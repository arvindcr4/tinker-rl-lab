"""
Multi-hop ReAct QA Agent — Atropos GRPO Environment.

Trains models to answer multi-hop questions using a ReAct loop over a
simple knowledge base.  This is a *genuinely agentic* RL task: the model
must plan which facts to retrieve and chain them across turns.

Tools:
    search(query)       — BM25-style keyword lookup over the knowledge base;
                          returns the top-3 matching paragraphs.
    lookup(title, sent) — Retrieve sentence `sent` (1-indexed) from the
                          paragraph titled `title`.
    finish(answer)      — Submit the final answer.

Dataset: HotpotQA (distractor setting, bridge questions).

Reward:
    1.0  — exact-match correct AND at least one search/lookup before finish
    0.5  — correct but went straight to finish (no retrieval)
    0.15 — wrong but produced at least two valid tool calls (exploration)
    0.0  — wrong with fewer than two valid tool calls
"""
import json
import os
import random
import re
import string
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from tinker_atropos.config import TinkerAtroposConfig


def _get_config_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get(
        "TINKER_CONFIG_PATH", "configs/multihop_react_qwen_8b.yaml"
    )


CONFIG_PATH = _get_config_path()

SYSTEM_PROMPT = """You are a research assistant that answers questions by searching a knowledge base.

Available tools:
1. search(query) — Search the knowledge base for paragraphs relevant to the query. Returns the top results.
2. lookup(title, sentence_num) — Look up a specific sentence (1-indexed) from the paragraph with the given title.
3. finish(answer) — Submit your final answer.

You MUST use a Thought/Action/Observation loop:

Thought: <reason about what information you need>
Action: search("query text")
Observation: <results will be filled in>
Thought: <reason about what you learned>
Action: lookup("Title", 2)
Observation: <result will be filled in>
...
Thought: <I now have enough information>
Action: finish("your answer")

Always search before answering. Do NOT guess without searching first."""


# ---------------------------------------------------------------------------
# Lightweight retrieval over HotpotQA context paragraphs
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _build_kb(context_titles: List[str], context_sentences: List[List[str]]) -> Dict[str, List[str]]:
    """Build a title → [sentences] knowledge base from HotpotQA context."""
    kb: Dict[str, List[str]] = {}
    for title, sents in zip(context_titles, context_sentences):
        kb[title] = sents
    return kb


def _bm25_search(query: str, kb: Dict[str, List[str]], top_k: int = 3) -> str:
    """Simple keyword overlap search (BM25-lite) over the knowledge base."""
    query_tokens = set(_normalize_answer(query).split())
    scored = []
    for title, sents in kb.items():
        full_text = " ".join(sents)
        doc_tokens = set(_normalize_answer(full_text).split())
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            scored.append((overlap, title, sents))
    scored.sort(key=lambda x: -x[0])
    results = []
    for _, title, sents in scored[:top_k]:
        preview = " ".join(sents[:3])
        if len(preview) > 300:
            preview = preview[:300] + "..."
        results.append(f"[{title}] {preview}")
    if not results:
        return "No results found."
    return "\n".join(results)


def _execute_react_trace(
    text: str, kb: Dict[str, List[str]]
) -> Tuple[Optional[str], int, int]:
    """
    Execute a ReAct trace against the knowledge base.
    Returns (submitted_answer, valid_tool_calls, total_actions).
    """
    submitted_answer = None
    valid_calls = 0
    total_actions = 0

    for line in text.split("\n"):
        stripped = line.strip()

        # search("query")
        search_match = re.match(
            r'Action:\s*search\s*\(\s*["\'](.+?)["\']\s*\)',
            stripped,
            re.IGNORECASE,
        )
        if search_match:
            total_actions += 1
            query = search_match.group(1)
            _bm25_search(query, kb)  # execute for side-effect validation
            valid_calls += 1
            continue

        # lookup("title", num)
        lookup_match = re.match(
            r'Action:\s*lookup\s*\(\s*["\'](.+?)["\']\s*,\s*(\d+)\s*\)',
            stripped,
            re.IGNORECASE,
        )
        if lookup_match:
            total_actions += 1
            title = lookup_match.group(1)
            sent_num = int(lookup_match.group(2))
            if title in kb and 1 <= sent_num <= len(kb[title]):
                valid_calls += 1
            continue

        # finish("answer")
        finish_match = re.match(
            r'Action:\s*finish\s*\(\s*["\'](.+?)["\']\s*\)',
            stripped,
            re.IGNORECASE,
        )
        if finish_match:
            total_actions += 1
            submitted_answer = finish_match.group(1)
            valid_calls += 1
            continue

    return submitted_answer, valid_calls, total_actions


def _score_react_trace(
    response: str,
    gold_answer: str,
    kb: Dict[str, List[str]],
) -> Tuple[float, dict]:
    """
    Score a ReAct trace:
        1.0  — correct + retrieved first
        0.5  — correct + no retrieval
        0.15 — wrong but >= 2 valid tool calls (exploration credit)
        0.0  — wrong + < 2 valid tool calls
    """
    submitted, valid_calls, total_actions = _execute_react_trace(response, kb)

    # Check correctness: either via finish() or fallback to last line
    answer_text = submitted
    if answer_text is None:
        # Fallback: try to find answer in last few lines
        for line in reversed(response.strip().split("\n")):
            line = line.strip()
            if line and not line.startswith(("Thought:", "Action:", "Observation:")):
                answer_text = line
                break

    correct = False
    if answer_text and gold_answer:
        correct = _exact_match(answer_text, gold_answer) or _f1_score(answer_text, gold_answer) >= 0.8

    # Did model search/lookup before answering?
    retrieval_calls = valid_calls - (1 if submitted else 0)

    if correct and retrieval_calls > 0:
        reward = 1.0
    elif correct and retrieval_calls == 0:
        reward = 0.5
    elif not correct and valid_calls >= 2:
        reward = 0.15
    else:
        reward = 0.0

    meta = {
        "submitted_answer": submitted,
        "answer_text": answer_text,
        "correct": correct,
        "valid_calls": valid_calls,
        "total_actions": total_actions,
        "retrieval_calls": retrieval_calls,
    }
    return reward, meta


# ---------------------------------------------------------------------------
# HotpotQA data loading
# ---------------------------------------------------------------------------

def _load_hotpotqa(split: str = "train", max_examples: int = 5000, seed: int = 42):
    """Load HotpotQA bridge questions with their context paragraphs."""
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    ds = ds.shuffle(seed=seed)

    examples = []
    for item in ds:
        if item.get("type") != "bridge":
            continue
        if not item.get("question") or not item.get("answer"):
            continue

        titles = item.get("context", {}).get("title", [])
        sentences = item.get("context", {}).get("sentences", [])
        if not titles or not sentences:
            continue

        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "context_titles": titles,
            "context_sentences": sentences,
        })
        if len(examples) >= max_examples:
            break
    return examples


# ---------------------------------------------------------------------------
# Atropos environment
# ---------------------------------------------------------------------------


class MultihopReactEnv(BaseEnv):
    """Multi-hop ReAct QA agent trained with GRPO."""

    name = "multihop_react"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.tool_use_buffer = []
        self.eval_metrics = []
        self.iter = 0

    @classmethod
    def config_init(cls):
        config = (
            TinkerAtroposConfig.from_yaml(CONFIG_PATH)
            if CONFIG_PATH
            else TinkerAtroposConfig()
        )
        cls._full_config = config
        env_config = BaseEnvConfig(
            tokenizer_name=config.base_model,
            group_size=config.group_size,
            use_wandb=config.use_wandb,
            rollout_server_url=config.atropos_api_url,
            total_steps=config.num_steps,
            batch_size=config.batch_size,
            steps_per_eval=config.steps_per_eval,
            max_token_length=config.max_token_env_length,
            max_num_workers=config.max_num_workers,
            max_batches_offpolicy=config.max_batches_offpolicy,
            wandb_name=f"{config.wandb_run_name}-env",
            ensure_scores_are_not_same=config.ensure_scores_are_not_same,
        )
        server_configs = [
            APIServerConfig(
                model_name=config.base_model,
                base_url=config.inference_api_url + "/v1",
                api_key="x",
                server_type="sglang",
                num_requests_for_eval=config.num_requests_for_eval,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        full_cfg = getattr(self.__class__, "_full_config", None)
        data_seed = full_cfg.data_seed if full_cfg else 42

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% if loop.last and message['role'] != 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        print("Loading HotpotQA dataset...")
        all_examples = _load_hotpotqa("train", max_examples=5000, seed=data_seed)
        split = int(0.9 * len(all_examples))
        self.train_examples = all_examples[:split]
        self.eval_examples = all_examples[split:]
        print(f"Loaded {len(self.train_examples)} train / {len(self.eval_examples)} eval examples")
        self.iter = 0

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        if self.tool_use_buffer:
            wandb_metrics["train/retrieval_rate"] = sum(self.tool_use_buffer) / len(
                self.tool_use_buffer
            )
        self.percent_correct_buffer = []
        self.tool_use_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, example: dict) -> dict:
        kb = _build_kb(example["context_titles"], example["context_sentences"])
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response = completion.choices[0].message.content
        reward, meta = _score_react_trace(response, example["answer"], kb)
        return {
            "score": reward,
            "response": response,
            "question": example["question"],
            "gold_answer": example["answer"],
            **meta,
        }

    async def evaluate(self, *args, **kwargs):
        import time
        from tqdm.asyncio import tqdm_asyncio

        start = time.time()
        tasks = [
            self.rollout_and_score_eval(ex) for ex in self.eval_examples[:200]
        ]
        results = await tqdm_asyncio.gather(*tasks)

        scores = [r["score"] for r in results]
        full_correct = sum(1 for s in scores if s >= 1.0) / len(scores)
        any_correct = sum(1 for s in scores if s >= 0.5) / len(scores)
        retrieval_rate = sum(
            1 for r in results if r["retrieval_calls"] > 0
        ) / len(results)
        avg_tool_calls = sum(r["valid_calls"] for r in results) / len(results)

        self.eval_metrics.append(("eval/agentic_accuracy", full_correct))
        self.eval_metrics.append(("eval/answer_accuracy", any_correct))
        self.eval_metrics.append(("eval/retrieval_rate", retrieval_rate))
        self.eval_metrics.append(("eval/avg_tool_calls", avg_tool_calls))

        await self.evaluate_log(
            metrics={
                "eval/agentic_accuracy": full_correct,
                "eval/answer_accuracy": any_correct,
                "eval/retrieval_rate": retrieval_rate,
                "eval/avg_tool_calls": avg_tool_calls,
            },
            samples=[
                {
                    "messages": [],
                    "score": r["score"],
                    "question": r["question"],
                    "gold_answer": r["gold_answer"],
                    "submitted": r.get("submitted_answer"),
                }
                for r in results[:10]
            ],
            start_time=start,
            end_time=time.time(),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        kb = _build_kb(item["context_titles"], item["context_sentences"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completions = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
                stop=[self.tokenizer.eos_token_id],
            )
            state = managed.get_state()
            nodes = state["nodes"]

        to_score = []
        for choice, node in zip(completions.choices, nodes):
            to_score.append(
                {
                    "messages": (
                        *messages,
                        {"role": "assistant", "content": choice.message.content},
                    ),
                    "answer": item["answer"],
                    "kb": kb,
                    "tokens": node.tokens,
                    "masked_tokens": node.masked_tokens,
                    "logprobs": node.logprobs,
                }
            )
        return await self.score(to_score), []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            response = item["messages"][-1]["content"]
            reward, meta = _score_react_trace(response, item["answer"], item["kb"])

            masked_tokens = item["masked_tokens"]
            if len([t for t in masked_tokens if t != -100]) < 10:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(masked_tokens)
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(float(reward))

            self.percent_correct_buffer.append(float(reward >= 0.5))
            self.tool_use_buffer.append(float(meta["retrieval_calls"] > 0))

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        item = self.train_examples[self.iter % len(self.train_examples)]
        self.iter += 1
        return item


if __name__ == "__main__":
    MultihopReactEnv.cli()
