"""
E5 — Bootstrap Threshold Sweep (GSM8K).

Tests Finding 1: GRPO requires a non-zero initialization accuracy (seed signal) to converge.
The hypothesis: GRPO fails when step-0 pass@1 ≈ 0% across the group; succeeds above ~7%.

Design:
  - Proxy for step-0 difficulty: gold solution word count (longer solution → more steps → harder)
  - Bin training problems into quintiles by solution length
  - Config `difficulty_bin` selects which quintile to train on
  - Extra metrics: frac_zero_signal (fraction of steps where ALL group members score 0)
  - Running this with difficulty_bin="easy" vs "hardest" directly tests the threshold claim

Config: configs/bootstrap_threshold_easy.yaml  (or hardest / medium)
"""

import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

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
    return os.environ.get("TINKER_CONFIG_PATH", "configs/bootstrap_threshold_easy.yaml")


CONFIG_PATH = _get_config_path()

question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

convo_prefix = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + question_suffix,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]

# Quintile bins by solution word count
# easy    = Q1 (shortest solutions, fewest steps, highest step-0 accuracy)
# medium  = Q3 (median difficulty)
# hard    = Q4 (complex multi-step)
# hardest = Q5 (longest solutions, most steps, lowest step-0 accuracy)
DIFFICULTY_BINS = {
    "easy": (0.0, 0.20),
    "medium": (0.40, 0.60),
    "hard": (0.70, 0.90),
    "hardest": (0.90, 1.00),
    "all": (0.0, 1.0),
}


def _solution_word_count(answer: str) -> int:
    """Proxy for problem difficulty: number of words in the gold solution."""
    return len(answer.split())


def _bin_by_difficulty(dataset, bin_name: str):
    """
    Filter dataset to a difficulty quintile using solution length as proxy.
    Returns the filtered subset.
    """
    if bin_name not in DIFFICULTY_BINS:
        raise ValueError(f"Unknown difficulty_bin '{bin_name}'. Choose from {list(DIFFICULTY_BINS)}")

    lo, hi = DIFFICULTY_BINS[bin_name]
    if bin_name == "all":
        return list(dataset)

    all_items = list(dataset)
    lengths = [_solution_word_count(item["answer"]) for item in all_items]
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    lo_thresh = sorted_lengths[int(lo * n)]
    hi_thresh = sorted_lengths[int(hi * n) - 1] if hi < 1.0 else sorted_lengths[-1]

    binned = [
        item for item in all_items
        if lo_thresh <= _solution_word_count(item["answer"]) <= hi_thresh
    ]
    return binned


class BootstrapThresholdEnv(BaseEnv):
    """
    E5: Bootstrap Threshold Sweep.
    Trains on a difficulty-stratified subset of GSM8K to test whether
    GRPO converges when step-0 accuracy is near zero.

    Key extra metrics:
      train/frac_zero_signal  — fraction of steps where all G rollouts scored 0
      train/mean_group_reward — mean reward per group (tracks whether signal exists)
      train/difficulty_bin    — which bin is being used (logged once)
    """

    name = "bootstrap_threshold"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        # Threshold-specific buffers
        self.zero_signal_steps = 0   # steps where all group scores == 0
        self.total_steps_logged = 0
        self.group_reward_buffer = []
        self.difficulty_bin = os.environ.get("BOOTSTRAP_DIFFICULTY_BIN", "easy")
        self.iter = 0

    @classmethod
    def config_init(cls):
        config = TinkerAtroposConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH else TinkerAtroposConfig()
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

        raw_train = load_dataset("gsm8k", "main", split="train")
        raw_test = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)

        self.train = _bin_by_difficulty(raw_train, self.difficulty_bin)
        random.shuffle(self.train)

        self.test = [
            {
                "question": item["question"],
                "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
            }
            for item in raw_test
        ]

        print(
            f"[BootstrapThreshold] difficulty_bin={self.difficulty_bin} | "
            f"train={len(self.train)} / full_test={len(self.test)}"
        )
        print(
            f"  Solution length range for bin '{self.difficulty_bin}': "
            f"{min(_solution_word_count(x['answer']) for x in self.train)} – "
            f"{max(_solution_word_count(x['answer']) for x in self.train)} words"
        )
        self.iter = 0

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
        self.percent_correct_buffer = []

        if self.group_reward_buffer:
            wandb_metrics["train/mean_group_reward"] = (
                sum(self.group_reward_buffer) / len(self.group_reward_buffer)
            )
        self.group_reward_buffer = []

        if self.total_steps_logged > 0:
            wandb_metrics["train/frac_zero_signal"] = (
                self.zero_signal_steps / self.total_steps_logged
            )
        self.zero_signal_steps = 0
        self.total_steps_logged = 0

        # Log which bin we're in (constant, but useful for dashboard filtering)
        wandb_metrics["train/difficulty_bin_id"] = list(DIFFICULTY_BINS.keys()).index(
            self.difficulty_bin
        )

        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, question: str, answer: str) -> dict:
        completion = await self.server.chat_completion(
            messages=[*convo_prefix, {"role": "user", "content": question + question_suffix}],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response_content = completion.choices[0].message.content
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            response_content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False, malformed_operators=False,
                        basic_latex=True, boxed="all", units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        return {"score": score, "question": question, "gold_answer": answer}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()
        tasks = [
            self.rollout_and_score_eval(item["question"], item["gold_answer"])
            for item in self.test
        ]
        results = await tqdm_asyncio.gather(*tasks)
        scores = [r["score"] for r in results]
        percent_correct = sum(scores) / len(scores)
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        await self.evaluate_log(
            metrics={"eval/percent_correct": percent_correct},
            samples=[{"question": r["question"], "score": r["score"]} for r in results[:20]],
            start_time=start_time,
            end_time=time.time(),
            generation_parameters={"temperature": 0.0, "max_tokens": self.config.max_token_length},
        )

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        user_message = {"role": "user", "content": item["question"] + question_suffix}
        gold_answer = "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"
        messages = [*convo_prefix, user_message]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completion = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
                stop=[self.tokenizer.eos_token_id],
            )
            state = managed.get_state()
            nodes = state["nodes"]

        to_score = []
        for choice, node in zip(chat_completion.choices, nodes):
            to_score.append({
                "messages": (*convo_prefix, user_message, {"role": "assistant", "content": choice.message.content}),
                "gold_answer": gold_answer,
                "finish_reason": choice.finish_reason,
                "tokens": node.tokens,
                "masked_tokens": node.masked_tokens,
                "logprobs": node.logprobs,
            })
        return await self.score(to_score), []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []

        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if not gold_parsed:
            return None

        random.shuffle(rollout_group_data)
        group_scores = []

        for item in rollout_group_data:
            answer_parsed = parse(
                item["messages"][-1]["content"].split("</think>")[-1],
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False, malformed_operators=False,
                            basic_latex=True, boxed="all", units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = 1.0 if verify(answer_parsed, gold_parsed) else 0.0

            if len([t for t in item["masked_tokens"] if t != -100]) < 10:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(item["masked_tokens"])
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(reward)
            group_scores.append(reward)
            self.percent_correct_buffer.append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Zero-signal tracking: did every completion score 0?
        if group_scores:
            mean_group = sum(group_scores) / len(group_scores)
            self.group_reward_buffer.append(mean_group)
            self.total_steps_logged += 1
            if all(s == 0.0 for s in group_scores):
                self.zero_signal_steps += 1

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return item


if __name__ == "__main__":
    BootstrapThresholdEnv.cli()
