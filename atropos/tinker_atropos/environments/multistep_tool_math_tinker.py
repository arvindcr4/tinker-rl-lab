"""
Multi-step Tool-Use Math Agent — Atropos GRPO Environment.

Trains models to solve math problems via a ReAct-style tool-calling loop:
    Thought → Action → Observation → … → Answer

Available tools (executed deterministically in a sandbox):
    calculate(expression)   — evaluate a Python math expression
    store(name, value)      — save a value for later retrieval
    recall(name)            — retrieve a previously stored value

Dataset: GSM8K (same problems, but the reward now requires multi-step
tool-use traces, not a single-shot boxed answer).

Reward:
    1.0  — final answer correct AND at least one valid tool call
    0.5  — final answer correct but no tool calls (single-shot shortcut)
    0.25 — wrong answer but produced at least one correct tool call
    0.0  — wrong answer with no valid tool calls
"""
import json
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

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
        "TINKER_CONFIG_PATH", "configs/multistep_tool_math_qwen_8b.yaml"
    )


CONFIG_PATH = _get_config_path()

TOOL_SCHEMA = """You have access to the following tools:

1. calculate(expression) — Evaluate a Python math expression and return the numeric result.
   Example: calculate(5 * 3.50 + 2.25)

2. store(name, value) — Store a numeric value under a variable name for later use.
   Example: store("total_cost", 17.50)

3. recall(name) — Retrieve a previously stored value.
   Example: recall("total_cost")
"""

SYSTEM_PROMPT = f"""You are a helpful math assistant that solves problems step by step using tools.

{TOOL_SCHEMA}

You MUST solve problems using a Thought/Action/Observation loop:

Thought: <reason about the next step>
Action: <tool_name>(<arguments>)
Observation: <result will be filled in>

Repeat until you have the final answer, then write:
Answer: \\boxed{{<number>}}

Always break the problem into explicit calculation steps. Do NOT skip to the answer."""

QUESTION_SUFFIX = " Solve step by step using the tools. Provide the final numerical answer inside \\boxed{}."


# ---------------------------------------------------------------------------
# Deterministic tool sandbox
# ---------------------------------------------------------------------------

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "int": int, "float": float,
    "math": math,
    "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
    "pow": pow, "sum": sum,
}


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a math expression in a restricted namespace."""
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)
        return float(result)
    except Exception:
        return None


def _execute_trace(text: str) -> Tuple[str, int, int]:
    """
    Walk through a ReAct trace, execute tool calls, and return:
        (filled_trace, num_valid_tool_calls, num_total_actions)
    """
    store: Dict[str, float] = {}
    filled_lines: List[str] = []
    valid_calls = 0
    total_actions = 0

    for line in text.split("\n"):
        stripped = line.strip()

        # Match Action: tool_name(args)
        action_match = re.match(
            r"Action:\s*(calculate|store|recall)\s*\((.+)\)\s*$",
            stripped,
            re.IGNORECASE,
        )
        if action_match:
            total_actions += 1
            tool_name = action_match.group(1).lower()
            raw_args = action_match.group(2).strip()

            observation = "Error: invalid call"

            if tool_name == "calculate":
                result = _safe_eval(raw_args)
                if result is not None:
                    observation = str(result)
                    valid_calls += 1
                else:
                    observation = "Error: could not evaluate expression"

            elif tool_name == "store":
                # store("name", value) or store(name, value)
                parts = [p.strip().strip('"').strip("'") for p in raw_args.split(",", 1)]
                if len(parts) == 2:
                    name = parts[0]
                    val = _safe_eval(parts[1])
                    if val is not None:
                        store[name] = val
                        observation = f"Stored {name} = {val}"
                        valid_calls += 1
                    else:
                        observation = "Error: could not evaluate value"

            elif tool_name == "recall":
                key = raw_args.strip().strip('"').strip("'")
                if key in store:
                    observation = str(store[key])
                    valid_calls += 1
                else:
                    observation = f"Error: '{key}' not found in store"

            filled_lines.append(line)
            filled_lines.append(f"Observation: {observation}")
            continue

        # Skip model-generated Observation lines (we replace them)
        if stripped.startswith("Observation:"):
            continue

        filled_lines.append(line)

    return "\n".join(filled_lines), valid_calls, total_actions


def _extract_answer(text: str) -> Optional[str]:
    """Extract the \\boxed{...} answer from the trace."""
    # Find last boxed answer
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def _score_trace(
    response: str,
    gold_answer: str,
) -> float:
    """
    Score a ReAct trace:
        1.0  correct + used tools
        0.5  correct + no tools (single-shot shortcut)
        0.25 wrong   + at least one valid tool call
        0.0  wrong   + no valid tool calls
    """
    _, valid_calls, _ = _execute_trace(response)
    used_tools = valid_calls > 0

    # Parse gold
    gold_parsed = parse(
        "\\boxed{" + gold_answer + "}",
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    # Parse model answer
    answer_parsed = parse(
        response.split("</think>")[-1],
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    correct = bool(verify(answer_parsed, gold_parsed)) if gold_parsed else False

    if correct and used_tools:
        return 1.0
    elif correct and not used_tools:
        return 0.5
    elif not correct and used_tools:
        return 0.25
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Atropos environment
# ---------------------------------------------------------------------------


class MultistepToolMathEnv(BaseEnv):
    """Multi-step tool-use math agent trained with GRPO."""

    name = "multistep_tool_math"

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

        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=data_seed)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=data_seed)
        self.test = [
            {
                "question": item["question"],
                "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
            }
            for item in test_data
        ]
        self.iter = 0

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        if self.tool_use_buffer:
            wandb_metrics["train/tool_use_rate"] = sum(self.tool_use_buffer) / len(
                self.tool_use_buffer
            )
        self.percent_correct_buffer = []
        self.tool_use_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, question: str, gold_answer: str) -> dict:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question + QUESTION_SUFFIX},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response = completion.choices[0].message.content
        score = _score_trace(response, gold_answer)
        _, valid_calls, total_actions = _execute_trace(response)
        return {
            "score": score,
            "response": response,
            "valid_tool_calls": valid_calls,
            "total_actions": total_actions,
            "question": question,
        }

    async def evaluate(self, *args, **kwargs):
        import time
        from tqdm.asyncio import tqdm_asyncio

        start = time.time()
        tasks = [
            self.rollout_and_score_eval(item["question"], item["gold_answer"])
            for item in self.test[:300]
        ]
        results = await tqdm_asyncio.gather(*tasks)

        scores = [r["score"] for r in results]
        full_correct = sum(1 for s in scores if s >= 1.0) / len(scores)
        any_correct = sum(1 for s in scores if s >= 0.5) / len(scores)
        tool_use_rate = sum(
            1 for r in results if r["valid_tool_calls"] > 0
        ) / len(results)
        avg_tool_calls = sum(r["valid_tool_calls"] for r in results) / len(results)

        self.eval_metrics.append(("eval/agentic_accuracy", full_correct))
        self.eval_metrics.append(("eval/answer_accuracy", any_correct))
        self.eval_metrics.append(("eval/tool_use_rate", tool_use_rate))
        self.eval_metrics.append(("eval/avg_tool_calls", avg_tool_calls))

        await self.evaluate_log(
            metrics={
                "eval/agentic_accuracy": full_correct,
                "eval/answer_accuracy": any_correct,
                "eval/tool_use_rate": tool_use_rate,
                "eval/avg_tool_calls": avg_tool_calls,
            },
            samples=[
                {
                    "messages": [],
                    "score": r["score"],
                    "question": r["question"],
                    "tool_calls": r["valid_tool_calls"],
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
        gold_answer = item["answer"].split("#")[-1].strip().replace(",", "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"] + QUESTION_SUFFIX},
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
                    "gold_answer": gold_answer,
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
            reward = _score_trace(response, item["gold_answer"])

            _, valid_calls, _ = _execute_trace(response)

            masked_tokens = item["masked_tokens"]
            if len([t for t in masked_tokens if t != -100]) < 10:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(masked_tokens)
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(float(reward))

            self.percent_correct_buffer.append(float(reward >= 0.5))
            self.tool_use_buffer.append(float(valid_calls > 0))

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return item


if __name__ == "__main__":
    MultistepToolMathEnv.cli()
