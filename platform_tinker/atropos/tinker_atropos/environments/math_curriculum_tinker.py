"""
E7 — Two-Phase Curriculum Reward (MATH Competition).

Tests Finding 2: The phase transition in GRPO training (slow format learning,
then fast reasoning) can be accelerated by explicitly rewarding format compliance
in early training before switching to answer correctness.

Hypothesis:
  Standard GRPO (answer-only reward throughout) gets stuck at ~14% on MATH because
  early completions don't even attempt \\boxed{} formatting — reward is always 0 →
  zero gradient signal (same as Llama-3.2-3B failure mode on GSM8K).

  Two-phase curriculum:
    Phase 1 (steps 0 → phase1_steps):  reward = format score only
      - 0.0  if no \\boxed{} in response at all
      - 0.5  if \\boxed{} present but may contain garbage
      - 1.0  if \\boxed{} present AND content is valid LaTeX (parseable)
    Phase 2 (steps > phase1_steps):    reward = answer correctness only
      - 1.0  if answer matches gold (symbolic verification)
      - 0.0  otherwise

  The format reward teaches the model WHERE to put the answer (format compliance).
  Once that's stable, the correctness reward teaches WHAT the answer should be.

Config: configs/math_curriculum_qwen8b.yaml
  Extra field: phase1_steps (default 15) — number of wandb-log cycles in phase 1
               Set via env var CURRICULUM_PHASE1_STEPS to override.
"""

import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset, concatenate_datasets
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
    return os.environ.get("TINKER_CONFIG_PATH", "configs/math_curriculum_qwen8b.yaml")


CONFIG_PATH = _get_config_path()

question_suffix = " Provide your answer inside \\boxed{}."

convo_prefix = [
    {
        "role": "user",
        "content": "What is the sum of all positive integers n such that n^2 divides 12!?" + question_suffix,
    },
    {
        "role": "assistant",
        "content": (
            "We need n^2 | 12!. First, 12! = 2^10 * 3^5 * 5^2 * 7 * 11. "
            "For n^2 | 12!, each prime in n can appear at most floor(exponent/2) times. "
            "So n divides 2^5 * 3^2 * 5 = 1440. "
            "The sum of divisors of 1440 = (1+2+4+8+16+32)(1+3+9)(1+5) = 63 * 13 * 6 = 4914. "
            "\\boxed{4914}"
        ),
    },
]

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Regex: is there a \boxed{...} anywhere in the response?
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _format_reward(response: str) -> float:
    """
    Phase 1 reward: measures format compliance only.

    Returns:
      0.0 — no \\boxed{} in response at all
      0.5 — \\boxed{} present but LaTeX inside is not parseable
      1.0 — \\boxed{} present AND parseable by math_verify
    """
    if not _BOXED_RE.search(response):
        return 0.0
    # Try to parse — if math_verify can extract something, it's valid enough
    parsed = parse(
        response.split("</think>")[-1],
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
    return 1.0 if parsed else 0.5


def _correctness_reward(response: str, gold_parsed) -> float:
    """Phase 2 reward: standard binary answer correctness."""
    answer_parsed = parse(
        response.split("</think>")[-1],
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
    return 1.0 if verify(answer_parsed, gold_parsed) else 0.0


class MATHCurriculumEnv(BaseEnv):
    """
    E7: Two-phase curriculum reward for MATH competition problems.

    Phase 1 — teaches the model to write answers inside \\boxed{}.
    Phase 2 — teaches the model to write the CORRECT answer inside \\boxed{}.

    Key metrics:
      train/current_phase     — 1 or 2 (which reward function is active)
      train/format_score      — avg format reward (tracked in both phases for comparison)
      train/correctness_score — avg correctness (tracked in both phases)
      train/percent_correct   — same as correctness_score (for continuity with other envs)
    """

    name = "math_curriculum"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        # Phase tracking
        # phase1_steps: number of wandb_log() calls (≈ training steps) before switching
        self.phase1_steps = int(os.environ.get("CURRICULUM_PHASE1_STEPS", "15"))
        self.wandb_log_calls = 0  # increments in wandb_log()
        # Per-step metric buffers
        self.format_score_buffer = []
        self.correctness_score_buffer = []
        self.iter = 0

    @property
    def current_phase(self) -> int:
        return 1 if self.wandb_log_calls < self.phase1_steps else 2

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

        train_splits = [
            load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            for subj in MATH_SUBJECTS
        ]
        test_splits = [
            load_dataset("EleutherAI/hendrycks_math", subj, split="test")
            for subj in MATH_SUBJECTS
        ]
        self.train = concatenate_datasets(train_splits).shuffle(seed=42)
        test_data = concatenate_datasets(test_splits).shuffle(seed=42)
        self.test = [{"problem": item["problem"], "solution": item["solution"]} for item in test_data]

        print(
            f"[MATHCurriculum] phase1_steps={self.phase1_steps} | "
            f"current_phase={self.current_phase} | "
            f"train={len(self.train)} | test={len(self.test)}"
        )
        self.iter = 0

    def _extract_boxed_answer(self, solution: str) -> str:
        idx = solution.rfind("\\boxed{")
        if idx == -1:
            return solution.strip()
        depth = 0
        start = idx + len("\\boxed{")
        for i in range(start, len(solution)):
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                if depth == 0:
                    return solution[start:i]
                depth -= 1
        return solution[start:].strip()

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
        self.percent_correct_buffer = []

        if self.format_score_buffer:
            wandb_metrics["train/format_score"] = (
                sum(self.format_score_buffer) / len(self.format_score_buffer)
            )
        self.format_score_buffer = []

        if self.correctness_score_buffer:
            wandb_metrics["train/correctness_score"] = (
                sum(self.correctness_score_buffer) / len(self.correctness_score_buffer)
            )
        self.correctness_score_buffer = []

        wandb_metrics["train/current_phase"] = self.current_phase
        wandb_metrics["train/wandb_log_calls"] = self.wandb_log_calls

        # Phase transition announcement
        if self.wandb_log_calls == self.phase1_steps:
            print(
                f"[MATHCurriculum] *** PHASE TRANSITION: switching from format reward "
                f"to correctness reward at wandb_log call {self.wandb_log_calls} ***"
            )

        self.wandb_log_calls += 1

        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, problem: str, solution: str) -> dict:
        gold_answer = self._extract_boxed_answer(solution)
        completion = await self.server.chat_completion(
            messages=[*convo_prefix, {"role": "user", "content": problem + question_suffix}],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response_content = completion.choices[0].message.content
        gold_parsed = parse(
            "\\boxed{" + gold_answer + "}",
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
        # Eval always uses correctness (phase-independent)
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        fmt_score = _format_reward(response_content)
        return {
            "score": score,
            "format_score": fmt_score,
            "problem": problem,
            "gold_answer": gold_answer,
        }

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()
        tasks = [
            self.rollout_and_score_eval(item["problem"], item["solution"])
            for item in self.test
        ]
        results = await tqdm_asyncio.gather(*tasks)
        scores = [r["score"] for r in results]
        fmt_scores = [r["format_score"] for r in results]
        percent_correct = sum(scores) / len(scores)
        avg_format = sum(fmt_scores) / len(fmt_scores)

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        self.eval_metrics.append(("eval/format_score", avg_format))
        self.eval_metrics.append(("eval/current_phase", float(self.current_phase)))

        await self.evaluate_log(
            metrics={
                "eval/percent_correct": percent_correct,
                "eval/format_score": avg_format,
                "eval/current_phase": float(self.current_phase),
            },
            samples=[
                {
                    "problem": r["problem"],
                    "gold_answer": r["gold_answer"],
                    "score": r["score"],
                    "format_score": r["format_score"],
                }
                for r in results[:20]
            ],
            start_time=start_time,
            end_time=time.time(),
            generation_parameters={"temperature": 0.0, "max_tokens": self.config.max_token_length},
        )

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        user_message = {"role": "user", "content": item["problem"] + question_suffix}
        gold_answer = "\\boxed{" + self._extract_boxed_answer(item["solution"]) + "}"
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

        phase = self.current_phase
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            response = item["messages"][-1]["content"]
            fmt_score = _format_reward(response)
            corr_score = _correctness_reward(response, gold_parsed)

            # Phase determines which signal trains the policy
            if phase == 1:
                reward = fmt_score
            else:
                reward = corr_score

            if len([t for t in item["masked_tokens"] if t != -100]) < 10:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(item["masked_tokens"])
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(reward)

            # Always track both metrics for analysis (phase-independent logging)
            self.format_score_buffer.append(fmt_score)
            self.correctness_score_buffer.append(corr_score)
            self.percent_correct_buffer.append(corr_score)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    MATHCurriculumEnv.cli()
