"""
E6 — MoE Routing Variance (GSM8K).

Tests Finding 4: Qwen3-30B MoE training volatility is caused by
sparse routing divergence within GRPO groups.

Mechanism:
  In GRPO, G completions are generated for the *same* prompt. With a MoE model,
  each token generated uses the router to pick K experts. Two completions that
  diverge early (different first token) will activate different expert subsets
  throughout generation, making within-group advantage estimates structurally
  noisy — they aren't comparing apples to apples.

What this env measures:
  - `train/reward_std`      — std dev of rewards within each group of G completions
  - `train/reward_variance` — variance (reward_std^2)
  - `train/frac_volatile`   — fraction of steps where reward_std > 0.4 (high noise)

Ablation: run with ROLLOUT_TEMPERATURE=1.0 (diverse routing, expected high variance)
          then with ROLLOUT_TEMPERATURE=0.3 (constrained routing, expected lower variance)
Compare reward_std trajectories. Lower temp → more consistent token sequences → more
consistent routing → lower within-group variance → smoother training curve.

Config: configs/moe_routing_temp1_0.yaml  (standard)
        configs/moe_routing_temp0_3.yaml  (constrained)
Set ROLLOUT_TEMPERATURE env var to override.
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
    return os.environ.get("TINKER_CONFIG_PATH", "configs/moe_routing_temp1_0.yaml")


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

VOLATILE_STD_THRESHOLD = 0.4  # reward_std above this → "volatile step"


def _reward_std(scores: List[float]) -> float:
    """Sample standard deviation of a list of scores."""
    if len(scores) < 2:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return variance ** 0.5


class MoERoutingEnv(BaseEnv):
    """
    E6: MoE Routing Variance ablation.

    Trains on GSM8K (already solved at temp=1.0 — good baseline) with
    configurable rollout temperature. Tracks within-group reward variance
    to test whether routing consistency reduces training volatility.

    Run twice:
      ROLLOUT_TEMPERATURE=1.0 python -m tinker_atropos.environments.moe_routing_tinker ...
      ROLLOUT_TEMPERATURE=0.3 python -m tinker_atropos.environments.moe_routing_tinker ...

    Compare train/reward_std curves.
    """

    name = "moe_routing"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        # Variance tracking buffers
        self.reward_std_buffer = []
        self.reward_var_buffer = []
        self.volatile_steps = 0
        self.total_steps_logged = 0
        # Temperature is read from env var so it can be overridden without config changes
        self.rollout_temperature = float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0"))
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

        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        raw_test = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = [
            {
                "question": item["question"],
                "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
            }
            for item in raw_test
        ]
        print(
            f"[MoERouting] rollout_temperature={self.rollout_temperature} | "
            f"train={len(self.train)} | test={len(self.test)}"
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

        if self.reward_std_buffer:
            wandb_metrics["train/reward_std"] = (
                sum(self.reward_std_buffer) / len(self.reward_std_buffer)
            )
            wandb_metrics["train/reward_variance"] = (
                sum(self.reward_var_buffer) / len(self.reward_var_buffer)
            )
        self.reward_std_buffer = []
        self.reward_var_buffer = []

        if self.total_steps_logged > 0:
            wandb_metrics["train/frac_volatile"] = (
                self.volatile_steps / self.total_steps_logged
            )
        self.volatile_steps = 0
        self.total_steps_logged = 0

        # Log the temperature used (for dashboard filtering across runs)
        wandb_metrics["train/rollout_temperature"] = self.rollout_temperature

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
        return {"score": score, "question": question}

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
                temperature=self.rollout_temperature,  # <-- ablation variable
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
        group_rewards = []

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
            group_rewards.append(reward)
            self.percent_correct_buffer.append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Variance tracking — the core metric for this experiment
        if len(group_rewards) >= 2:
            std = _reward_std(group_rewards)
            self.reward_std_buffer.append(std)
            self.reward_var_buffer.append(std ** 2)
            self.total_steps_logged += 1
            if std > VOLATILE_STD_THRESHOLD:
                self.volatile_steps += 1

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    MoERoutingEnv.cli()
