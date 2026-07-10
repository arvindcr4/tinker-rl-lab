"""
HumanEval GRPO Environment for Atropos.
Trains models to write correct Python functions verified by unit tests.

Dataset: openai/openai_humaneval
Reward: 1.0 if generated code passes all test cases, 0.0 otherwise.
Execution: subprocess with 10s timeout (safe, no network access).
"""
import ast
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
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
    return os.environ.get("TINKER_CONFIG_PATH", "configs/humaneval_qwen_8b.yaml")

CONFIG_PATH = _get_config_path()

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the function body for the given Python function signature and docstring. "
    "Respond with ONLY the complete function implementation (including the def line). "
    "Do not include any other text, markdown, or explanation."
)


def _extract_code(text: str) -> str:
    """Extract Python code from model response."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"```python\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _run_tests(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """
    Execute generated code + test harness in a subprocess.
    Returns True if all tests pass, False otherwise.
    """
    full_code = textwrap.dedent(f"""
import sys
import traceback

# Generated function
{code}

# Test harness
{test_code}

# Run tests
try:
    check({entry_point})
    sys.exit(0)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
""")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


class HumanEvalEnv(BaseEnv):
    """
    GRPO environment for HumanEval coding tasks.
    Reward: 1.0 if generated code passes all unit tests.
    """

    name = "humaneval"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.train_data = []
        self.eval_data = []
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
        print("Loading HumanEval dataset...")
        dataset = load_dataset("openai/openai_humaneval", split="test")
        all_items = list(dataset)
        random.shuffle(all_items)
        # Use 80% for train (cycle through), 20% for eval
        split = int(0.8 * len(all_items))
        self.train_data = all_items[:split]
        self.eval_data = all_items[split:]
        print(f"Loaded {len(self.train_data)} train / {len(self.eval_data)} eval problems")
        self.iter = 0

    def _make_prompt(self, item: Dict) -> str:
        return item["prompt"]

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_correct_buffer:
            wandb_metrics["train/pass_rate"] = sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
        self.percent_correct_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, item: Dict) -> Dict:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._make_prompt(item)},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response = completion.choices[0].message.content
        code = _extract_code(response)
        # Prepend prompt (contains imports + function signature) to completion
        full_code = item["prompt"] + "\n" + code
        passed = _run_tests(full_code, item["test"], item["entry_point"])
        return {"score": float(passed), "task_id": item["task_id"]}

    async def evaluate(self, *args, **kwargs):
        import time
        from tqdm.asyncio import tqdm_asyncio
        start = time.time()
        tasks = [self.rollout_and_score_eval(ex) for ex in self.eval_data]
        results = await tqdm_asyncio.gather(*tasks)
        scores = [r["score"] for r in results]
        pass_rate = sum(scores) / len(scores) if scores else 0.0
        self.eval_metrics.append(("eval/pass_at_1", pass_rate))
        await self.evaluate_log(
            metrics={"eval/pass_at_1": pass_rate},
            samples=[{"messages": [], "score": r["score"], "task_id": r["task_id"]} for r in results],
            start_time=start,
            end_time=time.time(),
            generation_parameters={"temperature": 0.0, "max_tokens": self.config.max_token_length},
        )

    async def collect_trajectories(self, item: Dict) -> Tuple[ScoredDataGroup, list]:
        prompt_text = self._make_prompt(item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
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
            to_score.append({
                "messages": (*messages, {"role": "assistant", "content": choice.message.content}),
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "tokens": node.tokens,
                "masked_tokens": node.masked_tokens,
                "logprobs": node.logprobs,
            })
        return await self.score(to_score), []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        import asyncio
        scores_obj = ScoredDataGroup()
        scores_obj["tokens"] = []
        scores_obj["masks"] = []
        scores_obj["scores"] = []
        scores_obj["inference_logprobs"] = []

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            response = item["messages"][-1]["content"]
            code = _extract_code(response)
            full_code = item["prompt"] + "\n" + code
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            passed = await loop.run_in_executor(
                None, _run_tests, full_code, item["test"], item["entry_point"]
            )
            reward = 1.0 if passed else 0.0

            masked_tokens = item["masked_tokens"]
            if len([t for t in masked_tokens if t != -100]) < 10:
                continue

            scores_obj["tokens"].append(item["tokens"])
            scores_obj["masks"].append(masked_tokens)
            scores_obj["inference_logprobs"].append(item["logprobs"])
            scores_obj["scores"].append(reward)
            self.percent_correct_buffer.append(reward)

            if len(scores_obj["tokens"]) >= self.config.group_size:
                break

        return scores_obj if scores_obj["tokens"] else None

    async def get_next_item(self) -> Dict:
        item = self.train_data[self.iter % len(self.train_data)]
        self.iter += 1
        return item


if __name__ == "__main__":
    HumanEvalEnv.cli()
