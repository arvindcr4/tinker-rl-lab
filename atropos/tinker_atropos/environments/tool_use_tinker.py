"""
Tool Use GRPO Environment for Atropos.
Trains models to correctly call functions/tools from a schema.

Dataset: glaiveai/glaive-function-calling-v2
Reward: binary — 1.0 if tool name + all required args match, 0.0 otherwise.
"""
import json
import os
import random
import re
import sys
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
    return os.environ.get("TINKER_CONFIG_PATH", "configs/tool_use_qwen_8b.yaml")

CONFIG_PATH = _get_config_path()

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When the user's request requires a tool, respond ONLY with a JSON object in this format:\n"
    '{"tool": "<tool_name>", "arguments": {<key>: <value>, ...}}\n'
    "Do not include any other text. If no tool is needed, answer directly."
)

def _parse_tool_call(text: str) -> Optional[Dict]:
    """Extract the first JSON object from model output."""
    # Try to find JSON block
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _score_tool_call(predicted: Optional[Dict], expected_name: str, expected_args: Dict) -> float:
    """
    Binary reward:
    - 0.0 if no valid JSON or wrong tool name
    - 0.5 if tool name correct but args wrong
    - 1.0 if tool name + all required args correct
    """
    if predicted is None:
        return 0.0
    pred_name = predicted.get("tool", predicted.get("name", "")).strip().lower()
    if pred_name != expected_name.strip().lower():
        return 0.0
    pred_args = predicted.get("arguments", predicted.get("parameters", predicted.get("args", {})))
    if not isinstance(pred_args, dict):
        return 0.5
    # Check all expected args are present and match (string comparison, lowercased)
    for key, val in expected_args.items():
        if key not in pred_args:
            return 0.5
        if str(pred_args[key]).lower().strip() != str(val).lower().strip():
            return 0.5
    return 1.0


def _build_glaive_examples(raw_dataset, max_examples: int = 5000):
    """
    Parse glaive-function-calling-v2 into (system_with_tools, user_query, tool_name, tool_args) tuples.
    The dataset has 'system' (with SYSTEM token + tool JSON) and 'chat' fields.
    """
    examples = []
    for item in raw_dataset:
        try:
            system_text = item.get("system", "")
            chat_text = item.get("chat", "")
            # Extract tool schema from system field: "SYSTEM: ... <tools>\n{...}\n</tools>"
            tool_match = re.search(r"<tools>(.*?)</tools>", system_text, re.DOTALL)
            if not tool_match:
                continue
            tools_json_str = tool_match.group(1).strip()
            tools = json.loads(tools_json_str) if tools_json_str.startswith("[") else [json.loads(tools_json_str)]

            # Extract first user turn and first function call from chat
            user_match = re.search(r"USER:\s*(.*?)(?=ASSISTANT:|$)", chat_text, re.DOTALL)
            func_match = re.search(r"<functioncall>\s*(\{.*?\})\s*</functioncall>", chat_text, re.DOTALL)
            if not user_match or not func_match:
                continue
            user_query = user_match.group(1).strip()
            func_call = json.loads(func_match.group(1))
            func_name = func_call.get("name", "")
            func_args = func_call.get("arguments", {})
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except Exception:
                    continue
            if not func_name or not isinstance(func_args, dict):
                continue

            # Build tool schema string for system prompt
            tools_str = json.dumps(tools, indent=2)
            system_with_tools = (
                SYSTEM_PROMPT
                + f"\n\nAvailable tools:\n{tools_str}"
            )
            examples.append({
                "system": system_with_tools,
                "user": user_query,
                "expected_tool": func_name,
                "expected_args": func_args,
            })
            if len(examples) >= max_examples:
                break
        except Exception:
            continue
    return examples


class ToolUseEnv(BaseEnv):
    """
    GRPO environment for tool/function calling.
    Reward: 1.0 if model calls correct tool with correct args.
    """

    name = "tool_use"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.examples = []
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
        print("Loading glaive function calling dataset...")
        raw = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        self.examples = _build_glaive_examples(raw, max_examples=5000)
        random.shuffle(self.examples)
        # Split: 90% train, 10% eval
        split = int(0.9 * len(self.examples))
        self.train_examples = self.examples[:split]
        self.eval_examples = self.examples[split:]
        print(f"Loaded {len(self.train_examples)} train / {len(self.eval_examples)} eval examples")
        self.iter = 0

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_correct_buffer:
            wandb_metrics["train/tool_accuracy"] = sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
        self.percent_correct_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, example: Dict) -> Dict:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["user"]},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response = completion.choices[0].message.content
        predicted = _parse_tool_call(response)
        score = _score_tool_call(predicted, example["expected_tool"], example["expected_args"])
        return {"score": score, "response": response, "expected_tool": example["expected_tool"]}

    async def evaluate(self, *args, **kwargs):
        import time
        from tqdm.asyncio import tqdm_asyncio
        start = time.time()
        tasks = [self.rollout_and_score_eval(ex) for ex in self.eval_examples[:200]]
        results = await tqdm_asyncio.gather(*tasks)
        scores = [r["score"] for r in results]
        accuracy = sum(1 for s in scores if s == 1.0) / len(scores)
        self.eval_metrics.append(("eval/tool_accuracy", accuracy))
        await self.evaluate_log(
            metrics={"eval/tool_accuracy": accuracy},
            samples=[{"messages": [], "score": r["score"], "expected": r["expected_tool"]} for r in results[:10]],
            start_time=start,
            end_time=time.time(),
            generation_parameters={"temperature": 0.0, "max_tokens": self.config.max_token_length},
        )

    async def collect_trajectories(self, item: Dict) -> Tuple[ScoredDataGroup, list]:
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
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
                "expected_tool": item["expected_tool"],
                "expected_args": item["expected_args"],
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

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            response = item["messages"][-1]["content"]
            predicted = _parse_tool_call(response)
            reward = _score_tool_call(predicted, item["expected_tool"], item["expected_args"])

            masked_tokens = item["masked_tokens"]
            if len([t for t in masked_tokens if t != -100]) < 5:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(masked_tokens)
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(float(reward))
            self.percent_correct_buffer.append(float(reward == 1.0))

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores if scores["tokens"] else None

    async def get_next_item(self) -> Dict:
        item = self.train_examples[self.iter % len(self.train_examples)]
        self.iter += 1
        return item


if __name__ == "__main__":
    ToolUseEnv.cli()
