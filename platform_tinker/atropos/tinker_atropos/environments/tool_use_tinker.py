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


def _score_tool_call_v2(
    raw_text: str,
    predicted: Optional[Dict],
    expected_name: str,
    expected_args: Dict,
) -> float:
    """
    Shaped 6-level reward designed to break ZVF saturation (counterfactual vs v1):
      0.0  -> no JSON object at all in output
      0.2  -> valid JSON but no recognizable tool/name field
      0.4  -> valid JSON with a tool/name field, but wrong tool
      0.6  -> correct tool name, arguments block missing or non-dict
      0.8  -> correct tool name + dict args, but at least one required arg wrong/missing
      1.0  -> correct tool name and every required arg matches
    Partial credit is assigned by *how many* required args are matched when > 0.
    Keeps v1 semantics at the endpoints (fail=0, perfect=1) so comparisons remain fair.
    """
    # level 0: no JSON detected at all
    if predicted is None:
        has_brace = "{" in raw_text and "}" in raw_text
        return 0.0 if not has_brace else 0.0

    # level 1: JSON but no tool/name key
    pred_name_raw = predicted.get("tool", predicted.get("name", ""))
    if not isinstance(pred_name_raw, str) or not pred_name_raw.strip():
        return 0.2

    pred_name = pred_name_raw.strip().lower()
    exp_name = expected_name.strip().lower()
    # level 2: wrong tool
    if pred_name != exp_name:
        return 0.4

    pred_args = predicted.get(
        "arguments", predicted.get("parameters", predicted.get("args", None))
    )
    # level 3: right tool, missing/non-dict args
    if not isinstance(pred_args, dict):
        return 0.6

    # partial credit across required args
    required = list(expected_args.items())
    if not required:
        return 1.0
    matches = 0
    for key, val in required:
        if key in pred_args and str(pred_args[key]).lower().strip() == str(val).lower().strip():
            matches += 1

    if matches == len(required):
        return 1.0
    # level 4: right tool + dict args, not all matching
    # scale inside [0.8, 1.0) by fraction matched so GRPO sees within-group variance
    frac = matches / len(required)
    return 0.8 + 0.19 * frac  # max 0.99 when all-but-one match; 0.8 when zero match


_REWARD_VERSION = os.environ.get("TOOL_USE_REWARD_VERSION", "v1").strip().lower()


def _score(raw_text: str, predicted: Optional[Dict], expected_name: str, expected_args: Dict) -> float:
    """Dispatch to v1 (binary-ish) or v2 (6-level shaped) based on env var."""
    if _REWARD_VERSION == "v2":
        return _score_tool_call_v2(raw_text, predicted, expected_name, expected_args)
    return _score_tool_call(predicted, expected_name, expected_args)


_JSON_DECODER = json.JSONDecoder()


def _extract_json_objects(text):
    """Pull all top-level JSON objects from a blob. The glaive-v2 'system' field
    lists one or more function schemas as raw JSON after 'Use them if required -'."""
    marker = "Use them if required -"
    idx = text.find(marker)
    seg = text[idx + len(marker):] if idx != -1 else text
    objs = []
    j = seg.find("{")
    while j != -1:
        try:
            obj, end = _JSON_DECODER.raw_decode(seg[j:])
            objs.append(obj)
            j = seg.find("{", j + end)
        except json.JSONDecodeError:
            j = seg.find("{", j + 1)
    return objs


def _build_glaive_examples(raw_dataset, max_examples: int = 5000):
    """
    Parse glaive-function-calling-v2 into (system_with_tools, user_query, tool_name, tool_args) tuples.
    Real format: 'system' is "SYSTEM: ... Use them if required -\n{function json}" (raw JSON,
    NOT wrapped in <tools> tags); 'chat' contains "<functioncall> {...} <|endoftext|>"
    (no closing tag; `arguments` is a single-quoted JSON string).
    """
    examples = []
    for item in raw_dataset:
        try:
            system_text = item.get("system", "")
            chat_text = item.get("chat", "")
            # Tool schemas: raw JSON object(s) after the "Use them if required -" marker.
            tools = _extract_json_objects(system_text)
            if not tools:
                continue

            # First user turn + first function call from chat.
            user_match = re.search(r"USER:\s*(.*?)(?=ASSISTANT:|$)", chat_text, re.DOTALL)
            # Real format has no closing tag; the call ends at <|endoftext|> or end-of-string.
            func_match = re.search(
                r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|</functioncall>|$)",
                chat_text, re.DOTALL,
            )
            if not user_match or not func_match:
                continue
            user_query = user_match.group(1).strip()

            # glaive wraps `arguments` in single quotes -> outer blob is invalid JSON.
            # Extract name + arguments-string directly instead of json.loads on the whole blob.
            blob = func_match.group(1)
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
            args_match = (
                re.search(r'"arguments"\s*:\s*[\'"](\{.*\})[\'"]', blob, re.DOTALL)
                or re.search(r'"arguments"\s*:\s*(\{.*\})', blob, re.DOTALL)
            )
            if not name_match or not args_match:
                continue
            func_name = name_match.group(1)
            try:
                func_args = json.loads(args_match.group(1))
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
        if not self.train_examples:
            raise RuntimeError(
                "Parsed 0 training examples from glaive-function-calling-v2 — dataset "
                "format may have changed (check _build_glaive_examples regexes)."
            )
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
        score = _score(response, predicted, example["expected_tool"], example["expected_args"])
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
            reward = _score(response, predicted, item["expected_tool"], item["expected_args"])

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
