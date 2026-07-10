"""
BrowserGym/WebArena GRPO Environment for Atropos + Tinker.

This environment turns browser tasks into verifiable GRPO rollouts. The model
generates a short BrowserGym action script from the initial observation. The
environment executes those actions in Chromium via BrowserGym and uses the
benchmark reward as the scalar training signal.

Start with MiniWoB for cheap local smoke tests, then switch the same env to
WebArena once the WebArena sites are running and configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from tinker_atropos.config import TinkerAtroposConfig

logger = logging.getLogger(__name__)


def _get_config_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get(
        "TINKER_CONFIG_PATH", "configs/browsergym_miniwob_qwen_8b_smoke.yaml"
    )


CONFIG_PATH = _get_config_path()

SYSTEM_PROMPT = """You are a browser-control agent.

You will receive a BrowserGym task goal and a text observation from the current
browser page. Return a short action script, one BrowserGym action per line.

Allowed actions:
- click('bid')
- dblclick('bid')
- fill('bid', 'text')
- clear('bid')
- select_option('bid', 'option')
- press('bid', 'Enter')
- focus('bid')
- hover('bid')
- scroll(0, 500)
- scroll(0, -500)
- keyboard_press('Enter')
- keyboard_type('text')
- noop()

Use only element ids that appear in the observation, such as 'a12'. Do not
write explanations, markdown, JSON, or Python code. If the task appears already
complete, output noop()."""

USER_TEMPLATE = """Task goal:
{goal}

Browser observation:
{observation}

Return only BrowserGym actions, one per line."""

ALLOWED_ACTIONS = {
    "click",
    "dblclick",
    "fill",
    "clear",
    "select_option",
    "press",
    "focus",
    "hover",
    "scroll",
    "keyboard_press",
    "keyboard_type",
    "noop",
}

ACTION_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>.*)\)\s*$"
)


def _import_browsergym(benchmark: str) -> None:
    """Import benchmark package so Gymnasium registers env IDs."""
    import browsergym.core  # noqa: F401

    benchmark = benchmark.lower().strip()
    if benchmark == "miniwob":
        import browsergym.miniwob  # noqa: F401
    elif benchmark in {"webarena", "visualwebarena", "webarena_verified"}:
        import browsergym.webarena  # noqa: F401
    elif benchmark == "workarena":
        import browsergym.workarena  # noqa: F401
    else:
        raise ValueError(f"Unsupported BrowserGym benchmark: {benchmark}")


def _clean_action_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = re.sub(r"^(Action|Next action)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip("`").strip()


def _extract_actions(text: str, max_steps: int) -> List[str]:
    """Extract safe BrowserGym action strings from model output."""
    text = re.sub(r"```(?:python|text)?\s*", "", text)
    text = text.replace("```", "")
    actions: List[str] = []
    for raw_line in text.splitlines():
        line = _clean_action_line(raw_line)
        if not line:
            continue
        match = ACTION_RE.match(line)
        if not match:
            continue
        if match.group("name") not in ALLOWED_ACTIONS:
            continue
        actions.append(line)
        if len(actions) >= max_steps:
            break
    return actions


def _shorten(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _stringify_observation(obs: Any, max_chars: int) -> Tuple[str, str]:
    """Return (goal, observation_text) from BrowserGym's observation dict."""
    if not isinstance(obs, dict):
        return "", _shorten(obs, max_chars)

    goal_parts: List[str] = []
    for key in ("goal", "task", "intent", "instruction"):
        if key in obs and obs[key]:
            goal_parts.append(f"{key}: {obs[key]}")

    observation_parts: List[str] = []
    priority_keys = [
        "url",
        "open_pages_urls",
        "axtree_txt",
        "accessibility_tree",
        "pruned_html",
        "text",
        "chat_messages",
        "screenshot",
    ]
    for key in priority_keys:
        if key in obs and obs[key] is not None:
            if key == "screenshot":
                observation_parts.append("screenshot: <image omitted in text-only GRPO prompt>")
            else:
                observation_parts.append(f"{key}:\n{_shorten(obs[key], max_chars)}")

    if not observation_parts:
        for key, value in obs.items():
            if value is None:
                continue
            observation_parts.append(f"{key}:\n{_shorten(value, max_chars)}")
            if sum(len(part) for part in observation_parts) >= max_chars:
                break

    goal = "\n".join(goal_parts)
    observation = "\n\n".join(observation_parts)
    return _shorten(goal, max_chars // 3), _shorten(observation, max_chars)


def _make_browsergym_env(env_id: str, cfg: TinkerAtroposConfig):
    import gymnasium as gym

    try:
        return gym.make(env_id, headless=cfg.browsergym_headless)
    except TypeError:
        return gym.make(env_id)


def _run_browsergym_episode(
    env_id: str,
    action_text: str,
    cfg: TinkerAtroposConfig,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a model action script in BrowserGym and return score metadata."""
    _import_browsergym(cfg.browsergym_benchmark)
    env = _make_browsergym_env(env_id, cfg)
    actions = _extract_actions(action_text, cfg.browsergym_max_browser_steps)

    total_reward = 0.0
    max_reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}

    try:
        obs, info = env.reset(seed=seed)
        if not actions:
            return {
                "score": 0.0,
                "actions": [],
                "valid_action_count": 0,
                "total_reward": 0.0,
                "max_reward": 0.0,
                "terminated": False,
                "truncated": False,
                "error": "no_valid_actions",
                "info": _shorten(info, 1000),
            }

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward or 0.0)
            total_reward += reward_f
            max_reward = max(max_reward, reward_f)
            if terminated or truncated:
                break
    except Exception as exc:
        logger.exception(
            "BrowserGym episode failed (env_id=%s, seed=%s): %r",
            env_id, seed, exc,
        )
        return {
            "score": 0.0,
            "actions": actions,
            "valid_action_count": len(actions),
            "total_reward": total_reward,
            "max_reward": max_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "error": repr(exc),
            "info": _shorten(info, 1000),
        }
    finally:
        env.close()

    score = max(max_reward, total_reward)
    score = max(0.0, min(1.0, float(score)))
    return {
        "score": score,
        "actions": actions,
        "valid_action_count": len(actions),
        "total_reward": total_reward,
        "max_reward": max_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "error": None,
        "info": _shorten(info, 1000),
    }


def _build_initial_item(env_id: str, cfg: TinkerAtroposConfig, seed: int) -> Dict[str, Any]:
    """Reset a BrowserGym env once to build the prompt observation."""
    _import_browsergym(cfg.browsergym_benchmark)
    env = _make_browsergym_env(env_id, cfg)
    try:
        obs, info = env.reset(seed=seed)
        goal, observation = _stringify_observation(
            obs, cfg.browsergym_observation_max_chars
        )
        return {
            "env_id": env_id,
            "seed": seed,
            "goal": goal or _shorten(info, 1500),
            "observation": observation,
        }
    finally:
        env.close()


class BrowserGymEnv(BaseEnv):
    """BrowserGym/WebArena browser-control environment trained with GRPO."""

    name = "browsergym"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_success_buffer: List[float] = []
        self.reward_buffer: List[float] = []
        self.action_count_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.items: List[Dict[str, Any]] = []
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
        if full_cfg is None:
            raise RuntimeError("BrowserGymEnv requires TinkerAtroposConfig")

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

        env_ids = full_cfg.browsergym_env_ids
        if not env_ids:
            if full_cfg.browsergym_benchmark == "miniwob":
                env_ids = ["browsergym/miniwob.choose-list"]
            elif full_cfg.browsergym_benchmark == "webarena":
                env_ids = ["browsergym/webarena.10"]
            else:
                raise ValueError("browsergym_env_ids must be set for this benchmark")

        print(
            f"Preparing BrowserGym benchmark={full_cfg.browsergym_benchmark} "
            f"env_ids={env_ids}"
        )
        self.items = []
        for offset, env_id in enumerate(env_ids):
            item = await asyncio.to_thread(
                _build_initial_item, env_id, full_cfg, full_cfg.data_seed + offset
            )
            self.items.append(item)
        random.Random(full_cfg.data_seed).shuffle(self.items)
        self.iter = 0

    async def wandb_log(self, wandb_metrics=None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_success_buffer:
            wandb_metrics["train/browser_success_rate"] = sum(
                self.percent_success_buffer
            ) / len(self.percent_success_buffer)
        if self.reward_buffer:
            wandb_metrics["train/browser_reward_mean"] = sum(
                self.reward_buffer
            ) / len(self.reward_buffer)
        if self.action_count_buffer:
            wandb_metrics["train/browser_action_count_mean"] = sum(
                self.action_count_buffer
            ) / len(self.action_count_buffer)

        self.percent_success_buffer = []
        self.reward_buffer = []
        self.action_count_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, item: Dict[str, Any]) -> Dict[str, Any]:
        full_cfg = getattr(self.__class__, "_full_config", None)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    goal=item["goal"], observation=item["observation"]
                ),
            },
        ]
        completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        response = completion.choices[0].message.content
        result = await asyncio.to_thread(
            _run_browsergym_episode,
            item["env_id"],
            response,
            full_cfg,
            item["seed"],
        )
        return {"response": response, "env_id": item["env_id"], **result}

    async def evaluate(self, *args, **kwargs):
        import time
        from tqdm.asyncio import tqdm_asyncio

        full_cfg = getattr(self.__class__, "_full_config", None)
        start = time.time()
        eval_count = max(1, min(len(self.items), full_cfg.browsergym_eval_episodes))
        eval_items = self.items[:eval_count]
        results = []
        for item in tqdm_asyncio(eval_items):
            results.append(await self.rollout_and_score_eval(item))
        scores = [float(r["score"]) for r in results]
        success_rate = sum(1 for s in scores if s >= 1.0) / len(scores)
        reward_mean = sum(scores) / len(scores)
        action_count = sum(r["valid_action_count"] for r in results) / len(results)

        self.eval_metrics.append(("eval/browser_success_rate", success_rate))
        self.eval_metrics.append(("eval/browser_reward_mean", reward_mean))
        self.eval_metrics.append(("eval/browser_action_count_mean", action_count))
        await self.evaluate_log(
            metrics={
                "eval/browser_success_rate": success_rate,
                "eval/browser_reward_mean": reward_mean,
                "eval/browser_action_count_mean": action_count,
            },
            samples=[
                {
                    "messages": [],
                    "score": r["score"],
                    "env_id": r["env_id"],
                    "actions": r["actions"],
                    "error": r["error"],
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    goal=item["goal"], observation=item["observation"]
                ),
            },
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

        rollout_group_data = []
        for choice, node in zip(completions.choices, nodes):
            rollout_group_data.append(
                {
                    "messages": (
                        *messages,
                        {"role": "assistant", "content": choice.message.content},
                    ),
                    "env_id": item["env_id"],
                    "seed": item["seed"],
                    "tokens": node.tokens,
                    "masked_tokens": node.masked_tokens,
                    "logprobs": node.logprobs,
                }
            )
        return await self.score(rollout_group_data), []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        full_cfg = getattr(self.__class__, "_full_config", None)
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            response = item["messages"][-1]["content"]
            result = await asyncio.to_thread(
                _run_browsergym_episode,
                item["env_id"],
                response,
                full_cfg,
                item["seed"],
            )
            reward = float(result["score"])

            masked_tokens = item["masked_tokens"]
            if len([t for t in masked_tokens if t != -100]) < 3:
                continue

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(masked_tokens)
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(reward)

            self.percent_success_buffer.append(float(reward >= 1.0))
            self.reward_buffer.append(reward)
            self.action_count_buffer.append(float(result["valid_action_count"]))

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores if scores["tokens"] else None

    async def get_next_item(self):
        item = self.items[self.iter % len(self.items)]
        self.iter += 1
        return item


if __name__ == "__main__":
    BrowserGymEnv.cli()
