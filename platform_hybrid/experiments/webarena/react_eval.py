"""
True multi-turn ReAct eval for BrowserGym (MiniWoB / WebArena-verified).

Per episode:
  1. env.reset -> obs
  2. loop:
       feed (goal + axtree_txt + last_action_error) to Tinker SamplingClient
       sample one action
       env.step(action)
       break on terminated/truncated or max_steps
  3. reward = task's terminal reward (max over episode as safety net)

Designed for parallel execution: pass --shard k/N to process only every N-th task
starting at offset k. Outputs one JSONL per shard; aggregate_webarena.py combines.

Usage:
    export TINKER_API_KEY=...
    export MINIWOB_URL=http://localhost:8000/miniwob/   # MiniWoB only
    export WEBARENA_SHOPPING_URL=...                    # WebArena only (etc.)
    python -m experiments.webarena.react_eval \\
        --benchmark miniwob --model Qwen/Qwen3-8B \\
        --tasks miniwob.choose-list,miniwob.click-button \\
        --max-steps 10 --group-size 1 \\
        --out results_shard_0.jsonl --shard 0/1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from numbers import Real
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("react_eval")

SYSTEM_PROMPT = """You are a browser automation agent. You will be shown a browser page as an accessibility tree and a goal. Emit exactly ONE action per turn using BrowserGym's HighLevelActionSet (bid-based):

  click('bid')              fill('bid', 'text')         select_option('bid', 'value')
  hover('bid')              press('bid', 'Enter')       clear('bid')
  goto('url')               go_back()                   go_forward()
  new_tab()                 tab_close()                 tab_focus(index)
  send_msg_to_user('text')  report_infeasible('reason')

Format each response as:
Thought: <one sentence>
Action: <exactly one call from above>

Do NOT emit multiple actions. Do NOT explain beyond one Thought line. If the task is done, emit send_msg_to_user with the final answer."""


USER_TEMPLATE = """Goal:
{goal}

Accessibility tree:
{axtree}

{error_block}Step {step}/{max_steps}. Emit the next single action."""


@dataclass
class StepRecord:
    step: int
    observation_len: int
    thought: str
    action: str
    reward: float
    terminated: bool
    truncated: bool
    error: Optional[str]


@dataclass
class EpisodeResult:
    env_id: str
    seed: int
    score: float | None       # None means invalid native evidence, never a failure score
    total_reward: float | None
    terminated: bool
    truncated: bool
    num_steps: int
    valid_action_count: int
    steps: List[StepRecord]
    wall_time_sec: float
    error: Optional[str] = None
    evaluation_id: str = ""


# ---------------------------------------------------------------------------
# BrowserGym helpers
# ---------------------------------------------------------------------------

def _import_benchmark(benchmark: str) -> None:
    """Lazy-register the gym environment ids for a benchmark."""
    if benchmark == "miniwob":
        import browsergym.miniwob  # noqa: F401
    elif benchmark == "webarena":
        import browsergym.webarena
    elif benchmark == "webarena_verified":
        import browsergym.webarena_verified
    elif benchmark == "workarena":
        import browsergym.workarena  # noqa: F401
    elif benchmark == "visualwebarena":
        import browsergym.visualwebarena  # noqa: F401
    elif benchmark == "assistantbench":
        import browsergym.assistantbench  # noqa: F401
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _make_env(env_id: str, headless: bool = True) -> Any:
    import gymnasium as gym
    try:
        return gym.make(env_id, headless=headless)
    except TypeError:
        return gym.make(env_id)


def _axtree_to_str(obs: Dict[str, Any], max_chars: int) -> str:
    """Flatten axtree to text with a hard char cap."""
    try:
        from browsergym.utils.obs import flatten_axtree_to_str
    except ImportError:
        # Very old browsergym layout
        from browsergym.core.observation import flatten_axtree_to_str  # type: ignore
    txt = flatten_axtree_to_str(obs["axtree_object"])
    if len(txt) > max_chars:
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        txt = head + "\n...[truncated]...\n" + tail
    return txt


def _goal_to_str(obs: Dict[str, Any]) -> str:
    goal = obs.get("goal")
    if goal:
        return str(goal)
    go = obs.get("goal_object")
    if go:
        return " ".join(str(x.get("text", x)) for x in go if isinstance(x, dict))
    return ""


ACTION_RE = __import__("re").compile(r"Action:\s*(.+?)(?:$|\n)", __import__("re").DOTALL)
THOUGHT_RE = __import__("re").compile(r"Thought:\s*(.+?)(?:$|\n)", __import__("re").DOTALL)


def _parse_response(text: str) -> tuple[str, Optional[str]]:
    """Extract (thought, action_str). Returns (thought, None) if no action found."""
    thought_m = THOUGHT_RE.search(text)
    action_m = ACTION_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    action = action_m.group(1).strip() if action_m else None
    if action:
        # Strip trailing code fences / backticks
        action = action.strip("` \n")
        # Take only the first line of the action
        action = action.split("\n")[0].strip()
    return thought, action


# ---------------------------------------------------------------------------
# Tinker sampling wrapper
# ---------------------------------------------------------------------------

class TinkerChatSampler:
    """Minimal wrapper: render chat messages -> tokens -> sc.sample -> decode."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512) -> None:
        import tinker
        from tinker.types import SamplingParams, ModelInput
        from transformers import AutoTokenizer
        self._SamplingParams = SamplingParams
        self._ModelInput = ModelInput
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        service = tinker.ServiceClient()
        self.sc = service.create_sampling_client(base_model=model)
        self._has_async = hasattr(self.sc, "sample_async")

    def sample(self, messages: List[Dict[str, str]]) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text)
        model_input = self._ModelInput.from_ints(prompt_ids)
        stop: list = []
        if self.tokenizer.eos_token_id is not None:
            stop.append(self.tokenizer.eos_token_id)
        params = self._SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=stop,
        )
        # Synchronous sampling only. Playwright sync API (used by BrowserGym)
        # cannot be called from within an asyncio event loop, so the whole
        # episode driver runs without asyncio. Tinker's .sample() returns a
        # future; block on .result() directly.
        future = self.sc.sample(
            prompt=model_input, num_samples=1, sampling_params=params,
        )
        result = future.result(timeout=120)
        completion_ids = result.sequences[0].tokens
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

class InvalidNativeRewardError(ValueError):
    """The native outcome cannot be interpreted as benchmark evidence."""


def run_episode(
    sampler: TinkerChatSampler,
    env_id: str,
    seed: int,
    max_steps: int,
    max_axtree_chars: int,
    benchmark: str,
) -> EpisodeResult:
    t0 = time.time()
    # Playwright sync API is incompatible with running from an asyncio executor
    # worker thread: it either errors with "cannot switch to a different thread"
    # (when different threads serve different calls) or "no running event loop"
    # (when the worker thread has no asyncio loop and Playwright tries to bootstrap).
    # With --concurrency 1 we can call env.reset/step/close synchronously from
    # this coroutine. They block the event loop briefly, but no other episode is
    # running concurrently in this process, so that is acceptable. The only awaits
    # needed are around sampler.sample() (LLM HTTP) which stays truly async.
    env = None
    steps: List[StepRecord] = []
    total_reward = 0.0
    max_reward = 0.0
    terminated = False
    truncated = False
    valid_action_count = 0
    last_error: Optional[str] = None
    try:
        _import_benchmark(benchmark)
        env = _make_env(env_id)
        obs, info = env.reset(seed=seed)
        goal = _goal_to_str(obs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for step in range(1, max_steps + 1):
            axtree = _axtree_to_str(obs, max_axtree_chars)
            error_block = f"Last action error:\n{last_error}\n\n" if last_error else ""
            user_msg = USER_TEMPLATE.format(
                goal=goal, axtree=axtree, error_block=error_block,
                step=step, max_steps=max_steps,
            )
            messages.append({"role": "user", "content": user_msg})
            response = sampler.sample(messages)
            messages.append({"role": "assistant", "content": response})
            thought, action = _parse_response(response)
            step_error: Optional[str] = None
            if not action:
                step_error = "no_action_parsed"
                steps.append(StepRecord(step, len(axtree), thought, "", 0.0, False, False, step_error))
                last_error = step_error
                break
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward) or not 0 <= reward <= 1:
                    raise InvalidNativeRewardError(f"invalid native reward: {reward!r}")
                reward_f = float(reward)
                step_error = obs.get("last_action_error") or info.get("action_exec_error")
                if not step_error:
                    valid_action_count += 1
                total_reward += reward_f
                max_reward = max(max_reward, reward_f)
                last_error = obs.get("last_action_error") or info.get("action_exec_error")
            except InvalidNativeRewardError:
                raise
            except Exception as exc:
                step_error = repr(exc)
                last_error = step_error
                reward_f = 0.0
            steps.append(StepRecord(
                step=step, observation_len=len(axtree), thought=thought, action=action,
                reward=reward_f, terminated=terminated, truncated=truncated, error=step_error,
            ))
            if terminated or truncated:
                break
            if len(messages) > 14:
                messages = [messages[0]] + messages[-12:]
    except Exception as exc:
        logger.exception("Episode crashed env_id=%s seed=%s: %r", env_id, seed, exc)
        try:
            env.close()
        except Exception as e:
            logger.warning("Failed to close env after crash env_id=%s: %s", env_id, e)
        return EpisodeResult(
            env_id=env_id, seed=seed,
            score=None if isinstance(exc, InvalidNativeRewardError) else 0.0,
            total_reward=None if isinstance(exc, InvalidNativeRewardError) else 0.0,
            terminated=False, truncated=False, num_steps=len(steps),
            valid_action_count=valid_action_count, steps=steps,
            wall_time_sec=time.time() - t0, error=repr(exc),
        )
    finally:
        try:
            env.close()
        except Exception as e:
            logger.warning("Failed to close env env_id=%s: %s", env_id, e)
    if not terminated and not truncated:
        truncated = True
    # Versioned maximum task-reward contract; never sum partial rewards into success.
    score = max_reward
    return EpisodeResult(
        env_id=env_id, seed=seed, score=score, total_reward=total_reward,
        terminated=terminated, truncated=truncated, num_steps=len(steps),
        valid_action_count=valid_action_count, steps=steps,
        wall_time_sec=time.time() - t0, error=last_error,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_tasks(spec: str, benchmark: str) -> List[str]:
    """Accept either a comma list or 'all' for benchmark's registered tasks."""
    import gymnasium as gym
    # webarena_verified is a semantic alias; tasks are registered under browsergym/webarena.
    prefix_name = "webarena" if benchmark == "webarena_verified" else benchmark
    if spec == "all":
        prefix = f"browsergym/{prefix_name}"
        return sorted(i for i in gym.envs.registry.keys() if i.startswith(prefix))
    out: List[str] = []
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        if not t.startswith("browsergym/"):
            t = f"browsergym/{t}"
        out.append(t)
    return out


def _shard(tasks: List[str], shard_spec: str) -> List[str]:
    """'k/N' -> take every N-th task starting at index k."""
    k, n = shard_spec.split("/")
    k, n = int(k), int(n)
    assert 0 <= k < n, f"invalid shard {shard_spec}"
    return tasks[k::n]


def _main_sync(args: argparse.Namespace) -> int:
    _import_benchmark(args.benchmark)
    tasks = _parse_tasks(args.tasks, args.benchmark)
    if args.concurrency != 1:
        raise ValueError("Only sequential execution (--concurrency 1) is supported")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    config = {k: getattr(args, k) for k in ("benchmark", "model", "seed", "max_steps", "max_axtree_chars", "action_tokens", "temperature")}
    config.update(protocol="browsergym-max-task-reward-v2", tasks=tasks)
    evaluation_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    seeds = {task: int.from_bytes(hashlib.sha256(f"{args.seed}:{task}".encode()).digest()[:4], "big") for task in tasks}
    tasks = _shard(tasks, args.shard)
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info("shard=%s tasks=%d benchmark=%s model=%s",
                args.shard, len(tasks), args.benchmark, args.model)

    # --- Optional: Weights & Biases ---
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            run_name = f"{args.benchmark}-{args.model.replace('/', '_')}-shard{args.shard.replace('/', 'of')}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name or run_name,
                group=args.run_name or args.benchmark,
                config={
                    "benchmark": args.benchmark, "model": args.model,
                    "max_steps": args.max_steps, "shard": args.shard,
                    "temperature": args.temperature, "concurrency": args.concurrency,
                    "max_axtree_chars": args.max_axtree_chars, "num_tasks": len(tasks),
                },
                tags=["webarena-eval", args.benchmark],
                reinit=True,
            )
            logger.info("wandb run: %s", wandb_run.url if wandb_run else None)
        except Exception as exc:
            logger.warning("wandb init failed (%r); continuing without wandb", exc)
            wandb_run = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = {"n": 0, "sum_score": 0.0, "sum_steps": 0, "invalid_scores": 0}

    def one(env_id: str, seed: int, idx: int) -> None:
            res = run_episode(
                sampler, env_id, seed,
                max_steps=args.max_steps,
                max_axtree_chars=args.max_axtree_chars,
                benchmark=args.benchmark,
            )
            res.evaluation_id = evaluation_id
            with out_path.open("a") as f:
                f.write(json.dumps(asdict(res), default=str) + "\n")
            completed["n"] += 1
            completed["invalid_scores"] += int(res.score is None)
            completed["sum_score"] += res.score if res.score is not None else 0.0
            completed["sum_steps"] += res.num_steps
            logger.info(
                "done[%d/%d] env=%s score=%s steps=%d err=%s",
                completed["n"], len(tasks), env_id, res.score, res.num_steps, res.error,
            )
            if wandb_run is not None:
                try:
                    import wandb as _wb  # noqa: F401
                    wandb_run.log(
                        {
                            "episode/score": res.score,
                            "episode/num_steps": res.num_steps,
                            "episode/valid_actions": res.valid_action_count,
                            "episode/wall_time_sec": res.wall_time_sec,
                            "episode/terminated": int(res.terminated),
                            "episode/truncated": int(res.truncated),
                            "episode/error": int(bool(res.error)),
                            "running/mean_score": None if completed["invalid_scores"] else completed["sum_score"] / completed["n"],
                            "running/mean_steps": completed["sum_steps"] / completed["n"],
                            "running/completed": completed["n"],
                        },
                        step=idx,
                    )
                except Exception as exc:
                    logger.warning("wandb.log failed: %r", exc)

    if out_path.exists() and not args.resume:
        raise ValueError("output already exists; choose a new versioned path or --resume")
    done_ids: set = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_id") != evaluation_id or row.get("seed") != seeds.get(row["env_id"]) or row["env_id"] not in tasks:
                raise ValueError("resume configuration mismatch; use a new output path")
            if row["env_id"] in done_ids:
                raise ValueError("duplicate resume episode")
            # Explicit policy: retain all recorded attempts, including errors.
            done_ids.add(row["env_id"])
            completed["n"] += 1
            completed["invalid_scores"] += int(row["score"] is None)
            completed["sum_score"] += row["score"] if row["score"] is not None else 0.0
            completed["sum_steps"] += row["num_steps"]
        logger.info("resuming: %d already done", len(done_ids))

    receipt_path = out_path.with_suffix(out_path.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps({"evaluation_id": evaluation_id, "effective_config": config, "actual_concurrency": 1, "sampling_mode": "synchronous", "sampling_timeout_seconds": 120, "coverage": [{"evaluation_id": evaluation_id, "env_id": task, "seed": seeds[task]} for task in tasks]}, indent=2))

    sampler = TinkerChatSampler(
        model=args.model, temperature=args.temperature, max_tokens=args.action_tokens,
    )

    for i, env_id in enumerate(tasks):
        if env_id in done_ids:
            continue
        one(env_id, seed=seeds[env_id], idx=i)

    # --- Optional: HuggingFace Hub upload ---
    if args.hf_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi()
            create_repo(
                args.hf_repo, repo_type="dataset", exist_ok=True, private=args.hf_private,
            )
            remote_name = f"{args.run_name or args.benchmark}/{out_path.name}"
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=remote_name,
                repo_id=args.hf_repo,
                repo_type="dataset",
                commit_message=f"results: {args.benchmark} shard {args.shard} model={args.model}",
            )
            logger.info("uploaded %s -> hf://%s/%s", out_path, args.hf_repo, remote_name)
        except Exception as exc:
            logger.warning("HF upload failed (%r); local file intact at %s", exc, out_path)

    if wandb_run is not None:
        try:
            wandb_run.summary["final/n_completed"] = completed["n"]
            wandb_run.summary["final/mean_score"] = (
                None if completed["invalid_scores"] else completed["sum_score"] / completed["n"] if completed["n"] else 0.0
            )
            wandb_run.summary["final/shard"] = args.shard
            # Log the JSONL as a W&B artifact for later reruns/diffs
            import wandb as _wb
            art = _wb.Artifact(
                name=f"results-{args.benchmark}-{args.shard.replace('/', '-of-')}",
                type="eval-results",
            )
            art.add_file(str(out_path))
            wandb_run.log_artifact(art)
            wandb_run.finish()
        except Exception as exc:
            logger.warning("wandb finalize failed: %r", exc)

    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["miniwob", "webarena", "webarena_verified",
                            "workarena", "visualwebarena", "assistantbench"])
    p.add_argument("--tasks", default="all",
                   help="Comma-separated task names, or 'all' for benchmark's full registry")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0/1", help="'k/N' -> every N-th task from k")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-axtree-chars", type=int, default=24000)
    p.add_argument("--action-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max concurrent browsergym envs / Chromiums")
    p.add_argument("--resume", action="store_true",
                   help="Skip env_ids already present in --out")
    p.add_argument("--log-level", default="INFO")

    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"),
                   help="Enable W&B logging to this project (requires WANDB_API_KEY)")
    p.add_argument("--run-name", default=os.environ.get("WANDB_RUN_NAME"),
                   help="Override default W&B run name / HF subdir")
    p.add_argument("--hf-repo", default=os.environ.get("HF_RESULTS_REPO"),
                   help="Upload results to this HF dataset repo (e.g. 'user/webarena-results'). Requires HF_TOKEN")
    p.add_argument("--hf-private", action="store_true",
                   help="Create/use private HF repo")

    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not os.environ.get("TINKER_API_KEY") and not Path(os.path.expanduser("~/.tinker_api_key")).exists():
        logger.error("TINKER_API_KEY not set and ~/.tinker_api_key missing")
        return 2
    return _main_sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
