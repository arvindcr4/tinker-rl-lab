#!/usr/bin/env python3
"""E13 OpenReward games -> Tinker LoRA RL training driver.

BUILD ONLY. The default mode is ``plan``: it validates every gate and prints a
cost projection without constructing a Tinker client or spending anything.
The paid path is behind ``--execute`` plus an explicit spend acknowledgement,
and still refuses to start if the projection exceeds the operational cap.

What this driver does
---------------------
* Trains an authorized Tinker model with LoRA on E13's **train** split.
* Evaluates on the **held-out** split, seeds ``seed_idx + 10000``.
* Uses the environment's own native reward. No shaped proxy, no surrogate.
* Enforces train/held-out separation with hard assertions at every seed use.

The split firewall
------------------
A leak here invalidates the whole campaign, so separation is enforced three
times over, in code, not by convention:

1. ``SplitFirewall`` refuses to construct unless ``prove_seed_separation``
   holds over the two manifests.
2. Every training rollout calls ``assert_train_seed``; every evaluation rollout
   calls ``assert_eval_seed``. Each raises ``SplitLeakError`` on a wrong-side
   seed, including a train seed that merely *looks* held-out.
3. The firewall records every seed it admits, and ``assert_no_leak`` re-checks
   the admitted train and eval sets are disjoint before a checkpoint is exported.

Licensing
---------
The pinned environment source has **no LICENSE file**. This driver does not
claim one. It emits ``observed_state: absent_at_pinned_revision`` with
``claimed_spdx: null`` and cites the owner's recorded risk acceptance as
``proceeding_under``. The validator is fail-closed and points at that record:
if the record disappears, the driver stops.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # package import (PYTHONPATH=zvf-program, `python -m flagship....`)
    from flagship.e13_openreward_games_local_runner import (
        EpisodeRecord,
        GameTaskSpec,
        ProgrammaticRewardVerifier,
        SplitManifest,
        VerifierOutcome,
        parse_split_manifest,
        prove_seed_separation,
    )
    from flagship.tinker_model_policy import assert_candidate_allowed
except ImportError:  # direct import with the flagship dir itself on sys.path
    from e13_openreward_games_local_runner import (
        EpisodeRecord,
        GameTaskSpec,
        ProgrammaticRewardVerifier,
        SplitManifest,
        VerifierOutcome,
        parse_split_manifest,
        prove_seed_separation,
    )
    from tinker_model_policy import assert_candidate_allowed

SCHEMA_VERSION = "e13-openreward-games-tinker-train-v1"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

#: Upstream's held-out offset. Train seeds are seed_idx; held-out are +10000.
HOLDOUT_SEED_BASE = 10000

BUDGET_PATH = HERE / "pavlov_tinker_budget.json"
LICENSE_RISK_RECORD = REPO_ROOT / "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md"

#: Conservative chars-per-token. English averages ~4; 3.5 overestimates token
#: counts, which is the direction we want when projecting spend.
CHARS_PER_TOKEN = 3.5

#: Measured from real local episodes at EnvCommons/wordle@92bea32e.
#: Worst case is Wordle-v0 / Wordle-v0-hardcore: 7 turns.
MEASURED_EPISODE_PROFILE = {
    "source": "outputs/e13_openreward_games/manifests/episode_token_profile.json",
    "worst_case_variant": "Wordle-v0",
    "prompt_chars": 748,
    "total_observation_chars": 5097,
    "max_turns": 7,
}


class SplitLeakError(RuntimeError):
    """Raised when a seed is used on the wrong side of the train/held-out split."""


class BudgetError(RuntimeError):
    """Raised when a projected or actual spend would breach the operational cap."""


class LicenseRecordError(RuntimeError):
    """Raised when the license risk-acceptance record is missing or unreadable."""


# --------------------------------------------------------------------------
# License: report the observed state, never invent an SPDX identifier
# --------------------------------------------------------------------------


def license_record(record_path: Path = LICENSE_RISK_RECORD) -> dict[str, Any]:
    """Fail-closed license block.

    The pinned EnvCommons revision carries no LICENSE file. This never returns
    an SPDX identifier; it returns the observed absence plus the owner's
    recorded acceptance. If the record is gone, work stops rather than
    silently proceeding unlicensed.
    """

    if not record_path.is_file():
        raise LicenseRecordError(
            f"license risk-acceptance record not found at {record_path}. "
            "The pinned environment source has no LICENSE file, so work cannot "
            "proceed without the owner's recorded acceptance."
        )
    text = record_path.read_text(encoding="utf-8")
    if "EnvCommons" not in text:
        raise LicenseRecordError(
            f"{record_path} does not cover EnvCommons; refusing to proceed."
        )
    return {
        "asset": "EnvCommons/wordle",
        "pinned_revision": "92bea32efa102e86275dedd2e0367e86d3754754",
        "observed_state": "absent_at_pinned_revision",
        "claimed_spdx": None,
        "proceeding_under": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
        "decision": "owner_risk_acceptance_2026-08-09",
    }


# --------------------------------------------------------------------------
# Split firewall
# --------------------------------------------------------------------------


class SplitFirewall:
    """Hard, stateful enforcement that train and held-out seeds never cross."""

    def __init__(self, train: SplitManifest, evaluation: SplitManifest) -> None:
        proof = prove_seed_separation(train, evaluation)
        if not proof.holds:
            raise SplitLeakError(
                "refusing to build a firewall over non-separated manifests: "
                + "; ".join(proof.violations)
            )
        self.proof = proof
        self.train = train
        self.evaluation = evaluation
        self._train_keys = {(t.variant, t.seed) for t in train.tasks}
        self._eval_keys = {(t.variant, t.seed) for t in evaluation.tasks}
        self.admitted_train: set[tuple[str, int]] = set()
        self.admitted_eval: set[tuple[str, int]] = set()

    def assert_train_seed(self, task: GameTaskSpec) -> GameTaskSpec:
        key = (task.variant, task.seed)
        if task.seed >= HOLDOUT_SEED_BASE:
            raise SplitLeakError(
                f"training rollout attempted on seed {task.seed} (>= {HOLDOUT_SEED_BASE}); "
                "that is the held-out range"
            )
        if key not in self._train_keys:
            raise SplitLeakError(f"{key} is not in the train manifest")
        if key in self._eval_keys:
            raise SplitLeakError(f"{key} appears in the held-out manifest; refusing to train on it")
        self.admitted_train.add(key)
        return task

    def assert_eval_seed(self, task: GameTaskSpec) -> GameTaskSpec:
        key = (task.variant, task.seed)
        if task.seed < HOLDOUT_SEED_BASE:
            raise SplitLeakError(
                f"held-out evaluation attempted on seed {task.seed} (< {HOLDOUT_SEED_BASE}); "
                "that is the train range"
            )
        if key not in self._eval_keys:
            raise SplitLeakError(f"{key} is not in the held-out manifest")
        if key in self._train_keys:
            raise SplitLeakError(f"{key} appears in the train manifest; refusing to evaluate on it")
        self.admitted_eval.add(key)
        return task

    def assert_no_leak(self) -> None:
        """Re-check admitted sets before anything derived from them is exported."""
        overlap = self.admitted_train & self.admitted_eval
        if overlap:
            raise SplitLeakError(
                f"{len(overlap)} instance(s) were used for BOTH training and held-out "
                f"evaluation: {sorted(overlap)[:10]}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "holdout_seed_base": HOLDOUT_SEED_BASE,
            "separation_proof": self.proof.as_dict(),
            "admitted_train_instances": len(self.admitted_train),
            "admitted_eval_instances": len(self.admitted_eval),
            "admitted_overlap": len(self.admitted_train & self.admitted_eval),
        }


# --------------------------------------------------------------------------
# Native reward — the environment's own verifier, never a proxy
# --------------------------------------------------------------------------


@dataclass
class NativeGameReward:
    """Reward source bound to the real environment's terminal reward.

    The value returned is exactly what the OpenReward environment produced,
    passed through the fail-closed ``ProgrammaticRewardVerifier``. There is no
    shaping, no partial credit, and no fallback: an episode the verifier
    rejects contributes no reward rather than a guessed one.
    """

    verifier: ProgrammaticRewardVerifier = field(default_factory=ProgrammaticRewardVerifier)
    rejected: int = 0

    def score(self, episode: EpisodeRecord) -> tuple[float | None, VerifierOutcome]:
        outcome = self.verifier.verify(episode)
        if not outcome.accepted:
            self.rejected += 1
            return None, outcome
        return outcome.reward, outcome


def normalize_group_rewards(rewards: Sequence[float]) -> list[float]:
    """GRPO advantage: centre and scale within the sample group."""
    usable = [r for r in rewards if r is not None]
    if not usable:
        return [0.0] * len(rewards)
    mean = sum(usable) / len(usable)
    var = sum((r - mean) ** 2 for r in usable) / len(usable)
    std = var ** 0.5
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [0.0 if r is None else (r - mean) / std for r in rewards]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class E13TrainConfig:
    name: str = "e13_openreward_wordle_grpo"
    model: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 32
    seed: int = 1337

    steps: int = 20
    batch_size: int = 4          # tasks per update
    group_size: int = 4          # samples per task (GRPO group)
    lr: float = 1e-5

    max_prompt_tokens: int = 4096
    max_response_tokens: int = 256
    max_turns: int = 7
    temperature: float = 1.0
    top_p: float = 1.0

    wandb_project: str = "tinker-rl-lab-pavlov"
    wandb_entity: str | None = None
    wandb_group: str | None = "e13-openreward-games"
    wandb_tags: tuple[str, ...] = ("e13", "openreward", "wordle", "grpo", "lora")

    eval_tasks: int = 20         # held-out tasks sampled for eval/reward
    eval_every: int = 10

    environment: str = "GeneralReasoning/Wordle"
    source_revision: str = "92bea32efa102e86275dedd2e0367e86d3754754"
    checkpoint_dir: Path = REPO_ROOT / "outputs/e13_openreward_games/checkpoints"
    base_url: str | None = None

    def validate(self) -> None:
        assert_candidate_allowed(self.model)
        for fieldname in ("steps", "batch_size", "group_size", "max_response_tokens", "max_turns"):
            if getattr(self, fieldname) <= 0:
                raise ValueError(f"{fieldname} must be positive")
        if self.wandb_project != "tinker-rl-lab-pavlov":
            raise ValueError(
                "wandb_project is a contract requirement and must be 'tinker-rl-lab-pavlov'"
            )


# --------------------------------------------------------------------------
# Cost projection
# --------------------------------------------------------------------------


def load_budget(path: Path = BUDGET_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _chars_to_tokens(chars: float) -> int:
    return int(chars / CHARS_PER_TOKEN + 0.5)


def project_episode_tokens(cfg: E13TrainConfig,
                           profile: Mapping[str, Any] = MEASURED_EPISODE_PROFILE) -> dict[str, int]:
    """Token accounting for ONE episode, charged conservatively.

    Every turn re-sends the whole conversation as uncached prefill and is
    charged the full ``max_response_tokens`` regardless of what the model
    actually emits.
    """

    turns = min(int(profile["max_turns"]), cfg.max_turns)
    prompt_tokens = _chars_to_tokens(profile["prompt_chars"])
    obs_tokens_total = _chars_to_tokens(profile["total_observation_chars"])
    obs_per_turn = obs_tokens_total / max(turns, 1)

    prefill = 0
    sample = 0
    for turn in range(turns):
        context = prompt_tokens + int(obs_per_turn * turn) + cfg.max_response_tokens * turn
        prefill += min(context, cfg.max_prompt_tokens)
        sample += cfg.max_response_tokens

    # Training sees the final full sequence once per sampled trajectory.
    train_tokens = min(prompt_tokens + obs_tokens_total, cfg.max_prompt_tokens) \
        + cfg.max_response_tokens * turns
    return {
        "turns": turns,
        "prefill_tokens": prefill,
        "sample_tokens": sample,
        "train_tokens": train_tokens,
    }


def project_cost(cfg: E13TrainConfig,
                 *,
                 episodes_sampled: int,
                 episodes_trained: int,
                 budget: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project USD for a given number of sampled and trained episodes."""

    b = dict(budget or load_budget())
    price = b["usd_per_million_tokens"]
    per = project_episode_tokens(cfg)

    prefill_tok = per["prefill_tokens"] * episodes_sampled
    sample_tok = per["sample_tokens"] * episodes_sampled
    train_tok = per["train_tokens"] * episodes_trained

    prefill_usd = prefill_tok / 1e6 * price["prefill"]
    sample_usd = sample_tok / 1e6 * price["sample"]
    train_usd = train_tok / 1e6 * price["train"]
    total = prefill_usd + sample_usd + train_usd

    return {
        "episodes_sampled": episodes_sampled,
        "episodes_trained": episodes_trained,
        "per_episode": per,
        "tokens": {"prefill": prefill_tok, "sample": sample_tok, "train": train_tok},
        "usd": {
            "prefill": round(prefill_usd, 4),
            "sample": round(sample_usd, 4),
            "train": round(train_usd, 4),
            "total": round(total, 4),
        },
        "pricing_source": b.get("pricing_source"),
        "accounting": "conservative: uncached prefill every turn, full max_response_tokens charged",
    }


def assert_within_cap(projection: Mapping[str, Any],
                      *,
                      already_spent_usd: float = 0.0,
                      budget: Mapping[str, Any] | None = None) -> None:
    b = dict(budget or load_budget())
    cap = float(b["operational_cap_usd"])
    reserve = float(b["safety_reserve_usd"])
    spendable = cap - reserve
    total = float(projection["usd"]["total"]) + already_spent_usd
    if total > spendable:
        raise BudgetError(
            f"projected ${total:.2f} exceeds spendable ${spendable:.2f} "
            f"(operational cap ${cap:.2f} minus ${reserve:.2f} reserve that must stay unspent)"
        )


# --------------------------------------------------------------------------
# W&B — online, and BEFORE any Tinker client exists
# --------------------------------------------------------------------------


def start_wandb_online(cfg: E13TrainConfig) -> Any:
    """Initialize an online W&B run. Mirrors grpo.py:_start_wandb invariants.

    Called before the Tinker client is constructed, so a tracking failure costs
    nothing. Fail-closed on every degraded mode W&B can return.
    """

    env_mode = os.environ.get("WANDB_MODE")
    if env_mode and env_mode.strip().lower() != "online":
        raise RuntimeError("W&B tracking requires WANDB_MODE=online")
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("W&B tracking is disabled by WANDB_DISABLED")
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(f"W&B dependency is unavailable: {exc}") from exc

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=f"{cfg.name}_seed{cfg.seed}",
        tags=list(cfg.wandb_tags),
        mode="online",
        config=asdict_config(cfg),
        reinit=True,
    )
    if run is None or not getattr(run, "id", None):
        raise RuntimeError("W&B online initialization returned no live run")
    mode = getattr(run, "mode", None)
    if mode is not None and str(mode).lower() != "online":
        raise RuntimeError("W&B online initialization returned a non-online run")
    if bool(getattr(run, "disabled", False)) or bool(getattr(run, "offline", False)):
        raise RuntimeError("W&B online initialization returned a disabled run")
    return run


def asdict_config(cfg: E13TrainConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["checkpoint_dir"] = str(cfg.checkpoint_dir)
    d["wandb_tags"] = list(cfg.wandb_tags)
    return d


def wandb_log(run: Any, payload: Mapping[str, Any]) -> None:
    log = getattr(run, "log", None)
    if not callable(log):
        raise RuntimeError("W&B run has no log method; receipt is inadmissible")
    if log(dict(payload)) is False:
        raise RuntimeError("W&B log was rejected; receipt is inadmissible")


# --------------------------------------------------------------------------
# Checkpoint export — shaped for e11_paid_run_driver.py --sampler-path
# --------------------------------------------------------------------------

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def build_checkpoint_record(
    *,
    cfg: E13TrainConfig,
    sampler_path: str,
    step: int,
    firewall: SplitFirewall,
    hf_repo: str | None = None,
    hf_revision: str | None = None,
    hf_commit: str | None = None,
) -> dict[str, Any]:
    """Emit a checkpoint record consumable by the E11 transfer probe.

    ``e11_paid_run_driver.py`` requires ``--sampler-path`` together with
    ``--hf-repo``, ``--hf-revision`` and a 40-hex ``--hf-commit``. Publishing to
    the Hub needs ``HF_TOKEN``, which is absent, so the HF fields stay null and
    the record reports the probe as not yet satisfiable rather than pretending.
    """

    firewall.assert_no_leak()
    if hf_commit is not None and not _HEX40.fullmatch(hf_commit):
        raise ValueError("hf_commit must be an immutable 40-hex commit")

    e11_ready = all((sampler_path, hf_repo, hf_revision, hf_commit))
    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": cfg.name,
        "model": cfg.model,
        "lora_rank": cfg.lora_rank,
        "step": step,
        "sampler_path": sampler_path,
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "hf_commit": hf_commit,
        "trained_on": {
            "environment": cfg.environment,
            "source_revision": cfg.source_revision,
            "split": "train",
            "holdout_seed_base": HOLDOUT_SEED_BASE,
        },
        "split_firewall": firewall.as_dict(),
        "license": license_record(),
        "e11_transfer_probe": {
            "ready": e11_ready,
            "driver": "outputs/e11_verilog_eval/e11_paid_run_driver.py",
            "command": (
                f"python outputs/e11_verilog_eval/e11_paid_run_driver.py "
                f"--sampler-path {sampler_path} --hf-repo {hf_repo} "
                f"--hf-revision {hf_revision} --hf-commit {hf_commit}"
            ) if e11_ready else None,
            "blocker": None if e11_ready else
                "HF_TOKEN is absent, so the sampler cannot be published to an "
                "immutable HF revision; --hf-repo/--hf-revision/--hf-commit "
                "cannot be supplied.",
        },
    }
    return record


# --------------------------------------------------------------------------
# Rollout (paid path — lazily imports tinker)
# --------------------------------------------------------------------------


def rollout_episode(
    *,
    task: GameTaskSpec,
    make_env: Callable[[GameTaskSpec], Any],
    sample_fn: Callable[[str], str],
    parse_action: Callable[[str], str],
    max_turns: int,
) -> EpisodeRecord:
    """Run one multi-turn episode against the real environment.

    ``sample_fn`` maps a prompt to a model response; ``make_env`` builds the
    real OpenReward environment. The terminal reward is whatever the
    environment returns — this function never computes a reward itself.
    """

    import asyncio

    async def _run() -> EpisodeRecord:
        env = make_env(task)
        blocks = await env.get_prompt()
        conversation = blocks[0].text
        steps = 0
        finished = False
        terminal_reward: float | None = None

        for _ in range(max_turns):
            response = sample_fn(conversation)
            action = parse_action(response)
            result = await env.guess_word(_guess_params(env, action))
            steps += 1
            conversation += f"\n{response}\n{result.blocks[0].text}"
            if result.finished:
                finished = True
                terminal_reward = float(result.reward)
                break
        return EpisodeRecord(
            task=task,
            steps=steps,
            finished=finished,
            terminal_reward=terminal_reward,
        )

    return asyncio.run(_run())


def _guess_params(env: Any, action: str) -> Any:
    """Build the env's own tool-parameter model, whatever it is called."""
    module = type(env).__module__
    import importlib
    mod = importlib.import_module(module)
    for name in ("GuessParams",):
        if hasattr(mod, name):
            return getattr(mod, name)(word=action)
    raise RuntimeError(f"cannot find a tool parameter model in {module}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load(path: Path) -> SplitManifest:
    return parse_split_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def plan(cfg: E13TrainConfig, firewall: SplitFirewall) -> dict[str, Any]:
    """Validate every gate and project cost. Spends nothing."""

    cfg.validate()
    budget = load_budget()

    smoke = project_cost(cfg, episodes_sampled=1, episodes_trained=1, budget=budget)

    pilot_sampled = cfg.steps * cfg.batch_size * cfg.group_size
    pilot_eval = cfg.eval_tasks * max(1, cfg.steps // max(cfg.eval_every, 1))
    pilot = project_cost(cfg,
                         episodes_sampled=pilot_sampled + pilot_eval,
                         episodes_trained=pilot_sampled,
                         budget=budget)

    n_train = len(firewall.train.tasks)
    full_sampled = n_train * cfg.group_size
    full = project_cost(cfg,
                        episodes_sampled=full_sampled + cfg.eval_tasks,
                        episodes_trained=full_sampled,
                        budget=budget)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "plan",
        "spent_usd": 0.0,
        "model": cfg.model,
        "model_policy": "assert_candidate_allowed passed",
        "config": asdict_config(cfg),
        "split_firewall": firewall.as_dict(),
        "license": license_record(),
        "measured_episode_profile": MEASURED_EPISODE_PROFILE,
        "projections": {
            "one_smoke_episode": smoke,
            "short_pilot": dict(pilot, shape={
                "updates": cfg.steps, "batch_size": cfg.batch_size,
                "group_size": cfg.group_size,
                "train_episodes": pilot_sampled, "eval_episodes": pilot_eval,
            }),
            "full_pass_200_train_tasks": dict(full, shape={
                "train_tasks": n_train, "group_size": cfg.group_size,
                "train_episodes": full_sampled, "eval_episodes": cfg.eval_tasks,
            }),
        },
        "budget": {
            "operational_cap_usd": budget["operational_cap_usd"],
            "safety_reserve_usd": budget["safety_reserve_usd"],
            "spendable_usd": budget["operational_cap_usd"] - budget["safety_reserve_usd"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="e13_openreward_games_tinker_train",
        description="Plan (default) or execute E13 OpenReward-games LoRA RL training on Tinker.",
    )
    default_manifests = REPO_ROOT / "outputs/e13_openreward_games/manifests"
    ap.add_argument("--train-manifest", type=Path, default=default_manifests / "wordle_train.json")
    ap.add_argument("--eval-manifest", type=Path, default=default_manifests / "wordle_test.json")
    ap.add_argument("--model", default=E13TrainConfig.model)
    ap.add_argument("--steps", type=int, default=E13TrainConfig.steps)
    ap.add_argument("--batch-size", type=int, default=E13TrainConfig.batch_size)
    ap.add_argument("--group-size", type=int, default=E13TrainConfig.group_size)
    ap.add_argument("--out", type=Path, default=None, help="write the plan/receipt JSON here")
    ap.add_argument("--execute", action="store_true",
                    help="PAID. Construct a Tinker client and train. Requires --i-accept-spend.")
    ap.add_argument("--i-accept-spend", action="store_true",
                    help="explicit spend acknowledgement; required with --execute")
    args = ap.parse_args(argv)

    cfg = E13TrainConfig(
        model=args.model, steps=args.steps,
        batch_size=args.batch_size, group_size=args.group_size,
    )
    firewall = SplitFirewall(_load(args.train_manifest), _load(args.eval_manifest))
    result = plan(cfg, firewall)

    if args.execute:
        if not args.i_accept_spend:
            print("REFUSED: --execute requires --i-accept-spend.")
            return 2
        assert_within_cap(result["projections"]["short_pilot"])
        print("Gates passed. The paid path is intentionally not wired in this build; "
              "enable it only under an explicit spend authorization.")
        return 3

    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    p = result["projections"]
    print(f"model              : {result['model']}")
    print(f"separation holds   : {result['split_firewall']['separation_proof']['holds']}")
    print(f"license            : {result['license']['observed_state']} "
          f"(claimed_spdx={result['license']['claimed_spdx']})")
    print(f"smoke  (1 episode) : ${p['one_smoke_episode']['usd']['total']:.4f}")
    print(f"pilot              : ${p['short_pilot']['usd']['total']:.2f}  "
          f"({p['short_pilot']['shape']['updates']} updates x "
          f"{p['short_pilot']['shape']['batch_size']} batch x "
          f"{p['short_pilot']['shape']['group_size']} group)")
    print(f"full 200-task pass : ${p['full_pass_200_train_tasks']['usd']['total']:.2f}")
    print(f"spendable          : ${result['budget']['spendable_usd']:.2f}")
    print(f"SPENT THIS RUN     : ${result['spent_usd']:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
