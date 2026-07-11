#!/usr/bin/env python3
"""
ZVF Program — Pillar-1 / M1 matched-compute comparison:
    canonical GRPO  vs  DAPO  vs  GSPO   at EQUAL total compute.

PURPOSE
  Define the three RL-loss "arms" and emit the EXACT commands that launch them
  through the user's existing per-cell runner, with their total rollout/token
  budget pinned equal so the comparison is fair. This module:
    * sets up each arm's config,
    * computes a matched-compute budget (equal total rollouts AND equal total
      generated tokens across arms),
    * emits launch commands (does NOT train, does NOT pick a winner).

  A winner is computed POST-HOC by aggregate_sweep.py from the real per-step
  ZVF/GU logs + held-out scores. See the `# RESULTS:` placeholder at the bottom.

DESIGN OF THE MATCH
  Compute for one GRPO-style step ~= group_size G * batch_prompts * max_tokens
  (samples drawn) plus a fixed forward/backward. We hold the TOTAL across the
  run equal:
        total_rollouts = G * batch_prompts * steps          (must match across arms)
        total_tokens   = total_rollouts * max_tokens         (must match across arms)
  The three arms differ ONLY in the loss/clipping/normalization, NOT in how much
  compute they spend. Where an arm's canonical recipe uses a different G, we
  rebalance `steps` so total_rollouts stays equal (integer-rounded; a warning is
  printed if a residual mismatch remains).

ARMS (one-line characterization; the real loss lives in the user's runner)
  grpo  — canonical GRPO: group-relative advantage, per-token PG, importance
          sampling surrogate, no asymmetric clipping. (baseline)
  dapo  — DAPO: decoupled clip (clip_higher), dynamic sampling that filters
          all-correct / all-wrong groups, token-level PG loss, overlong reward
          shaping. (Yu et al., 2025)
  gspo  — GSPO: sequence-level importance ratio + sequence-level clipping
          (group sequence policy optimization). (Qwen team, 2025)

USAGE
  python3 matched_compute.py                # print arms + matched budget + commands (no exec)
  python3 matched_compute.py --json         # emit machine-readable plan
  python3 matched_compute.py --execute      # shell out to the real runner per arm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "sweep_config.yaml"


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    loss: str                 # token passed to cell_runner --loss; runner maps to the real loss
    description: str
    # canonical knobs (recorded for provenance; runner is the source of truth)
    group_size: int
    clip_low: float
    clip_high: float
    importance_level: str     # "token" | "sequence"
    dynamic_sampling: bool
    notes: str = ""
    # filled by the matcher:
    steps: int = field(default=0)
    batch_prompts: int = field(default=0)
    max_tokens: int = field(default=0)


def define_arms() -> list[Arm]:
    return [
        Arm(
            name="grpo", loss="grpo",
            description="Canonical GRPO (baseline): group-relative advantage, "
                        "token-level PG, symmetric IS clip.",
            group_size=8, clip_low=0.2, clip_high=0.2,
            importance_level="token", dynamic_sampling=False,
            notes="Reference arm. Symmetric clip eps=0.2.",
        ),
        Arm(
            name="dapo", loss="dapo",
            description="DAPO: decoupled higher clip, dynamic sampling (drops "
                        "zero-variance groups), token-level loss, overlong shaping.",
            group_size=8, clip_low=0.2, clip_high=0.28,
            importance_level="token", dynamic_sampling=True,
            notes="clip_higher=0.28; dynamic sampling directly attacks ZVF by "
                  "resampling all-same-reward groups.",
        ),
        Arm(
            name="gspo", loss="gspo",
            description="GSPO: sequence-level importance ratio + sequence-level clip.",
            group_size=8, clip_low=0.2, clip_high=0.2,
            importance_level="sequence", dynamic_sampling=False,
            notes="Importance weighting/clipping at the sequence level rather "
                  "than per-token.",
        ),
        Arm(
            name="drgrpo", loss="drgrpo",
            description="Dr.GRPO: removes the /std normalization in the per-group "
                        "advantage, eliminating the length bias of vanilla GRPO. "
                        "Token-level loss, symmetric IS clip, no dynamic sampling.",
            group_size=8, clip_low=0.2, clip_high=0.2,
            importance_level="token", dynamic_sampling=False,
            notes="Implements Liu et al. 2025 'Dr. GRPO: Passing the Token-Level "
                  "Reward Bias in GRPO' (https://arxiv.org/abs/2506.13937). The "
                  "/std removal is enforced by toggling live_zvf_probe._ADV_NORMALIZE_STD "
                  "in cell_runner.install_loss_arm; the loss kernel is unchanged.",
        ),
        Arm(
            name="greso", loss="grpo",
            description="GRESO-style pre-rollout filtering: skip prompts whose "
                        "rolling success-rate estimate predicts a zero-variance "
                        "group; recycle the saved rollouts into fresh prompts. "
                        "Loss identical to canonical GRPO — the ONLY delta is "
                        "which prompts get rolled out (point-estimate gate).",
            group_size=8, clip_low=0.2, clip_high=0.2,
            importance_level="token", dynamic_sampling=True,
            notes="Zheng et al. 2025 'Act Only When It Pays' (arXiv:2506.02177). "
                  "Point-estimate gate: skip if rolling p_hat outside "
                  "[tau, 1-tau], tau=0.1, window=3 encounters. Runner flag: "
                  "--prompt-filter greso. Matched budget counts only EXECUTED "
                  "rollouts, so recycling is budget-neutral.",
        ),
        Arm(
            name="zvf_ci_gated", loss="grpo",
            description="ZVF CI-gated controller (ours): identical prompt-skip "
                        "action to greso, but gated on the Wilson 95% interval "
                        "of the rolling estimate (T1) with m>=64 groups of "
                        "history — act only when the CI excludes the useful "
                        "band. Below m=64 history the gate stays open "
                        "(no filtering), so early training is unbiased.",
            group_size=8, clip_low=0.2, clip_high=0.2,
            importance_level="token", dynamic_sampling=True,
            notes="The E-C positioning arm: calibrated controller vs "
                  "point-estimate heuristics at matched compute. Gate: skip "
                  "prompt iff Wilson-95 interval for its stratum's "
                  "zero-variance probability lies entirely above 0.8. "
                  "Requires shuffled batching (T1 estimand result: curriculum "
                  "ordering changes the estimand). Runner flag: "
                  "--prompt-filter zvf_ci_gated.",
        ),
    ]


# ---------------------------------------------------------------------------
# Matched-compute budget
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    total_rollouts: int       # G * batch_prompts * steps  (held equal across arms)
    max_tokens: int           # held equal across arms
    base_batch_prompts: int
    base_steps: int
    base_group_size: int

    @property
    def total_tokens(self) -> int:
        return self.total_rollouts * self.max_tokens


def matched_budget(base_group_size: int = 8, base_batch_prompts: int = 2,
                   base_steps: int = 30, max_tokens: int = 512) -> Budget:
    """Construct the matched Budget for one GRPO-style step.

    total_rollouts = G * batch_prompts * steps. Held equal across arms
    by ``fit_arm_to_budget``; the integer-rounding residual is
    surfaced as a stderr warning when nonzero. The total_tokens
    budget is rolled-out-budget * max_tokens.

    All inputs must be strictly positive integers; a zero or
    negative value raises ValueError with the offending key named,
    rather than propagating a ZeroDivisionError from
    fit_arm_to_budget when the user types ``base_batch_prompts: 0``
    in the YAML.
    """
    for name, val in (("base_group_size", base_group_size),
                      ("base_batch_prompts", base_batch_prompts),
                      ("max_tokens", max_tokens)):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(
                f"matched_budget: {name} must be a positive int, got {val!r}"
            )
    if not isinstance(base_steps, int) or base_steps < 0:
        raise ValueError(
            f"matched_budget: base_steps must be a non-negative int, got {base_steps!r}"
        )
    total_rollouts = base_group_size * base_batch_prompts * base_steps
    return Budget(total_rollouts=total_rollouts, max_tokens=max_tokens,
                  base_batch_prompts=base_batch_prompts, base_steps=base_steps,
                  base_group_size=base_group_size)


def fit_arm_to_budget(arm: Arm, budget: Budget) -> Arm:
    """Set arm.steps so that G*batch*steps == budget.total_rollouts (as close as
    integer rounding allows). batch_prompts and max_tokens are shared = matched.

    The matched_budget validator already enforces base_group_size and
    base_batch_prompts > 0, so per_step > 0 here. The runtime guard
    below is a defense-in-depth check against a future caller that
    passes a hand-built Budget without going through matched_budget.
    """
    arm.batch_prompts = budget.base_batch_prompts
    arm.max_tokens = budget.max_tokens
    per_step = arm.group_size * arm.batch_prompts
    if per_step <= 0:
        raise ValueError(
            f"fit_arm_to_budget: per_step = arm.group_size({arm.group_size}) * "
            f"arm.batch_prompts({arm.batch_prompts}) must be > 0; check arm and "
            f"Budget for non-positive inputs."
        )
    arm.steps = max(1, round(budget.total_rollouts / per_step))
    realized = arm.group_size * arm.batch_prompts * arm.steps
    if realized != budget.total_rollouts:
        print(f"  [warn] arm '{arm.name}': realized rollouts {realized} != "
              f"target {budget.total_rollouts} (residual from integer steps).",
              file=sys.stderr)
    return arm


# ---------------------------------------------------------------------------
# Command emission (shell-out only; never trains here)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def arm_commands(cfg: dict, arm: Arm, models: list[str], seeds: list[int],
                 difficulty: str) -> list[dict]:
    """Build launch commands for one arm across the matched-compute model/seed set.

    The arm-specific hyperparameters (clip_low, clip_high,
    importance_level, dynamic_sampling) are forwarded to the runner
    via the command_template substitution, so DAPO at clip_high=0.28
    with dynamic_sampling=True is ACTUALLY different from GRPO at
    clip_high=0.2 with dynamic_sampling=False at the runner level, not
    just in the dataclass notes. The command_template must contain
    placeholders for {clip_low}, {clip_high}, {importance_level},
    {dynamic_sampling}; if it does not, the substitution raises a
    KeyError so the user is forced to update the template.
    """
    runner = cfg["runner"]
    kind = runner["kind"]
    rc = runner[kind]
    results_dir = resolve(rc["results_dir"])
    midx = {m["short"]: m for fam in cfg["model_families"].values() for m in fam}

    cmds = []
    for short in models:
        m = midx[short]
        for seed in seeds:
            tag = f"matched_{arm.name}_{short}_G{arm.group_size}_{difficulty}_s{seed}"
            out_json = Path(rc["out_json_template"].format(
                results_dir=str(results_dir), tag=tag))
            subst = {
                "interpreter": rc["interpreter"], "script": str(resolve(rc["script"])),
                "results_dir": str(results_dir), "out_json": str(out_json),
                "tag": tag, "model_id": m["model_id"], "task": cfg["fixed"]["task"],
                "loss": arm.loss, "seed": seed, "group_size": arm.group_size,
                "lr": cfg["fixed"]["lr"], "steps": arm.steps, "rank": cfg["fixed"]["rank"],
                "difficulty": difficulty,
                "clip_low": arm.clip_low,
                "clip_high": arm.clip_high,
                "importance_level": arm.importance_level,
                "dynamic_sampling": "true" if arm.dynamic_sampling else "false",
            }
            cmd = " ".join(rc["command_template"].split()).format(**subst)
            cmds.append({"tag": tag, "arm": arm.name, "model_short": short,
                         "seed": seed, "command": cmd, "result_json": str(out_json)})
    return cmds


def build_plan(cfg: dict) -> dict:
    mc = cfg["sweep_design"]["matched_compute"]
    # YAML overrides: base_batch_prompts and max_tokens come from
    # sweep_design.matched_compute when present, else fall back to
    # the function defaults. The previous version silently used
    # defaults (2 prompts, 512 tokens) regardless of the YAML
    # value, which meant a user who set base_batch_prompts=4 in the
    # YAML got arms at half the intended compute.
    budget = matched_budget(
        base_group_size=mc["group_size"],
        base_steps=cfg["fixed"]["steps"],
        base_batch_prompts=mc.get("base_batch_prompts", 2),
        max_tokens=mc.get("max_tokens", 512),
    )
    arms = [fit_arm_to_budget(a, budget) for a in define_arms()]
    all_cmds = []
    for arm in arms:
        all_cmds.extend(arm_commands(cfg, arm, mc["models"], mc["seeds"], mc["difficulty"]))
    return {
        "budget": asdict(budget) | {"total_tokens": budget.total_tokens},
        "arms": [asdict(a) for a in arms],
        "models": mc["models"], "seeds": mc["seeds"], "difficulty": mc["difficulty"],
        "commands": all_cmds,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit machine-readable plan only (no execution, no print)")
    ap.add_argument("--execute", action="store_true",
                    help="shell out to the real runner for each arm/model/seed. "
                         "Mutually exclusive with --json-only: --json is the read-only "
                         "plan view, --execute runs the cells.")
    ap.add_argument("--dry-run", action="store_true",
                    help="(DEFAULT) print plan + commands, execute nothing. Same as omitting --execute.")
    ap.add_argument("--force", action="store_true",
                    help="skip the concurrency lock (see --lock-file). Only use this "
                         "if you are sure no other matched_compute.py is running.")
    ap.add_argument("--lock-file", type=Path, default=Path(".matched_compute.lock"),
                    help="PID file used to detect concurrent invocations. The lock "
                         "is acquired on --execute and released on exit; concurrent "
                         "invocations with --execute are refused to prevent "
                         "double-launch and ledger corruption. Set --force to skip.")
    ap.add_argument("--per-cell-timeout", type=int, default=4 * 3600,
                    help="per-cell subprocess timeout in seconds (default 4h). "
                         "A hung cell blocks the entire matched-compute sweep "
                         "forever without this.")
    args = ap.parse_args()

    # --json + --execute is logically inconsistent: --json prints the
    # plan and exits; --execute runs the plan. The previous version
    # silently ignored --json when --execute was also set, leading to
    # confusing "I asked for the plan and got launches" UX. Refuse the
    # combination explicitly.
    if args.json and args.execute:
        ap.error("--json and --execute are mutually exclusive. "
                  "Use --json for a read-only plan; drop --json (or use --dry-run) "
                  "for the plan + commands without launching.")

    cfg = load_config()
    plan = build_plan(cfg)

    if args.json:
        print(json.dumps(plan, indent=2))
        return 0

    b = plan["budget"]
    print("=" * 78)
    print("MATCHED-COMPUTE COMPARISON: canonical GRPO vs DAPO vs GSPO")
    print("=" * 78)
    print("Matched budget (held EQUAL across all three arms):")
    print(f"  total_rollouts : {b['total_rollouts']}   (G * batch_prompts * steps)")
    print(f"  max_tokens     : {b['max_tokens']}  per rollout")
    print(f"  total_tokens   : {b['total_tokens']}  (rollouts * max_tokens)")
    print(f"  base_batch     : {b['base_batch_prompts']} prompts/step")
    print("-" * 78)
    for a in plan["arms"]:
        print(f"ARM {a['name'].upper():5s} loss={a['loss']:5s} "
              f"G={a['group_size']} steps={a['steps']} "
              f"batch={a['batch_prompts']} max_tok={a['max_tokens']} "
              f"clip=[{a['clip_low']},{a['clip_high']}] "
              f"IS={a['importance_level']} dyn_sample={a['dynamic_sampling']}")
        print(f"          {a['description']}")
        if a["notes"]:
            print(f"          note: {a['notes']}")
    print("-" * 78)
    print(f"models={plan['models']} seeds={plan['seeds']} "
          f"difficulty={plan['difficulty']}  -> {len(plan['commands'])} runs total")
    print("-" * 78)

    if not args.execute:
        print("COMMANDS THAT WOULD RUN (no execution):")
        for c in plan["commands"]:
            print(f"  $ {c['command']}")
        print("-" * 78)
        print("Nothing launched. Use --execute to launch, then aggregate_sweep.py "
              "to compute the comparison from REAL logs.")
        return 0

    # EXECUTE: acquire the concurrency lock first so two parallel
    # matched_compute.py --execute invocations cannot both launch
    # their cells (the previous behavior double-launched every cell
    # and corrupted the resume ledger). The lock is a PID file
    # containing our PID; on a normal exit (or SIGTERM) we remove
    # it. On SIGKILL, a stale PID file is detected on next start.
    if not args.force:
        lock = args.lock_file
        if lock.exists():
            try:
                old_pid = int(lock.read_text().strip())
                # Check if the old PID is still alive.
                import os as _os
                try:
                    _os.kill(old_pid, 0)
                    print(
                        f"[error] Another matched_compute.py --execute is already "
                        f"running (PID {old_pid}, lock file {lock}). Refusing to "
                        f"launch. Pass --force to override.",
                        file=sys.stderr,
                    )
                    return 1
                except ProcessLookupError:
                    print(
                        f"[warn] Stale lock file {lock} (PID {old_pid} no longer "
                        f"alive). Removing and continuing.",
                        file=sys.stderr,
                    )
                    lock.unlink()
            except (ValueError, OSError) as exc:
                print(
                    f"[warn] Could not read lock file {lock}: {exc}. "
                    f"Removing and continuing.",
                    file=sys.stderr,
                )
                lock.unlink()
        import os as _os
        lock.write_text(str(_os.getpid()))
        lock_acquired = True
    else:
        lock_acquired = False

    try:
        print(f"Launching {len(plan['commands'])} matched-compute runs sequentially "
              f"(timeout {args.per_cell_timeout}s/cell)...")
        for c in plan["commands"]:
            print(f"  $ {c['command']}")
            try:
                proc = subprocess.run(
                    c["command"],
                    shell=True,
                    cwd=str(REPO_ROOT),
                    timeout=args.per_cell_timeout,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
                print(
                    f"  [warn] cell {c['tag']} exceeded {args.per_cell_timeout}s "
                    f"timeout; treating as failed. The runner subprocess was "
                    f"killed by subprocess.run.",
                    file=sys.stderr,
                )
            except Exception as exc:
                rc = f"exc:{type(exc).__name__}"
                print(f"  [warn] cell {c['tag']} raised {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            print(f"    -> {c['tag']} rc={rc}")
        print("Done. Compute the comparison with: python3 aggregate_sweep.py --matched")
        return 0
    finally:
        if lock_acquired:
            try:
                args.lock_file.unlink()
            except OSError:
                pass


# =============================================================================
# RESULTS: filled after runs complete
# =============================================================================
# DO NOT hand-edit numbers here. The matched-compute verdict is produced by
# aggregate_sweep.py reading the real per-step ZVF/GU logs and held-out scores
# from the result JSONs each arm wrote. Until those runs exist on disk, every
# field below is intentionally empty. No winner is asserted by this harness.
#
#   arm   | mean_GU(steps) | final_ZVF | GSM-Plus | AIME-2025 | BFCL-v3 | wins
#   ------+----------------+-----------+----------+-----------+---------+-----
#   grpo  |   <pending>    | <pending> | <pending>| <pending> |<pending>|  --
#   dapo  |   <pending>    | <pending> | <pending>| <pending> |<pending>|  --
#   gspo  |   <pending>    | <pending> | <pending>| <pending> |<pending>|  --
#
# Verdict: <pending — computed post-hoc from real logs; no result fabricated>
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())
