#!/usr/bin/env python3
"""Offline smoke test for the zvf-triage integration in live_zvf_probe.py.

Feeds SYNTHETIC per-step grouped rewards through the *same* integration glue the
live probe uses (``load_zvf_controller`` + the per-prompt group-id construction +
``ZVFController.step``) and prints the real controller decisions and the auto-stop
firing.

It does NOT call the Tinker API and does NOT invent any training metrics: every
number printed is computed by the zvf-triage package from the synthetic rewards.

Run::

    zvf-program/zvf-triage/.venv/bin/python \
        experiments/tinker-runs/smoke_zvf_triage_probe.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve()
PROBE_PATH = HERE.parent / "live_zvf_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("live_zvf_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top-level imports are light (no tinker/torch)
    return module


def _build_batch(probe_args, group_rewards):
    """Replicate the probe's flat-rewards + group-id construction.

    ``group_rewards`` is a list of per-prompt reward lists for one step. Mirrors
    the probe loop: ``batch_rewards.extend(rewards)`` /
    ``batch_group_ids.extend([prompt_idx] * len(rewards))``.
    """
    batch_rewards: list[float] = []
    batch_group_ids: list[int] = []
    for prompt_idx, rewards in enumerate(group_rewards):
        batch_rewards.extend(rewards)
        batch_group_ids.extend([prompt_idx] * len(rewards))
    return batch_rewards, batch_group_ids


def main() -> int:
    probe = _load_probe_module()

    # Build the same argparse Namespace the probe would; only the zvf-* fields and
    # --group matter for the controller.
    args = argparse.Namespace(
        group=2,
        use_zvf_triage=True,
        zvf_window=5,
        zvf_max=0.85,
        zvf_var_eps=1e-6,
        zvf_stop_k=3,
        zvf_drop_k=3,
        zvf_adaptive_g=True,
        zvf_gmax=32,
    )

    controller = probe.load_zvf_controller(args)
    if controller is None:
        print("FAIL: load_zvf_controller returned None (package not importable).")
        return 1
    print(f"controller loaded: {type(controller).__module__}.{type(controller).__name__}")
    print(f"  config: zvf_max={args.zvf_max} var_eps={args.zvf_var_eps} "
          f"stop_k={args.zvf_stop_k} adaptive_G={args.zvf_adaptive_g} "
          f"G0={args.group} Gmax={args.zvf_gmax}\n")

    # --- Synthetic per-step grouped rewards -------------------------------
    # Each row = one optimizer step; each sublist = one prompt-group's rollouts.
    #   step 1: mixed group -> within-group contrast -> ZVF=0, exploitable
    #   step 2: all-correct group (zero variance) -> ZVF=1, saturation. NOTE the
    #           global-collapse streak begins here (ZVF==1), independent of regime.
    #   steps 2-4: ZVF==1 for stop_k=3 consecutive steps -> AUTO-STOP at step 4.
    #   step 5 onward never runs (probe breaks on auto_stop).
    steps = [
        ("mixed contrast",      [[1.0, 0.0], [1.0, 0.0]]),  # ZVF=0
        ("all-correct (sat)",   [[1.0, 1.0], [1.0, 1.0]]),  # ZVF=1, reward=1 (streak=1)
        ("all-zero collapse 1", [[0.0, 0.0], [0.0, 0.0]]),  # ZVF=1, reward=0 (streak=2)
        ("all-zero collapse 2", [[0.0, 0.0], [0.0, 0.0]]),  # ZVF=1, reward=0 (streak=3) -> stop
        ("should-not-run",      [[1.0, 0.0], [1.0, 0.0]]),  # break before this
    ]

    auto_stopped_at = None
    header = (f"{'step':>4}  {'label':<20} {'zvf':>5} {'gu':>5} "
              f"{'reward':>6} {'regime':<22} {'signal':<14} {'G*':>3} {'stop':>5}")
    print(header)
    print("-" * len(header))

    for i, (label, group_rewards) in enumerate(steps, start=1):
        batch_rewards, batch_group_ids = _build_batch(args, group_rewards)
        # This is exactly the call the probe makes per step.
        decision = controller.step(batch_rewards, batch_group_ids)
        print(
            f"{decision.step:>4}  {label:<20} "
            f"{decision.zvf:>5.2f} {decision.gu:>5.2f} "
            f"{decision.mean_reward:>6.2f} "
            f"{decision.regime.value:<22} {decision.signal:<14} "
            f"{decision.group_size:>3} {str(decision.auto_stop):>5}"
        )
        # Probe behaviour: break on auto_stop.
        if decision.auto_stop:
            auto_stopped_at = decision.step
            print(f"\n>>> AUTO-STOP fired at step {decision.step} "
                  f"(global ZVF==1 for stop_k={args.zvf_stop_k} steps). "
                  f"Probe would break here.")
            break

    print()
    # --- Assertions (offline correctness of the integrated path) ----------
    decisions = controller.zvf_history
    checks = []
    checks.append(("step1 ZVF==0 (mixed)", decisions[0] == 0.0))
    checks.append(("step2 ZVF==1 (all-correct)", decisions[1] == 1.0))
    checks.append(("auto-stop fired at step 4 (3-step ZVF==1 streak)",
                   auto_stopped_at == 4))
    checks.append(("controller.stopped True", controller.stopped is True))

    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    print("\nSMOKE TEST:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
