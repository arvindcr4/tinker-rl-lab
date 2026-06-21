#!/usr/bin/env python3
"""End-to-end ZVF triage demo on SYNTHETIC GRPO rollouts (numpy only).

No GPU, no network, no torch/trl. We fabricate a per-step stream of grouped
binary rollout rewards that *drifts* from a healthy run (lots of within-group
contrast -> low ZVF) into a cold-start collapse (every group uniform -> ZVF == 1),
feed it through :class:`zvf_triage.ZVFCallback` (which wraps the controller +
panel), and print:

  * the per-step regime and how it transitions,
  * adaptive group-size (G) changes as ZVF rises,
  * per-prompt drops and the global auto-stop firing.

Run::

    python examples/quickstart.py
    python examples/quickstart.py --seed 7 --steps 24

The model behind the synthetic data: each prompt-group ``g`` has a latent solve
probability ``p_g(t)``. At ``K`` rollouts per group, a group is zero-variance
(contributes no GRPO gradient) iff all ``K`` rollouts agree -- probability
``p_g^K + (1 - p_g)^K``. As training "drifts", every ``p_g`` is pushed toward 0
(nothing gets solved), so groups collapse to all-zeros and ZVF -> 1: the
canonical cold-start collapse the triage callback is built to catch early.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from zvf_triage import ZVFCallback, ZVFPanel
from zvf_triage.controller import Regime


def simulate_step(
    rng: np.random.Generator,
    solve_probs: np.ndarray,
    group_size: int,
) -> Tuple[List[float], List[int]]:
    """Sample one GRPO step of binary rollout rewards.

    For each prompt-group ``g`` draw ``group_size`` Bernoulli(``solve_probs[g]``)
    rollout rewards. Returns a flat reward list and the matching group-id list.
    """
    rewards: List[float] = []
    group_ids: List[int] = []
    for g, p in enumerate(solve_probs):
        draws = (rng.random(group_size) < p).astype(float)
        rewards.extend(draws.tolist())
        group_ids.extend([g] * group_size)
    return rewards, group_ids


def drifting_solve_probs(
    n_groups: int,
    n_steps: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Build a per-step schedule of latent solve probabilities that collapses.

    Phase 1 (healthy): probabilities sit near 0.5 with mild spread -> groups are
    mixed -> high within-group contrast -> low ZVF.
    Phase 2 (drift):   every probability is dragged toward 0 -> groups go uniform
    all-wrong -> ZVF climbs to 1 -> cold-start collapse + auto-stop.
    """
    base = rng.uniform(0.4, 0.6, size=n_groups)
    drift_start = n_steps // 3
    # Reserve the final stretch for a *fully* collapsed tail so the global batch
    # holds ZVF == 1 for >= stop_k consecutive steps and auto-stop fires.
    collapse_tail = max(4, n_steps // 4)
    collapse_start = n_steps - collapse_tail
    schedule: List[np.ndarray] = []
    for t in range(n_steps):
        if t < drift_start:
            # Healthy: jitter around the per-group base solve rate.
            p = np.clip(base + rng.normal(0.0, 0.05, n_groups), 0.05, 0.95)
        elif t < collapse_start:
            # Drift: exponentially decay every solve prob toward ~0.
            frac = (t - drift_start + 1) / max(1, collapse_start - drift_start)
            p = np.clip(base * np.exp(-4.0 * frac), 0.0, 1.0)
        else:
            # Terminal collapse: nothing is solvable -> every group all-wrong ->
            # global ZVF == 1 every step.
            p = np.zeros(n_groups)
        schedule.append(p)
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--groups", type=int, default=6, help="prompt-groups / batch")
    parser.add_argument("--K", type=int, default=8, help="rollouts per group")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # In-memory panel so we can show the logged history at the end with no W&B /
    # TensorBoard dependency.
    panel = ZVFPanel(backend="memory")
    cb = ZVFCallback(
        window=5,
        zvf_max=0.85,
        eps_lo=0.05,
        eps_hi=0.05,
        on_collapse="warm_start",
        adaptive_G=True,
        G0=args.K,
        Gmin=2,
        Gmax=32,
        drop_k=3,
        stop_k=3,
        panel=panel,
    )

    schedule = drifting_solve_probs(args.groups, args.steps, rng)

    print("=" * 78)
    print("ZVF triage quickstart -- synthetic GRPO rollouts (numpy only)")
    print(f"  groups/batch={args.groups}  rollouts/group(K)={args.K}  "
          f"steps={args.steps}  seed={args.seed}")
    print("=" * 78)
    header = (
        f"{'step':>4} {'zvf':>6} {'roll':>6} {'r_bar':>6} "
        f"{'G':>4} {'regime':<22} {'signal':<14} {'note'}"
    )
    print(header)
    print("-" * 78)

    prev_regime = None
    prev_G = None
    regime_transitions: List[str] = []
    g_changes: List[str] = []
    stop_step = None

    for t, solve_probs in enumerate(schedule):
        rewards, group_ids = simulate_step(rng, solve_probs, args.K)
        d = cb.on_step(rewards, group_ids)

        flags = []
        if prev_regime is not None and d.regime is not prev_regime:
            flags.append("REGIME-CHANGE")
            regime_transitions.append(
                f"  step {d.step}: {prev_regime.value} -> {d.regime.value}"
            )
        if prev_G is not None and d.group_size != prev_G:
            flags.append(f"G {prev_G}->{d.group_size}")
            g_changes.append(
                f"  step {d.step}: G {prev_G} -> {d.group_size} (zvf={d.zvf:.2f})"
            )
        if d.dropped_prompts:
            flags.append(f"DROP {d.dropped_prompts}")
        if d.auto_stop:
            flags.append("AUTO-STOP")

        print(
            f"{d.step:>4} {d.zvf:>6.3f} {d.rolling_zvf:>6.3f} {d.mean_reward:>6.3f} "
            f"{d.group_size:>4} {d.regime.value:<22} {d.signal:<14} "
            f"{' '.join(flags)}"
        )

        prev_regime = d.regime
        prev_G = d.group_size

        if cb.should_stop:
            stop_step = d.step
            print("-" * 78)
            print(f"  cb.should_stop is True at step {d.step}: aborting the run.")
            break

    print("=" * 78)
    print("Summary")
    print("-" * 78)

    print("Regime transitions:")
    if regime_transitions:
        for line in regime_transitions:
            print(line)
    else:
        print("  (none)")

    print("Adaptive-G changes:")
    if g_changes:
        for line in g_changes:
            print(line)
    else:
        print("  (none -- ZVF never moved G off its base)")

    print(f"Dropped prompts (cumulative): {cb.dropped_prompts}")

    if stop_step is not None:
        print(f"Auto-stop fired at step {stop_step} "
              f"(global ZVF == 1 for stop_k={cb.controller.stop_k} consecutive steps).")
    else:
        print("Auto-stop did NOT fire within the simulated horizon.")

    # Show the panel history is real, captured, and consistent.
    final = panel.history[-1]
    print(
        f"Panel captured {len(panel.history)} steps; "
        f"last logged: zvf={final['zvf']:.3f}, gu={final['gu']:.3f}, "
        f"G={final['group_size']}, regime_code={final['regime_code']}, "
        f"auto_stop={final['auto_stop']}."
    )

    # Sanity assertions so the example doubles as a smoke test: the run must
    # start healthy (low ZVF, exploitable contrast) and end in a fired auto-stop.
    assert panel.history[0]["zvf"] < 0.85, "run should start with usable contrast"
    assert cb.decisions[0].regime is Regime.EXPLOITABLE_CONTRAST, \
        "first step should be healthy"
    assert stop_step is not None, "drifting run should trigger auto-stop"
    print("\nOK: started healthy (low ZVF) and ended in auto-stop. Demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
