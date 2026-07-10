#!/usr/bin/env python3
"""P7 --- ZVF closed-loop controller (promoted from designed-only to runnable).

Background
----------
In GRPO-style training the advantage of rollout i for a prompt is
``a_i = r_i - mean_j(r_j)`` over the G rollouts of that prompt's group. If all
G rollouts collapse to the *same* reward (all-pass or all-fail) the group has
zero reward variance, every advantage is 0, and the group contributes **no
gradient**. The Zero-Variance Fraction (ZVF) is the fraction of prompt-groups
in a step that are dead this way:

    zero-variance for prompt i  <=>  all G rollouts identical
    E[1{group i dead}]  =  p_i**G + (1 - p_i)**G        (Bernoulli rollouts)
    ZVF(G)              =  mean_i ( p_i**G + (1-p_i)**G )

where ``p_i`` is prompt i's true pass-rate. A high ZVF means most of the batch
is wasted compute. The *gradient-bearing fraction* is ``1 - ZVF``.

For a difficulty-heterogeneous population (many very-easy p~1 and very-hard p~0
prompts) ZVF is high and largely insensitive to G: (1-p)**G stays ~1 for hard
prompts. Simply raising G cannot rescue a U-shaped population --- you have to
*re-weight which prompts you train on*. That is the actuator here.

Controller
----------
Closed loop, one update per training step:

  * MEASUREMENT: the ZVF actually observed over the accepted groups this step.
  * SETPOINT   : target ZVF (default 0.30  ->  ~70% of groups carry gradient).
  * ACTUATOR   : a *difficulty-acceptance half-width* ``w``. We keep only
                 prompts whose running pass-rate estimate lies in the mid band
                 [0.5 - w, 0.5 + w]. Narrow w  -> only genuine mid-difficulty
                 prompts survive -> low ZVF. Wide w -> easy/hard prompts leak
                 back in -> high ZVF. So w is monotone in ZVF and a clean knob.
  * LAW        : a clamped PID on error = measured_zvf - target. Because ZVF is
                 *increasing* in w, we move w *against* the error:
                     w <- clamp( w_center - (Kp e + Ki integral + Kd dedt) ).

An alternative actuator --- the oversampling factor (draw more prompts, keep the
mid-difficulty survivors) --- is provided as ``actuator="oversample"`` and shares
the identical PID core; the threshold actuator is the default because it is the
one that can drive a U-shaped population down to an arbitrary ZVF target.

Simulation
----------
A synthetic prompt population with per-prompt true pass-rates (U-shaped +
mid mass) is trained under (a) an UNCONTROLLED baseline that uses every prompt
and (b) the closed-loop controller. We report before/after ZVF and the number
of steps the controller needs to pull ZVF into the tolerance band around target,
and dump plot-data to experiments/results/p7_controller/sim.json.

stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Synthetic prompt population
# ----------------------------------------------------------------------------
def make_population(n_prompts: int, rng: np.random.Generator) -> np.ndarray:
    """True per-prompt pass-rates p_i.

    A deliberately hard case for ZVF: a U-shaped mixture (lots of near-0 'too
    hard' and near-1 'too easy' prompts) plus a minority of mid-difficulty
    prompts. This is what makes the *uncontrolled* ZVF high and roughly G-proof,
    so the acceptance-threshold actuator has something real to do.
    """
    comp = rng.choice(3, size=n_prompts, p=[0.40, 0.40, 0.20])
    p = np.empty(n_prompts, dtype=float)
    hard = comp == 0
    easy = comp == 1
    mid = comp == 2
    p[hard] = rng.beta(0.6, 6.0, size=hard.sum())   # clustered near 0
    p[easy] = rng.beta(6.0, 0.6, size=easy.sum())   # clustered near 1
    p[mid] = rng.beta(3.0, 3.0, size=mid.sum())     # clustered near 0.5
    return np.clip(p, 1e-3, 1.0 - 1e-3)


def analytic_zvf(p: np.ndarray, g: int) -> float:
    """Expected ZVF for pass-rates p at group size g (Bernoulli rollouts)."""
    return float(np.mean(p ** g + (1.0 - p) ** g))


def rollout_group(p_i: float, g: int, rng: np.random.Generator) -> np.ndarray:
    """G Bernoulli rewards for one prompt."""
    return (rng.random(g) < p_i).astype(np.int8)


def group_is_dead(rewards: np.ndarray) -> bool:
    """Zero reward variance <=> all rollouts identical."""
    return bool(rewards.min() == rewards.max())


# ----------------------------------------------------------------------------
# PID core
# ----------------------------------------------------------------------------
@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    setpoint: float
    i_clamp: float = 5.0          # anti-windup on the integral term
    _integral: float = 0.0
    _prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def __call__(self, measurement: float) -> float:
        error = measurement - self.setpoint
        self._integral = float(np.clip(self._integral + error, -self.i_clamp, self.i_clamp))
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error)
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * deriv


# ----------------------------------------------------------------------------
# ZVF controller
# ----------------------------------------------------------------------------
class ZVFController:
    """Closed-loop controller that keeps measured ZVF near a target band.

    Actuator ``threshold`` (default): a mid-difficulty acceptance half-width w.
    Accept prompt i iff |p_hat_i - 0.5| <= w, where p_hat_i is an online EMA
    estimate of the prompt's pass-rate. The PID output shifts w against the ZVF
    error (ZVF is increasing in w).

    Actuator ``oversample``: draw ``ceil(base_batch * factor)`` candidate prompts
    and keep the ``base_batch`` closest to p=0.5 by p_hat; the PID drives the
    factor. Same core, different knob.
    """

    def __init__(
        self,
        target_zvf: float = 0.30,
        actuator: str = "threshold",
        kp: float = 0.9,
        ki: float = 0.35,
        kd: float = 0.10,
        w_center: float = 0.5,
        w_bounds=(0.03, 0.5),
        factor_bounds=(1.0, 6.0),
        ema_beta: float = 0.6,
    ):
        assert actuator in ("threshold", "oversample")
        self.target = target_zvf
        self.actuator = actuator
        self.pid = PID(kp=kp, ki=ki, kd=kd, setpoint=target_zvf)
        self.w_center = w_center
        self.w_bounds = w_bounds
        self.factor_bounds = factor_bounds
        self.ema_beta = ema_beta

        # actuator state
        self.w = w_center                       # start wide (== accept all)
        self.factor = float(np.mean(factor_bounds))  # start mid-range
        # online pass-rate estimates
        self.p_hat: Optional[np.ndarray] = None
        self.seen: Optional[np.ndarray] = None

    # -- estimate bookkeeping ------------------------------------------------
    def init_estimates(self, n_prompts: int) -> None:
        self.p_hat = np.full(n_prompts, 0.5, dtype=float)
        self.seen = np.zeros(n_prompts, dtype=bool)

    def observe(self, idx: int, rewards: np.ndarray) -> None:
        """Fold this step's rollouts for prompt idx into its EMA estimate."""
        obs = float(rewards.mean())
        if not self.seen[idx]:
            self.p_hat[idx] = obs
            self.seen[idx] = True
        else:
            b = self.ema_beta
            self.p_hat[idx] = b * self.p_hat[idx] + (1.0 - b) * obs

    # -- selection -----------------------------------------------------------
    def select(self, candidate_idx: np.ndarray, base_batch: int,
               rng: np.random.Generator) -> np.ndarray:
        """Choose which candidate prompts to train on this step."""
        dist = np.abs(self.p_hat[candidate_idx] - 0.5)
        if self.actuator == "threshold":
            keep = candidate_idx[dist <= self.w]
            if keep.size == 0:                      # never starve the step
                order = np.argsort(dist)
                keep = candidate_idx[order[:max(1, base_batch // 4)]]
            if keep.size > base_batch:
                keep = rng.choice(keep, size=base_batch, replace=False)
            return keep
        # oversample actuator: keep the base_batch prompts nearest p=0.5
        order = np.argsort(dist)
        return candidate_idx[order[:base_batch]]

    def oversample_candidates(self, all_idx: np.ndarray, base_batch: int,
                              rng: np.random.Generator) -> np.ndarray:
        n = min(all_idx.size, int(np.ceil(base_batch * self.factor)))
        return rng.choice(all_idx, size=n, replace=False)

    # -- control update ------------------------------------------------------
    def update(self, measured_zvf: float) -> None:
        u = self.pid(measured_zvf)     # >0 when ZVF too high
        if self.actuator == "threshold":
            # ZVF increasing in w -> move w against the error
            self.w = float(np.clip(self.w_center - u, *self.w_bounds))
        else:
            # More oversampling == more mid-difficulty filtering -> ZVF is
            # *decreasing* in factor, so move factor *with* the error: ZVF too
            # high (u>0) -> oversample harder.
            self.factor = float(np.clip(self.factor + 4.0 * u, *self.factor_bounds))


# ----------------------------------------------------------------------------
# One training-loop step (shared by baseline and controlled runs)
# ----------------------------------------------------------------------------
def measure_step_zvf(p: np.ndarray, idx: np.ndarray, g: int,
                     rng: np.random.Generator,
                     ctrl: Optional[ZVFController] = None):
    """Run G rollouts for each prompt in idx; return (zvf, n_groups, gradient_groups).

    If ctrl is given, fold observed rollouts into its pass-rate estimates.
    """
    if idx.size == 0:
        return 1.0, 0, 0
    dead = 0
    for i in idx:
        rewards = rollout_group(p[i], g, rng)
        if ctrl is not None:
            ctrl.observe(int(i), rewards)
        if group_is_dead(rewards):
            dead += 1
    zvf = dead / idx.size
    return zvf, idx.size, idx.size - dead


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
@dataclass
class RunTrace:
    zvf: List[float] = field(default_factory=list)
    grad_frac: List[float] = field(default_factory=list)
    n_groups: List[int] = field(default_factory=list)
    grad_groups: List[int] = field(default_factory=list)
    knob: List[float] = field(default_factory=list)   # w (threshold) / factor


def _rolling_mean(xs: List[float], window: int) -> List[float]:
    """Causal rolling mean (window shrinks at the head)."""
    out = []
    for k in range(len(xs)):
        lo = max(0, k - window + 1)
        out.append(float(np.mean(xs[lo:k + 1])))
    return out


def steps_to_converge(zvf_series: List[float], target: float, tol: float,
                      hold: int = 3, smooth: int = 5) -> Optional[int]:
    """First step (1-based) after which the causal rolling-mean ZVF stays within
    `tol` of target for `hold` consecutive steps.

    Smoothing is used because a single step's ZVF is a Bernoulli estimate over
    the batch (std ~ sqrt(z(1-z)/n_groups) ~ 0.06 at n=64), which is comparable
    to tol; the controller stabilises the *level*, not the per-step noise. The
    raw per-step series is still stored in the JSON for inspection."""
    sm = _rolling_mean(zvf_series, smooth)
    n = len(sm)
    for k in range(n - hold + 1):
        if all(abs(sm[k + j] - target) <= tol for j in range(hold)):
            return k + 1
    return None


def run_baseline(p, n_steps, base_batch, g, rng) -> RunTrace:
    """Uncontrolled: every step trains on a random batch of ALL prompts."""
    tr = RunTrace()
    all_idx = np.arange(p.size)
    for _ in range(n_steps):
        idx = rng.choice(all_idx, size=min(base_batch, p.size), replace=False)
        zvf, ng, gg = measure_step_zvf(p, idx, g, rng)
        tr.zvf.append(zvf)
        tr.grad_frac.append(1.0 - zvf)
        tr.n_groups.append(ng)
        tr.grad_groups.append(gg)
        tr.knob.append(1.0)   # no knob
    return tr


def run_controlled(p, n_steps, base_batch, g, rng, ctrl: ZVFController,
                   warmup_g: int = 4) -> RunTrace:
    """Closed loop. A short warmup seeds pass-rate estimates for every prompt,
    then each step: select -> measure ZVF -> PID update."""
    tr = RunTrace()
    all_idx = np.arange(p.size)
    ctrl.init_estimates(p.size)

    # warmup: cheap noisy estimate of every prompt (does not count as a step)
    for i in all_idx:
        ctrl.observe(int(i), rollout_group(p[i], warmup_g, rng))

    for _ in range(n_steps):
        if ctrl.actuator == "oversample":
            cand = ctrl.oversample_candidates(all_idx, base_batch, rng)
        else:
            cand = rng.choice(all_idx, size=min(4 * base_batch, p.size), replace=False)
        idx = ctrl.select(cand, base_batch, rng)
        zvf, ng, gg = measure_step_zvf(p, idx, g, rng, ctrl=ctrl)
        tr.zvf.append(zvf)
        tr.grad_frac.append(1.0 - zvf)
        tr.n_groups.append(ng)
        tr.grad_groups.append(gg)
        tr.knob.append(ctrl.w if ctrl.actuator == "threshold" else ctrl.factor)
        ctrl.update(zvf)
    return tr


def main() -> None:
    ap = argparse.ArgumentParser(description="P7 ZVF closed-loop controller + simulation")
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--actuator", choices=["threshold", "oversample"], default="threshold")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="experiments/results/p7_controller/sim.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    p = make_population(args.n_prompts, rng)

    a_zvf = analytic_zvf(p, args.g)

    # independent RNG streams so baseline vs controlled differ only by policy
    rng_base = np.random.default_rng(args.seed + 1)
    rng_ctrl = np.random.default_rng(args.seed + 2)

    base = run_baseline(p, args.n_steps, args.batch, args.g, rng_base)

    ctrl = ZVFController(target_zvf=args.target, actuator=args.actuator)
    controlled = run_controlled(p, args.n_steps, args.batch, args.g, rng_ctrl, ctrl)

    def _mean(xs, s):  # tail mean over last `s` steps
        return float(np.mean(xs[-s:]))

    tail = max(1, args.n_steps // 5)
    conv = steps_to_converge(controlled.zvf, args.target, args.tol)

    summary = {
        "config": {
            "n_prompts": args.n_prompts, "n_steps": args.n_steps,
            "batch": args.batch, "g": args.g, "target_zvf": args.target,
            "tol": args.tol, "actuator": args.actuator, "seed": args.seed,
        },
        "population": {
            "analytic_zvf_full_G": round(a_zvf, 4),
            "mean_pass_rate": round(float(p.mean()), 4),
            "frac_hard_lt0.1": round(float((p < 0.1).mean()), 4),
            "frac_easy_gt0.9": round(float((p > 0.9).mean()), 4),
            "frac_mid_0.3_0.7": round(float(((p >= 0.3) & (p <= 0.7)).mean()), 4),
        },
        "baseline": {
            "zvf_first": round(base.zvf[0], 4),
            "zvf_tail_mean": round(_mean(base.zvf, tail), 4),
            "grad_frac_tail_mean": round(_mean(base.grad_frac, tail), 4),
            "grad_groups_total": int(np.sum(base.grad_groups)),
        },
        "controlled": {
            "zvf_first": round(controlled.zvf[0], 4),
            "zvf_tail_mean": round(_mean(controlled.zvf, tail), 4),
            "grad_frac_tail_mean": round(_mean(controlled.grad_frac, tail), 4),
            "grad_groups_total": int(np.sum(controlled.grad_groups)),
            "steps_to_converge": conv,
            "final_knob": round(controlled.knob[-1], 4),
        },
    }
    summary["improvement"] = {
        "zvf_drop_tail": round(summary["baseline"]["zvf_tail_mean"]
                               - summary["controlled"]["zvf_tail_mean"], 4),
        "grad_groups_ratio": round(
            summary["controlled"]["grad_groups_total"]
            / max(1, summary["baseline"]["grad_groups_total"]), 3),
    }

    plot_data = {
        "summary": summary,
        "series": {
            "step": list(range(1, args.n_steps + 1)),
            "target": args.target,
            "baseline_zvf": [round(x, 4) for x in base.zvf],
            "controlled_zvf": [round(x, 4) for x in controlled.zvf],
            "controlled_knob": [round(x, 4) for x in controlled.knob],
            "baseline_grad_frac": [round(x, 4) for x in base.grad_frac],
            "controlled_grad_frac": [round(x, 4) for x in controlled.grad_frac],
        },
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(plot_data, f, indent=2)

    # ---- console report ----
    b, c = summary["baseline"], summary["controlled"]
    print("=" * 68)
    print("P7 ZVF CONTROLLER --- simulation report")
    print("=" * 68)
    print(f"actuator={args.actuator}  target ZVF={args.target}  tol=+/-{args.tol}"
          f"  G={args.g}  batch={args.batch}  prompts={args.n_prompts}")
    print(f"population analytic ZVF (train on everything, G={args.g}): {a_zvf:.3f}")
    print("-" * 68)
    print(f"{'':22s}{'ZVF step1':>11s}{'ZVF tail':>11s}{'grad-frac tail':>16s}")
    print(f"{'UNCONTROLLED baseline':22s}{b['zvf_first']:>11.3f}"
          f"{b['zvf_tail_mean']:>11.3f}{b['grad_frac_tail_mean']:>16.3f}")
    print(f"{'CLOSED-LOOP controller':22s}{c['zvf_first']:>11.3f}"
          f"{c['zvf_tail_mean']:>11.3f}{c['grad_frac_tail_mean']:>16.3f}")
    print("-" * 68)
    print(f"ZVF driven {b['zvf_tail_mean']:.3f} -> {c['zvf_tail_mean']:.3f}"
          f"  (target {args.target})")
    print(f"steps-to-converge (|ZVF-target|<={args.tol}, held 3): {c['steps_to_converge']}")
    print(f"gradient-bearing groups: baseline={b['grad_groups_total']}  "
          f"controlled={c['grad_groups_total']}  "
          f"ratio={summary['improvement']['grad_groups_ratio']}x")
    print(f"final actuator knob "
          f"({'w' if args.actuator=='threshold' else 'factor'})={c['final_knob']}")
    print(f"\nplot-data -> {out_path}")


if __name__ == "__main__":
    main()
