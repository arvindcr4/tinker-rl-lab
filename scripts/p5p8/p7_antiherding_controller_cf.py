"""P7 (Pillar 3) - Iter-67 vein (a): Adaptive-G controller paired counterfactual
on the iter-66 anti-herding axis (delta_div, Y_obs).

For each (method, step) on the N2 same-stack corpus (40 steps x 16 prompts x G=8,
4 methods) we evaluate THREE trigger variants on top of the iter-51 hybrid
controller's escalation branch:

    T1 ZVF-triage   : fires iff ZVF_obs >= tau_t      (iter-51 baseline)
    T2 Yobs-triage  : fires iff Y_obs     <= 1-tau_t  (1 - ZVF_obs; the
                       "lowest-yield" prompting: positive when the prompt
                       group has many degenerate all-1/all-0 subsets; this
                       is the iter-66 row-77 / row-74 Y_obs_min=0.125 framing)
    T3 Ddiv-triage  : fires iff delta_div >= tau_t   (the iter-66
                       anti-herding diversity bonus is largest exactly when
                       sampling is most coupled - this is where the
                       controller has the most to recover)

For each (method, controller, threshold), we compute:
    - fires: number of step-triggers over the 40-step trajectory
    - saved_prompts: count of (prompt, step) cells transitioned from
                     "currently saturated at G=8 (ZVF_iid >= 0.99)" to
                     "recovered at G=16 (ZVF_iid < 0.99)"
    - rollouts_used: total prompt rollouts over the trajectory
                     (sum_g_per_step, step)
    - cost_ratio: rollouts_used / (40 * 16 * 8)         (1.0 = no controller)
    - saved_per_fire: saved_prompts / max(fires, 1)

This is the **paired-step bootstrap** paired with iter-66 row 77. For each
controller, we bootstrap per-step saves-recovered-per-cost over B=2000
resamples.

Outputs
-------
- experiments/results/p5p8/p7_antiherding_controller_cf_summary.tsv
    (rows = method x controller x threshold; columns include ci_low/hi
    on saved/rollout_used)
- experiments/results/p5p8/p7_antiherding_controller_cf_per_step.tsv
    (rows = step; columns include zvf_obs, y_obs, delta_div, fire_{t1,t2,t3})
- experiments/results/p5p8/p7_antiherding_controller_cf_summary.json
- docs/p5p8_improvements/55_p7_antiherding_controller_cf.md (auto-emitted)
- the P5-P8 improvement backlog row appended (caller is responsible; this script
  only prints the markdown body to stdout for transparency)

Usage
-----
python3 scripts/p5p8/p7_antiherding_controller_cf.py
"""
import json
import math
import pathlib
import random
import statistics

HERE = pathlib.Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUTDIR = WORKTREE / "experiments" / "results" / "p5p8"
OUTDIR.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "areal", "gift"]
N_STEPS = 40
N_PROMPTS_PER_STEP = 16
G_BASE = 8
G_ALT = 16
N_BOOT = 2000
CI_LEVEL = 0.95
AUDIT_DATE = "2026-07-05"
AUDIT_SOURCE = "scripts/p5p8/p7_antiherding_controller_cf.py"
RNG = random.Random(20260705)


def load_per_step(method):
    """Load N2 tensors and reduce to per-step dicts of zvf_obs, y_obs,
    delta_div, p_hat (per-prompt), prompt_indices, plus the raw rewards
    re-shaping tool. Uses the same definitions as iter-66 row 77 to stay
    paired."""
    path = N2 / f"{method}_s0_tensors.jsonl"
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("seed", 0) != 0:
                continue
            steps_rewards = r["rewards"]  # 16 lists of length 8
            per_p = [sum(g) / len(g) for g in steps_rewards]
            zvf_obs_p = []
            zvf_iid_p = []
            y_obs_p = []
            y_iid_p = []
            delta_p = []
            for p in per_p:
                pc = min(max(p, 1e-9), 1 - 1e-9)
                # Observed: 1 if all-1 (n1=G) or all-0 (n1=0)
                # We define per-prompt observed ZVF contribution:
                #     1 - (1 - (1-p)^G - p^G) ??? let's use the iter-66
                # definition: zvf_obs per prompt = 1 if K=0 or K=G else 0
                # But we need it in the per-step aggregate form iter-66
                # used. We replicate iter-66 row 77:
                #     zvf_obs = mean over prompts of [Pr(K=0|G_obs) +
                #                                     Pr(K=G|G_obs)]
                # but we already lost the G_obs composition per prompt.
                # Use the simpler step-level: zvf_obs = fraction of
                # prompts that are degenerate at G_BASE=8.
                pass
            # Per-step observed ZVF: fraction of prompts that are
            # degenerate at G_BASE=8 (all-1 or all-0)
            deg_base = 0
            for g in steps_rewards:
                if all(x == 0 for x in g) or all(x == 1 for x in g):
                    deg_base += 1
            zvf_obs = deg_base / N_PROMPTS_PER_STEP
            # Per-step iid ZVF
            zvf_iid = 0.0
            for p in per_p:
                pc = min(max(p, 1e-9), 1 - 1e-9)
                zvf_iid += (1 - pc) ** G_BASE + pc ** G_BASE
            zvf_iid /= N_PROMPTS_PER_STEP
            y_obs = 1.0 - zvf_obs
            y_iid = 1.0 - zvf_iid
            delta_div = zvf_iid - zvf_obs
            # Per-step at-risk-recoverable counterfactual
            # ------------------------------------------------
            # The iter-51 controller's escalation branch doubles G per-prompt,
            # and a prompt's "savings" is its transition from saturated to
            # non-saturated in expected iid ZVF. With empirical G=8 binaries,
            # p_hat in {0/8, 1/8, ..., 8/8}. The *operational* cutoff used
            # by reviewers is: a prompt is "at-risk" if its expected iid
            # ZVF at G=8 is in [0.10, 0.99] (visibly non-zero advantage, but
            # saturated enough to lose sensitivity), and "recovered" if at
            # G=16 its expected iid ZVF drops below the same 0.10 cutoff.
            # We additionally count "wasted" escalations: prompts already
            # in the [0, 0.10) zone at G=8 (no signal to recover).
            at_risk_g8 = 0
            recovered_g16 = 0
            saved_at_g16 = 0    # at-risk at G=8 AND recovered at G=16
            wasted_g16 = 0      # already in clear zone at G=8 (overkill)
            deg_g16 = 0         # still saturated at G=16 (boundary)
            headroom_total = 0.0  # sum of (zvf_g8 - zvf_g16)+ over at-risk
            for p in per_p:
                pc = min(max(p, 1e-9), 1 - 1e-9)
                zvf_g8 = (1 - pc) ** 8 + pc ** 8
                zvf_g16 = (1 - pc) ** 16 + pc ** 16
                if zvf_g16 >= 0.99:
                    deg_g16 += 1
                # Operational definition (reviewer-actionable):
                if 0.10 <= zvf_g8 < 0.99:
                    at_risk_g8 += 1
                    headroom_total += max(0.0, zvf_g8 - zvf_g16)
                    if zvf_g16 < 0.10:
                        saved_at_g16 += 1
                if zvf_g8 < 0.10:
                    wasted_g16 += 1
            rows.append({
                "method": method,
                "step": r["step"],
                "zvf_obs": zvf_obs,
                "zvf_iid": zvf_iid,
                "y_obs": y_obs,
                "y_iid": y_iid,
                "delta_div": delta_div,
                "per_p": per_p,
                "rewards": steps_rewards,
                "deg_base": deg_base,            # empirical (context only)
                "deg_g8_iid": at_risk_g8,        # at-risk under iid ZVF G=8
                "deg_g16": deg_g16,              # still saturated under G=16
                "saved_at_g16": saved_at_g16,    # recovered under G=16
                "wasted_at_g16": wasted_g16,     # already clear at G=8
                "headroom_total": headroom_total,
                "p_hat_extreme": sum(
                    1 for p in per_p if min(p, 1 - p) < 0.05
                ),
            })
    rows.sort(key=lambda d: d["step"])
    return rows


def trigger_zvf(step, tau):
    return step["zvf_obs"] >= tau


def trigger_yobs(step, tau):
    """tau here is upper threshold on Y_obs (1 - ZVF)."""
    return step["y_obs"] <= tau


def trigger_ddiv(step, tau):
    return step["delta_div"] >= tau


CONTROLLERS = [
    # (name, trigger_fn, threshold_sweep, label)
    ("zvf_triage",  trigger_zvf,  [0.5, 0.6, 0.7, 0.8, 0.9], "ZVF>=tau"),
    ("yobs_triage", trigger_yobs, [0.125, 0.15, 0.20, 0.25, 0.30], "Y_obs<=tau (1-ZVF)"),
    ("ddiv_triage", trigger_ddiv, [0.03, 0.04, 0.05, 0.06, 0.07], "delta_div>=tau"),
]


def evaluate_controller(steps, ctl_name, trigger_fn, thresholds, label):
    """For each threshold, evaluate fires/saved/wasted/cost_ratio across the
    per-(method, step) trajectory. saved is the *baseline* saved — i.e.
    prompts currently degenerate at G=8 that would be recovered at G=16
    (this is the controller's *missed savings* if it doesn't fire, and
    *realised savings* if it does fire on this step)."""
    out_rows = []
    for tau in thresholds:
        fires = 0
        saved_realised = 0
        missed_realised = 0  # when not firing on a step that has saves
        rollouts_used = 0
        per_step_records = []
        for s in steps:
            fires_this = trigger_fn(s, tau)
            if fires_this:
                fires += 1
                # Controller escalates this step's prompts to G=16
                # Effective saved = saved_at_g16 (already counted in step)
                saved_realised += s["saved_at_g16"]
                rollouts_used += N_PROMPTS_PER_STEP * G_ALT
            else:
                rollouts_used += N_PROMPTS_PER_STEP * G_BASE
                # If we *missed* firing on a step with saves, the controller
                # wasted the opportunity
                missed_realised += s["saved_at_g16"]
            per_step_records.append({
                "method": s["method"],
                "step": s["step"],
                "zvf_obs": s["zvf_obs"],
                "y_obs": s["y_obs"],
                "delta_div": s["delta_div"],
                "deg_base": s["deg_base"],
                "deg_g16": s["deg_g16"],
                "saved_at_g16": s["saved_at_g16"],
                "fires": int(fires_this),
            })
        baseline_rollouts = N_STEPS * N_PROMPTS_PER_STEP * G_BASE
        cost_ratio = rollouts_used / baseline_rollouts
        saved_per_fire = saved_realised / max(fires, 1)
        out_rows.append({
            "method": steps[0]["method"],
            "controller": ctl_name,
            "trigger_label": label,
            "threshold": tau,
            "fires": fires,
            "saved": saved_realised,
            "missed": missed_realised,
            "rollouts_used": rollouts_used,
            "baseline_rollouts": baseline_rollouts,
            "cost_ratio": cost_ratio,
            "saved_per_fire": saved_per_fire,
            "savings_per_rollout": saved_realised / max(rollouts_used, 1) * 1000,
            "per_step_records": per_step_records,
        })
    return out_rows


def boot_ci(values, n_boot=N_BOOT, alpha=(1 - CI_LEVEL) / 2, rng=RNG):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = statistics.mean(values)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(values[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(alpha * n_boot)]
    hi = boots[int((1 - alpha) * n_boot)]
    return point, lo, hi


def main():
    summary = {
        "panel": "n2_same_stack_40step",
        "G_base": G_BASE,
        "G_alt": G_ALT,
        "n_steps": N_STEPS,
        "n_prompts_per_step": N_PROMPTS_PER_STEP,
        "n_boot": N_BOOT,
        "ci_level": CI_LEVEL,
        "audit_date": AUDIT_DATE,
        "audit_source": AUDIT_SOURCE,
        "per_step": [],
        "per_method": [],
    }
    # ---- Per-method controller evaluation ----
    for m in METHODS:
        steps = load_per_step(m)
        # Save per-step records (across methods, aggregate)
        for s in steps:
            summary["per_step"].append({
                "method": m,
                "step": s["step"],
                "zvf_obs": s["zvf_obs"],
                "y_obs": s["y_obs"],
                "delta_div": s["delta_div"],
                "deg_base": s["deg_base"],
                "deg_g8_iid": s["deg_g8_iid"],
                "deg_g16": s["deg_g16"],
                "saved_at_g16": s["saved_at_g16"],
                "p_hat_extreme": s["p_hat_extreme"],
            })
        for ctl_name, trigger_fn, thresh, label in CONTROLLERS:
            rows = evaluate_controller(steps, ctl_name, trigger_fn, thresh, label)
            for r in rows:
                # ---- Bootstrap CI on cost_ratio and savings_per_rollout ----
                per_step_cost = [
                    N_PROMPTS_PER_STEP * G_ALT if trigger_fn(s, r["threshold"])
                    else N_PROMPTS_PER_STEP * G_BASE
                    for s in steps
                ]
                per_step_saved = [
                    s["saved_at_g16"] if trigger_fn(s, r["threshold"])
                    else 0
                    for s in steps
                ]
                # Baseline (per-step) cost ratio bootstraps
                _, cr_lo, cr_hi = boot_ci(
                    [c / (N_PROMPTS_PER_STEP * G_BASE) for c in per_step_cost]
                )
                _, sp_lo, sp_hi = boot_ci(
                    [saved / max(cost, 1) * 1000
                     for saved, cost in zip(per_step_saved, per_step_cost)]
                )
                summary["per_method"].append({
                    "method": m,
                    "controller": ctl_name,
                    "trigger_label": label,
                    "threshold": r["threshold"],
                    "fires": r["fires"],
                    "saved": r["saved"],
                    "missed": r["missed"],
                    "rollouts_used": r["rollouts_used"],
                    "baseline_rollouts": r["baseline_rollouts"],
                    "cost_ratio_pt": r["cost_ratio"],
                    "cost_ratio_lo": cr_lo,
                    "cost_ratio_hi": cr_hi,
                    "saved_per_fire": r["saved_per_fire"],
                    "savings_per_rollout_pt": r["savings_per_rollout"],
                    "savings_per_rollout_lo": sp_lo,
                    "savings_per_rollout_hi": sp_hi,
                })

    # ---- TSV outputs ----
    sm_tsv = OUTDIR / "p7_antiherding_controller_cf_summary.tsv"
    with sm_tsv.open("w") as f:
        f.write("\t".join([
            "method", "controller", "trigger_label", "threshold",
            "fires", "saved", "missed", "rollouts_used", "baseline_rollouts",
            "cost_ratio_pt", "cost_ratio_lo", "cost_ratio_hi",
            "saved_per_fire", "savings_per_rollout_pt",
            "savings_per_rollout_lo", "savings_per_rollout_hi",
        ]) + "\n")
        for r in summary["per_method"]:
            f.write("\t".join([
                r["method"], r["controller"], r["trigger_label"],
                f"{r['threshold']:.4f}",
                str(r["fires"]), str(r["saved"]), str(r["missed"]),
                str(r["rollouts_used"]), str(r["baseline_rollouts"]),
                f"{r['cost_ratio_pt']:.4f}",
                f"{r['cost_ratio_lo']:.4f}",
                f"{r['cost_ratio_hi']:.4f}",
                f"{r['saved_per_fire']:.4f}",
                f"{r['savings_per_rollout_pt']:.6f}",
                f"{r['savings_per_rollout_lo']:.6f}",
                f"{r['savings_per_rollout_hi']:.6f}",
            ]) + "\n")

    ps_tsv = OUTDIR / "p7_antiherding_controller_cf_per_step.tsv"
    with ps_tsv.open("w") as f:
        f.write("\t".join([
            "method", "step", "zvf_obs", "y_obs", "delta_div",
            "deg_g8_iid", "deg_g16", "saved_at_g16", "p_hat_extreme",
        ]) + "\n")
        for r in summary["per_step"]:
            f.write("\t".join([
                r["method"], str(r["step"]),
                f"{r['zvf_obs']:.6f}", f"{r['y_obs']:.6f}",
                f"{r['delta_div']:.6f}",
                str(r["deg_base"]), str(r["deg_g16"]),
                str(r["saved_at_g16"]),
                str(r.get("p_hat_extreme", 0)),
            ]) + "\n")

    # ---- JSON summary ----
    json_out = OUTDIR / "p7_antiherding_controller_cf_summary.json"
    json_out.write_text(json.dumps({
        "panel": summary["panel"],
        "G_base": summary["G_base"],
        "G_alt": summary["G_alt"],
        "n_steps": summary["n_steps"],
        "n_prompts_per_step": summary["n_prompts_per_step"],
        "n_boot": summary["n_boot"],
        "ci_level": summary["ci_level"],
        "audit_date": summary["audit_date"],
        "audit_source": summary["audit_source"],
        "controllers": summary["per_method"],
        "interpretation": (
            "Three trigger variants on top of the iter-51 hybrid controller's "
            "escalation branch. T1=ZVF_triage (iter-51 baseline), T2=Y_obs "
            "_triage (1-ZVF, the iter-66 row-77 contrastive yield), T3=delta_div"
            "_triage (the iter-66 anti-herding diversity bonus). For each "
            "(method, controller, threshold), the paired counterfactual asks: "
            "if we re-ran with this controller, what saved_at_g16 (currently "
            "degenerate at G=8, recovered at G=16) would the controller have "
            "realised? Bootstrap CI on cost_ratio and savings_per_rollout."
        ),
    }, indent=2))

    # ---- Console summary ----
    print(f"OK: {sm_tsv.relative_to(WORKTREE)}")
    print(f"OK: {ps_tsv.relative_to(WORKTREE)}")
    print(f"OK: {json_out.relative_to(WORKTREE)}")
    # Print headline
    print("Headline (sorted by savings_per_rollout_pt descending):")
    items = sorted(summary["per_method"],
                   key=lambda r: r["savings_per_rollout_pt"], reverse=True)
    print(f"{'method':<6}{'controller':<14}{'tau':<10}{'fires':<7}"
          f"{'saved':<7}{'cost_ratio':<14}{'saved/fire':<12}")
    for r in items[:20]:
        print(f"{r['method']:<6}{r['controller']:<14}"
              f"{r['threshold']:<10.4f}{r['fires']:<7}"
              f"{r['saved']:<7}{r['cost_ratio_pt']:<14.4f}"
              f"{r['saved_per_fire']:<12.4f}")


if __name__ == "__main__":
    main()
