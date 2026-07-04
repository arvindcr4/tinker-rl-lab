"""Iter 23 — P7 vein (d): bootstrap CIs on the cost-efficiency
Pareto-restoration metric + per-regime stratification.

Item 26 (P7 vein (e)) presented the cost-efficiency Pareto with a
symmetry-based caveat ("sub-0.5/1k by symmetry across methods"); the
95% CIs on `restored/1k extra rollouts` for each controller were not
formally bootstrapped. This iter closes that statistical-rigor gap.

Output:
  experiments/results/p5p8/p7_costeff_boot_summary.tsv
  experiments/results/p5p8/p7_costeff_boot_summary.json
  experiments/results/p5p8/p7_costeff_boot_regime.tsv
  experiments/results/p5p8/p7_costeff_boot_step.tsv
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
TENSOR_DIR = WORKTREE / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORKTREE / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ("grpo", "aero", "gift", "areal")
G_BASE, G_NEW = 8, 16
BOOT, RNG_SEED = 4000, 20260704


def betaln(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bb_postpred(k: int, n: int, yp: int, gp: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    log_p = (
        math.lgamma(gp + 1)
        - math.lgamma(yp + 1)
        - math.lgamma(gp - yp + 1)
        + betaln(alpha + k + yp, beta + n - k + gp - yp)
        - betaln(alpha + k, beta + n - k)
    )
    return max(0.0, min(1.0, math.exp(log_p)))


def restore_prob(k: int, n: int = G_BASE, gp: int = G_NEW) -> float:
    p0 = bb_postpred(k, n, 0, gp)
    pgp = bb_postpred(k, n, gp, gp)
    return max(0.0, min(1.0, 1.0 - p0 - pgp))


def midrange_prob(k: int, n: int = G_BASE, alpha: float = 1.0, beta: float = 1.0) -> float:
    a = k + alpha
    b = n - k + beta
    lo, hi, grid = 0.05, 0.95, 1024
    dx = (hi - lo) / grid
    log_norm = betaln(a, b)
    total = 0.0
    prev = math.exp((a - 1.0) * math.log(lo) + (b - 1.0) * math.log(1.0 - lo) - log_norm)
    for i in range(1, grid + 1):
        x = lo + i * dx
        cur = math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_norm)
        total += 0.5 * (prev + cur) * dx
        prev = cur
    return total


def load_tensors(method: str):
    fp = TENSOR_DIR / f"{method}_s0_tensors.jsonl"
    out = []
    with fp.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


# Pre-compute per-step records ONCE and store as parallel arrays for fast
# O(n) bootstrap resampling (no Python-level per-element dict rebuilds).
class StepMatrix:
    """Per-step parallel arrays: each step has 16 prompts with k values."""

    def __init__(self):
        # per-step: list of (k_arr_16, zvf_step, pcd_step, method_idx, step_idx)
        self.steps = []  # one entry per (method, step)
        self.records = []  # flat list of dicts (used for printing / breakdown)

    def add_method(self, method: str, method_idx: int):
        tensors = load_tensors(method)
        for step_idx, step_rec in enumerate(tensors):
            ks = [int(round(sum(r))) for r in step_rec["rewards"]]
            self.steps.append({
                "method": method,
                "method_idx": method_idx,
                "step": step_rec["step"],
                "ks": ks,
                "zvf": step_rec["zvf"],
                "pcd": step_rec["pcd"],
                "restores": [restore_prob(k) for k in ks],
                "midranges": [midrange_prob(k) for k in ks],
            })

    def add_all_records(self):
        """For regime-stratified reporting."""
        out = []
        for s in self.steps:
            for j, k in enumerate(s["ks"]):
                out.append({
                    "method": s["method"], "method_idx": s["method_idx"],
                    "step_idx": s["step"], "k": k,
                    "zvf_step": s["zvf"], "pcd_step": s["pcd"],
                    "restore": s["restores"][j], "midrange": s["midranges"][j],
                    "is_degenerate": (k == 0 or k == 8),
                    "is_boundary": (k in (1, 7)),
                    "is_mid": (k in (2, 3, 4, 5, 6)),
                })
        self.records = out

    # ---- fast per-step masks for a controller ----
    def zvf_triage_mask(self, tau: float, step_filter=None):
        """For each (method, step) return True/False mask + restore sum."""
        n = 0
        mask = [False] * (len(self.steps) * 16)
        restore_total = 0.0
        for s_idx, s in enumerate(self.steps):
            fire = (s["zvf"] >= tau and s["pcd"] <= 0.20)
            if step_filter and not step_filter(s):
                fire = False
            for j in range(16):
                if fire:
                    mask[s_idx * 16 + j] = True
                    restore_total += s["restores"][j]
        return mask, restore_total

    def bayes_mask(self, tau: float):
        """Per-prompt: fire iff currently degenerate AND m(k,8) > tau."""
        n = len(self.steps) * 16
        mask = [False] * n
        restore_total = 0.0
        for s_idx, s in enumerate(self.steps):
            for j, k in enumerate(s["ks"]):
                if (k == 0 or k == 8) and s["midranges"][j] > tau:
                    mask[s_idx * 16 + j] = True
                    restore_total += s["restores"][j]
        return mask, restore_total

    def dualformer_mask(self):
        n = len(self.steps) * 16
        mask = [False] * n
        restore_total = 0.0
        for s_idx, s in enumerate(self.steps):
            for j, k in enumerate(s["ks"]):
                if k <= 1 or k >= 7:
                    mask[s_idx * 16 + j] = True
                    restore_total += s["restores"][j]
        return mask, restore_total


def bootstrap_ci_mean(values, boot=BOOT, seed=RNG_SEED):
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    pts = []
    for _ in range(boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        pts.append(statistics.mean(sample))
    pts.sort()
    return (statistics.mean(values), pts[int(0.025 * boot)], pts[int(0.975 * boot)])


def rest_per_k_extra(n_fires: int, total_restore: float) -> float:
    if n_fires == 0:
        return 0.0
    return 1000.0 * total_restore / (8.0 * n_fires)


def main():
    print("[p7_costeff_boot] loading tensors ...")
    sm = StepMatrix()
    for i, m in enumerate(METHODS):
        sm.add_method(m, i)
    sm.add_all_records()

    n_total = len(sm.records)
    n_by_method = {m: sum(1 for r in sm.records if r["method"] == m) for m in METHODS}
    n_degenerate = sum(1 for r in sm.records if r["is_degenerate"])
    n_boundary = sum(1 for r in sm.records if r["is_boundary"])
    n_mid = sum(1 for r in sm.records if r["is_mid"])

    print(f"Total prompt-step obs: {n_total}")
    print(f"  by method: {n_by_method}")
    print(f"  regimes: degen={n_degenerate} boundary={n_boundary} mid={n_mid}")

    # ---- job 1: per-controller real-data + bootstrap ----
    n_steps = len(sm.steps)  # 160 steps
    n_prompts_total = n_steps * 16  # 2560
    rng = random.Random(RNG_SEED)

    def ctrl_real(ctrl_callable):
        mask, restore_total = ctrl_callable()
        n_fires = sum(1 for m in mask if m)
        return n_fires, restore_total, rest_per_k_extra(n_fires, restore_total)

    controllers = [
        ("zvf_triage", 0.50, lambda: sm.zvf_triage_mask(0.50)),
        ("zvf_triage", 0.70, lambda: sm.zvf_triage_mask(0.70)),
        ("zvf_triage", 0.90, lambda: sm.zvf_triage_mask(0.90)),
        ("dualformer", None, lambda: sm.dualformer_mask()),
        ("bayes", 0.60, lambda: sm.bayes_mask(0.60)),
        ("bayes", 0.65, lambda: sm.bayes_mask(0.65)),  # sanity: silenced
    ]

    print("[p7_costeff_boot] per-controller real-data ...")
    real_per_ctrl = {}
    for ctrl, tau, fn in controllers:
        n_fires, r_total, rpk = ctrl_real(fn)
        real_per_ctrl[(ctrl, tau)] = {
            "n_fires": n_fires, "fires_per_method": n_fires / 4.0,
            "restore_total": r_total,
            "extra": n_fires * 8,
            "rpk_real": rpk,
        }
        print(f"  {ctrl}@{tau}: fires={n_fires} restore={r_total:.1f} rpk={rpk:.3f}")

    # ---- job 2: bootstrap CI on rest/1k-extra (step-level resampling) ----
    # For each controller, store samples
    boot_samples = {(ctrl, tau): [] for (ctrl, tau, _) in controllers}

    print(f"[p7_costeff_boot] bootstrapping {BOOT} iter x {len(controllers)} controllers ...")
    for b in range(BOOT):
        if (b + 1) % 500 == 0:
            print(f"  bootstrap iter {b + 1}/{BOOT}")
        # Step-level resampling: build a temporary StepMatrix-like view
        # Approach: sample step indices with replacement, then evaluate masks
        # in one pass over the sampled steps. Faster to inline than to rebuild.
        sampled_step_indices = [rng.randrange(n_steps) for _ in range(n_steps)]

        for (ctrl, tau, _) in controllers:
            n_fires_b = 0
            restore_b = 0.0
            for s_idx in sampled_step_indices:
                s = sm.steps[s_idx]
                if ctrl == "zvf_triage":
                    if s["zvf"] >= tau and s["pcd"] <= 0.20:
                        n_fires_b += 16
                        restore_b += sum(s["restores"])
                elif ctrl == "bayes":
                    for j, k in enumerate(s["ks"]):
                        if (k == 0 or k == 8) and s["midranges"][j] > tau:
                            n_fires_b += 1
                            restore_b += s["restores"][j]
                elif ctrl == "dualformer":
                    for j, k in enumerate(s["ks"]):
                        if k <= 1 or k >= 7:
                            n_fires_b += 1
                            restore_b += s["restores"][j]
            boot_samples[(ctrl, tau)].append(
                1000.0 * restore_b / (8.0 * n_fires_b) if n_fires_b > 0 else 0.0
            )

    boot_lo = int(0.025 * BOOT); boot_hi = int(0.975 * BOOT)
    summary_rows = [
        "controller\ttau\tn_fires\tfires_per_method\trestored\t"
        "extra\trpk_real\trpk_boot_mean\trpk_boot_lo\trpk_boot_hi\t"
        "Δvs_zvf_0.5_mean\tΔvs_zvf_0.5_lo\tΔvs_zvf_0.5_hi\tci_excludes_zero"
    ]
    base_real = real_per_ctrl[("zvf_triage", 0.50)]["rpk_real"]
    base_samples = boot_samples[("zvf_triage", 0.50)]
    base_mean = statistics.mean(base_samples)
    print(f"[p7_costeff_boot] pareto winner (zvf_triage@0.50) boot mean = {base_mean:.3f}")

    for ctrl, tau, _ in controllers:
        real = real_per_ctrl[(ctrl, tau)]
        samples = boot_samples[(ctrl, tau)]
        samples_sorted = sorted(samples)
        b_mean = statistics.mean(samples)
        b_lo = samples_sorted[boot_lo]
        b_hi = samples_sorted[boot_hi]
        # delta samples
        delta = [base_samples[i] - samples[i] for i in range(BOOT)]
        delta_sorted = sorted(delta)
        d_mean = base_mean - b_mean
        d_lo = delta_sorted[boot_lo]
        d_hi = delta_sorted[boot_hi]
        excludes = (d_lo > 0) or (d_hi < 0)
        ctrl_label = f"{ctrl}@{tau}" if tau is not None else ctrl
        ci_str = "EXCLUDES_0" if excludes else "INCLUDES_0"
        summary_rows.append(
            f"{ctrl_label}\t{tau if tau is not None else '-'}\t"
            f"{real['n_fires']}\t{real['fires_per_method']:.1f}\t"
            f"{real['restore_total']:.2f}\t{real['extra']}\t"
            f"{real['rpk_real']:.4f}\t{b_mean:.4f}\t{b_lo:.4f}\t{b_hi:.4f}\t"
            f"{d_mean:+.4f}\t{d_lo:+.4f}\t{d_hi:+.4f}\t{ci_str}"
        )

    summary_fp = OUT_DIR / "p7_costeff_boot_summary.tsv"
    summary_fp.write_text("\n".join(summary_rows) + "\n")
    print(f"wrote {summary_fp}")

    # ---- job 3: per-regime stratification ----
    regime_rows = [
        "regime\tdefinition\tn_obs\tmean_k\tmean_restore\t"
        "zvf_t50_n_fires\tzvf_t50_fire_rate\tzvf_t50_rpk_real\t"
        "bayes_60_n_fires\tbayes_60_fire_rate\tbayes_60_rpk_real\t"
        "dualf_n_fires\tdualf_fire_rate\tdualf_rpk_real"
    ]
    sub_filter = lambda pred: [r for r in sm.records if pred(r)]
    regimes = [
        ("degenerate_k0_k8", lambda r: r["is_degenerate"]),
        ("boundary_k1_k7", lambda r: r["is_boundary"]),
        ("mid_k2_to_6", lambda r: r["is_mid"]),
        ("non_degenerate", lambda r: not r["is_degenerate"]),
        ("full_all_2k560", lambda r: True),
    ]
    for rname, fn in regimes:
        sub = sub_filter(fn)
        if not sub:
            continue
        mean_k = statistics.mean(r["k"] for r in sub)
        mean_r = statistics.mean(r["restore"] for r in sub)
        # zvf-triage@0.5 step-level: fire iff zvf>=0.5 AND pcd<=0.20
        n_fired_z = 0; restore_z = 0.0
        for s in sm.steps:
            n_in_sub_step = sum(1 for r in sub if r["method_idx"] == s["method_idx"] and r["step_idx"] == s["step"])
            if n_in_sub_step == 0:
                continue
            if s["zvf"] >= 0.50 and s["pcd"] <= 0.20:
                # use the in-substep records for actual restore values
                in_sub_recs = [r for r in sub if r["method_idx"] == s["method_idx"] and r["step_idx"] == s["step"]]
                n_fired_z += len(in_sub_recs)
                restore_z += sum(r["restore"] for r in in_sub_recs)
        zvf_rpk = 1000.0 * restore_z / (8 * n_fired_z) if n_fired_z else 0.0
        # bayesian
        n_fired_b = 0; restore_b = 0.0
        for r in sub:
            if r["is_degenerate"] and r["midrange"] > 0.60:
                n_fired_b += 1; restore_b += r["restore"]
        b_rpk = 1000.0 * restore_b / (8 * n_fired_b) if n_fired_b else 0.0
        # dualformer
        n_fired_d = 0; restore_d = 0.0
        for r in sub:
            if r["k"] <= 1 or r["k"] >= 7:
                n_fired_d += 1; restore_d += r["restore"]
        d_rpk = 1000.0 * restore_d / (8 * n_fired_d) if n_fired_d else 0.0
        zvf_rate = n_fired_z / len(sub) if sub else 0.0
        b_rate = n_fired_b / len(sub) if sub else 0.0
        d_rate = n_fired_d / len(sub) if sub else 0.0
        regime_rows.append(
            f"{rname}\t--\t{len(sub)}\t{mean_k:.3f}\t{mean_r:.4f}\t"
            f"{n_fired_z}\t{zvf_rate:.4f}\t{zvf_rpk:.4f}\t"
            f"{n_fired_b}\t{b_rate:.4f}\t{b_rpk:.4f}\t"
            f"{n_fired_d}\t{d_rate:.4f}\t{d_rpk:.4f}"
        )

    regime_fp = OUT_DIR / "p7_costeff_boot_regime.tsv"
    regime_fp.write_text("\n".join(regime_rows) + "\n")
    print(f"wrote {regime_fp}")

    # ---- job 4: per-fired-step decomposition of zvf-triage@0.5 ----
    step_rows = [
        "method\tstep\tzvf_step\tpcd_step\tn_prompts_in_step\t"
        "n_degen\tn_bd\tn_md\tmean_restore_in_step\trestore_sum_in_step\tfired"
    ]
    fired_count = 0
    fired_restore_total = 0.0
    fired_degen_restore_sum = 0.0
    fired_bd_restore_sum = 0.0
    fired_md_restore_sum = 0.0
    fired_steps_list = []
    for s in sm.steps:
        n = 16
        n_degen = sum(1 for k in s["ks"] if k == 0 or k == 8)
        n_bd = sum(1 for k in s["ks"] if k in (1, 7))
        n_md = sum(1 for k in s["ks"] if k in (2, 3, 4, 5, 6))
        mean_r = statistics.mean(s["restores"])
        r_sum = sum(s["restores"])
        fired = (s["zvf"] >= 0.50 and s["pcd"] <= 0.20)
        step_rows.append(
            f"{s['method']}\t{s['step']}\t{s['zvf']:.4f}\t{s['pcd']:.4f}\t{n}\t"
            f"{n_degen}\t{n_bd}\t{n_md}\t{mean_r:.4f}\t{r_sum:.4f}\t{int(fired)}"
        )
        if fired:
            fired_count += 1
            fired_restore_total += r_sum
            fired_degen_restore_sum += sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (0, 8))
            fired_bd_restore_sum += sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (1, 7))
            fired_md_restore_sum += sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (2, 3, 4, 5, 6))
            fired_steps_list.append({
                "method": s["method"], "step": s["step"],
                "zvf": s["zvf"], "pcd": s["pcd"],
                "n": n, "n_degen": n_degen, "n_bd": n_bd, "n_md": n_md,
                "mean_restore": mean_r, "restore_total": r_sum,
                "d_restore_frac": sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (0, 8)) / max(1e-9, r_sum),
                "b_restore_frac": sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (1, 7)) / max(1e-9, r_sum),
                "m_restore_frac": sum(r for j, r in enumerate(s["restores"]) if s["ks"][j] in (2, 3, 4, 5, 6)) / max(1e-9, r_sum),
            })

    step_fp = OUT_DIR / "p7_costeff_boot_step.tsv"
    step_fp.write_text("\n".join(step_rows) + "\n")
    print(f"wrote {step_fp}")

    # ---- summary JSON ----
    summary_json = {
        "evidence_base": "N2 four-method reward tensors (40 steps x 4 methods x 16 prompts = 2560 prompt-step obs)",
        "method": "step-level bootstrap (resample 160 (method, step) indices with replacement, n_boot=4000, seed=20260704) on rest/1k-extra-rollouts",
        "regime_counts": {
            "degenerate_k0_k8": n_degenerate, "boundary_k1_k7": n_boundary,
            "mid_k2_to_6": n_mid, "total": n_total,
        },
        "controllers": [
            {
                "name": f"{ctrl}@{tau}" if tau is not None else ctrl,
                "n_fires_total": real_per_ctrl[(ctrl, tau)]["n_fires"],
                "restored_total": round(real_per_ctrl[(ctrl, tau)]["restore_total"], 2),
                "extra_rollouts": real_per_ctrl[(ctrl, tau)]["extra"],
                "rpk_real": round(real_per_ctrl[(ctrl, tau)]["rpk_real"], 4),
                "rpk_boot_mean": round(statistics.mean(boot_samples[(ctrl, tau)]), 4),
                "rpk_boot_ci": [
                    round(sorted(boot_samples[(ctrl, tau)])[boot_lo], 4),
                    round(sorted(boot_samples[(ctrl, tau)])[boot_hi], 4),
                ],
            }
            for (ctrl, tau, _) in controllers
        ],
        "deltas_vs_zvf_triage_0.5": [
            {
                "name": f"{ctrl}@{tau}" if tau is not None else ctrl,
                "delta_mean": round(base_mean - statistics.mean(boot_samples[(ctrl, tau)]), 4),
                "delta_ci": [
                    round(sorted([base_samples[i] - boot_samples[(ctrl, tau)][i] for i in range(BOOT)])[boot_lo], 4),
                    round(sorted([base_samples[i] - boot_samples[(ctrl, tau)][i] for i in range(BOOT)])[boot_hi], 4),
                ],
                "excludes_zero": (
                    sorted([base_samples[i] - boot_samples[(ctrl, tau)][i] for i in range(BOOT)])[boot_lo] > 0
                    or sorted([base_samples[i] - boot_samples[(ctrl, tau)][i] for i in range(BOOT)])[boot_hi] < 0
                ),
            }
            for (ctrl, tau, _) in controllers
            if (ctrl, tau) != ("zvf_triage", 0.50)
        ],
        "zvf_triage_0.5_fired_steps": {
            "n_fired_steps": fired_count,
            "n_fired_steps_per_method": fired_count / 4.0,
            "restore_sum_per_fired_step_mean": (
                statistics.mean(s["restore_total"] for s in fired_steps_list) if fired_steps_list else 0.0
            ),
            "fraction_d_restore": fired_degen_restore_sum / max(1e-9, fired_restore_total),
            "fraction_b_restore": fired_bd_restore_sum / max(1e-9, fired_restore_total),
            "fraction_m_restore": fired_md_restore_sum / max(1e-9, fired_restore_total),
            "first_5_fired_steps": fired_steps_list[:5],
        },
    }
    json_fp = OUT_DIR / "p7_costeff_boot_summary.json"
    json_fp.write_text(json.dumps(summary_json, indent=2))
    print(f"wrote {json_fp}")

    # ---- console headline ----
    print("\n=== HEADLINE: bootstrap CIs on cost-efficiency (n_boot=4000) ===")
    print(f"{'controller':<22}{'fires/m':>10}{'rpk_real':>10}{'rpk_boot_mean':>14}{'95% CI':>22}")
    for ctrl, tau, _ in controllers:
        real = real_per_ctrl[(ctrl, tau)]
        samples = sorted(boot_samples[(ctrl, tau)])
        b_mean = statistics.mean(boot_samples[(ctrl, tau)])
        ctrl_label = f"{ctrl}@{tau}" if tau is not None else ctrl
        print(f"{ctrl_label:<22}{real['fires_per_method']:>10.1f}{real['rpk_real']:>10.2f}"
              f"{b_mean:>14.4f}[{samples[boot_lo]:.2f}, {samples[boot_hi]:.2f}]")

    print("\n=== DELTAS vs zvf_triage@0.50 ===")
    print(f"{'controller':<22}{'Δ_mean':>10}{'95% CI':>22}{'excludes_zero':>15}")
    for ctrl, tau, _ in controllers:
        if (ctrl, tau) == ("zvf_triage", 0.50):
            continue
        samples = boot_samples[(ctrl, tau)]
        delta = sorted([base_samples[i] - samples[i] for i in range(BOOT)])
        b_mean = statistics.mean(samples)
        d_mean = base_mean - b_mean
        excl_zero = (delta[boot_lo] > 0 or delta[boot_hi] < 0)
        ctrl_label = f"{ctrl}@{tau}" if tau is not None else ctrl
        print(f"{ctrl_label:<22}{d_mean:>10.3f}[{delta[boot_lo]:+.3f}, {delta[boot_hi]:+.3f}]{str(excl_zero):>15}")

    print("\n=== Per-regime (cost-efficiency, real data) ===")
    for line in regime_rows[1:]:
        print(f"  {line}")

    print("\n=== Fired-steps decomposition (zvf_triage@0.50) ===")
    print(f"  n_fired_steps (across all methods): {fired_count} / {n_steps} = {fired_count / n_steps:.4f}")
    print(f"  mean restore_sum per fired step: {(fired_restore_total / fired_count) if fired_count else 0:.3f}")
    print(f"  fired-step restore composition:")
    print(f"    degenerate k in {{0,8}} fraction: {fired_degen_restore_sum / max(1e-9, fired_restore_total):.4f}")
    print(f"    boundary k in {{1,7}}: {fired_bd_restore_sum / max(1e-9, fired_restore_total):.4f}")
    print(f"    mid k in {{2..6}}: {fired_md_restore_sum / max(1e-9, fired_restore_total):.4f}")


if __name__ == "__main__":
    main()
