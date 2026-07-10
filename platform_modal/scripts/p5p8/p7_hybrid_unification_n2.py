#!/usr/bin/env python3
"""
Pillar-7 (P7) per-(method, step) Hybrid C3 unification replay on the N2
four-method reward-tensor panel.

Iter 31: validates iter-27's panel-conditional unification prediction --
"Hybrid (C3) strictly dominates both zvf-triage (C1) and Dualformer-Auto
(C2) only when the per-step ZVF trajectory reaches the saturation band
zvf >= tau+delta" -- on the N2 four-method data.

KEY EVIDENCE: gift's N2 trajectory reaches zvf>=0.9 in 8 of 40 steps
(the only N2 method that does). This is the FIRST panel where iter-27's
prediction is actually testable: Hybrid C3 should strictly dominate C1
(only escalates) on gift because C3 de-escalates the saturation-band
steps where escalation would only waste rollouts.

Per-(method, step) controllers (all dispatch on per-step ZVF z_t,
G_base=8):

  C0 baseline     : G_t = 8                           (compute = 320/method)
  C1 zvf-triage@τ : G_t = 16 if z_t >= τ else 8       (escalate on boundary)
  C2 Dualformer@τ : G_t = 4 if z_t >= τ else 8        (de-escalate easy steps)
  C3 Hybrid@τ+δ   : G_t = 16 if τ <= z_t < τ+δ, 4 if z_t >= τ+δ, 8 otherwise

Per-method metrics (n=4 methods × 40 steps = 160 step-units):
  total_G         sum of G_t over 40 steps (= compute proxy)
  savings_vs_C0   (total_G_C0 - total_G_Ci) / total_G_C0
  n_fire          number of steps with G_t != 8
  select_rate     n_fire / 40
  headroom_bad    steps fired on zvf >= 0.99 (saturated, no escalation value)

Statistical rigor:
  - Per-method total_G (point estimate; n=40 steps too few to bootstrap).
  - Bootstrap-CI (B=2000, percentile) on per-method total_G treating the
    40 steps as iid (steps in a single GRPO run are serially correlated
    but the bootstrap still gives a meaningful uncertainty band).
  - Paired bootstrap-CI on per-method Δtotal_G (C3-C1, C3-C2, C2-C1) --
    the headline falsifiable claim.
  - Method-stratified contrasts: gift is the saturation-band panel;
    grpo/aero/areal are interior-only panels (C3 should reduce to C1).

References (verified):
  - su2024dualformer     (Su et al., 2024, "Dualformer")
  - alphaproof2025nature (AlphaProof, Nature 2025)

Outputs (worktree-relative paths):
  experiments/results/p5p8/p7_hybrid_n2_per_step.tsv
  experiments/results/p5p8/p7_hybrid_n2_per_method.tsv
  experiments/results/p5p8/p7_hybrid_n2_summary.json
"""
from __future__ import annotations
import csv
import json
import random
import statistics
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = ROOT / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 0.7           # primary zvf-triage threshold
TAU_DELTA = 0.2     # Hybrid band width: z in [tau, tau+tau_delta) escalates
G_BASE = 8          # N2 fixed group size
G_ESC = 16          # escalation: 2x base
G_DES = 4           # de-escalation: 0.5x base
N_STEPS = 40        # N2 panel step count per method
N_BOOT = 2000       # bootstrap replicates
RNG_SEED = 20260704
SATURATION_THRESHOLD = 0.9  # saturation band: zvf >= tau+delta


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_n2_methods() -> list[dict]:
    """Load every N2 method JSONL on disk; return list of dicts with
    method + step-level zvf trajectory (40 entries each)."""
    out = []
    for path in sorted(N2_DIR.glob("*_s0_tensors.jsonl")):
        method = path.stem.replace("_s0_tensors", "")
        rows = []
        with path.open() as fh:
            for line in fh:
                d = json.loads(line)
                rows.append(d)
        if len(rows) != N_STEPS:
            continue
        zvfs = [float(d.get("zvf", 0.0)) for d in rows]
        out.append({"method": method, "zvfs": zvfs})
    return out


# ----------------------------------------------------------------------------
# Controller dispatch
# ----------------------------------------------------------------------------
def c0(z: list[float]) -> list[int]:
    return [G_BASE] * len(z)


def c1_zvf_triage(z: list[float], tau: float) -> list[int]:
    return [G_ESC if zt >= tau else G_BASE for zt in z]


def c2_dualformer(z: list[float], tau: float) -> list[int]:
    return [G_DES if zt >= tau else G_BASE for zt in z]


def c3_hybrid(z: list[float], tau: float, delta: float) -> list[int]:
    out = []
    for zt in z:
        if zt >= tau + delta:
            out.append(G_DES)        # saturation band: de-escalate
        elif zt >= tau:
            out.append(G_ESC)        # boundary band: escalate
        else:
            out.append(G_BASE)
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(G_t: list[int], z: list[float]) -> dict:
    n = len(G_t)
    total = sum(G_t)
    n_fire = sum(1 for g in G_t if g != G_BASE)
    sel = n_fire / n
    headroom_bad = sum(1 for g, zt in zip(G_t, z) if g != G_BASE and zt >= 0.99)
    sat_band_fired = sum(1 for g, zt in zip(G_t, z) if g != G_BASE and zt >= SATURATION_THRESHOLD)
    return {
        "total_G": total,
        "n_fire": n_fire,
        "select_rate": sel,
        "headroom_bad": headroom_bad,
        "sat_band_fired": sat_band_fired,
    }


# ----------------------------------------------------------------------------
# Bootstrap CI (percentile) on a list of step-level scalars
# ----------------------------------------------------------------------------
def boot_ci(values: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sd": float("nan")}
    rng = random.Random(seed)
    n = len(values)
    sums = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sums.append(sum(sample))
    sums.sort()
    lo = sums[int(0.025 * n_boot)]
    hi = sums[int(0.975 * n_boot)]
    return {
        "mean": statistics.mean(values),
        "sum": sum(values),
        "lo": lo,
        "hi": hi,
        "sd": statistics.stdev(values) if n > 1 else 0.0,
        "n": n,
    }


def paired_boot_ci_delta(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """Paired bootstrap on the (a-b) total over the SAME 40 steps.

    Returns the bootstrap distribution of sum(a_sample) - sum(b_sample) where
    a_sample, b_sample are resampled from (a, b) using the same index."""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    diffs_total = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_sum = sum(a[i] for i in idx)
        b_sum = sum(b[i] for i in idx)
        diffs_total.append(a_sum - b_sum)
    diffs_total.sort()
    diffs_pa = [a[i] - b[i] for i in range(n)]
    return {
        "mean_diff_total": sum(diffs_pa),
        "mean_diff_per_step": statistics.mean(diffs_pa),
        "lo": diffs_total[int(0.025 * n_boot)],
        "hi": diffs_total[int(0.975 * n_boot)],
        "n": n,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    methods = load_n2_methods()
    n_methods = len(methods)
    assert n_methods >= 1, f"Need >=1 N2 method; got {n_methods}"

    controllers = {
        "C0_baseline": lambda z: c0(z),
        f"C1_zvf_triage@{TAU:.2f}": lambda z: c1_zvf_triage(z, TAU),
        f"C2_dualformer@{TAU:.2f}": lambda z: c2_dualformer(z, TAU),
        f"C3_hybrid@{TAU:.2f}+{TAU_DELTA:.2f}": lambda z: c3_hybrid(z, TAU, TAU_DELTA),
    }
    c1_key = f"C1_zvf_triage@{TAU:.2f}"
    c2_key = f"C2_dualformer@{TAU:.2f}"
    c3_key = f"C3_hybrid@{TAU:.2f}+{TAU_DELTA:.2f}"

    # ---- Per-(method, step) table (for transparency)
    per_step_rows = []
    for m in methods:
        for ctrl_name, fn in controllers.items():
            G_t = fn(m["zvfs"])
            for step_idx, (g, zt) in enumerate(zip(G_t, m["zvfs"])):
                per_step_rows.append({
                    "method": m["method"],
                    "controller": ctrl_name,
                    "step": step_idx,
                    "zvf": zt,
                    "G_t": g,
                    "fired": int(g != G_BASE),
                    "sat_band_fired": int(g != G_BASE and zt >= SATURATION_THRESHOLD),
                })

    out_step_tsv = OUT_DIR / "p7_hybrid_n2_per_step.tsv"
    with out_step_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_step_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_step_rows)

    # ---- Per-method metrics table + paired contrasts
    per_method_rows = []
    per_method_totalG = {name: {} for name in controllers}
    per_method_G_per_step = {name: {} for name in controllers}  # method -> list of G_t
    per_method_savings = {name: {} for name in controllers}

    for m in methods:
        row = {"method": m["method"], "n_steps": N_STEPS,
               "zvf_min": min(m["zvfs"]), "zvf_max": max(m["zvfs"]),
               "zvf_mean": sum(m["zvfs"]) / N_STEPS,
               "n_sat_band": sum(1 for zt in m["zvfs"] if zt >= SATURATION_THRESHOLD)}
        base_total = None
        for ctrl_name, fn in controllers.items():
            G_t = fn(m["zvfs"])
            met = metrics(G_t, m["zvfs"])
            row[f"{ctrl_name}_total_G"] = met["total_G"]
            row[f"{ctrl_name}_n_fire"] = met["n_fire"]
            row[f"{ctrl_name}_select_rate"] = met["select_rate"]
            row[f"{ctrl_name}_headroom_bad"] = met["headroom_bad"]
            row[f"{ctrl_name}_sat_band_fired"] = met["sat_band_fired"]
            per_method_totalG[ctrl_name][m["method"]] = met["total_G"]
            per_method_G_per_step[ctrl_name][m["method"]] = G_t
            if ctrl_name == "C0_baseline":
                base_total = met["total_G"]
        for ctrl_name in controllers:
            if ctrl_name == "C0_baseline":
                row[f"{ctrl_name}_savings"] = 0.0
                continue
            saving = (base_total - row[f"{ctrl_name}_total_G"]) / base_total
            row[f"{ctrl_name}_savings"] = saving
            per_method_savings[ctrl_name][m["method"]] = saving
        per_method_rows.append(row)

    # Per-method table with bootstrap CIs on total_G (B=2000 resamples of
    # the 40 steps within the same method).
    for row in per_method_rows:
        method = row["method"]
        for ctrl_name in controllers:
            G_t_list = per_method_G_per_step[ctrl_name][method]
            ci = boot_ci([float(g) for g in G_t_list])
            row[f"{ctrl_name}_total_G_lo"] = ci["lo"]
            row[f"{ctrl_name}_total_G_hi"] = ci["hi"]

    out_method_tsv = OUT_DIR / "p7_hybrid_n2_per_method.tsv"
    with out_method_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_method_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_method_rows)

    # ---- Headline summary
    summary = {
        "config": {
            "tau": TAU,
            "tau_delta": TAU_DELTA,
            "saturation_threshold": SATURATION_THRESHOLD,
            "G_base": G_BASE, "G_esc": G_ESC, "G_des": G_DES,
            "n_steps_per_method": N_STEPS,
            "n_methods": n_methods,
            "n_boot": N_BOOT, "rng_seed": RNG_SEED,
        },
        "per_method_summaries": {},
        "per_controller_pooled": {},
        "paired_contrasts_per_method": {},
        "paired_contrasts_pooled": {},
        "headroom_bad_per_controller": {},
        "panel_conditional_unification_test": {},
    }

    # Per-method summary dicts (raw + bootstrap)
    for row in per_method_rows:
        method = row["method"]
        ms = {
            "n_sat_band_steps": row["n_sat_band"],
            "zvf_min": row["zvf_min"], "zvf_max": row["zvf_max"], "zvf_mean": row["zvf_mean"],
            "controllers": {},
        }
        for ctrl_name in controllers:
            ms["controllers"][ctrl_name] = {
                "total_G": row[f"{ctrl_name}_total_G"],
                "total_G_lo": row[f"{ctrl_name}_total_G_lo"],
                "total_G_hi": row[f"{ctrl_name}_total_G_hi"],
                "savings_vs_C0": row[f"{ctrl_name}_savings"],
                "n_fire": row[f"{ctrl_name}_n_fire"],
                "sat_band_fired": row[f"{ctrl_name}_sat_band_fired"],
                "headroom_bad": row[f"{ctrl_name}_headroom_bad"],
            }
        summary["per_method_summaries"][method] = ms

    # Pooled per-controller (treat the 4 methods × 40 steps = 160 step-units
    # as the iid sample; report mean ± CI on per-(method,step) G_t).
    pooled_per_step = {name: [] for name in controllers}
    for m in methods:
        for ctrl_name in controllers:
            pooled_per_step[ctrl_name].extend(per_method_G_per_step[ctrl_name][m["method"]])
    for ctrl_name in controllers:
        ci = boot_ci([float(g) for g in pooled_per_step[ctrl_name]])
        summary["per_controller_pooled"][ctrl_name] = {
            "n_step_units": len(pooled_per_step[ctrl_name]),
            "mean_G_t": ci["mean"],
            "mean_G_t_lo": ci["lo"] / N_STEPS,  # per-method total_G mean from per-step CI sum / 40
            "mean_G_t_hi": ci["hi"] / N_STEPS,
            "total_G_pooled": ci["sum"],
            "total_G_pooled_lo": ci["lo"],
            "total_G_pooled_hi": ci["hi"],
        }

    # Paired bootstrap contrasts per method (per-(method, step) G_t)
    for m in methods:
        method = m["method"]
        contrasts = {}
        for (a_name, b_name, key) in [
            (c3_key, c1_key, "C3_minus_C1"),
            (c3_key, c2_key, "C3_minus_C2"),
            (c2_key, c1_key, "C2_minus_C1"),
        ]:
            a_per_step = [float(g) for g in per_method_G_per_step[a_name][method]]
            b_per_step = [float(g) for g in per_method_G_per_step[b_name][method]]
            contrasts[key] = paired_boot_ci_delta(a_per_step, b_per_step)
        summary["paired_contrasts_per_method"][method] = contrasts

    # Pooled paired contrasts (160 paired step-units across all methods)
    contrasts_pooled = {}
    for (a_name, b_name, key) in [
        (c3_key, c1_key, "C3_minus_C1"),
        (c3_key, c2_key, "C3_minus_C2"),
        (c2_key, c1_key, "C2_minus_C1"),
    ]:
        a_pooled = []
        b_pooled = []
        for m in methods:
            a_pooled.extend([float(g) for g in per_method_G_per_step[a_name][m["method"]]])
            b_pooled.extend([float(g) for g in per_method_G_per_step[b_name][m["method"]]])
        contrasts_pooled[key] = paired_boot_ci_delta(a_pooled, b_pooled)
    summary["paired_contrasts_pooled"] = contrasts_pooled

    # Headroom (well-calibration check)
    for ctrl_name in controllers:
        if ctrl_name == "C0_baseline":
            continue
        headroom_total = sum(row[f"{ctrl_name}_headroom_bad"] for row in per_method_rows)
        sat_band_total = sum(row[f"{ctrl_name}_sat_band_fired"] for row in per_method_rows)
        summary["headroom_bad_per_controller"][ctrl_name] = {
            "headroom_bad_total": headroom_total,
            "sat_band_fired_total": sat_band_total,
            "n_sat_band_total": sum(row["n_sat_band"] for row in per_method_rows),
        }

    # Panel-conditional unification test:
    # - On saturation-band panels (n_sat_band > 0, i.e., gift): C3 strictly
    #   dominates C1 (C3 de-escalates saturation band; C1 wrongly escalates).
    # - On interior-only panels (n_sat_band == 0): C3 reduces to C1.
    summary["panel_conditional_unification_test"] = {
        "n_methods_with_saturation_band": sum(1 for r in per_method_rows if r["n_sat_band"] > 0),
        "saturation_band_methods": [r["method"] for r in per_method_rows if r["n_sat_band"] > 0],
        "interior_only_methods": [r["method"] for r in per_method_rows if r["n_sat_band"] == 0],
        "c3_equals_c1_check": {},
    }
    for m in methods:
        method = m["method"]
        c1_g = per_method_G_per_step[c1_key][method]
        c3_g = per_method_G_per_step[c3_key][method]
        diffs = [a - b for a, b in zip(c3_g, c1_g)]
        n_diff = sum(1 for d in diffs if d != 0)
        summary["panel_conditional_unification_test"]["c3_equals_c1_check"][method] = {
            "n_steps_where_C3_differs_from_C1": n_diff,
            "is_bit_identical_to_C1": n_diff == 0,
        }

    out_json = OUT_DIR / "p7_hybrid_n2_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- Console echo
    print("=" * 76)
    print("P7 Hybrid unification replay on N2 four-method step-level ZVF trajectories")
    print(f"tau={TAU}, tau_delta={TAU_DELTA}, G_base={G_BASE}, G_esc={G_ESC}, G_des={G_DES}")
    print("=" * 76)
    print(f"\nPer-method summary (n_sat_band = steps with zvf >= {SATURATION_THRESHOLD}):")
    for row in per_method_rows:
        print(f"  {row['method']:6s}  zvf=[{row['zvf_min']:.3f}, {row['zvf_max']:.3f}]  mean={row['zvf_mean']:.3f}  "
              f"n_sat={row['n_sat_band']:2d}  C1_total={row[f'{c1_key}_total_G']}  "
              f"C2_total={row[f'{c2_key}_total_G']}  C3_total={row[f'{c3_key}_total_G']}  "
              f"C0_total={row['C0_baseline_total_G']}")

    print(f"\nPer-method savings vs C0 baseline (fraction of compute NOT spent):")
    for row in per_method_rows:
        print(f"  {row['method']:6s}  C1={row[f'{c1_key}_savings']:+.4f}  C2={row[f'{c2_key}_savings']:+.4f}  "
              f"C3={row[f'{c3_key}_savings']:+.4f}  "
              f"  C3_sat_fired={row[f'{c3_key}_sat_band_fired']}")

    print(f"\nPer-method paired bootstrap contrasts on per-step G_t (n_boot={N_BOOT}):")
    for method, contrasts in summary["paired_contrasts_per_method"].items():
        c3mc1 = contrasts["C3_minus_C1"]
        c3mc2 = contrasts["C3_minus_C2"]
        c2mc1 = contrasts["C2_minus_C1"]
        sig3_1 = "***" if c3mc1["hi"] < 0 or c3mc1["lo"] > 0 else "n.s."
        sig3_2 = "***" if c3mc2["hi"] < 0 or c3mc2["lo"] > 0 else "n.s."
        sig2_1 = "***" if c2mc1["hi"] < 0 or c2mc1["lo"] > 0 else "n.s."
        print(f"  {method:6s}  C3-C1 Δ={c3mc1['mean_diff_total']:+5.0f} [{c3mc1['lo']:+5.0f}, {c3mc1['hi']:+5.0f}] {sig3_1}  "
              f"C3-C2 Δ={c3mc2['mean_diff_total']:+5.0f} [{c3mc2['lo']:+5.0f}, {c3mc2['hi']:+5.0f}] {sig3_2}  "
              f"C2-C1 Δ={c2mc1['mean_diff_total']:+5.0f} [{c2mc1['lo']:+5.0f}, {c2mc1['hi']:+5.0f}] {sig2_1}")

    print(f"\nPooled (160 step-units) paired bootstrap contrasts:")
    for key, v in summary["paired_contrasts_pooled"].items():
        sig = "***" if v["lo"] > 0 or v["hi"] < 0 else "n.s."
        print(f"  {key:12s}  Δ_total={v['mean_diff_total']:+6.0f}  "
              f"95%CI=[{v['lo']:+6.0f}, {v['hi']:+6.0f}]  {sig}")

    print(f"\nPanel-conditional unification test:")
    pcut = summary["panel_conditional_unification_test"]
    print(f"  saturation-band methods ({pcut['n_methods_with_saturation_band']}): {pcut['saturation_band_methods']}")
    print(f"  interior-only methods: {pcut['interior_only_methods']}")
    for method, chk in pcut["c3_equals_c1_check"].items():
        print(f"  {method:6s}  C3≡C1? {chk['is_bit_identical_to_C1']}  "
              f"({chk['n_steps_where_C3_differs_from_C1']} steps differ)")

    print(f"\nHeadroom-bad / sat-band fires (saturated, no escalation value):")
    for ctrl, d in summary["headroom_bad_per_controller"].items():
        print(f"  {ctrl:35s}  headroom_bad={d['headroom_bad_total']}  "
              f"sat_band_fired={d['sat_band_fired_total']}  / {d['n_sat_band_total']} sat-band steps")

    print(f"\nOutputs:\n  {out_step_tsv.relative_to(ROOT)}\n  "
          f"{out_method_tsv.relative_to(ROOT)}\n  {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()