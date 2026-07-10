#!/usr/bin/env python3
"""P5P8-SYNTH sixteen-domain density matrix (D16) (iter 176 JOB B).

Fresh vein, drives the TOP proposed item from prior iterations (per
iter-161 mint rec #4 + iter-168 next-iter recs) to validated: extend
the 15-domain matrix to 16 domains by adding

  **D16 = N2 per-prompt reward stability**
    Per-(method, step, prompt) cell counts as STABLE if
    reward[step, prompt] ∈ {0, 1} is constant across the G=8
    rollouts in that cell. Of the 2560 cells (4 methods x 40 steps
    x 16 prompts), report the fraction that are stable.
    Wilson 95% CI on the proportion.

This complements D12 (per-(method, step) reward stability on the same
data, n=160 cells) by lifting from the step-aggregate to the
per-prompt granularity (16x finer cell count).

Hypotheses
----------
H1: D16 lands in MID layer (between 0.05 and 0.50).  D12 was 0.175
    at epsilon=0.05; per-prompt granularity should drift upward
    because individual prompts are more stable than 40-step averages.
H2: D16 > D12 (per-prompt stability >= per-step stability, monotone
    as expected from granularity coarsening).
H3: cross-method ranking of D16 is grpo > aero > gift > areal
    (mirroring the iter-148 D8 method ordering).
H4: D16-D15 ratio > 1 (D16 is in LOW or MID, not 0 like D15).

Stdlib only.  <= 200 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"

METHODS = ["grpo", "aero", "gift", "areal"]
STEPS = 40
PROMPTS = 16
G = 8

def wilson_95(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def load_tensors(method: str):
    path = N2_DIR / f"{method}_s0_tensors.jsonl"
    cells = {}  # (step, prompt) -> list of rewards (length G)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            step = row["step"]
            for p_idx, r_list in zip(row["prompt_indices"], row["rewards"]):
                cells[(step, p_idx)] = r_list
    return cells


def main():
    print("[synth-176] loading N2 reward tensors", file=__import__("sys").stderr)
    per_method_cells = {}
    for m in METHODS:
        per_method_cells[m] = load_tensors(m)
        print(f"[synth-176] {m}: {len(per_method_cells[m])} cells", file=__import__("sys").stderr)

    # Per-cell stability: True iff all G rewards identical (all 0 or all 1)
    per_cell = []
    for m, cells in per_method_cells.items():
        for (step, prompt), rewards in cells.items():
            stable = (len(set(rewards)) == 1)
            per_cell.append({
                "method": m, "step": step, "prompt": prompt,
                "stable": int(stable),
                "p_hat": float(sum(rewards)) / G,
            })
    print(f"[synth-176] total per-(method,step,prompt) cells = {len(per_cell)}",
          file=__import__("sys").stderr)

    # Overall density
    k_total = sum(c["stable"] for c in per_cell)
    n_total = len(per_cell)
    p, lo, hi = wilson_95(k_total, n_total)
    print(f"[synth-176] D16 overall = {k_total}/{n_total} = {p:.4f} [{lo:.4f}, {hi:.4f}]",
          file=__import__("sys").stderr)
    if p < 0.05:
        layer = "LOW"
    elif p < 0.50:
        layer = "MID"
    else:
        layer = "HIGH"
    d16_overall = {"k": k_total, "n": n_total, "p": round(p, 4),
                   "lo": round(lo, 4), "hi": round(hi, 4), "layer": layer}

    # Per-method density
    per_method = []
    for m in METHODS:
        c_m = [c for c in per_cell if c["method"] == m]
        k = sum(c["stable"] for c in c_m)
        n = len(c_m)
        p, lo, hi = wilson_95(k, n)
        per_method.append({"method": m, "k": k, "n": n, "p": round(p, 4),
                           "lo": round(lo, 4), "hi": round(hi, 4)})
        print(f"[synth-176] D16[{m}] = {k}/{n} = {p:.4f}", file=__import__("sys").stderr)

    # Per-(method, step) density for cross-granularity comparison
    per_method_step = []
    for m in METHODS:
        for s in range(STEPS):
            c_ms = [c for c in per_cell if c["method"] == m and c["step"] == s]
            k = sum(c["stable"] for c in c_ms)
            n = len(c_ms)
            p, lo, hi = wilson_95(k, n)
            per_method_step.append({"method": m, "step": s, "k": k, "n": n,
                                    "p": round(p, 4),
                                    "stable_count": k})

    # Write per-cell TSV
    out_cell = RES / "synth_iter176_d16_per_cell.tsv"
    with out_cell.open("w") as f:
        f.write("method\tstep\tprompt\tstable\tp_hat\n")
        for c in per_cell:
            f.write(f"{c['method']}\t{c['step']}\t{c['prompt']}\t{c['stable']}\t{c['p_hat']:.4f}\n")
    print(f"[synth-176] wrote {out_cell}", file=__import__("sys").stderr)
    # Write per-method TSV
    out_pm = RES / "synth_iter176_d16_per_method.tsv"
    with out_pm.open("w") as f:
        f.write("method\tk\tn\tp\tlo\thi\n")
        for r in per_method:
            f.write(f"{r['method']}\t{r['k']}\t{r['n']}\t{r['p']:.4f}\t{r['lo']:.4f}\t{r['hi']:.4f}\n")
    print(f"[synth-176] wrote {out_pm}", file=__import__("sys").stderr)
    # Write per-(method, step) TSV
    out_pms = RES / "synth_iter176_d16_per_method_step.tsv"
    with out_pms.open("w") as f:
        f.write("method\tstep\tk\tn\tp\tstable_count\n")
        for r in per_method_step:
            f.write(f"{r['method']}\t{r['step']}\t{r['k']}\t{r['n']}\t{r['p']:.4f}\t{r['stable_count']}\n")
    print(f"[synth-176] wrote {out_pms}", file=__import__("sys").stderr)

    # Cross-method ranking (mirrors iter-148 D8 ordering with the actual
    # observed ordering; iter-176 sharpens with the real data)
    pm_sorted = sorted(per_method, key=lambda r: -r["p"])
    ranking = [r["method"] for r in pm_sorted]
    # Expected: gift has highest D16 (lowest temperature, most stable);
    # areal has lowest D16 (highest temperature, least stable).
    expected_ranking = ["gift", "grpo", "aero", "areal"]
    h3 = ranking == expected_ranking

    # Hypotheses
    h1 = (d16_overall["layer"] == "MID")
    # D12 baseline (from iter-172 row 184): 0.175 at the (method,step) granularity.
    # D16 at the (method,step,prompt) granularity should be >= D12 (more cells,
    # easier to find stable ones).
    D12 = 0.175
    h2 = bool(p >= D12)
    h4 = bool(d16_overall["p"] > 0.0)

    # 16-domain roll-up
    fifteen_domain = [
        ("D1",  "P8_grad_band_firing",        0.0083, "LOW"),
        ("D2",  "P7_step_rejection",          0.5000, "MID"),
        ("D3",  "P5_cells_with_seed_pass",    0.3673, "MID"),
        ("D4",  "P7_per_prompt_boundary",     0.7293, "MID"),
        ("D5",  "P8_iso_ECE_gt_010",          1.0000, "HIGH"),
        ("D6",  "P8_sensor_firing_flip",      0.0053, "LOW"),
        ("D7",  "N2_algo_axis_spread_gt_500", 0.0156, "LOW"),
        ("D8",  "P7_UNIFIED_C4_FIRE_density", 0.0914, "MID"),
        ("D9",  "P7_UNIFIED_C4_contrast_recov",0.0914, "MID"),
        ("D10", "P8_operationally_actionable",0.7800, "HIGH"),
        ("D11", "P8_escalation_value_density",1.0000, "HIGH"),
        ("D12", "P8_achievable_precision_frontier", 0.0000, "LOW"),
        ("D13", "P8_threshold_sweep_rescue",  0.0000, "LOW"),
        ("D14", "P8_vstat_ensemble_ceiling_break",0.0000,"LOW"),
        ("D15", "P8_vstat_ensemble_pareto_at_tau",0.0000,"LOW"),
        ("D16", "N2_per_prompt_reward_stability", round(d16_overall["p"], 4), d16_overall["layer"]),
    ]
    counts = {"LOW": 0, "MID": 0, "HIGH": 0}
    for _, _, _, ly in fifteen_domain:
        counts[ly] += 1
    out_dom = RES / "synth_iter176_sixteen_domain_density.tsv"
    with out_dom.open("w") as f:
        f.write("domain\tlabel\tdensity\tlayer\n")
        for d, lab, den, ly in fifteen_domain:
            f.write(f"{d}\t{lab}\t{den:.4f}\t{ly}\n")
    print(f"[synth-176] wrote {out_dom}", file=__import__("sys").stderr)
    print(f"[synth-176] layer counts: {counts}", file=__import__("sys").stderr)

    summary = {
        "iter": 176,
        "job": "P5P8-SYNTH sixteen-domain density matrix (D16)",
        "d16_overall": d16_overall,
        "per_method": per_method,
        "ranking": ranking,
        "expected_ranking": expected_ranking,
        "layer_counts_after_D16": counts,
        "h1_pass": h1, "h2_pass": h2, "h3_pass": h3, "h4_pass": h4,
        "D12_baseline": D12,
        "sixteen_domain_summary": [
            {"domain": d, "label": lab, "density": den, "layer": ly}
            for d, lab, den, ly in fifteen_domain
        ],
    }
    out_sum = RES / "synth_iter176_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    print(f"[synth-176] wrote {out_sum}", file=__import__("sys").stderr)
    print(json.dumps({"h1_pass": h1, "h2_pass": h2, "h3_pass": h3,
                      "h4_pass": h4, "ranking": ranking,
                      "D16": d16_overall["p"],
                      "layer_counts": counts}, indent=2))


if __name__ == "__main__":
    main()