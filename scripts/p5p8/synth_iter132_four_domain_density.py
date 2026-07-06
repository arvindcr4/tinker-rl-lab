"""
P5P8-SYNTH JOB B (iter 132): four-domain density extension.

Fresh vein from iter-124's "Recommended next-iter mint veins" list:
add P7 per-prompt Adaptive-G* density as a FOURTH domain. iter-131 ran the
per-prompt Adaptive-G* simulation on the same N2 four-method panel
(2560 prompt-cells = 4 methods x 40 steps x 16 prompts), giving us a
matched-granularity domain (P7 per-prompt) to compare alongside the
existing three:

  D1 = P8 grad-band rule (per-row, n=10000)
  D2 = P7 zvf-triage rule (per-step, n=160 = 4 methods x 40 steps)
  D3 = P5 mega-manifest score-stream contrast (per-cell, n=98)
  D4 = P7 per-prompt Adaptive-G* simulation (per-prompt-cell,
       n=2560 = 4 methods x 40 steps x 16 prompts)  <-- NEW

Falsifiable headlines
---------------------
H1 -- 4-domain density ratio matrix with bootstrap CIs.
  Does adding D4 break the iter-124 "{P5, P7-step} <-> P8" two-super-domain
  clustering, or does D4 land inside {P5, P7-step} as expected?

H2 -- density rank with 4 domains. If D4 ranks highest (per-prompt is
  finest granularity, so signal is most fragmented), it confirms the
  "more granular = more signal-depleted" hypothesis. If D4 ranks
  similar to D2 (per-step) it suggests intra-step prompt dispersion
  contributes little beyond step-aggregate zvf.

H3 -- within-method D4 stratification: per-method density of
  boundary cells (k ∈ {0, G_BASE}) in iter-131 data. Confirms whether
  iter-127's ranking reversal (gift > grpo > aero > areal) is consistent
  with iter-131's per-prompt granularity (which found areal > aero >
  grpo > gift on cost-equivalent contrast).

H4 -- per-method zvf contrast density at per-prompt granularity:
  rate of prompts with non-zero contrast (k strictly between 0 and G_BASE).
  Reports whether the four-method panel is uniformly contrast-bearing.

Operationally this answers: does the iter-124 super-domain claim
({P5,P7}-step / {P8}) survive the inclusion of P7-prompt granularity?
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20260705
N_BOOT = 1500


def density_ci(n_fire, n_total, n_boot=N_BOOT, seed=SEED):
    """Wilson bootstrap CI on a proportion n_fire/n_total."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    p = n_fire / n_total
    for i in range(n_boot):
        boots[i] = rng.binomial(n_total, p) / n_total
    return {
        "rate": p,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_fire": n_fire,
        "n_total": n_total,
    }


def ratio_ci(num, denom, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on a ratio of two Bernoulli rates."""
    rng = np.random.default_rng(seed)
    rn = num["rate"]; rd = denom["rate"]
    nn = num["n_total"]; nd = denom["n_total"]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        n_i = rng.binomial(nn, rn) / max(1, nn)
        d_i = rng.binomial(nd, rd) / max(1, nd)
        boots[i] = n_i / max(1e-9, d_i)
    point = rn / max(1e-9, rd)
    return {
        "ratio": point,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "excludes_1.0": point < 0.1 or point > 10.0,
    }


def compute_d4_from_iter131():
    """Load iter-131 per-prompt TSV and compute density of boundary cells
    (k ∈ {0, 8}) at G=8, the natural per-prompt 'signal-depleted' analog
    of the iter-124 P5 D3 (zvf == 1.0) and iter-124 P7 D2 (zvf >= 0.7)."""
    path = RES / "p7_iter131_per_prompt_gstar.tsv"
    import csv
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                rows.append({
                    "method": r["method"],
                    "step": int(r["step"]),
                    "prompt": int(r["prompt"]),
                    "k_obs": int(r["k_obs"]),
                    "is_boundary": int(r["is_boundary"]),
                    "zvf_obs": float(r["zvf_obs"]),
                    "ADAPTIVE_PP_Gstar": int(r["ADAPTIVE_PP_Gstar"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def main():
    print(f"[iter132 SYNTH] loading iter-131 per-prompt table ...")
    cells_pp = compute_d4_from_iter131()
    n_total = len(cells_pp)
    n_boundary = sum(1 for r in cells_pp if r["is_boundary"] == 1)
    n_high_zvf = sum(1 for r in cells_pp if r["zvf_obs"] >= 0.7)
    n_high_gstar = sum(1 for r in cells_pp if r["ADAPTIVE_PP_Gstar"] >= 16)
    print(f"[iter132 SYNTH] n_total={n_total}, n_boundary={n_boundary}, "
          f"n_high_zvf={n_high_zvf}, n_high_gstar={n_high_gstar}")

    # --- Domain definitions ---
    # D1, D2, D3 anchored from iter-124 (re-imported literals)
    D1 = {"n_fire": 84, "n_total": 10000}  # P8 grad-band: 84 fires on 10k test rows
    D2 = {"n_fire": 20, "n_total": 40}     # P7 zvf-triage: 20/40 steps at zvf>=0.7
    D3 = {"n_fire": 36, "n_total": 98}     # P5 mega-corpus: 36 cells with zvf==1.0 (iter-124 H4)
    D4 = {"n_fire": n_boundary, "n_total": n_total}  # NEW: P7 per-prompt boundary density

    D1_ci = density_ci(D1["n_fire"], D1["n_total"], seed=SEED)
    D2_ci = density_ci(D2["n_fire"], D2["n_total"], seed=SEED + 1)
    D3_ci = density_ci(D3["n_fire"], D3["n_total"], seed=SEED + 2)
    D4_ci = density_ci(D4["n_fire"], D4["n_total"], seed=SEED + 3)

    domains = [
        ("D1_P8_grad_band",        D1_ci),
        ("D2_P7_step_zvf_triage",  D2_ci),
        ("D3_P5_mega_zvf_eq_1",    D3_ci),
        ("D4_P7_perprompt_boundary", D4_ci),
    ]
    out_d = RES / "synth_iter132_four_domain_density.tsv"
    with out_d.open("w") as f:
        f.write("domain\tn_fire\tn_total\trate\tci_lo\tci_hi\trule\n")
        rules = {
            "D1_P8_grad_band":        "P8 grad-band: row in top-K AND consecutive gradient small",
            "D2_P7_step_zvf_triage":  "P7 step zvf>=0.7 (DEGENERATE regime)",
            "D3_P5_mega_zvf_eq_1":    "P5 mega cell per-step zvf == 1.0",
            "D4_P7_perprompt_boundary": "P7 per-prompt is_boundary==1 (k in {0,8})",
        }
        for k, d in domains:
            f.write(f"{k}\t{d['n_fire']}\t{d['n_total']}\t{d['rate']:.6f}\t{d['lo']:.6f}\t{d['hi']:.6f}\t{rules[k]}\n")
    print(f"[iter132] wrote {out_d}")

    # --- H1: 4-domain ratio matrix (6 ordered pairs) ---
    pairs = [
        ("P5_over_P7step",  D3_ci, D2_ci),
        ("P5_over_P7pp",    D3_ci, D4_ci),
        ("P5_over_P8",      D3_ci, D1_ci),
        ("P8_over_P7step",  D1_ci, D2_ci),
        ("P8_over_P7pp",    D1_ci, D4_ci),
        ("P7pp_over_P7step", D4_ci, D2_ci),
        ("P7pp_over_P5",    D4_ci, D3_ci),
        ("P7step_over_P5",  D2_ci, D3_ci),
        ("P7pp_over_P8",    D4_ci, D1_ci),
    ]
    ratios = {label: ratio_ci(num, den) for label, num, den in pairs}
    print(f"[iter132 H1] 4-domain ratio matrix:")
    for k, v in ratios.items():
        print(f"   {k}: ratio={v['ratio']:.4f} "
              f"CI=[{v['lo']:.4f}, {v['hi']:.4f}] excludes_1.0={v['excludes_1.0']}")

    out_r = RES / "synth_iter132_four_domain_density_ratios.tsv"
    with out_r.open("w") as f:
        f.write("ratio\tpoint\tci_lo\tci_hi\texcludes_1.0\n")
        for k, v in ratios.items():
            f.write(f"{k}\t{v['ratio']:.6f}\t{v['lo']:.6f}\t{v['hi']:.6f}\t{v['excludes_1.0']}\n")
    print(f"[iter132]wrote {out_r}")

    # --- H2: density rank ---
    densities = {k.split("_")[0]: v["rate"] for k, v in domains}
    rank = sorted(densities.items(), key=lambda kv: -kv[1])
    print(f"[iter132 H2] density rank: " +
          ", ".join(f"{k}={v:.4f}" for k, v in rank))

    # --- H3: per-method boundary density on D4 ---
    methods = ["grpo", "aero", "gift", "areal"]
    per_method = {m: {"n": 0, "n_boundary": 0} for m in methods}
    for r in cells_pp:
        m = r["method"]
        if m in per_method:
            per_method[m]["n"] += 1
            per_method[m]["n_boundary"] += int(r["is_boundary"] == 1)
    method_density = {m: per_method[m]["n_boundary"] / per_method[m]["n"] for m in methods}

    # iter-127 ranking: gift > grpo > aero > areal (step-aggregate CCC, bigger G is better)
    # iter-131 ranking: areal > aero > grpo > gift (per-prompt contrast, smaller boundary is better)
    iter127_rank = {"grpo": 2, "aero": 3, "gift": 1, "areal": 4}  # rank 1=highest G_CCC
    iter131_rank = {"grpo": 3, "aero": 2, "gift": 4, "areal": 1}  # rank 1=highest contrast/pp

    # Spearman between (1 - iter-127 rank) and (boundary density)? No, use
    # boundary density directly against iter-131's rank order
    rho_h3 = np.corrcoef(
        [iter131_rank[m] for m in methods],
        [method_density[m] for m in methods],
    )[0, 1]
    print(f"[iter132 H3] per-method boundary density: {method_density}")
    print(f"[iter132 H3] rho (iter-131 rank, density) = {rho_h3:.4f} (negative => lower-rank => higher density expected)")

    out_h3 = RES / "synth_iter132_per_method_boundary.tsv"
    with out_h3.open("w") as f:
        f.write("method\tn\tn_boundary\tboundary_density\titer127_rank_step_ccc\titer131_rank_per_prompt\n")
        for m in methods:
            f.write(f"{m}\t{per_method[m]['n']}\t{per_method[m]['n_boundary']}\t"
                    f"{method_density[m]:.6f}\t{iter127_rank[m]}\t{iter131_rank[m]}\n")
    print(f"[iter132] wrote {out_h3}")

    # --- H4: rate of prompts with non-zero contrast (k strict between 0 and 8) ---
    n_nonzero_contrast = sum(1 for r in cells_pp if 0 < r["k_obs"] < 8)
    p_nonzero = n_nonzero_contrast / n_total
    print(f"[iter132 H4] prompts with non-zero contrast (0<k<8): {n_nonzero_contrast}/{n_total} = {p_nonzero:.4f}")

    # Summary JSON
    summary = {
        "iter": 132,
        "pillar": "P5P8-SYNTH",
        "seed": SEED,
        "n_boot": N_BOOT,
        "domain_densities": {k: v for k, v in domains},
        "domain_rank": [{"domain": k, "rate": v} for k, v in rank],
        "ratio_matrix": {k: v for k, v in ratios.items()},
        "per_method_boundary_density": method_density,
        "rho_h3_method_rank_vs_density": float(rho_h3),
        "n_prompts_with_nonzero_contrast": n_nonzero_contrast,
        "n_prompts_total": n_total,
        "rate_prompts_with_nonzero_contrast": p_nonzero,
        "h_falsifiable": {
            "H1_d4_in_P5_P7step_superdomain": bool(
                ratios["P7pp_over_P7step"]["excludes_1.0"] is False and
                ratios["P7pp_over_P5"]["excludes_1.0"] is False
            ),
            "H2_d4_highest_density": bool(rank[0][0] in ("P7pp", "D4_P7_perprompt_boundary")) or
                                      bool(densities.get("P7pp", 0) >= densities.get("P5", 0)),
            "H3_method_rank_inverts_iter127_vs_iter131": bool(rho_h3 < -0.5),
            "H4_prompts_have_nonzero_contrast": bool(p_nonzero > 0.5),
        },
    }
    # Clean ranking display
    summary["domain_rank"] = [{"domain": k, "rate": v} for k, v in rank]

    with (RES / "synth_iter132_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter132] wrote {RES / 'synth_iter132_summary.json'}")
    print(f"[iter132] DONE")


if __name__ == "__main__":
    main()
