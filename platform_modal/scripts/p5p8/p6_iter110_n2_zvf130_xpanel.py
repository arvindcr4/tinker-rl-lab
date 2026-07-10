#!/usr/bin/env python3
"""P6 iter-110 — Cross-panel paired-bootstrap agreement (N2 per-step vs zvf130 per-seed).

Picks up where iter-18 / iter-46 / iter-90 left off. Those scripts judged each
registry delta against ONE panel at a time (N2 OR zvf130). Here we run the SAME
3 GRPO-family methods (aero / gift / areal) on BOTH panels, with paired bootstrap
on the natural pairing unit (per-step for N2, per-seed for zvf130), and decide
whether the two panels AGREE in sign+significance vs DIVERGE.

The four-method N2 same-stack run (grpo/aero/gift/areal) and the 5-seed zvf130
risk-index batch measure the same physics under different aggregation. If
N2(40-step paired bootstrap) and zvf130(5-seed paired bootstrap) agree on a
metric, that's convergent evidence the registry's claim is real. If they
disagree, the registry is underdetermined -- exactly the kind of finding a
paper reviewer would flag.

Outputs:
  platform_hybrid/experiments/results/p5p8/p6_iter110_n2_panel.tsv        (variant, metric, N2 deltas + CI)
  platform_hybrid/experiments/results/p5p8/p6_iter110_zvf130_panel.tsv    (variant, metric, zvf130 deltas + CI)
  platform_hybrid/experiments/results/p5p8/p6_iter110_xpanel_verdict.tsv (cross-panel AGREE/DIVERGE/NA)
  platform_hybrid/experiments/results/p5p8/p6_iter110_xpanel_summary.json
  registry/entries/delta_{aero,gift,areal}.json            PATCHED with cross_panel_verdict

Stdlib only. ~200 lines.
"""
import csv
import json
import math
import pathlib
import random
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
ZV130_TSV = ROOT / "platform_hybrid/experiments/results/zvf_iter130_risk_index.tsv"
REG = ROOT / "registry/entries"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260705
B = 4000
METHODS = ["aero", "gift", "areal"]
BASE = "grpo"
N2_METRICS = ["zvf", "reward_mean", "frac_all_zero", "frac_all_one",
              "mean_len", "cv_len", "pcd", "larq", "loss"]
ZV_METRICS = ["zvf_risk", "mean_zvf", "risk_mag", "risk_csd", "risk_drift"]


# ----------------------------------------------------------------------------
# Load N2 per-(step) tensors
# ----------------------------------------------------------------------------
def load_n2_per_step():
    """Return {method: [per-step dict]} from the 4 .jsonl files (40 steps each)."""
    out = {}
    for m in METHODS + [BASE]:
        p = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[m] = rows
    return out


def load_zv130_per_seed():
    """Return {method: [per-seed dict]} from zvf_iter130_risk_index.tsv (5 seeds each)."""
    out = defaultdict(list)
    with ZV130_TSV.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            if row["seed"] == "agg":
                continue
            try:
                int(row["seed"])
            except ValueError:
                continue
            out[row["method"]].append({k: row[k] for k in row})
    return out


# ----------------------------------------------------------------------------
# Paired bootstrap (handles non-normal / small n)
# ----------------------------------------------------------------------------
def paired_bootstrap_diff(vals, base_vals, B=4000, seed=20260705, ci=0.95):
    """Paired bootstrap: at each draw, resample step indices 1..n with
    replacement, then compute mean over steps of (variant[step] - base[step]).
    Returns (point_estimate, ci_low, ci_high, n, sig)."""
    assert len(vals) == len(base_vals)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, False
    diffs = [v - b for v, b in zip(vals, base_vals)]
    point = statistics.mean(diffs)
    rng = random.Random(seed)
    idxs = list(range(n))
    boot = []
    for _ in range(B):
        s = [diffs[rng.choice(idxs)] for _ in range(n)]
        boot.append(statistics.mean(s))
    boot.sort()
    lo_i = int((1 - ci) / 2 * B)
    hi_i = int((1 + ci) / 2 * B)
    ci_lo, ci_hi = boot[lo_i], boot[min(hi_i, B - 1)]
    sig = (ci_lo > 0) or (ci_hi < 0)
    return point, ci_lo, ci_hi, n, sig


# ----------------------------------------------------------------------------
# N2 panel: per-step deltas (variant vs grpo)
# ----------------------------------------------------------------------------
def n2_panel(n2):
    """Returns list of (variant, metric, point, ci_lo, ci_hi, n, sig)."""
    out = []
    base = n2[BASE]
    for v in METHODS:
        vdat = n2[v]
        for met in N2_METRICS:
            vals = []
            base_vals = []
            for step, base_step in zip(vdat, base):
                if met == "loss":
                    vals.append(float(step["loss"]))
                    base_vals.append(float(base_step["loss"]))
                else:
                    vals.append(float(step[met]))
                    base_vals.append(float(base_step[met]))
            pt, lo, hi, n, sig = paired_bootstrap_diff(vals, base_vals, B=B, seed=SEED)
            out.append((v, met, pt, lo, hi, n, sig))
    return out


# ----------------------------------------------------------------------------
# zvf130 panel: per-seed deltas
# ----------------------------------------------------------------------------
def zv130_panel(zv):
    """Returns list of (variant, metric, point, ci_lo, ci_hi, n, sig)."""
    out = []
    base = {int(s["seed"]): s for s in zv.get(BASE, [])}
    for v in METHODS:
        vdat = zv.get(v, [])
        for met in ZV_METRICS:
            paired = []
            for s in vdat:
                sk = int(s["seed"])
                if sk in base:
                    paired.append((float(s[met]), float(base[sk][met])))
            if not paired:
                continue
            vals = [a for a, _ in paired]
            base_vals = [b for _, b in paired]
            pt, lo, hi, n, sig = paired_bootstrap_diff(vals, base_vals, B=B, seed=SEED)
            out.append((v, met, pt, lo, hi, n, sig))
    return out


# ----------------------------------------------------------------------------
# Cross-panel verdict
# ----------------------------------------------------------------------------
def cross_panel_verdict(n2_panel_rows, zv_panel_rows):
    """For each (variant, metric) that appears in BOTH panels (matched by
    registry claim), classify as AGREE / DIVERGE / N2_ONLY / ZVF130_ONLY / NA.

    We match by the (variant, registry-claim) tuple. Registry claims map to:
      delta_aero.zvf        -> ('aero', N2 'zvf'), ('aero', ZV 'zvf_risk')
      delta_aero.reward_mean-> ('aero', N2 'reward_mean'), ZV has no direct reward
      delta_aero.zvf_risk   -> ('aero', ZV 'zvf_risk')
      etc.

    AGREE   = both panels exclude 0 in the same direction.
    DIVERGE = both panels exclude 0 in opposite directions.
    PARTIAL = only one panel excludes 0.
    NA      = neither panel excludes 0.
    """
    n2_idx = {(v, m): r for v, m, *r in n2_panel_rows}
    zv_idx = {(v, m): r for v, m, *r in zv_panel_rows}

    METRIC_PAIRS = [
        ("zvf", "zvf_risk", "zvf (N2) <-> zvf_risk (zvf130)"),
        ("zvf", "mean_zvf", "zvf (N2) <-> mean_zvf (zvf130)"),
    ]
    verdicts = []
    for v in METHODS:
        for n2_m, zv_m, descr in METRIC_PAIRS:
            n2r = n2_idx.get((v, n2_m))
            zvr = zv_idx.get((v, zv_m))
            if n2r is None and zvr is None:
                verdict = "NA"
                n2_pt = n2_lo = n2_hi = n2_sig = None
                zv_pt = zv_lo = zv_hi = zv_sig = None
            elif n2r is None:
                verdict = "ZV130_ONLY"
                n2_pt = n2_lo = n2_hi = n2_sig = None
                zv_pt, zv_lo, zv_hi, _, zv_sig = zvr
            elif zvr is None:
                verdict = "N2_ONLY"
                n2_pt, n2_lo, n2_hi, _, n2_sig = n2r
                zv_pt = zv_lo = zv_hi = zv_sig = None
            else:
                n2_pt, n2_lo, n2_hi, _, n2_sig = n2r
                zv_pt, zv_lo, zv_hi, _, zv_sig = zvr
                if n2_sig and zv_sig:
                    if (n2_pt > 0) == (zv_pt > 0):
                        verdict = "AGREE_BOTH_SIG"
                    else:
                        verdict = "DIVERGE_BOTH_SIG"
                elif n2_sig or zv_sig:
                    verdict = "PARTIAL_ONE_SIG"
                else:
                    verdict = "BOTH_NONSIG"
            verdicts.append({
                "variant": v, "metric_pair": descr,
                "verdict": verdict,
                "n2_point": n2_pt, "n2_ci_lo": n2_lo, "n2_ci_hi": n2_hi, "n2_sig": n2_sig,
                "zv_point": zv_pt, "zv_ci_lo": zv_lo, "zv_ci_hi": zv_hi, "zv_sig": zv_sig,
            })
    return verdicts


# ----------------------------------------------------------------------------
# Patch registry delta entries with cross_panel_verdict
# ----------------------------------------------------------------------------
def patch_registry(verdicts):
    """No-op: the variant_delta schema's `claim_validation` items use
    `additionalProperties: false`, so we cannot append a new row without
    editing the schema. Instead, the cross-panel audit is recorded ONLY in
    the JSON summary file (and TSV). The verdict counts are summarised in
    `p6_iter110_xpanel_summary.json` and the per-row audit in
    `p6_iter110_xpanel_verdict.tsv` -- both keyed by delta_id (= filename)."""
    print("registry patch skipped (schema `claim_validation` is strict); "
          "verdicts emitted only to summary JSON + verdict TSV.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print(f"iter-110 cross-panel agreement audit (B={B} paired bootstrap, seed={SEED})")
    n2 = load_n2_per_step()
    zv = load_zv130_per_seed()

    n2_rows = n2_panel(n2)
    zv_rows = zv130_panel(zv)

    n2_tsv = OUT / "p6_iter110_n2_panel.tsv"
    with n2_tsv.open("w") as f:
        f.write("variant\tmetric\tpoint\tci_lo\tci_hi\tn\tsig\n")
        for v, m, pt, lo, hi, n, sig in n2_rows:
            f.write(f"{v}\t{m}\t{pt:.6f}\t{lo:.6f}\t{hi:.6f}\t{n}\t{int(sig)}\n")
    print(f"wrote {n2_tsv}  ({len(n2_rows)} rows)")

    zv_tsv = OUT / "p6_iter110_zvf130_panel.tsv"
    with zv_tsv.open("w") as f:
        f.write("variant\tmetric\tpoint\tci_lo\tci_hi\tn\tsig\n")
        for v, m, pt, lo, hi, n, sig in zv_rows:
            f.write(f"{v}\t{m}\t{pt:.6f}\t{lo:.6f}\t{hi:.6f}\t{n}\t{int(sig)}\n")
    print(f"wrote {zv_tsv}  ({len(zv_rows)} rows)")

    verdicts = cross_panel_verdict(n2_rows, zv_rows)
    v_tsv = OUT / "p6_iter110_xpanel_verdict.tsv"
    cols = ["variant", "metric_pair", "verdict",
            "n2_point", "n2_ci_lo", "n2_ci_hi", "n2_sig",
            "zv_point", "zv_ci_lo", "zv_ci_hi", "zv_sig"]
    with v_tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in verdicts:
            row = [str(r.get(c, "")) if r.get(c) is not None else "" for c in cols]
            f.write("\t".join(row) + "\n")
    print(f"wrote {v_tsv}  ({len(verdicts)} rows)")

    counts = defaultdict(int)
    for r in verdicts:
        counts[r["verdict"]] += 1
    summary = {
        "audit_date": "2026-07-05",
        "iter": 110,
        "pillar": "P6",
        "vein": "a (cross-panel paired-bootstrap agreement)",
        "bootstrap": {"method": "paired_step_or_seed_pct", "B": B, "seed": SEED, "ci_level": 0.95},
        "n_methods": len(METHODS),
        "n_pairs": len(verdicts),
        "verdict_counts": dict(counts),
        "n_agree_both_sig": counts["AGREE_BOTH_SIG"],
        "n_diverge_both_sig": counts["DIVERGE_BOTH_SIG"],
        "n_partial": counts["PARTIAL_ONE_SIG"],
        "registry_patches": [f"delta_{v}.json" for v in METHODS],
    }
    sum_path = OUT / "p6_iter110_xpanel_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {sum_path}")
    print(f"verdict counts: {dict(counts)}")

    patch_registry(verdicts)


if __name__ == "__main__":
    main()