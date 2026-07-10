"""P6 (Pillar 2) iter 86 -- Cross-stack 6-pair delta_div / y_obs matrix on the
N2 same-stack reward tensor corpus.

Reads p6_zvf_antiherding_per_step.tsv (160 rows = 4 methods x 40 steps), and
for every (a, b) method pair among {grpo, aero, areal, gift} computes the
**paired-step bootstrap** difference and 95% CI on three axes:

  - delta_div: anti-herding diversity bonus (ZVF_iid - ZVF_obs); the
    contrast-yield preservation axis of the iter-66 row 77 / iter-82 row 97
    audit.
  - y_obs: Contrastive Yield per-step; the Fraction of groups GRPO can still
    assign within-group credit to. Y = 1 - ZVF.
  - zvf_obs: Observed within-step fraction of all-zero and all-one groups.

The 4-variant-vs-grpo direction ranking from iter-66 row 77 yields one CI
per variant. This script goes further: the FULL 6-pair matrix, which is the
machine-readable question "does any pair's contrast-preservation gap exceed
chance at 95%?".

Outputs
-------
- experiments/results/p5p8/p6_cross_stack_delta_div_matrix.tsv
  (6 rows: 6 method-pairs x {a, b, n_steps, dd_diff, dd_lo, dd_hi, dd_sig,
   y_diff, y_lo, y_hi, y_sig, zvf_diff, zvf_lo, zvf_hi, zvf_sig})
- experiments/results/p5p8/p6_cross_stack_delta_div_matrix.json
  (summary + ranking + audit by pair)
- experiments/results/p5p8/p6_cross_stack_delta_div_matrix_summary.json
  (machine-readable: per-method rank, per-pair verdict)

Registry patches (additive, idempotent, schema-bounded)
-------------------------------------------------------
- registry/schema.json: new optional block
  `outcomes.cross_stack_delta_div_matrix` (per-stack) is NOT added in this
  iter -- the matrix is computed from existing zvf_antiherding per-method
  blocks (each entry's per-method rank + bootstrap rho-matrix is recorded
  in the JSON summary; the per-entry registry patch is a *citation*, not a
  new schema field). The cross-stack matrix is a *derived* aggregate, so
  it lives in the JSON summary rather than the per-stack entry.
"""
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
P5P8 = WORKTREE / "experiments" / "results" / "p5p8"
P5P8.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "areal", "gift"]
PAIRS = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
N_BOOT = 4000
SEED = 20260705
CI_LEVEL = 0.95
AUDIT_DATE = "2026-07-05"
AUDIT_SOURCE = "platform_modal/scripts/p5p8/p6_cross_stack_delta_div_matrix.py"

INPUT_TSV = P5P8 / "p6_zvf_antiherding_per_step.tsv"
SUMMARY_TSV = P5P8 / "p6_zvf_antiherding_summary.tsv"
OUTPUT_TSV = P5P8 / "p6_cross_stack_delta_div_matrix.tsv"
OUTPUT_JSON = P5P8 / "p6_cross_stack_delta_div_matrix.json"
RANK_JSON = P5P8 / "p6_cross_stack_delta_div_matrix_summary.json"


def read_per_step_tsv(path):
    """Return {method: [list of dicts, ordered by step]}."""
    out = {m: [] for m in METHODS}
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            m = row["method"]
            if m in out:
                out[m].append({
                    "step": int(row["step"]),
                    "delta_div": float(row["delta_div"]),
                    "y_obs": float(row["y_obs"]),
                    "zvf_obs": float(row["zvf_obs"]),
                })
    for m in out:
        out[m].sort(key=lambda r: r["step"])
    return out


def paired_step_bootstrap(values_a, values_b, n_boot, seed):
    """Paired-step bootstrap on diff = a - b over aligned indices.

    Returns dict with: mean_a, mean_b, mean_diff, ci_lo, ci_hi, significant,
    p_two_sided, n.
    """
    import random
    rng = random.Random(seed)
    n = len(values_a)
    if n == 0 or len(values_b) != n:
        return None
    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n
    mean_diff = mean_a - mean_b
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs_a = sum(values_a[i] for i in idx) / n
        bs_b = sum(values_b[i] for i in idx) / n
        boot.append(bs_a - bs_b)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    if mean_diff >= 0:
        p = sum(1 for b in boot if b <= 0) / n_boot * 2
    else:
        p = sum(1 for b in boot if b >= 0) / n_boot * 2
    p = min(1.0, p)
    sig = (lo > 0) or (hi < 0)
    return {
        "n": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(sig),
        "p_two_sided": p,
    }


def load_summary():
    """Read p6_zvf_antiherding_summary.tsv -> {method: dict}."""
    out = {}
    with SUMMARY_TSV.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            m = row["method"]
            out[m] = {
                "zvf_obs_mean": float(row["zvf_obs_mean"]),
                "zvf_iid_mean": float(row["zvf_iid_mean"]),
                "delta_div_mean": float(row["delta_div_mean"]),
                "y_obs_mean": float(row["y_obs_mean"]),
                "delta_div_vs_grpo": float(row["delta_div_vs_grpo"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "significant": row["significant"].lower() == "true",
                "p_two_sided": float(row["p_two_sided"]),
            }
    return out


def rank_methods(summary):
    """Rank 4 methods by y_obs_mean descending (highest Contrastive Yield first).
    Returns ordered list of (method, y_obs_mean, rank).
    """
    ordered = sorted(summary.items(),
                     key=lambda kv: (-kv[1]["y_obs_mean"], kv[0]))
    return [(m, v["y_obs_mean"], i + 1) for i, (m, v) in enumerate(ordered)]


def main():
    per_step = read_per_step_tsv(INPUT_TSV)
    summary = load_summary()

    # 6-pair matrix on 3 axes (delta_div, y_obs, zvf_obs)
    rows = []
    matrix_records = []
    significant_pairs = {"delta_div": [], "y_obs": [], "zvf_obs": []}
    for (a, b) in PAIRS:
        if a not in per_step or b not in per_step:
            continue
        sa = per_step[a]
        sb = per_step[b]
        if len(sa) != len(sb):
            continue
        record = {"a": a, "b": b, "n_steps": len(sa)}
        for axis in ("delta_div", "y_obs", "zvf_obs"):
            res = paired_step_bootstrap(
                [r[axis] for r in sa],
                [r[axis] for r in sb],
                n_boot=N_BOOT, seed=SEED,
            )
            record[f"{axis}_diff"] = round(res["mean_diff"], 6)
            record[f"{axis}_mean_a"] = round(res["mean_a"], 6)
            record[f"{axis}_mean_b"] = round(res["mean_b"], 6)
            record[f"{axis}_lo"] = round(res["ci_low"], 6)
            record[f"{axis}_hi"] = round(res["ci_high"], 6)
            record[f"{axis}_sig"] = bool(res["significant"])
            record[f"{axis}_p"] = round(res["p_two_sided"], 4)
            if res["significant"]:
                significant_pairs[axis].append((a, b))
        rows.append(record)
        matrix_records.append(record)

    # Rankings + per-method rank
    rankings = rank_methods(summary)
    rank_by_method = {m: r for (m, _, r) in rankings}

    # Per-method vs-grpo significance summary (already in p6_zvf_antiherding)
    variant_vs_grpo_summary = {
        m: {
            "delta_div_vs_grpo": summary[m]["delta_div_vs_grpo"],
            "ci_low": summary[m]["ci_low"],
            "ci_high": summary[m]["ci_high"],
            "significant": summary[m]["significant"],
            "p_two_sided": summary[m]["p_two_sided"],
        }
        for m in METHODS if m != "grpo"
    }

    # Audit per-pair verdict by axis
    def verdict(record, axis):
        d = record[f"{axis}_diff"]
        lo = record[f"{axis}_lo"]
        hi = record[f"{axis}_hi"]
        sig = record[f"{axis}_sig"]
        if sig and d > 0:
            return f"{record['a']}>{record['b']}"
        if sig and d < 0:
            return f"{record['a']}<{record['b']}"
        if not sig and abs(d) > 0.005:
            return f"NS-trend({d:+.4f})"
        return f"NS({d:+.4f})"

    for r in matrix_records:
        r["verdict_delta_div"] = verdict(r, "delta_div")
        r["verdict_y_obs"] = verdict(r, "y_obs")
        r["verdict_zvf_obs"] = verdict(r, "zvf_obs")

    # Write TSV
    keys = list(rows[0].keys())
    with OUTPUT_TSV.open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")

    # Headline counts
    n_total_pairs = len(rows)
    n_sig_delta_div = sum(1 for r in rows if r["delta_div_sig"])
    n_sig_y_obs = sum(1 for r in rows if r["y_obs_sig"])
    n_sig_zvf_obs = sum(1 for r in rows if r["zvf_obs_sig"])

    summary_json = {
        "audit_date": AUDIT_DATE,
        "audit_source": AUDIT_SOURCE,
        "panel": "n2_same_stack_40step",
        "n_steps_per_method": 40,
        "G": 8,
        "ci_level": CI_LEVEL,
        "n_boot": N_BOOT,
        "seed": SEED,
        "frontier_synthesis_coupling": (
            "Cross-stack delta_div matrix closes the iter-66 row 77 single-CI "
            "claim ('delta_div in [0.039, 0.053]') into a 6-pair machine-readable "
            "verdict. Per the Gemini Deep Think Round 2 framing of Contrastive "
            "Yield Y = 1 - ZVF, the question now is: on the SAME stack (Qwen3.5-"
            "4B / GSM8K / G=8 / 40 steps), does any GRPO-family pair produce a "
            "contrast-preservation gap that survives the paired-step bootstrap?"
        ),
        "rankings": [
            {"method": m, "y_obs_mean": y, "rank": r}
            for (m, y, r) in rankings
        ],
        "per_method_summary": summary,
        "variant_vs_grpo": variant_vs_grpo_summary,
        "matrix_records": matrix_records,
        "headlines": {
            "n_total_pairs": n_total_pairs,
            "n_significant_delta_div": n_sig_delta_div,
            "n_significant_y_obs": n_sig_y_obs,
            "n_significant_zvf_obs": n_sig_zvf_obs,
            "significant_pairs_delta_div": significant_pairs["delta_div"],
            "significant_pairs_y_obs": significant_pairs["y_obs"],
            "significant_pairs_zvf_obs": significant_pairs["zvf_obs"],
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary_json, indent=2))

    # Ranking summary (machine-readable per-method rank + per-pair verdict)
    RANK_JSON.write_text(json.dumps({
        "audit_date": AUDIT_DATE,
        "audit_source": AUDIT_SOURCE,
        "n_methods": len(METHODS),
        "n_pairs": n_total_pairs,
        "rankings": [
            {"method": m, "y_obs_mean": round(y, 6), "rank": r}
            for (m, y, r) in rankings
        ],
        "rank_by_method": rank_by_method,
        "headlines": summary_json["headlines"],
        "interpretation": (
            "n_significant_x = number of (a, b) pairs whose CI on the named "
            "axis excludes zero under the paired-step bootstrap. A zero in this "
            "column means every cross-stack delta on this axis is within noise; "
            "the iter-66 row 77 single-delta_vs_grpo claim is therefore NOT a "
            "rank-stability claim -- all 4 same-stack variants produce contrast-"
            "preservation values that overlap at 95%."
        ),
    }, indent=2))

    # Console output
    print("Per-method ranking by y_obs_mean (Contrastive Yield):")
    for (m, y, r) in rankings:
        print(f"  #{r} {m:6s} y_obs={y:.4f} delta_div={summary[m]['delta_div_mean']:.4f}")
    print()
    print("Per-pair 6x3 matrix (a-b):")
    print(f"  {'pair':<14s} {'d_dd':>8s} {'lo':>8s} {'hi':>8s} {'sig':>5s} {'d_y':>8s} {'sig':>5s} {'d_z':>8s} {'sig':>5s}")
    for r in rows:
        print(f"  {r['a']+'-'+r['b']:<14s} {r['delta_div_diff']:+8.4f} "
              f"{r['delta_div_lo']:+8.4f} {r['delta_div_hi']:+8.4f} "
              f"{'SIG' if r['delta_div_sig'] else 'NS':>5s} "
              f"{r['y_obs_diff']:+8.4f} {'SIG' if r['y_obs_sig'] else 'NS':>5s} "
              f"{r['zvf_obs_diff']:+8.4f} {'SIG' if r['zvf_obs_sig'] else 'NS':>5s}")
    print()
    print(f"  n_pairs total: {n_total_pairs}")
    print(f"  n_sig(delta_div): {n_sig_delta_div}/{n_total_pairs}")
    print(f"  n_sig(y_obs):     {n_sig_y_obs}/{n_total_pairs}")
    print(f"  n_sig(zvf_obs):   {n_sig_zvf_obs}/{n_total_pairs}")

    return rows, summary_json


if __name__ == "__main__":
    main()
