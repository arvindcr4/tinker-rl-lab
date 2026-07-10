#!/usr/bin/env python3
"""P6 iter-170 — coverage audit at field granularity.

Fresh vein. Closes brief vein (b) at the per-leaf level: which of the
26 MIN-REPORT leaves (7 items x leaves each) are systematically null,
which (framework, label) combinations are missing, and which leaves
are the binding MIN-REPORT-coverage ceiling.

Inputs:
  registry/entries/*.json   -- 43 records (26 stack + 17 variant_delta)

Outputs (under platform_hybrid/experiments/results/p5p8/):
  p6_iter170_per_leaf_null_rate.tsv
  p6_iter170_per_entry_coverage.tsv
  p6_iter170_framework_matrix.tsv
  p6_iter170_label_matrix.tsv
  p6_iter170_summary.json

Falsifiable hypotheses (H1-H4):
  H1 (PASS if): mean null-rate across 26 leaves is < 0.50
  H2 (PASS if): at least 18/26 leaves have null-rate < 0.50
  H3 (PASS if): every of 6 frameworks has >= 1 stack entry
  H4 (PASS if): every of 9 method labels has >= 1 stack entry
                 (9 = the labels present in registry/entries/)
"""
from __future__ import annotations
import json, glob, os
from collections import defaultdict, Counter
from statistics import mean

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENT = os.path.join(REPO, "registry", "entries")
OUT = os.path.join(REPO, "experiments", "results", "p5p8")
os.makedirs(OUT, exist_ok=True)

# MR item -> leaves (as enumerated from the registry entries on 2026-07-05)
MR_LEAVES = {
    "loss_form": ["advantage_normalization", "clip_eps_high", "clip_eps_low",
                  "importance_ratio_level", "length_normalization",
                  "reward_shaping_type", "sampling_dynamic_filter",
                  "token_aggregation", "token_mask"],
    "reference_kl": ["kl_beta", "kl_estimator", "reference_policy"],
    "sampler_backend": ["backend", "precision", "temperature", "top_p"],
    "telemetry": ["per_step_gu", "per_step_zvf", "source"],
    "group_size_schedule": ["adaptation_rule", "initial_g", "schedule"],
    "heldout_split": ["description", "disjoint_from_reward_env"],
    "decontamination": ["parser_robustness_probe", "performed"],
}
ALL_LEAVES = [(it, lf) for it, ls in MR_LEAVES.items() for lf in ls]
N_STACK_HARD = 9  # the 9 method labels present in registry/entries
N_FW_HARD = 6     # 6 framework names actually represented


def leaf_null_rate(entries):
    """null-rate per (item, leaf) over stacks with item present."""
    stacks = [e for e in entries if e["record_type"] == "stack"]
    n = len(stacks)
    out = []
    for it, leaves in MR_LEAVES.items():
        present = [e for e in stacks if it in e.get("min_report", {})]
        np_ = len(present)
        for lf in leaves:
            n_nn = sum(1 for e in present
                       if e["min_report"][it].get(lf) is not None)
            out.append({"item": it, "leaf": lf,
                        "stacks_with_item": np_,
                        "n_non_null": n_nn,
                        "null_rate": 1.0 - (n_nn / np_) if np_ else 1.0})
    return out, n


def entry_coverage(entries):
    """per-entry non-null rate across 26 leaves + 7-item presence."""
    stacks = [e for e in entries if e["record_type"] == "stack"]
    rows = []
    for e in stacks:
        leaves_non_null = 0
        items_present = 0
        for it, leaves in MR_LEAVES.items():
            sub = e.get("min_report", {}).get(it, {})
            if sub:
                items_present += 1
                for lf in leaves:
                    if sub.get(lf) is not None:
                        leaves_non_null += 1
        rows.append({
            "id": e["id"],
            "framework": e["framework"]["name"],
            "label": e.get("label_claimed", "?"),
            "items_present": items_present,
            "items_present_pct": items_present / 7,
            "leaves_non_null": leaves_non_null,
            "leaves_total": 26,
            "leaves_non_null_pct": leaves_non_null / 26,
        })
    return rows


def cross_matrix(entries, dim):
    """per-(dim-value) coverage of each MR item; reports (n entries, % covered)."""
    stacks = [e for e in entries if e["record_type"] == "stack"]
    buckets = defaultdict(list)
    for e in stacks:
        if dim == "framework":
            v = e["framework"]["name"]
        else:
            v = e.get(dim, "?")
        buckets[v].append(e)
    rows = []
    for k, v in sorted(buckets.items()):
        nn = len(v)
        item_cov = {}
        for it in MR_LEAVES:
            pres = sum(1 for e in v if it in e.get("min_report", {}))
            item_cov[it] = pres / nn if nn else 0.0
        rows.append({
            "dim": dim, "value": k, "n_entries": nn,
            **{f"cov_{it}": round(item_cov[it], 4) for it in MR_LEAVES},
            "mean_item_cov": round(mean(item_cov.values()), 4),
        })
    return rows


def h1(per_leaf, n_stacks):
    mean_null = mean(r["null_rate"] for r in per_leaf)
    return {"hypothesis": "H1: mean null-rate across 26 leaves < 0.50",
            "mean_null_rate": round(mean_null, 4),
            "pass": mean_null < 0.50,
            "n_stacks": n_stacks}


def h2(per_leaf):
    n_lt_50 = sum(1 for r in per_leaf if r["null_rate"] < 0.50)
    return {"hypothesis": "H2: at least 18/26 leaves have null-rate < 0.50",
            "n_leaves_lt_50": n_lt_50, "n_leaves_total": len(per_leaf),
            "pass": n_lt_50 >= 18}


def h3(per_entry):
    fws = {r["framework"] for r in per_entry}
    n = len(fws)
    return {"hypothesis": f"H3: every of {N_FW_HARD} frameworks has >= 1 stack entry",
            "n_frameworks": n, "frameworks": sorted(fws),
            "pass": n >= N_FW_HARD}


def h4(per_entry):
    labels = {r["label"] for r in per_entry}
    n = len(labels)
    return {"hypothesis": f"H4: every of {N_STACK_HARD} method labels has >= 1 stack entry",
            "n_labels": n, "labels": sorted(labels),
            "pass": n >= N_STACK_HARD}


def write_tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def main():
    entries = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ENT, "*.json")))]
    per_leaf, n_stacks = leaf_null_rate(entries)
    per_entry = entry_coverage(entries)
    fw_rows = cross_matrix(entries, "framework")
    lbl_rows = cross_matrix(entries, "label")

    H = [h1(per_leaf, n_stacks), h2(per_leaf), h3(per_entry), h4(per_entry)]
    n_pass = sum(1 for h in H if h["pass"])
    summary = {
        "iter": 170,
        "pillar": "P6",
        "n_entries": len(entries),
        "n_stack": sum(1 for e in entries if e["record_type"] == "stack"),
        "n_variant_delta": sum(1 for e in entries if e["record_type"] == "variant_delta"),
        "n_stacks_with_full_mr": sum(1 for r in per_entry if r["items_present"] == 7),
        "n_leaves_total": len(per_leaf),
        "n_leaves_lt_50": sum(1 for r in per_leaf if r["null_rate"] < 0.50),
        "max_null_leaf": max(per_leaf, key=lambda r: r["null_rate"])["leaf"],
        "min_null_leaf": min(per_leaf, key=lambda r: r["null_rate"])["leaf"],
        "hypotheses": H,
        "n_pass": n_pass,
        "n_total": len(H),
        "tsv_files": {
            "per_leaf_null_rate": "p6_iter170_per_leaf_null_rate.tsv",
            "per_entry_coverage": "p6_iter170_per_entry_coverage.tsv",
            "framework_matrix": "p6_iter170_framework_matrix.tsv",
            "label_matrix": "p6_iter170_label_matrix.tsv",
        },
    }

    write_tsv(os.path.join(OUT, "p6_iter170_per_leaf_null_rate.tsv"),
              per_leaf, ["item", "leaf", "stacks_with_item", "n_non_null", "null_rate"])
    write_tsv(os.path.join(OUT, "p6_iter170_per_entry_coverage.tsv"),
              per_entry, ["id", "framework", "label", "items_present",
                          "items_present_pct", "leaves_non_null",
                          "leaves_total", "leaves_non_null_pct"])
    fw_cols = ["dim", "value", "n_entries"] + [f"cov_{it}" for it in MR_LEAVES] + ["mean_item_cov"]
    write_tsv(os.path.join(OUT, "p6_iter170_framework_matrix.tsv"),
              fw_rows, fw_cols)
    write_tsv(os.path.join(OUT, "p6_iter170_label_matrix.tsv"),
              lbl_rows, fw_cols)
    with open(os.path.join(OUT, "p6_iter170_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tsv_files",)}, indent=2))


if __name__ == "__main__":
    main()
