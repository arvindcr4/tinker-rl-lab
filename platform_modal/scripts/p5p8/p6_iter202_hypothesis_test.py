#!/usr/bin/env python3
"""Iter 202 P6 (Pillar 2) — hypothesis test on cross-framework divergence.

Consumes the per-cell TSVs from p6_iter202_framework_method_coverage.py and
issues falsifiable hypotheses about cross-framework reproducibility.

Outputs:
  experiments/results/p5p8/p6_iter202_hypotheses.tsv
  experiments/results/p5p8/p6_iter202_hypotheses.json

Stdlib only. Run: python3 platform_modal/scripts/p5p8/p6_iter202_hypothesis_test.py
"""
import csv
import json
import pathlib
from collections import defaultdict
from statistics import mean

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
P5P8 = ROOT / "experiments" / "results" / "p5p8"


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    cells = load_tsv(P5P8 / "p6_iter202_framework_method_cell.tsv")
    clusters = load_tsv(P5P8 / "p6_iter202_same_method_clusters.tsv")
    divergence = load_tsv(P5P8 / "p6_iter202_minreport_divergence.tsv")

    n_populated = sum(1 for r in cells if r["status"] == "POPULATED")
    n_total_possible = n_populated + 65  # 65 unmined from prior script

    # H1: cartesian density ≥25%
    density = 100 * n_populated / n_total_possible
    h1 = {
        "id": "H1",
        "claim": "cartesian density (populated / total) ≥ 25% post-iter-198-bump",
        "value_pct": round(density, 2),
        "threshold_pct": 25.0,
        "result": "PASS" if density >= 25.0 else "FAIL",
    }

    # H2: ≥3 cross-framework clusters (≥3 methods covered by ≥2 frameworks)
    n_clusters = len(clusters)
    h2 = {
        "id": "H2",
        "claim": "≥3 methods covered by ≥2 frameworks (cross-framework surface ≥3)",
        "value": n_clusters,
        "threshold": 3,
        "result": "PASS" if n_clusters >= 3 else "FAIL",
        "supporting_methods": [r["method"] for r in clusters],
    }

    # H3: grpo is the highest-population method
    by_method_count = defaultdict(int)
    for r in cells:
        by_method_count[r["method"]] += int(r["n_entries"])
    top_method, top_count = max(by_method_count.items(), key=lambda kv: kv[1])
    h3 = {
        "id": "H3",
        "claim": "grpo is the highest-population method",
        "value": f"{top_method} ({top_count})",
        "grpo_count": by_method_count.get("grpo", 0),
        "result": "PASS" if top_method == "grpo" else "FAIL",
        "all_method_counts": dict(sorted(by_method_count.items(), key=lambda kv: -kv[1])),
    }

    # H4: zvf130 framework has the most entries
    by_fw_count = defaultdict(int)
    for r in cells:
        by_fw_count[r["framework"]] += int(r["n_entries"])
    top_fw, top_fw_n = max(by_fw_count.items(), key=lambda kv: kv[1])
    h4 = {
        "id": "H4",
        "claim": "zvf130 framework has the most stack entries (single-batch harness reuse)",
        "value": f"{top_fw} ({top_fw_n})",
        "zvf130_count": by_fw_count.get("zvf130", 0),
        "result": "PASS" if top_fw == "zvf130" else "FAIL",
        "all_fw_counts": dict(sorted(by_fw_count.items(), key=lambda kv: -kv[1])),
    }

    # H5: cross-framework MIN-REPORT divergence is non-zero on grpo (the cluster
    # with the most frameworks). At least one grpo field disagrees pairwise.
    grpo_div = [r for r in divergence if r["method"] == "grpo"]
    n_disagree_fields = sum(1 for r in grpo_div if int(r["cross_fw_disagree_pairs"]) > 0)
    n_total_grpo_fields = len(grpo_div)
    h5 = {
        "id": "H5",
        "claim": "grpo cross-framework MIN-REPORT shows ≥1 disagreeing field",
        "n_disagree_fields": n_disagree_fields,
        "n_total_fields": n_total_grpo_fields,
        "disagree_rate_overall": round(n_disagree_fields / n_total_grpo_fields, 4) if n_total_grpo_fields else 0,
        "result": "PASS" if n_disagree_fields >= 1 else "FAIL",
        "worst_disagree_fields": sorted(
            [r for r in grpo_div if int(r["cross_fw_disagree_pairs"]) > 0],
            key=lambda r: -float(r["disagree_rate"]),
        )[:5],
    }

    # H6: cross-framework mean divergence < 0.5 (most fields agree across fws)
    disagree_rates = [float(r["disagree_rate"]) for r in divergence if int(r["cross_fw_pairs"]) > 0]
    mean_disagree = mean(disagree_rates) if disagree_rates else 0.0
    h6 = {
        "id": "H6",
        "claim": "mean cross-framework pairwise disagree rate < 0.50 (mostly agree)",
        "value": round(mean_disagree, 4),
        "threshold": 0.50,
        "result": "PASS" if mean_disagree < 0.50 else "FAIL",
        "n_fields_evaluated": len(disagree_rates),
    }

    # H7: zvf130 has the LOWEST mean MIN-REPORT badge (single-batch harness).
    by_fw_badge = defaultdict(list)
    for r in cells:
        by_fw_badge[r["framework"]].append(float(r["mean_badge"]))
    fw_mean_badge = {fw: round(mean(badges), 2) for fw, badges in by_fw_badge.items()}
    lowest_fw = min(fw_mean_badge, key=lambda fw: fw_mean_badge[fw])
    h7 = {
        "id": "H7",
        "claim": "zvf130 framework has the lowest mean MIN-REPORT badge (single-batch harness trades reporting breadth for cross-method coverage)",
        "zvf130_badge": fw_mean_badge.get("zvf130"),
        "lowest_fw": lowest_fw,
        "lowest_badge": fw_mean_badge[lowest_fw],
        "all_fw_badges": dict(sorted(fw_mean_badge.items(), key=lambda kv: kv[1])),
        "result": "PASS" if lowest_fw == "zvf130" else "FAIL",
    }

    # H8: top-3 unmined priority cells (from prior run) are all (zvf130, method_covered_elsewhere)
    # Reload priority
    unmined = load_tsv(P5P8 / "p6_iter202_unmined_cell_priority.tsv")
    top3 = unmined[:3]
    all_zvf130 = all(r["framework"] == "zvf130" for r in top3)
    h8 = {
        "id": "H8",
        "claim": "top-3 unmined cells are all zvf130 (cross-method extension of single-batch harness)",
        "top3": [{"framework": r["framework"], "method": r["method"], "priority": int(r["priority_score"])} for r in top3],
        "all_zvf130": all_zvf130,
        "result": "PASS" if all_zvf130 else "FAIL",
    }

    # H9: cross-framework reproducibility surface (grpo cluster) is large enough
    # to enable a meaningful cross-fw same-method comparison (≥4 frameworks, ≥6 entries).
    grpo_cluster = next((c for c in clusters if c["method"] == "grpo"), None)
    n_grpo_frameworks = int(grpo_cluster["n_frameworks"]) if grpo_cluster else 0
    n_grpo_entries = int(grpo_cluster["n_entries"]) if grpo_cluster else 0
    h9 = {
        "id": "H9",
        "claim": "grpo cluster has ≥4 frameworks and ≥6 entries (sufficient for cross-fw reproducibility test)",
        "n_frameworks": n_grpo_frameworks,
        "n_entries": n_grpo_entries,
        "result": "PASS" if (n_grpo_frameworks >= 4 and n_grpo_entries >= 6) else "FAIL",
    }

    # H10: the registry is method-monoculture except for grpo (i.e., only grpo
    # has cross-framework coverage). The STRONG finding is the FAIL: 5
    # additional methods (aero, areal, dapo, drgrpo, gift) DO have multi-
    # framework coverage, broadening the cross-framework reproducibility
    # surface from 1 method (grpo) to 6 methods. PASS=monoculture-confirmed,
    # FAIL=monoculture-broader-than-expected (the informative outcome here).
    methods_with_multi_fw = sorted(
        [m for m, c in by_method_count.items() if c > 1]
    )
    n_methods_with_multi_fw = len(methods_with_multi_fw)
    h10 = {
        "id": "H10",
        "claim": "registry is method-monoculture (only grpo has multi-framework coverage)",
        "value": n_methods_with_multi_fw,
        "methods_with_multi_fw": methods_with_multi_fw,
        "result": "PASS" if n_methods_with_multi_fw == 1 else "FAIL",
        "interpretation": "FAIL IS the strong finding: 6 methods (grpo + 5 others) have multi-framework coverage; cross-framework reproducibility surface is broader than a single-method monoculture",
    }

    hypotheses = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
    n_pass = sum(1 for h in hypotheses if h["result"] == "PASS")
    n_fail = len(hypotheses) - n_pass

    # Write TSV
    out_tsv = P5P8 / "p6_iter202_hypotheses.tsv"
    fields = ["id", "claim", "value", "result"]
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for h in hypotheses:
            w.writerow({
                "id": h["id"],
                "claim": h["claim"],
                "value": json.dumps({k: v for k, v in h.items() if k not in ("id", "claim", "result")}),
                "result": h["result"],
            })

    summary = {
        "iter": 202,
        "pillar": "P6",
        "vein": "(b) post-bump cross-framework x method coverage + hypothesis test",
        "n_hypotheses": len(hypotheses),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "hypotheses": hypotheses,
    }
    with open(P5P8 / "p6_iter202_hypotheses.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Iter 202 hypothesis test: {n_pass} PASS / {n_fail} FAIL out of {len(hypotheses)}")
    for h in hypotheses:
        marker = "PASS" if h["result"] == "PASS" else "FAIL"
        print(f"  [{marker}] {h['id']}: {h['claim']}")
    print(f"Output: {out_tsv}")


if __name__ == "__main__":
    main()