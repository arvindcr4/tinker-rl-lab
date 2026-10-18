"""Recompute flagship Table 3 (tab:runs) and gate counts from r4-2 acceptance records.

Usage: python3 tools/review_ledger/Flagship_Main_table3_recompute.py [repo_root]
Writes platform_hybrid/experiments/results/review_2026-10-18/Flagship_Main/table3_recompute.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
acc = root / "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/acceptance"
out = {}
for p in sorted(acc.glob("fpilot__*.json")):
    d = json.loads(p.read_text())
    m = d["full_record"]["manifest"]
    rs = m["gradient_receipts"]
    ev = m["evaluations"]
    rel = Counter(r["gradient_relation"] for r in rs)
    nz = [r for r in rs if r["gradient_relation"] == "nonzero"]
    nz_pass = sum(1 for r in nz if r["gradient_cosine"] >= 0.999 and r["gradient_relative_l2"] <= 0.01)
    snz = [r for r in rs if r["selected_vs_intended_relation"] == "nonzero"]
    srel = Counter(r["selected_vs_intended_relation"] for r in rs)
    sel = {
        "relations": dict(srel),
        "joint_criterion_pass": srel.get("joint_zero", 0) + sum(
            1 for r in snz
            if r["selected_vs_intended_cosine"] >= 0.999 and r["selected_vs_intended_relative_l2"] <= 0.01),
        "min_cos": min((r["selected_vs_intended_cosine"] for r in snz), default=None),
        "max_rel_l2": max((r["selected_vs_intended_relative_l2"] for r in snz), default=None),
    }
    out[p.stem] = {
        "selected_vs_intended": sel,
        "condition": d["full_record"]["condition"],
        "start_correct": ev[0]["correct"],
        "final_correct": ev[-1]["correct"],
        "heldout_n": ev[-1]["heldout_n"],
        "relations": dict(rel),
        "min_cos": min(r["gradient_cosine"] for r in nz) if nz else None,
        "max_rel_l2": max(r["gradient_relative_l2"] for r in nz) if nz else None,
        "nonzero": len(nz),
        "nonzero_pass": nz_pass,
        "joint_criterion_pass": rel.get("joint_zero", 0) + nz_pass,
        "nonzero_cos_fail": sum(1 for r in nz if r["gradient_cosine"] < 0.999),
        "nonzero_rel_l2_fail": sum(1 for r in nz if r["gradient_relative_l2"] > 0.01),
        "charged_generated_tokens": m["token_flop_ledger"]["charged_generated_tokens"],
        "active_tokens_per_step": {str(k): v for k, v in Counter(r["active_tokens"] for r in rs).items()},
        "active_rows_per_step": {str(k): v for k, v in Counter(r["active_rows"] for r in rs).items()},
        "eval_generated_tokens_sum": sum(e["generated_tokens"] for e in ev),
        "condition_factors": m["plan"]["condition_factors"],
        "runtime": m["plan"]["runtime"],
        "regime_contract": m["plan"]["regime_contract"],
    }
dst = root / "platform_hybrid/experiments/results/review_2026-10-18/Flagship_Main"
dst.mkdir(parents=True, exist_ok=True)
(dst / "table3_recompute.json").write_text(json.dumps(out, indent=1, sort_keys=True))
for k, v in out.items():
    print(k, v["start_correct"], v["final_correct"], v["relations"], v["min_cos"], v["max_rel_l2"],
          "pass", v["joint_criterion_pass"], "nzpass", v["nonzero_pass"], "cosfail", v["nonzero_cos_fail"],
          "l2fail", v["nonzero_rel_l2_fail"], v["charged_generated_tokens"], v["active_tokens_per_step"],
          v["eval_generated_tokens_sum"])
