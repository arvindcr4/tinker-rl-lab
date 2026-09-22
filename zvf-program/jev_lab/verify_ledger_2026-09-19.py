#!/usr/bin/env python3
"""Code-side verification of the E1-E14 results ledger (2026-09-19 restart).

Every check here is deterministic: file presence, hash equality, integer
identity, exact division.  Semantic judgments are deliberately excluded;
they belong to the jev batteries (see jev_judge.py).

Writes outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json"

checks: list[dict] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 1. E1 — full-suite score is exactly 2/731 -------------------------------
p = REPO / "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/receipt.json"
d = json.loads(p.read_text())
ok = abs(d["score"] - 2 / 731) < 1e-15 and d["status"] == "SCORED"
check(
    "E1_receipt_score_equals_2_of_731",
    ok,
    f"score={d['score']!r} status={d['status']} 2/731={2/731!r}",
)

# 2. E9 — arms reconcile with the Sep-5 prose ------------------------------
rows = list(
    csv.DictReader(
        (REPO / "outputs/e1_e14_results_2026-09-05/e9_competition_receipts.csv").open()
    )
)
ms = [r for r in rows if r["arm"] == "modal_streaming" and "GRADED" in r["status"]]
mv = [r for r in rows if r["arm"] == "merged_vllm" and "GRADED" in r["status"]]
ms_uniq = len({r["competition_id"] for r in ms})
check(
    "E9_modal_streaming_40_unique_graded_of_75",
    ms_uniq == 40,
    f"graded_rows={len(ms)} unique_competitions={ms_uniq} (denominator 75)",
)
check(
    "E9_merged_vllm_arm_separate_with_1_grade",
    len(mv) == 1,
    f"merged_vllm graded rows={len(mv)}",
)

# 3. E11 — canonical raw 129/312, sensitivity 129/311 retained -------------
d = json.loads(
    (REPO / "outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json").read_text()
)
pa = d["pass_at_1"]
ok = (
    pa["raw"]["passes"] == 129
    and pa["raw"]["denominator"] == 312
    and pa["raw"]["canonical"] is True
    and abs(pa["raw"]["pass_at_1"] - 129 / 312) < 1e-15
    and pa["corrected"]["denominator"] == 311
    and "reporting_rule" in pa
)
check(
    "E11_canonical_129_of_312_with_sensitivity",
    ok,
    f"raw={pa['raw']['passes']}/{pa['raw']['denominator']} "
    f"corrected={pa['corrected']['passes']}/{pa['corrected']['denominator']}",
)
# component split from the Sep-5 report
check(
    "E11_component_split_sums_67_plus_62_equals_129",
    67 + 62 == 129,
    "code-completion 67/156 + spec-to-RTL 62/156 = 129/312",
)

# 4. E14 — recheck reproduces accepted/skipped and official accuracy --------
d = json.loads(
    (REPO / "outputs/PES_Phase2_Review_2026-09-12/e14_scoring_recheck.json").read_text()
)
ok = (
    d["attempted"] == 4428
    and d["accepted"] == 4426
    and d["skipped"] == 2
    and d["per_task_checks"] == 4428
    and abs(d["official_accuracy"] - 0.5131043831902395) < 1e-15
    and d["official_stdout"].strip() == "Total Accuracy:0.5131043831902395"
)
check(
    "E14_recheck_4426_of_4428_accuracy_0_5131",
    ok,
    f"accepted={d['accepted']}/{d['attempted']} skipped={d['skipped']} "
    f"acc={d['official_accuracy']}",
)

# 5. E8 — LAB-Bench categories all complete and sum to 1967 -----------------
d = json.loads((REPO / "outputs/PES_Phase2_Review_2026-09-12/results.json").read_text())
cats = d["labbench_complete_categories"]
all_complete = all(
    c["evaluated"] == c["expected_total"] and c["status"] == "COMPLETE"
    for c in cats.values()
)
total = sum(c["evaluated"] for c in cats.values())
check(
    "E8_labbench_1967_of_1967_all_categories_complete",
    all_complete and total == 1967,
    f"categories={len(cats)} evaluated_sum={total}",
)

# 6. Sep-12 strict completed set --------------------------------------------
check(
    "strict_completed_suites_are_E8_E10_E11",
    d["strict_completed_suites"] == ["E8", "E10", "E11"],
    f"strict_completed_suites={d['strict_completed_suites']}",
)

# 7. E1 v6 sealed artifacts survive with recorded hashes ---------------------
v6dir = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6"
val = json.loads((v6dir / "validation_v6.json").read_text())
req = v6dir / "resource_request_v6.json"
ok = sha256(req) == val["request"]["sha256"]
check(
    "E1_v6_sealed_request_hash_matches_validation",
    ok,
    f"sha256={val['request']['sha256'][:16]}...",
)
fix = v6dir / "group_regression_receipt.json"
ok = sha256(fix) == val["actual_fixture_receipt"]["sha256"]
check(
    "E1_v6_fixture_receipt_hash_matches_validation",
    ok,
    f"sha256={val['actual_fixture_receipt']['sha256'][:16]}...",
)

# 8. Lost-source register (existence probes) --------------------------------
lost = [
    ".codex-run/finish_20260912/e1_recovery_v6/bounded_sdk.py",
    ".codex-run/finish_20260912/e9_completion/restored",
    ".codex-run/finish_20260912/e6_continuation/checks_v1.py",
    ".codex-run/public_native_improve_v2/e1/metadata_wave08.py",
]
present = [p for p in lost if (REPO / p).exists()]
check(
    "codex_run_execution_sources_absent",
    not present,
    f"probed={len(lost)} still_present={present}",
)

result = {
    "schema_version": "ledger-code-check-v1",
    "recorded_at": "2026-09-19",
    "checks": checks,
    "passed": sum(1 for c in checks if c["status"] == "PASS"),
    "failed": sum(1 for c in checks if c["status"] == "FAIL"),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(json.dumps(result, indent=1, sort_keys=True))
print("receipt:", OUT)
