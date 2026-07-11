from __future__ import annotations

import json
from pathlib import Path

from platform_hybrid.registry.provenance import minreport


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "platform_hybrid/registry/provenance/campaign.provenance.json"


def test_minreport_resolves_restructured_source_without_mutating_record():
    record = json.loads(CAMPAIGN.read_text())
    assert record["provenance"]["code_file"] == "experiments/openings/campaign.py"

    report = minreport.verify(record, strict=True)
    source_checks = [check for check in report["checks"] if "provenance source" in check["msg"]]
    assert source_checks == [
        {
            "level": "PASS",
            "weight": 2,
            "msg": "provenance source exists: platform_hybrid/experiments/openings/campaign.py",
        }
    ]


def test_minreport_fails_when_provenance_source_is_missing():
    record = json.loads(CAMPAIGN.read_text())
    record["provenance"]["code_file"] = "missing/source.py"

    report = minreport.verify(record, strict=True)

    assert any(
        check["level"] == "FAIL" and "provenance source missing" in check["msg"]
        for check in report["checks"]
    )
