from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "zvf-program/experiments-next/verify_rlhfbook_followup.py"
PROTOCOL_PATH = REPO_ROOT / "zvf-program/experiments-next/rlhfbook_followup_preregistration.json"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_rlhfbook_followup", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_followup_contract_is_inert_and_evidence_bounded() -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    result = verifier.verify_contract(payload, REPO_ROOT)

    assert result["status"] == "RLHFBOOK_FOLLOWUP_CONTRACT_PASS"
    assert result["gpu_authorized"] is False
    assert isinstance(result["live_checkout_matches_accepted_source"], bool)
    assert result["live_checkout_matches_accepted_source"] is (
        result["live_objective_sha256"] == result["accepted_objective_sha256"]
    )


def test_proxy_signal_cannot_complete_the_learning_claim() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    boundary = payload["decision_rules"]["claim_boundary"]

    assert "cannot support a learning" in boundary
    assert payload["evaluation_contract"]["proxy_metrics_are_not_quality_metrics"] is True
    assert payload["authorization"]["amendment_required_before_execution"] is True
