#!/usr/bin/env python3
"""Fixed adversarial mutation score for the post-training follow-up contract."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = RUN_DIR.parents[1]
VERIFIER_RELATIVE = Path("zvf-program/experiments-next/verify_rlhfbook_followup.py")
PROTOCOL_RELATIVE = Path("zvf-program/experiments-next/rlhfbook_followup_preregistration.json")


def load_verifier(repo_root: Path):
    path = repo_root / VERIFIER_RELATIVE
    spec = importlib.util.spec_from_file_location("followup_metric_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_hypothesis_falsifier_blank(payload: dict[str, Any]) -> None:
    payload["hypotheses"][0]["falsified_if"] = ""


def _remove_foundations_stage(payload: dict[str, Any]) -> None:
    payload["stages"] = [
        stage for stage in payload["stages"] if stage["id"] != "S1_foundations_mapping"
    ]


def _blank_foundations_evidence(payload: dict[str, Any]) -> None:
    stage = next(stage for stage in payload["stages"] if stage["id"] == "S1_foundations_mapping")
    stage["evidence"] = ""


def _remove_required_telemetry(payload: dict[str, Any]) -> None:
    payload["required_telemetry"].remove("correct_completion_coverage")


def _remove_frozen_review_bundle(payload: dict[str, Any]) -> None:
    payload["scope"]["must_not_modify_or_relabel"].remove(
        "zvf-program/flagship/paper/review_bundle.zip"
    )


def _reduce_answer_checks(payload: dict[str, Any]) -> None:
    payload["evaluation_contract"]["answer_checks"] = ["registered_strict_parser"]


def mutation_cases() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    return [
        ("schema_drift", lambda p: p.__setitem__("schema_version", "unknown-v0")),
        ("authorization_status", lambda p: p.__setitem__("status", "authorized")),
        (
            "book_commit_drift",
            lambda p: p["book_binding"].__setitem__("source_commit", "0" * 40),
        ),
        (
            "course_commit_drift",
            lambda p: p["course_binding"].__setitem__("source_commit", "0" * 40),
        ),
        ("missing_course_binding", lambda p: p.__delitem__("course_binding")),
        (
            "missing_course_transfer_boundary",
            lambda p: p["course_binding"].__setitem__("use_boundary", ""),
        ),
        ("unprotected_review_bundle", _remove_frozen_review_bundle),
        (
            "missing_coverage_hypothesis",
            lambda p: p.__setitem__(
                "hypotheses", [h for h in p["hypotheses"] if h["id"] != "H5_coverage"]
            ),
        ),
        ("blank_hypothesis_falsifier", _set_hypothesis_falsifier_blank),
        ("missing_foundations_stage", _remove_foundations_stage),
        ("blank_foundations_evidence", _blank_foundations_evidence),
        ("missing_coverage_telemetry", _remove_required_telemetry),
        (
            "missing_theory_boundary",
            lambda p: p["decision_rules"].__delitem__("theory_boundary"),
        ),
        (
            "gpu_authorized",
            lambda p: p["authorization"].__setitem__("gpu", True),
        ),
        ("single_answer_checker", _reduce_answer_checks),
        (
            "unevidenced_accepted_source",
            lambda p: p["scope"]["live_checkout_observation"].__setitem__(
                "accepted_unit_source_sha256", "0" * 64
            ),
        ),
    ]


def evaluate(repo_root: Path) -> dict[str, Any]:
    verifier = load_verifier(repo_root)
    protocol_path = repo_root / PROTOCOL_RELATIVE
    base = json.loads(protocol_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for name, mutate in mutation_cases():
        candidate = copy.deepcopy(base)
        mutate(candidate)
        try:
            verifier.verify_contract(candidate, repo_root, protocol_path)
        except verifier.FollowupContractError as exc:
            results.append({"name": name, "rejected": True, "reason": str(exc)})
        except Exception as exc:  # A crash is not a controlled rejection.
            results.append(
                {
                    "name": name,
                    "rejected": False,
                    "reason": f"unexpected {type(exc).__name__}: {exc}",
                }
            )
        else:
            results.append({"name": name, "rejected": False, "reason": "accepted"})

    score = sum(result["rejected"] for result in results)
    return {"score": score, "maximum": len(results), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.repo_root.resolve())
    if args.details:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
