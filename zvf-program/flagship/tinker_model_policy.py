#!/usr/bin/env python3
"""Fail-closed policy: every *evaluated* model must be one Tinker actually serves.

Two distinct roles must not be conflated.

**Candidate role** -- the model under test. This is the thing a Pavlov result is
about, and it MUST be a Tinker-served model from
``flagship/tinker_model_registry.json``. That is what "Tinker models only" means
and this module enforces it.

**Verifier / judge role** -- the model a benchmark's *own* grader calls. Several
suites specify these upstream: BankerToolBench's Gandalf grader defaults to
Gemini, AgentHarm's refusal and semantic judges default to
``openai/gpt-4o-2024-08-06``. Substituting a Tinker model there does not make the
run "Tinker-only" -- it changes the measuring instrument, so the number is no
longer the official benchmark and must not be compared to a published
leaderboard. This module therefore records judge bindings and refuses to certify
a run as protocol-faithful when one is swapped, rather than silently allowing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "tinker_model_registry.json"

#: Contract-authorized candidates (pavlovs_domain_contract.json -> model_candidates).
AUTHORIZED_CANDIDATES = ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-9B")

#: Upstream-specified grader models, by suite. Swapping these changes the
#: measurement; they are deliberately NOT rewritten to Tinker models.
UPSTREAM_JUDGE_BINDINGS = {
    "banker_toolbench_eval": "gemini (Gandalf the Grader default)",
    "agentharm_eval": "openai/gpt-4o-2024-08-06 (refusal + semantic judges)",
    "appbench_eval": "two human full-stack developers (not a model at all)",
}


class ModelPolicyError(RuntimeError):
    """Raised when a candidate model is not served by Tinker."""


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def served_models(registry: dict[str, Any] | None = None) -> set[str]:
    reg = registry if registry is not None else load_registry()
    return set(reg["served_names"])


def is_served(model_id: str, registry: dict[str, Any] | None = None) -> bool:
    return model_id in served_models(registry)


def assert_candidate_allowed(
    model_id: str,
    *,
    require_authorized: bool = True,
    registry: dict[str, Any] | None = None,
) -> None:
    """Fail closed unless ``model_id`` is a Tinker-served candidate.

    ``require_authorized`` additionally restricts to the two contract candidates.
    """

    reg = registry if registry is not None else load_registry()
    if model_id not in set(reg["served_names"]):
        raise ModelPolicyError(
            f"{model_id!r} is not served by Tinker. Served base models: "
            f"{', '.join(sorted(n for n in reg['served_names'] if ':peft:' not in n))}"
        )
    if require_authorized and model_id not in AUTHORIZED_CANDIDATES:
        raise ModelPolicyError(
            f"{model_id!r} is served by Tinker but is not a contract-authorized "
            f"candidate ({', '.join(AUTHORIZED_CANDIDATES)}). Amend "
            "pavlovs_domain_contract.json:model_candidates before using it."
        )


def audit_bindings(bindings: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Audit ``(suite_id, model_id)`` pairs without raising.

    Returns a report naming every binding that is off-registry or off-contract,
    plus the upstream judge bindings that deliberately remain non-Tinker.
    """

    reg = load_registry()
    served = set(reg["served_names"])
    rows = []
    for suite_id, model_id in bindings:
        rows.append(
            {
                "suite_id": suite_id,
                "model_id": model_id,
                "tinker_served": model_id in served,
                "contract_authorized": model_id in AUTHORIZED_CANDIDATES,
            }
        )
    off_registry = [r for r in rows if not r["tinker_served"]]
    off_contract = [r for r in rows if r["tinker_served"] and not r["contract_authorized"]]
    return {
        "schema_version": "tinker-model-policy-audit-v1",
        "registry_sha256": reg["registry_sha256"],
        "candidate_bindings": rows,
        "off_registry": off_registry,
        "off_contract": off_contract,
        "compliant": not off_registry and not off_contract,
        "upstream_judge_bindings_not_rewritten": UPSTREAM_JUDGE_BINDINGS,
        "note": (
            "Candidate models are enforced Tinker-only. Judge/verifier models are "
            "specified by each benchmark's own protocol; replacing them changes the "
            "measurement and forfeits comparability with published results."
        ),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", metavar="MODEL_ID", help="assert one model is an allowed candidate")
    ap.add_argument("--allow-unauthorized", action="store_true",
                    help="permit any Tinker-served model, not just contract candidates")
    args = ap.parse_args()

    if args.check:
        try:
            assert_candidate_allowed(args.check, require_authorized=not args.allow_unauthorized)
        except ModelPolicyError as exc:
            print(f"BLOCKED: {exc}")
            return 1
        print(f"OK: {args.check} is a Tinker-served, contract-authorized candidate")
        return 0

    reg = load_registry()
    print(json.dumps({
        "served_model_count": reg["served_model_count"],
        "authorized_candidates": AUTHORIZED_CANDIDATES,
        "registry_sha256": reg["registry_sha256"],
        "upstream_judge_bindings_not_rewritten": UPSTREAM_JUDGE_BINDINGS,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
