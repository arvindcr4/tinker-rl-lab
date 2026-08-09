#!/usr/bin/env python3
"""Fail-closed validation for the Pavlov's List domain campaign contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).with_name("pavlovs_domain_contract.json")
PRIMARY_ROLES = {"primary_eval"}
TRAIN_ROLES = {"train"}
REQUIRED_CAPABILITIES = {
    "multimodal_input",
    "tool_calling",
    "long_context",
    "computer_use",
    "code_generation",
}


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    if not isinstance(contract, dict):
        raise ValueError("contract root must be a JSON object")
    return contract


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    snapshot = contract.get("source_snapshot", {})
    domains = contract.get("domains", [])
    domain_set = set(domains)
    companies = contract.get("companies", [])
    suites = contract.get("suite_registry", {})

    if len(domains) != len(domain_set):
        errors.append("domains must be unique")
    if len(domain_set) != snapshot.get("domain_count"):
        errors.append(
            f"domain count {len(domain_set)} != snapshot {snapshot.get('domain_count')}"
        )
    names = [entry.get("name") for entry in companies]
    if len(names) != len(set(names)):
        errors.append("company names must be unique")
    if len(companies) != snapshot.get("company_count"):
        errors.append(
            f"company count {len(companies)} != snapshot {snapshot.get('company_count')}"
        )

    company_domain_counts: Counter[str] = Counter()
    for company in companies:
        name = company.get("name", "<unnamed>")
        listed = set(company.get("domains", []))
        if not listed:
            errors.append(f"{name}: no domains")
        unknown = listed - domain_set
        if unknown:
            errors.append(f"{name}: unknown domains {sorted(unknown)}")
        company_domain_counts.update(listed)

    if set(company_domain_counts) != domain_set:
        missing = sorted(domain_set - set(company_domain_counts))
        errors.append(f"domains without companies: {missing}")

    train_coverage: Counter[str] = Counter()
    eval_coverage: Counter[str] = Counter()
    stateful_primary = 0
    artifact_primary = 0
    primary_count = 0
    for suite_id, suite in suites.items():
        role = suite.get("role")
        listed = set(suite.get("domains", []))
        unknown = listed - domain_set
        if unknown:
            errors.append(f"{suite_id}: unknown domains {sorted(unknown)}")
        if role in TRAIN_ROLES:
            train_coverage.update(listed)
        if role in PRIMARY_ROLES:
            primary_count += 1
            eval_coverage.update(listed)
            stateful_primary += bool(suite.get("stateful"))
            artifact_primary += bool(suite.get("artifact_or_side_effect"))

    for domain in sorted(domain_set):
        if train_coverage[domain] == 0:
            errors.append(f"{domain}: no training suite")
        if eval_coverage[domain] == 0:
            errors.append(f"{domain}: no primary evaluation suite")

    if primary_count == 0:
        errors.append("no primary evaluation suites")
    else:
        if stateful_primary / primary_count < 0.6:
            errors.append("fewer than 60% of primary suites are stateful")
        if artifact_primary / primary_count < 0.5:
            errors.append(
                "fewer than 50% of primary suites inspect artifacts or side effects"
            )

    model_roles = Counter()
    for model in contract.get("model_candidates", []):
        role = model.get("role", "<missing>")
        model_roles[role] += 1
        caps = set(model.get("required_capabilities", []))
        missing_caps = REQUIRED_CAPABILITIES - caps
        if missing_caps:
            errors.append(
                f"model {model.get('model_id')}: missing capabilities "
                f"{sorted(missing_caps)}"
            )
    if model_roles["primary"] != 1 or model_roles["replication"] != 1:
        errors.append("exactly one primary and one replication model are required")

    gsm = suites.get("gsm8k_calibration", {})
    if gsm.get("role") != "calibration_only":
        errors.append("GSM8K must be calibration_only")
    if any(
        "gsm8k" in (suite_id + " " + str(suite.get("url", ""))).lower()
        and suite.get("role") in (PRIMARY_ROLES | TRAIN_ROLES)
        for suite_id, suite in suites.items()
    ):
        errors.append("GSM8K may not be a training or primary evaluation suite")
    sampling = contract.get("sampling_contract", {})
    if sampling.get("maximum_math_fraction", 1.0) > 0.05:
        errors.append("math may occupy at most 5% of the training mixture")
    if sampling.get("minimum_primary_domains_per_training_batch", 0) < 6:
        errors.append("every training batch must cover at least six domains")
    if sampling.get("minimum_stateful_fraction", 0.0) < 0.6:
        errors.append("at least 60% of the training mixture must be stateful")
    if sampling.get("minimum_artifact_or_side_effect_fraction", 0.0) < 0.5:
        errors.append(
            "at least 50% of the training mixture must inspect artifacts or side effects"
        )
    if sampling.get("unit") != "environment episode, not isolated prompt":
        errors.append("training unit must be an environment episode")

    budget = contract.get("budget_gate", {})
    if budget.get("paid_jobs_may_launch"):
        maximum_usd = budget.get("maximum_usd")
        if not isinstance(maximum_usd, (int, float)) or maximum_usd <= 0:
            errors.append("paid jobs require a positive explicit maximum_usd")

    if not contract.get("non_claims"):
        errors.append("non_claims must be explicit")
    return errors


def coverage_report(contract: dict[str, Any]) -> dict[str, Any]:
    suites = contract["suite_registry"]
    return {
        "companies": len(contract["companies"]),
        "domains": len(contract["domains"]),
        "training_suites": sum(s["role"] == "train" for s in suites.values()),
        "primary_eval_suites": sum(
            s["role"] == "primary_eval" for s in suites.values()
        ),
        "gsm8k_role": suites["gsm8k_calibration"]["role"],
        "paid_jobs_may_launch": contract["budget_gate"]["paid_jobs_may_launch"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    errors = validate_contract(contract)
    report = coverage_report(contract)
    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["status"])
        print(
            "COMPANIES={companies} DOMAINS={domains} TRAIN_SUITES={training_suites} "
            "PRIMARY_EVAL_SUITES={primary_eval_suites} GSM8K_ROLE={gsm8k_role} "
            "PAID_JOBS={paid_jobs_may_launch}".format(**report)
        )
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
