#!/usr/bin/env python3
"""Fail-closed, deterministic accounting for the Pavlov Tinker budget.

The budget JSON is the authority for prices and the $18/$16.50/$1.50
boundaries.  This module deliberately keeps three kinds of money separate:

* ``known_usd`` is money reported as live billed by a receipt;
* ``base_eval_estimated_usd`` is the estimate retained for audit from a
  base/primary-eval evaluation receipt (even after reconciliation); its
  unbilled ``base_eval_unreconciled_usd`` portion is charged conservatively;
  and
* ``rejected_run_allowance_usd`` is the conservative allowance held for a
  rejected/failed launch.

Anything that is not explicitly billed is included in the conservative
pending total, but it is never relabelled as actual billing.  Token arithmetic
uses :class:`decimal.Decimal` throughout and is converted to JSON numbers only
at the final display boundary.

The command line interface is intentionally useful for audits as well as
launch preflights.  It accepts the budget path, repeated receipt paths, an
optional projection JSON file, and direct projected token/step inputs.  A
portfolio view is loaded from ``pavlovs_domain_contract.json`` when it is next
to the budget; this makes the 12 training suites (T1-T12), 14 ``primary_eval``
suites (E1-E14), and 16 required domains visible without changing that
contract.  A primary-eval suite is not called held-out evidence until its
immutable split, license, task, container, and decontamination receipts are
present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_BUDGET_PATH = HERE / "pavlov_tinker_budget.json"
DEFAULT_PORTFOLIO_PATH = HERE / "pavlovs_domain_contract.json"

HARD_CAP = Decimal("18.00")
OPERATIONAL_CAP = Decimal("16.50")
SAFETY_RESERVE = Decimal("1.50")
MILLION = Decimal("1000000")
DISPLAY_QUANTUM = Decimal("0.000001")

REQUIRED_PRICES = ("prefill", "cached_prefill", "sample", "train")
TRAINING_ROLE = "train"
PRIMARY_EVAL_ROLE = "primary_eval"
TERMINAL_STATUSES = {"completed", "complete", "succeeded", "success", "done"}
PENDING_STATUSES = {"pending", "running", "active", "queued", "launched"}
REJECTED_STATUSES = {
    "rejected",
    "reject",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "aborted",
}
HELDOUT_RECEIPT_FIELDS = {
    "split": (
        "split_receipt",
        "split",
        "immutable_split_receipt",
    ),
    "license": (
        "license_receipt",
        "license",
        "immutable_license_receipt",
    ),
    "task": (
        "task_receipt",
        "task",
        "task_id_receipt",
        "immutable_task_receipt",
    ),
    "container": (
        "container_receipt",
        "container",
        "environment_receipt",
        "immutable_container_receipt",
    ),
    "decontamination": (
        "decontamination_receipt",
        "decontamination",
        "contamination_receipt",
        "immutable_decontamination_receipt",
    ),
}
HELDOUT_CLAIM_KEYS = (
    "heldout_claim_requested",
    "claim_heldout",
    "request_heldout_claim",
    "held_out_claim",
)


class BudgetError(ValueError):
    """A malformed or unsafe budget/receipt input."""


class LaunchRejected(BudgetError):
    """Raised by :func:`authorize_arm` when the launch guard rejects an arm."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def _json_load(path: Path, label: str) -> Any:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise BudgetError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BudgetError(f"{label} is malformed: {path}: {exc}") from exc


def _decimal(value: Any, field: str, *, allow_zero: bool = True) -> Decimal:
    """Parse a finite, non-negative decimal without going through ``float``."""

    if isinstance(value, bool) or value is None:
        raise BudgetError(f"{field} must be a finite non-negative number")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BudgetError(f"{field} must be a finite non-negative number") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise BudgetError(f"{field} must be a finite non-negative number")
    return parsed


def _integer(value: Any, field: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool):
        raise BudgetError(f"{field} must be a non-negative integer")
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise BudgetError(f"{field} must be a non-negative integer")
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed_decimal = Decimal(value)
        except InvalidOperation as exc:
            raise BudgetError(f"{field} must be a non-negative integer") from exc
        if not parsed_decimal.is_finite() or parsed_decimal != parsed_decimal.to_integral_value():
            raise BudgetError(f"{field} must be a non-negative integer")
        parsed = int(parsed_decimal)
    else:
        raise BudgetError(f"{field} must be a non-negative integer")
    if parsed < 0 or (not allow_zero and parsed == 0):
        raise BudgetError(f"{field} must be a non-negative integer")
    return parsed


def _normalise_status(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("status", receipt.get("run_status", receipt.get("state")))
    if value is None:
        # The existing eval receipt has no status, but has an explicit
        # estimated_cost_usd and token accounting.  Treat that as a completed
        # *estimate*, never as a live bill.
        if any(key in receipt for key in ("estimated_cost_usd", "base_eval_estimated_usd")):
            return "completed"
        if any(
            key in receipt
            for key in (
                "billed_cost_usd",
                "actual_billed_usd",
                "actual_cost_usd",
                "provider_billed_cost_usd",
                "live_billed_cost_usd",
            )
        ):
            return "completed"
        raise BudgetError("receipt status is missing")
    if not isinstance(value, str) or not value.strip():
        raise BudgetError("receipt status must be a non-empty string")
    return value.strip().lower()


def _receipt_model(receipt: Mapping[str, Any]) -> str | None:
    pricing = receipt.get("pricing")
    nested_model = (
        (pricing.get("model") or pricing.get("model_id"))
        if isinstance(pricing, Mapping)
        else None
    )
    for key in (
        "pricing_model",
        "price_model",
        "billing_model",
        "model",
        "tokenizer_model",
        "model_id",
    ):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(nested_model, str) and nested_model.strip():
        return nested_model.strip()
    return None


def _budget_model(budget: Mapping[str, Any]) -> str:
    model = budget.get("model")
    if not isinstance(model, str) or not model.strip():
        raise BudgetError("budget model is missing")
    return model.strip()


def validate_budget(budget: Mapping[str, Any], *, enforce_pavlov_caps: bool = True) -> list[str]:
    """Return validation errors for a loaded budget contract.

    The default strict mode protects this particular authorization.  The
    optional relaxed mode is useful to callers that want to inspect a copied
    contract in tests, while the CLI always uses strict mode.
    """

    errors: list[str] = []
    if not isinstance(budget, Mapping):
        return ["budget root must be a JSON object"]
    try:
        maximum = _decimal(budget.get("maximum_usd"), "maximum_usd", allow_zero=False)
        operational = _decimal(
            budget.get("operational_cap_usd"), "operational_cap_usd", allow_zero=False
        )
        reserve = _decimal(
            budget.get("safety_reserve_usd"), "safety_reserve_usd", allow_zero=False
        )
    except BudgetError as exc:
        errors.append(str(exc))
    else:
        if operational > maximum:
            errors.append("operational_cap_usd cannot exceed maximum_usd")
        if reserve != maximum - operational:
            errors.append("safety_reserve_usd must equal maximum_usd - operational_cap_usd")
        if enforce_pavlov_caps:
            if maximum != HARD_CAP:
                errors.append("maximum_usd must preserve the $18.00 hard cap")
            if operational != OPERATIONAL_CAP:
                errors.append("operational_cap_usd must preserve the $16.50 operational cap")
            if reserve != SAFETY_RESERVE:
                errors.append("safety_reserve_usd must preserve the $1.50 safety reserve")

    try:
        _budget_model(budget)
    except BudgetError as exc:
        errors.append(str(exc))

    prices = budget.get("usd_per_million_tokens")
    if not isinstance(prices, Mapping):
        errors.append("usd_per_million_tokens must be an object")
    else:
        for name in REQUIRED_PRICES:
            if name not in prices:
                errors.append(f"unknown/incomplete price model: missing {name}")
                continue
            try:
                _decimal(prices[name], f"usd_per_million_tokens.{name}")
            except BudgetError as exc:
                errors.append(str(exc))
    if budget.get("provider") != "Tinker":
        errors.append("budget provider must be Tinker")
    return errors


def load_budget(path: Path = DEFAULT_BUDGET_PATH) -> dict[str, Any]:
    budget = _json_load(Path(path), "budget contract")
    if not isinstance(budget, dict):
        raise BudgetError("budget contract root must be a JSON object")
    errors = validate_budget(budget)
    if errors:
        raise BudgetError("invalid budget contract: " + "; ".join(errors))
    return budget


def _price_table(budget: Mapping[str, Any]) -> dict[str, Decimal]:
    errors = validate_budget(budget)
    if errors:
        raise BudgetError("invalid budget contract: " + "; ".join(errors))
    return {
        name: _decimal(budget["usd_per_million_tokens"][name], f"price {name}")
        for name in REQUIRED_PRICES
    }


def _extract_number(mapping: Mapping[str, Any], keys: Sequence[str], field: str) -> Decimal | None:
    values = [(key, mapping[key]) for key in keys if key in mapping]
    if not values:
        return None
    # Duplicate aliases are accepted only when they agree.  This prevents a
    # stale estimated value from silently overriding a live value.
    parsed = [_decimal(value, f"{field}.{key}") for key, value in values]
    if len(set(parsed)) != 1:
        raise BudgetError(f"receipt has conflicting {field} aliases")
    return parsed[0]


def _nested_number(mapping: Mapping[str, Any], section: str, keys: Sequence[str], field: str) -> Decimal | None:
    nested = mapping.get(section)
    if not isinstance(nested, Mapping):
        return None
    return _extract_number(nested, keys, field)


def _token_value(mapping: Mapping[str, Any], keys: Sequence[str], field: str) -> int | None:
    values = [(key, mapping[key]) for key in keys if key in mapping]
    if not values:
        return None
    parsed = [_integer(value, f"{field}.{key}") for key, value in values]
    if len(set(parsed)) != 1:
        raise BudgetError(f"receipt has conflicting {field} aliases")
    return parsed[0]


def estimate_token_cost(
    budget: Mapping[str, Any],
    *,
    prefill_tokens: int = 0,
    cached_prefill_tokens: int = 0,
    sample_tokens: int = 0,
    train_tokens: int = 0,
) -> Decimal:
    """Conservatively price token counts using uncached prefill for all input."""

    prices = _price_table(budget)
    prefill = _integer(prefill_tokens, "prefill_tokens")
    cached = _integer(cached_prefill_tokens, "cached_prefill_tokens")
    sample = _integer(sample_tokens, "sample_tokens")
    train = _integer(train_tokens, "train_tokens")
    # The contract explicitly says to charge cached input as uncached input
    # for a conservative launch decision.  The cached rate is retained in the
    # contract for reporting/audit only.
    return (
        (Decimal(prefill + cached) * prices["prefill"])
        + (Decimal(sample) * prices["sample"])
        + (Decimal(train) * prices["train"])
    ) / MILLION


def estimate_cost(
    budget: Mapping[str, Any],
    *,
    prefill_tokens: int = 0,
    cached_prefill_tokens: int = 0,
    sample_tokens: int = 0,
    train_tokens: int = 0,
    steps: int = 0,
    train_tokens_per_step: int | None = None,
    tokens_per_step: int | None = None,
    cost_per_step_usd: Decimal | int | str | None = None,
    samples: int = 0,
    max_prompt_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> Decimal:
    """Estimate a proposed run from token/step inputs.

    ``steps`` is meaningful only with ``train_tokens_per_step`` (or an
    equivalent explicit train token count); silently assigning a made-up cost
    to an otherwise unpriced step would violate the fail-closed boundary.
    ``samples * max_output_tokens`` and ``samples * max_prompt_tokens`` are
    accepted for the conservative primary-eval evaluation ceiling.
    """

    for name, value in (("steps", steps), ("samples", samples)):
        _integer(value, name)
    prefill = _integer(prefill_tokens, "prefill_tokens")
    cached = _integer(cached_prefill_tokens, "cached_prefill_tokens")
    sample = _integer(sample_tokens, "sample_tokens")
    train = _integer(train_tokens, "train_tokens")
    if train_tokens_per_step is not None and tokens_per_step is not None:
        if _integer(train_tokens_per_step, "train_tokens_per_step") != _integer(
            tokens_per_step, "tokens_per_step"
        ):
            raise BudgetError("train_tokens_per_step and tokens_per_step disagree")
    if train_tokens_per_step is None:
        train_tokens_per_step = tokens_per_step
    if cost_per_step_usd is not None:
        return _decimal(cost_per_step_usd, "cost_per_step_usd") * _integer(steps, "steps")
    if train_tokens_per_step is not None:
        train += _integer(train_tokens_per_step, "train_tokens_per_step") * _integer(steps, "steps")
    if max_prompt_tokens is not None:
        prefill += _integer(max_prompt_tokens, "max_prompt_tokens") * _integer(samples, "samples")
    if max_output_tokens is not None:
        sample += _integer(max_output_tokens, "max_output_tokens") * _integer(samples, "samples")
    if _integer(steps, "steps") and not train:
        # A budget can provide an explicit cost-per-step in a custom projection
        # (handled by _projection_cost); a bare step count is never enough.
        raise BudgetError("projected steps require projected train tokens or train_tokens_per_step")
    return estimate_token_cost(
        budget,
        prefill_tokens=prefill,
        cached_prefill_tokens=cached,
        sample_tokens=sample,
        train_tokens=train,
    )


# Readable aliases for callers that use the wording from the contract.
conservative_token_cost = estimate_token_cost
estimate_conservative_cost = estimate_cost
project_cost = estimate_cost


def _portfolio_from_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise BudgetError("portfolio contract root must be a JSON object")
    suites = contract.get("suite_registry")
    if not isinstance(suites, Mapping):
        raise BudgetError("portfolio contract suite_registry is missing")
    domains = contract.get("domains")
    if not isinstance(domains, list) or not domains or any(not isinstance(x, str) for x in domains):
        raise BudgetError("portfolio contract domains are missing")
    # Preserve the frozen registry order so T1-T12/E1-E14 are stable labels,
    # rather than silently changing when a caller sorts a copied contract.
    training = [
        str(sid)
        for sid, suite in suites.items()
        if isinstance(suite, Mapping) and suite.get("role") == TRAINING_ROLE
    ]
    primary = [
        str(sid)
        for sid, suite in suites.items()
        if isinstance(suite, Mapping) and suite.get("role") == PRIMARY_EVAL_ROLE
    ]
    if len(training) != 12:
        raise BudgetError(f"portfolio must contain exactly 12 training suites, found {len(training)}")
    if len(primary) != 14:
        raise BudgetError(f"portfolio must contain exactly 14 primary_eval suites, found {len(primary)}")
    domain_set = set(domains)
    suite_domains: dict[str, tuple[str, ...]] = {}
    for sid, suite in suites.items():
        if not isinstance(suite, Mapping):
            raise BudgetError(f"portfolio suite {sid} is malformed")
        listed = suite.get("domains", [])
        if not isinstance(listed, list) or any(not isinstance(x, str) for x in listed):
            raise BudgetError(f"portfolio suite {sid} domains are malformed")
        unknown = set(listed) - domain_set
        if unknown:
            raise BudgetError(f"portfolio suite {sid} has unknown domains {sorted(unknown)}")
        suite_domains[str(sid)] = tuple(sorted(set(listed)))
    covered_train = set().union(*(set(suite_domains[sid]) for sid in training))
    covered_primary = set().union(*(set(suite_domains[sid]) for sid in primary))
    if covered_train != domain_set:
        raise BudgetError("training portfolio does not cover every required domain")
    if covered_primary != domain_set:
        raise BudgetError("primary_eval portfolio does not cover every required domain")
    return {
        "domains": tuple(sorted(domain_set)),
        "suites": {str(k): dict(v) for k, v in suites.items()},
        "suite_domains": suite_domains,
        "training_suite_ids": tuple(training),
        "primary_eval_suite_ids": tuple(primary),
        "training_suite_labels": {
            f"T{index}": suite_id for index, suite_id in enumerate(training, start=1)
        },
        "primary_eval_suite_labels": {
            f"E{index}": suite_id for index, suite_id in enumerate(primary, start=1)
        },
        "suite_labels": {
            **{
                f"T{index}": suite_id
                for index, suite_id in enumerate(training, start=1)
            },
            **{
                f"E{index}": suite_id
                for index, suite_id in enumerate(primary, start=1)
            },
        },
        "required_suite_ids": tuple(training + primary),
    }


def load_portfolio_contract(path: Path = DEFAULT_PORTFOLIO_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return _portfolio_from_contract(_json_load(path, "portfolio contract"))


def _portfolio_for_budget(
    budget: Mapping[str, Any], portfolio_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if portfolio_contract is not None:
        # Callers may pass a raw contract or an already normalized report.
        if "suite_domains" in portfolio_contract and "required_suite_ids" in portfolio_contract:
            return dict(portfolio_contract)
        return _portfolio_from_contract(portfolio_contract)
    embedded = budget.get("portfolio_contract") or budget.get("portfolio")
    if isinstance(embedded, Mapping) and "suite_registry" in embedded:
        return _portfolio_from_contract(embedded)
    # The Pavlov budget and domain contract are intentionally separate files;
    # use the adjacent domain contract as a read-only portfolio index.
    candidate = Path(budget.get("_portfolio_path", DEFAULT_PORTFOLIO_PATH))
    if candidate.exists():
        return load_portfolio_contract(candidate)
    return None


def _normalise_suite_id(
    suite: str | None, portfolio: Mapping[str, Any] | None
) -> str | None:
    if suite is None or portfolio is None:
        return suite
    return portfolio.get("suite_labels", {}).get(suite, suite)


def _validate_receipt_model(receipt: Mapping[str, Any], budget: Mapping[str, Any]) -> None:
    model = _receipt_model(receipt)
    if not model:
        raise BudgetError("receipt has an unknown price model")
    expected = _budget_model(budget)
    if model != expected:
        raise BudgetError(f"receipt price model {model!r} is not authorized for {expected!r}")
    pricing = receipt.get("pricing")
    if isinstance(pricing, Mapping):
        nested_model = pricing.get("model") or pricing.get("model_id")
        if nested_model is not None and nested_model != expected:
            raise BudgetError(f"receipt nested price model {nested_model!r} is unknown")
        # Receipts produced by eval_pavlov_xlam carry a pricing snapshot.  A
        # snapshot with a different rate is a different price model, even if
        # the model name was copied correctly; fail closed instead of silently
        # mixing rates.
        prices = _price_table(budget)
        aliases = {
            "prefill": ("prefill", "prefill_usd_per_million"),
            "cached_prefill": (
                "cached_prefill",
                "cached_prefill_usd_per_million",
            ),
            "sample": ("sample", "sample_usd_per_million"),
            "train": ("train", "train_usd_per_million"),
        }
        for name, keys in aliases.items():
            supplied = _extract_number(pricing, keys, f"receipt.pricing.{name}")
            if supplied is not None and supplied != prices[name]:
                raise BudgetError(
                    f"receipt pricing for {name} is not authorized for {expected!r}"
                )


def _receipt_suite(receipt: Mapping[str, Any]) -> str | None:
    for key in ("suite_id", "suite", "suite_name", "evaluation_suite", "training_suite"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _receipt_domains(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("domain", "domain_id"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for key in ("domains", "domain_ids"):
        value = receipt.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            raise BudgetError(f"receipt {key} must be a list of non-empty strings")
        values.extend(x.strip() for x in value)
    return tuple(sorted(set(values)))


def _canonical_json_value(value: Any) -> Any:
    """Return a JSON-safe, float-free value for receipt identity hashing."""

    if isinstance(value, Decimal):
        return {"__decimal__": format(value, "f")}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def _receipt_identity(receipt: Mapping[str, Any]) -> str:
    """Get a stable run identity, rejecting duplicate receipts later."""

    for key in (
        "receipt_id",
        "run_id",
        "attempt_id",
        "fingerprint",
        "receipt_sha256",
        "sha256",
    ):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip()}"
    try:
        encoded = json.dumps(
            _canonical_json_value(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BudgetError("receipt cannot be canonically identified") from exc
    return "canonical-sha256:" + hashlib.sha256(encoded).hexdigest()


def _immutable_reference(value: Any, *, allow_plain_string: bool = True) -> bool:
    """Whether an evidence entry is an immutable receipt reference."""

    if isinstance(value, str):
        return allow_plain_string and bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    if value.get("immutable") is False or value.get("verified") is False:
        return False
    return any(
        isinstance(value.get(key), str) and value[key].strip()
        for key in ("receipt_id", "id", "digest", "sha256", "hash", "fingerprint")
    )


def _heldout_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the five immutable receipts needed for a held-out claim."""

    containers: list[tuple[Mapping[str, Any], bool]] = [(receipt, False)]
    for key in (
        "heldout_receipts",
        "held_out_receipts",
        "immutable_receipts",
        "receipts",
        "heldout_evidence",
        "evidence",
    ):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            containers.append((value, key in {"immutable_receipts", "immutable_evidence"}))
    found: dict[str, Any] = {}
    missing: list[str] = []
    for required, aliases in HELDOUT_RECEIPT_FIELDS.items():
        proven_value: Any = None
        for container, immutable_container in containers:
            for alias in aliases:
                if alias in container:
                    candidate = container[alias]
                    plain_string_ok = (
                        immutable_container
                        or alias.endswith("_receipt")
                        or alias.startswith("immutable_")
                    )
                    if _immutable_reference(
                        candidate, allow_plain_string=plain_string_ok
                    ):
                        proven_value = candidate
                        break
            if proven_value is not None:
                break
        if proven_value is None:
            missing.append(required)
        else:
            found[required] = proven_value
    return {"proven": not missing, "missing": tuple(missing), "receipts": found}


def _claim_requested(value: Any, field: str) -> bool:
    """Parse a held-out claim flag without treating malformed input as false."""

    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"", "false", "0", "no", "none"}:
            return False
        if normalised in {
            "true",
            "1",
            "yes",
            "heldout",
            "held-out",
            "primary_eval",
            "primary-eval",
        }:
            return True
    raise BudgetError(
        f"{field} must be a boolean or an explicit held-out claim value"
    )


def _mapping_claim_requested(mapping: Mapping[str, Any], field: str) -> bool:
    values = [mapping[key] for key in HELDOUT_CLAIM_KEYS if key in mapping]
    if not values:
        return False
    parsed = [_claim_requested(value, field) for value in values]
    if len(set(parsed)) != 1:
        raise BudgetError(f"{field} aliases disagree")
    return parsed[0]


def _receipt_is_primary_eval(
    receipt: Mapping[str, Any], portfolio: Mapping[str, Any] | None
) -> bool:
    if portfolio is None:
        return False
    suite = _normalise_suite_id(_receipt_suite(receipt), portfolio)
    return bool(
        suite
        and suite in portfolio["suites"]
        and portfolio["suites"][suite].get("role") == PRIMARY_EVAL_ROLE
    )


def _receipt_tokens(receipt: Mapping[str, Any]) -> dict[str, int] | None:
    prefill = _token_value(receipt, ("prefill_tokens", "prompt_tokens", "input_tokens"), "prefill_tokens")
    cached = _token_value(receipt, ("cached_prefill_tokens", "cached_prompt_tokens"), "cached_prefill_tokens")
    sample = _token_value(receipt, ("sample_tokens", "completion_tokens", "output_tokens"), "sample_tokens")
    train = _token_value(receipt, ("train_tokens", "training_tokens", "trained_tokens"), "train_tokens")
    steps = _token_value(receipt, ("steps", "training_steps", "step_count"), "steps")
    per_step = _token_value(receipt, ("train_tokens_per_step", "tokens_per_step"), "train_tokens_per_step")
    samples = _token_value(receipt, ("samples", "examples", "num_samples"), "samples")
    max_prompt = _token_value(receipt, ("max_prompt_tokens", "max_input_tokens"), "max_prompt_tokens")
    max_output = _token_value(receipt, ("max_output_tokens", "max_response_tokens"), "max_output_tokens")
    if all(value is None for value in (prefill, cached, sample, train, steps, per_step, samples, max_prompt, max_output)):
        return None
    train_total = train or 0
    if per_step is not None:
        train_total += per_step * (steps or 0)
    prefill_total = prefill or 0
    sample_total = sample or 0
    if samples is not None and max_prompt is not None:
        # The maximum is a ceiling, not an amount to add on top of observed
        # tokens.  This is the conservative charge without double counting.
        prefill_total = max(prefill_total, samples * max_prompt)
    if samples is not None and max_output is not None:
        sample_total = max(sample_total, samples * max_output)
    if steps and (train or 0) == 0 and per_step is None:
        raise BudgetError("receipt step count has no train-token accounting")
    return {
        "prefill_tokens": prefill_total,
        "cached_prefill_tokens": cached or 0,
        "sample_tokens": sample_total,
        "train_tokens": train_total,
    }


def _receipt_explicit_cost(receipt: Mapping[str, Any], names: Sequence[str], field: str) -> Decimal | None:
    value = _extract_number(receipt, names, field)
    if value is not None:
        return value
    for section in ("billing", "cost", "accounting", "projection"):
        value = _nested_number(receipt, section, names, field)
        if value is not None:
            return value
    return None


def validate_receipt(
    receipt: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    portfolio: Mapping[str, Any] | None = None,
    source: str = "receipt",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return [f"{source}: receipt root must be a JSON object"]
    try:
        _validate_receipt_model(receipt, budget)
        status = _normalise_status(receipt)
        _receipt_tokens(receipt)
        _receipt_domains(receipt)
    except BudgetError as exc:
        errors.append(f"{source}: {exc}")
        return errors
    raw_suite = _receipt_suite(receipt)
    suite = _normalise_suite_id(raw_suite, portfolio)
    if portfolio is not None and suite is None:
        errors.append(f"{source}: receipt is missing a suite allocation")
    if suite and portfolio is not None and suite not in portfolio["suites"]:
        errors.append(f"{source}: unknown portfolio suite {raw_suite!r}")
    if portfolio is not None:
        domains = set(_receipt_domains(receipt))
        unknown = domains - set(portfolio["domains"])
        if unknown:
            errors.append(f"{source}: unknown portfolio domains {sorted(unknown)}")
        if suite and domains:
            expected = set(portfolio["suite_domains"].get(suite, ()))
            if not domains.issubset(expected):
                errors.append(f"{source}: receipt domains do not belong to suite {suite!r}")
    try:
        claim_requested = _mapping_claim_requested(receipt, f"{source}: held-out claim")
    except BudgetError as exc:
        errors.append(str(exc))
        claim_requested = False
    if claim_requested:
        if not _receipt_is_primary_eval(receipt, portfolio):
            errors.append(f"{source}: held-out claim requires a primary_eval suite")
        elif status not in TERMINAL_STATUSES:
            errors.append(f"{source}: held-out claim requires a completed evidence receipt")
        else:
            evidence = _heldout_evidence(receipt)
            if not evidence["proven"]:
                errors.append(
                    f"{source}: held-out claim missing immutable receipts: "
                    + ", ".join(evidence["missing"])
                )
    if status not in TERMINAL_STATUSES | PENDING_STATUSES | REJECTED_STATUSES:
        errors.append(f"{source}: unknown receipt status {status!r}")
    if status in REJECTED_STATUSES:
        allowance = _receipt_explicit_cost(
            receipt,
            (
                "rejected_run_allowance_usd",
                "rejected_cost_usd",
                "maximum_authorized_cost_usd",
                "conservative_cost_usd",
                "projected_cost_usd",
                "estimated_cost_usd",
                "estimated_cost",
                "cost_usd",
            ),
            "rejected_run_allowance_usd",
        )
        if allowance is None and _receipt_tokens(receipt) is None:
            errors.append(f"{source}: rejected receipt needs an explicit allowance or token projection")
    if (
        status in PENDING_STATUSES
        and _receipt_explicit_cost(
            receipt,
            (
                "conservative_pending_usd",
                "pending_cost_usd",
                "projected_cost_usd",
                "estimated_cost_usd",
                "base_eval_estimated_usd",
                "estimated_cost",
                "cost_usd",
            ),
            "conservative_pending_usd",
        )
        is None
        and _receipt_tokens(receipt) is None
    ):
        errors.append(f"{source}: pending receipt needs an explicit estimate or token projection")
    return errors


def load_receipt(
    path: Path,
    budget: Mapping[str, Any] | None = None,
    *,
    portfolio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _json_load(Path(path), "run receipt")
    if not isinstance(receipt, dict):
        raise BudgetError(f"run receipt root must be a JSON object: {path}")
    if budget is not None:
        errors = validate_receipt(receipt, budget, portfolio=portfolio, source=str(path))
        if errors:
            raise BudgetError("; ".join(errors))
    return receipt


def _projection_cost(
    budget: Mapping[str, Any], projection: Mapping[str, Any], *, field: str = "projection"
) -> Decimal:
    explicit = _extract_number(
        projection,
        ("cost_usd", "projected_cost_usd", "conservative_cost_usd", "estimated_cost_usd"),
        field,
    )
    if explicit is not None:
        return explicit
    nested_tokens = projection.get("tokens")
    if nested_tokens is not None and not isinstance(nested_tokens, Mapping):
        raise BudgetError(f"{field}.tokens must be an object")
    source = nested_tokens if isinstance(nested_tokens, Mapping) else projection
    prefill = source.get("prefill_tokens", source.get("prompt_tokens", 0))
    cached = source.get("cached_prefill_tokens", source.get("cached_prompt_tokens", 0))
    sample = source.get("sample_tokens", source.get("completion_tokens", 0))
    train = source.get("train_tokens", source.get("training_tokens", 0))
    steps = source.get("steps", source.get("training_steps", 0))
    per_step = source.get("train_tokens_per_step", source.get("tokens_per_step"))
    samples = source.get("samples", source.get("examples", 0))
    max_prompt = source.get("max_prompt_tokens", source.get("max_input_tokens"))
    max_output = source.get("max_output_tokens", source.get("max_response_tokens"))
    cost_per_step = _extract_number(source, ("cost_per_step_usd", "usd_per_step"), f"{field}.cost_per_step_usd")
    if cost_per_step is not None:
        return cost_per_step * _integer(steps, f"{field}.steps")
    return estimate_cost(
        budget,
        prefill_tokens=_integer(prefill, f"{field}.prefill_tokens"),
        cached_prefill_tokens=_integer(cached, f"{field}.cached_prefill_tokens"),
        sample_tokens=_integer(sample, f"{field}.sample_tokens"),
        train_tokens=_integer(train, f"{field}.train_tokens"),
        steps=_integer(steps, f"{field}.steps"),
        train_tokens_per_step=None if per_step is None else _integer(per_step, f"{field}.train_tokens_per_step"),
        samples=_integer(samples, f"{field}.samples"),
        max_prompt_tokens=None if max_prompt is None else _integer(max_prompt, f"{field}.max_prompt_tokens"),
        max_output_tokens=None if max_output is None else _integer(max_output, f"{field}.max_output_tokens"),
    )


def _receipt_breakdown(
    receipt: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    source: str,
    portfolio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = _normalise_status(receipt)
    live = _receipt_explicit_cost(
        receipt,
        (
            "live_billed_cost_usd",
            "billed_cost_usd",
            "actual_billed_usd",
            "actual_cost_usd",
            "provider_billed_cost_usd",
            "charged_cost_usd",
        ),
        "live_billed_usd",
    )
    base_estimate = _receipt_explicit_cost(
        receipt,
        ("base_eval_estimated_usd",),
        "base_eval_estimated_usd",
    )
    estimated = _receipt_explicit_cost(
        receipt,
        (
            "conservative_pending_usd",
            "pending_cost_usd",
            "conservative_cost_usd",
            "projected_cost_usd",
            "estimated_cost_usd",
            "estimated_cost",
            "cost_usd",
        ),
        "conservative_pending_usd",
    )
    allowance = _receipt_explicit_cost(
        receipt,
        (
            "rejected_run_allowance_usd",
            "rejected_cost_usd",
            "maximum_authorized_cost_usd",
        ),
        "rejected_run_allowance_usd",
    )
    tokens = _receipt_tokens(receipt)
    token_estimate = (
        None
        if tokens is None
        else estimate_token_cost(
            budget,
            prefill_tokens=tokens["prefill_tokens"],
            cached_prefill_tokens=tokens["cached_prefill_tokens"],
            sample_tokens=tokens["sample_tokens"],
            train_tokens=tokens["train_tokens"],
        )
    )
    if status in REJECTED_STATUSES:
        if allowance is None:
            allowance = estimated if estimated is not None else token_estimate
        if allowance is None:
            raise BudgetError(f"{source}: rejected receipt has no conservative allowance")
        # A rejected run's estimate is an allowance, not an additional pending
        # charge.  Explicit live billing remains known and is added separately.
        pending = allowance
    elif live is not None:
        pending = Decimal("0")
    elif estimated is not None:
        pending = estimated
    elif base_estimate is not None:
        pending = base_estimate
    elif token_estimate is not None:
        pending = token_estimate
    else:
        raise BudgetError(f"{source}: receipt has neither live billing nor a conservative estimate")

    is_base_eval = (
        str(receipt.get("schema_version", "")).startswith("pavlov-xlam-eval")
        or str(receipt.get("stage", "")).lower() in {"heldout-evaluation", "base-evaluation", "base_eval"}
        or receipt.get("kind") in {"base_eval", "base_evaluation"}
        or base_estimate is not None
    )
    if is_base_eval and base_estimate is None:
        base_estimate = estimated if estimated is not None else token_estimate
    if live is not None and is_base_eval:
        # A receipt may carry both the estimate and a later billing reconciliation;
        # keep the estimate visible but do not double charge it.
        pending = Decimal("0")
    if status in REJECTED_STATUSES:
        base_estimate = Decimal("0")
    pending_billing = (
        Decimal("0")
        if is_base_eval or status in REJECTED_STATUSES
        else pending
    )
    heldout = _heldout_evidence(receipt) if _receipt_is_primary_eval(receipt, portfolio) else {
        "proven": False,
        "missing": tuple(),
        "receipts": {},
    }
    heldout_proven = bool(heldout["proven"] and status in TERMINAL_STATUSES)
    suite_id = _normalise_suite_id(_receipt_suite(receipt), portfolio)
    return {
        "source": source,
        "status": status,
        "suite_id": suite_id,
        "domains": _receipt_domains(receipt),
        "live_billed_usd": live or Decimal("0"),
        "pending_usd": pending,
        "pending_billing_usd": pending_billing,
        "base_eval_estimated_usd": base_estimate or Decimal("0"),
        "base_eval_unreconciled_usd": (
            (base_estimate or Decimal("0"))
            if is_base_eval and live is None
            else Decimal("0")
        ),
        "rejected_run_allowance_usd": allowance or Decimal("0"),
        "primary_eval": _receipt_is_primary_eval(receipt, portfolio),
        "heldout_receipt_proven": heldout_proven,
        "heldout_missing_receipts": heldout["missing"],
    }


def _mapping_decimal_values(value: Any, field: str) -> dict[str, Decimal]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BudgetError(f"{field} must be an object mapping names to amounts")
    result: dict[str, Decimal] = {}
    for key, amount in value.items():
        if not isinstance(key, str) or not key.strip():
            raise BudgetError(f"{field} contains an empty name")
        result[key.strip()] = _decimal(amount, f"{field}.{key}")
    return result


def _projection_mapping(
    projection: Mapping[str, Any], primary: str, alias: str, field: str
) -> dict[str, Decimal]:
    """Read a canonical/legacy allocation key pair without silent override."""

    primary_present = primary in projection
    alias_present = alias in projection
    primary_values = (
        _mapping_decimal_values(projection[primary], field) if primary_present else None
    )
    alias_values = (
        _mapping_decimal_values(projection[alias], field) if alias_present else None
    )
    if primary_values is not None and alias_values is not None:
        if primary_values != alias_values:
            raise BudgetError(f"{field} and legacy alias {alias} disagree")
        return primary_values
    return primary_values if primary_values is not None else (alias_values or {})


def _normalise_suite_allocations(
    values: Mapping[str, Decimal], portfolio: Mapping[str, Any] | None, field: str
) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    for name, amount in values.items():
        suite_id = _normalise_suite_id(name, portfolio)
        if suite_id is None:
            raise BudgetError(f"{field} contains an empty suite")
        if suite_id in result:
            raise BudgetError(f"{field} contains duplicate suite allocation {name!r}")
        result[suite_id] = amount
    return result


def _split_amount(amount: Decimal, labels: Sequence[str]) -> dict[str, Decimal]:
    """Split an amount deterministically while preserving an exact sum."""

    ordered = tuple(sorted(set(labels)))
    if not ordered or amount == 0:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: amount}
    share = amount / Decimal(len(ordered))
    result = {label: share for label in ordered[:-1]}
    result[ordered[-1]] = amount - sum(result.values(), Decimal("0"))
    return result


def _add_amount(target: dict[str, Decimal], key: str, amount: Decimal) -> None:
    target[key] = target.get(key, Decimal("0")) + amount


def _projection_parts(
    budget: Mapping[str, Any], projection: Mapping[str, Any] | None, portfolio: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if projection is None:
        # A future revision of the budget may carry frozen portfolio/phase
        # reservations.  They are read-only defaults; the current contract has
        # none, so an ordinary audit remains a zero-cost projection.
        defaults: dict[str, Any] = {}
        for key in (
            "suite_costs",
            "domain_costs",
            "suite_reservations",
            "domain_reservations",
            "phase_reservations",
            "cost_usd",
            "projected_cost_usd",
            *HELDOUT_CLAIM_KEYS,
            "heldout_suite_ids",
            "heldout_suites",
        ):
            if key in budget:
                defaults[key] = budget[key]
        if defaults:
            projection = defaults
        else:
            return {
                "cost_usd": Decimal("0"),
                "direct_cost_usd": Decimal("0"),
                "reservation_total_usd": Decimal("0"),
                "suite_costs": {},
                "domain_costs": {},
                "suite_reservations": {},
                "domain_reservations": {},
                "phase_reservations": {},
                "arm": None,
                "phase": None,
                "coverage_required": False,
                "allocation_complete_required": False,
                "claim_heldout": False,
                "heldout_suite_ids": (),
            }
    if not isinstance(projection, Mapping):
        raise BudgetError("projection must be a JSON object")
    claim_heldout = _mapping_claim_requested(projection, "projection held-out claim")
    if "heldout_suite_ids" in projection and "heldout_suites" in projection:
        if projection["heldout_suite_ids"] != projection["heldout_suites"]:
            raise BudgetError(
                "projection heldout_suite_ids and heldout_suites aliases disagree"
            )
    heldout_values = projection.get(
        "heldout_suite_ids", projection.get("heldout_suites")
    )
    if heldout_values is None:
        heldout_suite_ids: tuple[str, ...] = (
            tuple(portfolio["primary_eval_suite_ids"]) if claim_heldout and portfolio else ()
        )
    else:
        if not isinstance(heldout_values, (list, tuple, set)):
            raise BudgetError("projection.heldout_suite_ids must be a list of strings")
        normalised_heldout: list[str] = []
        heldout_iterable = (
            sorted(heldout_values, key=str)
            if isinstance(heldout_values, set)
            else heldout_values
        )
        for index, value in enumerate(heldout_iterable):
            if not isinstance(value, str) or not value.strip():
                raise BudgetError(
                    f"projection.heldout_suite_ids[{index}] must be a non-empty string"
                )
            suite_id = _normalise_suite_id(value.strip(), portfolio)
            if suite_id is None:
                raise BudgetError("projection held-out claim requires a portfolio contract")
            if suite_id in normalised_heldout:
                raise BudgetError(
                    f"projection.heldout_suite_ids contains duplicate suite {value!r}"
                )
            normalised_heldout.append(suite_id)
        heldout_suite_ids = tuple(normalised_heldout)
    if claim_heldout and portfolio is None:
        raise BudgetError("projection held-out claim requires a portfolio contract")
    suite_costs = _projection_mapping(
        projection, "suite_costs", "suites_cost_usd", "projection.suite_costs"
    )
    domain_costs = _projection_mapping(
        projection, "domain_costs", "domains_cost_usd", "projection.domain_costs"
    )
    suite_reservations = _projection_mapping(
        projection,
        "suite_reservations",
        "reservations_by_suite",
        "projection.suite_reservations",
    )
    domain_reservations = _projection_mapping(
        projection,
        "domain_reservations",
        "reservations_by_domain",
        "projection.domain_reservations",
    )
    phase_reservations = _projection_mapping(
        projection,
        "phase_reservations",
        "reservations_by_phase",
        "projection.phase_reservations",
    )
    suite_allocation_supplied = bool(
        "suite_costs" in projection
        or "suites_cost_usd" in projection
        or "suite_reservations" in projection
        or "reservations_by_suite" in projection
        or isinstance(projection.get("suites"), list)
    )
    domain_allocation_supplied = bool(
        "domain_costs" in projection
        or "domains_cost_usd" in projection
        or "domain_reservations" in projection
        or "reservations_by_domain" in projection
        or isinstance(projection.get("domains"), list)
    )

    # A list form is convenient for generated manifests and retains per-suite
    # token accounting.  Explicit cost maps remain the canonical output.
    suite_entries = projection.get("suites")
    seen_suite_entries: set[str] = set()
    if isinstance(suite_entries, list):
        for index, entry in enumerate(suite_entries):
            if not isinstance(entry, Mapping):
                raise BudgetError(f"projection.suites[{index}] must be an object")
            suite_id = entry.get("suite_id", entry.get("id", entry.get("name")))
            if not isinstance(suite_id, str) or not suite_id.strip():
                raise BudgetError(f"projection.suites[{index}] has no suite_id")
            suite_id = suite_id.strip()
            if suite_id in seen_suite_entries:
                raise BudgetError(
                    f"projection.suites contains duplicate suite allocation {suite_id!r}"
                )
            seen_suite_entries.add(suite_id)
            suite_costs[suite_id] = suite_costs.get(suite_id, Decimal("0")) + _projection_cost(
                budget, entry, field=f"projection.suites[{index}]"
            )
    domain_entries = projection.get("domains")
    seen_domain_entries: set[str] = set()
    if isinstance(domain_entries, list):
        for index, entry in enumerate(domain_entries):
            if not isinstance(entry, Mapping):
                raise BudgetError(f"projection.domains[{index}] must be an object")
            domain_id = entry.get("domain_id", entry.get("id", entry.get("name")))
            if not isinstance(domain_id, str) or not domain_id.strip():
                raise BudgetError(f"projection.domains[{index}] has no domain_id")
            domain_id = domain_id.strip()
            if domain_id in seen_domain_entries:
                raise BudgetError(
                    f"projection.domains contains duplicate domain allocation {domain_id!r}"
                )
            seen_domain_entries.add(domain_id)
            domain_costs[domain_id] = domain_costs.get(domain_id, Decimal("0")) + _projection_cost(
                budget, entry, field=f"projection.domains[{index}]"
            )

    suite_costs = _normalise_suite_allocations(
        suite_costs, portfolio, "projection.suite_costs"
    )
    suite_reservations = _normalise_suite_allocations(
        suite_reservations, portfolio, "projection.suite_reservations"
    )

    direct_keys = (
        "cost_usd",
        "projected_cost_usd",
        "conservative_cost_usd",
        "estimated_cost_usd",
    )
    token_keys = (
        "prefill_tokens",
        "cached_prefill_tokens",
        "sample_tokens",
        "train_tokens",
        "training_tokens",
        "steps",
        "samples",
        "tokens",
    )
    explicit_cost = _extract_number(projection, direct_keys, "projection.cost_usd")
    token_cost = _projection_cost(budget, projection, field="projection") if any(
        key in projection
        for key in token_keys
    ) else None
    suite_total = sum(suite_costs.values(), Decimal("0"))
    domain_total = sum(domain_costs.values(), Decimal("0"))
    if suite_costs and domain_costs and suite_total != domain_total:
        raise BudgetError(
            "projection suite_costs and domain_costs disagree; "
            "provide one allocation or matching breakdowns"
        )
    allocation_total = suite_total or domain_total
    if explicit_cost is not None:
        if allocation_total and explicit_cost != allocation_total:
            raise BudgetError(
                "projection cost_usd disagrees with suite/domain allocation"
            )
        direct_cost = explicit_cost
    elif token_cost is not None:
        if allocation_total:
            raise BudgetError(
                "projection token inputs and suite/domain allocation are ambiguous"
            )
        direct_cost = token_cost
    else:
        direct_cost = allocation_total
    suite_reservation_total = sum(suite_reservations.values(), Decimal("0"))
    domain_reservation_total = sum(domain_reservations.values(), Decimal("0"))
    if suite_reservations and domain_reservations and suite_reservation_total != domain_reservation_total:
        raise BudgetError(
            "projection suite_reservations and domain_reservations disagree; "
            "provide one allocation or matching breakdowns"
        )
    reservation_total = (
        (
            suite_reservation_total
            if suite_reservations
            else domain_reservation_total
        )
        + sum(phase_reservations.values(), Decimal("0"))
    )
    arm = projection.get("arm", projection.get("arm_name"))
    if arm is not None and not isinstance(arm, str):
        raise BudgetError("projection arm must be a string")
    if isinstance(arm, str):
        planned = budget.get("planned_arms")
        if isinstance(planned, list):
            names = {
                entry.get("name")
                for entry in planned
                if isinstance(entry, Mapping) and isinstance(entry.get("name"), str)
            }
            if arm.strip() not in names:
                raise BudgetError(f"projection arm {arm.strip()!r} is not in planned_arms")
    phase = projection.get("phase")
    if phase is not None and not isinstance(phase, str):
        raise BudgetError("projection phase must be a string")
    required_domains_value = projection.get(
        "required_domains", portfolio["domains"] if portfolio and (
            projection.get("enforce_domain_coverage", False)
            or suite_costs
            or domain_costs
            or suite_reservations
            or domain_reservations
        ) else ()
    )
    if not isinstance(required_domains_value, (list, tuple, set)) or any(
        not isinstance(value, str) or not value.strip() for value in required_domains_value
    ):
        raise BudgetError("projection.required_domains must be a list of strings")
    required_domain_items = tuple(value.strip() for value in required_domains_value)
    if len(set(required_domain_items)) != len(required_domain_items):
        raise BudgetError("projection.required_domains contains duplicate domains")
    coverage_required = bool(
        projection.get("enforce_domain_coverage", False)
        or required_domains_value
        or suite_costs
        or domain_costs
        or suite_reservations
        or domain_reservations
    )
    allocation_complete_required = bool(
        portfolio
        and (
            projection.get("complete_portfolio", False)
            or projection.get("require_complete_allocations", False)
            or suite_allocation_supplied
            or domain_allocation_supplied
        )
    )
    if isinstance(arm, str) and direct_cost == 0 and not allocation_total and not reservation_total:
        raise BudgetError("projection arm has no projected cost or reservation")
    if portfolio is not None:
        valid_suites = set(portfolio["required_suite_ids"])
        valid_domains = set(portfolio["domains"])
        for name, values in (
            ("suite_costs", suite_costs),
            ("suite_reservations", suite_reservations),
        ):
            unknown = set(values) - valid_suites
            if unknown:
                raise BudgetError(f"projection.{name} has unknown suites {sorted(unknown)}")
        for name, values in (
            ("domain_costs", domain_costs),
            ("domain_reservations", domain_reservations),
        ):
            unknown = set(values) - valid_domains
            if unknown:
                raise BudgetError(f"projection.{name} has unknown domains {sorted(unknown)}")
        heldout_unknown = set(heldout_suite_ids) - set(portfolio["primary_eval_suite_ids"])
        if heldout_unknown:
            raise BudgetError(
                "projection held-out claim names non-primary_eval suites "
                f"{sorted(heldout_unknown)}"
            )
        if allocation_complete_required:
            required_suites = set(portfolio["required_suite_ids"])
            required_domains = set(portfolio["domains"])
            if not suite_allocation_supplied:
                raise BudgetError(
                    "complete portfolio allocation is missing suite allocations "
                    "for T1-T12/E1-E14"
                )
            if not domain_allocation_supplied:
                raise BudgetError(
                    "complete portfolio allocation is missing domain allocations"
                )
            missing_suites = sorted(required_suites - set(suite_costs))
            extra_suites = sorted(set(suite_costs) - required_suites)
            missing_domains = sorted(required_domains - set(domain_costs))
            extra_domains = sorted(set(domain_costs) - required_domains)
            if missing_suites or extra_suites:
                raise BudgetError(
                    "suite allocation must cover exactly T1-T12/E1-E14; "
                    f"missing={missing_suites} extra={extra_suites}"
                )
            if missing_domains or extra_domains:
                raise BudgetError(
                    "domain allocation must cover exactly all required domains; "
                    f"missing={missing_domains} extra={extra_domains}"
                )
    required_domains = tuple(
        sorted(required_domain_items, key=str)
        if isinstance(required_domains_value, set)
        else required_domain_items
    )
    return {
        "cost_usd": direct_cost + reservation_total,
        "direct_cost_usd": direct_cost,
        "reservation_total_usd": reservation_total,
        "suite_costs": suite_costs,
        "domain_costs": domain_costs,
        "suite_reservations": suite_reservations,
        "domain_reservations": domain_reservations,
        "phase_reservations": phase_reservations,
        "arm": arm.strip() if isinstance(arm, str) else None,
        "phase": phase.strip() if isinstance(phase, str) else None,
        "coverage_required": coverage_required,
        "allocation_complete_required": allocation_complete_required,
        "required_domains": required_domains,
        "claim_heldout": claim_heldout,
        "heldout_suite_ids": heldout_suite_ids,
    }


def _coverage_issues(
    parts: Mapping[str, Any], portfolio: Mapping[str, Any] | None,
) -> list[str]:
    if portfolio is None or not parts.get("coverage_required"):
        return []
    valid_domains = set(portfolio["domains"])
    required_domains = set(parts.get("required_domains", ()))
    unknown = required_domains - valid_domains
    if unknown:
        return [f"projection required unknown domains {sorted(unknown)}"]
    if not required_domains:
        return []

    if not any(
        amount > 0
        for key in ("suite_costs", "domain_costs", "suite_reservations", "domain_reservations")
        for amount in parts.get(key, {}).values()
    ):
        # A zero-cost audit has no suite capable of crowding a domain out.
        # Complete-allocation validation still runs before this check when a
        # portfolio breakdown was supplied.
        return []

    # A domain is reserved if an explicit domain reservation/cost exists or a
    # suite reservation/cost covering that domain exists.  This is deliberately
    # a positive reservation test: spending the whole budget on one suite must
    # not silently crowd every other required domain out.
    covered: set[str] = set()
    suite_amounts: dict[str, Decimal] = {}
    for key in ("suite_costs", "suite_reservations"):
        for suite_id, amount in parts.get(key, {}).items():
            suite_amounts[suite_id] = suite_amounts.get(suite_id, Decimal("0")) + amount
    for suite_id, amount in suite_amounts.items():
        if amount > 0:
            covered.update(portfolio["suite_domains"].get(suite_id, ()))
    for key in ("domain_costs", "domain_reservations"):
        covered.update(name for name, amount in parts.get(key, {}).items() if amount > 0)
    missing = sorted(required_domains - covered)
    if missing:
        return [
            "projection would crowd out required domain coverage: "
            + ", ".join(missing)
        ]
    return []


def build_ledger(
    budget: Mapping[str, Any],
    receipts: Iterable[Mapping[str, Any]] = (),
    projection: Mapping[str, Any] | None = None,
    *,
    portfolio_contract: Mapping[str, Any] | None = None,
    receipt_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build an exact-decimal ledger and launch preflight report."""

    errors = validate_budget(budget)
    if errors:
        raise BudgetError("invalid budget contract: " + "; ".join(errors))
    portfolio = _portfolio_for_budget(budget, portfolio_contract)
    receipt_items = list(receipts)
    sources = list(
        receipt_sources
        or [
            str(item) if isinstance(item, (str, Path)) else f"receipt[{i}]"
            for i, item in enumerate(receipt_items)
        ]
    )
    if len(sources) != len(receipt_items):
        raise BudgetError("receipt_sources length must match receipts")
    breakdowns: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    for item, source in zip(receipt_items, sources):
        receipt = (
            load_receipt(Path(item), budget, portfolio=portfolio)
            if isinstance(item, (str, Path))
            else item
        )
        receipt_errors = validate_receipt(receipt, budget, portfolio=portfolio, source=source)
        if receipt_errors:
            raise BudgetError("; ".join(receipt_errors))
        identity = _receipt_identity(receipt)
        if identity in seen_receipts:
            raise BudgetError(f"duplicate receipt: {identity}")
        seen_receipts.add(identity)
        breakdowns.append(
            _receipt_breakdown(
                receipt, budget, source=source, portfolio=portfolio
            )
        )
    parts = _projection_parts(budget, projection, portfolio)
    coverage_issues = _coverage_issues(parts, portfolio)
    heldout_proven_ids: set[str] = {
        entry["suite_id"]
        for entry in breakdowns
        if entry["primary_eval"] and entry["heldout_receipt_proven"]
    }

    known = sum((entry["live_billed_usd"] for entry in breakdowns), Decimal("0"))
    pending_billing = sum(
        (entry["pending_billing_usd"] for entry in breakdowns), Decimal("0")
    )
    base_eval = sum((entry["base_eval_estimated_usd"] for entry in breakdowns), Decimal("0"))
    base_eval_unreconciled = sum(
        (entry["base_eval_unreconciled_usd"] for entry in breakdowns), Decimal("0")
    )
    rejected = sum((entry["rejected_run_allowance_usd"] for entry in breakdowns), Decimal("0"))
    pending = pending_billing + base_eval_unreconciled + rejected
    proposed = parts["cost_usd"]
    projected = known + pending + proposed
    maximum = _decimal(budget["maximum_usd"], "maximum_usd")
    operational = _decimal(budget["operational_cap_usd"], "operational_cap_usd")
    reserve = _decimal(budget["safety_reserve_usd"], "safety_reserve_usd")
    op_remaining = operational - projected
    hard_remaining = maximum - projected
    # Reserve is preserved exactly when the operational cap is respected.  The
    # value is a displayable amount of the hard cap not available for spending.
    reserve_preserved = projected <= operational
    reasons = list(coverage_issues)
    if parts["claim_heldout"]:
        missing_heldout = tuple(
            suite_id
            for suite_id in parts["heldout_suite_ids"]
            if suite_id not in heldout_proven_ids
        )
        if missing_heldout:
            reasons.append(
                "held-out claim requires immutable split/license/task/container/"
                "decontamination receipts for: "
                + ", ".join(missing_heldout)
            )
    if projected > operational:
        reasons.append(
            f"projected spend ${projected} exceeds operational cap ${operational}"
        )
    if projected > maximum:
        reasons.append(f"projected spend ${projected} exceeds hard cap ${maximum}")
    status = "ALLOW" if not reasons else "REJECT"
    categories = (
        "live_billed",
        "pending_billing",
        "base_eval_estimated",
        "rejected_unreconciled",
        "projected",
        "reservation",
    )
    suite_allocations: dict[str, dict[str, Decimal]] = {
        category: {} for category in categories
    }
    domain_allocations: dict[str, dict[str, Decimal]] = {
        category: {} for category in categories
    }
    unallocated_suite: dict[str, Decimal] = {
        category: Decimal("0") for category in categories
    }
    unallocated_domain: dict[str, Decimal] = {
        category: Decimal("0") for category in categories
    }

    def allocate_receipt(entry: Mapping[str, Any]) -> None:
        suite = entry["suite_id"]
        domains = entry["domains"] or (
            portfolio["suite_domains"].get(suite, ()) if portfolio and suite else ()
        )
        values = {
            "live_billed": entry["live_billed_usd"],
            "pending_billing": entry["pending_billing_usd"],
            "base_eval_estimated": entry["base_eval_unreconciled_usd"],
            "rejected_unreconciled": entry["rejected_run_allowance_usd"],
        }
        for category, amount in values.items():
            if suite:
                _add_amount(suite_allocations[category], suite, amount)
            else:
                unallocated_suite[category] += amount
            shares = _split_amount(amount, domains)
            if shares:
                for domain, share in shares.items():
                    _add_amount(domain_allocations[category], domain, share)
            elif suite:
                # A suite without a domain list is still owned by the suite;
                # only the domain side is unallocated.  Do not count the
                # amount twice on the suite side.
                unallocated_domain[category] += amount
            else:
                # Keep an unallocated amount in both ledgers when neither
                # dimension can own it; this makes omission visible while
                # preserving exact suite/domain reconciliation.
                unallocated_suite[category] += amount
                unallocated_domain[category] += amount

    for entry in breakdowns:
        allocate_receipt(entry)

    suite_cost_total = sum(parts["suite_costs"].values(), Decimal("0"))
    domain_cost_total = sum(parts["domain_costs"].values(), Decimal("0"))
    if parts["suite_costs"]:
        for suite, amount in parts["suite_costs"].items():
            _add_amount(suite_allocations["projected"], suite, amount)
        if parts["domain_costs"]:
            for domain, amount in parts["domain_costs"].items():
                _add_amount(domain_allocations["projected"], domain, amount)
        else:
            for suite, amount in parts["suite_costs"].items():
                domains = portfolio["suite_domains"].get(suite, ()) if portfolio else ()
                for domain, share in _split_amount(amount, domains).items():
                    _add_amount(domain_allocations["projected"], domain, share)
    elif parts["domain_costs"]:
        for domain, amount in parts["domain_costs"].items():
            _add_amount(domain_allocations["projected"], domain, amount)
        unallocated_suite["projected"] += domain_cost_total
    else:
        unallocated_suite["projected"] += parts["direct_cost_usd"]
        unallocated_domain["projected"] += parts["direct_cost_usd"]

    suite_reservation_total = sum(
        parts["suite_reservations"].values(), Decimal("0")
    )
    domain_reservation_total = sum(
        parts["domain_reservations"].values(), Decimal("0")
    )
    if parts["suite_reservations"]:
        for suite, amount in parts["suite_reservations"].items():
            _add_amount(suite_allocations["reservation"], suite, amount)
        if parts["domain_reservations"]:
            for domain, amount in parts["domain_reservations"].items():
                _add_amount(domain_allocations["reservation"], domain, amount)
        else:
            for suite, amount in parts["suite_reservations"].items():
                domains = portfolio["suite_domains"].get(suite, ()) if portfolio else ()
                for domain, share in _split_amount(amount, domains).items():
                    _add_amount(domain_allocations["reservation"], domain, share)
    elif parts["domain_reservations"]:
        for domain, amount in parts["domain_reservations"].items():
            _add_amount(domain_allocations["reservation"], domain, amount)
        unallocated_suite["reservation"] += domain_reservation_total
    for amount in parts["phase_reservations"].values():
        unallocated_suite["reservation"] += amount
        unallocated_domain["reservation"] += amount

    # Any direct token/step projection that is not keyed to suites/domains is
    # visible as unallocated prospective usage, never silently discarded.
    if not parts["suite_costs"] and not parts["domain_costs"]:
        unallocated_suite["projected"] = parts["direct_cost_usd"]
        unallocated_domain["projected"] = parts["direct_cost_usd"]

    suite_total = sum(
        (sum(values.values(), Decimal("0")) for values in suite_allocations.values()),
        Decimal("0"),
    ) + sum(unallocated_suite.values(), Decimal("0"))
    domain_total = sum(
        (sum(values.values(), Decimal("0")) for values in domain_allocations.values()),
        Decimal("0"),
    ) + sum(unallocated_domain.values(), Decimal("0"))
    if suite_total != domain_total or suite_total != projected:
        raise BudgetError(
            "allocation reconciliation failed: "
            f"suite_total={suite_total} domain_total={domain_total} projected={projected}"
        )
    heldout_pending_ids: tuple[str, ...] = tuple(
        suite_id
        for suite_id in (portfolio["primary_eval_suite_ids"] if portfolio else ())
        if suite_id not in heldout_proven_ids
    )
    heldout_proven_labels: tuple[str, ...] = tuple(
        label
        for label, suite_id in (
            portfolio["primary_eval_suite_labels"].items() if portfolio else ()
        )
        if suite_id in heldout_proven_ids
    )
    heldout_pending_labels: tuple[str, ...] = tuple(
        label
        for label, suite_id in (
            portfolio["primary_eval_suite_labels"].items() if portfolio else ()
        )
        if suite_id in set(heldout_pending_ids)
    )

    def _suite_report(category: str) -> dict[str, Decimal]:
        return {
            suite_id: suite_allocations[category].get(suite_id, Decimal("0"))
            for suite_id in (portfolio["required_suite_ids"] if portfolio else ())
        }

    def _domain_report(category: str) -> dict[str, Decimal]:
        return {
            domain: domain_allocations[category].get(domain, Decimal("0"))
            for domain in (portfolio["domains"] if portfolio else ())
        }

    def _label_suite_report(category: str) -> dict[str, Decimal]:
        return {
            label: suite_allocations[category].get(suite_id, Decimal("0"))
            for label, suite_id in (
                portfolio["suite_labels"].items() if portfolio else ()
            )
        }

    portfolio_report: dict[str, Any] | None = None
    if portfolio is not None:
        suite_total_report = {
            suite_id: sum(
                (
                    suite_allocations[category].get(suite_id, Decimal("0"))
                    for category in categories
                ),
                Decimal("0"),
            )
            for suite_id in portfolio["required_suite_ids"]
        }
        domain_total_report = {
            domain: sum(
                (
                    domain_allocations[category].get(domain, Decimal("0"))
                    for category in categories
                ),
                Decimal("0"),
            )
            for domain in portfolio["domains"]
        }
        portfolio_report = {
            "training_suite_count": len(portfolio["training_suite_ids"]),
            "primary_eval_suite_count": len(portfolio["primary_eval_suite_ids"]),
            "training_suite_ids": list(portfolio["training_suite_ids"]),
            "primary_eval_suite_ids": list(portfolio["primary_eval_suite_ids"]),
            "training_suite_labels": dict(portfolio["training_suite_labels"]),
            "primary_eval_suite_labels": dict(portfolio["primary_eval_suite_labels"]),
            "required_domain_count": len(portfolio["domains"]),
            "required_domains": list(portfolio["domains"]),
            # Legacy aliases remain, but the explicit category maps are the
            # reconciliation source of truth.
            "suite_known_usd": _suite_report("live_billed"),
            "suite_pending_usd": _suite_report("pending_billing"),
            "suite_projected_usd": _suite_report("projected"),
            "suite_reservations_usd": _suite_report("reservation"),
            "domain_known_usd": _domain_report("live_billed"),
            "domain_pending_usd": _domain_report("pending_billing"),
            "domain_projected_usd": _domain_report("projected"),
            "domain_reservations_usd": _domain_report("reservation"),
            "suite_allocations": {
                category: _label_suite_report(category) for category in categories
            },
            "domain_allocations": {
                category: _domain_report(category) for category in categories
            },
            "suite_total_usd": suite_total_report,
            "domain_total_usd": domain_total_report,
            "phase_reservations_usd": dict(parts["phase_reservations"]),
            "heldout_receipt_proven_suite_ids": sorted(heldout_proven_ids),
            "heldout_pending_suite_ids": list(heldout_pending_ids),
            "heldout_receipt_proven_labels": list(heldout_proven_labels),
            "heldout_pending_labels": list(heldout_pending_labels),
            "heldout_receipt_proven_count": len(heldout_proven_ids),
            "heldout_pending_count": len(heldout_pending_ids),
            "unallocated_suite_usd": dict(unallocated_suite),
            "unallocated_domain_usd": dict(unallocated_domain),
            "unallocated_usd": sum(unallocated_suite.values(), Decimal("0")),
            "coverage_issues": coverage_issues,
        }
    return {
        "schema_version": "pavlov-tinker-budget-ledger-v1",
        "status": status,
        # Short names mirror the wording used by the launch checklist; the
        # *_usd names remain the unambiguous machine-facing interface.
        "known": known,
        "known_usd": known,
        "live": known,
        "live_billed_usd": known,
        "receipt_pending_usd": pending,
        "pending_billing": pending_billing,
        "pending_billing_usd": pending_billing,
        "base_eval_estimated_usd": base_eval,
        "base_eval_unreconciled_usd": base_eval_unreconciled,
        "rejected_run_allowance_usd": rejected,
        "rejected_unreconciled": rejected,
        "rejected_unreconciled_usd": rejected,
        "conservative_pending": pending,
        "conservative_pending_usd": pending,
        "proposed_arm_usd": proposed,
        "projected": projected,
        "projected_usage": projected,
        "projected_usd": projected,
        "projected_usage_usd": projected,
        "usage_by_state": {
            "live_billed_usd": known,
            "pending_billing_usd": pending_billing,
            "base_eval_estimated_usd": base_eval,
            "base_eval_unreconciled_usd": base_eval_unreconciled,
            "rejected_unreconciled_usd": rejected,
            "projected_usage_usd": projected,
        },
        "remaining": op_remaining,
        "remaining_usd": op_remaining,
        "operational_remaining_usd": op_remaining,
        "hard_remaining_usd": hard_remaining,
        "hard_cap_usd": maximum,
        "operational_cap_usd": operational,
        "safety_reserve_usd": reserve,
        "reserve_preserved": reserve_preserved,
        "arm": parts["arm"],
        "phase": parts["phase"],
        "direct_projection_usd": parts["direct_cost_usd"],
        "portfolio_reservation_usd": parts["reservation_total_usd"],
        "phase_reservations_usd": parts["phase_reservations"],
        "allocation_complete_required": parts["allocation_complete_required"],
        "allocation_reconciled": True,
        "reconciliation": {
            "suite_total_usd": suite_total,
            "domain_total_usd": domain_total,
            "projected_usage_usd": projected,
        },
        "suite_allocations": {
            category: dict(values) for category, values in suite_allocations.items()
        },
        "suite_label_allocations": {
            category: _label_suite_report(category) for category in categories
        },
        "domain_allocations": {
            category: dict(values) for category, values in domain_allocations.items()
        },
        "unallocated_suite_usd": dict(unallocated_suite),
        "unallocated_domain_usd": dict(unallocated_domain),
        "unallocated_usd": sum(unallocated_suite.values(), Decimal("0")),
        "heldout_claim_requested": parts["claim_heldout"],
        "heldout_claim_suite_ids": list(parts["heldout_suite_ids"]),
        "heldout_receipt_proven_suite_ids": sorted(heldout_proven_ids),
        "heldout_pending_suite_ids": list(heldout_pending_ids),
        "heldout_receipt_proven_labels": list(heldout_proven_labels),
        "heldout_pending_labels": list(heldout_pending_labels),
        "heldout_receipt_proven_count": len(heldout_proven_ids),
        "heldout_pending_count": len(heldout_pending_ids),
        "reasons": reasons,
        "receipts": breakdowns,
        "portfolio": portfolio_report,
    }


def can_launch(ledger: Mapping[str, Any]) -> bool:
    return ledger.get("status") == "ALLOW" and bool(ledger.get("reserve_preserved", False))


# Compatibility names for callers that describe the same side-effect-free
# operation as a calculation/check rather than a ledger build/guard.
calculate_ledger = build_ledger


def guard_launch(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return the launch decision without mutating or launching anything."""

    report = dict(ledger)
    report["allowed"] = can_launch(ledger)
    report["decision"] = "ALLOW" if report["allowed"] else "REJECT"
    return report


check_launch = guard_launch


def authorize_arm(
    budget: Mapping[str, Any],
    receipts: Iterable[Mapping[str, Any]] = (),
    projection: Mapping[str, Any] | None = None,
    *,
    portfolio_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a guard report or raise :class:`LaunchRejected`.

    This is the side-effect-free boundary a launcher can call immediately
    before creating a Tinker job.
    """

    ledger = build_ledger(
        budget,
        receipts,
        projection,
        portfolio_contract=portfolio_contract,
    )
    report = guard_launch(ledger)
    if not report["allowed"]:
        raise LaunchRejected("Pavlov Tinker launch rejected: " + "; ".join(report["reasons"]), report)
    return report


def _parse_name_amount(values: Sequence[str], field: str) -> dict[str, Decimal]:
    parsed: dict[str, Decimal] = {}
    for item in values:
        if "=" not in item:
            raise BudgetError(f"{field} expects NAME=USD, got {item!r}")
        name, amount = item.split("=", 1)
        if not name.strip():
            raise BudgetError(f"{field} has an empty name")
        normalised = name.strip()
        if normalised in parsed:
            raise BudgetError(f"{field} contains duplicate allocation {normalised!r}")
        parsed[normalised] = _decimal(amount, f"{field}.{normalised}")
    return parsed


def _parse_cli_projection(args: argparse.Namespace, budget: Mapping[str, Any]) -> dict[str, Any] | None:
    projection: dict[str, Any] = {}
    if args.projection:
        raw = _json_load(Path(args.projection), "projection JSON")
        if not isinstance(raw, dict):
            raise BudgetError("projection JSON root must be an object")
        projection.update(raw)
    if args.projected_tokens_json:
        token_path = Path(args.projected_tokens_json)
        if token_path.exists():
            raw_tokens = _json_load(token_path, "projected token JSON")
        else:
            try:
                raw_tokens = json.loads(args.projected_tokens_json, parse_float=Decimal)
            except json.JSONDecodeError as exc:
                raise BudgetError("projected token JSON is malformed") from exc
        if not isinstance(raw_tokens, Mapping):
            raise BudgetError("projected token JSON root must be an object")
        projection.update(raw_tokens)
    suite_costs = _parse_name_amount(args.suite_cost, "--suite-cost")
    domain_costs = _parse_name_amount(args.domain_cost, "--domain-cost")
    suite_reservations = _parse_name_amount(args.suite_reservation, "--suite-reservation")
    domain_reservations = _parse_name_amount(args.domain_reservation, "--domain-reservation")
    phase_reservations = _parse_name_amount(args.phase_reservation, "--phase-reservation")

    def merge_mapping(field: str, updates: Mapping[str, Decimal], *keys: str) -> None:
        if not updates:
            return
        existing: Any = None
        source_key: str | None = None
        for key in keys:
            if key in projection:
                existing = projection[key]
                source_key = key
                break
        if existing is None:
            existing = {}
        if not isinstance(existing, Mapping):
            raise BudgetError(f"projection.{field} must be an object mapping names to amounts")
        duplicate_names = set(existing) & set(updates)
        if duplicate_names:
            raise BudgetError(
                f"projection.{field} contains duplicate allocations "
                f"{sorted(duplicate_names, key=str)}"
            )
        projection[field] = {**existing, **updates}
        if source_key is not None and source_key != field:
            projection.pop(source_key, None)

    merge_mapping("suite_costs", suite_costs, "suite_costs", "suites_cost_usd")
    merge_mapping("domain_costs", domain_costs, "domain_costs", "domains_cost_usd")
    merge_mapping(
        "suite_reservations", suite_reservations, "suite_reservations", "reservations_by_suite"
    )
    merge_mapping(
        "domain_reservations", domain_reservations, "domain_reservations", "reservations_by_domain"
    )
    merge_mapping(
        "phase_reservations", phase_reservations, "phase_reservations", "reservations_by_phase"
    )
    direct = {
        "prefill_tokens": args.projected_prefill_tokens,
        "cached_prefill_tokens": args.projected_cached_prefill_tokens,
        "sample_tokens": args.projected_sample_tokens,
        "train_tokens": args.projected_train_tokens,
        "steps": args.projected_steps,
        "train_tokens_per_step": args.projected_train_tokens_per_step,
        "samples": args.projected_samples,
        "max_prompt_tokens": args.projected_max_prompt_tokens,
        "max_output_tokens": args.projected_max_output_tokens,
    }
    if any(value is not None for value in direct.values()):
        projection.update({key: value for key, value in direct.items() if value is not None})
    if args.arm:
        projection["arm"] = args.arm
    if args.phase:
        projection["phase"] = args.phase
    if args.projected_cost_usd is not None:
        projection["cost_usd"] = args.projected_cost_usd
    if args.enforce_domain_coverage:
        projection["enforce_domain_coverage"] = True
    if args.claim_heldout:
        projection["claim_heldout"] = True
    if args.heldout_suite:
        existing = projection.get("heldout_suite_ids", projection.get("heldout_suites", []))
        if not isinstance(existing, list):
            raise BudgetError("projection held-out suite list must be a JSON list")
        projection["heldout_suite_ids"] = [*existing, *args.heldout_suite]
        projection["claim_heldout"] = True
    if not projection:
        return None
    return projection


def _display(value: Any) -> Any:
    if isinstance(value, Decimal):
        rounded = value.quantize(DISPLAY_QUANTUM, rounding=ROUND_HALF_UP)
        return float(rounded)
    if isinstance(value, Mapping):
        return {str(key): _display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_display(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="budget path (when --budget is omitted), then receipt paths")
    parser.add_argument("--budget", "--budget-json", dest="budget", type=Path)
    parser.add_argument(
        "--receipt",
        "--receipts",
        "--run-receipt",
        dest="receipts",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--portfolio-contract", "--portfolio", dest="portfolio_contract", type=Path, default=DEFAULT_PORTFOLIO_PATH)
    parser.add_argument("--projection", type=Path, help="JSON object containing projected portfolio/token reservations")
    parser.add_argument(
        "--projected-tokens-json",
        "--token-inputs",
        dest="projected_tokens_json",
        help="path or inline JSON object containing projected token/step inputs",
    )
    parser.add_argument("--projected-prefill-tokens", "--prefill-tokens", dest="projected_prefill_tokens", type=int)
    parser.add_argument("--projected-cached-prefill-tokens", "--cached-prefill-tokens", dest="projected_cached_prefill_tokens", type=int)
    parser.add_argument("--projected-sample-tokens", "--sample-tokens", dest="projected_sample_tokens", type=int)
    parser.add_argument("--projected-train-tokens", "--train-tokens", dest="projected_train_tokens", type=int)
    parser.add_argument("--projected-steps", "--steps", dest="projected_steps", type=int)
    parser.add_argument("--projected-train-tokens-per-step", "--train-tokens-per-step", dest="projected_train_tokens_per_step", type=int)
    parser.add_argument("--projected-samples", "--samples", dest="projected_samples", type=int)
    parser.add_argument("--projected-max-prompt-tokens", "--max-prompt-tokens", dest="projected_max_prompt_tokens", type=int)
    parser.add_argument("--projected-max-output-tokens", "--max-output-tokens", dest="projected_max_output_tokens", type=int)
    parser.add_argument("--projected-cost-usd", "--cost-usd", dest="projected_cost_usd", type=Decimal)
    parser.add_argument("--arm", help="planned arm name for the proposed projection")
    parser.add_argument("--phase", help="portfolio phase for the proposed projection")
    parser.add_argument("--suite-cost", action="append", default=[], metavar="SUITE=USD")
    parser.add_argument("--domain-cost", action="append", default=[], metavar="DOMAIN=USD")
    parser.add_argument("--suite-reservation", action="append", default=[], metavar="SUITE=USD")
    parser.add_argument("--domain-reservation", action="append", default=[], metavar="DOMAIN=USD")
    parser.add_argument("--phase-reservation", action="append", default=[], metavar="PHASE=USD")
    parser.add_argument("--enforce-domain-coverage", action="store_true")
    parser.add_argument(
        "--claim-heldout",
        "--request-heldout-claim",
        dest="claim_heldout",
        action="store_true",
        help="request a held-out claim; immutable evidence receipts are mandatory",
    )
    parser.add_argument(
        "--heldout-suite",
        action="append",
        default=[],
        metavar="SUITE",
        help="primary_eval suite (or E1-E14 label) covered by a held-out claim",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.budget is None:
            if not args.paths:
                budget_path = DEFAULT_BUDGET_PATH
                positional_receipts: list[Path] = []
            else:
                budget_path = args.paths[0]
                positional_receipts = args.paths[1:]
        else:
            budget_path = args.budget
            positional_receipts = args.paths
        budget = load_budget(budget_path)
        receipt_paths = list(args.receipts) + list(positional_receipts)
        portfolio = load_portfolio_contract(args.portfolio_contract)
        receipts = [load_receipt(path, budget, portfolio=portfolio) for path in receipt_paths]
        projection = _parse_cli_projection(args, budget)
        ledger = build_ledger(
            budget,
            receipts,
            projection,
            portfolio_contract=portfolio,
            receipt_sources=[str(path) for path in receipt_paths],
        )
        report = guard_launch(ledger)
        print(json.dumps(_display(report), indent=2, sort_keys=True))
        return 0 if report["allowed"] else 1
    except BudgetError as exc:
        error = {"schema_version": "pavlov-tinker-budget-ledger-v1", "status": "ERROR", "allowed": False, "errors": [str(exc)]}
        print(json.dumps(error, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
