#!/usr/bin/env python3
"""Offline reconciliation of Tinker billing exports with the Pavlov ledger.

The reconciler is deliberately independent of the Tinker service.  It accepts
already-exported JSON rows and a local Decimal ledger, matches every row to an
immutable run identifier, and replaces estimates with settled billing exactly
once.  Pending, rejected-run allowances, base-evaluation estimates, and
component line items remain visible in separate fields.

Any missing/ambiguous run identity, unknown billing run, conflicting amount,
or explicitly lagging export raises :class:`BillingReconciliationError`.
Cap decisions are returned as a report so callers can inspect the exact
remaining authorized amount before starting another paid action.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


try:  # Package import used by the flagship tests.
    from .pavlov_tinker_budget import (
        DEFAULT_BUDGET_PATH,
        HARD_CAP as CONTRACT_HARD_CAP,
        OPERATIONAL_CAP as CONTRACT_OPERATIONAL_CAP,
        SAFETY_RESERVE as CONTRACT_SAFETY_RESERVE,
        load_budget,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    from pavlov_tinker_budget import (  # type: ignore[no-redef]
        DEFAULT_BUDGET_PATH,
        HARD_CAP as CONTRACT_HARD_CAP,
        OPERATIONAL_CAP as CONTRACT_OPERATIONAL_CAP,
        SAFETY_RESERVE as CONTRACT_SAFETY_RESERVE,
        load_budget,
    )


HARD_CAP = Decimal("18.00")
OPERATIONAL_CAP = Decimal("16.50")
SAFETY_RESERVE = Decimal("1.50")
DISPLAY_QUANTUM = Decimal("0.000001")

if (CONTRACT_HARD_CAP, CONTRACT_OPERATIONAL_CAP, CONTRACT_SAFETY_RESERVE) != (
    HARD_CAP,
    OPERATIONAL_CAP,
    SAFETY_RESERVE,
):  # pragma: no cover - protects drift between the two owned interfaces.
    raise RuntimeError("Pavlov billing reconciler cap constants drifted from the ledger contract")


TERMINAL_STATUSES = {"completed", "complete", "succeeded", "success", "done", "billed", "paid", "settled"}
PENDING_STATUSES = {"pending", "running", "active", "queued", "launched", "estimated", "in_progress"}
REJECTED_STATUSES = {"rejected", "reject", "failed", "failure", "cancelled", "canceled", "aborted"}

RUN_ID_KEYS = (
    "run_id",
    "tinker_run_id",
    "training_run_id",
    "receipt_id",
    "job_id",
    "id",
)
RUN_NESTED_KEYS = ("run", "receipt", "job")
PRICE_MODEL_KEYS = (
    "pricing_model",
    "price_model",
    "billing_model",
    "model",
    "model_id",
    "tokenizer_model",
)

LIVE_AMOUNT_KEYS = (
    "live_billed_usd",
    "live_billed_cost_usd",
    "billed_cost_usd",
    "actual_billed_usd",
    "actual_billed_cost_usd",
    "actual_cost_usd",
    "provider_billed_cost_usd",
    "charged_cost_usd",
    "base_eval_billed_usd",
    "base_eval_actual_usd",
    "base_eval_cost_usd",
    "amount_usd",
    "cost_usd",
    "total_usd",
    "charge_usd",
    "amount",
    "cost",
    "total",
)
PENDING_AMOUNT_KEYS = (
    "pending_billing_usd",
    "pending_cost_usd",
    "conservative_pending_usd",
    "estimated_cost_usd",
    "estimated_cost",
    "base_eval_estimated_usd",
    "base_eval_unreconciled_usd",
) + LIVE_AMOUNT_KEYS
REJECTED_AMOUNT_KEYS = (
    "rejected_unreconciled_usd",
    "rejected_run_allowance_usd",
    "rejected_cost_usd",
    "maximum_authorized_cost_usd",
) + LIVE_AMOUNT_KEYS


class BillingReconciliationError(ValueError):
    """A billing export or local ledger cannot be reconciled safely."""


@dataclass(frozen=True)
class _LedgerRun:
    run_id: str
    status: str
    live: Decimal
    pending: Decimal
    base_eval_estimated: Decimal
    base_eval_unreconciled: Decimal
    rejected: Decimal
    components: dict[str, Decimal]
    terminal_at: datetime | None
    billing_exempt: bool

    @property
    def conservative_total(self) -> Decimal:
        return self.live + self.pending + self.base_eval_unreconciled + self.rejected

    @property
    def estimate_total(self) -> Decimal:
        return self.pending + self.base_eval_unreconciled + self.rejected


@dataclass
class _BillingLine:
    run_id: str
    state: str
    component: str
    amount: Decimal
    row_index: int
    row_id: str | None
    billed_at: datetime | None
    lagging: bool


@dataclass
class _BillingRun:
    run_id: str
    live: Decimal = Decimal("0")
    pending: Decimal = Decimal("0")
    rejected: Decimal = Decimal("0")
    has_live: bool = False
    has_pending: bool = False
    has_rejected: bool = False
    components: dict[str, Decimal] = field(default_factory=dict)
    components_by_state: dict[str, dict[str, Decimal]] = field(
        default_factory=lambda: {"live": {}, "pending": {}, "rejected": {}}
    )
    row_count: int = 0
    lagging: bool = False
    latest_billed_at: datetime | None = None

    @property
    def raw_total(self) -> Decimal:
        return self.live + self.pending + self.rejected


def _decimal(value: Any, field_name: str, *, allow_zero: bool = True) -> Decimal:
    if value is None or isinstance(value, bool):
        raise BillingReconciliationError(f"{field_name} must be a finite non-negative number")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BillingReconciliationError(
            f"{field_name} must be a finite non-negative number"
        ) from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise BillingReconciliationError(f"{field_name} must be a finite non-negative number")
    return parsed


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1", "yes", "y"}:
            return True
        if normalised in {"false", "0", "no", "n", ""}:
            return False
    raise BillingReconciliationError(f"{field_name} must be boolean")


def _first_non_none(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _model_value(mapping: Mapping[str, Any], field_name: str) -> str | None:
    """Return one unambiguous pricing-model identity from a record."""

    values: list[str] = []
    for key in PRICE_MODEL_KEYS:
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            if not isinstance(value, str) or not value.strip():
                raise BillingReconciliationError(f"{field_name}.{key} must be a non-empty string")
            values.append(value.strip())
    for section in ("billing", "cost", "accounting", "projection", "pricing"):
        nested = mapping.get(section)
        if isinstance(nested, Mapping):
            for key in PRICE_MODEL_KEYS:
                if key in nested and nested[key] is not None:
                    value = nested[key]
                    if not isinstance(value, str) or not value.strip():
                        raise BillingReconciliationError(
                            f"{field_name}.{section}.{key} must be a non-empty string"
                        )
                    values.append(value.strip())
    if not values:
        return None
    if len(set(values)) != 1:
        raise BillingReconciliationError(f"conflicting {field_name} price-model aliases")
    return values[0]


def _contract_price_model() -> str | None:
    """Read the local Pavlov contract model when one is available.

    This is intentionally a local read only.  A billing row that names a model
    must match the same contract model (or an explicit model on the ledger),
    otherwise the reconciler cannot safely apply its amount.
    """

    try:
        budget = load_budget(DEFAULT_BUDGET_PATH)
    except Exception:  # pragma: no cover - a copied standalone tool may omit the contract.
        return None
    model = budget.get("model") if isinstance(budget, Mapping) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _money_values(
    mapping: Mapping[str, Any], aliases: Sequence[str], field_name: str
) -> list[Decimal]:
    values: list[Decimal] = []
    for key in aliases:
        if key in mapping:
            values.append(_decimal(mapping[key], f"{field_name}.{key}"))
    for section in ("billing", "cost", "accounting", "projection"):
        nested = mapping.get(section)
        if isinstance(nested, Mapping):
            for key in aliases:
                if key in nested:
                    values.append(_decimal(nested[key], f"{field_name}.{section}.{key}"))
    return values


def _optional_money(
    mapping: Mapping[str, Any], aliases: Sequence[str], field_name: str
) -> Decimal | None:
    values = _money_values(mapping, aliases, field_name)
    if not values:
        return None
    if len(set(values)) != 1:
        raise BillingReconciliationError(f"conflicting {field_name} aliases")
    return values[0]


def _money(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
    field_name: str,
    *,
    default: Decimal = Decimal("0"),
) -> Decimal:
    value = _optional_money(mapping, aliases, field_name)
    return default if value is None else value


def _run_id(mapping: Mapping[str, Any], field_name: str) -> str:
    candidates: list[str] = []
    present_keys = [key for key in RUN_ID_KEYS if key in mapping]
    for key in present_keys:
        value = mapping[key]
        if not isinstance(value, str) or not value.strip():
            raise BillingReconciliationError(f"{field_name} has an invalid {key}")
        candidates.append(value.strip())
    for nested_key in RUN_NESTED_KEYS:
        nested = mapping.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        nested_values = [nested[key] for key in RUN_ID_KEYS if key in nested]
        for value in nested_values:
            if not isinstance(value, str) or not value.strip():
                raise BillingReconciliationError(f"{field_name} has an invalid nested run ID")
            candidates.append(value.strip())
    if not candidates:
        raise BillingReconciliationError(f"{field_name} is missing a run ID")
    if len(set(candidates)) != 1:
        raise BillingReconciliationError(
            f"{field_name} has ambiguous run IDs: {sorted(set(candidates))}"
        )
    return candidates[0]


def _status(mapping: Mapping[str, Any], *, component: str | None = None) -> str:
    raw = _first_non_none(mapping, ("billing_status", "status", "run_status", "state"))
    if raw is not None:
        if not isinstance(raw, str) or not raw.strip():
            raise BillingReconciliationError("status must be a non-empty string")
        normalised = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if normalised in TERMINAL_STATUSES:
            return "live"
        if normalised in PENDING_STATUSES:
            return "pending"
        if normalised in REJECTED_STATUSES:
            return "rejected"
        raise BillingReconciliationError(f"unknown billing status {raw!r}")
    if component in {"pending", "base_eval_pending"}:
        return "pending"
    if component in {"rejected", "rejected_unreconciled"}:
        return "rejected"
    if any(
        _money_values(
            mapping,
            (
                "rejected_unreconciled_usd",
                "rejected_run_allowance_usd",
                "rejected_cost_usd",
                "maximum_authorized_cost_usd",
            ),
            "rejected_cost",
        )
    ):
        return "rejected"
    if any(_money_values(mapping, LIVE_AMOUNT_KEYS, "live_cost")):
        return "live"
    if _money_values(
        mapping,
        (
            "pending_billing_usd",
            "pending_cost_usd",
            "conservative_pending_usd",
            "estimated_cost_usd",
            "estimated_cost",
            "base_eval_estimated_usd",
            "base_eval_unreconciled_usd",
        ),
        "pending_cost",
    ):
        return "pending"
    return "live"


def _component(value: Any, field_name: str) -> str:
    if value is None:
        return "unclassified"
    if not isinstance(value, str) or not value.strip():
        raise BillingReconciliationError(f"{field_name} must be a non-empty string")
    normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "billed": "live",
        "actual": "live",
        "charge": "live",
        "live_billed": "live",
        "pending_billing": "pending",
        "pending_billing_usd": "pending",
        "pending_cost": "pending",
        "estimate": "pending",
        "estimated": "pending",
        "base_eval_unreconciled_usd": "base_eval",
        "base_eval_estimated_usd": "base_eval",
        "base": "base_eval",
        "base_evaluation": "base_eval",
        "evaluation": "base_eval",
        "base_eval_estimated": "base_eval",
        "rejected_run_allowance": "rejected",
        "rejected_unreconciled_usd": "rejected",
        "rejected_cost": "rejected",
        "allowance": "rejected",
        "training": "train",
        "input": "prefill",
        "output": "sample",
    }
    return aliases.get(normalised, normalised)


def _timestamp(value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BillingReconciliationError(f"{field_name} must be an ISO timestamp")
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BillingReconciliationError(f"{field_name} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _flag(mapping: Mapping[str, Any], keys: Sequence[str], field_name: str) -> bool:
    items = [(key, mapping[key]) for key in keys if key in mapping]
    if not items:
        return False
    parsed = [_bool(value, f"{field_name}.{key}") for key, value in items]
    if len(set(parsed)) != 1:
        raise BillingReconciliationError(f"conflicting {field_name} flags")
    return parsed[0]


def _component_mapping(mapping: Mapping[str, Any], field_name: str) -> dict[str, Decimal]:
    component_containers = [
        (key, mapping[key])
        for key in ("component_costs", "components")
        if key in mapping and mapping[key] is not None
    ]
    if len(component_containers) > 1 and any(
        value != component_containers[0][1] for _, value in component_containers[1:]
    ):
        raise BillingReconciliationError(f"{field_name} has conflicting component containers")
    raw = component_containers[0][1] if component_containers else None
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise BillingReconciliationError(f"{field_name}.components must be an object")
    result: dict[str, Decimal] = {}
    for name, value in raw.items():
        component = _component(name, f"{field_name}.components")
        amount = _decimal(value, f"{field_name}.components.{name}")
        if component in result:
            raise BillingReconciliationError(
                f"{field_name}.components contains duplicate component {name!r}"
            )
        result[component] = amount
    return result


def _add(target: dict[str, Decimal], key: str, amount: Decimal) -> None:
    target[key] = target.get(key, Decimal("0")) + amount


def _normalise_local_ledger(
    ledger: Mapping[str, Any],
) -> tuple[list[_LedgerRun], Decimal, dict[str, Decimal], dict[str, Any], set[str]]:
    if not isinstance(ledger, Mapping):
        raise BillingReconciliationError("local ledger root must be an object")
    root = ledger.get("ledger") if isinstance(ledger.get("ledger"), Mapping) else ledger
    local_model = _model_value(root, "ledger")
    allowed_models: set[str] = set()
    if local_model is not None:
        allowed_models.add(local_model)
    model_allowlist = root.get("known_price_models", root.get("allowed_price_models"))
    if model_allowlist is not None:
        if not isinstance(model_allowlist, (list, tuple, set)):
            raise BillingReconciliationError("ledger.known_price_models must be a list")
        for index, value in enumerate(model_allowlist):
            if not isinstance(value, str) or not value.strip():
                raise BillingReconciliationError(
                    f"ledger.known_price_models[{index}] must be a non-empty string"
                )
            allowed_models.add(value.strip())
    if not allowed_models:
        contract_model = _contract_price_model()
        if contract_model is not None:
            allowed_models.add(contract_model)
    receipt_containers = [
        (key, root[key])
        for key in ("receipts", "receipt_rows", "runs")
        if key in root and root[key] is not None
    ]
    if len(receipt_containers) > 1 and any(
        value != receipt_containers[0][1] for _, value in receipt_containers[1:]
    ):
        raise BillingReconciliationError(
            "local ledger contains conflicting receipt containers"
        )
    receipts_value = receipt_containers[0][1] if receipt_containers else None
    if receipts_value is None:
        receipts: list[Any] = []
    elif isinstance(receipts_value, list):
        receipts = receipts_value
    else:
        raise BillingReconciliationError("local ledger receipts must be a list")

    runs: list[_LedgerRun] = []
    seen: set[str] = set()
    local_components: dict[str, Decimal] = {}
    for index, raw in enumerate(receipts):
        if not isinstance(raw, Mapping):
            raise BillingReconciliationError(f"ledger.receipts[{index}] must be an object")
        run_id = _run_id(raw, f"ledger.receipts[{index}]")
        receipt_model = _model_value(raw, f"ledger.receipts[{index}]")
        if receipt_model is not None and allowed_models and receipt_model not in allowed_models:
            raise BillingReconciliationError(
                f"unknown price model {receipt_model!r} in ledger.receipts[{index}]"
            )
        if run_id in seen:
            raise BillingReconciliationError(f"duplicate ledger run ID {run_id!r}")
        seen.add(run_id)
        status = _status(raw)
        live = _money(
            raw,
            ("live_billed_usd", "live_billed_cost_usd", "known_usd", "billed_cost_usd"),
            f"ledger.receipts[{index}].live",
        )
        base_estimated = _money(raw, ("base_eval_estimated_usd",), f"ledger.receipts[{index}].base_eval_estimated")
        base_unreconciled_value = _optional_money(
            raw,
            ("base_eval_unreconciled_usd",),
            f"ledger.receipts[{index}].base_eval_unreconciled",
        )
        base_unreconciled = (
            base_unreconciled_value
            if base_unreconciled_value is not None
            else (base_estimated if live == 0 else Decimal("0"))
        )
        rejected = _money(
            raw,
            ("rejected_unreconciled_usd", "rejected_run_allowance_usd", "rejected_cost_usd"),
            f"ledger.receipts[{index}].rejected",
        )
        pending_value = _optional_money(
            raw,
            ("pending_billing_usd",),
            f"ledger.receipts[{index}].pending_billing",
        )
        if pending_value is None:
            pending_total = _optional_money(
                raw, ("pending_usd", "conservative_pending_usd"), f"ledger.receipts[{index}].pending"
            )
            if pending_total is None:
                pending = Decimal("0")
            else:
                pending = pending_total - base_unreconciled - rejected
                if pending < 0:
                    raise BillingReconciliationError(
                        f"ledger.receipts[{index}] pending total is below its base/rejected components"
                    )
        else:
            pending = pending_value
        components = _component_mapping(raw, f"ledger.receipts[{index}]")
        if not components:
            for name, amount in (
                ("live", live),
                ("pending", pending),
                ("base_eval", base_unreconciled),
                ("rejected", rejected),
            ):
                if amount:
                    components[name] = amount
        for name, amount in components.items():
            _add(local_components, name, amount)
        terminal_at = _timestamp(
            _first_non_none(raw, ("completed_at", "finished_at", "ended_at", "updated_at")),
            f"ledger.receipts[{index}].completed_at",
        )
        exempt = _flag(
            raw,
            ("billing_exempt", "billing_not_required"),
            f"ledger.receipts[{index}].billing_exempt",
        )
        runs.append(
            _LedgerRun(
                run_id=run_id,
                status=status,
                live=live,
                pending=pending,
                base_eval_estimated=base_estimated,
                base_eval_unreconciled=base_unreconciled,
                rejected=rejected,
                components=components,
                terminal_at=terminal_at,
                billing_exempt=exempt,
            )
        )

    sums = {
        "live": sum((run.live for run in runs), Decimal("0")),
        "pending": sum((run.pending for run in runs), Decimal("0")),
        "base_eval_unreconciled": sum(
            (run.base_eval_unreconciled for run in runs), Decimal("0")
        ),
        "base_eval_estimated": sum(
            (run.base_eval_estimated for run in runs), Decimal("0")
        ),
        "rejected": sum((run.rejected for run in runs), Decimal("0")),
    }
    aggregate_aliases = {
        "live": ("live_billed_usd", "known_usd"),
        "pending": ("pending_billing_usd",),
        "base_eval_unreconciled": ("base_eval_unreconciled_usd",),
        "base_eval_estimated": ("base_eval_estimated_usd",),
        "rejected": ("rejected_unreconciled_usd", "rejected_run_allowance_usd"),
    }
    for name, aliases in aggregate_aliases.items():
        supplied = _optional_money(root, aliases, f"ledger.{name}")
        if supplied is not None and supplied != sums[name]:
            raise BillingReconciliationError(
                f"ledger {name} aggregate {supplied} does not reconcile to receipts {sums[name]}"
            )
    supplied_conservative = _optional_money(
        root, ("conservative_pending_usd",), "ledger.conservative_pending"
    )
    conservative = sums["pending"] + sums["base_eval_unreconciled"] + sums["rejected"]
    if supplied_conservative is not None and supplied_conservative != conservative:
        raise BillingReconciliationError(
            f"ledger conservative pending aggregate {supplied_conservative} does not reconcile to receipts {conservative}"
        )
    current = sums["live"] + conservative
    supplied_proposed = _optional_money(
        root,
        ("proposed_arm_usd", "proposed_cost_usd", "direct_projection_usd"),
        "ledger.proposed",
    )
    supplied_projected = _optional_money(
        root, ("projected_usage_usd", "projected_usd", "projected"), "ledger.projected"
    )
    if supplied_proposed is None:
        proposed = (
            supplied_projected - current if supplied_projected is not None else Decimal("0")
        )
        if proposed < 0:
            raise BillingReconciliationError("ledger projected total is below receipt usage")
    else:
        proposed = supplied_proposed
    if supplied_projected is not None and supplied_projected != current + proposed:
        raise BillingReconciliationError(
            f"ledger projected aggregate {supplied_projected} does not reconcile to receipts plus proposal {current + proposed}"
        )
    if not runs and (current or proposed) and receipts_value is None:
        # An aggregate-only ledger cannot be joined to billing rows safely.
        if current:
            raise BillingReconciliationError(
                "local ledger has spend aggregates but no run receipts"
            )
    return runs, proposed, local_components, sums, allowed_models


def _normalise_export(export: Any) -> tuple[list[Mapping[str, Any]], datetime | None, Mapping[str, Any]]:
    if isinstance(export, list):
        return export, None, {}
    if not isinstance(export, Mapping):
        raise BillingReconciliationError("billing export root must be an object or list")
    row_containers = [
        (key, export[key])
        for key in ("rows", "billing_rows", "line_items", "charges", "export")
        if key in export and export[key] is not None
    ]
    if len(row_containers) > 1 and any(
        value != row_containers[0][1] for _, value in row_containers[1:]
    ):
        raise BillingReconciliationError("billing export contains conflicting row containers")
    rows_value = row_containers[0][1] if row_containers else None
    if rows_value is None and any(key in export for key in RUN_ID_KEYS):
        rows_value = [export]
    if not isinstance(rows_value, list):
        raise BillingReconciliationError("billing export rows must be a list")
    as_of = _timestamp(
        _first_non_none(export, ("as_of", "exported_at", "retrieved_at", "billing_as_of")),
        "billing_export.as_of",
    )
    root_status = _first_non_none(export, ("billing_status", "status", "state"))
    if isinstance(root_status, str) and root_status.strip().lower().replace("-", "_") in PENDING_STATUSES:
        raise BillingReconciliationError("billing export is explicitly lagging")
    if _flag(export, ("billing_lagging", "lagging"), "billing_export.lagging") or _row_lagging(
        export, "billing_export"
    ):
        raise BillingReconciliationError("billing export is explicitly lagging")
    return rows_value, as_of, export


def _billing_amount(row: Mapping[str, Any], state: str, field_name: str) -> Decimal:
    if state == "pending":
        aliases = PENDING_AMOUNT_KEYS
    elif state == "rejected":
        aliases = REJECTED_AMOUNT_KEYS
    else:
        aliases = LIVE_AMOUNT_KEYS
    amount = _optional_money(row, aliases, field_name)
    if amount is None:
        raise BillingReconciliationError(f"{field_name} is missing a billing amount")
    return amount


def _component_row_total(
    row: Mapping[str, Any], field_name: str, state: str | None = None
) -> Decimal | None:
    """Read one unambiguous scalar total when a row also carries components."""

    if state is not None:
        aliases = {
            "live": LIVE_AMOUNT_KEYS,
            "pending": PENDING_AMOUNT_KEYS,
            "rejected": REJECTED_AMOUNT_KEYS,
        }[state]
        return _optional_money(row, aliases, f"{field_name}.total")
    candidates: list[Decimal] = []
    for aliases in (LIVE_AMOUNT_KEYS, PENDING_AMOUNT_KEYS, REJECTED_AMOUNT_KEYS):
        value = _optional_money(row, aliases, f"{field_name}.total")
        if value is not None and value not in candidates:
            candidates.append(value)
    if len(candidates) > 1:
        raise BillingReconciliationError(f"conflicting {field_name}.total aliases")
    return candidates[0] if candidates else None


def _row_id(row: Mapping[str, Any], field_name: str) -> str | None:
    values: list[str] = []
    for key in ("billing_row_id", "line_item_id", "entry_id", "charge_id"):
        if key in row:
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise BillingReconciliationError(f"{field_name}.{key} must be non-empty")
            values.append(value.strip())
    if not values:
        return None
    if len(set(values)) != 1:
        raise BillingReconciliationError(f"{field_name} has ambiguous billing row IDs")
    return values[0]


def _row_lagging(row: Mapping[str, Any], field_name: str) -> bool:
    flags = []
    for key in ("billing_lagging", "lagging"):
        if key in row:
            flags.append(_bool(row[key], f"{field_name}.{key}"))
    if "billing_complete" in row and not _bool(row["billing_complete"], f"{field_name}.billing_complete"):
        flags.append(True)
    if "is_final" in row and not _bool(row["is_final"], f"{field_name}.is_final"):
        flags.append(True)
    if "settled" in row and not _bool(row["settled"], f"{field_name}.settled"):
        flags.append(True)
    return any(flags)


def _billing_lines(export: Any) -> tuple[list[_BillingLine], datetime | None, Mapping[str, Any]]:
    rows, as_of, metadata = _normalise_export(export)
    export_model = _model_value(metadata, "billing_export")
    row_models: set[str] = set()
    lines: list[_BillingLine] = []
    seen_row_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise BillingReconciliationError(f"billing_rows[{index}] must be an object")
        field_name = f"billing_rows[{index}]"
        run_id = _run_id(raw, field_name)
        row_model = _model_value(raw, field_name)
        if row_model is not None:
            row_models.add(row_model)
            if export_model is not None and row_model != export_model:
                raise BillingReconciliationError(
                    f"billing row {index} price model {row_model!r} disagrees with export {export_model!r}"
                )
        row_id = _row_id(raw, field_name)
        if row_id is not None:
            if row_id in seen_row_ids:
                raise BillingReconciliationError(f"duplicate billing row ID {row_id!r}")
            seen_row_ids.add(row_id)
        billed_at = _timestamp(
            _first_non_none(raw, ("billed_at", "charged_at", "created_at", "timestamp")),
            f"{field_name}.billed_at",
        )
        if as_of is not None and billed_at is not None and billed_at > as_of:
            raise BillingReconciliationError(
                f"{field_name}.billed_at {billed_at.isoformat()} is after billing export as_of {as_of.isoformat()}"
            )
        lagging = _row_lagging(raw, field_name)
        component_value = _first_non_none(raw, ("component", "cost_component", "category"))
        component_hint = (
            _component(component_value, f"{field_name}.component")
            if component_value is not None
            else None
        )
        state_hint = _status(raw, component=component_hint)
        component_containers = [
            (key, raw[key])
            for key in ("component_costs", "components")
            if key in raw and raw[key] is not None
        ]
        if len(component_containers) > 1 and any(
            value != component_containers[0][1] for _, value in component_containers[1:]
        ):
            raise BillingReconciliationError(
                f"{field_name} has conflicting component containers"
            )
        raw_components = component_containers[0][1] if component_containers else None
        if raw_components is not None:
            if not isinstance(raw_components, Mapping) or not raw_components:
                raise BillingReconciliationError(f"{field_name}.components must be a non-empty object")
            component_amounts: dict[str, Decimal] = {}
            for name, value in raw_components.items():
                component = _component(name, f"{field_name}.components")
                if component in component_amounts:
                    raise BillingReconciliationError(
                        f"{field_name}.components contains duplicate component {name!r}"
                    )
                component_amounts[component] = _decimal(
                    value, f"{field_name}.components.{name}"
                )
            total = _component_row_total(raw, field_name, state_hint)
            if total is not None and total != sum(component_amounts.values(), Decimal("0")):
                raise BillingReconciliationError(
                    f"{field_name} total does not reconcile to component costs"
                )
            for name, amount in component_amounts.items():
                state = state_hint if component_value is not None else _status(raw, component=name)
                lines.append(
                    _BillingLine(
                        run_id=run_id,
                        state=state,
                        component=name,
                        amount=amount,
                        row_index=index,
                        row_id=row_id,
                        billed_at=billed_at,
                        lagging=lagging,
                    )
                )
            continue
        component = component_hint or "unclassified"
        state = state_hint
        amount = _billing_amount(raw, state, field_name)
        lines.append(
            _BillingLine(
                run_id=run_id,
                state=state,
                component=component,
                amount=amount,
                row_index=index,
                row_id=row_id,
                billed_at=billed_at,
                lagging=lagging,
            )
        )
    if len(row_models) > 1:
        raise BillingReconciliationError(
            f"billing export contains multiple price models: {sorted(row_models)}"
        )
    if export_model is not None or row_models:
        metadata = dict(metadata)
        metadata["_reconciler_price_model"] = export_model or next(iter(row_models))
    return lines, as_of, metadata


def _aggregate_billing(lines: Iterable[_BillingLine]) -> dict[str, _BillingRun]:
    runs: dict[str, _BillingRun] = {}
    seen_components: dict[tuple[str, str], list[str | None]] = {}
    for line in lines:
        key = (line.run_id, line.component)
        prior_ids = seen_components.setdefault(key, [])
        if prior_ids and (line.row_id is None or any(value is None for value in prior_ids)):
            raise BillingReconciliationError(
                f"ambiguous duplicate billing component {line.component!r} for run {line.run_id!r}"
            )
        prior_ids.append(line.row_id)
        run = runs.setdefault(line.run_id, _BillingRun(run_id=line.run_id))
        run.row_count += 1
        run.lagging = run.lagging or line.lagging
        if line.billed_at is not None and (
            run.latest_billed_at is None or line.billed_at > run.latest_billed_at
        ):
            run.latest_billed_at = line.billed_at
        _add(run.components, line.component, line.amount)
        _add(run.components_by_state[line.state], line.component, line.amount)
        if line.state == "live":
            run.has_live = True
            run.live += line.amount
        elif line.state == "pending":
            run.has_pending = True
            run.pending += line.amount
        elif line.state == "rejected":
            run.has_rejected = True
            run.rejected += line.amount
        else:  # pragma: no cover - _status validates this invariant.
            raise BillingReconciliationError(f"unknown billing state {line.state!r}")
    for run in runs.values():
        if "total" in run.components and len(run.components) > 1:
            raise BillingReconciliationError(
                f"billing run {run.run_id!r} mixes a total row with component rows"
            )
    return runs


@dataclass(frozen=True)
class _EffectiveRun:
    run_id: str
    live: Decimal
    pending: Decimal
    base_eval_unreconciled: Decimal
    rejected: Decimal
    components: dict[str, Decimal]
    billing_state: str
    billing_rows: int
    cleared_estimate: Decimal
    superseded_pending: Decimal

    @property
    def total(self) -> Decimal:
        return self.live + self.pending + self.base_eval_unreconciled + self.rejected


def _effective_run(local: _LedgerRun, billing: _BillingRun | None) -> _EffectiveRun:
    if billing is None:
        if (
            not local.billing_exempt
            and local.status == "live"
            and (local.live or local.estimate_total or local.terminal_at is not None)
        ):
            raise BillingReconciliationError(
                f"billing export is lagging for completed run {local.run_id!r}"
            )
        return _EffectiveRun(
            run_id=local.run_id,
            live=local.live,
            pending=local.pending,
            base_eval_unreconciled=local.base_eval_unreconciled,
            rejected=local.rejected,
            components=dict(local.components),
            billing_state="missing_pending" if local.status == "pending" else "local_only",
            billing_rows=0,
            cleared_estimate=Decimal("0"),
            superseded_pending=Decimal("0"),
        )
    if billing.lagging:
        raise BillingReconciliationError(
            f"billing export is explicitly lagging for run {local.run_id!r}"
        )
    if billing.has_live:
        if local.live and local.live != billing.live:
            raise BillingReconciliationError(
                f"live billing mismatch for run {local.run_id!r}: local={local.live} export={billing.live}"
            )
        cleared = local.estimate_total
        return _EffectiveRun(
            run_id=local.run_id,
            live=billing.live,
            pending=Decimal("0"),
            base_eval_unreconciled=Decimal("0"),
            rejected=Decimal("0"),
            components=dict(billing.components_by_state["live"]),
            billing_state="live",
            billing_rows=billing.row_count,
            cleared_estimate=cleared,
            superseded_pending=billing.pending + billing.rejected,
        )
    if billing.has_rejected:
        if local.live:
            raise BillingReconciliationError(
                f"rejected billing conflicts with local live billing for run {local.run_id!r}"
            )
        if local.rejected and local.rejected != billing.rejected:
            raise BillingReconciliationError(
                f"rejected allowance mismatch for run {local.run_id!r}: local={local.rejected} export={billing.rejected}"
            )
        return _EffectiveRun(
            run_id=local.run_id,
            live=Decimal("0"),
            pending=Decimal("0"),
            base_eval_unreconciled=Decimal("0"),
            rejected=billing.rejected,
            components=dict(billing.components_by_state["rejected"]),
            billing_state="rejected",
            billing_rows=billing.row_count,
            cleared_estimate=local.pending + local.base_eval_unreconciled,
            superseded_pending=billing.pending,
        )
    if billing.has_pending:
        billing_base = sum(
            billing.components_by_state["pending"].get(name, Decimal("0"))
            for name in ("base_eval",)
        )
        billing_pending = billing.pending - billing_base
        if billing_pending < 0:
            raise BillingReconciliationError(
                f"pending billing components exceed pending total for run {local.run_id!r}"
            )
        if local.status == "live" and local.live:
            raise BillingReconciliationError(
                f"billing export is lagging for completed run {local.run_id!r}"
            )
        if local.status == "live" and billing.pending < local.estimate_total:
            raise BillingReconciliationError(
                f"billing export is lagging for completed run {local.run_id!r}"
            )
        pending = max(local.pending, billing_pending)
        base = max(local.base_eval_unreconciled, billing_base)
        components = dict(billing.components_by_state["pending"])
        if local.pending > billing_pending:
            _add(components, "pending", local.pending - billing_pending)
        if local.base_eval_unreconciled > billing_base:
            _add(components, "base_eval", local.base_eval_unreconciled - billing_base)
        if local.rejected:
            _add(components, "rejected", local.rejected)
        return _EffectiveRun(
            run_id=local.run_id,
            live=Decimal("0"),
            pending=pending,
            base_eval_unreconciled=base,
            rejected=local.rejected,
            components=components,
            billing_state="pending",
            billing_rows=billing.row_count,
            cleared_estimate=Decimal("0"),
            superseded_pending=Decimal("0"),
        )
    raise BillingReconciliationError(
        f"billing export has no billable state for run {local.run_id!r}"
    )


def _validate_caps(ledger: Mapping[str, Any]) -> None:
    if not isinstance(ledger, Mapping):
        raise BillingReconciliationError("local ledger root must be an object")
    root = ledger.get("ledger") if isinstance(ledger.get("ledger"), Mapping) else ledger
    for key, expected in (
        ("maximum_usd", HARD_CAP),
        ("hard_cap_usd", HARD_CAP),
        ("operational_cap_usd", OPERATIONAL_CAP),
        ("safety_reserve_usd", SAFETY_RESERVE),
    ):
        if key in root and _decimal(root[key], f"ledger.{key}") != expected:
            raise BillingReconciliationError(
                f"ledger {key} does not preserve the authorized {expected} boundary"
            )


def reconcile_billing_export(
    ledger: Mapping[str, Any], billing_export: Any
) -> dict[str, Any]:
    """Reconcile an offline billing export with a local Decimal ledger.

    Structural uncertainty raises ``BillingReconciliationError``.  A valid
    reconciliation that crosses a cap returns a ``REJECT`` report instead.
    """

    _validate_caps(ledger)
    local_runs, proposed, local_components, local_sums, allowed_models = _normalise_local_ledger(
        ledger
    )
    lines, export_as_of, metadata = _billing_lines(billing_export)
    billing_model = metadata.get("_reconciler_price_model")
    if billing_model is not None and (
        not allowed_models or billing_model not in allowed_models
    ):
        raise BillingReconciliationError(f"unknown price model {billing_model!r}")
    billing_runs = _aggregate_billing(lines)
    local_by_id = {run.run_id: run for run in local_runs}
    unknown = sorted(set(billing_runs) - set(local_by_id))
    if unknown:
        raise BillingReconciliationError(
            f"billing export contains unknown run IDs {unknown}"
        )
    if billing_runs and not local_runs:
        raise BillingReconciliationError("billing export cannot be reconciled without local run IDs")
    if export_as_of is not None:
        late_runs = [
            run.run_id
            for run in local_runs
            if run.terminal_at is not None and run.terminal_at > export_as_of
        ]
        if late_runs:
            raise BillingReconciliationError(
                f"billing export as_of {export_as_of.isoformat()} predates completed runs {sorted(late_runs)}"
            )
    effective: list[_EffectiveRun] = []
    for run in local_runs:
        effective.append(_effective_run(run, billing_runs.get(run.run_id)))

    live = sum((run.live for run in effective), Decimal("0"))
    pending = sum((run.pending for run in effective), Decimal("0"))
    base_unreconciled = sum(
        (run.base_eval_unreconciled for run in effective), Decimal("0")
    )
    rejected = sum((run.rejected for run in effective), Decimal("0"))
    base_estimated = sum(
        (run.base_eval_estimated for run in local_runs), Decimal("0")
    )
    projected = live + pending + base_unreconciled + rejected + proposed
    operational_remaining = OPERATIONAL_CAP - projected
    hard_remaining = HARD_CAP - projected
    remaining_authorized = max(Decimal("0"), operational_remaining)
    reserve_preserved = projected <= OPERATIONAL_CAP
    reasons: list[str] = []
    if projected > OPERATIONAL_CAP:
        reasons.append(
            f"projected reconciled usage ${projected} exceeds operational cap ${OPERATIONAL_CAP}"
        )
    if projected > HARD_CAP:
        reasons.append(
            f"projected reconciled usage ${projected} exceeds hard cap ${HARD_CAP}"
        )
    if remaining_authorized == 0:
        reasons.append("no positive remaining authorized amount for another paid action")

    billing_components: dict[str, Decimal] = {}
    reconciled_components: dict[str, Decimal] = {}
    for run in billing_runs.values():
        for name, amount in run.components.items():
            _add(billing_components, name, amount)
    for run in effective:
        for name, amount in run.components.items():
            _add(reconciled_components, name, amount)

    reconciled_components_total = sum(reconciled_components.values(), Decimal("0"))
    reconciled_usage_without_proposal = live + pending + base_unreconciled + rejected
    if reconciled_components_total != reconciled_usage_without_proposal:
        raise BillingReconciliationError(
            "reconciled component total does not reconcile to live/pending/base-eval/rejected usage"
        )

    billing_live = sum((run.live for run in billing_runs.values()), Decimal("0"))
    billing_pending = sum((run.pending for run in billing_runs.values()), Decimal("0"))
    billing_rejected = sum((run.rejected for run in billing_runs.values()), Decimal("0"))
    cleared_estimates = sum((run.cleared_estimate for run in effective), Decimal("0"))
    superseded_pending = sum((run.superseded_pending for run in effective), Decimal("0"))
    matched = sorted(set(billing_runs) & set(local_by_id))
    pending_ids = sorted(
        run.run_id for run in effective if run.pending or run.base_eval_unreconciled
    )
    rejected_ids = sorted(run.run_id for run in effective if run.rejected)
    report_status = "ALLOW" if not reasons else "REJECT"
    report_metadata = {
        key: value
        for key, value in metadata.items()
        if key in {"export_id", "source", "row_count", "provider"}
    }
    if billing_model is not None:
        report_metadata["price_model"] = billing_model
    run_reports = [
        {
            "run_id": run.run_id,
            "status": run.billing_state,
            "billing_rows": run.billing_rows,
            "local_live_billed_usd": local_by_id[run.run_id].live,
            "local_pending_billing_usd": local_by_id[run.run_id].pending,
            "local_base_eval_estimated_usd": local_by_id[run.run_id].base_eval_estimated,
            "local_base_eval_unreconciled_usd": local_by_id[run.run_id].base_eval_unreconciled,
            "local_rejected_unreconciled_usd": local_by_id[run.run_id].rejected,
            "reconciled_live_billed_usd": run.live,
            "reconciled_pending_billing_usd": run.pending,
            "reconciled_base_eval_unreconciled_usd": run.base_eval_unreconciled,
            "reconciled_rejected_unreconciled_usd": run.rejected,
            "reconciled_total_usd": run.total,
            "cleared_estimate_usd": run.cleared_estimate,
            "superseded_pending_usd": run.superseded_pending,
            "components_usd": dict(run.components),
        }
        for run in effective
    ]
    return {
        "schema_version": "pavlov-tinker-billing-reconcile-v1",
        "status": report_status,
        "decision": report_status,
        "allowed": report_status == "ALLOW",
        "another_paid_action_may_start": report_status == "ALLOW",
        "maximum_usd": HARD_CAP,
        "hard_cap_usd": HARD_CAP,
        "operational_cap_usd": OPERATIONAL_CAP,
        "safety_reserve_usd": SAFETY_RESERVE,
        "live_billed_usd": live,
        "known_usd": live,
        "pending_billing_usd": pending,
        "base_eval_estimated_usd": base_estimated,
        "base_eval_unreconciled_usd": base_unreconciled,
        "rejected_unreconciled_usd": rejected,
        "rejected_run_allowance_usd": rejected,
        "conservative_pending_usd": pending + base_unreconciled + rejected,
        "proposed_arm_usd": proposed,
        "projected_usage_usd": projected,
        "projected_usd": projected,
        "operational_remaining_usd": operational_remaining,
        "hard_remaining_usd": hard_remaining,
        "remaining_authorized_usd": remaining_authorized,
        "remaining_usd": remaining_authorized,
        "reserve_preserved": reserve_preserved,
        "billing_export_live_usd": billing_live,
        "billing_export_pending_usd": billing_pending,
        "billing_export_rejected_usd": billing_rejected,
        "billing_export_raw_total_usd": billing_live + billing_pending + billing_rejected,
        "cleared_estimate_usd": cleared_estimates,
        "superseded_pending_usd": superseded_pending,
        "matched_run_ids": matched,
        "pending_run_ids": pending_ids,
        "rejected_run_ids": rejected_ids,
        "billing_as_of": export_as_of.isoformat() if export_as_of else None,
        "components_usd": {
            "ledger_local": dict(local_components),
            "billing_export": billing_components,
            "reconciled": reconciled_components,
        },
        "component_reconciliation": {
            "ledger_local_total_usd": sum(local_components.values(), Decimal("0")),
            "billing_export_total_usd": sum(billing_components.values(), Decimal("0")),
            "reconciled_total_usd": reconciled_components_total,
            "reconciled_usage_without_proposal_usd": reconciled_usage_without_proposal,
            "reconciled": reconciled_components_total == reconciled_usage_without_proposal,
        },
        "run_reports": run_reports,
        "local_ledger_sums": dict(local_sums),
        "reconciliation": {
            "live_billed_usd": live,
            "pending_billing_usd": pending,
            "base_eval_unreconciled_usd": base_unreconciled,
            "rejected_unreconciled_usd": rejected,
            "proposed_arm_usd": proposed,
            "projected_usage_usd": projected,
        },
        "reasons": reasons,
        "metadata": report_metadata,
    }


reconcile_billing = reconcile_billing_export
reconcile = reconcile_billing_export
reconcile_ledger = reconcile_billing_export


def _json_load(path: Path, label: str) -> Any:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise BillingReconciliationError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BillingReconciliationError(f"{label} is malformed: {path}: {exc}") from exc


def _display(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value.quantize(DISPLAY_QUANTUM, rounding=ROUND_HALF_UP))
    if isinstance(value, Mapping):
        return {str(key): _display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_display(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="ledger path followed by billing export path")
    parser.add_argument("--ledger", "--ledger-json", dest="ledger", type=Path)
    parser.add_argument(
        "--billing-export",
        "--export",
        dest="billing_export",
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = list(args.paths)
        ledger_path = args.ledger or (paths[0] if paths else None)
        export_path = args.billing_export or (paths[1] if len(paths) > 1 else None)
        if ledger_path is None or export_path is None:
            raise BillingReconciliationError(
                "provide --ledger and --billing-export (or ledger and export positional paths)"
            )
        if len(paths) > 2:
            raise BillingReconciliationError("expected exactly ledger and billing export paths")
        ledger = _json_load(ledger_path, "local ledger")
        billing_export = _json_load(export_path, "billing export")
        report = reconcile_billing_export(ledger, billing_export)
        print(json.dumps(_display(report), indent=2, sort_keys=True))
        return 0 if report["another_paid_action_may_start"] else 1
    except BillingReconciliationError as exc:
        report = {
            "schema_version": "pavlov-tinker-billing-reconcile-v1",
            "status": "ERROR",
            "decision": "ERROR",
            "allowed": False,
            "another_paid_action_may_start": False,
            "maximum_usd": HARD_CAP,
            "hard_cap_usd": HARD_CAP,
            "operational_cap_usd": OPERATIONAL_CAP,
            "safety_reserve_usd": SAFETY_RESERVE,
            "remaining_authorized_usd": Decimal("0"),
            "remaining_usd": Decimal("0"),
            "operational_remaining_usd": Decimal("0"),
            "hard_remaining_usd": Decimal("0"),
            "errors": [str(exc)],
        }
        print(json.dumps(_display(report), indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
