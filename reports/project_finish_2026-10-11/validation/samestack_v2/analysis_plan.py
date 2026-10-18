"""Canonical, retrospective primary-analysis artifact (schema version 1).

No data-driven family substitution or automatic environment/protocol selection.
An explicit plan binds the cohort. Historical records without evidence remain
excluded, not repaired into apparently verified provenance. A repaired plan is
not evidence of prospective registration, even when its endpoint is complete.
"""

from copy import deepcopy
import hashlib
import json
import math
import re

try:
    from .stats_lib import holm, paired
except ImportError:
    from stats_lib import holm, paired

ARMS = ("grpo_g8", "grpo_g2", "ppo", "ppo_8x8")
SEEDS = (42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777)
PRIMARY = {
    "g8-ppo": ("grpo_g8", "ppo"),
    "g8-ppo_8x8": ("grpo_g8", "ppo_8x8"),
    "g8-g2": ("grpo_g8", "grpo_g2"),
}
EXPLORATORY = {"g2-ppo": ("grpo_g2", "ppo")}
IDENTITY_FIELDS = (
    "run_id",
    "environment",
    "environment_hash",
    "protocol_hash",
    "ordered_test_item_hash",
    "shared_stack_hash",
)
# These are intentional treatment/replicate differences, not shared stack fields.
ARM_SEED_CONFIG_FIELDS = frozenset(
    ("arm", "seed", "group", "prompts_per_step", "critic", "advantage_normalization")
)


def shared_stack_fingerprint(record):
    """Derive a common-stack fingerprint, or None for incomplete provenance.

    This identifies recorded evidence, not an independent attestation. Canonical
    selection also requires the shared run validator's verified result. Never
    accept a caller-supplied shared_stack_hash in place of this derivation.
    """
    if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
        return None
    manifest = record["manifest"]
    source, revisions = manifest.get("source"), manifest.get("revisions")
    config, generation = manifest.get("effective_config"), manifest.get("generation")
    is_hash = lambda value: (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )
    if not all(
        isinstance(value, dict) and value for value in (source, revisions, config, generation)
    ):
        return None
    if (
        source.get("status") != "observed"
        or type(source.get("dirty")) is not bool
        or not isinstance(source.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40,64}", source["commit"]) is None
        or not is_hash(source.get("dirty_hash"))
    ):
        return None
    files = source.get("source_files")
    if (
        not isinstance(files, dict)
        or not files
        or not all(
            isinstance(path, str) and path and is_hash(digest) for path, digest in files.items()
        )
    ):
        return None
    for component in ("model", "tokenizer", "dataset"):
        revision = revisions.get(component)
        if (
            not isinstance(revision, dict)
            or not isinstance(revision.get("revision"), str)
            or re.fullmatch(r"[0-9a-f]{40,64}", revision["revision"]) is None
            or revision["revision"] != config.get(component + "_revision")
        ):
            return None
    if not is_hash(manifest.get("chat_template_hash")):
        return None
    if generation != config.get("generation") or any(
        not isinstance(generation.get(mode), dict) or not generation[mode]
        for mode in ("train", "eval")
    ):
        return None
    for key in ("train_item_ids", "test_item_ids"):
        ids = manifest.get(key)
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(value, str) and value for value in ids)
            or len(set(ids)) != len(ids)
        ):
            return None
    payload = {
        "fingerprint_schema": "samestack-shared-stack-v1",
        "source": {key: source[key] for key in ("commit", "dirty", "dirty_hash", "source_files")},
        "revisions": revisions,
        "chat_template_hash": manifest["chat_template_hash"],
        "generation": generation,
        "train_item_ids": manifest["train_item_ids"],
        "test_item_ids": manifest["test_item_ids"],
        "shared_effective_config": {
            key: value for key, value in config.items() if key not in ARM_SEED_CONFIG_FIELDS
        },
    }
    try:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (ValueError, TypeError):
        return None


def default_plan():
    return {
        "plan_id": "samestack-v3-seed-paired-analysis-v1",
        "planned_seeds": list(SEEDS),
        "environment": None,
        "environment_hash": None,
        "shared_stack_hash": None,
        "protocol_hash": None,
        "ordered_test_item_hash": None,
        "primary_contrast_ids": list(PRIMARY),
        "endpoint": {"declared": False, "description": "All four arms at every planned seed"},
        "equivalence_margin": 0.02,
        "margin_justification": None,
        "prespecification_status": "retrospective_repair",
    }


def validate_plan(plan):
    plan = deepcopy(plan)
    if plan.get("primary_contrast_ids") != list(PRIMARY):
        raise ValueError("primary family must be the frozen three contrasts in fixed order")
    seeds = plan.get("planned_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(type(s) is not int for s in seeds):
        raise ValueError("planned_seeds must contain at least two integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate planned seed")
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"]:
        raise ValueError("plan_id is required")
    endpoint = plan.get("endpoint")
    if not isinstance(endpoint, dict) or type(endpoint.get("declared")) is not bool:
        raise ValueError("endpoint.declared must be an explicit boolean")
    margin = plan.get("equivalence_margin")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("equivalence_margin must be finite and positive")
    for field in IDENTITY_FIELDS[1:]:
        value = plan.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"plan {field} must be a nonempty string or null")
    # This software repair cannot retroactively establish prespecification.
    if plan.get("prespecification_status") != "retrospective_repair":
        raise ValueError("this analysis supports retrospective_repair plans only")
    json.dumps(plan, allow_nan=False)
    return plan


def _record_identity(record):
    manifest = record.get("manifest") or {}
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    return {
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        **{key: manifest.get(key) for key in IDENTITY_FIELDS},
        "shared_stack_hash": shared_stack_fingerprint(record),
    }


def _validate_values(record):
    n = record.get("n_eval")
    if type(n) is not int or n < 1:
        raise ValueError("n_eval must be a positive integer")
    for field in ("pre_correct", "post_correct"):
        values = record.get(field)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"{field} length must equal n_eval")
        if any(type(v) not in (int, bool) or v not in (0, 1) for v in values):
            raise ValueError(f"{field} must contain binary verdicts")
    for field, values in (
        ("heldout_pre_acc", record["pre_correct"]),
        ("heldout_post_acc", record["post_correct"]),
    ):
        if field in record:
            value = record[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, sum(values) / n, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError(f"{field} disagrees with verdicts")


def _contrast(key, arms, selected, plan):
    a, b = arms
    seeds = [s for s in plan["planned_seeds"] if (a, s) in selected and (b, s) in selected]
    result = {
        "id": key,
        "arms": list(arms),
        "n": len(seeds),
        "seeds": seeds,
        "missing_seeds": [s for s in plan["planned_seeds"] if s not in seeds],
        "pairs": [
            {
                "seed": s,
                "a_run_id": _record_identity(selected[a, s])["run_id"],
                "b_run_id": _record_identity(selected[b, s])["run_id"],
            }
            for s in seeds
        ],
        "status": "available" if len(seeds) >= 2 else "unavailable",
        "mean": None,
        "sd": None,
        "diffs": [],
        "t_ci": None,
        "tost_ci90": None,
        "p_t": 1.0,
        "p_signflip": 1.0,
        "tost_p": 1.0,
        "equivalence_claim_eligible": False,
    }
    for seed in seeds:
        x, y = selected[a, seed], selected[b, seed]
        xi, yi = _record_identity(x), _record_identity(y)
        if (
            any(xi[field] != yi[field] for field in IDENTITY_FIELDS[1:])
            or x["n_eval"] != y["n_eval"]
        ):
            raise ValueError(
                f"unmatched environment/shared stack/protocol/items for {key}, seed {seed}"
            )
    if len(seeds) >= 2:
        accuracies = lambda arm: {
            s: sum(selected[arm, s]["post_correct"]) / selected[arm, s]["n_eval"] for s in seeds
        }
        result.update(paired(accuracies(a), accuracies(b), margin=plan["equivalence_margin"]))
    return result


def analyze_runs(
    records, plan=None, *, completion_validated_run_ids=(), input_artifact_exclusions=()
):
    """Analyze a materialized list of raw records without filesystem/CLI effects.

    Duplicate identities are errors. Out-of-plan, unverified or incompatible
    records are excluded with reasons; they never enter a paired comparison.
    Missing tests consume a p=1 slot in each fixed three-member Holm family.
    """
    plan = validate_plan(default_plan() if plan is None else plan)
    checked_ids = set(completion_validated_run_ids)
    if any(not isinstance(run_id, str) or not run_id for run_id in checked_ids):
        raise ValueError("completion-validated run IDs must be nonempty strings")
    loader_exclusions = deepcopy(list(input_artifact_exclusions))
    for exclusion in loader_exclusions:
        if (
            not isinstance(exclusion, dict)
            or not isinstance(exclusion.get("reasons"), list)
            or not exclusion["reasons"]
            or not all(isinstance(reason, str) and reason for reason in exclusion["reasons"])
        ):
            raise ValueError("input artifact exclusions require nonempty string reasons")
    json.dumps(loader_exclusions, allow_nan=False)
    selected, excluded = {}, []
    seen_run_ids, seen_identities = set(), set()
    planned = {(arm, seed) for arm in ARMS for seed in plan["planned_seeds"]}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each run must be an object")
        identity = _record_identity(record)
        arm, seed = identity["arm"], identity["seed"]
        if not isinstance(arm, str) or type(seed) is not int:
            raise ValueError("run arm/seed identity is invalid")
        run_id = identity["run_id"]
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("run_id must be a nonempty string")
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate run_id: {run_id}")
            seen_run_ids.add(run_id)
        for field in IDENTITY_FIELDS[1:]:
            if identity[field] is not None and not isinstance(identity[field], str):
                raise ValueError(f"run {field} must be a string or null")
        semantic = (arm, seed, identity["environment"], identity["protocol_hash"])
        if semantic in seen_identities:
            raise ValueError(f"duplicate run identity: {arm}, seed {seed}")
        seen_identities.add(semantic)
        _validate_values(record)
        reasons = []
        if (arm, seed) not in planned:
            reasons.append("outside planned arm/seed cohort")
        for field in IDENTITY_FIELDS:
            if not isinstance(identity[field], str) or not identity[field]:
                reasons.append(f"unverified {field}")
        for field in IDENTITY_FIELDS[1:]:
            if plan.get(field) is None:
                reasons.append(f"plan has not bound {field}")
            elif identity[field] != plan[field]:
                reasons.append(f"{field} does not match plan")
        from platform_hybrid.experiments.samestack_contract import validate_run

        validation = validate_run(record, filename=record.get("source_path"), require_texts=False)
        if not validation["verified"]:
            reasons.extend(validation["errors"] + validation["unverified"])
        if (record.get("manifest") or {}).get("analysis_plan_id") != plan["plan_id"]:
            reasons.append("manifest analysis_plan_id does not match plan_id")
        recorded_stack = (record.get("manifest") or {}).get("shared_stack_hash")
        if recorded_stack is not None and recorded_stack != identity["shared_stack_hash"]:
            reasons.append("recorded shared_stack_hash contradicts derived identity")
        if record.get("smoke") is True:
            reasons.append("smoke run is not a primary run")
        if reasons:
            excluded.append({**identity, "reasons": reasons})
            continue
        if (arm, seed) in selected:
            raise ValueError(f"duplicate selected arm/seed: {arm}, {seed}")
        selected[arm, seed] = record
    missing = [
        {"arm": arm, "seed": seed}
        for arm in ARMS
        for seed in plan["planned_seeds"]
        if (arm, seed) not in selected
    ]
    primary = {key: _contrast(key, arms, selected, plan) for key, arms in PRIMARY.items()}
    for raw, adjusted in (
        ("p_t", "holm_p_t"),
        ("p_signflip", "holm_p_signflip"),
        ("tost_p", "holm_tost_p"),
    ):
        corrected = holm({key: value[raw] for key, value in primary.items()})
        for key, value in primary.items():
            value[adjusted] = corrected[key]
    interim = []
    if not plan["endpoint"]["declared"]:
        interim.append("analysis endpoint has not been declared")
    if missing:
        interim.append(f"{len(missing)} planned runs are missing or excluded")
    if any(plan.get(field) is None for field in IDENTITY_FIELDS[1:]):
        interim.append("analysis plan has unbound provenance fields")
    identities = [_record_identity(selected[key]) for key in sorted(selected)]
    selected_ids = {identity["run_id"] for identity in identities}
    if any(row.get("run_id") in selected_ids for row in loader_exclusions):
        raise ValueError("a run cannot be selected and excluded by artifact validation")
    for identity in identities:
        identity["completion_bundle_validated"] = identity["run_id"] in checked_ids
    return {
        "schema_version": 1,
        "status": "interim" if interim else "endpoint_complete",
        "final_eligible": False,
        "interim_reasons": interim,
        "plan": plan,
        "plan_hash": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "selected_runs": identities,
        "missing_planned_runs": missing,
        "excluded_runs": excluded + loader_exclusions,
        "artifact_validation": {
            "scope": "Pure API validates record schema; completion checks are explicitly supplied by the file-boundary caller",
            "completion_validated_selected_run_ids": sorted(selected_ids & checked_ids),
            "unchecked_selected_run_ids": sorted(selected_ids - checked_ids),
            "input_artifact_exclusion_count": len(loader_exclusions),
        },
        "completion": {"planned": len(planned), "selected": len(selected), "missing": len(missing)},
        "primary_contrasts": primary,
        "exploratory_contrasts": {
            key: _contrast(key, arms, selected, plan) for key, arms in EXPLORATORY.items()
        },
        "inference": {
            "primary_unit": "training seed",
            "primary_test": "paired t, Holm across fixed three contrasts",
            "paired_t_assumptions": "Independent training-seed pairs and approximately normal paired differences; many test items do not increase seed replication",
            "signflip": "Sensitivity test requiring independent paired seeds and symmetric/exchangeable signs; not a treatment-randomization test",
            "five_pair_minimum_two_sided_signflip_p": 0.0625,
            "item_only": "Conditional on observed training seeds; not training-run uncertainty",
            "pooled_mcnemar": "Nonprimary descriptive diagnostic; repeated seed-item observations are not independent",
            "equivalence": f"90% TOST CI and combined max(one-sided p) Holm-adjusted across the fixed family; +/-{plan['equivalence_margin'] * 100:g}pp margin; "
            + (
                f"justification: {plan['margin_justification']}"
                if plan.get("margin_justification")
                else "no supplied scientific justification"
            ),
            "prespecification": "Retrospective repair; no prospective registration claim",
            "power": "Pilot unadjusted paired-t sensitivity only; actual-design joint seed/item, family and TOST simulation remains unmet without justified variance/correlation assumptions",
        },
    }
