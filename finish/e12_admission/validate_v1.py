"""Offline evidence admission only. Never launches, downloads, or authenticates grants."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

FLOOR = 6 * 1024**3
REVISION = "de80d5bcd404adee5307311571e512b5c37e6112"
PINS = {
    "csv": "b28b74959e81602f0a8b7e8985547915cf3de98822b87f736bc86269518900ba",
    "split": "ee9de54f596322d8eced32a26917bffe1ee38c962721be1c5e69ae9d126f0415",
    "disjointness": "ed3f86ec71477d420d5d82b98beb43679bdd0c4c0c1175f84793f8b62e3bc0e6",
}
EVIDENCE = ("permission", "holdout", "environment", "grading", "attempt_ledger")


def validate(manifest, root):
    """Validate local files and explicit attestations; READY is not launch authority.

    Paths are relative to root. Root must be the repository or a test fixture.
    Parent review must independently establish authenticity/completeness of evidence.
    Disk is checked before each file and after validation; no writes occur here.
    """
    root = Path(root).resolve()
    errors = []

    def require(condition, code):
        if not condition:
            errors.append(code)

    def disk():
        if shutil.disk_usage(root).free < FLOOR:
            raise ValueError("DISK_BELOW_6_GIB")

    def file(ref, label, pin=None, parse=True):
        disk()
        if not isinstance(ref, dict):
            raise ValueError(label + ":MISSING_DESCRIPTOR")
        name, digest = ref.get("path"), ref.get("sha256")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError(label + ":RELATIVE_PATH_REQUIRED")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(label + ":MISSING_OR_OUTSIDE_ROOT")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(label + ":SHA256_REQUIRED")
        if path.stat().st_size > 2_000_000 or path.stat().st_size == 0:
            raise ValueError(label + ":EMPTY_OR_TOO_LARGE")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or (pin and digest != pin):
            raise ValueError(label + ":HASH_MISMATCH")
        if not parse:
            return raw
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(label + ":OBJECT_REQUIRED")
        return value

    try:
        disk()
        require(manifest.get("schema_version") == "e12-offline-admission-v1", "SCHEMA")
        require(manifest.get("suite_id") == "appbench_eval", "SUITE")
        require(manifest.get("dataset_revision") == REVISION, "REVISION")
        require(manifest.get("scope") == "original_heldout", "ORIGINAL_SCOPE_REQUIRED")
        refs = manifest.get("assets", {})
        docs = {}
        for name, pin in PINS.items():
            try:
                docs[name] = file(refs.get(name), name, pin, name != "csv")
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        for name in EVIDENCE:
            try:
                docs[name] = file(refs.get(name), name)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        split = docs.get("split", {})
        ids = [t["task_id"] for t in split.get("tasks", [])]
        require(len(ids) == 6 and len(set(ids)) == 6, "EXACT_SIX_TASK_IDS_REQUIRED")
        model = manifest.get("model") or {}
        require(bool(model.get("id")) and bool(re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", "")))), "IMMUTABLE_MODEL_REQUIRED")
        binding = {"suite_id": "appbench_eval", "dataset_revision": REVISION,
                   "split_sha256": PINS["split"], "model": model}
        for name in EVIDENCE:
            doc = docs.get(name, {})
            require(doc.get("binding") == binding, name + ":BINDING")
        permission = docs.get("permission", {})
        require(permission.get("issuer") == "AfterQuery" and permission.get("evaluation_allowed") is True
                and permission.get("aggregate_publication_allowed") is True, "PERMISSION_REQUIRED")
        holdout = docs.get("holdout", {})
        require(holdout.get("issuer") == "AfterQuery" and holdout.get("held_out_from_model_training") is True
                and holdout.get("task_ids") == ids and bool(ids), "MODEL_SPECIFIC_HOLDOUT_REQUIRED")
        env = docs.get("environment", {})
        require(env.get("issuer") == "AfterQuery" and env.get("official_exact_environment") is True,
                "OFFICIAL_ENVIRONMENT_REQUIRED")
        require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(env.get("image_digest", "")))), "ENV_IMAGE_DIGEST")
        for name in ("template", "runtime_reset", "deployment", "artifact_verification", "side_effect_verification", "credentials_policy"):
            try:
                file(env.get(name), "environment:" + name, parse=False)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        grading = docs.get("grading", {})
        graders = grading.get("graders", [])
        require(isinstance(graders, list) and len(graders) == 2
                and all(isinstance(g, dict) and g.get("qualified_full_stack") is True and g.get("id") for g in graders)
                and len({str(g.get("id")) for g in graders if isinstance(g, dict)}) == 2, "TWO_QUALIFIED_GRADERS")
        require(grading.get("issuer") == "AfterQuery" and grading.get("independent_then_consensus") is True
                and grading.get("binary_per_item") is True and grading.get("attempts_per_task") == 3
                and grading.get("aggregation") == "sum_best_of_three_per_task_over_151"
                and grading.get("rubric_counts") == [24, 33, 22, 25, 23, 24]
                and grading.get("website_23_vs_csv_24_resolved") is True, "EXACT_GRADING_PROTOCOL")
        attempt = manifest.get("attempt") or {}
        require(isinstance(attempt.get("run_id"), str) and bool(attempt.get("run_id", "").strip()), "RUN_ID_REQUIRED")
        require(attempt.get("task_id") in ids and type(attempt.get("ordinal")) is int
                and 1 <= attempt.get("ordinal", 0) <= 3, "EXACT_ATTEMPT_IDENTITY")
        ledger = docs.get("attempt_ledger", {})
        require(ledger.get("authoritative_complete") is True, "ATTEMPT_HISTORY_UNKNOWN")
        records = ledger.get("attempts", [])
        expected = {(task, n) for task in ids for n in (1, 2, 3)}
        keys = [(r["task_id"], r["ordinal"]) for r in records]
        require(len(keys) == 18 and set(keys) == expected and len(set(keys)) == len(keys), "COMPLETE_18_SLOT_LEDGER")
        require(all(r.get("state") in ("not_started", "active", "completed", "failed", "unknown") for r in records), "LEDGER_STATE")
        selected = [r for r in records if r.get("task_id") == attempt.get("task_id") and r.get("ordinal") == attempt.get("ordinal")]
        require(len(selected) == 1 and selected[0].get("state") == "not_started", "NO_REPLAY_OR_UNKNOWN")
        require(not any(r.get("state") in ("active", "unknown") for r in records), "RECONCILE_ACTIVE_OR_UNKNOWN")
        require(not any(r.get("run_id") == attempt.get("run_id") for r in records), "RUN_ID_ALREADY_USED")
        for r in records:
            if r.get("state") in ("completed", "failed"):
                file(r.get("receipt"), "attempt_receipt")
        review = file(refs.get("parent_review"), "parent_review")
        require(review.get("binding") == binding and review.get("attempt") == attempt
                and review.get("authenticity_and_history_verified") is True and bool(review.get("reviewer"))
                and review.get("evidence_sha256") == {n: refs.get(n, {}).get("sha256") for n in EVIDENCE}, "PARENT_EVIDENCE_REVIEW_REQUIRED")
        disk()
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        errors.append("INVALID_OR_MISSING_INPUT:" + str(exc))
    return {"schema_version": "e12-offline-admission-result-v1",
            "status": "BLOCKED" if errors else "OFFLINE_EVIDENCE_VALIDATED",
            "blockers": sorted(set(errors)), "launch_authorized": False,
            "score": None, "is_model_score": False,
            "claim_boundary": "Local consistency only; parent must verify authentic grants, current attempt history, disk, and allocation at launch."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text())
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "blockers": [str(exc)], "launch_authorized": False}))
        return 2
    result = validate(data, args.root)
    print(json.dumps(result, indent=2))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
