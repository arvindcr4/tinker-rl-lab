#!/usr/bin/env python3
"""Read-only Hugging Face provenance audit for the NeurIPS 36320 evidence.

The token is read once from stdin and passed directly to the Hub client. It is
never printed, written, or installed in the user's Hugging Face configuration.
All downloaded manifests live in a temporary directory that is deleted before
the process exits.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "zvf-program/audit/results/campaign-verification.json"
LOCAL_MANIFESTS = ROOT / "zvf-program/audit/results/full/manifests"
EXPECTED_STEPS = [5, 10, 15, 20, 25, 30]
CHECKPOINT_RE = re.compile(r"^checkpoints/checkpoint-(\d+)/trainer_state\.json$")
MATCHED_TERMS = ("matched", "36320", "reinforce")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def compact_model(model: Any) -> dict[str, Any]:
    return {
        "id": getattr(model, "id", None) or getattr(model, "modelId", None),
        "private": getattr(model, "private", None),
        "sha": getattr(model, "sha", None),
        "created_at": jsonable(getattr(model, "created_at", None)),
        "last_modified": jsonable(getattr(model, "last_modified", None)),
    }


def main() -> int:
    token = sys.stdin.readline().strip()
    if not token:
        print(json.dumps({"error": "empty Hugging Face token"}))
        return 2

    try:
        from huggingface_hub import HfApi, hf_hub_download
        import huggingface_hub

        api = HfApi(token=token)
        identity = api.whoami(token=token, cache=False)
        username = str(identity.get("name") or identity.get("fullname") or "")
        if not username:
            raise RuntimeError("Hugging Face identity has no username")

        models = list(
            api.list_models(
                author=username,
                full=True,
                limit=None,
                token=token,
            )
        )
        compact_models = [compact_model(model) for model in models]
        model_ids = {str(row["id"]) for row in compact_models if row.get("id")}

        campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        units = campaign["units"]
        expected_repos = {str(unit["hf_repo"]) for unit in units}
        expected_by_repo = {str(unit["hf_repo"]): unit for unit in units}

        checked: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        with tempfile.TemporaryDirectory(prefix="hf-evidence-audit-") as tmp:
            tmp_path = Path(tmp)
            for repo_id in sorted(expected_repos):
                unit = expected_by_repo[repo_id]
                revision = str(unit["hf_commit"])
                try:
                    info = api.model_info(
                        repo_id,
                        revision=revision,
                        files_metadata=False,
                        token=token,
                    )
                    files = sorted(
                        sibling.rfilename
                        for sibling in (info.siblings or [])
                        if getattr(sibling, "rfilename", None)
                    )
                    if not files:
                        files = sorted(
                            api.list_repo_files(
                                repo_id,
                                revision=revision,
                                repo_type="model",
                                token=token,
                            )
                        )
                    steps = sorted(
                        {
                            int(match.group(1))
                            for filename in files
                            if (match := CHECKPOINT_RE.match(filename))
                        }
                    )
                    manifest_path = Path(
                        hf_hub_download(
                            repo_id=repo_id,
                            repo_type="model",
                            filename="run_manifest.json",
                            revision=revision,
                            token=token,
                            local_dir=tmp_path / repo_id.rsplit("/", 1)[-1],
                        )
                    )
                    manifest_bytes = manifest_path.read_bytes()
                    manifest = json.loads(manifest_bytes)
                    audit_record = manifest.get("audit_record", {})
                    run_config = manifest.get("run_config", {})
                    local_manifest = LOCAL_MANIFESTS / (
                        str(unit["unit"]).replace("/seed-", "-seed-") + ".json"
                    )
                    local_sha = (
                        sha256_bytes(local_manifest.read_bytes())
                        if local_manifest.is_file()
                        else None
                    )
                    remote_sha = sha256_bytes(manifest_bytes)
                    manifest_checks = {
                        "schema_confirmatory": manifest.get("schema_version")
                        == "e1-colab-confirmatory-run-v1",
                        "evidence_confirmatory": manifest.get("evidence_class")
                        == "confirmatory",
                        "arm_matches": audit_record.get("arm")
                        == str(unit["unit"]).split("/", 1)[0],
                        "seed_matches": audit_record.get("seed")
                        == int(str(unit["unit"]).rsplit("seed-", 1)[1]),
                        "heldout_n_500": audit_record.get("heldout_n") == 500,
                        "heldout_score_matches": audit_record.get("heldout_score")
                        == unit.get("heldout_score"),
                        "unit_fingerprint_matches": run_config.get("unit_fingerprint")
                        == unit.get("unit_fingerprint"),
                        "stack_fingerprint_matches": run_config.get("stack_fingerprint")
                        == unit.get("stack_fingerprint"),
                        "remote_steps_match": manifest.get("remote_checkpoint_steps")
                        == EXPECTED_STEPS,
                        "trace_has_500_rows": len(manifest.get("heldout_trace", []))
                        == 500,
                        "local_manifest_sha_matches": local_sha == remote_sha,
                    }
                    required_files = {
                        "run_manifest.json": "run_manifest.json" in files,
                        "final_adapter": "final/adapter_model.safetensors" in files,
                        "evaluation_progress": "evaluation/progress.json" in files,
                    }
                    checked.append(
                        {
                            "unit": unit["unit"],
                            "repo_id": repo_id,
                            "expected_commit": revision,
                            "resolved_commit": getattr(info, "sha", None),
                            "private": getattr(info, "private", None),
                            "file_count": len(files),
                            "checkpoint_steps": steps,
                            "required_files": required_files,
                            "manifest_sha256": remote_sha,
                            "expected_manifest_sha256": unit.get("manifest_sha256"),
                            "manifest_checks": manifest_checks,
                        }
                    )
                except Exception as exc:
                    message = str(exc).replace(token, "<redacted>")
                    errors.append(
                        {
                            "unit": str(unit.get("unit")),
                            "repo_id": repo_id,
                            "type": type(exc).__name__,
                            "message": message,
                        }
                    )

        matched_candidates = [
            row
            for row in compact_models
            if row.get("id")
            and any(term in str(row["id"]).lower() for term in MATCHED_TERMS)
            and str(row["id"]) not in expected_repos
        ]
        e1_models = [
            row
            for row in compact_models
            if str(row.get("id", "")).startswith(f"{username}/tinker-rl-lab-e1-")
        ]
        missing_from_listing = sorted(expected_repos - model_ids)

        checks = {
            "expected_units": len(units),
            "checked_units": len(checked),
            "error_units": len(errors),
            "expected_repos_listed": len(expected_repos & model_ids),
            "missing_expected_repos_from_listing": missing_from_listing,
            "commit_matches": sum(
                row["expected_commit"] == row["resolved_commit"] for row in checked
            ),
            "private_repos": sum(row["private"] is True for row in checked),
            "all_six_checkpoints": sum(
                row["checkpoint_steps"] == EXPECTED_STEPS for row in checked
            ),
            "all_required_files": sum(
                all(row["required_files"].values()) for row in checked
            ),
            "all_manifest_checks": sum(
                all(row["manifest_checks"].values()) for row in checked
            ),
            "local_manifest_sha_matches": sum(
                row["manifest_checks"]["local_manifest_sha_matches"] for row in checked
            ),
            "campaign_manifest_sha_matches": sum(
                row["manifest_sha256"] == row["expected_manifest_sha256"]
                for row in checked
            ),
        }
        result = {
            "sdk_version": getattr(huggingface_hub, "__version__", None),
            "identity": {
                "name": username,
                "type": identity.get("type"),
                "auth_type": identity.get("auth", {}).get("type")
                if isinstance(identity.get("auth"), dict)
                else None,
            },
            "author_model_count": len(models),
            "author_private_model_count": sum(
                row.get("private") is True for row in compact_models
            ),
            "e1_model_count": len(e1_models),
            "e1_models": e1_models,
            "matched_candidate_count": len(matched_candidates),
            "matched_candidates": matched_candidates,
            "checks": checks,
            "errors": errors,
            "units": checked,
            "model_visibility_counts": dict(
                Counter("private" if row.get("private") else "public" for row in compact_models)
            ),
        }
        print(json.dumps(result, sort_keys=True, default=jsonable))
        return 0 if not errors else 1
    except Exception as exc:
        message = str(exc).replace(token, "<redacted>")
        print(json.dumps({"error": type(exc).__name__, "message": message}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
