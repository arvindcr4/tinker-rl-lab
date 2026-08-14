#!/usr/bin/env python3
"""Seal the full 480-task APEX-Agents split manifest at the pinned revision.

Emits a manifest that pins, by content hash, exactly which tasks the benchmark
contains -- so any later subset can be shown to be a subset of THIS revision,
and any drift in the upstream dataset is detectable without re-downloading.

Training task IDs are empty and must stay empty: the dataset card states
"Any use of this dataset for training, fine-tuning, or parameter fitting is
forbidden."

Writes: outputs/e5_apex_agents/split_manifest_480.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASET = HERE / "hf_dataset"
SIBLINGS = HERE / "evidence" / "hf_dataset_meta.json"
OUT = HERE / "split_manifest_480.json"

DATASET_ID = "mercor/apex-agents"
DATASET_REVISION = "92c86856cf1b11f9833a8a076b3a45a63afa3929"

sys.path.insert(0, str(REPO / "zvf-program"))
from flagship import eval_apex_agents as runner  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    tasks_bytes = (DATASET / "tasks_and_rubrics.json").read_bytes()
    worlds_bytes = (DATASET / "world_descriptions.json").read_bytes()
    tasks = json.loads(tasks_bytes)
    worlds = json.loads(worlds_bytes)

    report = runner.dataset_ingestion_report(tasks, worlds)
    if not report["valid"]:
        print("REFUSING to seal a manifest over an invalid dataset:", file=sys.stderr)
        for error in report["errors"][:20]:
            print("  -", error, file=sys.stderr)
        return 1

    siblings = [
        s["rfilename"]
        for s in json.loads(SIBLINGS.read_text(encoding="utf-8"))["siblings"]
    ]
    world_zips = {f.split("/")[1][:-4] for f in siblings if f.startswith("world_files_zipped/")}
    task_file_dirs = {f.split("/")[1] for f in siblings if f.startswith("task_files/")}
    gold_file_dirs = {f.split("/")[1] for f in siblings if f.startswith("gold_files/")}
    junk = {name for name in task_file_dirs | gold_file_dirs if name.startswith(".")}
    task_file_dirs -= junk
    gold_file_dirs -= junk

    task_ids = sorted(t["task_id"] for t in tasks)
    world_ids = sorted(w["world_id"] for w in worlds)
    declared_inputs = {t["task_id"] for t in tasks if t.get("task_input_files")}
    file_output_tasks = {t["task_id"] for t in tasks if t["gold_response_type"] == "file"}

    by_domain = collections.defaultdict(list)
    for task in tasks:
        by_domain[task["domain"]].append(task)

    manifest = {
        "schema_version": "pavlov-e5-apex-agents-split-manifest-v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "suite_id": "apex_agents_eval",
        "suite_role": "primary_eval",
        "intended_use": "model evaluation only",
        "training_prohibited": True,
        "training_prohibition_source": (
            "dataset card: 'Any use of this dataset for training, fine-tuning, "
            "or parameter fitting is forbidden.'"
        ),
        "training_task_ids": [],
        "training_scope": "none (eval-only)",
        "evaluation_task_ids": task_ids,
        "world_ids": world_ids,
        "file_digests": {
            "tasks_and_rubrics.json": sha256_bytes(tasks_bytes),
            "world_descriptions.json": sha256_bytes(worlds_bytes),
            "eval.yaml": sha256_bytes((DATASET / "eval.yaml").read_bytes()),
            "metadata.json": sha256_bytes((DATASET / "metadata.json").read_bytes()),
            "README.md": sha256_bytes((DATASET / "README.md").read_bytes()),
        },
        "task_id_sha256": report["task_id_sha256"],
        "world_id_sha256": report["world_id_sha256"],
        "counts": {
            "tasks": len(tasks),
            "worlds": len(worlds),
            "rubric_criteria": sum(len(t["rubric"]) for t in tasks),
            "tasks_with_input_files": len(declared_inputs),
            "tasks_with_file_output": len(file_output_tasks),
            "by_domain": {
                domain: {
                    "tasks": len(rows),
                    "criteria": sum(len(t["rubric"]) for t in rows),
                    "mean_criteria": round(
                        sum(len(t["rubric"]) for t in rows) / len(rows), 3
                    ),
                }
                for domain, rows in sorted(by_domain.items())
            },
        },
        "referential_integrity": {
            "world_refs_resolve": sorted({t["world_id"] for t in tasks}) == world_ids,
            "world_zip_set_matches": set(world_ids) == world_zips,
            "task_input_files_match_task_files_dirs": declared_inputs == task_file_dirs,
            "file_output_tasks_match_gold_files_dirs": file_output_tasks == gold_file_dirs,
            "ignored_junk_entries": sorted(junk),
            "checked_against": "the 319-entry Hugging Face file listing; no bulk assets downloaded",
        },
        "tasks": [
            {
                "task_id": t["task_id"],
                "world_id": t["world_id"],
                "domain": t["domain"],
                "task_name": t["task_name"],
                "rubric_criteria": len(t["rubric"]),
                "verifier_ids": [c["verifier_id"] for c in t["rubric"]],
                "expected_output": t["expected_output"],
                "gold_response_type": t["gold_response_type"],
                "gold_response_sha256": sha256_bytes(t["gold_response"].encode("utf-8")),
                "has_task_input_files": bool(t.get("task_input_files")),
            }
            for t in sorted(tasks, key=lambda t: t["task_id"])
        ],
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ri = manifest["referential_integrity"]
    print(f"sealed {len(task_ids)} tasks / {len(world_ids)} worlds at {DATASET_REVISION}")
    print(f"  task_id_sha256  = {manifest['task_id_sha256']}")
    print(f"  world_id_sha256 = {manifest['world_id_sha256']}")
    for key, value in ri.items():
        if isinstance(value, bool):
            print(f"  {key:44s} {value}")
    print(f"written: {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
