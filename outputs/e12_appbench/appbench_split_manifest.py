"""E12 AppBench evaluation split manifest: immutable task IDs, hashes, disjointness.

Scope of what this module proves and what it deliberately does NOT prove.

PROVES (offline, no credentials, no network):
  * Every task in `AppBench vExternal.csv` gets a deterministic, content-addressed
    64-hex task ID that is stable across machines, Python versions and CSV
    line-ending transport, and that changes if the upstream task text changes.
  * A deterministic split hash over the eval task ID set, byte-compatible with
    ``pavlov_appbench_openreward_games_adapter._deterministic_split_hash``.
  * An aggregate manifest hash covering every field of the manifest.
  * Set-disjointness between the E12 eval split and any other named split.

DOES NOT PROVE:
  * That the split is *held out from model training*. The source file is the
    public ``vExternal`` CSV published on the Hugging Face Hub on 2025-12-10 with
    a nonzero public download count. Contamination for any model whose training
    data postdates that publication is an open question this manifest cannot
    close. See ``heldout_claim`` in the emitted manifest.
  * That the task data is licensed for evaluation use. See the lane receipt.

Task ID construction
--------------------
``task_id = sha256(canonical_json(identity_payload))`` where ``identity_payload``
binds the task content to the exact upstream dataset revision::

    {"content": <content_payload>, "dataset_id": ..., "revision": ...,
     "source_file": ..., "suite_id": "appbench_eval"}

``content_sha256 = sha256(canonical_json(content_payload))`` is the same digest
with the revision pin removed, so a republished-but-identical task can be
recognised across revisions.

Canonical JSON is ``json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)``.

Text normalisation before hashing is intentionally minimal and is part of the
contract: CRLF/CR are folded to LF (the upstream CSV is CRLF), and leading and
trailing whitespace of each field is stripped. Interior whitespace, casing and
punctuation are preserved, so any substantive upstream edit changes the ID.

CLI
---
    python3 outputs/e12_appbench/appbench_split_manifest.py build --out <path>
    python3 outputs/e12_appbench/appbench_split_manifest.py verify --manifest <path>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "e12-appbench-split-manifest-v1"
SUITE_ID = "appbench_eval"
BOUNDARY_ID = "E12"

DATASET_ID = "AfterQuery/App-Bench"
DATASET_REVISION = "de80d5bcd404adee5307311571e512b5c37e6112"
SOURCE_FILE = "AppBench vExternal.csv"
SOURCE_BLOB_SHA1 = "f156b93ede86d3c03daa82d75e2ca3b2612fefc2"
SOURCE_SHA256 = "b28b74959e81602f0a8b7e8985547915cf3de98822b87f736bc86269518900ba"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "outputs" / "e12_appbench" / "hf_dataset" / SOURCE_FILE
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "e12_appbench" / "split_manifest.json"

COLUMN_ORDINAL = "#"
COLUMN_APP_NAME = "App Name"
COLUMN_APP_DESCRIPTION = "App Description"
COLUMN_PROMPT = "Prompt"
COLUMN_CLI_ADDITION = "Addition for CLI Tools"
COLUMN_RUBRIC = "Rubric"

TASK_COLUMNS = (
    COLUMN_ORDINAL,
    COLUMN_APP_NAME,
    COLUMN_APP_DESCRIPTION,
    COLUMN_PROMPT,
    COLUMN_CLI_ADDITION,
    COLUMN_RUBRIC,
)

# Columns present in the CSV that hold per-tool leaderboard results. They are
# empty at the pinned revision and are excluded from the task identity so that a
# future upstream score backfill does not rotate every task ID.
RESULT_COLUMNS = (
    "Claude Code",
    "Codex CLI",
    "Cursor CLI",
    "Gemini CLI",
    "Orchids",
    "Lovable",
    "Bolt",
    "Replit",
    "v0",
)


class AppBenchSplitManifestError(ValueError):
    """Raised when the AppBench source CSV or a manifest fails validation."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    """Fold CRLF/CR to LF and strip leading/trailing whitespace."""

    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_rubric_items(rubric: str) -> list[str]:
    """Return the numbered rubric lines, in file order, with the number stripped."""

    items: list[str] = []
    for line in normalize_text(rubric).split("\n"):
        stripped = line.strip()
        head, sep, tail = stripped.partition(".")
        if sep and head.isdigit() and tail.strip():
            items.append(tail.strip())
    return items


def parse_requirement_items(prompt: str) -> list[str]:
    """Return the numbered ``## Feature Requirements`` lines from the prompt."""

    return parse_rubric_items(prompt)


def load_tasks(csv_path: Path | str = DEFAULT_CSV) -> list[dict[str, Any]]:
    """Read the AppBench CSV into normalized task records, in file order."""

    path = Path(csv_path)
    if not path.is_file():
        raise AppBenchSplitManifestError(f"AppBench CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [c for c in TASK_COLUMNS if c not in fieldnames]
        if missing:
            raise AppBenchSplitManifestError(
                f"AppBench CSV is missing required columns: {missing}"
            )
        rows = list(reader)

    if not rows:
        raise AppBenchSplitManifestError("AppBench CSV contains no task rows")

    tasks: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        ordinal_raw = normalize_text(row.get(COLUMN_ORDINAL) or "")
        if not ordinal_raw.isdigit():
            raise AppBenchSplitManifestError(
                f"row {position}: '#' column is not an integer: {ordinal_raw!r}"
            )
        app_name = normalize_text(row.get(COLUMN_APP_NAME) or "")
        prompt = normalize_text(row.get(COLUMN_PROMPT) or "")
        rubric = normalize_text(row.get(COLUMN_RUBRIC) or "")
        if not app_name:
            raise AppBenchSplitManifestError(f"row {position}: empty App Name")
        if not prompt:
            raise AppBenchSplitManifestError(f"row {position}: empty Prompt")
        if not rubric:
            raise AppBenchSplitManifestError(f"row {position}: empty Rubric")

        tasks.append(
            {
                "ordinal": int(ordinal_raw),
                "app_name": app_name,
                "app_description": normalize_text(row.get(COLUMN_APP_DESCRIPTION) or ""),
                "prompt": prompt,
                "cli_addition": normalize_text(row.get(COLUMN_CLI_ADDITION) or ""),
                "rubric": rubric,
                "rubric_items": parse_rubric_items(rubric),
                "requirement_items": parse_requirement_items(prompt),
            }
        )

    ordinals = [t["ordinal"] for t in tasks]
    if len(set(ordinals)) != len(ordinals):
        raise AppBenchSplitManifestError(f"duplicate '#' ordinals in CSV: {ordinals}")
    return tasks


def content_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    """Revision-independent identity payload for one task."""

    return {
        "ordinal": int(task["ordinal"]),
        "app_name": task["app_name"],
        "app_description": task["app_description"],
        "prompt": task["prompt"],
        "cli_addition": task["cli_addition"],
        "rubric": task["rubric"],
    }


def content_sha256(task: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(content_payload(task)))


def identity_payload(
    task: Mapping[str, Any],
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    source_file: str = SOURCE_FILE,
) -> dict[str, Any]:
    return {
        "content": content_payload(task),
        "dataset_id": dataset_id,
        "revision": revision,
        "source_file": source_file,
        "suite_id": SUITE_ID,
    }


def task_id(
    task: Mapping[str, Any],
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    source_file: str = SOURCE_FILE,
) -> str:
    return _sha256_text(
        canonical_json(identity_payload(task, dataset_id, revision, source_file))
    )


def split_hash(task_ids: Iterable[str]) -> str:
    """Deterministic split hash.

    Byte-for-byte identical to
    ``pavlov_appbench_openreward_games_adapter._deterministic_split_hash`` so a
    manifest emitted here can be dropped straight into the E12 contract boundary.
    Order-independent by construction (the IDs are sorted first).
    """

    payload = canonical_json(sorted(task_ids))
    return _sha256_text(payload)


def assert_disjoint(named_splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Prove pairwise set-disjointness across named ID collections.

    Raises ``AppBenchSplitManifestError`` on any intra-split duplicate or any
    cross-split intersection. Returns a proof record on success.
    """

    pairs: list[dict[str, Any]] = []
    names = sorted(named_splits)
    for name in names:
        ids = list(named_splits[name])
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise AppBenchSplitManifestError(
                f"split {name!r} contains duplicate task IDs: {duplicates}"
            )
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(named_splits[left]) & set(named_splits[right]))
            if overlap:
                raise AppBenchSplitManifestError(
                    f"splits {left!r} and {right!r} overlap on {len(overlap)} task ID(s): "
                    f"{overlap[:3]}"
                )
            pairs.append({"left": left, "right": right, "intersection_size": 0})
    return {
        "method": "exact set intersection over 64-hex content-addressed task IDs",
        "splits": {name: len(set(named_splits[name])) for name in names},
        "pairs_checked": pairs,
        "disjoint": True,
    }


def aggregate_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash of the whole manifest with the aggregate field itself removed."""

    body = {k: v for k, v in manifest.items() if k != "aggregate_sha256"}
    return _sha256_text(canonical_json(body))


def build_manifest(
    csv_path: Path | str = DEFAULT_CSV,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    source_file: str = SOURCE_FILE,
) -> dict[str, Any]:
    """Build the full E12 eval split manifest from the pinned CSV."""

    path = Path(csv_path)
    raw = path.read_bytes()
    tasks = load_tasks(path)

    entries: list[dict[str, Any]] = []
    for task in tasks:
        entries.append(
            {
                "task_id": task_id(task, dataset_id, revision, source_file),
                "content_sha256": content_sha256(task),
                "ordinal": task["ordinal"],
                "app_name": task["app_name"],
                "rubric_item_count": len(task["rubric_items"]),
                "requirement_item_count": len(task["requirement_items"]),
                "prompt_chars": len(task["prompt"]),
                "rubric_chars": len(task["rubric"]),
                "cli_addition_chars": len(task["cli_addition"]),
            }
        )

    ids = [entry["task_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise AppBenchSplitManifestError("task ID collision within the eval split")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "boundary_id": BOUNDARY_ID,
        "suite_id": SUITE_ID,
        "source": {
            "dataset_id": dataset_id,
            "revision": revision,
            "revision_kind": "hugging_face_dataset_commit_sha",
            "source_file": source_file,
            "file_bytes": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_git_blob_sha1": hashlib.sha1(
                b"blob %d\0" % len(raw) + raw
            ).hexdigest(),
            "expected_file_sha256": SOURCE_SHA256,
            "expected_file_git_blob_sha1": SOURCE_BLOB_SHA1,
            "local_path": str(path),
        },
        "identity": {
            "task_id_algorithm": (
                "sha256(canonical_json({content, dataset_id, revision, source_file, suite_id}))"
            ),
            "content_sha256_algorithm": "sha256(canonical_json(content))",
            "canonical_json": 'json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)',
            "text_normalization": "CRLF/CR -> LF; strip leading/trailing whitespace per field",
            "content_fields": list(content_payload(tasks[0]).keys()),
            "excluded_fields": list(RESULT_COLUMNS),
            "excluded_fields_reason": (
                "per-tool leaderboard result columns are empty at the pinned revision; "
                "excluding them keeps task IDs stable if upstream backfills scores"
            ),
        },
        "split": {
            "name": "eval",
            "role": "receipt_proven_heldout",
            "task_count": len(entries),
            "unique_task_count": len(set(ids)),
            "split_hash": split_hash(ids),
            "split_hash_algorithm": (
                "sha256(canonical_json(sorted(task_ids))) - identical to "
                "pavlov_appbench_openreward_games_adapter._deterministic_split_hash"
            ),
        },
        "tasks": entries,
        "heldout_claim": {
            "immutable_task_ids": True,
            "split_manifest_hashed": True,
            "intra_split_disjointness_proven": True,
            "held_out_from_model_training": None,
            "held_out_reason": (
                "The source is the public 'vExternal' CSV on the Hugging Face Hub "
                "(published 2025-12-10, publicly downloadable, no gating). Publication "
                "predates evaluation, so contamination cannot be excluded for any model "
                "whose training corpus postdates that date. Upstream publishes no private "
                "held-out partition that this lane can access."
            ),
        },
        "aggregate_sha256": None,
    }

    manifest["aggregate_sha256"] = aggregate_sha256(manifest)
    return manifest


def verify_manifest(
    manifest: Mapping[str, Any], csv_path: Path | str = DEFAULT_CSV
) -> dict[str, Any]:
    """Rebuild the manifest from the CSV and assert every hash still matches."""

    rebuilt = build_manifest(
        csv_path,
        dataset_id=manifest["source"]["dataset_id"],
        revision=manifest["source"]["revision"],
        source_file=manifest["source"]["source_file"],
    )
    problems: list[str] = []

    if manifest["source"]["file_sha256"] != rebuilt["source"]["file_sha256"]:
        problems.append("source.file_sha256 mismatch")
    if manifest["split"]["split_hash"] != rebuilt["split"]["split_hash"]:
        problems.append("split.split_hash mismatch")
    if aggregate_sha256(manifest) != manifest.get("aggregate_sha256"):
        problems.append("aggregate_sha256 does not cover the manifest body")
    if manifest["aggregate_sha256"] != rebuilt["aggregate_sha256"]:
        problems.append("aggregate_sha256 mismatch against rebuilt manifest")

    recorded = [t["task_id"] for t in manifest["tasks"]]
    expected = [t["task_id"] for t in rebuilt["tasks"]]
    if recorded != expected:
        problems.append("task_id list mismatch against rebuilt manifest")
    if manifest["split"]["split_hash"] != split_hash(recorded):
        problems.append("split_hash is not the deterministic hash of the recorded task_ids")

    if problems:
        raise AppBenchSplitManifestError("; ".join(problems))
    return {"verified": True, "task_count": len(recorded), "checks": 6}


_HEX64_PATTERN = __import__("re").compile(r"^[0-9a-f]{64}$")

# Directory names never worth scanning: vendored dependencies, caches, and the
# raw upstream data drop. Anything starting with "venv" is a virtualenv.
_SCAN_SKIP_DIRS = frozenset(
    {"venvs", "site-packages", "node_modules", "__pycache__", "hf_dataset", ".git"}
)
_SCAN_MAX_BYTES = 2_000_000


def _is_skipped_dir(name: str) -> bool:
    return name in _SCAN_SKIP_DIRS or name.startswith("venv")


def collect_foreign_task_ids(search_root: Path | str, exclude_dir: Path | str) -> dict[str, list[str]]:
    """Collect 64-hex task-ID-shaped strings from JSON files outside this lane.

    Used to show that the E12 eval IDs do not collide with any other boundary's
    recorded IDs in this checkout. A conservative scan: it keys on JSON string
    values matching 64-hex under a key mentioning ``task`` or ``id``.
    """

    import os

    root = Path(search_root)
    excluded = Path(exclude_dir).resolve()
    found: dict[str, list[str]] = {}

    json_paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune vendored trees in place so the walk never descends into them.
        dirnames[:] = [
            d
            for d in dirnames
            if not _is_skipped_dir(d) and Path(dirpath, d).resolve() != excluded
        ]
        for name in filenames:
            if name.endswith(".json"):
                json_paths.append(Path(dirpath, name))

    for path in sorted(json_paths):
        resolved = path.resolve()
        if excluded in resolved.parents or resolved == excluded:
            continue
        try:
            if path.stat().st_size > _SCAN_MAX_BYTES:
                continue
        except OSError:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            continue

        ids: set[str] = set()

        def walk(node: Any, key: str | None = None) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
            elif (
                isinstance(node, str)
                and key
                and _HEX64_PATTERN.match(node)
                and ("task" in key.lower() or "id" in key.lower())
            ):
                ids.add(node)

        walk(payload)
        if ids:
            found[str(path)] = sorted(ids)
    return found


def _cmd_disjoint(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    eval_ids = [t["task_id"] for t in manifest["tasks"]]
    foreign = collect_foreign_task_ids(args.search_root, Path(args.manifest).parent)

    # Foreign files legitimately share IDs with each other (a lane's receipt and
    # its own manifest cite the same IDs). The claim under test is only that the
    # E12 eval set does not intersect the union of everything else.
    union = sorted({i for ids in foreign.values() for i in ids})
    proof_core = assert_disjoint(
        {"e12_appbench_eval": eval_ids, "all_other_boundaries": union}
    )
    proof = {
        "schema_version": "e12-appbench-disjointness-proof-v1",
        "boundary_id": BOUNDARY_ID,
        "eval_split_hash": manifest["split"]["split_hash"],
        "eval_task_count": len(eval_ids),
        "method": proof_core["method"],
        "search_root": str(args.search_root),
        "foreign_files_scanned": len(foreign),
        "foreign_unique_ids": len(union),
        "intersection_with_eval": sorted(set(eval_ids) & set(union)),
        "pairs_checked": len(proof_core["pairs_checked"]),
        "disjoint": True,
        "e13_openreward_games": {
            "task_ids_available": False,
            "note": (
                "E13's boundary has no task IDs in this checkout, so the adapter's "
                "E12/E13 cross-boundary disjointness rule is vacuously satisfiable "
                "but not yet exercised against real E13 IDs."
            ),
        },
        "does_not_prove": (
            "Set-disjointness across boundaries is not evidence that the tasks were "
            "held out from model training. See heldout_claim in split_manifest.json."
        ),
    }
    out = Path(args.out)
    out.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"  foreign files scanned : {proof['foreign_files_scanned']}")
    print(f"  foreign unique ids    : {proof['foreign_unique_ids']}")
    print(f"  intersection with e12 : {len(proof['intersection_with_eval'])}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    proof = assert_disjoint({"e12_eval": [t["task_id"] for t in manifest["tasks"]]})
    print(f"wrote {out}")
    print(f"  tasks           : {manifest['split']['task_count']}")
    print(f"  split_hash      : {manifest['split']['split_hash']}")
    print(f"  aggregate_sha256: {manifest['aggregate_sha256']}")
    print(f"  disjointness    : {proof['disjoint']}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = verify_manifest(manifest, args.csv)
    print(f"verified {args.manifest}: {result}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the split manifest from the pinned CSV")
    build.add_argument("--csv", default=str(DEFAULT_CSV))
    build.add_argument("--out", default=str(DEFAULT_MANIFEST))
    build.set_defaults(func=_cmd_build)

    verify = sub.add_parser("verify", help="re-derive every hash and compare")
    verify.add_argument("--csv", default=str(DEFAULT_CSV))
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    verify.set_defaults(func=_cmd_verify)

    disjoint = sub.add_parser(
        "disjoint", help="prove the eval IDs collide with no other boundary's IDs"
    )
    disjoint.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    disjoint.add_argument("--search-root", default=str(REPO_ROOT / "outputs"))
    disjoint.add_argument(
        "--out",
        default=str(REPO_ROOT / "outputs" / "e12_appbench" / "disjointness_proof.json"),
    )
    disjoint.set_defaults(func=_cmd_disjoint)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
