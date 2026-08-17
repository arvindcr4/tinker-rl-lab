#!/usr/bin/env python3
"""Build a content-backed E11 decontamination and final-launch receipt.

The trained sampler used only API-Bank RLVR + SWE-Gym.  The training run's
W&B record retained the aggregate dataset revision and suite IDs, but not the
dirty-checkout source blob that transformed the rows.  This tool therefore
uses the safer boundary: it loads *every row* from both historically cached,
pinned source snapshots and compares every task-bearing source field against
all 312 VerilogEval prompts and references.  That source envelope is a strict
superset of the 512 selected training rows and does not depend on recreating
the missing transform implementation.

Only aggregate hashes and collision counts are emitted; no benchmark
solution or training-source row is copied into the public receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


SCHEMA_VERSION = "pavlov-e11-decontamination-v1"
RUN_DATE = "2026-08-16"
TRAINING_RUN_ID = "cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0"
TRAINING_WANDB_RUN_ID = "bsv8vx04"
TRAINING_DATASET_REVISION = "bd1f0db6056091e99d5e28e3a7b3d05d77b21876ad65e5b31bf6f1b470e73e5c"
TRAINING_SOURCE_REVISIONS = {
    "Simu-Env/API-Bank-RLVR": "bf67c42626f02c305514b1df16dcabc5fc616333",
    "SWE-Gym/SWE-Gym": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
}
TRAINING_WANDB_DIR = "wandb/run-20260809_140744-bsv8vx04"
E11_REVISION = "c498220d0a52248f8e3fdffe279075215bde2da6"
E11_DATASETS = ("code-complete-iccad2023", "spec-to-rtl")
NGRAM_SIZE = 8
NEAR_DUPLICATE_THRESHOLD = 0.80

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
FINAL_SAMPLER_PATH = (
    "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/"
    "sampler_weights/seed809_final"
)
FINAL_HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-"
    "tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
FINAL_HF_REVISION = "checkpoint-seed809-stepfinal-9f777c4018b6"
FINAL_HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_hash(values: Iterable[str]) -> str:
    return _sha256_text("\n".join(sorted(values)) + "\n")


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", value.lower()))


def _ngrams(value: str) -> set[str]:
    tokens = value.split()
    if not tokens:
        return set()
    width = min(NGRAM_SIZE, len(tokens))
    return {
        _sha256_text(" ".join(tokens[index : index + width]))
        for index in range(len(tokens) - width + 1)
    }


def _max_near_duplicate(
    training_documents: Sequence[tuple[str, str]],
    evaluation_documents: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    training_grams: list[set[str]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for index, (_, text) in enumerate(training_documents):
        grams = _ngrams(_normalise(text))
        training_grams.append(grams)
        for gram in grams:
            inverted[gram].append(index)

    maximum = 0.0
    threshold_matches = 0
    for _, text in evaluation_documents:
        eval_grams = _ngrams(_normalise(text))
        intersections: dict[int, int] = defaultdict(int)
        for gram in eval_grams:
            for train_index in inverted.get(gram, ()):
                intersections[train_index] += 1
        for train_index, intersection in intersections.items():
            denominator = len(eval_grams) + len(training_grams[train_index]) - intersection
            similarity = intersection / denominator if denominator else 1.0
            maximum = max(maximum, similarity)
            threshold_matches += int(similarity >= NEAR_DUPLICATE_THRESHOLD)
    return {
        "algorithm": f"lowercase_alnum_word_{NGRAM_SIZE}gram_jaccard",
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "maximum_similarity": round(maximum, 8),
        "pairs_at_or_above_threshold": threshold_matches,
    }


def _wandb_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _verify_local_training_receipt(root: Path) -> dict[str, Any]:
    run_dir = root / TRAINING_WANDB_DIR
    config_path = run_dir / "files/config.yaml"
    metadata_path = run_dir / "files/wandb-metadata.json"
    output_path = run_dir / "files/output.log"
    for required in (config_path, metadata_path, output_path):
        if not required.is_file():
            raise RuntimeError(f"missing local W&B training receipt: {required}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("local W&B training config is malformed")
    expected = {
        "dataset_revision": TRAINING_DATASET_REVISION,
        "seed": 809,
        "training_suite_ids": ["api_bank_rlvr_train", "swe_gym_train"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    observed = {key: _wandb_value(config, key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"local W&B training identity drifted: {observed!r}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("startedAt") != "2026-08-09T08:37:44.026800Z":
        raise RuntimeError("local W&B start-time receipt drifted")
    if metadata.get("program") != "-m platform_tinker.tinkerrl.grpo_cli":
        raise RuntimeError("local W&B program receipt drifted")
    output = output_path.read_text(encoding="utf-8")
    if TRAINING_RUN_ID not in output or FINAL_HF_REPO not in output:
        raise RuntimeError("local W&B output does not bind the expected Tinker run")
    if "Published checkpoint step=final:" not in output:
        raise RuntimeError("local W&B output lacks the final checkpoint publication")

    return {
        "run_id": TRAINING_WANDB_RUN_ID,
        "dataset_revision": TRAINING_DATASET_REVISION,
        "config_sha256": _sha256_text(config_path.read_text(encoding="utf-8")),
        "metadata_sha256": _sha256_text(metadata_path.read_text(encoding="utf-8")),
        "output_sha256": _sha256_text(output),
        "started_at": metadata["startedAt"],
        "recorded_git_commit": metadata.get("git", {}).get("commit"),
        "source_capture_boundary": (
            "W&B recorded the dirty checkout's base Git commit and aggregate dataset "
            "revision, but did not upload the dirty source diff."
        ),
    }


def _source_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _load_training_source_envelope(
) -> tuple[list[tuple[str, str]], list[str], dict[str, Any]]:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from datasets import load_dataset

    documents: list[tuple[str, str]] = []
    source_ids: list[str] = []
    row_counts: dict[str, int] = {}

    api_revision = TRAINING_SOURCE_REVISIONS["Simu-Env/API-Bank-RLVR"]
    api = load_dataset("Simu-Env/API-Bank-RLVR", revision=api_revision)
    api_rows = 0
    for split_name in ("train", "validation"):
        for row_index, raw in enumerate(api[split_name]):
            row = dict(raw)
            extra = json.loads(str(row["extra_info"]))
            source_id = f"api_bank_rlvr_train:{extra.get('index')}"
            source_ids.append(source_id)
            for field in ("prompt", "ground_truth", "extra_info"):
                documents.append((f"{source_id}:{split_name}:{row_index}:{field}", _source_text(row[field])))
            api_rows += 1
    row_counts["Simu-Env/API-Bank-RLVR"] = api_rows

    swe_revision = TRAINING_SOURCE_REVISIONS["SWE-Gym/SWE-Gym"]
    swe = load_dataset("SWE-Gym/SWE-Gym", split="train", revision=swe_revision)
    for row_index, raw in enumerate(swe):
        row = dict(raw)
        source_id = f"swe_gym_train:{row.get('instance_id')}"
        source_ids.append(source_id)
        for field in (
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "hints_text",
            "patch",
            "test_patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ):
            documents.append((f"{source_id}:train:{row_index}:{field}", _source_text(row.get(field))))
    row_counts["SWE-Gym/SWE-Gym"] = len(swe)

    expected_counts = {
        "Simu-Env/API-Bank-RLVR": 597,
        "SWE-Gym/SWE-Gym": 2438,
    }
    if row_counts != expected_counts:
        raise RuntimeError(f"training-source envelope row counts drifted: {row_counts!r}")
    source_receipts = {
        "revisions": dict(TRAINING_SOURCE_REVISIONS),
        "row_counts": row_counts,
        "document_count": len(documents),
        "boundary": (
            "All task-bearing fields from every row in both pinned sources; this is a "
            "strict superset of the 512 rows selected for training."
        ),
    }
    return documents, source_ids, source_receipts


def _load_e11(checkout: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    prompts: list[tuple[str, str]] = []
    references: list[tuple[str, str]] = []
    task_ids: list[str] = []
    for dataset in E11_DATASETS:
        directory = checkout / f"dataset_{dataset}"
        for prompt_path in sorted(directory.glob("*_prompt.txt")):
            problem_id = prompt_path.name[: -len("_prompt.txt")]
            key = f"verilog_eval/{dataset}/{problem_id}"
            reference_path = directory / f"{problem_id}_ref.sv"
            if not reference_path.is_file():
                raise RuntimeError(f"missing E11 reference artifact for {key}")
            prompts.append((key, prompt_path.read_text(encoding="utf-8")))
            references.append((key, reference_path.read_text(encoding="utf-8")))
            task_ids.append(key)
    if len(prompts) != 312:
        raise RuntimeError(f"expected 312 exact E11 tasks, found {len(prompts)}")
    return prompts, references, task_ids


def build_decontamination_receipt(checkout: Path) -> dict[str, Any]:
    training_wandb = _verify_local_training_receipt(_repo_root())
    training_documents, training_ids, source_receipts = _load_training_source_envelope()
    eval_prompts, eval_references, eval_ids = _load_e11(checkout)

    training_document_hashes = {_sha256_text(text) for _, text in training_documents}
    eval_prompt_hashes = {_sha256_text(text) for _, text in eval_prompts}
    eval_reference_hashes = {_sha256_text(text) for _, text in eval_references}
    normalised_training = {_sha256_text(_normalise(text)) for _, text in training_documents}
    normalised_eval = {_sha256_text(_normalise(text)) for _, text in eval_prompts + eval_references}

    comparisons = {
        "task_or_source_id_overlap_count": len(set(training_ids).intersection(eval_ids)),
        "exact_source_document_to_eval_prompt_overlap_count": len(
            training_document_hashes.intersection(eval_prompt_hashes)
        ),
        "exact_source_document_to_eval_reference_overlap_count": len(
            training_document_hashes.intersection(eval_reference_hashes)
        ),
        "cross_content_normalised_overlap_count": len(normalised_training.intersection(normalised_eval)),
        "near_duplicate": _max_near_duplicate(
            training_documents,
            eval_prompts + eval_references,
        ),
    }
    verified = all(
        comparisons[key] == 0
        for key in (
            "task_or_source_id_overlap_count",
            "exact_source_document_to_eval_prompt_overlap_count",
            "exact_source_document_to_eval_reference_overlap_count",
            "cross_content_normalised_overlap_count",
        )
    ) and comparisons["near_duplicate"]["pairs_at_or_above_threshold"] == 0

    identity = {
        "schema_version": SCHEMA_VERSION,
        "training": {
            "tinker_run_id": TRAINING_RUN_ID,
            "wandb_run_id": TRAINING_WANDB_RUN_ID,
            "dataset_revision": TRAINING_DATASET_REVISION,
            "suite_ids": ["api_bank_rlvr_train", "swe_gym_train"],
            "selected_training_row_count": 512,
            "source_envelope": source_receipts,
            "source_id_manifest_sha256": _manifest_hash(training_ids),
            "source_document_manifest_sha256": _manifest_hash(training_document_hashes),
            "wandb_receipt": training_wandb,
        },
        "evaluation": {
            "suite_id": "verilog_eval",
            "revision": E11_REVISION,
            "datasets": list(E11_DATASETS),
            "task_count": len(eval_prompts),
            "task_id_manifest_sha256": _manifest_hash(eval_ids),
            "prompt_manifest_sha256": _manifest_hash(eval_prompt_hashes),
            "reference_manifest_sha256": _manifest_hash(eval_reference_hashes),
        },
        "comparisons": comparisons,
    }
    receipt_id = hashlib.sha1(_canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        **identity,
        "recorded_at": _utc_now(),
        "status": "VERIFIED" if verified else "BLOCKED",
        "receipt_id": receipt_id,
        "visibility": "private",
        "safe_public_artifact": True,
        "contains_raw_training_or_eval_content": False,
        "blockers": [] if verified else ["training/evaluation content collision detected"],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=root / "outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / f"outputs/modal_e1_e14/{RUN_DATE}/e11",
    )
    args = parser.parse_args(argv)

    receipt = build_decontamination_receipt(args.checkout.resolve())
    _write_json(args.output_dir / "decontamination_receipt.json", receipt)

    if receipt["status"] != "VERIFIED":
        print(json.dumps({"status": receipt["status"], "blockers": receipt["blockers"]}, indent=2))
        return 2

    flagship = Path(__file__).resolve().parent
    if str(flagship.parent) not in sys.path:
        sys.path.insert(0, str(flagship.parent))
    from flagship.pavlov_verilog_eval_split_manifest import build_split_manifest_receipt

    split_receipt = build_split_manifest_receipt(
        args.checkout.resolve(),
        decontamination={
            "status": "verified",
            "receipt_id": receipt["receipt_id"],
            "visibility": receipt["visibility"],
            "safe_public_artifact": receipt["safe_public_artifact"],
            "url": None,
        },
    )
    _write_json(args.output_dir / "split_manifest_receipt.json", split_receipt)

    split_ready = split_receipt.get("status") == "READY"
    launch = {
        "schema_version": "pavlov-modal-e11-final-launch-preflight-v1",
        "recorded_at": _utc_now(),
        "lane": "E11",
        "suite_id": "verilog_eval",
        "suite_role": "primary_eval",
        "status": "READY" if split_ready else "BLOCKED",
        "score": None,
        "is_model_score": False,
        "launch_allowed": split_ready,
        "claim_boundary": "Launch preflight only; no final-checkpoint E11 score exists yet.",
        "decontamination": {
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": _sha256_text(_canonical_json(receipt)),
            "path": str((args.output_dir / "decontamination_receipt.json").relative_to(root)),
        },
        "split_manifest": {
            "status": split_receipt.get("status"),
            "receipt_sha256": _sha256_text(_canonical_json(split_receipt)),
            "path": str((args.output_dir / "split_manifest_receipt.json").relative_to(root)),
        },
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": FINAL_SAMPLER_PATH,
            "hf_repo": FINAL_HF_REPO,
            "hf_revision": FINAL_HF_REVISION,
            "hf_commit": FINAL_HF_COMMIT,
            "training_wandb_run_id": TRAINING_WANDB_RUN_ID,
            "training_tinker_run_id": TRAINING_RUN_ID,
        },
        "blockers": [] if split_ready else split_receipt.get("validation", {}).get("blockers", []),
    }
    launch["receipt_sha256"] = _sha256_text(_canonical_json(launch))
    launch_path = args.output_dir / "launch_preflight_receipt.json"
    _write_json(launch_path, launch)
    print(
        json.dumps(
            {
                "status": launch["status"],
                "decontamination_receipt_id": receipt["receipt_id"],
                "selected_training_rows": receipt["training"]["selected_training_row_count"],
                "source_envelope_rows": receipt["training"]["source_envelope"]["row_counts"],
                "evaluation_tasks": receipt["evaluation"]["task_count"],
                "near_duplicate": receipt["comparisons"]["near_duplicate"],
                "launch_receipt": str(launch_path),
            },
            indent=2,
        )
    )
    return 0 if split_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
