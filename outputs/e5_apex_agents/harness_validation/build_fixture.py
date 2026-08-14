#!/usr/bin/env python3
"""Build a SYNTHETIC harness-validation fixture for the Archipelago grading runner.

SYNTHETIC FIXTURE — NOT APEX-Agents BENCHMARK DATA.
Nothing here comes from `mercor/apex-agents`. The snapshots are derived from the
Archipelago repo's own `examples/simple_task/original_snapshot.zip`, plus one
file this script writes. The purpose is solely to prove the native grading
runner executes end to end (snapshot diff -> helper -> programmatic verifier ->
scoring method -> grades.json). It produces NO model score.

Verifiers used are `content_length_check`, which is registered with
`eval_types=[EvalType.PROGRAMMATIC]` -> no LLM judge, no paid API call.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIPELAGO = HERE.parent / "sprint.8cEFaN" / "archipelago"
FIXTURES = HERE / "fixtures"

# 12 words, 1 sentence, 1 paragraph, 1 line. Deterministic by construction.
SUMMARY_WORDS = [
    "The",
    "gorilla",
    "image",
    "is",
    "located",
    "at",
    "animals",
    "xk92m",
    "qz7fw",
    "png",
    "in",
    "filesystem",
]
SUMMARY_TEXT = " ".join(SUMMARY_WORDS) + "."
EXPECTED_WORD_COUNT = len(SUMMARY_WORDS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_snapshots() -> dict[str, str]:
    src = ARCHIPELAGO / "examples" / "simple_task" / "original_snapshot.zip"
    if not src.is_file():
        raise SystemExit(f"missing upstream example snapshot: {src}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    initial = FIXTURES / "initial_snapshot.zip"
    final = FIXTURES / "final_snapshot.zip"
    shutil.copyfile(src, initial)

    # Final snapshot == initial snapshot + one created file.
    with zipfile.ZipFile(initial) as zin, zipfile.ZipFile(
        final, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            zout.writestr(info, zin.read(info.filename))
        zout.writestr("filesystem/summary.txt", SUMMARY_TEXT)

    return {
        "upstream_example_snapshot_sha256": sha256(src),
        "initial_snapshot_sha256": sha256(initial),
        "final_snapshot_sha256": sha256(final),
    }


def build_configs() -> None:
    # Minimal trajectory. `status` must be a member of AgentStatus.
    (FIXTURES / "trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Find the gorilla image and write summary.txt.",
                    },
                    {"role": "assistant", "content": SUMMARY_TEXT},
                ],
                "output": {"source": "harness_validation_synthetic_fixture"},
                "status": "completed",
                "time_elapsed": 0.0,
            },
            indent=2,
        )
    )

    # llm_judge_model is a required field on GradingSettings but is never read:
    # every verifier below is PROGRAMMATIC. The sentinel makes an accidental
    # LLM call fail loudly instead of silently billing a provider.
    (FIXTURES / "grading_settings.json").write_text(
        json.dumps(
            {
                "llm_judge_model": "unused/no-llm-judge-in-this-run",
                "llm_judge_extra_args": None,
            },
            indent=2,
        )
    )

    (FIXTURES / "eval_configs.json").write_text(
        json.dumps(
            [
                {
                    "eval_config_id": "ec_content_length",
                    "eval_config_name": "Content Length Check (programmatic)",
                    "eval_defn_id": "content_length_check",
                    "eval_config_values": {},
                }
            ],
            indent=2,
        )
    )

    def verifier(idx: int, vid: str, values: dict) -> dict:
        return {
            "verifier_id": vid,
            "verifier_version": 1,
            "world_id": None,
            "task_id": "harness_validation_synthetic",
            "eval_config_id": "ec_content_length",
            "verifier_values": values,
            "verifier_index": idx,
            "verifier_dependencies": None,
        }

    (FIXTURES / "verifiers.json").write_text(
        json.dumps(
            [
                # Expected PASS: the created file has >= 5 words.
                verifier(
                    0,
                    "ver_pass_at_least",
                    {
                        "is_primary_objective": True,
                        "content_source": "Created/Modified Files",
                        "target_file": "summary.txt",
                        "metric_type": "Word Count",
                        "comparison_type": "At least",
                        "min_value": 5,
                        "aggregation_mode": "sum",
                    },
                ),
                # Expected FAIL: proves the harness discriminates rather than
                # returning 1.0 unconditionally.
                verifier(
                    1,
                    "ver_fail_at_least",
                    {
                        "is_primary_objective": False,
                        "content_source": "Created/Modified Files",
                        "target_file": "summary.txt",
                        "metric_type": "Word Count",
                        "comparison_type": "At least",
                        "min_value": 10_000,
                        "aggregation_mode": "sum",
                    },
                ),
                # Expected PASS: exact word count -> proves the snapshot diff
                # surfaced the created file's real content, not a placeholder.
                verifier(
                    2,
                    "ver_pass_exact",
                    {
                        "is_primary_objective": False,
                        "content_source": "Created/Modified Files",
                        "target_file": "summary.txt",
                        "metric_type": "Word Count",
                        "comparison_type": "Exactly",
                        "expected_value": EXPECTED_WORD_COUNT,
                        "tolerance": 0,
                        "aggregation_mode": "sum",
                    },
                ),
            ],
            indent=2,
        )
    )

    (FIXTURES / "scoring_config.json").write_text(
        json.dumps(
            {
                "scoring_config_id": "sc_harness_validation",
                "scoring_config_name": "Task Score Unweighted + Universal Penalty",
                "scoring_defn_id": "task_score_unweighted_and_universal_penalty",
                "scoring_config_values": {
                    "task_primary_objective_scaling_factor": 2.0,
                    "task_non_primary_objective_scaling_factor": 1.0,
                    "task_negative_scaling_factor": 2.0,
                    "universal_penalty_cap": 0.2,
                    "final_score_ceiling": 1.0,
                    "final_score_floor": 0.0,
                },
            },
            indent=2,
        )
    )


def main() -> None:
    digests = build_snapshots()
    build_configs()
    manifest = {
        "fixture_kind": "SYNTHETIC — not APEX-Agents benchmark data",
        "purpose": "harness_validation of the Archipelago native grading runner",
        "is_model_score": False,
        "expected_word_count": EXPECTED_WORD_COUNT,
        "expected_verifier_scores": {
            "ver_pass_at_least": 1.0,
            "ver_fail_at_least": 0.0,
            "ver_pass_exact": 1.0,
        },
        **digests,
    }
    (FIXTURES / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
