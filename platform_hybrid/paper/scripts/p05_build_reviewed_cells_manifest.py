#!/usr/bin/env python3
"""Build the immutable manifest for paper 05's two reviewed tool-use cells.

Everything computable from the repository is recomputed here (file SHA-256,
git blob IDs, step-level ZVF/reward recomputation, candidate runner source).
Hugging Face revisions were retrieved on 2026-10-19 via the public Hub API and
are recorded as constants; re-verify them with
``https://huggingface.co/api/models/<repo>`` before relying on them.
Fields that no artifact supplies are written as ``"UNBOUND"`` with a reason.

Usage: python3 platform_hybrid/paper/scripts/p05_build_reviewed_cells_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = "platform_hybrid/experiments/tinker-runs/results"
OUT = REPO / "platform_hybrid/paper/manifests/p05_reviewed_cells_manifest.json"

# Earliest committed version of the runner whose experiment table contains the
# two reviewed tags (cross_tool_qwen3-32b, cross_tool_llama-8b-inst).
RUNNER_COMMIT = "c68110a252edee61dc0d800a07f74ecf72794efa"
RUNNER_PATH = "experiments/tinker-runs/scripts/tinker_parallel_runner.py"

# Retrieved 2026-10-19 from https://huggingface.co/api/models/<repo>.
HF = {
    "cross_tool_qwen3-32b": {
        "repo": "arvindcr4/tinker-rl-bench-cross_tool_qwen3-32b",
        "revision": "3f95b71094cd92c6f2107c103dee73979144494d",
        "files": ["README.md", "results.json", "step_log.json"],
    },
    "cross_tool_llama-8b-inst": {
        "repo": "arvindcr4/tinker-rl-bench-cross_tool_llama-8b-inst",
        "revision": "4f25a91dc334d3c27a7dd050924c8c97221f675c",
        "files": ["README.md", "results.json", "step_log.json"],
    },
}
HF_RESULTS_SHA256 = {  # sha256 of results.json downloaded from the Hub, 2026-10-19
    "cross_tool_qwen3-32b": "55f0b6b237cab0ed0f80c280ba1ad282f1664668ea3499f890574d04acb8f4f8",
    "cross_tool_llama-8b-inst": "d4babb2f92318c0bfa6b3b4c6ecffaaf4e234864a4c0a737f83b6f23cf622ea3",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def function_source(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    nxt = text.find("\ndef ", start + 1)
    return text[start:nxt] if nxt != -1 else text[start:]


def main() -> int:
    runner = git("show", f"{RUNNER_COMMIT}:{RUNNER_PATH}")
    runner_blob = git("rev-parse", f"{RUNNER_COMMIT}:{RUNNER_PATH}")
    tool_src = function_source(runner, "get_tool_use")
    reward_src = function_source(runner, "reward_tool")
    system_start = runner.index("SYSTEM_PROMPT_TOOL")
    system_src = runner[system_start : runner.index('"""', runner.index('"""', system_start) + 3) + 3]

    cells = []
    for tag in ("cross_tool_qwen3-32b", "cross_tool_llama-8b-inst"):
        rel = f"{RESULTS}/{tag}.json"
        raw = (REPO / rel).read_bytes()
        rec = json.loads(raw)
        steps = rec["step_log"]
        cells.append(
            {
                "tag": tag,
                "model": rec["model"],
                "local_artifact": {
                    "path": rel,
                    "sha256": sha256(raw),
                    "git_blob_at_HEAD": git("rev-parse", f"HEAD:{rel}"),
                },
                "hub_mirror": {
                    **HF[tag],
                    "results_json_sha256": HF_RESULTS_SHA256[tag],
                    "results_json_identical_to_local": HF_RESULTS_SHA256[tag] == sha256(raw),
                    "contains_adapter_weights": False,
                },
                "tinker_run_id": rec["run_id"],
                "tinker_checkpoint_uri": rec["checkpoint"],
                "config": {k: rec[k] for k in ("seed", "rank", "lr", "group", "steps")},
                "recorded_timestamp": rec.get("timestamp"),
                "recomputation_from_stored_step_log": {
                    "logged_steps": len(steps),
                    "steps_with_reward_0": sum(1 for s in steps if s["reward"] == 0.0),
                    "steps_with_zvf_1": sum(1 for s in steps if s["zvf"] == 1.0),
                    "mean_reward": sum(s["reward"] for s in steps) / len(steps),
                    "mean_zvf": sum(s["zvf"] for s in steps) / len(steps),
                },
                "raw_per_group_reward_vectors": "UNBOUND: not stored locally or on the Hub; "
                "only per-step mean reward and per-step ZVF were logged, so ZVF cannot be "
                "recomputed independently from raw rewards",
                "raw_completions": "UNBOUND: not stored; parser/template failure cannot be "
                "distinguished from policy failure",
            }
        )

    manifest = {
        "schema": "p05-reviewed-cells-manifest/1",
        "generated_by": "platform_hybrid/paper/scripts/p05_build_reviewed_cells_manifest.py",
        "cells": cells,
        "candidate_runner": {
            "commit": RUNNER_COMMIT,
            "path_at_commit": RUNNER_PATH,
            "git_blob": runner_blob,
            "binding": "CANDIDATE: earliest committed source whose experiment list contains "
            "both tags; that this exact source executed the runs is not proven",
            "prompt_set": {
                "function": "get_tool_use",
                "n_prompts": tool_src.count('{"name"'),
                "source_sha256": sha256(tool_src.encode()),
                "chat_template": "Qwen ChatML markers (<|im_start|>) hard-coded for every "
                "model, including Llama-3.1-8B-Instruct",
            },
            "system_prompt_sha256": sha256(system_src.encode()),
            "reward_function": {
                "function": "reward_tool",
                "source_sha256": sha256(reward_src.encode()),
                "semantics": "0.3 if json.loads(response.strip()) succeeds, +0.4 tool name "
                "match, +0.3 expected argument keys present; 0.0 otherwise",
            },
            "batch_prompts_per_step": 2,
        },
        "unbound_fields": {
            "evaluator_revision": "UNBOUND: no evaluator beyond the in-runner reward exists for "
            "these cells; held-out evaluation was not run",
            "checkpoint_revision": "UNBOUND beyond the tinker:// sampler URI; managed backend",
            "tinker_sdk_version": "CONFLICT: ethics text states tinker==0.16.1; Hub model "
            "cards state Tinker API v0.18.1",
            "dataset_card": "Hub model cards list openai/gsm8k for these tool_use runs; the "
            "candidate runner uses the 10 in-source prompts above",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
