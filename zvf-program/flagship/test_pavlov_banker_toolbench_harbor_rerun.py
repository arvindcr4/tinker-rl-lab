from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_banker_toolbench_harbor_rerun import (
    PINNED_TASK_ID,
    HarborRerunReceiptError,
    build_harbor_rerun_receipt,
)


def write_native_task(root: Path) -> Path:
    task = root / "btb-707cba99"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "task.toml").write_text(
        "[environment]\ncpus=4\n[verifier.env]\nLLM_API_KEY='${GEMINI_API_KEY}'\n",
        encoding="utf-8",
    )
    (task / "tests" / "grader.toml").write_text(
        "model='gemini/gemini-3-flash-preview'\nmode='batch'\nrubric_path='/tests/rubric.json'\n"
        "mcp_servers=[{name='mcp-server', transport='stdio', command='/usr/bin/mcp-server'}]\n",
        encoding="utf-8",
    )
    (task / "environment" / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    return task


class HarborRerunReceiptTests(unittest.TestCase):
    def test_receipt_is_fail_closed_and_uses_exact_public_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "tasks.jsonl"
            ids = [PINNED_TASK_ID, *[f"task-{index}" for index in range(99)]]
            manifest.write_text("\n".join(json.dumps({"task_id": item}) for item in ids) + "\n")
            receipt = build_harbor_rerun_receipt(
                manifest_path=manifest,
                task_dir=write_native_task(root),
                harbor_version="0.20.0",
                native_start_state="BUILD_IN_PROGRESS_RETAINED_SINGLE_ATTEMPT",
                credential_presence={"WANDB_API_KEY": "ABSENT", "TINKER_API_KEY": "ABSENT", "GEMINI_API_KEY": "ABSENT"},
                native_start_command="harbor task start-env --path btb-707cba99 --env docker --non-interactive",
                operational_event="one retained native build remains in progress",
            )
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertEqual(receipt["execution"]["score"], None)
        self.assertEqual(receipt["execution"]["paid_calls"], 0)
        self.assertEqual(receipt["source"]["selected_task_id_sha256"], hashlib.sha256(PINNED_TASK_ID.encode()).hexdigest())
        self.assertIn("GEMINI_API_KEY", " ".join(receipt["blockers"]))

    def test_rejects_non_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "tasks.jsonl"
            manifest.write_text(json.dumps({"task_id": PINNED_TASK_ID}) + "\n")
            with self.assertRaises(HarborRerunReceiptError):
                build_harbor_rerun_receipt(
                    manifest_path=manifest,
                    task_dir=write_native_task(root),
                    harbor_version="0.20.0",
                    native_start_state="READY",
                    credential_presence={},
                    native_start_command="harbor task start-env",
                    operational_event="not reached",
                )
