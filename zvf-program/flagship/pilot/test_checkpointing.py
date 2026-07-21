from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from pilot.artifacts import with_fingerprint
from pilot.checkpointing import (
    CheckpointContractError,
    load_checkpoint_bundle,
    load_replay_batch,
    save_checkpoint_bundle,
)
from pilot.flops import PROFILED_STEPS, TrainingFlopLedger
from pilot.protocol import build_screening_plan, load_protocol


class TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        self.assertTrue if False else None
        if not safe_serialization:
            raise AssertionError("test requires safetensors")
        path.mkdir(parents=True, exist_ok=True)
        save_file({"weight": self.weight.detach().cpu()}, path / "adapter_model.safetensors")
        (path / "adapter_config.json").write_text(json.dumps({"type": "tiny"}) + "\n")


def load_tiny(model: TinyAdapter, path: Path) -> None:
    state = load_file(path / "adapter_model.safetensors")
    model.load_state_dict(state)


class CheckpointingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()
        self.plan = build_screening_plan(self.protocol, next(self.protocol.screening_units()))
        self.corpus = with_fingerprint(
            {
                "fingerprint_seed": "corpus",
                "charged_generated_tokens": 6400,
                "wandb": {
                    "run_id": "corpus-run",
                    "run_url": "https://wandb.ai/entity/tinker-rl-lab/runs/corpus-run",
                    "entity": "entity",
                    "project": "tinker-rl-lab",
                },
            }
        )

    def ledger(self, through: int) -> TrainingFlopLedger:
        ledger = TrainingFlopLedger()
        for step in range(1, through + 1):
            phases = None
            if step in PROFILED_STEPS:
                phases = {
                    "policy_forward": 1.0,
                    "optimizer_backward": 2.0,
                    "diagnostic_backward": 3.0,
                }
            ledger.add_step(
                step=step,
                active_tokens=8,
                padded_tokens=10,
                phase_flops=phases,
            )
        return ledger

    def test_checkpoint_round_trip_restores_adapter_optimizer_scheduler_and_rng(self) -> None:
        model = TinyAdapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=100
        )
        optimizer.zero_grad()
        model.weight.sum().backward()
        optimizer.step()
        scheduler.step()
        expected_weight = model.weight.detach().clone()
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint-20"
            evaluation_zero = Path(temporary) / "step-000.jsonl"
            evaluation_twenty = Path(temporary) / "step-020.jsonl"
            evaluation_zero.write_text('{"index": 0}\n')
            evaluation_twenty.write_text('{"index": 0}\n')
            manifest = save_checkpoint_bundle(
                destination=root,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=20,
                plan=self.plan,
                corpus=self.corpus,
                receipts=[{"step": step} for step in range(1, 21)],
                evaluations=[{"step": 0}, {"step": 20}],
                flop_ledger=self.ledger(20),
                source_hashes={"source.py": "a" * 64},
                evaluation_files={0: evaluation_zero, 20: evaluation_twenty},
            )
            self.assertEqual(manifest["step"], 20)
            with torch.no_grad():
                model.weight.add_(10)
            optimizer.state.clear()
            training_state, ledger, restored_manifest = load_checkpoint_bundle(
                root=root,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                plan=self.plan,
                corpus=self.corpus,
                adapter_loader=load_tiny,
            )
            torch.testing.assert_close(model.weight, expected_weight)
            self.assertEqual(optimizer.state_dict()["param_groups"], expected_optimizer["param_groups"])
            self.assertEqual(training_state["step"], 20)
            self.assertEqual(ledger.profiled_steps, [1, 20])
            self.assertEqual(restored_manifest["fingerprint"], manifest["fingerprint"])

    def test_checkpoint_refuses_receipt_mismatch_and_tampering(self) -> None:
        model = TinyAdapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=100)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint-20"
            evaluation_zero = Path(temporary) / "step-000.jsonl"
            evaluation_zero.write_text('{"index": 0}\n')
            with self.assertRaisesRegex(CheckpointContractError, "receipt count"):
                save_checkpoint_bundle(
                    destination=root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=20,
                    plan=self.plan,
                    corpus=self.corpus,
                    receipts=[],
                    evaluations=[],
                    flop_ledger=self.ledger(20),
                    source_hashes={"source.py": "a" * 64},
                    evaluation_files={},
                )
            save_checkpoint_bundle(
                destination=root,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=20,
                plan=self.plan,
                corpus=self.corpus,
                receipts=[{"step": step} for step in range(1, 21)],
                evaluations=[],
                flop_ledger=self.ledger(20),
                source_hashes={"source.py": "a" * 64},
                evaluation_files={},
            )
            with (root / "training_state.json").open("a", encoding="utf-8") as stream:
                stream.write("tamper")
            with self.assertRaisesRegex(CheckpointContractError, "hash mismatch"):
                load_checkpoint_bundle(
                    root=root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    plan=self.plan,
                    corpus=self.corpus,
                    adapter_loader=load_tiny,
                )

    def test_replay_loader_requires_schema_fingerprint_and_finite_old_logps(self) -> None:
        payload = {
            "schema_version": "flagship-pilot-replay-group-v1",
            "group": {"fingerprint": "f" * 64},
            "prompt_ids": torch.ones((8, 2), dtype=torch.long),
            "prompt_mask": torch.ones((8, 2), dtype=torch.long),
            "completion_ids": torch.ones((8, 3), dtype=torch.long),
            "completion_mask": torch.ones((8, 3)),
            "rewards": torch.zeros(8),
            "active_rows": torch.ones(8, dtype=torch.bool),
            "old_logps": torch.zeros((8, 3)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "group.pt"
            torch.save(payload, path)
            batch = load_replay_batch(path, expected_fingerprint="f" * 64)
            self.assertEqual(batch.old_logps.shape, (8, 3))
            with self.assertRaisesRegex(CheckpointContractError, "fingerprint mismatch"):
                load_replay_batch(path, expected_fingerprint="e" * 64)


if __name__ == "__main__":
    unittest.main()
