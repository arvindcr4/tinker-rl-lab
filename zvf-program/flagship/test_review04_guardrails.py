"""Offline adversarial reproductions for flagship Pro review 04.

Modal decoration is stubbed at import; no remote, billing, training or network
calls are possible. All signer keys, checkpoint files and ledgers are temporary.
"""

# ruff: noqa: PT009, PT027
# Keep this suite executable with stdlib unittest like adjacent offline tests.
from __future__ import annotations
import asyncio
import copy
import hashlib
import importlib
import json
import sys
import sqlite3
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from flagship import campaign_plan_v2 as plan
from flagship import evidence_v2 as evidence
from flagship.campaign_ledger_v2 import CampaignLedger
from flagship import pavlov_tinker_budget as budget
from flagship import build_pavlov_receipt_bundle as bundle
from flagship import build_pavlovs_campaign_manifest as campaign
from flagship.s1 import reference, receipt
from flagship.s1.combine_receipts import combine
from flagship.theory import action_reversal, reward_channel_regimes


class DecorationOnly:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        if name in {"function", "local_entrypoint", "cls"}:
            return lambda *a, **kw: lambda obj: obj
        return lambda *a, **kw: self


def import_modal_module(name):
    fake = types.ModuleType("modal")
    for attr in ("Image", "Secret", "Volume", "Dict"):
        setattr(fake, attr, DecorationOnly())
    fake.App = DecorationOnly
    fake.is_local = lambda: False
    for attr in ("enter", "exit", "concurrent", "asgi_app"):
        setattr(fake, attr, lambda *a, **kw: lambda obj: obj)
    with patch.dict(sys.modules, {"modal": fake}):
        module = importlib.import_module("flagship." + name)
    sys.modules["flagship." + name] = module
    return module


modal_e11 = import_modal_module("modal_e1_e14")
bridge = import_modal_module("modal_tinker_openai_bridge")
proxy = import_modal_module("modal_proxy_evals")


class Review04Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = budget.load_budget()
        self.budget["_portfolio_path"] = str(self.root / "absent-contract.json")

    def artifact(self, name="evidence.json", contents=b"trusted evidence"):
        path = self.root / name
        path.write_bytes(contents)
        return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}

    def signed(self, kind, identity, **fields):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        key = Ed25519PrivateKey.generate()
        root = {
            "test-provider": key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        }
        value = {
            "schema_version": "pavlov-approval-v2",
            "kind": kind,
            "approved": True,
            "status": "approved",
            "identity": identity,
            "issuer": "test-provider",
            "artifacts": [self.artifact()],
            **fields,
        }
        value["signature_hex"] = key.sign(evidence.canonical(value)).hex()
        return value, root

    def identity(self):
        return dict(
            zip(
                evidence.IDENTITY_FIELDS,
                ["campaign", "phase", "suite", "run", "checkpoint", "plan"],
            )
        )

    def test_e11_spoofed_contradictory_success_and_missing_log_are_not_scores(self):
        task = "Prob1"
        p = self.root / task
        p.mkdir()
        (p / f"{task}_sample01.sv").write_text(
            'module TopModule; initial $display("Mismatches: 0 in 100 samples"); endmodule'
        )
        (p / f"{task}_sample01-sv-iv-test.log").write_text(
            "Mismatches: 0 in 100 samples\nMismatches: 12 in 100 samples\n"
        )
        completed = types.SimpleNamespace(returncode=2, stdout="", stderr="")
        with patch.object(modal_e11.subprocess, "run", return_value=completed):
            result = modal_e11._e11_run_harness(self.root, [task, "Missing"])
        self.assertEqual(result["direct_results"], {})
        self.assertEqual(result["outcomes"], {task: "HARNESS_ERROR", "Missing": "MISSING_EVIDENCE"})
        self.assertFalse(result["complete"])

    def test_e11_trusted_candidate_failure_can_be_complete_with_nonzero_make(self):
        task = "Prob1"
        p = self.root / task
        p.mkdir()
        source = p / f"{task}_sample01.sv"
        source.write_bytes(b"invalid module")
        log = p / f"{task}_sample01-sv-iv-test.log"
        log.write_bytes(b"compile failure")
        identity = self.identity()
        signed, roots = self.signed(
            "e11_outcomes",
            identity,
            outcomes=[
                {
                    "task_id": task,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    "outcome": "MODEL_COMPILE_FAILURE",
                }
            ],
        )
        with patch.object(
            modal_e11.subprocess,
            "run",
            return_value=types.SimpleNamespace(returncode=2, stdout="", stderr=""),
        ):
            result = modal_e11._e11_run_harness(
                self.root,
                [task],
                supervisor_receipt=signed,
                trusted_issuers=roots,
                expected_identity=identity,
            )
        self.assertTrue(result["complete"])
        self.assertEqual(result["direct_results"], {task: False})

    def test_bridge_rejects_unset_arbitrary_and_unsigned_sampler(self):
        for config in (
            {},
            {"SERVING_MODE": "final_adapter", "SAMPLER_PATH": "other"},
            {"SERVING_MODE": "final_adapter", "SAMPLER_PATH": bridge.SAMPLER_PATH},
        ):
            with self.assertRaises(RuntimeError):
                bridge.resolve_serving_identity(config, binding={}, trusted_issuers={})

    def test_bridge_signed_sampler_identity_and_base_alias_are_distinct(self):
        for mode, sampler, checkpoint in [
            ("final_adapter", bridge.SAMPLER_PATH, bridge.HF_COMMIT),
            ("base", "tinker://base/frozen", bridge.MODEL_REVISION),
        ]:
            identity = {
                "campaign": bridge.CAMPAIGN_ID,
                "phase": "serving",
                "suite": "bridge",
                "run": sampler,
                "checkpoint": checkpoint,
                "plan": "pavlov-serving-v2",
            }
            signed, roots = self.signed("sampler_export_binding", identity)
            result = bridge.resolve_serving_identity(
                {"SERVING_MODE": mode, "SAMPLER_PATH": sampler},
                binding=signed,
                trusted_issuers=roots,
            )
            self.assertEqual(
                result["model"],
                bridge.MODEL_ALIAS if mode == "final_adapter" else bridge.BASE_MODEL_ALIAS,
            )
            self.assertEqual(len(result["run_manifest_sha256"]), 64)

    def test_partial_billing_never_releases_running_commitment(self):
        raw = {"status": "running", "estimated_cost_usd": 10, "model": self.budget["model"]}
        before = budget._receipt_breakdown(raw, self.budget, source="test")
        after = budget._receipt_breakdown({**raw, "billed_cost_usd": 1}, self.budget, source="test")
        self.assertEqual(before["pending_usd"], Decimal(10))
        self.assertEqual(after["pending_usd"], Decimal(9))
        result = budget.build_ledger(
            self.budget,
            [{**raw, "billed_cost_usd": 1}],
            projection={"cost_usd": 10},
            portfolio_contract=None,
        )
        self.assertEqual(result["projected_usd"], Decimal(20))
        self.assertEqual(result["status"], "REJECT")

    def test_projection_never_prefers_underpriced_explicit_cost(self):
        result = budget._projection_cost(
            self.budget, {"cost_usd": ".01", "tokens": {"sample_tokens": 100_000_000}}
        )
        self.assertGreater(result, Decimal(100))

    def test_ledger_missing_state_and_transaction_failure_and_idempotency(self):
        path = self.root / "ledger.sqlite"
        with self.assertRaises(RuntimeError):
            CampaignLedger(path, campaign="c")
        ledger = CampaignLedger.provision(
            path, campaign="c", maximum_usd=18, reconciled_spend_usd=0
        )
        ledger.reserve("r", 2, identity="manifest")
        with self.assertRaises(RuntimeError):
            with ledger._transaction() as db:
                db.execute("UPDATE requests SET ceiling=0 WHERE id='r'")
                raise RuntimeError("crash before commit")
        self.assertEqual(ledger.snapshot()["reserved_usd"], "2")
        ledger.settle("r", 1)
        ledger.settle("r", 1)
        ledger.fail("r")
        self.assertEqual(ledger.snapshot()["reserved_usd"], "2")
        self.assertEqual(ledger.snapshot()["charged_usd"], "0")
        ledger.reconcile("r", 1)
        ledger.reconcile("r", 1)
        self.assertEqual(ledger.snapshot()["charged_usd"], "1")
        with self.assertRaises(RuntimeError):
            ledger.reserve("r", 2, identity="manifest")
        path.unlink()
        with self.assertRaises(sqlite3.OperationalError):
            ledger.snapshot()

    def test_tracking_failure_cannot_recharge(self):
        obj = bridge.TinkerOpenAIBridge()
        obj.wandb_run = types.SimpleNamespace(
            log=lambda _: (_ for _ in ()).throw(RuntimeError("tracking down"))
        )
        with self.assertWarns(UserWarning):
            obj._track({"event": 1})

    def test_negative_and_content_free_approvals_rejected_in_both_builders(self):
        for predicate in (bundle._license_receipt_valid, campaign._license_receipt_present):
            self.assertFalse(predicate("not approved"))
        for predicate in (bundle._budget_receipt_valid, campaign._budget_receipt_present):
            for amount in (10, True, float("inf")):
                self.assertFalse(
                    predicate(
                        {
                            "authorized": False,
                            "status": "rejected",
                            "sha256": "a" * 64,
                            "maximum_usd": amount,
                        }
                    )
                )
        for value in ("some receipt", {"sha256": "x"}, {"sha256": "a" * 64}):
            self.assertFalse(budget._immutable_reference(value))

    def test_signed_approval_requires_external_trust_context_and_artifact_bytes(self):
        identity = self.identity()
        value, roots = self.signed("budget", identity, maximum_usd=10)
        self.assertTrue(
            evidence.approval_valid(
                value, "budget", trusted_issuers=roots, expected_identity=identity
            )
        )
        self.assertFalse(
            evidence.approval_valid(value, "budget", trusted_issuers={}, expected_identity=identity)
        )
        self.assertFalse(
            evidence.approval_valid(
                value,
                "budget",
                trusted_issuers=roots,
                expected_identity={**identity, "run": "wrong"},
            )
        )
        Path(value["artifacts"][0]["path"]).write_text("changed")
        self.assertFalse(
            evidence.approval_valid(
                value, "budget", trusted_issuers=roots, expected_identity=identity
            )
        )

    def test_model_receipt_revision_must_match_candidate(self):
        contract = {
            "model_candidates": [
                {"role": "primary", "revision": "a" * 40},
                {"role": "other", "revision": "b" * 40},
            ]
        }
        result = campaign._model_revision_status(
            contract, [{"model_revision": {"primary": "c" * 40, "other": "d" * 40}}]
        )
        self.assertFalse(result[0])

    def test_s1_minimal_matching_receipts_cannot_pass(self):
        a = self.root / "a.json"
        b = self.root / "b.json"
        raw = {
            "status": "PASS",
            "intended_cases": [{"verdict": "PASS"}],
            "native_cases": [{"verdict": "NOT_TESTED"}],
        }
        a.write_text(json.dumps({**raw, "stack": "trl"}))
        b.write_text(json.dumps({**raw, "stack": "verl"}))
        self.assertEqual(combine(a, b)["status"], "S1_FAIL")

    def test_controller_inverted_actions_and_history_fixtures_fail(self):
        matrix = receipt._controller_matrix()
        self.assertTrue(receipt.validate_controller_matrix(matrix))
        for row in matrix:
            if row["decision"]["action"] in {"keep", "drop"}:
                row["decision"]["action"] = {"keep": "drop", "drop": "keep"}[
                    row["decision"]["action"]
                ]
        self.assertFalse(receipt.validate_controller_matrix(matrix))

    def test_rejected_ready_and_missing_hashes_are_rejected(self):
        contract = bundle.load_contract()
        value = bundle.build_bundle(contract)
        for row in value["suites"]:
            row["evidence_status"] = "rejected"
            row["launchable"] = False
            row.pop("entry_hash", None)
        value["status"] = "READY"
        value.pop("bundle_hash", None)
        errors = bundle.validate_bundle(value, contract)
        self.assertTrue(any("rejected evidence" in e for e in errors))
        self.assertTrue(any("entry_hash is required" in e for e in errors))
        self.assertIn("bundle_hash is required", errors)

    def test_resume_reextracts_but_rejects_unbound_or_corrupt_generation(self):
        identity = modal_e11._e11_generation_identity(
            prompt_sha256="a" * 64, seed=1, max_tokens=10, temperature=0.2, enable_thinking=True
        )
        raw = "<think>module TopModule; endmodule"
        record = {
            "generation_identity": identity,
            "response_text": raw,
            "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "module_source": "old draft",
        }
        self.assertIsNone(modal_e11._e11_resume_record(record, identity)["module_source"])
        for bad in (
            {**record, "generation_identity": {}},
            {**record, "response_text": "corrupt"},
            {"module_source": "historical"},
        ):
            with self.assertRaises(ValueError):
                modal_e11._e11_resume_record(bad, identity)
        checkpoint = self.root / "record.json"
        modal_e11._e11_atomic_checkpoint(checkpoint, record)
        self.assertEqual(json.loads(checkpoint.read_bytes()), record)

    def test_frozen_task_manifest_rejects_missing_extra_and_changed_prompts(self):
        rows = {"a": [("id", "prompt")]}
        frozen = {"a/id": hashlib.sha256(b"prompt").hexdigest()}
        self.assertEqual(len(modal_e11._e11_validate_prompts(rows, frozen)), 64)
        for bad in (
            {},
            {"a": []},
            {"a": [("id", "changed")]},
            {"a": [("id", "prompt"), ("id", "prompt")]},
        ):
            with self.assertRaises(ValueError):
                modal_e11._e11_validate_prompts(bad, frozen)

    def test_reward_nan_history_and_thresholds_and_interior_plateau(self):
        for rewards in ([float("nan"), 0], [float("inf"), 0]):
            with self.assertRaises(ValueError):
                reference.decide_policy("full_triage", rewards)
        with self.assertRaises(ValueError):
            reference.decide_policy("full_triage", [1, 1], history_successes=-1, history_total=0)
        with self.assertRaises(ValueError):
            reference.decide_policy_observation(
                "full_triage", [None, None], status="missing", history_successes=1, history_total=0
            )
        self.assertTrue(reference.reward_degenerate([0, 1.2e-8]))
        self.assertEqual(reference.decide_policy("symmetric_zvf", [0, 1.2e-8]).action, "escalate")
        self.assertEqual(reference.decide_policy("full_triage", [0.9, 0.9]).action, "recheck")
        with self.assertRaises(ValueError):
            reference.objective_trace(
                arm="grpo",
                rewards=[[float("nan"), 0]],
                logps=[[0], [0]],
                old_logps=[[0], [0]],
                mask=[[1], [1]],
            )

    def test_smaller_reproductions(self):
        self.assertFalse(proxy.exact_proxy_match("1", "10"))
        self.assertTrue(proxy.exact_proxy_match("1.0", "1"))
        self.assertIsNone(proxy.final_choice("A might fit but final answer B"))
        self.assertEqual(proxy.final_choice("B"), "B")
        for n in (43, 100):
            self.assertEqual(
                reward_channel_regimes.regime_receipt(n)["clean_hard"]["primary_rewards"], [0.0] * n
            )
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                action_reversal.validate(action_reversal.Construction(sample_cost=value))
        with self.assertRaises(ValueError):
            action_reversal.validate(action_reversal.Construction(group_size=True))
        self.assertFalse(
            campaign._xlam_only_scope(
                {"non_claims": ["xLAM-only evidence never proves portfolio usefulness"]}
            )
        )
        result = budget.build_ledger(
            self.budget,
            [
                {
                    "status": "completed",
                    "billing_final": True,
                    "billed_cost_usd": 1,
                    "model": self.budget["model"],
                }
            ],
            portfolio_contract=None,
        )
        self.assertEqual(result["projected_usd"], Decimal(1))

    def test_plan_rounds_and_safety_precision(self):
        self.assertEqual([p["total_steps"] for p in plan.PLAN["phases"][:4]], [10, 50, 100, 200])
        self.assertEqual(bundle._successive_halving(), campaign._successive_halving_contract())
        self.assertFalse(plan.safety_feasibility(100)["feasible"])
        self.assertTrue(plan.safety_feasibility(149)["feasible"])
        self.assertAlmostEqual(plan.safety_feasibility(100)["best_case_upper"], 0.0295130496)

    def test_checkpoint_launch_needs_initial_only_running_needs_every_due_step(self):
        identity = {"campaign": "c", "run": "r"}

        def row(step):
            artifact = self.artifact(str(step))
            return {
                "step": step,
                "identity": identity,
                "checkpoint_sha256": artifact["sha256"],
                "export_revision": "b" * 40,
                "artifact": artifact,
            }

        self.assertEqual(
            plan.checkpoint_errors(
                [row(0)],
                stage="launch",
                expected_identity=identity,
                scheduled_steps=[5, 10],
                current_step=0,
                final_step=10,
            ),
            [],
        )
        self.assertTrue(
            plan.checkpoint_errors(
                [row(0), row(10)],
                stage="completion",
                expected_identity=identity,
                scheduled_steps=[5, 10],
                current_step=10,
                final_step=10,
            )
        )
        self.assertTrue(
            plan.checkpoint_errors(
                [{**row(0), "identity": {"campaign": "old", "run": "old"}}],
                stage="launch",
                expected_identity=identity,
                scheduled_steps=[],
                current_step=0,
                final_step=10,
            )
        )

    def test_scientific_completion_cannot_be_promoted_from_status_assertions(self):
        result = plan.evaluate_campaign_completion(
            {"status": "READY", "science_pass": True}, {}, trusted_issuers={}
        )
        self.assertEqual(result["stage"], "BLOCKED")
        self.assertFalse(result["scientific_claim_eligible"])

    def test_api_rejects_ignored_controls_and_propagates_length(self):
        from fastapi.testclient import TestClient

        obj = bridge.TinkerOpenAIBridge()
        obj.identity = {"model": bridge.MODEL_ALIAS, "serving_mode": "final_adapter"}
        obj._authorize = lambda auth: None

        async def complete(**kwargs):
            return {
                "content": "truncated",
                "tool_calls": [],
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "finish_reason": "length",
            }

        obj._complete = complete
        with TestClient(obj.web()) as client:
            base = {"model": bridge.MODEL_ALIAS, "messages": [{"role": "user", "content": "x"}]}
            for extra in (
                {"tool_choice": "required"},
                {"tool_choice": {"type": "function", "function": {"name": "x"}}},
                {"unknown_control": True},
                {"temperature": float("inf")},
            ):
                # JSON transport itself correctly refuses infinity; exercise finite invalid controls here.
                if "temperature" in extra:
                    continue
                self.assertEqual(
                    client.post("/v1/chat/completions", json={**base, **extra}).status_code, 400
                )
            result = client.post("/v1/chat/completions", json=base)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["choices"][0]["finish_reason"], "length")
            for extra in (
                {"max_output_tokens": 0},
                {"max_output_tokens": True},
                {"temperature": -1},
                {"top_p": 2},
                {"unknown": 1},
                {"instructions": ["x"]},
            ):
                self.assertEqual(
                    client.post("/v1/responses", json={"input": "x", **extra}).status_code, 400
                )
            result = client.post("/v1/responses", json={"input": "x"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "incomplete")
            result = client.post("/v1/responses", json={"input": "x", "stream": True})
            self.assertIn("event: response.incomplete", result.text)

    def test_overlong_prompts_rejected_before_reservation(self):
        from fastapi import HTTPException

        obj = bridge.TinkerOpenAIBridge()
        obj.tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda *a, **kw: "prompt",
            encode=lambda *a, **kw: list(range(bridge.MODEL_CONTEXT_TOKENS)),
        )
        obj._reserve = lambda *a: self.fail("overlong prompt reached paid reservation")
        with self.assertRaises(HTTPException) as error:
            asyncio.run(
                obj._complete(
                    messages=[{"role": "user", "content": "x"}],
                    tools=None,
                    max_tokens=10,
                    temperature=0.2,
                    top_p=0.9,
                    stop=None,
                    seed=None,
                    enable_thinking=True,
                )
            )
        self.assertEqual(error.exception.status_code, 400)

    def test_audit_sources_are_declarations_with_same_byte_hashes(self):
        from flagship import e1_e14_completion_gate as gate

        # Derive the accepted inventory labels without reading omitted historical outputs.
        inverse = {value: key for key, value in gate.SOURCE_STATUS_TO_CLASS.items()}
        lanes = [
            {"lane": lane, "benchmark_status": inverse.get(spec["class"], "BLOCKED"), "score": None}
            for lane, spec in gate.LANE_SPECS.items()
        ]
        for lane in ("E1", "E11"):
            lanes = [row for row in lanes if row["lane"] != lane] + [
                {"lane": lane, "benchmark_status": "SCORED_EXACT_SUITE", "score": 50}
            ]
        # Exact source validation itself must never assert verification.
        self.assertEqual(len(gate._declared_exact_source_lanes({"lanes": lanes})), 2)
        for bad in (-50, 500, True, float("inf")):
            changed = copy.deepcopy(lanes)
            next(row for row in changed if row["lane"] == "E1")["score"] = bad
            with self.assertRaises(gate.PackageValidationError):
                gate._declared_exact_source_lanes({"lanes": changed})
        a = self.root / "status.json"
        b = self.root / "followup.json"
        a.write_text(json.dumps({"lanes": lanes}))
        b.write_text("{}")
        result = gate.audit(a, b)
        self.assertEqual(result["verified_exact_source_lanes"], [])
        self.assertFalse(result["scientific_claim_eligible"])
        self.assertEqual(
            result["input_sha256"],
            {
                "status": hashlib.sha256(a.read_bytes()).hexdigest(),
                "followup": hashlib.sha256(b.read_bytes()).hexdigest(),
            },
        )

    def test_partial_invoice_with_explicit_remaining_estimate_is_additive(self):
        raw = {
            "status": "running",
            "model": self.budget["model"],
            "billed_cost_usd": 1,
            "remaining_estimate_usd": 10,
        }
        result = budget._receipt_breakdown(raw, self.budget, source="test")
        self.assertEqual(result["pending_usd"], Decimal(10))
        result = budget._projection_cost(
            self.budget, {"cost_usd": 0.01, "conservative_cost_usd": 10}
        )
        self.assertEqual(result, Decimal(10))

    def test_concurrent_ledger_consumers_share_one_cap(self):
        path = self.root / "shared.sqlite"
        one = CampaignLedger.provision(path, campaign="c", maximum_usd=3, reconciled_spend_usd=0)
        two = CampaignLedger(path, campaign="c")
        one.reserve("a", 2, identity="i")
        with self.assertRaises(RuntimeError):
            two.reserve("b", 2, identity="i")

    def test_execution_completion_checks_actual_coverage_without_claim_promotion(self):
        identity = self.identity()
        train_ids = [f"train-{i}" for i in range(20)]
        eval_ids = [f"eval-{i}" for i in range(14)]
        suites = [f"suite-{i}" for i in range(14)]
        frozen = {
            "identity": identity,
            "plan_sha256": plan.plan_contract()["plan_sha256"],
            "training_task_ids": train_ids,
            "evaluation_task_ids": eval_ids,
            "training_suite_ids": [f"train-suite-{i}" for i in range(12)],
            "evaluation_tasks": dict(zip(eval_ids, suites)),
            "checkpoint_steps": [5],
            "final_step": 5,
            "base_checkpoint": "a" * 64,
            "final_checkpoint": "b" * 64,
            "normalization": {
                suite: {
                    "minimum": 0,
                    "maximum": 1,
                    "direction": "higher",
                    "transform": "fixed_linear",
                    "reference": "preregistered_range",
                    "missing_policy": "incomplete",
                }
                for suite in suites
            },
        }
        episodes = [
            {
                "task_id": task,
                "suite_id": f"train-suite-{i % 12}",
                "domains": [f"domain-{i % 6}"],
                "stateful": True,
                "artifact": True,
            }
            for i, task in enumerate(train_ids)
        ]
        checkpoints = [
            {
                "step": step,
                "identity": {"campaign": identity["campaign"], "run": identity["run"]},
                "checkpoint_sha256": "c" * 64,
                "export_revision": "d" * 40,
                "artifact": self.artifact(f"checkpoint-{step}"),
            }
            for step in [0, 5]
        ]
        for checkpoint in checkpoints:
            checkpoint["checkpoint_sha256"] = checkpoint["artifact"]["sha256"]
        execution = {
            "training_task_ids": train_ids,
            "evaluation_task_ids": eval_ids,
            "selection_task_ids": ["selection"],
            "final_consulted_during_selection": False,
            "batches": [{"episodes": episodes}],
            "checkpoints": checkpoints,
            "final_step": 5,
            "paired_results": [
                {
                    "task_id": task,
                    "suite_id": suite,
                    "base_checkpoint": "a" * 64,
                    "final_checkpoint": "b" * 64,
                    "infrastructure_error": False,
                    "attempt_count": 1,
                    "base_result": 0,
                    "final_result": 1,
                }
                for task, suite in zip(eval_ids, suites)
            ],
            "infrastructure_errors": 0,
        }
        signed, roots = self.signed("campaign_execution", identity, **execution)
        result = plan.evaluate_campaign_completion(signed, frozen, trusted_issuers=roots)
        self.assertEqual(result["stage"], "EXECUTION_COMPLETE")
        self.assertFalse(result["scientific_claim_eligible"])
        for mutation in (
            lambda x: x["batches"][0]["episodes"].pop(),
            lambda x: x["checkpoints"].pop(),
            lambda x: x["paired_results"][0].update(attempt_count=True),
            lambda x: x["paired_results"][0].update(final_checkpoint="wrong"),
        ):
            modified = copy.deepcopy(execution)
            mutation(modified)
            signed, roots = self.signed("campaign_execution", identity, **modified)
            self.assertEqual(
                plan.evaluate_campaign_completion(signed, frozen, trusted_issuers=roots)["stage"],
                "BLOCKED",
            )

    def test_new_paid_e11_is_blocked_before_any_external_call(self):
        with patch.object(
            modal_e11.Path,
            "read_bytes",
            side_effect=AssertionError("must not read launch artifacts"),
        ):
            with self.assertRaisesRegex(RuntimeError, "trusted supervisor"):
                modal_e11.run_e11_full(billed_tinker_usd_before=0)

    def test_safety_planning_blocks_launch_at_100_independent_pairs(self):
        contract = bundle.load_contract()
        counts = {
            suite: 100
            for suite, spec in contract["suite_registry"].items()
            if spec["role"] == "primary_eval"
        }
        contract["safety_planning"] = {
            "independent_pairs_by_suite": counts,
            "method": plan.PLAN["safety_method"],
            "independence_design": "independent_pairs",
            "projected_cost_usd": 1.0,
        }
        self.assertTrue(plan.launch_feasibility_errors(contract))
        contract["safety_planning"]["independent_pairs_by_suite"] = {suite: 149 for suite in counts}
        self.assertEqual(plan.launch_feasibility_errors(contract), [])
        self.assertTrue(
            plan.normalization_errors(
                {
                    "minimum": 0,
                    "maximum": 1,
                    "direction": "higher",
                    "transform": "fit_to_final_results",
                    "reference": "final_model",
                    "missing_policy": "zero",
                }
            )
        )

    def test_sampling_tracking_exception_does_not_charge_twice_or_fail_response(self):
        obj = bridge.TinkerOpenAIBridge()
        obj.maximum_usd = Decimal(18)
        obj.identity = {"run_manifest_sha256": "a" * 64}
        obj.ledger = CampaignLedger.provision(
            self.root / "request.sqlite",
            campaign=bridge.CAMPAIGN_ID,
            maximum_usd=18,
            reconciled_spend_usd=0,
        )
        obj.tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda *a, **kw: "prompt",
            encode=lambda *a, **kw: [1, 2],
            decode=lambda *a, **kw: "answer",
        )
        obj.sampling_client = types.SimpleNamespace(
            sample=lambda *a, **kw: types.SimpleNamespace(
                result=lambda: types.SimpleNamespace(
                    sequences=[types.SimpleNamespace(tokens=[3, 4, 5], stop_reason="stop")]
                )
            )
        )
        obj.wandb_run = types.SimpleNamespace(
            log=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("tracking unavailable"))
        )
        fake = types.ModuleType("tinker")
        fake_types = types.ModuleType("tinker.types")
        fake_types.ModelInput = types.SimpleNamespace(from_ints=lambda value: value)
        fake_types.SamplingParams = lambda **kw: kw
        fake.types = fake_types
        with (
            patch.dict(sys.modules, {"tinker": fake, "tinker.types": fake_types}),
            patch.object(
                bridge,
                "estimate_tinker_usd",
                side_effect=lambda prompt, output: 2 if output == 10 else 1,
            ),
        ):
            with self.assertWarns(UserWarning):
                result = asyncio.run(
                    obj._complete(
                        messages=[{"role": "user", "content": "x"}],
                        tools=None,
                        max_tokens=10,
                        temperature=0.2,
                        top_p=0.9,
                        stop=None,
                        seed=None,
                        enable_thinking=True,
                    )
                )
        self.assertEqual(result["content"], "answer")
        self.assertEqual(obj.ledger.snapshot()["reserved_usd"], "2")
        self.assertEqual(obj.ledger.snapshot()["estimated_usd"], "1")
        self.assertEqual(obj.ledger.snapshot()["charged_usd"], "0")

    def test_rejected_ready_fixture_fails_even_when_all_other_receipts_are_valid(self):
        contract = bundle.load_contract()
        overrides = {
            suite: {"evidence_status": "rejected"}
            for suite, spec in contract["suite_registry"].items()
            if spec["role"] in {"train", "primary_eval"}
        }
        with (
            patch.object(
                bundle,
                "_receipt_completeness",
                return_value={field: True for field in bundle.REQUIRED_RECEIPT_FIELDS},
            ),
            patch.object(bundle, "_budget_guard", return_value=({}, [])),
        ):
            value = bundle.build_bundle(contract, overrides)
            self.assertEqual(value["status"], "BLOCKED")
            self.assertTrue(all(not row["launchable"] for row in value["suites"]))
            value["status"] = "READY"
            value["bundle_hash"] = bundle.sha256_json(bundle._bundle_hash_payload(value))
            errors = bundle.validate_bundle(value, contract)
            self.assertTrue(any("rejected evidence" in error for error in errors))
            self.assertTrue(any("aggregate status" in error for error in errors))

    def test_reserve_is_subtracted_once_from_hard_cap(self):
        self.assertEqual(modal_e11.TINKER_HARD_CAP_USD, 18)
        self.assertEqual(modal_e11.TINKER_OPERATIONAL_CAP_USD, 16.5)
        self.assertEqual(
            modal_e11.TINKER_HARD_CAP_USD - modal_e11.TINKER_SAFETY_RESERVE_USD,
            modal_e11.TINKER_OPERATIONAL_CAP_USD,
        )


if __name__ == "__main__":
    unittest.main()
