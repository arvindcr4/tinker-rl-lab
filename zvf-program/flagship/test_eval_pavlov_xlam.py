from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from flagship import eval_pavlov_xlam as harness
from flagship.eval_pavlov_xlam import maximum_eval_cost


_VALID_PROVENANCE_ARGS = [
    "--dataset-revision",
    "a" * 40,
    "--split-manifest-sha256",
    "b" * 64,
    "--task-id-sha256",
    "c" * 64,
    "--license-id",
    "spdx:Apache-2.0",
    "--license-receipt",
    "license-receipt-v1",
    "--decontamination-sha256",
    "7" * 64,
    "--decontamination-receipt",
    "decontamination-receipt-v1",
    "--container-digest",
    "sha256:" + "d" * 64,
    "--runtime-digest",
    "sha256:" + "e" * 64,
    "--verifier-sha256",
    "f" * 64,
    "--base-model-revision",
    "1" * 40,
    "--tokenizer-revision",
    "2" * 40,
]


class PavlovXlamCostTests(unittest.TestCase):
    def test_conservative_100_example_ceiling(self) -> None:
        self.assertAlmostEqual(maximum_eval_cost(100, 1200, 128), 0.081888)

    def test_cost_scales_linearly(self) -> None:
        self.assertAlmostEqual(
            maximum_eval_cost(500, 1200, 128),
            5 * maximum_eval_cost(100, 1200, 128),
        )


class _FakeRun:
    def __init__(self, events: list[tuple[str, object]], *, fail_log: bool = False) -> None:
        self.events = events
        self.config: _FakeConfig = _FakeConfig(events)
        self.mode = "online"
        self.id = "fake-run"
        self.url = "https://wandb.invalid/fake-run"
        self.artifact_acknowledged = False
        self.fail_log = fail_log

    def log(self, metrics: dict[str, object]) -> None:
        self.events.append(("log", dict(metrics)))
        if self.fail_log:
            raise RuntimeError("fake W&B log failure")

    def log_artifact(self, artifact: object) -> None:
        self.events.append(("artifact", artifact))
        self.artifact_acknowledged = True

    def finish(self, *, exit_code: int) -> None:
        self.events.append(("finish", exit_code))


class _FakeConfig(dict[str, object]):
    def __init__(self, events: list[tuple[str, object]]) -> None:
        super().__init__()
        self.events = events

    def update(
        self,
        values: dict[str, object],
        *,
        allow_val_change: bool | None = None,
    ) -> None:
        self.events.append(("config_update", allow_val_change))
        super().update(values)


class _FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict[str, object]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str | None]] = []

    def add_file(self, path: str, *, name: str | None = None) -> None:
        self.files.append((path, name))


class _FakeWandb(types.ModuleType):
    def __init__(self, events: list[tuple[str, object]], *, fail_log: bool = False) -> None:
        super().__init__("wandb")
        self.events = events
        self.run: _FakeRun | None = None
        self.fail_log = fail_log

    def init(self, **kwargs: object) -> _FakeRun:
        self.events.append(("init", kwargs))
        self.run = _FakeRun(self.events, fail_log=self.fail_log)
        return self.run

    Artifact = _FakeArtifact


def _fake_runtime(events: list[tuple[str, object]]) -> dict[str, types.ModuleType]:
    class Example:
        prompt = "call the tool"
        target = {"name": "tool", "arguments": {"x": 1}}

    class Dataset:
        def test_examples(self) -> list[Example]:
            return [Example()]

    class Tokenizer:
        def __init__(self, model_name: str, revision: str | None) -> None:
            self.name_or_path = model_name
            self.revision = revision
            self.init_kwargs = {"revision": revision}

        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            del prompt, add_special_tokens
            return [1, 2, 3]

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            del tokens, skip_special_tokens
            return "raw model response must not be logged"

    class Future:
        def result(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                sequences=[types.SimpleNamespace(tokens=[4, 5])]
            )

    class Sampler:
        def __init__(self, *, name_or_path: str, revision: str | None) -> None:
            self.name_or_path = name_or_path
            self.revision = revision

        def sample(self, *args: object, **kwargs: object) -> Future:
            events.append(("sample", (args, kwargs)))
            return Future()

    class ServiceClient:
        def __init__(
            self,
            *,
            user_metadata: dict[str, str],
            base_model_revision: str | None = None,
        ) -> None:
            events.append(
                (
                    "service",
                    {
                        "user_metadata": user_metadata,
                        "base_model_revision": base_model_revision,
                    },
                )
            )

        def create_sampling_client(
            self,
            *,
            base_model: str | None = None,
            model_path: str | None = None,
            revision: str | None = None,
        ) -> Sampler:
            kwargs = {
                key: value
                for key, value in {
                    "base_model": base_model,
                    "model_path": model_path,
                    "revision": revision,
                }.items()
                if value is not None
            }
            events.append(("sampling_client", kwargs))
            return Sampler(
                name_or_path=base_model or model_path or "",
                revision=revision,
            )

    class ModelInput:
        @staticmethod
        def from_ints(tokens: list[int]) -> tuple[int, ...]:
            return tuple(tokens)

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    tinker = types.ModuleType("tinker")
    tinker.ServiceClient = ServiceClient
    tinker_types = types.ModuleType("tinker.types")
    tinker_types.ModelInput = ModelInput
    tinker_types.SamplingParams = SamplingParams

    transformers = types.ModuleType("transformers")
    def from_pretrained(*args: object, **kwargs: object) -> Tokenizer:
        model_name = str(args[0])
        revision = kwargs.get("revision")
        events.append(("tokenizer", {"model": model_name, "revision": revision}))
        return Tokenizer(model_name, str(revision) if revision is not None else None)

    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)

    grpo = types.ModuleType("platform_tinker.tinkerrl.grpo")
    grpo.StrictToolCallReward = type(
        "StrictToolCallReward",
        (),
        {"score": lambda self, response, example: 1.0},
    )
    def make_xlam_dataset(*, seed: int, revision: str) -> Dataset:
        events.append(("dataset", {"seed": seed, "revision": revision}))
        return Dataset()

    grpo.make_xlam_dataset = make_xlam_dataset
    grpo.Dataset = Dataset
    return {
        "tinker": tinker,
        "tinker.types": tinker_types,
        "transformers": transformers,
        "platform_tinker.tinkerrl.grpo": grpo,
    }


class PavlovXlamWandbTests(unittest.TestCase):
    def _build_argv(
        self,
        out: Path,
        *,
        sampler_path: bool = False,
        adapter_revision: str | None = None,
        provenance_args: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        source_args = (
            ["--sampler-path", "fake/adapter"]
            if sampler_path
            else ["--base-model", "fake/base"]
        )
        argv = [
            "eval_pavlov_xlam.py",
            *source_args,
            "--limit",
            "1",
            "--max-cost-usd",
            "1",
            "--out",
            str(out),
        ]
        argv.extend(
            _VALID_PROVENANCE_ARGS
            if provenance_args is None
            else provenance_args
        )
        if adapter_revision is not None:
            argv.extend(["--adapter-revision", adapter_revision])
        if extra_args:
            argv.extend(extra_args)
        return argv

    def _run_main(
        self,
        wandb: _FakeWandb,
        events: list[tuple[str, object]],
        *,
        sampler_path: bool = False,
        adapter_revision: str | None = None,
        provenance_args: list[str] | None = None,
        extra_args: list[str] | None = None,
        runtime_modules: dict[str, types.ModuleType] | None = None,
    ) -> Path:
        modules = _fake_runtime(events) if runtime_modules is None else runtime_modules
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "receipt.json"
            argv = self._build_argv(
                out,
                sampler_path=sampler_path,
                adapter_revision=adapter_revision,
                provenance_args=provenance_args,
                extra_args=extra_args,
            )
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                self.assertEqual(harness.main(), 0)
            # Keep the content available after TemporaryDirectory cleanup for
            # assertions by returning a copy into a second temporary location.
            kept = Path(tempfile.mkdtemp()) / "receipt.json"
            kept.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            return kept

    def test_online_wandb_precedes_tinker_and_logs_portfolio_receipt(self) -> None:
        events: list[tuple[str, object]] = []
        wandb = _FakeWandb(events)
        receipt_path = self._run_main(
            wandb,
            events,
            extra_args=["--suite-id", "xlam_component", "--domains", "tool_use"],
        )

        event_names = [name for name, _ in events]
        self.assertLess(event_names.index("init"), event_names.index("service"))
        init_kwargs = next(value for name, value in events if name == "init")
        self.assertEqual(init_kwargs["mode"], "online")
        self.assertEqual(init_kwargs["project"], harness.PAVLOV_WANDB_PROJECT)
        self.assertEqual(init_kwargs["group"], harness.PAVLOV_WANDB_GROUP)
        self.assertTrue(init_kwargs["config"]["config_immutable"])
        self.assertEqual(
            [value for name, value in events if name == "config_update"], [False]
        )
        self.assertEqual(init_kwargs["config"]["portfolio_suite_count"], 14)
        self.assertEqual(len(init_kwargs["config"]["portfolio_suite_ids"]), 14)
        self.assertEqual(
            init_kwargs["config"]["portfolio_id"],
            "pavlov-primary-eval-14-suite-v1",
        )
        self.assertEqual(init_kwargs["config"]["portfolio_role"], "primary_eval")
        self.assertEqual(init_kwargs["config"]["stage"], "primary-evaluation")
        self.assertEqual(init_kwargs["config"]["suite_id"], "xlam_component")
        self.assertEqual(init_kwargs["config"]["domains"], ["tool_use"])
        self.assertEqual(
            init_kwargs["config"]["provenance"]["dataset_revision"], "a" * 40
        )
        self.assertEqual(
            init_kwargs["config"]["provenance"]["split_manifest_sha256"], "b" * 64
        )
        self.assertEqual(
            init_kwargs["config"]["provenance"]["container_digest"],
            "sha256:" + "d" * 64,
        )
        self.assertEqual(
            init_kwargs["config"]["provenance"]["decontamination_sha256"],
            "7" * 64,
        )
        self.assertEqual(
            init_kwargs["config"]["phase0_provenance_sha256"],
            init_kwargs["config"]["provenance_sha256"],
        )

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["suite_id"], "xlam_component")
        self.assertEqual(receipt["domains"], ["tool_use"])
        self.assertEqual(
            receipt["provenance_schema_version"], "pavlov-phase0-provenance-v1"
        )
        self.assertEqual(receipt["provenance"]["adapter_revision"], None)
        self.assertEqual(
            receipt["phase0_provenance_sha256"], receipt["provenance_sha256"]
        )
        self.assertEqual(receipt["portfolio"]["suite_count"], 14)
        self.assertTrue(receipt["portfolio"]["component_only"])
        self.assertIn("does not cover all", receipt["portfolio"]["coverage_claim"])
        self.assertEqual(len(receipt["portfolio"]["suite_ids"]), 14)
        self.assertEqual(len(receipt["portfolio"]["suites"]), 14)
        self.assertEqual(
            receipt["portfolio"]["suites"][0],
            {
                "suite_id": "swe_bench_pro_eval",
                "domains": ["code", "long_horizon"],
                "role": "primary_eval",
            },
        )
        self.assertNotIn("raw model response must not be logged", json.dumps(receipt))

        log_metrics = [value for name, value in events if name == "log"]
        self.assertEqual(len(log_metrics), 1)
        self.assertIn("prompt_tokens", log_metrics[0])
        self.assertIn("sample_tokens", log_metrics[0])
        self.assertIn("estimated_cost_usd", log_metrics[0])
        artifacts = [value for name, value in events if name == "artifact"]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].type, "evaluation-receipt")
        self.assertEqual(artifacts[0].files[0][1], "receipt.json")
        self.assertTrue(wandb.run.artifact_acknowledged)
        self.assertEqual([value for name, value in events if name == "finish"], [0])

        dataset_event = next(value for name, value in events if name == "dataset")
        self.assertEqual(dataset_event["seed"], 809)
        self.assertEqual(dataset_event["revision"], "a" * 40)
        tokenizer_event = next(value for name, value in events if name == "tokenizer")
        self.assertEqual(tokenizer_event["revision"], "2" * 40)
        service_event = next(value for name, value in events if name == "service")
        self.assertEqual(service_event["base_model_revision"], "1" * 40)

    def test_valid_sampler_path_preserves_order_and_records_adapter_revision(self) -> None:
        events: list[tuple[str, object]] = []
        receipt_path = self._run_main(
            _FakeWandb(events),
            events,
            sampler_path=True,
            adapter_revision="3" * 40,
        )

        event_names = [name for name, _ in events]
        self.assertLess(event_names.index("init"), event_names.index("service"))
        self.assertLess(event_names.index("service"), event_names.index("sampling_client"))
        sampling_kwargs = next(
            value for name, value in events if name == "sampling_client"
        )
        self.assertEqual(
            sampling_kwargs,
            {"model_path": "fake/adapter", "revision": "3" * 40},
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["source_kind"], "sampler_path")
        self.assertEqual(receipt["provenance"]["adapter_revision"], "3" * 40)

    def test_phase0_missing_provenance_blocks_wandb_and_tinker(self) -> None:
        events: list[tuple[str, object]] = []
        wandb = _FakeWandb(events)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "receipt.json"
            argv = self._build_argv(out, provenance_args=[])
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    harness.main()
        event_names = [name for name, _ in events]
        self.assertNotIn("init", event_names)
        self.assertNotIn("service", event_names)
        self.assertNotIn("sample", event_names)

    def test_phase0_placeholder_provenance_blocks_wandb_and_tinker(self) -> None:
        events: list[tuple[str, object]] = []
        wandb = _FakeWandb(events)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        placeholder_args = list(_VALID_PROVENANCE_ARGS)
        placeholder_args[
            placeholder_args.index("--split-manifest-sha256") + 1
        ] = "PLACEHOLDER"
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "receipt.json"
            argv = self._build_argv(out, provenance_args=placeholder_args)
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    harness.main()
        event_names = [name for name, _ in events]
        self.assertNotIn("init", event_names)
        self.assertNotIn("service", event_names)
        self.assertNotIn("sample", event_names)

    def test_sampler_path_missing_adapter_revision_blocks_before_side_effects(self) -> None:
        events: list[tuple[str, object]] = []
        wandb = _FakeWandb(events)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "receipt.json"
            argv = self._build_argv(out, sampler_path=True)
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    harness.main()
        event_names = [name for name, _ in events]
        self.assertNotIn("init", event_names)
        self.assertNotIn("service", event_names)
        self.assertNotIn("sample", event_names)

    def test_wandb_initialization_failure_blocks_service_client(self) -> None:
        events: list[tuple[str, object]] = []

        class FailingWandb(_FakeWandb):
            def init(self, **kwargs: object) -> _FakeRun:
                self.events.append(("init", kwargs))
                raise RuntimeError("network unavailable")

        wandb = FailingWandb(events)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = self._build_argv(Path(temp_dir) / "receipt.json")
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    harness.main()
        self.assertNotIn("service", [name for name, _ in events])
        self.assertNotIn("sample", [name for name, _ in events])

    def test_non_online_wandb_run_blocks_service_and_finishes_failed(self) -> None:
        events: list[tuple[str, object]] = []

        class OfflineWandb(_FakeWandb):
            def init(self, **kwargs: object) -> _FakeRun:
                run = super().init(**kwargs)
                run.mode = "offline"
                return run

        wandb = OfflineWandb(events)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = self._build_argv(Path(temp_dir) / "receipt.json")
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    harness.main()
        self.assertEqual([value for name, value in events if name == "finish"], [1])
        self.assertNotIn("service", [name for name, _ in events])
        self.assertNotIn("sample", [name for name, _ in events])

    def test_metric_logging_failure_finishes_failed(self) -> None:
        events: list[tuple[str, object]] = []
        wandb = _FakeWandb(events, fail_log=True)
        modules = _fake_runtime(events)
        modules["wandb"] = wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = self._build_argv(Path(temp_dir) / "receipt.json")
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    harness.main()
        self.assertEqual([value for name, value in events if name == "finish"], [1])
        self.assertNotIn("artifact", [name for name, _ in events])

    def test_revision_bound_fallback_loader_receives_dataset_revision(self) -> None:
        events: list[tuple[str, object]] = []
        modules = _fake_runtime(events)
        grpo = modules["platform_tinker.tinkerrl.grpo"]

        def unbound_loader(*, seed: int) -> object:
            events.append(("unbound_dataset_loader", seed))
            return object()

        grpo.make_xlam_dataset = unbound_loader
        row = {
            "tools": '[{"name": "tool"}]',
            "answers": '[{"name": "tool", "arguments": {"x": 1}}]',
            "query": "call the tool",
        }
        datasets = types.ModuleType("datasets")

        def load_dataset(
            dataset_id: str, *, split: str, revision: str
        ) -> list[dict[str, str]]:
            events.append(
                (
                    "load_dataset",
                    {"dataset_id": dataset_id, "split": split, "revision": revision},
                )
            )
            return [row] * 3501

        datasets.load_dataset = load_dataset
        modules["datasets"] = datasets
        wandb = _FakeWandb(events)
        receipt_path = self._run_main(
            wandb,
            events,
            runtime_modules=modules,
        )
        load_event = next(value for name, value in events if name == "load_dataset")
        self.assertEqual(load_event["dataset_id"], harness.XLAM_DATASET_ID)
        self.assertEqual(load_event["split"], "train")
        self.assertEqual(load_event["revision"], "a" * 40)
        self.assertNotIn("unbound_dataset_loader", [name for name, _ in events])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            receipt["dataset_loader_binding"]["mode"], "local_revision_bound_loader"
        )

    def test_non_positive_or_non_finite_limits_block_before_wandb_and_tinker(self) -> None:
        cases = (
            ("--limit", "0"),
            ("--max-prompt-tokens", "-1"),
            ("--max-response-tokens", "0"),
            ("--max-cost-usd", "0"),
            ("--max-cost-usd", "nan"),
            ("--max-cost-usd", "-1"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                events: list[tuple[str, object]] = []
                wandb = _FakeWandb(events)
                modules = _fake_runtime(events)
                with tempfile.TemporaryDirectory() as temp_dir:
                    argv = self._build_argv(Path(temp_dir) / "receipt.json")
                    if option in argv:
                        argv[argv.index(option) + 1] = value
                    else:
                        argv.extend([option, value])
                    modules["wandb"] = wandb
                    with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                        with self.assertRaises(SystemExit):
                            harness.main()
                event_names = [name for name, _ in events]
                self.assertNotIn("init", event_names)
                self.assertNotIn("service", event_names)
                self.assertNotIn("sampling_client", event_names)
                self.assertNotIn("sample", event_names)

    def test_empty_examples_fail_after_wandb_but_before_tinker(self) -> None:
        events: list[tuple[str, object]] = []
        modules = _fake_runtime(events)

        class EmptyDataset:
            def test_examples(self) -> list[object]:
                return []

        def empty_loader(*, seed: int, revision: str) -> EmptyDataset:
            events.append(("dataset", {"seed": seed, "revision": revision}))
            return EmptyDataset()

        modules["platform_tinker.tinkerrl.grpo"].make_xlam_dataset = empty_loader
        wandb = _FakeWandb(events)
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = self._build_argv(Path(temp_dir) / "receipt.json")
            modules["wandb"] = wandb
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    harness.main()
        self.assertEqual([value for name, value in events if name == "finish"], [1])
        self.assertNotIn("service", [name for name, _ in events])
        self.assertNotIn("sample", [name for name, _ in events])

    def test_primary_suite_and_unknown_domain_overrides_are_rejected(self) -> None:
        cases = (
            ("--suite-id", harness.PAVLOV_PRIMARY_EVAL_SUITE_IDS[0]),
            ("--domains", "browser"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                events: list[tuple[str, object]] = []
                wandb = _FakeWandb(events)
                modules = _fake_runtime(events)
                with tempfile.TemporaryDirectory() as temp_dir:
                    argv = self._build_argv(Path(temp_dir) / "receipt.json")
                    argv.extend([option, value])
                    modules["wandb"] = wandb
                    with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                        with self.assertRaises(SystemExit):
                            harness.main()
                self.assertNotIn("init", [name for name, _ in events])

    def test_wandb_identity_config_and_artifact_ack_are_required(self) -> None:
        class MissingIdentityRun(_FakeRun):
            def __init__(self, events: list[tuple[str, object]]) -> None:
                super().__init__(events)
                self.id = ""

        class MissingIdentityWandb(_FakeWandb):
            def init(self, **kwargs: object) -> _FakeRun:
                self.events.append(("init", kwargs))
                self.run = MissingIdentityRun(self.events)
                return self.run

        class MissingConfigRun(_FakeRun):
            def __init__(self, events: list[tuple[str, object]]) -> None:
                super().__init__(events)
                self.config = None

        class MissingConfigWandb(_FakeWandb):
            def init(self, **kwargs: object) -> _FakeRun:
                self.events.append(("init", kwargs))
                self.run = MissingConfigRun(self.events)
                return self.run

        class MissingAckRun(_FakeRun):
            def log_artifact(self, artifact: object) -> None:
                super().log_artifact(artifact)
                self.artifact_acknowledged = False

        class MissingAckWandb(_FakeWandb):
            def init(self, **kwargs: object) -> _FakeRun:
                self.events.append(("init", kwargs))
                self.run = MissingAckRun(self.events)
                return self.run

        for wandb_class in (
            MissingIdentityWandb,
            MissingConfigWandb,
            MissingAckWandb,
        ):
            with self.subTest(wandb=wandb_class.__name__):
                events: list[tuple[str, object]] = []
                wandb = wandb_class(events)
                modules = _fake_runtime(events)
                with tempfile.TemporaryDirectory() as temp_dir:
                    argv = self._build_argv(Path(temp_dir) / "receipt.json")
                    modules["wandb"] = wandb
                    with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                        with self.assertRaises(RuntimeError):
                            harness.main()
                self.assertEqual(
                    [value for name, value in events if name == "finish"], [1]
                )
                if wandb_class is not MissingAckWandb:
                    self.assertNotIn("service", [name for name, _ in events])
                    self.assertNotIn("sample", [name for name, _ in events])


if __name__ == "__main__":
    unittest.main()
