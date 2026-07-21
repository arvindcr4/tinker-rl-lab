from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilot.remote_unit import main, parse_args


class PromptTokenTests(unittest.TestCase):
    def test_prompt_tokens_unwraps_transformers_v5_batch_encoding(self) -> None:
        import torch

        from pilot.remote_unit import _prompt_tokens

        class FakeEncoding:
            def __init__(self, input_ids: object) -> None:
                self.input_ids = input_ids

            def __getitem__(self, key: str) -> object:
                return getattr(self, key)

        class FakeTokenizer:
            def __init__(self, encoded: object) -> None:
                self.encoded = encoded

            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                return self.encoded

        tensor = torch.tensor([[5, 6, 7, 8]])
        result = _prompt_tokens(
            FakeTokenizer(FakeEncoding(tensor)),
            regime="balanced_equal_length",
            question="1+1?",
            max_length=64,
        )
        self.assertTrue(torch.equal(result, tensor))
        list_result = _prompt_tokens(
            FakeTokenizer([[5, 6, 7, 8]]),
            regime="balanced_equal_length",
            question="1+1?",
            max_length=2,
        )
        self.assertTrue(torch.equal(list_result, torch.tensor([[7, 8]])))


class GenerateCandidateTests(unittest.TestCase):
    def test_candidates_use_scoped_seeded_rng_without_generator_kwarg(self) -> None:
        import torch
        from torch.nn.attention import SDPBackend

        from pilot.remote_unit import _generate_candidates

        recorded: dict[str, object] = {}
        sdpa_calls: list[list[object]] = []

        @contextlib.contextmanager
        def recorder(backends: object):
            sdpa_calls.append(list(backends))
            yield

        class FakeModel:
            def __init__(self) -> None:
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def parameters(self) -> object:
                return iter([self.anchor])

            def generate(self, **kwargs: object) -> object:
                recorded.clear()
                recorded.update(kwargs)
                prompt_width = kwargs["input_ids"].shape[1]
                draws = torch.randint(
                    0,
                    50,
                    (kwargs["num_return_sequences"], prompt_width + kwargs["max_new_tokens"]),
                )
                draws[:, :prompt_width] = kwargs["input_ids"]
                return draws

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 49

            def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
                return "#### 1"

        def call() -> list[tuple[int, ...]]:
            candidates = _generate_candidates(
                model=FakeModel(),
                tokenizer=FakeTokenizer(),
                prompt_ids=torch.tensor([[1, 2]]),
                regime="balanced_equal_length",
                answer="1",
                group_index=0,
                seed=11,
                count=2,
                max_new_tokens=3,
            )
            return [candidate.token_ids for candidate in candidates]

        with mock.patch("torch.nn.attention.sdpa_kernel", recorder):
            state_before = torch.random.get_rng_state()
            first = call()
            state_after = torch.random.get_rng_state()
            self.assertNotIn("generator", recorded)
            self.assertTrue(torch.equal(state_before, state_after))
            self.assertEqual(first, call())
        self.assertEqual(sdpa_calls, [[SDPBackend.MATH], [SDPBackend.MATH]])


class RemoteUnitSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        source = Path(__file__).resolve().parents[1] / "pilot_preregistration.json"
        payload = json.loads(source.read_text())
        payload["status"] = "locked_not_authorized"
        payload["authorization"]["gpu"] = False
        self.locked_protocol = Path(self.temporary.name) / "pilot_preregistration.json"
        self.locked_protocol.write_text(json.dumps(payload))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_describe_reports_authorized_protocol_without_side_effects(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["describe"]), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "ready_to_run")
        self.assertTrue(record["gpu_authorized"])
        self.assertEqual(record["execution_blockers"], [])

    def test_build_corpus_refuses_before_credentials_or_heavy_remote_work(self) -> None:
        with self.assertRaisesRegex(SystemExit, "allocation is forbidden"):
            main(
                [
                    "--protocol",
                    str(self.locked_protocol),
                    "build-corpus",
                    "--regime",
                    "balanced_equal_length",
                    "--seed",
                    "11",
                    "--hf-repo",
                    "arvindcr4/tinker-rl-lab-flagship-pilot-corpus-test",
                ]
            )

    def test_build_corpus_cli_requires_isolated_identity_fields(self) -> None:
        args = parse_args(
            [
                "build-corpus",
                "--regime",
                "filtered_variable_length",
                "--seed",
                "23",
                "--hf-repo",
                "arvindcr4/tinker-rl-lab-flagship-pilot-corpus-filtered-s23",
            ]
        )
        self.assertEqual(args.command, "build-corpus")
        self.assertEqual(args.seed, 23)
        self.assertEqual(args.regime, "filtered_variable_length")

    def test_train_unit_refuses_before_remote_writes_while_locked(self) -> None:
        with self.assertRaisesRegex(SystemExit, "allocation is forbidden"):
            main(
                [
                    "--protocol",
                    str(self.locked_protocol),
                    "train-unit",
                    "--condition",
                    "intended_full",
                    "--regime",
                    "balanced_equal_length",
                    "--seed",
                    "11",
                ]
            )

    def test_smoke_refuses_while_locked(self) -> None:
        with self.assertRaisesRegex(SystemExit, "allocation is forbidden"):
            main(["--protocol", str(self.locked_protocol), "smoke"])


if __name__ == "__main__":
    unittest.main()
