from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from pilot.remote_core import (
    RemoteContractError,
    canonical_order_hash,
    expected_runtime_versions,
    frozen_train_order,
    gsm8k_reward,
    last_boxed,
    math500_reward,
    normalize_math,
    prompt_messages,
    require_a100,
)
from pilot.protocol import load_protocol


class FakeCuda:
    def __init__(self, name: str, available: bool = True, bf16: bool = True) -> None:
        self.name = name
        self.available = available
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, _: int) -> str:
        return self.name

    def is_bf16_supported(self) -> bool:
        return self.bf16


class RemoteCoreTests(unittest.TestCase):
    def test_reward_parsers_are_strict_and_nested_box_aware(self) -> None:
        self.assertEqual(gsm8k_reward("work\n#### 1,024", "#### 1024"), 1.0)
        self.assertEqual(gsm8k_reward("answer 1024", "#### 1024"), 0.0)
        self.assertEqual(last_boxed(r"x \\boxed{\\frac{1}{2}}"), r"\\frac{1}{2}")
        self.assertEqual(math500_reward(r"x \\boxed{\\dfrac{1}{2}}", r"\\frac{1}{2}"), 1.0)
        self.assertEqual(math500_reward(r"x \\boxed{0.5}", r"\\frac{1}{2}"), 0.0)
        self.assertEqual(normalize_math(r" { \\dfrac{1}{2} } . "), r"\\frac{1}{2}")

    def test_prompt_contracts_are_regime_specific(self) -> None:
        gsm = prompt_messages("balanced_equal_length", "question")
        math = prompt_messages("filtered_variable_length", "problem")
        self.assertIn("####", gsm[0]["content"])
        self.assertIn("\\boxed", math[0]["content"])
        with self.assertRaisesRegex(RemoteContractError, "unknown pilot regime"):
            prompt_messages("unknown", "x")

    def test_seeded_order_and_hash_are_deterministic(self) -> None:
        rows = [
            {"question": f"q{index}", "answer": f"a{index}"}
            for index in range(20)
        ]
        first, first_hash = frozen_train_order(
            rows,
            eligible_indices=range(20),
            seed=11,
            keys=("question", "answer"),
            count=10,
        )
        second, second_hash = frozen_train_order(
            rows,
            eligible_indices=range(20),
            seed=11,
            keys=("question", "answer"),
            count=10,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_hash, canonical_order_hash(rows, first, keys=("question", "answer")))
        different, _ = frozen_train_order(
            rows,
            eligible_indices=range(20),
            seed=23,
            keys=("question", "answer"),
            count=10,
        )
        self.assertNotEqual(first, different)

    def test_runtime_pins_are_exact_and_complete(self) -> None:
        versions = expected_runtime_versions(load_protocol())
        self.assertEqual(versions["trl"], "1.2.0")
        self.assertEqual(versions["datasets"], "4.8.4")
        self.assertEqual(versions["huggingface-hub"], "1.11.0")
        self.assertEqual(versions["numpy"], np.__version__)

    def test_a100_check_rejects_substitution(self) -> None:
        torch_module = mock.Mock(cuda=FakeCuda("NVIDIA A100-SXM4-40GB"))
        self.assertIn("A100", require_a100(torch_module))
        with self.assertRaisesRegex(RemoteContractError, "requires A100"):
            require_a100(mock.Mock(cuda=FakeCuda("NVIDIA L4")))
        with self.assertRaisesRegex(RemoteContractError, "no CUDA"):
            require_a100(mock.Mock(cuda=FakeCuda("none", available=False)))


if __name__ == "__main__":
    unittest.main()
