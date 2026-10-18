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
import pytest


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
        assert gsm8k_reward("work\n#### 1,024", "#### 1024") == 1.0
        assert gsm8k_reward("answer 1024", "#### 1024") == 0.0
        assert last_boxed(r"x \\boxed{\\frac{1}{2}}") == r"\\frac{1}{2}"
        assert math500_reward(r"x \\boxed{\\dfrac{1}{2}}", r"\\frac{1}{2}") == 1.0
        assert math500_reward(r"x \\boxed{0.5}", r"\\frac{1}{2}") == 0.0
        assert normalize_math(r" { \\dfrac{1}{2} } . ") == r"\\frac{1}{2}"

    def test_prompt_contracts_are_regime_specific(self) -> None:
        gsm = prompt_messages("balanced_equal_length", "question")
        math = prompt_messages("filtered_variable_length", "problem")
        assert "####" in gsm[0]["content"]
        assert "\\boxed" in math[0]["content"]
        with pytest.raises(RemoteContractError, match="unknown pilot regime"):
            prompt_messages("unknown", "x")

    def test_seeded_order_and_hash_are_deterministic(self) -> None:
        rows = [{"question": f"q{index}", "answer": f"a{index}"} for index in range(20)]
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
        assert first == second
        assert first_hash == second_hash
        assert first_hash == canonical_order_hash(rows, first, keys=("question", "answer"))
        different, _ = frozen_train_order(
            rows,
            eligible_indices=range(20),
            seed=23,
            keys=("question", "answer"),
            count=10,
        )
        assert first != different

    def test_runtime_pins_are_exact_and_complete(self) -> None:
        versions = expected_runtime_versions(load_protocol())
        assert versions["trl"] == "1.2.0"
        assert versions["datasets"] == "4.8.4"
        assert versions["huggingface-hub"] == "1.11.0"
        assert versions["numpy"] == np.__version__

    def test_a100_check_rejects_substitution(self) -> None:
        torch_module = mock.Mock(cuda=FakeCuda("NVIDIA A100-SXM4-40GB"))
        assert "A100" in require_a100(torch_module)
        with pytest.raises(RemoteContractError, match="requires A100"):
            require_a100(mock.Mock(cuda=FakeCuda("NVIDIA L4")))
        with pytest.raises(RemoteContractError, match="no CUDA"):
            require_a100(mock.Mock(cuda=FakeCuda("none", available=False)))


if __name__ == "__main__":
    unittest.main()
