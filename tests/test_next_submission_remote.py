from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NEXT_SUBMISSION = ROOT / "zvf-program/next-submission"
sys.path.insert(0, str(NEXT_SUBMISSION))
PATH = NEXT_SUBMISSION / "remote_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_remote_preflight", PATH)
assert SPEC is not None and SPEC.loader is not None
REMOTE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REMOTE
SPEC.loader.exec_module(REMOTE)


def test_last_boxed_accepts_nested_braced_latex():
    assert REMOTE.last_boxed(r"work \boxed{\frac{3}{2}}.") == r"\frac{3}{2}"


def test_last_boxed_accepts_only_unbraced_numeric_atoms():
    assert REMOTE.last_boxed(r"largest root is $\boxed 2$.") == "2"
    assert REMOTE.last_boxed(r"negative root is $\boxed -3/2$.") == "-3/2"
    assert REMOTE.last_boxed(r"ambiguous prose \boxed answer") is None


def test_frozen_math_training_exceptions_are_parseable():
    assert REMOTE.last_boxed(r"the largest value is $\boxed 2$.") == "2"
    assert REMOTE.last_boxed(r"therefore the sum is $\boxed 9$.") == "9"


def test_qwen_decoder_contract_disables_thinking_and_pins_sampling():
    assert REMOTE.ENABLE_THINKING is False
    assert REMOTE.TRAINING_DECODER == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }


def test_latest_metric_requires_observed_training_telemetry():
    history = [
        {"loss": 1.0},
        {"completions/clipped_ratio": 0.25},
        {"loss": 0.5},
    ]
    assert REMOTE.latest_metric(history, "completions/clipped_ratio") == 0.25
