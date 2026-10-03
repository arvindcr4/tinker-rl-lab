"""Shared test fakes, deduplicated from the suites that used verbatim copies.

Consolidates helpers previously copy-pasted across test modules:

- ``load_module`` (from ``test_next_submission_trl_adapter``; the other
  ``test_next_submission_*`` suites inlined the same file-loading boilerplate)
- ``embedded_payload`` (from the gcp/hf_jobs/kaggle preflight suites)
- ``_Future`` / ``_Tokenizer`` / ``_config`` / ``_example`` / ``HfApi``
  (from ``test_grpo_coverage`` and ``test_grpo_caliber_fixes``)
- ``make_fake_run`` (wraps the ``fake_run`` closures of ``test_grpo_caliber_fixes``)
- ``fast_settings`` (the ``FAST`` hypothesis profile of the properties suites)

Third-party imports (``hypothesis``, ``tinkerrl.grpo``) stay function-local so
importing this module never adds import-time dependencies to a consumer.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

__all__ = [
    "HfApi",
    "_Future",
    "_Tokenizer",
    "_config",
    "_example",
    "embedded_payload",
    "fast_settings",
    "load_module",
    "make_fake_run",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def embedded_payload(script: str, pattern=r"FILES = json\.loads\((.+)\)\n") -> dict[str, str]:
    match = re.search(pattern, script)
    assert match is not None
    encoded_json = json.loads(match.group(1))
    return json.loads(encoded_json)


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Tokenizer:
    """Fake tokenizer. Defaults match test_grpo_coverage; caliber passes
    ``tokens=PROMPT`` with ``decode_first_only=True``."""

    def __init__(self, tokens=(1, 2, 3, 4, 5), decode_first_only=False):
        self._tokens = list(tokens)
        self._decode_first_only = decode_first_only

    def encode(self, _prompt, add_special_tokens=False):
        return list(self._tokens)

    def decode(self, tokens, skip_special_tokens=True):
        seq = list(tokens)
        if self._decode_first_only:
            seq = seq[:1]
        return "1" if seq == [1] else "0"


def _example(target="1", prompt="q"):
    from platform_tinker.tinkerrl.grpo import TrainingExample

    return TrainingExample(prompt=prompt, target=target)


def _config(tmp_path, name="cov", **kwargs):
    from platform_tinker.tinkerrl.grpo import GRPOConfig

    base = dict(name=name, steps=1, group_size=2, batch_size=1, checkpoint_dir=str(tmp_path))
    base.update(kwargs)
    return GRPOConfig(**base)


class HfApi:
    info_count = 0

    def __init__(self, **_kwargs):
        pass

    def whoami(self, **_kwargs):
        return {"name": "owner"}

    def model_info(self, _repo_id, revision):
        type(self).info_count += 1
        return SimpleNamespace(sha=f"{type(self).info_count:040x}")

    def create_repo(self, **_kwargs):
        return None

    def create_branch(self, **_kwargs):
        return None


def make_fake_run(seen):
    def fake_run(_config, _dataset, reward):
        seen["reward"] = reward
        return []

    return fake_run


def fast_settings():
    from hypothesis import HealthCheck, settings

    return settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
