"""Static guard: a pinned-seed ``sample(num_samples=N>1)`` call collapses GRPO groups.

On the Tinker service, ``sample(prompt, num_samples=N, sampling_params=SamplingParams(seed=S))``
returns N identical sequences. Every group then has zero reward variance, so GRPO gets no
gradient signal, silently. The fix is N requests of ``num_samples=1`` with a per-member seed.

This test walks every ``.py`` file under ``platform_hybrid/experiments/tinker-runs/`` and
flags a call whose function attribute is ``sample`` when BOTH hold:

  1. ``num_samples`` is passed and is not the literal ``1``;
  2. the ``sampling_params`` argument (keyword or positional) is either
     - a ``SamplingParams(...)`` call with a ``seed=`` keyword, or
     - a ``Name`` bound in the same function to such a ``SamplingParams(...)`` call, or to
       ``<expr>.model_copy(update={... 'seed': ...})``.

Known limitation: no dataflow beyond simple same-function assignment is tracked. Rebinding
through tuples/containers, arguments, ``global`` names, closures, or module-level constants is
not followed, so a pinned seed that reaches ``sampling_params`` indirectly can be missed.

The repo scans as currently buggy, so the scan test is an expected failure until the runners
are fixed in their own change. ``TestCheckerOnSnippets`` pins the checker's behaviour either way.
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNNERS_DIR = (
    Path(__file__).resolve().parents[1] / "platform_hybrid" / "experiments" / "tinker-runs"
)
SAMPLE_ATTRS = {"sample", "sample_async"}


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _callee_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Attribute):
        return expr.attr
    if isinstance(expr, ast.Name):
        return expr.id
    return None


def _is_sampling_params_seeded(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and _callee_name(expr.func) == "SamplingParams"
        and _kwarg(expr, "seed") is not None
    )


def _is_seed_model_copy(expr: ast.expr) -> bool:
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)):
        return False
    if expr.func.attr != "model_copy":
        return False
    update = _kwarg(expr, "update")
    return isinstance(update, ast.Dict) and any(
        isinstance(k, ast.Constant) and k.value == "seed" for k in update.keys
    )


def _same_function_seeded_params(func: ast.AST) -> set[str]:
    """Names assigned in `func` (anywhere) to a seeded SamplingParams or seeded model_copy."""
    seeded: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not (_is_sampling_params_seeded(value) or _is_seed_model_copy(value)):
            continue
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    seeded.add(sub.id)
    return seeded


def _is_bad_sample_call(node: ast.Call, seeded_names: set[str]) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr in SAMPLE_ATTRS):
        return False

    sampling_params = _kwarg(node, "sampling_params")
    if sampling_params is None and len(node.args) >= 2:
        sampling_params = node.args[1]

    num_samples = _kwarg(node, "num_samples")
    if num_samples is None and len(node.args) >= 3:
        num_samples = node.args[2]
    if num_samples is None:
        return False
    if isinstance(num_samples, ast.Constant) and num_samples.value == 1:
        return False

    if sampling_params is None:
        return False
    if _is_sampling_params_seeded(sampling_params) or _is_seed_model_copy(sampling_params):
        return True
    return isinstance(sampling_params, ast.Name) and sampling_params.id in seeded_names


def find_seeded_group_sampling_offenders(source: str) -> list[int]:
    """1-based line numbers of offending sample() calls. Raises SyntaxError on bad source."""
    tree = ast.parse(source)
    offenders: set[int] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seeded_names = _same_function_seeded_params(func)
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and _is_bad_sample_call(node, seeded_names):
                offenders.add(node.lineno)
    return sorted(offenders)


BAD_SNIPPET = """
def rollout(sc, prompt, group_size, seed):
    sp = T.SamplingParams(max_tokens=512, temperature=0.8, seed=seed)
    return sc.sample(prompt, num_samples=group_size, sampling_params=sp).result()
"""

FIXED_SNIPPET = """
def rollout(sc, prompt, group_size, seed):
    return [
        sc.sample(
            prompt,
            num_samples=1,
            sampling_params=T.SamplingParams(temperature=0.8, seed=seed + i),
        ).result()
        for i in range(group_size)
    ]
"""


class TestCheckerOnSnippets:
    def test_flags_seeded_multi_sample_call(self):
        assert find_seeded_group_sampling_offenders(BAD_SNIPPET) == [4]

    def test_accepts_per_member_single_sample_pattern(self):
        assert find_seeded_group_sampling_offenders(FIXED_SNIPPET) == []

    def test_inline_sampling_params_with_seed_is_flagged(self):
        source = "def f(sc, p):\n    sc.sample(p, num_samples=8, sampling_params=T.SamplingParams(seed=1))\n"
        assert find_seeded_group_sampling_offenders(source) == [2]

    def test_positional_sampling_params_is_checked(self):
        source = "def f(sc, p):\n    sc.sample(p, T.SamplingParams(seed=1), 8)\n"
        assert find_seeded_group_sampling_offenders(source) == [2]

    def test_unseeded_sampling_params_is_ignored(self):
        source = (
            "def f(sc, p):\n"
            "    sc.sample(p, num_samples=8, sampling_params=T.SamplingParams(temperature=1.0))\n"
        )
        assert find_seeded_group_sampling_offenders(source) == []

    def test_seeded_name_from_model_copy_is_flagged(self):
        source = (
            "def f(base):\n"
            "    params = base.model_copy(update={'temperature': 1.0, 'seed': 7})\n"
            "    return sc.sample(p, num_samples=4, sampling_params=params)\n"
        )
        assert find_seeded_group_sampling_offenders(source) == [3]

    def test_inline_seed_model_copy_async_is_flagged(self):
        source = (
            "async def f(sc, p, base, n):\n"
            "    await sc.sample_async(prompt=p, num_samples=n,\n"
            "        sampling_params=base.model_copy(update={'seed': 3}))\n"
        )
        assert find_seeded_group_sampling_offenders(source) == [2]

    def test_plain_list_sample_is_ignored(self):
        source = "import random\n\ndef f(items):\n    return random.Random(0).sample(items, 3)\n"
        assert find_seeded_group_sampling_offenders(source) == []


def _collect_offenders() -> list[str]:
    offenders: list[str] = []
    for path in sorted(RUNNERS_DIR.rglob("*.py")):
        lines = find_seeded_group_sampling_offenders(path.read_text(encoding="utf-8"))
        rel = path.relative_to(RUNNERS_DIR.parents[2])
        offenders.extend(f"{rel}:{line}" for line in lines)
    return offenders


def test_no_seeded_group_sampling_under_tinker_runs():
    assert RUNNERS_DIR.is_dir(), f"runner directory missing: {RUNNERS_DIR}"
    offenders = _collect_offenders()
    assert not offenders, (
        "sample() with num_samples > 1 and a pinned seed returns identical sequences "
        "(zero GRPO group variance). Issue one num_samples=1 request per member with "
        "per-member seeds. Offenders:\n  " + "\n  ".join(offenders)
    )
