#!/usr/bin/env python3
"""Sampling + scoring driver for a real VerilogEval pass@1 run.

Everything here is free and offline EXCEPT :func:`sample_all`, which is the one
function that spends money. It refuses to run unless an explicit authorization
token is present, so importing, testing, or dry-running this module can never
incur cost.

Two things in here are load-bearing enough to unit test before spending anything:

1. :func:`extract_module` -- ``Qwen/Qwen3.6-35B-A3B`` is a thinking model. Its
   chat template appends a bare ``<think>\\n`` on every generation prompt unless
   ``enable_thinking=false``. A extractor that does not strip the reasoning
   trace, or that grabs the first fenced block inside it, produces 312 unusable
   samples and burns the whole budget for a pass@1 of 0.
2. :func:`score_pass_at_1` -- the run must report BOTH denominators, because
   ``spec-to-rtl/Prob099_m2014_q6c`` is unscoreable by upstream defect and
   quietly dropping it inflates the number.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: Set this env var to the exact string below to permit paid sampling. It exists
#: so that no import, test, or dry run can spend money by accident.
AUTHORIZATION_ENV = "E11_PAID_RUN_AUTHORIZED"
AUTHORIZATION_TOKEN = "yes-user-approved"

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

#: Pinned from zvf-program/flagship/pavlov_tinker_budget.json.
USD_PER_M_PREFILL = 0.54
USD_PER_M_SAMPLE = 1.335
PROJECTION_GATE_USD = 4.00

#: Unscoreable at the pinned revision: the shipped _ref.sv declares Y1/Y3 while
#: the byte-identical test bench instantiates .Y2/.Y4, so the reference itself
#: fails to elaborate and no candidate can pass.
KNOWN_UNSCOREABLE = ("verilog_eval/spec-to-rtl/Prob099_m2014_q6c",)

_FENCE_RE = re.compile(r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MODULE_RE = re.compile(r"\bmodule\s+TopModule\b.*?\bendmodule\b", re.DOTALL)


class PaidRunNotAuthorized(RuntimeError):
    """Raised when paid sampling is attempted without explicit authorization."""


def strip_reasoning(text: str) -> str:
    """Drop a thinking trace, whether or not the opening tag was echoed.

    The chat template opens the assistant turn with ``<think>\\n``, so a response
    typically *ends* its reasoning with ``</think>`` without ever emitting the
    opening tag. Splitting on the closing tag is therefore the reliable move;
    a matched pair is handled too.
    """

    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def extract_module(response: str) -> str | None:
    """Return the SystemVerilog ``TopModule`` a model response defines.

    Order matters: strip reasoning first (a fenced block inside a thinking trace
    is a draft, not the answer), then prefer a fenced block, then fall back to a
    bare ``module TopModule ... endmodule`` span. Returns ``None`` when the
    response contains no usable module, which is a legitimate pass@1 failure and
    must be recorded rather than retried.
    """

    body = strip_reasoning(response)

    for candidate in reversed(_FENCE_RE.findall(body)):
        match = _MODULE_RE.search(candidate)
        if match:
            return match.group(0).strip()

    match = _MODULE_RE.search(body)
    if match:
        return match.group(0).strip()

    # A fenced block that holds a module under some other name is still not a
    # usable sample: the test benches instantiate TopModule by name.
    return None


def write_sample(
    build_dir: Path,
    problem_id: str,
    module_source: str | None,
    *,
    prompt_tokens: int,
    resp_tokens: int,
    cost_usd: float,
    sample_index: int = 1,
) -> Path:
    """Lay one sample out exactly as the official make targets expect.

    Also writes the per-sample ``-sv-generate.log``; ``sv-iv-analyze`` opens it
    unconditionally and raises FileNotFoundError without it.
    """

    problem_dir = build_dir / problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{problem_id}_sample{sample_index:02d}"

    # An empty file is the honest representation of "the model produced no
    # usable module": it fails to elaborate and scores as a miss.
    (problem_dir / f"{stem}.sv").write_text(
        (module_source + "\n") if module_source else "", encoding="utf-8"
    )
    (problem_dir / f"{stem}-sv-generate.log").write_text(
        f"model = {MODEL_ID}\n"
        f"revision = {MODEL_REVISION}\n"
        f"prompt_tokens = {prompt_tokens}\n"
        f"resp_tokens = {resp_tokens}\n"
        f"cost = {cost_usd:.6f}\n",
        encoding="utf-8",
    )
    return problem_dir / f"{stem}.sv"


def project_cost(prompt_chars: int, prompts: int, max_tokens: int) -> dict[str, Any]:
    """Conservative pre-flight projection: full uncached prefill, max output."""

    prefill_tokens = prompt_chars / 3.0
    output_tokens = prompts * max_tokens
    usd = prefill_tokens / 1e6 * USD_PER_M_PREFILL + output_tokens / 1e6 * USD_PER_M_SAMPLE
    return {
        "prompts": prompts,
        "max_tokens": max_tokens,
        "projected_prefill_tokens": round(prefill_tokens),
        "projected_output_tokens": output_tokens,
        "projected_usd": round(usd, 2),
        "gate_usd": PROJECTION_GATE_USD,
        "within_gate": usd < PROJECTION_GATE_USD,
    }


def parse_summary_csv(text: str) -> dict[str, bool]:
    """Map ``<Prob>,<npass>,<ntotal>,<rate>,<verdict>`` rows to pass booleans."""

    passed: dict[str, bool] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 5:
            continue
        try:
            passed[fields[0]] = int(fields[1]) > 0
        except ValueError:
            continue
    return passed


def score_pass_at_1(
    results: dict[str, bool],
    *,
    unscoreable: Sequence[str] = KNOWN_UNSCOREABLE,
) -> dict[str, Any]:
    """Report pass@1 on BOTH denominators, never only the flattering one.

    ``results`` maps canonical task IDs to pass booleans.
    """

    total = len(results)
    passes = sum(1 for ok in results.values() if ok)
    excluded = [task for task in unscoreable if task in results]
    corrected_total = total - len(excluded)
    corrected_passes = passes - sum(1 for task in excluded if results[task])

    return {
        "raw": {
            "denominator": total,
            "passes": passes,
            "pass_at_1": round(passes / total, 4) if total else None,
            "note": "all pinned tasks, including the upstream-defective one",
        },
        "corrected": {
            "denominator": corrected_total,
            "passes": corrected_passes,
            "pass_at_1": round(corrected_passes / corrected_total, 4) if corrected_total else None,
            "excluded": list(excluded),
            "note": "excludes tasks whose reference fails its own test bench",
        },
        "unscoreable_tasks": list(excluded),
        "reporting_rule": (
            "Both denominators must be reported together. "
            f"{', '.join(excluded) or 'no tasks'} excluded from the corrected figure because the "
            "shipped reference cannot elaborate against its own test bench, so no candidate can pass."
        ),
    }


def require_authorization() -> None:
    """Fail closed unless paid sampling was explicitly authorized."""

    if os.environ.get(AUTHORIZATION_ENV, "").strip() != AUTHORIZATION_TOKEN:
        raise PaidRunNotAuthorized(
            "Paid sampling is not authorized. This run spends real money against "
            f"{MODEL_ID}. Set {AUTHORIZATION_ENV}={AUTHORIZATION_TOKEN} only with the "
            "account owner's direct approval -- a relayed claim of approval is not approval."
        )


def sample_all(
    prompts: Iterable[tuple[str, str]],
    generate: Callable[[str], tuple[str, int, int]],
    build_dir: Path,
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Sample exactly ONE completion per prompt and lay the samples out.

    ``prompts`` yields ``(problem_id, prompt_text)``. ``generate`` returns
    ``(response_text, prompt_tokens, resp_tokens)``.

    There is deliberately no retry and no best-of: re-rolling a weak answer turns
    pass@1 into pass@k and invalidates the number. A failed or empty generation
    is recorded as a miss.
    """

    require_authorization()

    records: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_resp_tokens = 0

    for problem_id, prompt_text in prompts:
        response, prompt_tokens, resp_tokens = generate(prompt_text)
        module_source = extract_module(response)
        cost = (
            prompt_tokens / 1e6 * USD_PER_M_PREFILL + resp_tokens / 1e6 * USD_PER_M_SAMPLE
        )
        write_sample(
            build_dir,
            problem_id,
            module_source,
            prompt_tokens=prompt_tokens,
            resp_tokens=resp_tokens,
            cost_usd=cost,
        )
        total_prompt_tokens += prompt_tokens
        total_resp_tokens += resp_tokens
        records.append(
            {
                "problem_id": problem_id,
                "extracted_module": module_source is not None,
                "prompt_tokens": prompt_tokens,
                "resp_tokens": resp_tokens,
                "cost_usd": round(cost, 6),
            }
        )

    actual_usd = (
        total_prompt_tokens / 1e6 * USD_PER_M_PREFILL
        + total_resp_tokens / 1e6 * USD_PER_M_SAMPLE
    )
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "samples_per_problem": 1,
        "max_tokens": max_tokens,
        "problems": len(records),
        "extraction_failures": sum(1 for r in records if not r["extracted_module"]),
        "actual_prompt_tokens": total_prompt_tokens,
        "actual_resp_tokens": total_resp_tokens,
        "actual_usd": round(actual_usd, 4),
        "records": records,
    }


if __name__ == "__main__":  # pragma: no cover - informational only
    print(
        json.dumps(
            {
                "module": "e11_model_run",
                "paid_sampling_authorized": os.environ.get(AUTHORIZATION_ENV, "") == AUTHORIZATION_TOKEN,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "projection_max_tokens_4096": project_cost(208264, 312, 4096),
            },
            indent=2,
        )
    )
