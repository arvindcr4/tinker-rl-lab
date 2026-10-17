#!/usr/bin/env python3
"""Read and audit the complete 18-paper Tinker RL manuscript program.

Rendered PDF text is used so every page that a reviewer sees is included.
Page boundaries are preserved, and each paper receives a structured editorial
audit from the repository-mandated Groq/Kimi model. Results are resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = "kimi-k2-0905-preview"
REQUEST_BUDGET = 0
REQUESTS_USED = 0

PAPERS = [
    (
        "P01",
        "Scaling laws",
        "platform_hybrid/paper/paper_P1_scaling.tex",
        "platform_hybrid/paper/paper_P1_scaling.pdf",
    ),
    (
        "P02",
        "Zero-variance fraction",
        "platform_hybrid/paper/paper_P2_zvf.tex",
        "platform_hybrid/paper/paper_P2_zvf.pdf",
    ),
    (
        "P03",
        "Group size",
        "platform_hybrid/paper/paper_P3_group_size.tex",
        "platform_hybrid/paper/paper_P3_group_size.pdf",
    ),
    (
        "P04",
        "Length bias",
        "platform_hybrid/paper/paper_P4_length_bias.tex",
        "platform_hybrid/paper/paper_P4_length_bias.pdf",
    ),
    (
        "P05",
        "MIN-REPORT-RL",
        "platform_hybrid/paper/paper_P5_minreport.tex",
        "platform_hybrid/paper/paper_P5_minreport.pdf",
    ),
    (
        "P06",
        "GRPO registry",
        "platform_hybrid/paper/paper_P6_registry.tex",
        "platform_hybrid/paper/paper_P6_registry.pdf",
    ),
    (
        "P07",
        "ZVF controller",
        "platform_hybrid/paper/paper_P7_zvf_controller.tex",
        "platform_hybrid/paper/paper_P7_zvf_controller.pdf",
    ),
    (
        "P08",
        "Fraud detection",
        "platform_hybrid/paper/paper_P8_fraud.tex",
        "platform_hybrid/paper/paper_P8_fraud.pdf",
    ),
    (
        "R01",
        "ACM benchmark variant",
        "platform_hybrid/paper/acm_main.tex",
        "platform_hybrid/paper/acm_main.pdf",
    ),
    (
        "R02",
        "NeurIPS ZVF variant",
        "platform_hybrid/paper/neurips_2026_variants/main_zvf.tex",
        "platform_hybrid/paper/neurips_2026_variants/main_zvf.pdf",
    ),
    (
        "R03",
        "NeurIPS workshop artifact",
        "platform_hybrid/paper/neurips_2026_variants/main_workshop.tex",
        "platform_hybrid/paper/neurips_2026_variants/main_workshop.pdf",
    ),
    (
        "R04",
        "NeurIPS DNB benchmark",
        "platform_hybrid/paper/neurips_2026_variants/main_dnb.tex",
        "platform_hybrid/paper/neurips_2026_variants/main_dnb.pdf",
    ),
    ("R05", "ZVF theory", "zvf-program/theory/zvf_theory.tex", "zvf-program/theory/zvf_theory.pdf"),
    (
        "R06",
        "MIN-REPORT position",
        "zvf-program/position/min_report_rl.tex",
        "zvf-program/position/min_report_rl.pdf",
    ),
    (
        "R07",
        "Living GRPO registry",
        "zvf-program/registry/grpo_registry.tex",
        "zvf-program/registry/grpo_registry.pdf",
    ),
    (
        "R08",
        "Reproducibility audit",
        "zvf-program/audit/reproducibility_audit.tex",
        "zvf-program/audit/reproducibility_audit.pdf",
    ),
    (
        "U01",
        "Umbrella benchmark",
        "platform_hybrid/paper/main.tex",
        "platform_hybrid/paper/main.pdf",
    ),
    (
        "N01",
        "Unified signal starvation",
        "platform_hybrid/paper/unified_signal_starvation/main.tex",
        "output/pdf/signal-starvation-grpo-ppo-sao.pdf",
    ),
]

SYSTEM = """You are a meticulous senior ML/RL paper editor. Read the complete
paper supplied by the user. Distinguish verified evidence, mathematical
identity, interpretation, proposal, and untested causal claim. Do not invent
missing experiments or citations. Return only one JSON object."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    result = subprocess.run(["pdfinfo", str(path)], check=True, text=True, capture_output=True)
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from {path}")
    return int(match.group(1))


def extract_pdf(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        text=True,
        capture_output=True,
    )
    pages = result.stdout.split("\f")
    return "\n".join(
        f"\n[PAGE {index}]\n{page.strip()}"
        for index, page in enumerate(pages, start=1)
        if page.strip()
    )


def parse_json(raw: str) -> dict[str, object]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def call_with_retry(client: object, messages: list[dict[str, str]]) -> str:
    global REQUESTS_USED
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            if REQUESTS_USED >= REQUEST_BUDGET:
                raise ValueError("aggregate request budget exhausted before dispatch")
            REQUESTS_USED += 1
            completion = client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.1,
                max_tokens=5000,
            )
            if completion.choices[0].finish_reason != "stop":
                raise ValueError("incomplete model output")
            return completion.choices[0].message.content or ""
        except Exception as exc:  # network/provider errors vary by SDK version
            if isinstance(exc, ValueError) or getattr(exc, "status_code", None) in {
                400,
                401,
                403,
                404,
                422,
            }:
                raise
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def paper_prompt(identifier: str, label: str, text: str) -> str:
    return f"""Audit paper {identifier}: {label}. This is one manuscript in an
overlapping 18-paper program. The durable thesis is a bounded contribution on
ZVF as a diagnostic for signal starvation in group-relative RL, matched-budget
group-size failure modes, and reproducibility/stack reporting. Some papers are
venue variants or companion scopes; duplication is acceptable only when the
scope is explicit.

Read every supplied page. Return a JSON object with exactly these keys:
  identity: {{title, one_sentence_thesis, intended_paper_type}}
  claim_status: [{{claim, status, evidence_in_paper, page}}]
    where status is one of verified, algebraic, descriptive, proposed,
    externally_sourced, unsupported, or contradicted
  strongest_contribution: string
  major_problems: [{{severity, problem, exact_anchor, page, repair}}]
  overlap_and_scope: {{duplicates, unique_material, scope_sentence_to_add}}
  consistency_checks: [{{topic, verdict, exact_anchor, page}}]
  concrete_edits: [{{priority, target_anchor, replacement_or_action, reason}}]
  evidence_needed: [string]
  verdict: one of keep, merge, venue_variant, park, or retire

For each problem or edit, anchor it to text actually present in the paper.
Prefer high-impact scientific edits over stylistic rewriting. Flag causal
overreach, stale numerical claims, mismatched sample counts, undefined scope,
cross-reference failures, contradictory conclusions, missing limitations,
and duplicated novelty. Do not punish a paper merely for being a declared
venue variant.

COMPLETE PAPER TEXT:
{text}
"""


SCHEMA_VERSION = 2
PAPER_KEYS = {
    "identity",
    "claim_status",
    "strongest_contribution",
    "major_problems",
    "overlap_and_scope",
    "consistency_checks",
    "concrete_edits",
    "evidence_needed",
    "verdict",
}
CROSS_KEYS = {
    "canonical_map",
    "program_contradictions",
    "shared_edits",
    "paper_specific_edits",
    "evidence_firewall",
    "recommended_order",
    "completion_checks",
}


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_audit(value, cross=False):
    keys = CROSS_KEYS if cross else PAPER_KEYS
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("audit schema keys mismatch")
    if cross:
        if any(not isinstance(v, (dict, list, str)) for v in value.values()):
            raise ValueError("invalid synthesis field")
        edits = json.dumps(value["paper_specific_edits"])
        if any(identifier not in edits for identifier, *_ in PAPERS):
            raise ValueError("synthesis lacks paper coverage")
    else:
        for key in ("identity", "overlap_and_scope"):
            if not isinstance(value[key], dict):
                raise ValueError("expected object: " + key)
        for key in (
            "claim_status",
            "major_problems",
            "consistency_checks",
            "concrete_edits",
            "evidence_needed",
        ):
            if not isinstance(value[key], list):
                raise ValueError("expected array: " + key)
        if not isinstance(value["strongest_contribution"], str) or value["verdict"] not in {
            "keep",
            "merge",
            "venue_variant",
            "park",
            "retire",
        }:
            raise ValueError("invalid paper verdict/contribution")
        nested = {
            "identity": {"title", "one_sentence_thesis", "intended_paper_type"},
            "overlap_and_scope": {"duplicates", "unique_material", "scope_sentence_to_add"},
        }
        for key, fields in nested.items():
            if set(value[key]) != fields:
                raise ValueError("invalid " + key)
        rows = {
            "claim_status": {"claim", "status", "evidence_in_paper", "page"},
            "major_problems": {"severity", "problem", "exact_anchor", "page", "repair"},
            "consistency_checks": {"topic", "verdict", "exact_anchor", "page"},
            "concrete_edits": {"priority", "target_anchor", "replacement_or_action", "reason"},
        }
        for key, fields in rows.items():
            if any(not isinstance(row, dict) or set(row) != fields for row in value[key]):
                raise ValueError("invalid " + key)
        if any(
            row["status"]
            not in {
                "verified",
                "algebraic",
                "descriptive",
                "proposed",
                "externally_sourced",
                "unsupported",
                "contradicted",
            }
            for row in value["claim_status"]
        ):
            raise ValueError("invalid claim status")
        if any(not isinstance(v, str) for v in value["evidence_needed"]):
            raise ValueError("invalid evidence requirement")
    return value


def cached_review(client, directory, prompt, inputs, cross=False):
    key = digest_json(
        {
            "inputs": inputs,
            "system": SYSTEM,
            "prompt": prompt,
            "model": MODEL,
            "generation": {"temperature": 0.1, "max_tokens": 5000},
            "schema_version": SCHEMA_VERSION,
        }
    )
    path = directory / (key + ".json")
    if path.exists():
        try:
            record = json.loads(path.read_text())
            if (
                record["cache_key"] == key
                and record["status"] == "success"
                and record["schema_version"] == SCHEMA_VERSION
            ):
                return validate_audit(record["audit"], cross)
        except (ValueError, KeyError, TypeError):
            pass
    audit = validate_audit(
        parse_json(
            call_with_retry(
                client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            )
        ),
        cross,
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Immutable successful cache entries; temporary sibling prevents partial reads.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "cache_key": key,
                "status": "success",
                "audit": audit,
            },
            handle,
            indent=2,
        )
        temporary = Path(handle.name)
    temporary.replace(path)
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-requests",
        type=int,
        required=True,
        help="Authorized aggregate attempts including retries",
    )
    args = parser.parse_args()
    if args.max_requests < 1:
        parser.error("--max-requests must be positive")
    global REQUEST_BUDGET, REQUESTS_USED
    REQUEST_BUDGET, REQUESTS_USED = args.max_requests, 0
    output = args.output_dir.resolve()
    if output == OUT.resolve() or OUT.resolve() in output.parents:
        parser.error("historical audit directory is frozen; choose a new output directory")
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=240.0)
    audits = []
    for identifier, label, _tex_rel, pdf_rel in PAPERS:
        pdf = ROOT / pdf_rel
        text = extract_pdf(pdf)
        if len(text) > 300000:
            raise ValueError("paper exceeds input-size budget")
        audit = cached_review(
            client,
            output / "audits" / identifier,
            paper_prompt(identifier, label, text),
            {"pdf_sha256": sha256(pdf), "text_sha256": hashlib.sha256(text.encode()).hexdigest()},
        )
        audits.append({"paper_id": identifier, "audit": audit})
    cross_prompt = (
        "Synthesize these audits without adding evidence. Return exactly these keys: "
        + ", ".join(sorted(CROSS_KEYS))
        + ". Include every paper ID in paper_specific_edits. AUDITS: "
        + json.dumps(audits, sort_keys=True)
    )
    cached_review(
        client,
        output / "synthesis",
        cross_prompt,
        {"upstream": [digest_json(audit) for audit in audits]},
        cross=True,
    )
    print(f"Successful content-addressed audit records: {output}")


if __name__ == "__main__":
    main()
