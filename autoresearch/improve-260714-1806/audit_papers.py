#!/usr/bin/env python3
"""Read and audit the complete 18-paper Tinker RL manuscript program.

Rendered PDF text is used so every page that a reviewer sees is included.
Page boundaries are preserved, and each paper receives a structured editorial
audit from the repository-mandated Groq/Kimi model. Results are resumable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from groq import Groq


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = "kimi-k2-0905-preview"

PAPERS = [
    ("P01", "Scaling laws", "platform_hybrid/paper/paper_P1_scaling.tex", "platform_hybrid/paper/paper_P1_scaling.pdf"),
    ("P02", "Zero-variance fraction", "platform_hybrid/paper/paper_P2_zvf.tex", "platform_hybrid/paper/paper_P2_zvf.pdf"),
    ("P03", "Group size", "platform_hybrid/paper/paper_P3_group_size.tex", "platform_hybrid/paper/paper_P3_group_size.pdf"),
    ("P04", "Length bias", "platform_hybrid/paper/paper_P4_length_bias.tex", "platform_hybrid/paper/paper_P4_length_bias.pdf"),
    ("P05", "MIN-REPORT-RL", "platform_hybrid/paper/paper_P5_minreport.tex", "platform_hybrid/paper/paper_P5_minreport.pdf"),
    ("P06", "GRPO registry", "platform_hybrid/paper/paper_P6_registry.tex", "platform_hybrid/paper/paper_P6_registry.pdf"),
    ("P07", "ZVF controller", "platform_hybrid/paper/paper_P7_zvf_controller.tex", "platform_hybrid/paper/paper_P7_zvf_controller.pdf"),
    ("P08", "Fraud detection", "platform_hybrid/paper/paper_P8_fraud.tex", "platform_hybrid/paper/paper_P8_fraud.pdf"),
    ("R01", "ACM benchmark variant", "platform_hybrid/paper/acm_main.tex", "platform_hybrid/paper/acm_main.pdf"),
    ("R02", "NeurIPS ZVF variant", "platform_hybrid/paper/neurips_2026_variants/main_zvf.tex", "platform_hybrid/paper/neurips_2026_variants/main_zvf.pdf"),
    ("R03", "NeurIPS workshop artifact", "platform_hybrid/paper/neurips_2026_variants/main_workshop.tex", "platform_hybrid/paper/neurips_2026_variants/main_workshop.pdf"),
    ("R04", "NeurIPS DNB benchmark", "platform_hybrid/paper/neurips_2026_variants/main_dnb.tex", "platform_hybrid/paper/neurips_2026_variants/main_dnb.pdf"),
    ("R05", "ZVF theory", "zvf-program/theory/zvf_theory.tex", "zvf-program/theory/zvf_theory.pdf"),
    ("R06", "MIN-REPORT position", "zvf-program/position/min_report_rl.tex", "zvf-program/position/min_report_rl.pdf"),
    ("R07", "Living GRPO registry", "zvf-program/registry/grpo_registry.tex", "zvf-program/registry/grpo_registry.pdf"),
    ("R08", "Reproducibility audit", "zvf-program/audit/reproducibility_audit.tex", "zvf-program/audit/reproducibility_audit.pdf"),
    ("U01", "Umbrella benchmark", "platform_hybrid/paper/main.tex", "platform_hybrid/paper/main.pdf"),
    ("N01", "Unified signal starvation", "platform_hybrid/paper/unified_signal_starvation/main.tex", "output/pdf/signal-starvation-grpo-ppo-sao.pdf"),
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
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, text=True, capture_output=True
    )
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


def call_with_retry(client: Groq, messages: list[dict[str, str]]) -> str:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.1,
                max_tokens=5000,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:  # network/provider errors vary by SDK version
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 ** attempt)
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


def main() -> None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=240.0)
    inventory: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for index, (identifier, label, tex_rel, pdf_rel) in enumerate(PAPERS, start=1):
        tex_path, pdf_path = ROOT / tex_rel, ROOT / pdf_rel
        if not tex_path.exists() or not pdf_path.exists():
            raise FileNotFoundError(f"Missing root for {identifier}: {tex_path} / {pdf_path}")

        text_path = OUT / "text" / f"{identifier}.txt"
        audit_path = OUT / "audits" / f"{identifier}.json"
        text = extract_pdf(pdf_path)
        text_path.write_text(text)
        row = {
            "id": identifier,
            "label": label,
            "tex": tex_rel,
            "pdf": pdf_rel,
            "pages": pdf_pages(pdf_path),
            "words": len(text.split()),
            "characters": len(text),
            "pdf_sha256": sha256(pdf_path),
        }
        inventory.append(row)

        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            print(f"[{index:02d}/{len(PAPERS)}] reuse {identifier}", flush=True)
        else:
            print(
                f"[{index:02d}/{len(PAPERS)}] audit {identifier}: "
                f"{row['pages']} pages, {row['words']} words",
                flush=True,
            )
            raw = call_with_retry(
                client,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": paper_prompt(identifier, label, text)},
                ],
            )
            try:
                audit = parse_json(raw)
            except Exception as exc:
                (OUT / "audits" / f"{identifier}.raw.txt").write_text(raw)
                audit = {"parse_error": str(exc), "raw_saved": True}
            audit["paper_id"] = identifier
            audit["paper_label"] = label
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        audits.append(audit)

    with (OUT / "inventory.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(inventory)

    cross_path = OUT / "cross_paper_audit.json"
    if not cross_path.exists():
        compact = json.dumps(audits, ensure_ascii=True)
        cross_prompt = f"""Synthesize the following 18 complete-paper audits into
a program-level editorial plan. Return JSON with keys: canonical_map,
program_contradictions, shared_edits, paper_specific_edits, evidence_firewall,
recommended_order, and completion_checks. Every paper ID P01-P08, R01-R08,
U01, and N01 must appear in paper_specific_edits. Preserve the thesis-first
direction; do not fabricate new results. Prefer editing shared source files
when several roots include them, but require a paper-specific scope improvement
for every root.

AUDITS:
{compact}
"""
        print("[19/19] synthesize cross-paper audit", flush=True)
        raw = call_with_retry(
            client,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": cross_prompt},
            ],
        )
        try:
            cross = parse_json(raw)
        except Exception as exc:
            (OUT / "cross_paper_audit.raw.txt").write_text(raw)
            cross = {"parse_error": str(exc), "raw_saved": True}
        cross_path.write_text(json.dumps(cross, indent=2) + "\n")

    print(f"Wrote {OUT / 'inventory.tsv'}", flush=True)
    print(f"Wrote {cross_path}", flush=True)


if __name__ == "__main__":
    main()
