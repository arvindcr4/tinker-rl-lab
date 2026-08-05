#!/usr/bin/env python3
"""Mechanical evidence-boundary checks for the Reviewer #36320 corpus revision."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_worthiness_check import ACTIVE_ROSTER, ABSORBED_ARCHIVED  # noqa: E402

DESCENDANT = "platform_hybrid/paper/paper.tex"
MANIFEST = "platform_hybrid/paper/REVIEWER_36320_CORRECTION_MANIFEST.md"
LIVE_BOUNDARIES = {
    "P1": "platform_hybrid/paper/sections/p1_abstract.tex",
    "P2": "platform_hybrid/paper/sections/p2_abstract.tex",
    "P3": "platform_hybrid/paper/sections/p3_abstract.tex",
    "P4": "platform_hybrid/paper/sections/p4_abstract.tex",
    "P5": "platform_hybrid/paper/sections/p5_abstract.tex",
    "P6": "platform_hybrid/paper/sections/p6_abstract.tex",
    "P7": "platform_hybrid/paper/sections/p7_abstract.tex",
    "P8": "platform_hybrid/paper/neurips_2026_variants/sections/abstract_workshop.tex",
    "P9": "platform_hybrid/paper/neurips_2026_variants/sections/abstract_dnb.tex",
    "P10": "zvf-program/theory/paper_P10_zvf_theory.tex",
    "P11": "zvf-program/audit/paper_P11_reproducibility_audit.tex",
    "P12": "platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex",
}
HISTORICAL_ROOTS = {
    "R01": "platform_hybrid/paper/archive/absorbed/R01_acm/acm_main.tex",
    "R02": "platform_hybrid/paper/archive/absorbed/R02_main_zvf/main_zvf.tex",
    "R06": "platform_hybrid/paper/archive/absorbed/R06_min_report/min_report_rl.tex",
    "R07": "platform_hybrid/paper/archive/absorbed/R07_grpo_registry/grpo_registry.tex",
    "U01": "platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex",
    "P08_fraud": "platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex",
}


def read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")


def inclusion_closure(rel: str, seen: set[str] | None = None) -> str:
    """Return the TeX inclusion closure rooted at *rel*, ignoring missing variants."""
    seen = set() if seen is None else seen
    path = (REPO_ROOT / rel).resolve()
    if str(path) in seen or not path.is_file():
        return ""
    seen.add(str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [text]
    for include in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
        child = path.parent / include
        if child.suffix != ".tex":
            child = child.with_suffix(".tex")
        try:
            child_rel = child.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        chunks.append(inclusion_closure(child_rel, seen))
    return "\n".join(chunks)


def qwen_ppo_derived_errors(text: str) -> list[str]:
    """Reject any live numeric/statistical Qwen3-8B PPO row or comparison."""
    errors: list[str] = []
    for raw_row in re.split(r"\\\\", text.lower()):
        row = re.sub(r"\s+", " ", raw_row)
        for qwen in re.finditer(r"qwen3[-_ ]8b", row):
            window = row[max(0, qwen.start() - 100):min(len(row), qwen.end() + 180)]
            if not re.search(r"\bppo\b|\bppo(?=[_\s-])", window):
                continue
            derived_statistic = re.search(
                r"\b(?:cohen|welch|mann(?:-| )?whitney|t-test|p\s*(?:=|<|>|value))\b"
                r"|[&;]\s*[+-]?\d+\.\d+%?"
                r"|\d+\.\d+\s*%"
                r"|\b0\.\d+\b[^.\n]{0,24}\b(?:ppo|grpo)\b"
                r"|\b(?:ppo|grpo)\b[^.\n]{0,24}\b0\.\d+\b",
                window,
            )
            quarantined = any(marker in window for marker in ("provenance conflict", "not estimable", "quarantined"))
            comparison = re.search(r"(?:ppo\s+vs\s+grpo|grpo\s+vs\s+ppo|ppo[_\s-]+qwen3[-_ ]8b|ppo\s+trace)", window)
            if derived_statistic or (comparison and not quarantined):
                errors.append("#36320 inclusion closure retains a derived Qwen PPO comparison statistic")
                break
        if errors:
            break
    return errors


def comparison_claim_scope(text: str, match: re.Match[str]) -> str:
    """Return this comparison's sentence/clause, including pronoun continuations only."""
    paragraph_start = text.rfind("\n\n", 0, match.start()) + 2
    paragraph_end = text.find("\n\n", match.end())
    if paragraph_end == -1:
        paragraph_end = len(text)
    row_start = text.rfind("\\\\", paragraph_start, match.start()) + 2
    row_end = text.find("\\\\", match.end(), paragraph_end)
    if row_end != -1:
        return text[max(paragraph_start, row_start):row_end]

    sentence_start = max(paragraph_start, text.rfind(";", paragraph_start, match.start()) + 1)
    for boundary in re.finditer(r"(?<!\d)[.!?](?=\s|$)", text[paragraph_start:match.start()]):
        sentence_start = paragraph_start + boundary.end()
    sentence_end_match = re.search(r"(?:;|(?<!\d)[.!?](?=\s|$))", text[match.end():paragraph_end])
    sentence_end = match.end() + sentence_end_match.end() if sentence_end_match else paragraph_end
    while sentence_end < paragraph_end:
        continuation = re.match(r"\s*(?:its|it|this|that)\b", text[sentence_end:paragraph_end])
        if not continuation:
            break
        next_end = re.search(r"(?<!\d)[.!?](?=\s|$)", text[sentence_end + continuation.end():paragraph_end])
        if not next_end:
            sentence_end = paragraph_end
            break
        sentence_end += continuation.end() + next_end.end()
    return text[sentence_start:sentence_end]


def descendant_claim_errors(text: str) -> list[str]:
    """Reject residual live claims without scanning withdrawn audit prose."""
    errors: list[str] = []
    lowered = text.lower()
    forbidden = (
        "tell a practitioner what action to take next",
        "clean held-out",
        "every prompt contributes zero gradient",
        "no gradient flows",
        "contributes no gradient",
        "nulls the update",
        "mediated entirely by the within-group spread of rewards",
        "recommended deployment",
        "deploy iso-g@",
        "deploy hysteresis@",
        "default for budget-constrained deployment",
        "improves grade-school arithmetic and multi-step reasoning",
        "zero-gradient theorem",
    )
    for phrase in forbidden:
        if phrase in lowered:
            errors.append(f"#36320 inclusion closure retains forbidden claim: {phrase}")
    for match in re.finditer(r"gradient vanishes", lowered):
        prefix = lowered[max(0, match.start() - 80):match.start()]
        if "does not assert that the total " not in prefix:
            errors.append("#36320 inclusion closure retains forbidden claim: gradient vanishes")
            break
    if re.search(r"qwen3-8b.{0,300}(22\.5|35\.0)", text, flags=re.IGNORECASE | re.DOTALL):
        errors.append("#36320 inclusion closure retains a live Qwen PPO numeric comparison")
    errors.extend(qwen_ppo_derived_errors(text))
    for match in re.finditer(r"\badopt\b.{0,160}\b(?:controller|deployment|default)\b", lowered, flags=re.DOTALL):
        prefix = lowered[max(0, match.start() - 40):match.start()]
        if "rather than " not in prefix and "do not " not in prefix:
            errors.append("#36320 inclusion closure retains an adoption prescription")
            break
    comparison_82_83 = re.compile(
        r"(?:82(?:\.0)?(?:\\?%)?.{0,140}?83\.3|83\.3.{0,140}?82(?:\.0)?(?:\\?%)?)",
        flags=re.DOTALL,
    )
    for match in comparison_82_83.finditer(lowered):
        local = comparison_claim_scope(lowered, match)
        has_one_sample_fixed_baseline = "one-sample" in local and bool(
            re.search(r"fixed(?:-|\s+)bas(?:eline|e)", local)
        )
        has_paired_boundary = bool(re.search(r"not\s+(?:a\s+)?paired-seed", local))
        has_reviewed_boundary = (
            "not reviewed-record evidence" in local
            or "not a paired-seed or reviewed-record evidence" in local
            or "not paired-seed or reviewed-record evidence" in local
            or "not part of the reviewed record" in local
            or "not reviewed record" in local
            or "not reviewed" in local
            or bool(re.search(r"not\s+(?:a\s+)?paired-seed[^.]{0,120}reviewed-record", local))
        )
        if not (has_one_sample_fixed_baseline and has_paired_boundary and has_reviewed_boundary):
            errors.append("#36320 inclusion closure has an unsafe local 82.0-to-83.3 statement")
            break
    if re.search(r"primary inferential claims?.{0,500}held-out 5-seed", lowered, flags=re.DOTALL):
        errors.append("#36320 inclusion closure treats a held-out 5-seed comparison as a primary inferential claim")
    return errors


def p7_operational_errors(text: str) -> list[str]:
    """Reject P7 imperatives while allowing retrospective vocabulary."""
    lowered = text.lower()
    patterns = (
        r"\b(?:updated|operational|design) recommendation\b",
        r"\bprincipled recommendation\b",
        r"\bbest-\$?\\?tau\$? recommendation\b",
        r"\b(?:we )?adopt\b.{0,160}\b(?:controller|deployment|default|operating|setting)\b",
        r"\b(?:we\s+)?recommend\s+(?:\\(?:textbf|emph)\{)?c_?\d\b",
        r"\bc_?\d\s+is\s+recommended\b",
        r"\bpick\s+(?:\\(?:textbf|emph)\{)?c_?\d\b",
        r"\bdefault\s+to\b",
        r"\badopt\s+(?:\\(?:textbf|emph)\{)?c_?\d\b",
        r"\bshould\s+be\s+adopted\b",
        r"\b(?:use|deploy|avoid|replace)\s+(?:the\s+)?(?:\\textbf\{)?(?:c\d|ccc|bayesian|hybrid|controller|policy)\b",
        r"\b(?:recommended|default) controller\b",
        r"\brecommended default\b",
        r"\b(?:best|recommended) operating point\b",
        r"\bshould (?:pick|use)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.DOTALL)
        if not match:
            continue
        clause_start = max(
            lowered.rfind(".", 0, match.start()),
            lowered.rfind(";", 0, match.start()),
            lowered.rfind("\n", 0, match.start()),
        ) + 1
        negated = re.search(r"\b(?:do\s+not|not|never)\b", lowered[clause_start:match.start()])
        if not negated:
            return ["#36320 P7 inclusion closure retains an operational imperative"]
    return []


def p10_centered_contribution_errors(text: str) -> list[str]:
    """Reject T2 shorthand that can be read as a total-update guarantee."""
    lowered = text.lower()
    semantic = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", lowered)
    semantic = re.sub(r"\\[a-zA-Z]+", " ", semantic)
    semantic = semantic.replace("{}", "")
    semantic = re.sub(r"[-_~]", " ", semantic)
    shorthand = re.compile(
        r"(?:non\s*zero\s+(?:(?:reward\s+)?gradient|(?:policy\s+)?update)"
        r"|rollouts\s*to\s*(?:potentially\s*)?non\s*zero[^\n]{0,100}(?:gradient|update))"
    )
    for match in shorthand.finditer(semantic):
        local = semantic[max(0, match.start() - 180):min(len(semantic), match.end() + 260)]
        if "potentially nonzero centered reward contrast contribution" not in local:
            return ["#36320 P10 inclusion closure retains an unscoped nonzero update/gradient shorthand"]
    return []


def live_root_claim_errors(paper_id: str, text: str) -> list[str]:
    """Scope semantic claim checks to one active root's inclusion closure."""
    errors = descendant_claim_errors(text)
    if paper_id == "P7":
        errors.extend(p7_operational_errors(text))
    if paper_id == "P10":
        errors.extend(p10_centered_contribution_errors(text))
    return [f"{paper_id}: {error}" for error in errors]


def check() -> list[str]:
    errors: list[str] = []
    if set(ACTIVE_ROSTER) != {f"P{i}" for i in range(1, 13)}:
        errors.append("canonical live roster is not exactly P1--P12")
    if set(ABSORBED_ARCHIVED) != set(HISTORICAL_ROOTS):
        errors.append("canonical absorbed set differs from the six audit roots")
    manifest = read(MANIFEST)
    for paper_id, root in ACTIVE_ROSTER.items():
        if not (REPO_ROOT / root).is_file():
            errors.append(f"{paper_id}: missing live root {root}")
        boundary = LIVE_BOUNDARIES[paper_id]
        if "August 2026 evidence boundary" not in read(boundary):
            errors.append(f"{paper_id}: missing scope boundary in {boundary}")
        if paper_id not in manifest or root not in manifest:
            errors.append(f"{paper_id}: manifest does not enumerate {root}")
        errors.extend(live_root_claim_errors(paper_id, inclusion_closure(root)))
    for archive_id, root in HISTORICAL_ROOTS.items():
        text = read(root)
        if "August 2026 claim-consistency audit note" not in text:
            errors.append(f"{archive_id}: missing historical audit note")
        if archive_id not in manifest:
            errors.append(f"{archive_id}: manifest does not enumerate archive")
    descendant = inclusion_closure(DESCENDANT)
    errors.extend(live_root_claim_errors("DESC", descendant))
    required = (
        "not its authenticated or byte-identical",
        "sections/diagnostic_method",
        "provenance conflict; not available",
        "quarantined",
        "not evaluated",
    )
    for marker in required:
        if marker not in descendant:
            errors.append(f"#36320 descendant missing marker: {marker}")
    for forbidden in ("92.08\\%", "mean $91.6\\%", "22.5\\% in ledger"):
        if forbidden in descendant:
            errors.append(f"#36320 descendant retains withdrawn aggregate/conflict: {forbidden}")
    method = read("platform_hybrid/paper/sections/diagnostic_method.tex")
    for marker in ("not canonical GRPO", "centered reward-contrast term", "completion-only token masking", "future-paper evidence"):
        if marker not in method:
            errors.append(f"diagnostic method missing scope marker: {marker}")
    return errors


def main() -> int:
    errors = check()
    print(f"LIVE_ROOTS={len(ACTIVE_ROSTER)}")
    print(f"HISTORICAL_AUDIT_ROOTS={len(HISTORICAL_ROOTS)}")
    print(f"DESCENDANT={DESCENDANT}")
    if errors:
        print("FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
