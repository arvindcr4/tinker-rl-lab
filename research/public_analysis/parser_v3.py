"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

_ATOM = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"


_NUMBER = re.compile(rf"({_ATOM})(?:\s*/\s*({_ATOM}))?\Z")


def normalize_number(text: str) -> str | None:
    text = text.strip().replace("−", "-")
    if len(text) > 100:
        return None
    for left, right in (("$", "$"), (r"\(", r"\)")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    # Accept a simple LaTeX fraction; no expression evaluation or recursive parser.
    frac = re.fullmatch(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if frac:
        text = frac.group(1) + "/" + frac.group(2)
    match = _NUMBER.fullmatch(text)
    if not match:
        return None
    try:
        numerator = Fraction(Decimal(match.group(1).replace(",", "")))
        denominator = Fraction(Decimal(match.group(2).replace(",", ""))) if match.group(2) else 1
        value = numerator / denominator
        return str(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _boxed_values(text: str) -> list[str] | None:
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                values.append(text[start:index])
                break
        else:
            return None
    return values


_HEADING = re.compile(r"(?:final\s+answer|answer)\s*:", re.IGNORECASE)


_EXPLICIT = re.compile(r"(?:final\s+answer|answer)\s*(?:is\s*|:\s*|=\s*)(.+)", re.IGNORECASE)


_XML = re.compile(r"<answer>([^<>]+)</answer>")


_XML_LIKE = re.compile(r"<\s*/?\s*answer", re.IGNORECASE)


def _canonical_line(line: str) -> str:
    line = line.strip()
    if line.startswith("**####") and line.endswith("**"):
        return line[2:-2].strip()
    return line


def _whole_numeric_box(line: str) -> str | None:
    """Return one complete numeric box payload, with no surrounding content."""
    match = re.fullmatch(r"\\boxed\s*\{(.*)\}", line)
    if match and normalize_number(match.group(1)) is not None:
        return match.group(1)
    return None


def parse_answer(text: str) -> dict[str, str | None]:
    """Parse explicit numeric answers; malformed/invalid/conflicting claims fail closed.

    As in v1, a bare numeric final line is allowed; numbers embedded in prose are
    never searched. Unlike v1, earlier anchored 'Answer:' claims and a bare final
    numeric line participate in the ambiguity guard even when other markers exist.
    New R2/R3 compound forms require physical adjacency within their three-line
    blocks, not merely adjacency after blank-line filtering. Outer whitespace and
    blank lines between an ordinary heading and its successor retain v2 behavior.
    Known terminal-token removal remains the caller's responsibility, as in v1.
    """
    def failed(status: str) -> dict[str, str | None]:
        return {"status": status, "source": None, "value": None}

    text = text.strip()
    if not text:
        return failed("missing_final_answer")
    boxes = _boxed_values(text)
    if boxes is None:
        return failed("malformed_box")
    candidates = [("boxed", value) for value in boxes]
    numbered_lines = [(number, line.strip()) for number, line in enumerate(text.splitlines())
                      if line.strip()]
    positions = [number for number, _ in numbered_lines]
    lines = [line for _, line in numbered_lines]
    pending_heading = False
    pending_textual_heading = False
    consumed_until = 0
    for index, raw in enumerate(lines):
        if index < consumed_until:
            continue
        line = _canonical_line(raw)
        adjacent_triplet = (index + 2 < len(lines)
                            and positions[index + 1] == positions[index] + 1
                            and positions[index + 2] == positions[index] + 2)
        if pending_heading and raw == "$$":
            # R2: consume delimiters only after verifying the exact entire block.
            if (not pending_textual_heading or not adjacent_triplet or lines[index + 2] != "$$"
                    or _whole_numeric_box(lines[index + 1]) is None):
                return failed("invalid_numeric_answer")
            # The unchanged global box scan already added this numeric claim.
            consumed_until = index + 3
            pending_heading = False
            continue
        source = None
        value = None
        if "####" in line:
            if not line.startswith("####") or line.count("####") != 1:
                return failed("malformed_marker")
            payload = line[4:].strip()
            # R1: only a complete bold heading label, never arbitrary bold text.
            if (raw == line and payload.startswith("**") and payload.endswith("**")
                    and _HEADING.fullmatch(payload[2:-2])):
                payload = payload[2:-2]
            # R3: anchored opener/body/closer; do not skip blank or body lines.
            # A full-line bold wrapper is not one of these literal three lines.
            if (raw == line and payload == "<answer>" and adjacent_triplet
                    and lines[index + 1].startswith("####")
                    and lines[index + 2].startswith("####")
                    and lines[index + 2][4:].strip() == "</answer>"
                    and normalize_number(lines[index + 1][4:].strip()) is not None):
                if pending_heading:
                    return failed("invalid_numeric_answer")
                candidates.append(("hash_xml_three_line", lines[index + 1][4:].strip()))
                consumed_until = index + 3
                pending_heading = False
                continue
            if _HEADING.fullmatch(payload) or payload == "<answer>":
                if pending_heading:
                    return failed("invalid_numeric_answer")
                pending_heading = True
                pending_textual_heading = payload != "<answer>"
                continue
            source, value = "hash_marker", payload
        elif _XML_LIKE.search(line):
            source, value = "xml_answer", line
        else:
            explicit = _EXPLICIT.fullmatch(line)
            if explicit:
                source, value = "explicit_final", explicit.group(1).strip().removesuffix(".")
            elif (pending_heading or index == len(lines) - 1) and normalize_number(line) is not None:
                source, value = ("heading_numeric_line" if pending_heading else "standalone_final_line"), line
            elif pending_heading:
                # Only an entire balanced box can satisfy a heading, never prose.
                if re.fullmatch(r"\\boxed\s*\{.*\}", line) and _boxed_values(line):
                    pending_heading = False
                    continue
                return failed("invalid_numeric_answer")
        if value is not None:
            if _XML_LIKE.search(value):
                xml = _XML.fullmatch(value)
                if not xml:
                    return failed("malformed_answer_tag")
                value = xml.group(1).strip()
                source = "hash_xml_answer" if source == "hash_marker" else "xml_answer"
            candidates.append((source, value))
            pending_heading = False
    if pending_heading:
        return failed("invalid_numeric_answer")
    if not candidates:
        return failed("missing_final_answer")
    parsed = [(source, normalize_number(value)) for source, value in candidates]
    if any(value is None for _, value in parsed):
        return failed("invalid_numeric_answer")
    if len({value for _, value in parsed}) != 1:
        return failed("ambiguous_final_answers")
    return {"status": "ok", "source": "+".join(dict.fromkeys(source for source, _ in parsed)),
            "value": parsed[0][1]}

PARSER_VERSION = 3
PARSER_STATUS = "prospective_candidate_not_deployed"
