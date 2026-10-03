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


def parse_answer(text: str) -> dict[str, str | None]:
    """Parse explicit numeric answers; malformed/invalid/conflicting claims fail closed.

    As in v1, a bare numeric final line is allowed; numbers embedded in prose are
    never searched. Unlike v1, earlier anchored 'Answer:' claims and a bare final
    numeric line participate in the ambiguity guard even when other markers exist.
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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pending_heading = False
    for index, raw in enumerate(lines):
        line = _canonical_line(raw)
        source = None
        value = None
        if "####" in line:
            if not line.startswith("####") or line.count("####") != 1:
                return failed("malformed_marker")
            payload = line[4:].strip()
            if _HEADING.fullmatch(payload) or payload == "<answer>":
                if pending_heading:
                    return failed("invalid_numeric_answer")
                pending_heading = True
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

PARSER_VERSION = 2
