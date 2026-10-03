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


def strip_terminal_tokens(text: str) -> str:
    """Only remove known terminal markers, retaining original text in raw receipts."""
    text = text.rstrip()
    while True:
        before = text
        for token in ("<|im_end|>", "<|endoftext|>"):
            if text.endswith(token):
                text = text[:-len(token)].rstrip()
        if text == before:
            return text


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


def parse_answer(text: str) -> dict[str, str | None]:
    """Never search for the last number in prose; ambiguity fails closed."""
    failed = {"status": "missing_final_answer", "source": None, "value": None}
    text = text.strip()
    if not text:
        return failed
    candidates: list[tuple[str, str]] = []
    boxes = _boxed_values(text)
    if boxes is None:
        return {**failed, "status": "malformed_box"}
    candidates.extend(("boxed", value) for value in boxes)
    for line in text.splitlines():
        if "####" in line:
            if not line.lstrip().startswith("####") or line.count("####") != 1:
                return {**failed, "status": "malformed_marker"}
            candidates.append(("hash_marker", line.lstrip()[4:].strip()))
    last_line = text.splitlines()[-1].strip()
    explicit = re.fullmatch(r"(?:final\s+answer|answer)\s*(?:is\s*|:\s*|=\s*)(.+)",
                            last_line, flags=re.IGNORECASE)
    if explicit:
        candidates.append(("explicit_final", explicit.group(1).strip().removesuffix(".")))
    if not candidates:
        numeric = normalize_number(last_line)
        if numeric is None:
            return failed
        candidates = [("standalone_final_line", last_line)]
    parsed = [(source, normalize_number(value)) for source, value in candidates]
    if any(value is None for _, value in parsed):
        return {**failed, "status": "invalid_numeric_answer"}
    if len({value for _, value in parsed}) != 1:
        return {**failed, "status": "ambiguous_final_answers"}
    return {"status": "ok", "source": "+".join(dict.fromkeys(source for source, _ in parsed)),
            "value": parsed[0][1]}

PARSER_VERSION = 1
