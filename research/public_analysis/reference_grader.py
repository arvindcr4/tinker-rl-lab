"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

import re
from typing import Optional

def extract_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from response.

    Handles multiple output formats:
    - GSM8K standard: #### <number>
    - LaTeX boxed: \\boxed{<number>}
    - Explicit statement: "the answer is <number>"
    - Qwen3 <think> reasoning: looks for the last numeric conclusion
    """
    # Strip <think>...</think> wrapper if present — answer is usually restated after
    # or at the end of reasoning
    clean = text
    # If there's content after </think>, prefer that
    think_end = text.find('</think>')
    if think_end != -1:
        after_think = text[think_end + 8:].strip()
        if after_think:
            clean = after_think  # Use post-thinking answer

    # GSM8K format: answer ends with "#### <number>"
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', clean)
    if match:
        return match.group(1).replace(',', '')

    boxed = re.search(r'\\boxed\{\s*(-?\d[\d,]*\.?\d*)\s*\}', clean)
    if boxed:
        return boxed.group(1).replace(',', '')

    # "the answer is X" / "answer: X" patterns
    explicit = re.search(r'(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)', clean)
    if explicit:
        return explicit.group(1).replace(',', '')

    # "= X cups/dollars/etc" at end of reasoning — take the last "= <number>"
    equals_matches = re.findall(r'=\s*(-?\d[\d,]*\.?\d*)\s*(?:cups?|dollars?|\$|%|items?|people|hours?|minutes?|days?|miles?|meters?|kg|lbs?|pounds?|gallons?|liters?|feet|inches|years?|months?|weeks?|seconds?|cents?|\.?\s*$)', clean, re.I)
    if equals_matches:
        return equals_matches[-1].replace(',', '')

    # Last resort: search the FULL text (including <think>) for the patterns above
    if clean != text:
        # Try "the answer is X" in the thinking section
        explicit_full = re.search(r'(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=|would be)\s*\$?\s*(-?\d[\d,]*\.?\d*)', text)
        if explicit_full:
            return explicit_full.group(1).replace(',', '')

        # "So, X cups/dollars" pattern common in Qwen3 reasoning
        so_pattern = re.findall(r'(?i)(?:so|therefore|thus|hence),?\s*(?:the\s+)?(?:answer\s+is\s+)?(?:it\s+(?:is|would be)\s+)?\$?\s*(-?\d[\d,]*\.?\d*)\s*(?:cups?|dollars?|\$|%|items?|people|hours?|minutes?|days?|miles?|meters?|kg|lbs?|pounds?|gallons?|liters?|feet|inches|years?|months?|weeks?|seconds?|cents?)', text)
        if so_pattern:
            return so_pattern[-1].replace(',', '')

    return None


def normalize_number(s: str) -> str:
    """Normalize numbers for comparison."""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return s
