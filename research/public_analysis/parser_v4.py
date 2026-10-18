"""Candidate wrapper-closure revision. V2/V3 historical outputs remain immutable.

This explicit policy change can reject previously accepted malformed objects;
no preservation of every V3 accepted object and no deployment are claimed.
"""
import re
from .parser_v3 import parse_answer as historical_parse

PARSER_VERSION = 4
PARSER_STATUS = "prospective_candidate_not_deployed"

def parse_answer(text):
    if not isinstance(text, str):
        return {"status": "invalid_text", "source": None, "value": None}
    # Wrapper state is separate from textual heading state. All literal answer
    # openers need an ordered matching closer, with no nesting or stray closer.
    opened = False
    for token in re.finditer(r"<\s*(/?)\s*answer\b[^>]*>", text, re.IGNORECASE):
        if token.group() not in ("<answer>", "</answer>") or bool(token[1]) != opened:
            return {"status": "malformed_answer_tag", "source": None, "value": None}
        opened = not opened
    if opened:
        return {"status": "malformed_answer_tag", "source": None, "value": None}
    return historical_parse(text)
