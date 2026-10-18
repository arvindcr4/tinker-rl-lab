"""Exact candidate comparator v2; reference_grader.py remains historical v1.

Whole explicit candidates only; no prose scavenging or binary floating-point
normalization. Disagreement records retain item identity and both policies.
"""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from . import reference_grader as historical

GRADER_VERSION = "exact-reference-v2-candidate"

def normalize_number(value):
    if not isinstance(value, str) or len(value) > 4096: return None
    value = value.strip().replace("−", "-")
    value = re.sub(r"(?<![\d,])([+-]?\d{1,3}(?:,\d{3})+)(?![\d,])", lambda m: m[0].replace(',', ''), value)
    atom = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if not re.fullmatch(rf"{atom}(?:\s*/\s*{atom})?", value): return None
    try:
        parts = [Decimal(x.strip()) for x in value.split('/')]
        if not all(x.is_finite() and abs(x.adjusted()) <= 10000 for x in parts): return None
        result = Fraction(parts[0]) / (Fraction(parts[1]) if len(parts) == 2 else 1)
        return str(result)
    except (ValueError, InvalidOperation, ZeroDivisionError, OverflowError): return None

def extract_answer(text):
    if not isinstance(text, str): return None
    candidates = re.findall(r"(?m)^\s*####\s*([^\n]*)$", text)
    candidates += re.findall(r"\\boxed\s*\{([^{}]*)\}", text)
    candidates += re.findall(r"(?im)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*(.*?)\s*$", text)
    if not candidates: return None
    values = [normalize_number(x) for x in candidates]
    return values[0] if None not in values and len(set(values)) == 1 else None

def compare_items(items):
    seen, disagreements = set(), []
    for row in items:
        identity = row['item_id']
        if not isinstance(identity, str) or not identity or identity in seen: raise ValueError('unique item IDs required')
        seen.add(identity)
        old = historical.extract_answer(row['text'])
        old = historical.normalize_number(old) if old is not None else None
        new = extract_answer(row['text'])
        if old != new: disagreements.append({'item_id':identity, 'historical_value':old, 'exact_value':new})
    return {'schema_version':GRADER_VERSION, 'items_compared':len(seen), 'disagreements':disagreements,
            'scope':'comparator disagreement, not independent task correctness'}
