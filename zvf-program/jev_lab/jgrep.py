#!/usr/bin/env python3
"""jgrep: semantic filter for files, exit-code gated for CI (pattern U8).

    python3 zvf-program/jev_lab/jgrep.py '<natural-language predicate>' PATH...

Exit codes: 0 = at least one file satisfies the predicate; 1 = none do;
2 = service outage or abstain on every file (fail closed, jgrep contract).

Deterministic prefilter: files whose text shares no token with the predicate
are excluded before any judgment (keeps cost near zero on big trees).
Respects the briefing caveat: files are judged one state at a time, never
batched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from usecases import AbstainPolicy, judged  # noqa: E402


def prefilter(text: str, predicate: str) -> bool:
    pt = {t for t in re.findall(r"[a-z0-9]+", predicate.lower()) if len(t) > 3}
    tt = {t for t in re.findall(r"[a-z0-9]+", text.lower())}
    return not pt or bool(pt & tt)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 64
    predicate, paths = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    hits, abstains = [], 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if not prefilter(text, predicate):
            continue
        r = judged(
            {"file": path.name, "text": text[:12000]},
            f"Predicate: {predicate} — does this file satisfy it?",
            label=f"jgrep_{path.name[:48]}",
        )
        if r.decision == "YES":
            hits.append(str(path))
        elif r.decision in ("ABSTAIN", "FALLBACK_OUTAGE", "FALLBACK_CHOICE"):
            abstains += 1
    for h in hits:
        print(h)
    if hits:
        return 0
    if abstains:
        print(f"jgrep: {abstains} file(s) abstained or service out; failing closed", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
