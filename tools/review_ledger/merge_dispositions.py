"""Resolve add/add merge conflicts in reports/deepreview/{dispositions,ledgers}.

Usage (during a conflicted merge): python3 merge_dispositions.py <path> [<path> ...]

Dispositions (.jsonl): per finding id keep the strongest status; on a tie keep
the incoming branch's entry (fix agents carry verified evidence). Ledgers
(.tsv): union of rows, ours first, header once.
"""

import json
import subprocess
import sys

RANK = ["fixed", "already-fixed", "unsourced-removed", "refuted", "needs-compute", "wontfix-scope"]


def stage(path, n):
    return subprocess.run(
        ["git", "show", f":{n}:{path}"], capture_output=True, text=True, check=True
    ).stdout


def merge_jsonl(ours, theirs):
    rows = {}
    for text, src in ((ours, "ours"), (theirs, "theirs")):
        for line in text.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            prev = rows.get(d["id"])
            if prev is None or RANK.index(d["status"]) <= RANK.index(prev["status"]):
                rows[d["id"]] = d
    return "".join(json.dumps(rows[k]) + "\n" for k in sorted(rows))


def merge_tsv(ours, theirs):
    seen, out = set(), []
    for line in ours.splitlines() + theirs.splitlines():
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out) + "\n"


for path in sys.argv[1:]:
    ours, theirs = stage(path, 2), stage(path, 3)
    merged = merge_jsonl(ours, theirs) if path.endswith(".jsonl") else merge_tsv(ours, theirs)
    open(path, "w").write(merged)
    subprocess.run(["git", "add", path], check=True)
    print("resolved", path)
