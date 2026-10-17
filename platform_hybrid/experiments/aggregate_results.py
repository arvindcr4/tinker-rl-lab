"""Aggregate explicitly selected empirical run receipts; no legacy inference."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import require_empirical, EvidenceUnavailable, PROTOCOL_VERSION

import argparse
import json
from pathlib import Path

def aggregate(receipts):
    rows = [require_empirical(dict(record)) for record in receipts]
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise EvidenceUnavailable("Duplicate run IDs in selected invocation")
    return {"protocol_version": PROTOCOL_VERSION, "experiments": rows}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.receipts]
    args.out.write_text(json.dumps(aggregate(rows), indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
