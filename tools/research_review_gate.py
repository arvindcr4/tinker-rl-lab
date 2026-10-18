"""Final delivered-byte review gate v2. Offline; never launches an agent."""

import argparse
import hashlib
import json
from pathlib import Path

CRITICS = {"dialectic", "depth", "width", "instruction"}


def validate(report, manifest, reviews):
    digest = hashlib.sha256(report).hexdigest()
    if (
        manifest.get("schema_version") != 2
        or not manifest.get("run_id")
        or not manifest.get("query_sha256")
        or not manifest.get("corpus_sha256")
    ):
        raise ValueError("missing run identity")
    if manifest.get("final_sha256") != digest or manifest.get("lint_status") != "passed":
        raise ValueError("final hash/lint gate failed")
    required = CRITICS if manifest.get("tier") == "full" else set()
    if manifest.get("tier") not in {"full", "light"}:
        raise ValueError("unknown tier")
    if len(reviews) != len(required) or {r.get("critic") for r in reviews} != required:
        raise ValueError("required critic coverage incomplete")
    for review in reviews:
        if (
            any(
                review.get(k) != manifest.get(k)
                for k in ("run_id", "query_sha256", "corpus_sha256")
            )
            or review.get("draft_sha256") != digest
        ):
            raise ValueError("stale review identity")
        if review.get("status") != "completed" or not isinstance(review.get("findings"), list):
            raise ValueError("required review failed")
        if any(
            not isinstance(f, dict)
            or (f.get("severity") == "critical" and f.get("resolution") != "resolved")
            for f in review["findings"]
        ):
            raise ValueError("unresolved critical finding")
    return {
        "schema_version": 2,
        "review_status": "complete",
        "run_id": manifest["run_id"],
        "final_sha256": digest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.report.read_bytes(),
                json.loads(args.manifest.read_text()),
                [json.loads(p.read_text()) for p in args.reviews],
            )
        )
    )
