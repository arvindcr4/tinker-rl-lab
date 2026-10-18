#!/usr/bin/env python3
"""Fail-closed task/site contract for the supported WebArena subset.

The input manifest digest must be obtained from an independent reviewed source.
Forum reset, maps and Wikipedia assets are not implemented by this image recipe;
those tasks are rejected before any cloud allocation rather than scored as fails.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

SUPPORTED_SITES = {"shopping", "shopping_admin", "gitlab", "homepage"}


def validate(path, digest, benchmark):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Task/site manifest digest mismatch")
    data = json.loads(raw)
    if data.get("schema") != "webarena-service-contract-v1" or data.get("benchmark") != benchmark:
        raise ValueError("Wrong task/site contract schema or benchmark")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Explicit nonempty task manifest is required")
    ids, sites = [], set()
    for task in tasks:
        identity = task.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", identity) or identity in ids or identity == "all":
            raise ValueError("Task IDs must be explicit, unique and safe")
        required = task.get("required_sites")
        if not isinstance(required, list) or not required:
            raise ValueError("Every task must declare required sites")
        if not set(required).issubset(SUPPORTED_SITES):
            raise ValueError(
                "Unsupported service assets/reset contract; full benchmark deployment is blocked"
            )
        if task.get("requires_judge"):
            raise ValueError("Judge-dependent task contract is not yet supported")
        ids.append(identity)
        sites.update(required)
    return {
        "tasks": ",".join(ids),
        "required_sites": " ".join(sorted(sites)),
        "protocol": "webarena-restricted-services-v2",
        "manifest_sha256": digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("digest")
    parser.add_argument("benchmark")
    parser.add_argument("--field", choices=("tasks", "required_sites"), required=True)
    args = parser.parse_args()
    print(validate(args.manifest, args.digest, args.benchmark)[args.field])


if __name__ == "__main__":
    main()
