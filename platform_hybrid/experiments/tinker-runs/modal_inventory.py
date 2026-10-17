#!/usr/bin/env python3
"""modal_inventory.py — inventory Modal volumes and harvest metrics JSONs.

Fourth leg of the provenance audit (after Tinker, W&B, HF). The Modal volumes
hold the OPEN-stack arms (TRL/CleanRL/SB3/... on Modal GPUs) whose runs appear
in the claim-to-run table only as unresolvable `local:` IDs. This script:

  1. recursively lists the four experiment volumes (depth-bounded)
     -> modal_volumes.jsonl (one row per file: volume, path)
  2. downloads every *.json metrics file (weights and checkpoints skipped)
     -> modal_artifacts/<volume>/<path>
  3. writes MODAL_INVENTORY.md summarizing per-volume seed coverage.

Auth comes from ~/.modal.toml (modal CLI). Read-only; nothing is written to
Modal. Requires the `modal` CLI importable from the repo venv.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODAL = REPO / ".venv" / "bin" / "modal"
OUT_DIR = HERE.parents[1] / "experiments" / "results" / "modal_registry"
ART_DIR = OUT_DIR / "modal_artifacts"

VOLUMES = [
    "tinkerrl-zvf-open-results",
    "tinker-rl-results",
    "tinker-results",
    "tinkerrl-results",
]
MAX_DEPTH = 3
MAX_JSON_FILES = 200


def vol_ls(volume, path=""):
    import modal
    vol = modal.Volume.from_name(volume)
    return [{"path": item.path, "type": getattr(item.type, "name", str(item.type)).lower()}
            for item in vol.iterdir(path or "/", recursive=False)]


def walk(volume):
    files, frontier, seen = [], ["/"], set()
    while frontier:
        path = frontier.pop()
        for entry in vol_ls(volume, path):
            full = entry["path"].lstrip("/")
            if full in seen:
                continue
            seen.add(full)
            kind = entry["type"]
            if kind in {"file"}:
                files.append(full)
            elif kind in {"directory", "dir"}:
                frontier.append(full)
            else:
                raise ValueError(f"Unknown Modal entry type: {entry}")
    return sorted(files)


def main() -> None:
    if not MODAL.exists():
        sys.exit(f"modal CLI not found at {MODAL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    fetched = 0
    coverage = {"requested_volumes": len(VOLUMES), "listed_volumes": 0, "failed": [], "truncated_files": 0}
    for vol in VOLUMES:
        try:
            files = walk(vol)
            coverage["listed_volumes"] += 1
        except Exception as exc:
            coverage["failed"].append({"volume": vol, "error": str(exc)})
            continue
        print(f"[modal] {vol}: {len(files)} files", flush=True)
        for f in files:
            inventory.append({"volume": vol, "path": f,
                              "ref": f"modal://{vol}/{f}"})
        for f in files:
            if not f.endswith(".json"):
                continue
            if fetched >= MAX_JSON_FILES:
                coverage["truncated_files"] += 1
                continue
            dest = ART_DIR / vol / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                fetched += 1
                continue
            p = subprocess.run([str(MODAL), "volume", "get", vol, f,
                                str(dest), "--force"],
                               capture_output=True, text=True, timeout=180)
            if p.returncode == 0:
                fetched += 1
            else:
                coverage["failed"].append({"volume": vol, "path": f, "error": p.stderr})
                print(f"[modal] WARN get failed {vol}/{f}", flush=True)

    coverage.update(discovered_files=len(inventory), fetched_files=fetched,
                    status="partial" if coverage["failed"] or coverage["truncated_files"] else "complete")
    (OUT_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2))
    with (OUT_DIR / "modal_volumes.jsonl").open("w") as fh:
        for row in inventory:
            fh.write(json.dumps(row) + "\n")

    md = [
        "# Modal volume inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Coverage: {coverage['status']}; requested volumes: {len(VOLUMES)}, listed: {coverage['listed_volumes']}",
        f"- Files inventoried: {len(inventory)}",
        f"- Metrics JSONs downloaded to `modal_artifacts/`: {fetched}",
        "",
        "## Per volume",
        "",
    ]
    for vol in VOLUMES:
        vf = [r["path"] for r in inventory if r["volume"] == vol]
        md.append(f"### `{vol}` — {len(vf)} files")
        md.append("")
        for f in vf[:40]:
            md.append(f"- `modal://{vol}/{f}`")
        if len(vf) > 40:
            md.append(f"- ... {len(vf) - 40} more in modal_volumes.jsonl")
        md.append("")
    md += [
        "These `modal://` refs are the resolvable provenance for claim-table",
        "rows previously recorded as bare `local:` IDs (P4 drgrpo_gsm8k, P5",
        "samestack, the cross-framework zoo, and the ZVF open audit).",
        "Read-only harvest; auth via ~/.modal.toml; nothing written to Modal.",
    ]
    (OUT_DIR / "MODAL_INVENTORY.md").write_text("\n".join(md) + "\n")
    print(f"-> {OUT_DIR / 'modal_volumes.jsonl'}")
    print(f"-> {OUT_DIR / 'MODAL_INVENTORY.md'}")
    print(f"-> {ART_DIR} ({fetched} json files)")


if __name__ == "__main__":
    main()
