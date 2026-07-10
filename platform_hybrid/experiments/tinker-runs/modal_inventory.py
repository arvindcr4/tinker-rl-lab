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


def vol_ls(volume: str, path: str = "") -> list[str]:
    cmd = [str(MODAL), "volume", "ls", volume]
    if path:
        cmd.append(path)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return []
    entries = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("┏", "┡", "└", "│", "Name", "===")):
            # plain output has no table borders when piped; keep simple lines
            if line.startswith("│"):
                continue
        if line and " " not in line:
            entries.append(line)
    return entries


def walk(volume: str) -> list[str]:
    files: list[str] = []
    frontier = [("", 0)]
    seen: set[str] = set()
    while frontier:
        path, depth = frontier.pop()
        for entry in vol_ls(volume, path):
            # `modal volume ls <vol> <dir>` returns entries already prefixed
            # with <dir>/ — don't double the prefix.
            if path and entry.startswith(path + "/"):
                full = entry
            elif path:
                full = f"{path}/{entry}"
            else:
                full = entry
            if full in seen:
                continue
            seen.add(full)
            if "." in full.rsplit("/", 1)[-1]:
                files.append(full)
            elif depth + 1 <= MAX_DEPTH:
                frontier.append((full, depth + 1))
    return sorted(set(files))


def main() -> None:
    if not MODAL.exists():
        sys.exit(f"modal CLI not found at {MODAL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    fetched = 0
    for vol in VOLUMES:
        files = walk(vol)
        print(f"[modal] {vol}: {len(files)} files", flush=True)
        for f in files:
            inventory.append({"volume": vol, "path": f,
                              "ref": f"modal://{vol}/{f}"})
        for f in files:
            if not f.endswith(".json") or fetched >= MAX_JSON_FILES:
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
                print(f"[modal] WARN get failed {vol}/{f}", flush=True)

    with (OUT_DIR / "modal_volumes.jsonl").open("w") as fh:
        for row in inventory:
            fh.write(json.dumps(row) + "\n")

    md = [
        "# Modal volume inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Volumes scanned: {len(VOLUMES)}",
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
