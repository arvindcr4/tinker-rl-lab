#!/usr/bin/env python3
"""wandb_backfill.py — ensure every experiment artifact is logged in W&B.

Scans two artifact sources that currently have no W&B identity and creates
one W&B run per result file, tagged `backfill`, with the source path (local
or modal://) recorded in the run notes and the raw JSON attached as an
artifact. IDEMPOTENT: existing run names in the target project are skipped,
so re-running after new results land keeps W&B complete.

Sources -> projects:
  zvf-program/experiments-next/results/*.json      -> zvf-experiments-next
  .../results/modal_registry/modal_artifacts/**/*.json -> modal-open-stack
  tinker-runs/results/*.json (completed)           -> zvf-training

Backfilled runs carry `backfill: true` in config; W&B created-at reflects
backfill time, NOT original run time — original timing lives in the JSON.
Auth: ambient (netrc / WANDB_API_KEY). Nothing here trains or samples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ENTITY = "arvindcr4-pes-university"

EXPNEXT = REPO / "zvf-program" / "experiments-next" / "results"
MODAL_ART = (HERE.parents[1] / "experiments" / "results" / "modal_registry"
             / "modal_artifacts")
TRAIN_RESULTS = HERE / "results"

MAX_CONFIG_STR = 200


def flat_scalars(obj: dict, prefix: str = "", depth: int = 0) -> dict:
    """Scalar (and short-string) fields for config/summary; skips lists/logs."""
    out = {}
    if depth > 2 or not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, (int, float, bool)):
            out[key] = v
        elif isinstance(v, str) and len(v) <= MAX_CONFIG_STR:
            out[key] = v
        elif isinstance(v, dict):
            out.update(flat_scalars(v, key + ".", depth + 1))
        elif isinstance(v, list):
            out[key + ".len"] = len(v)
    return out


def existing_names(api: wandb.Api, project: str) -> set[str]:
    try:
        return {r.name for r in api.runs(f"{ENTITY}/{project}", per_page=500)}
    except Exception:
        return set()


def log_file(project: str, name: str, path: Path, tags: list[str],
             notes: str, existing: set[str]) -> bool:
    if name in existing:
        print(f"[skip] {project}/{name} (exists)", flush=True)
        return False
    try:
        data = json.loads(path.read_text())
    except Exception as ex:
        print(f"[warn] unreadable {path}: {ex}", flush=True)
        return False
    cfg = flat_scalars(data if isinstance(data, dict) else {})
    cfg["backfill"] = True
    cfg["source_path"] = str(path.relative_to(REPO))
    run = wandb.init(entity=ENTITY, project=project, name=name, tags=tags,
                     notes=notes, config=cfg, reinit=True,
                     settings=wandb.Settings(silent=True))
    numeric = {k: v for k, v in cfg.items() if isinstance(v, (int, float))
               and not isinstance(v, bool)}
    if numeric:
        run.summary.update(numeric)
    art = wandb.Artifact(name=f"{name}-json", type="result-json")
    art.add_file(str(path))
    run.log_artifact(art)
    run.finish()
    print(f"[logged] {project}/{name}", flush=True)
    return True


def main() -> None:
    api = wandb.Api()
    total = 0

    # 1. experiments-next results
    proj = "zvf-experiments-next"
    seen = existing_names(api, proj)
    for p in sorted(EXPNEXT.glob("*.json")):
        data = json.loads(p.read_text())
        if data.get("status") not in (None, "complete"):
            print(f"[skip] {p.name} status={data.get('status')!r}", flush=True)
            continue
        kind = data.get("kind", p.stem.split("_")[0])
        name = p.stem
        notes = (f"Backfilled from {p.relative_to(REPO)}; kind={kind}; "
                 "Tinker sampling experiments (experiments-next suite)")
        total += log_file(proj, name, p, ["backfill", "tinker", kind], notes,
                          seen)

    # 2. modal artifacts
    proj = "modal-open-stack"
    seen = existing_names(api, proj)
    if MODAL_ART.exists():
        for p in sorted(MODAL_ART.rglob("*.json")):
            rel = p.relative_to(MODAL_ART)
            volume = rel.parts[0]
            name = "-".join(rel.parts[1:]).removesuffix(".json") or p.stem
            name = f"{volume}--{name}"[:120]
            modal_ref = f"modal://{volume}/{'/'.join(rel.parts[1:])}"
            notes = f"Backfilled from {modal_ref} (open-stack Modal artifact)"
            total += log_file(proj, name, p,
                              ["backfill", "modal", volume], notes, seen)

    # 3. Tinker training runs (live_zvf_probe outputs; no native W&B logging)
    proj = "zvf-training"
    seen = existing_names(api, proj)
    if TRAIN_RESULTS.exists():
        for p in sorted(TRAIN_RESULTS.glob("*.json")):
            if ".orig" in p.name:
                continue  # preserved originals; the canonical tag is logged
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue  # legacy batch files are top-level lists; skip
            if data.get("status") not in ("completed", "complete"):
                print(f"[skip] {p.name} status={data.get('status')!r}",
                      flush=True)
                continue
            name = p.stem
            tags = ["backfill", "tinker-train"]
            if "drgrpo" in name:
                tags.append("drgrpo")
            elif "grpo" in name:
                tags.append("grpo")
            notes = (f"Backfilled from {p.relative_to(REPO)}; "
                     "live_zvf_probe training run (P4/E-R2b)")
            total += log_file(proj, name, p, tags, notes, seen)

    print(f"done: {total} new runs logged", flush=True)


if __name__ == "__main__":
    main()
