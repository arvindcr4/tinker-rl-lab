#!/usr/bin/env python3
"""rectify_correlation_with_hf.py — three-way Tinker x W&B x HF correlation.

The original registry (tinker_wandb_registry.py, 2026-07-10) joined 949 Tinker
runs against 940 W&B runs: 41 exact-ID links of which 8 were model-conflicting.
The Hugging Face audit (2026-07-11) showed HF adapter repos preserve exact
Tinker run IDs and model labels, which can ARBITRATE those conflicts: when the
HF metadata for a Tinker ID agrees with the live Tinker registry, the
disagreeing W&B run is a MISLINK (wrong wandb row recorded the ID), not a
corrupted Tinker record.

This script:
  1. harvests every `arvindcr4/*tinker*` HF model repo (README + *.json files)
     for Tinker run IDs and model labels  -> hf_runs.jsonl
  2. left-joins onto tinker_wandb_correlation.csv adding columns:
     hf_repos, hf_model_label, hf_model_consistent, resolution
  3. writes tinker_wandb_hf_correlation.csv + RECTIFICATION.md summarizing
     every conflict and its arbitration verdict.

Resolution vocabulary (per Tinker run):
  confirmed_wandb_hf   exact W&B link, model-consistent, HF agrees
  confirmed_wandb      exact W&B link, model-consistent, no HF evidence
  hf_arbitrated_mislink exact W&B link conflicts, HF agrees with Tinker
                        => the W&B link is quarantined as a mislink
  conflict_unresolved  exact W&B link conflicts, no (or disagreeing) HF evidence
  hf_only              no exact W&B link, but HF preserves the run
  candidate | unmatched  unchanged from the v1 registry

Requires HF_TOKEN in the environment (read-only listing + raw file fetches).
Never writes to HF. Never stores the token.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG_DIR = HERE.parents[1] / "experiments" / "results" / "tinker_wandb_registry"
CSV_IN = REG_DIR / "tinker_wandb_correlation.csv"
HF_OUT = REG_DIR / "hf_runs.jsonl"
CSV_OUT = REG_DIR / "tinker_wandb_hf_correlation.csv"
MD_OUT = REG_DIR / "RECTIFICATION.md"

RUN_ID_RE = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}(?::train:\d+)?)\b"
)
AUTHOR = "arvindcr4"


def hf_get(path: str, token: str, raw: bool = False):
    url = f"https://huggingface.co{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "tinker-rl-lab-rectify",
        },
    )
    body = urllib.request.urlopen(req, timeout=30).read()
    return body.decode("utf-8", "replace") if raw else json.loads(body)


def normalize_run_id(rid):
    """Never invent a training suffix for an ambiguous UUID."""
    return rid.lower()


def harvest_hf(token):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    records = []
    for model in api.list_models(author=AUTHOR):
        repo = model.id
        try:
            refs = api.list_repo_refs(repo)
            revisions = {r.target_commit for r in [*refs.branches, *refs.tags]}
            for revision in sorted(revisions):
                files = api.list_repo_files(repo, revision=revision)
                targets = [f for f in files if f.lower().endswith((".md", ".json"))]
                for filename in targets:
                    record = {
                        "hf_repo": repo,
                        "revision": revision,
                        "source_file": filename,
                        "tinker_run_ids": [],
                        "hf_model_labels": [],
                        "files_requested": 1,
                        "files_fetched": 0,
                        "files_failed": 0,
                        "files_scanned": 0,
                        "coverage": "partial",
                        "evidence_role": "textual_reference",
                    }
                    try:
                        path = hf_hub_download(repo, filename, revision=revision, token=token)
                        text = Path(path).read_text()
                        record["tinker_run_ids"] = sorted(
                            {
                                normalize_run_id(m[1])
                                for m in RUN_ID_RE.finditer(text)
                                if ":train:" in m[1]
                            }
                        )
                        if filename.endswith(".json"):
                            obj = json.loads(text)
                            if isinstance(obj, dict):
                                record["hf_model_labels"] = sorted(
                                    {
                                        obj[k]
                                        for k in ("model", "base_model", "model_id", "model_name")
                                        if isinstance(obj.get(k), str)
                                    }
                                )
                                record["evidence_role"] = obj.get(
                                    "evidence_role", "textual_reference"
                                )
                        record.update(
                            files_fetched=1, files_scanned=1, coverage="metadata_file_complete"
                        )
                    except Exception as exc:
                        record.update(files_failed=1, error=str(exc))
                    records.append(record)
        except Exception as exc:
            records.append(
                {
                    "hf_repo": repo,
                    "tinker_run_ids": [],
                    "hf_model_labels": [],
                    "files_scanned": 0,
                    "coverage": "partial",
                    "error": str(exc),
                }
            )
    return records


def canon(model):
    return (model or "").strip().lower()


def labels_consistent(a, b):
    return bool(canon(a) and canon(a) == canon(b))


def main() -> None:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        sys.exit("HF_TOKEN not set")
    if not CSV_IN.exists():
        sys.exit(f"missing {CSV_IN}")

    hf_records = harvest_hf(token)
    with HF_OUT.open("w") as f:
        for r in hf_records:
            f.write(json.dumps(r) + "\n")

    # index: tinker run id -> hf evidence
    hf_by_run: dict[str, list[dict]] = defaultdict(list)
    for r in hf_records:
        for rid in r["tinker_run_ids"]:
            hf_by_run[rid].append(r)

    # index: tinker run id -> wandb rows that explicitly reference it
    wandb_rows = [json.loads(l) for l in (REG_DIR / "wandb_runs.jsonl").open()]
    wandb_by_run: dict[str, list[dict]] = defaultdict(list)
    for wr in wandb_rows:
        for rid in wr.get("referenced_tinker_ids") or []:
            wandb_by_run[normalize_run_id(rid)].append(wr)

    rows = list(csv.DictReader(CSV_IN.open()))
    out_rows: list[dict] = []
    stats = defaultdict(int)
    conflict_details: list[dict] = []

    for row in rows:
        rid = row["training_run_id"]
        tinker_model = row["base_model"]

        hf_hits = hf_by_run.get(rid, [])
        hf_repos = ";".join(h["hf_repo"] for h in hf_hits)
        hf_labels = sorted({lb for h in hf_hits for lb in h["hf_model_labels"]})
        hf_consistent = ""
        if hf_hits:
            if not hf_labels:
                hf_consistent = "no_label"
            elif all(labels_consistent(lb, tinker_model) for lb in hf_labels):
                hf_consistent = "consistent"
            else:
                hf_consistent = "conflict"

        wb_hits = wandb_by_run.get(rid, [])
        wb_models = sorted({wr.get("model") or "" for wr in wb_hits} - {""})
        wandb_consistent = ""
        if wb_hits:
            if not wb_models:
                wandb_consistent = "no_label"
            elif all(labels_consistent(m, tinker_model) for m in wb_models):
                wandb_consistent = "consistent"
            else:
                wandb_consistent = "conflict"

        method = row["match_method"]
        if method == "verified_training_owner":
            if wandb_consistent == "consistent":
                resolution = (
                    "confirmed_wandb_hf" if hf_consistent == "consistent" else "confirmed_wandb"
                )
            else:  # wandb conflict
                if hf_consistent == "consistent":
                    # Tinker registry + HF agree; the disagreeing W&B row is a
                    # mislink and is quarantined for claims purposes.
                    resolution = "hf_arbitrated_wandb_mislink"
                elif hf_consistent == "conflict":
                    resolution = "three_way_conflict"
                else:
                    resolution = "conflict_unresolved"
        elif method == "mutual_model_time_candidate":
            resolution = "candidate_hf" if hf_hits else "candidate"
        elif hf_hits:
            resolution = "hf_only"
        else:
            resolution = "unmatched"
        stats[resolution] += 1

        if "conflict" in resolution or "mislink" in resolution:
            conflict_details.append(
                {
                    "run_id": rid,
                    "tinker_model": tinker_model,
                    "wandb_models": wb_models,
                    "hf_labels": hf_labels,
                    "hf_repos": hf_repos,
                    "resolution": resolution,
                    "wandb_url": row.get("wandb_url", ""),
                }
            )

        row.update(
            {
                "wandb_model_labels": ";".join(wb_models),
                "wandb_model_consistent": wandb_consistent,
                "hf_repos": hf_repos,
                "hf_model_label": ";".join(hf_labels),
                "hf_model_consistent": hf_consistent,
                "resolution": resolution,
            }
        )
        out_rows.append(row)

    fieldnames = list(out_rows[0].keys())
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    hf_run_ids = set(hf_by_run)
    csv_ids = {r["training_run_id"] for r in rows}
    md = [
        "# Correlation rectification with Hugging Face evidence",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- HF repos scanned: {len(hf_records)}",
        f"- Unique Tinker run IDs found on HF: {len(hf_run_ids)}",
        f"- ...of which present in the 949-run registry: {len(hf_run_ids & csv_ids)}",
        f"- ...NOT in the registry (orphans): {sorted(hf_run_ids - csv_ids)}",
        "",
        "## Resolution distribution",
        "",
    ]
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        md.append(f"- `{k}`: {v}")
    md += ["", "## Conflicts and arbitrations", ""]
    if conflict_details:
        for c in conflict_details:
            md.append(
                f"- **{c['resolution']}** `{c['run_id']}` "
                f"Tinker=`{c['tinker_model']}` "
                f"W&B={c['wandb_models']} HF={c['hf_labels']} "
                f"({c['wandb_url'] or 'no wandb url'})"
            )
    else:
        md.append("- none")
    md += [
        "",
        "## Files",
        "",
        "- `hf_runs.jsonl`: per-repo harvest (run IDs + model labels).",
        "- `tinker_wandb_hf_correlation.csv`: v1 correlation + 4 HF columns.",
        "",
        "Provenance: read-only HF listing; no token stored; nothing written",
        "to HF. The v1 CSV is left untouched.",
    ]
    MD_OUT.write_text("\n".join(md) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"-> {CSV_OUT}\n-> {MD_OUT}\n-> {HF_OUT}")


if __name__ == "__main__":
    main()
