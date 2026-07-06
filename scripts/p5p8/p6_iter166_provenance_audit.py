#!/usr/bin/env python3
"""
Iter-166 — P6 registry provenance-source audit.

Vein (fresh): each registry entry declares `provenance.source_artifacts` as a
list of free-text strings. The strings blend three archetypes:
  (A) clean relative path (e.g. "experiments/results/foo.tsv") that should
      exist on disk;
  (B) wandb handle (e.g. "W&B <project> / <run>") — opaque URL-like token;
  (C) free-text prose that may embed path tokens (e.g. "...see
      experiments/results/n2_reward_tensor_resume/aero_s0_tensors.jsonl").

Iter-166 classifies each `source_artifacts` element into one of six types,
resolves path tokens, and combines artifact resolvability with citation
completeness into a `provenance_completeness_score` per entry.

Inputs : registry/entries/*.json
Outputs: experiments/results/p5p8/p6_iter166_per_entry.tsv
         experiments/results/p5p8/p6_iter166_per_artifact.tsv
         experiments/results/p5p8/p6_iter166_type_counts.tsv
         experiments/results/p5p8/p6_iter166_summary.json
Stdlib only.
"""
import csv
import json
import os
import re
from collections import defaultdict, Counter

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENT_DIR = os.path.join(WORKTREE, "registry", "entries")
OUT_DIR = os.path.join(WORKTREE, "experiments", "results", "p5p8")

# Regex to extract candidate relative paths from a free-text string.
# Use lookahead so the alternation matches the LONGEST extension first
# (e.g. `jsonl` wins over `json`). The body is greedy and includes `/`, `.`,
# `_`, `-` so multi-segment paths like
# `experiments/results/n2_reward_tensor_resume/aero_s0_tensors.jsonl` match.
# Two flavors are recognized:
#   (i) full-prefix path: experiments/...tsv
#   (ii) bare filename in the worktree root: foo.tsv
PATH_RE = re.compile(
    r"(?P<path>(?:(?:experiments|paper|registry|scripts|docs)/"
    r"[A-Za-z0-9_./-]+|[A-Za-z0-9_][A-Za-z0-9_.-]*)"
    r"(?=\.(?:tsv|jsonl|json|csv|yaml|yml|tex|md|py))\."
    r"(?:tsv|jsonl|json|csv|yaml|yml|tex|md|py))"
)
WANDB_RE = re.compile(r"\bW(?:&|\b)\s*B\b", re.IGNORECASE)


def classify(artifact: str):
    """Return (type, resolved_paths) where type in {PATH_OK, PATH_MISSING,
    WANDB, DESC, EMPTY}; resolved_paths is the list of file paths extracted
    from the prose that exist (or don't)."""
    if not artifact or not artifact.strip():
        return ("EMPTY", [])

    s = artifact.strip()

    # Wandb handle — explicit W&B token at start
    if WANDB_RE.match(s):
        return ("WANDB", [])

    # Pure path?
    if s.startswith(("experiments/", "paper/", "registry/", "scripts/", "docs/")):
        full = os.path.join(WORKTREE, s)
        return ("PATH_OK" if os.path.exists(full) else "PATH_MISSING", [s])

    # Else: prose — extract candidate paths
    matches = PATH_RE.findall(s)
    # Filter out things that look like extensions but aren't real paths —
    # e.g. "G=8" matches as "G=8" only if our pattern accepts `=` which it
    # doesn't. But "see row method=grpo" still extracts "method=grpo" as a
    # candidate. So post-filter: only keep candidates whose joined path
    # exists OR is exactly one of the known top-level tsv/json files.
    candidates = []
    for m in matches:
        if "/" in m:
            # Multi-segment path
            candidates.append(m)
        else:
            # Bare filename — only keep if it looks like a real file
            # (lowercase letters + underscores/digits)
            if re.match(r"^[a-z][a-z0-9_]+\.[a-z]+$", m):
                candidates.append(m)
    if not candidates:
        return ("DESC", [])
    resolved = []
    for m in candidates:
        # Try the literal candidate first; if it doesn't exist, fall back
        # to the canonical worktree location experiments/results/<name> for
        # bare top-level filenames like `zvf_iter130_method_risk.tsv`.
        full = os.path.join(WORKTREE, m)
        if os.path.exists(full):
            resolved.append(m)
            continue
        if "/" not in m:
            canon = os.path.join(WORKTREE, "experiments", "results", m)
            if os.path.exists(canon):
                resolved.append(canon[len(WORKTREE) + 1:])
    if len(resolved) == len(candidates):
        return ("DESC_PATH_OK", resolved)
    elif len(resolved) == 0:
        return ("DESC_PATH_MISSING", [])
    else:
        return ("DESC_PATH_PARTIAL", resolved)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []
    for fname in sorted(os.listdir(ENT_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ENT_DIR, fname)
        with open(path) as f:
            d = json.load(f)
        entries.append(d)

    per_entry = []
    per_artifact = []
    type_counts = Counter()
    record_type_counts = defaultdict(Counter)

    for ent in entries:
        eid = ent["id"]
        record_type = ent.get("record_type", "unknown")
        arts = ent.get("provenance", {}).get("source_artifacts", []) or []
        n_arts = len(arts)
        n_path_ok = n_path_missing = n_wandb = n_desc = n_desc_path_ok = 0
        n_desc_path_missing = n_desc_path_partial = n_empty = 0
        for a in arts:
            t, paths = classify(a)
            type_counts[t] += 1
            record_type_counts[record_type][t] += 1
            if t == "PATH_OK": n_path_ok += 1
            elif t == "PATH_MISSING": n_path_missing += 1
            elif t == "WANDB": n_wandb += 1
            elif t == "DESC": n_desc += 1
            elif t == "DESC_PATH_OK": n_desc_path_ok += 1
            elif t == "DESC_PATH_PARTIAL": n_desc_path_partial += 1
            elif t == "DESC_PATH_MISSING": n_desc_path_missing += 1
            elif t == "EMPTY": n_empty += 1

            per_artifact.append({
                "entry_id": eid, "record_type": record_type,
                "artifact": a, "type": t,
                "n_paths_in_artifact": len(paths),
                "all_paths_resolve": int(len(paths) > 0 and t.endswith("OK")),
                "resolved_paths": "|".join(paths),
            })

        # provenance_completeness_score: fraction of declared artifacts whose
        # type-token resolves to a real file on disk. PATH_OK and DESC_PATH_OK
        # and DESC_PATH_PARTIAL count as resolvable (partial counts 0.5).
        if n_arts == 0:
            score = None
        else:
            score = (
                n_path_ok
                + n_desc_path_ok
                + 0.5 * n_desc_path_partial
            ) / n_arts

        # Detect citation channel (used by variant_delta entries).
        cite = ent.get("citation", {}) or {}
        has_cite_bibkey = bool((cite.get("bibkey") or "").strip())
        has_cite_arxiv = bool((cite.get("arxiv") or "").strip())
        has_cite_title = bool((cite.get("title") or "").strip())
        citation_score = (
            (1 if has_cite_bibkey else 0)
            + (1 if has_cite_arxiv else 0)
            + (1 if has_cite_title else 0)
        ) / 3.0  # 0.0 - 1.0

        # Combined provenance_score: artifact-resolvability is primary
        # (weight 0.7), citation completeness is secondary (weight 0.3).
        # For entries with no source_artifacts, citation_score is the
        # primary channel and gets full weight.
        if n_arts == 0:
            combined_score = citation_score
        else:
            combined_score = 0.7 * score + 0.3 * citation_score

        per_entry.append({
            "entry_id": eid, "record_type": record_type,
            "n_artifacts": n_arts,
            "n_path_ok": n_path_ok, "n_path_missing": n_path_missing,
            "n_desc_path_ok": n_desc_path_ok,
            "n_desc_path_partial": n_desc_path_partial,
            "n_desc_path_missing": n_desc_path_missing,
            "n_desc": n_desc, "n_wandb": n_wandb, "n_empty": n_empty,
            "artifact_resolvability_score": (
                None if score is None else round(score, 4)),
            "citation_score": round(citation_score, 4),
            "has_cite_bibkey": int(has_cite_bibkey),
            "has_cite_arxiv": int(has_cite_arxiv),
            "provenance_completeness_score": round(combined_score, 4),
        })

    # Write per-entry
    pe_cols = [
        "entry_id", "record_type", "n_artifacts", "n_path_ok", "n_path_missing",
        "n_desc_path_ok", "n_desc_path_partial", "n_desc_path_missing",
        "n_desc", "n_wandb", "n_empty", "artifact_resolvability_score",
        "citation_score", "has_cite_bibkey", "has_cite_arxiv",
        "provenance_completeness_score",
    ]
    with open(os.path.join(OUT_DIR, "p6_iter166_per_entry.tsv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=pe_cols, delimiter="\t")
        w.writeheader()
        for row in per_entry:
            w.writerow({k: row.get(k, "") for k in pe_cols})

    # Write per-artifact
    pa_cols = [
        "entry_id", "record_type", "artifact", "type",
        "n_paths_in_artifact", "all_paths_resolve", "resolved_paths",
    ]
    with open(os.path.join(OUT_DIR, "p6_iter166_per_artifact.tsv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=pa_cols, delimiter="\t")
        w.writeheader()
        for row in per_artifact:
            w.writerow({k: row.get(k, "") for k in pa_cols})

    # Type counts overall + per record_type
    tc_rows = [{"scope": "ALL", "record_type": "*", "type": t, "n": n}
               for t, n in sorted(type_counts.items())]
    for rt, ctr in sorted(record_type_counts.items()):
        for t, n in sorted(ctr.items()):
            tc_rows.append({"scope": "BY_RECORD_TYPE", "record_type": rt,
                            "type": t, "n": n})
    with open(os.path.join(OUT_DIR, "p6_iter166_type_counts.tsv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "record_type", "type", "n"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(tc_rows)

    # Headline verdicts (H1..H5)
    n_entries = len(entries)
    n_entries_with_artifacts = sum(1 for r in per_entry if r["n_artifacts"] > 0)
    n_wandb = sum(r["n_wandb"] for r in per_entry)
    n_desc_path_missing = sum(r["n_desc_path_missing"] for r in per_entry)
    n_path_missing = sum(r["n_path_missing"] for r in per_entry)

    summary = {
        "n_entries": n_entries,
        "n_entries_with_artifacts": n_entries_with_artifacts,
        "n_entries_with_zero_artifact_list": n_entries - n_entries_with_artifacts,
        "type_counts_overall": dict(type_counts),
        "n_entries_with_combined_score_1.0": sum(
            1 for r in per_entry if r["provenance_completeness_score"] == 1.0),
        "n_entries_with_combined_score_0.0": sum(
            1 for r in per_entry if r["provenance_completeness_score"] == 0.0),
        "mean_combined_provenance_completeness_score": round(
            sum(r["provenance_completeness_score"] for r in per_entry)
            / max(1, len(per_entry)), 4),
        "mean_citation_score": round(
            sum(r["citation_score"] for r in per_entry)
            / max(1, len(per_entry)), 4),
        "n_wandb_handles": n_wandb,
        "n_path_missing_clean_paths": n_path_missing,
        "n_desc_path_missing_prose": n_desc_path_missing,
        "headline": {
            "H1_pct_entries_with_at_least_one_artifact": round(
                100 * n_entries_with_artifacts / n_entries, 2),
            "H2_pct_entries_with_full_combined_score_1.0": round(
                100 * sum(1 for r in per_entry
                          if r["provenance_completeness_score"] == 1.0)
                / n_entries, 2),
            "H3_pct_wandb_handles_among_declared": round(
                100 * n_wandb / max(1, sum(type_counts.values())), 2),
            "H4_mean_combined_score_all_entries": round(
                sum(r["provenance_completeness_score"] for r in per_entry)
                / max(1, len(per_entry)), 4),
        },
    }
    with open(os.path.join(OUT_DIR, "p6_iter166_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Iter-166 provenance audit complete.")
    print(f"  entries: {n_entries}")
    print(f"  entries with ≥1 source_artifacts: {n_entries_with_artifacts}")
    print(f"  type counts: {dict(type_counts)}")
    mean_score = (sum(r["provenance_completeness_score"] for r in per_entry)
                  / n_entries)
    print(f"  mean combined provenance_completeness_score: {mean_score:.4f}")


if __name__ == "__main__":
    main()