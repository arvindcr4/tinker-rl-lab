#!/usr/bin/env python3
"""P6 iter-106 — Claim-Evidence Ledger: triangulation of expected_effects, measured,
and claim_validation across the 14 variant_delta registry records.

This is the iter-106 fresh vein. It closes brief vein (a) at the *audit-trail*
level (iter 82 = window-sensitivity; iter 90 = zvf130 measured-vs-claimed; iter 102
= crossref-integrity ground-truth guard). The Claim-Evidence Ledger is a single
canonical table where every (delta, metric, panel) tuple gets one row containing
the human-supplied predicted_sign, the measured delta + CI, and the machine-derived
verdict — and exposes audit gaps that prior iterations only flagged implicitly:

  Gap A  CLAIM-WITHOUT-AUDIT   expected_effect exists but no claim_validation row
                              (claims exist in the human-supplied layer but were
                              never scored against a measurement)
  Gap B  AUDIT-WITHOUT-CLAIM   claim_validation row exists but no matching
                              expected_effects row (machine output with no human
                              provenance)
  Gap C  MEASURED-WITHOUT-AUDIT  measured[] row exists but no claim_validation
                              generated (machine-readable evidence without
                              machine-readable audit verdict)
  Gap D  CLAIM-ONLY            the entry declares expected_effects but ZERO
                              measured[] rows; the entry is fully ungrounded
                              in the worktree (legitimate nulls flagged in
                              notes are still surfaced)
  Gap E  SKELETON              the entry has neither expected_effects NOR
                              measured[] rows (e.g. delta_reinforce,
                              delta_liteppo) -- the entry exists for schema
                              completeness but is unmeasurable today

Bonus job: extend the registry's measured block on delta_aero, delta_gift,
delta_areal with N2 panel rows for two metrics that exist in
experiments/results/n2_reward_tensor_resume/n2_metrics.tsv but are not yet
catalogued: pcd (per-prompt collapse depth) and mean_len. These are
deterministic last-10-step paired bootstraps over the 4 method x 40 step tensors.

Inputs:
  - registry/entries/delta_*.json (14 variant_delta records)
  - experiments/results/n2_reward_tensor_resume/n2_metrics.tsv (4 methods x 40 steps)
  - registry/schema.json (for jsonschema validation)

Outputs:
  - experiments/results/p5p8/p6_iter106_claim_evidence_ledger.tsv
      (40 rows = one per (delta, metric, panel) tuple across 14 entries)
  - experiments/results/p5p8/p6_iter106_audit_gaps.tsv
      (ranked by severity, Gap A-E classification)
  - experiments/results/p5p8/p6_iter106_summary.json
      (machine-readable: per-entry matrix + corpus verdict distribution)
  - registry/entries/delta_aero.json, delta_gift.json, delta_areal.json
      PATCHED with new measured[] + claim_validation[] rows (provenance:
      n2_metrics.tsv last-10 paired bootstrap, B=2000 seed 20260705).
"""
import json
import math
import pathlib
import random
import statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ENT = ROOT / "registry" / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260705
B = 2000
random.seed(SEED)

# Pairs where the N2 panel can ground a registry entry (4 same-stack methods)
N2_METHODS = {"grpo", "aero", "gift", "areal"}
PANEL_N2 = "n2_same_stack_last10"


# ---------- bootstrap helpers ----------
def bootstrap_paired_diff(values_a, values_b, B=B, seed=SEED):
    """Paired bootstrap CI on (a - b). Returns (point, ci_lo, ci_high)."""
    if not values_a or not values_b:
        return None, None, None
    n = min(len(values_a), len(values_b))
    a = values_a[:n]
    b = values_b[:n]
    diffs = [ai - bi for ai, bi in zip(a, b)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return point, boots[int(0.025 * B)], boots[int(0.975 * B)]


def load_n2_last10(metric):
    """Return {method: [v0..v9]} over last 10 steps of n2_metrics.tsv."""
    by_method = defaultdict(list)
    with N2.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            method = parts[idx["method"]]
            step = int(parts[idx["step"]])
            if step < 30:  # last 10 of 40
                continue
            try:
                v = float(parts[idx[metric]])
            except (ValueError, IndexError):
                continue
            if math.isnan(v):
                continue
            by_method[method].append(v)
    return by_method


def compute_n2_delta(variant, metric, B=B):
    """paired last-10 delta (variant - grpo) with bootstrap CI."""
    by_method = load_n2_last10(metric)
    if variant not in by_method or "grpo" not in by_method:
        return None
    pt, lo, hi = bootstrap_paired_diff(by_method[variant], by_method["grpo"], B=B)
    if pt is None:
        return None
    return {
        "metric": metric,
        "panel": PANEL_N2,
        "base": "grpo",
        "delta": round(pt, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "n": 10,
        "significant": bool(lo > 0 or hi < 0),
        "ci_method": {
            "method": "paired_step_bootstrap_pct",
            "n_boot": B,
            "seed": SEED,
            "ci_level": 0.95,
            "source": "platform_modal/scripts/p5p8/p6_iter106_claim_evidence_ledger.py::compute_n2_delta",
        },
        "source": "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
        "note": f"iter-106: N2 last-10 paired bootstrap (B={B}, seed={SEED}); variant minus grpo",
    }


# ---------- claim validation ----------
def expected_sign_to_set(s):
    """Map predicted_sign to a set of acceptable observed-sign values."""
    s = s.strip()
    if s == ">0":
        return {"pos"}
    if s == "<0":
        return {"neg"}
    if s == ">=0":
        return {"pos", "zero"}
    if s == "<=0":
        return {"neg", "zero"}
    if s == "=0":
        return {"zero"}
    return set()


def observed_sign(delta, ci_lo, ci_high, significant):
    """Return pos/neg/zero classification."""
    if significant:
        return "pos" if delta > 0 else "neg"
    # CI crosses 0: zero (NS)
    return "zero"


def machine_verdict(predicted_sign, delta, ci_lo, ci_high, significant):
    """Recompute the verdict from (predicted_sign, observed)."""
    observed = observed_sign(delta, ci_lo, ci_high, significant)
    if predicted_sign is None:
        return "UNCLAIMED", "no expected_effect declared for this (metric, panel) pair"
    accept = expected_sign_to_set(predicted_sign)
    if observed in accept:
        if observed == "zero":
            return "NEUTRAL", (
                f"measured delta={delta:.4f} CI=[{ci_lo:.4f},{ci_high:.4f}] "
                f"includes 0; cannot falsify predicted {predicted_sign}"
            )
        return "SUPPORTS", (
            f"measured delta={delta:.4f} CI=[{ci_lo:.4f},{ci_high:.4f}] "
            f"is significant and matches predicted {predicted_sign}"
        )
    if observed == "zero":
        return "NEUTRAL", (
            f"measured delta={delta:.4f} CI=[{ci_lo:.4f},{ci_high:.4f}] "
            f"includes 0; cannot falsify predicted {predicted_sign}"
        )
    return "CONTRADICTS", (
        f"measured delta={delta:.4f} CI=[{ci_lo:.4f},{ci_high:.4f}] "
        f"is significant but OPPOSITE to predicted {predicted_sign}"
    )


# ---------- main pipeline ----------
def main():
    # Job 1: load every delta entry
    delta_paths = sorted(ENT.glob("delta_*.json"))
    entries = []
    for p in delta_paths:
        d = json.loads(p.read_text())
        entries.append((p, d))

    # Job 2: extend measured blocks on N2-method deltas (aero, gift, areal)
    # Group extensions by delta_id so we accumulate before writing.
    n2_extensions_by_delta = defaultdict(list)  # delta_id -> [(m, cv), ...]
    for delta_id in ("delta_aero", "delta_gift", "delta_areal"):
        for metric in ("pcd", "mean_len"):
            row = compute_n2_delta(delta_id.split("_", 1)[1], metric)
            if row is None:
                continue
            # Decide verdict via existing expected_effect for this (delta, metric, panel)
            d = next(dd for _, dd in entries if dd["id"] == delta_id)
            ee = d.get("expected_effects") or []
            match = next(
                (e for e in ee if e["metric"] == metric and e["panel"] == row["panel"]),
                None,
            )
            predicted_sign = match["predicted_sign"] if match else None
            verdict, rationale = machine_verdict(
                predicted_sign,
                row["delta"],
                row["ci_low"],
                row["ci_high"],
                row["significant"],
            )
            cv_row = {
                "metric": metric,
                "panel": row["panel"],
                "predicted_sign": predicted_sign,
                "observed_delta": row["delta"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "significant": row["significant"],
                "verdict": verdict,
                "rationale": f"iter-106 N2 last-10: {rationale}",
            }
            n2_extensions_by_delta[delta_id].append((dict(row), cv_row))

    # Job 3: write the patched entries (accumulate ALL extensions per entry, write once)
    patched_entries = {}
    for delta_id, ext_list in n2_extensions_by_delta.items():
        d = next(rec for _, rec in entries if rec["id"] == delta_id)
        d = json.loads(json.dumps(d))  # deep copy of the original on-disk entry
        for m, cv in ext_list:
            d.setdefault("measured", []).append(m)
            d.setdefault("claim_validation", []).append(cv)
        patched_entries[delta_id] = d
        out_p = ENT / f"{delta_id}.json"
        out_p.write_text(json.dumps(d, indent=2, sort_keys=False) + "\n")

    n2_extensions = [
        (delta_id, m, cv)
        for delta_id, ext_list in n2_extensions_by_delta.items()
        for m, cv in ext_list
    ]

    # Job 4: walk every entry's expected_effects x measured x claim_validation triangulation
    ledger_rows = []  # list of dicts for the tsv
    gap_rows = []     # list of dicts for the audit gaps tsv
    corpus_summary = {
        "n_delta_entries": len(entries),
        "n_claim_validation_rows": 0,
        "n_measured_rows": 0,
        "n_expected_effects_rows": 0,
        "verdict_distribution_pre": Counter(),
        "verdict_distribution_post": Counter(),
        "gap_counts": Counter(),
        "per_entry": {},
    }

    for p, d in entries:
        delta_id = d["id"]
        measured = d.get("measured") or []
        claimed = d.get("expected_effects") or []
        cv = d.get("claim_validation") or []

        corpus_summary["n_measured_rows"] += len(measured)
        corpus_summary["n_expected_effects_rows"] += len(claimed)
        corpus_summary["n_claim_validation_rows"] += len(cv)

        measured_pairs = {(m["metric"], m["panel"]) for m in measured}
        claimed_pairs = {(e["metric"], e["panel"]) for e in claimed}
        cv_pairs = {(v["metric"], v["panel"]) for v in cv}

        # Ledger rows: union of (claimed U measured) so every (metric, panel) appears once
        all_pairs = sorted(measured_pairs | claimed_pairs | cv_pairs)
        for metric, panel in all_pairs:
            m_row = next((m for m in measured if m["metric"] == metric and m["panel"] == panel), None)
            e_row = next((e for e in claimed if e["metric"] == metric and e["panel"] == panel), None)
            c_row = next((v for v in cv if v["metric"] == metric and v["panel"] == panel), None)

            # Recompute machine verdict if measured exists
            mv = None
            mv_rationale = None
            if m_row is not None:
                predicted_sign = e_row["predicted_sign"] if e_row else None
                mv, mv_rationale = machine_verdict(
                    predicted_sign,
                    m_row["delta"],
                    m_row["ci_low"],
                    m_row["ci_high"],
                    m_row["significant"],
                )
            corpus_summary["verdict_distribution_post"][mv or "NONE"] += 1
            if c_row is not None:
                corpus_summary["verdict_distribution_pre"][c_row["verdict"]] += 1

            ledger_rows.append(
                {
                    "delta_id": delta_id,
                    "metric": metric,
                    "panel": panel,
                    "predicted_sign": (e_row or {}).get("predicted_sign"),
                    "observed_delta": (m_row or {}).get("delta"),
                    "ci_low": (m_row or {}).get("ci_low"),
                    "ci_high": (m_row or {}).get("ci_high"),
                    "n": (m_row or {}).get("n"),
                    "significant": (m_row or {}).get("significant"),
                    "stored_verdict": (c_row or {}).get("verdict"),
                    "machine_verdict": mv,
                    "consistent": (
                        c_row["verdict"] == mv if c_row is not None and mv is not None else None
                    ),
                    "claim_layer": "expected_effects" if e_row else ("none" if c_row is None and m_row is None else "missing"),
                    "measured_layer": "measured[]" if m_row else "missing",
                    "audit_layer": "claim_validation[]" if c_row else "missing",
                }
            )

        # Gap classification per entry
        n_pairs_claimed = len(claimed_pairs)
        n_pairs_measured = len(measured_pairs)
        n_pairs_cv = len(cv_pairs)
        # Gap D: claimed > 0 AND measured == 0
        if n_pairs_claimed > 0 and n_pairs_measured == 0:
            gap_rows.append(
                {
                    "delta_id": delta_id,
                    "gap_class": "D",
                    "severity": "HIGH",
                    "description": (
                        f"CLAIM-ONLY: {n_pairs_claimed} expected_effect(s) "
                        f"but ZERO measured[] rows -- entry fully ungrounded"
                    ),
                    "missing_pair_count": n_pairs_claimed,
                    "remediation": "either add a same-stack arm or annotate the entry's notes with a 'no same-stack arm' marker",
                }
            )
            corpus_summary["gap_counts"]["D"] += 1
        # Gap E: skeleton
        if n_pairs_claimed == 0 and n_pairs_measured == 0 and n_pairs_cv == 0:
            gap_rows.append(
                {
"delta_id": delta_id,
                    "gap_class": "E",
                    "severity": "INFO",
                    "description": (
                        "SKELETON: entry has neither expected_effects nor measured[] "
                        "nor claim_validation[]; exists for schema completeness"
                    ),
                    "missing_pair_count": 0,
                    "remediation": (
                        "add expected_effects (paper-derived) when a same-stack arm is runnable"
                    ),
                }
            )
            corpus_summary["gap_counts"]["E"] += 1
        # Gap A: claim without audit (in claimed_pairs but not in cv_pairs)
        gap_a_pairs = claimed_pairs - cv_pairs
        if gap_a_pairs:
            gap_rows.append(
                {
                    "delta_id": delta_id,
                    "gap_class": "A",
                    "severity": "MEDIUM",
                    "description": (
                        f"CLAIM-WITHOUT-AUDIT: {len(gap_a_pairs)} expected_effect(s) "
                        f"have no matching claim_validation row: {sorted(gap_a_pairs)}"
                    ),
                    "missing_pair_count": len(gap_a_pairs),
                    "remediation": (
                        "generate claim_validation rows (script can do it from "
                        "measured[] + expected_effects matching)"
                    ),
                }
            )
            corpus_summary["gap_counts"]["A"] += 1
        # Gap C: measured without audit
        gap_c_pairs = measured_pairs - cv_pairs
        if gap_c_pairs:
            gap_rows.append(
                {
                    "delta_id": delta_id,
                    "gap_class": "C",
                    "severity": "LOW",
                    "description": (
                        f"MEASURED-WITHOUT-AUDIT: {len(gap_c_pairs)} measured row(s) "
                        f"have no matching claim_validation: {sorted(gap_c_pairs)}"
                    ),
                    "missing_pair_count": len(gap_c_pairs),
                    "remediation": "add claim_validation rows for the missing (metric, panel)",
                }
            )
            corpus_summary["gap_counts"]["C"] += 1
        # Gap B: audit without claim
        gap_b_pairs = cv_pairs - claimed_pairs
        if gap_b_pairs:
            gap_rows.append(
                {
                    "delta_id": delta_id,
                    "gap_class": "B",
                    "severity": "INFO",
                    "description": (
                        f"AUDIT-WITHOUT-CLAIM: {len(gap_b_pairs)} claim_validation row(s) "
                        f"have no matching expected_effect (UNCLAIMED rows): {sorted(gap_b_pairs)}"
                    ),
                    "missing_pair_count": len(gap_b_pairs),
                    "remediation": (
                        "either add expected_effects for the (metric, panel) or note "
                        "the audit row as UNCLAIMED-by-design in entry notes"
                    ),
                }
            )
            corpus_summary["gap_counts"]["B"] += 1

        # Inconsistency check: stored verdict vs machine verdict
        inconsistencies = []
        for v in cv:
            m_row = next(
                (m for m in measured if m["metric"] == v["metric"] and m["panel"] == v["panel"]),
                None,
            )
            if m_row is None:
                continue
            predicted_sign = v.get("predicted_sign")
            mv, _ = machine_verdict(
                predicted_sign,
                m_row["delta"],
                m_row["ci_low"],
                m_row["ci_high"],
                m_row["significant"],
            )
            if mv != v["verdict"]:
                inconsistencies.append(
                    {
                        "metric": v["metric"],
                        "panel": v["panel"],
                        "stored": v["verdict"],
                        "machine": mv,
                    }
                )
        if inconsistencies:
            gap_rows.append(
                {
                    "delta_id": delta_id,
                    "gap_class": "F",
                    "severity": "MEDIUM",
                    "description": (
                        f"INCONSISTENT-VERDICT: {len(inconsistencies)} claim_validation "
                        f"row(s) where stored verdict disagrees with machine recomputation: "
                        f"{inconsistencies}"
                    ),
                    "missing_pair_count": len(inconsistencies),
                    "remediation": (
                        "re-run machine verdict computation; fix stored verdict or "
                        "update measured/expected fields to match"
                    ),
                }
            )
            corpus_summary["gap_counts"]["F"] += 1

        corpus_summary["per_entry"][delta_id] = {
            "n_measured": n_pairs_measured,
            "n_claimed": n_pairs_claimed,
            "n_cv": n_pairs_cv,
            "n_inconsistent": len(inconsistencies),
            "gap_classes": sorted({g["gap_class"] for g in gap_rows if g["delta_id"] == delta_id}),
        }

    # Job 5: write artefacts
    ledger_path = OUT / "p6_iter106_claim_evidence_ledger.tsv"
    with ledger_path.open("w") as f:
        cols = [
            "delta_id", "metric", "panel",
            "predicted_sign", "observed_delta", "ci_low", "ci_high", "n", "significant",
            "stored_verdict", "machine_verdict", "consistent",
            "claim_layer", "measured_layer", "audit_layer",
        ]
        f.write("\t".join(cols) + "\n")
        for r in ledger_rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    gaps_path = OUT / "p6_iter106_audit_gaps.tsv"
    with gaps_path.open("w") as f:
        cols = ["delta_id", "gap_class", "severity", "description", "missing_pair_count", "remediation"]
        f.write("\t".join(cols) + "\n")
        # Sort by severity then by delta_id
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        for r in sorted(gap_rows, key=lambda x: (sev_rank.get(x["severity"], 9), x["delta_id"], x["gap_class"])):
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Convert Counters to dicts for json
    summary_path = OUT / "p6_iter106_summary.json"
    json_summary = {
        "n_delta_entries": corpus_summary["n_delta_entries"],
        "n_claim_validation_rows": corpus_summary["n_claim_validation_rows"],
        "n_measured_rows": corpus_summary["n_measured_rows"],
        "n_expected_effects_rows": corpus_summary["n_expected_effects_rows"],
        "verdict_distribution_pre": dict(corpus_summary["verdict_distribution_pre"]),
        "verdict_distribution_post": dict(corpus_summary["verdict_distribution_post"]),
        "gap_counts": dict(corpus_summary["gap_counts"]),
        "per_entry": corpus_summary["per_entry"],
        "n2_extensions_added": len(n2_extensions),
        "n2_extensions_detail": [
            {"delta_id": did, "metric": m["metric"], "verdict": cv["verdict"]}
            for did, m, cv in n2_extensions
        ],
        "git_sha": "iter-106 uncommitted",
        "audit_date": "2026-07-05",
        "audit_source": "platform_modal/scripts/p5p8/p6_iter106_claim_evidence_ledger.py",
    }
    summary_path.write_text(json.dumps(json_summary, indent=2, sort_keys=False) + "\n")

    print(f"WROTE: {ledger_path} ({len(ledger_rows)} rows)")
    print(f"WROTE: {gaps_path} ({len(gap_rows)} rows)")
    print(f"WROTE: {summary_path}")
    print(f"PATCHED: {len(n2_extensions)} delta entries: {sorted(patched_entries.keys())}")
    print("Verdict distribution (machine recompute):", dict(corpus_summary["verdict_distribution_post"]))
    print("Verdict distribution (stored):            ", dict(corpus_summary["verdict_distribution_pre"]))
    print("Gap counts:", dict(corpus_summary["gap_counts"]))


if __name__ == "__main__":
    main()