#!/usr/bin/env python3
"""P6 iter-46: Measured-effect vs claimed-effect validation of the registry.

Each `delta_*.json` carries:
  - `deltas[].change` -- a free-text description of WHAT the variant does;
  - `measured[]`      -- a list of (metric, panel, base, delta, ci, ...)
                        grounded measurement rows from
                        `platform_modal/scripts/p5p8/p6_measured_delta_block.py`.

The audit gap that this iter closes: the registry records both
"what the variant claims to change" and "what we measured", but
nothing in the worktree had been linking them. A drift between the
two is exactly the kind of paper-reviewer-fatal finding that a
benchmark paper must pre-empt: if DAPO claims to RAISE clip_eps_high
and the measured effect is a LOWER mean reward on the same stack, the
reader deserves to see that surfaced.

This script:

  1. Optionally extends each `delta_*.json` with a new OPTIONAL
     `expected_effects` block: a forward-referenced per-(metric, panel)
     tuple of {predicted_sign, rationale} that the human-readable
     change text implies. The block is additive; entries that don't
     have it are still audited with verdict = "UNCLAIMED".

  2. For every `measured[]` row, looks up the matching `expected_effect`
     (or marks UNCLAIMED), and classifies the (predicted, observed)
     pair into one of four verdicts:

       SUPPORTS   -- measured CI excludes 0 AND observed sign matches
                    predicted sign (significant in the right direction).
       NEUTRAL    -- measured CI includes 0 (or sign matches but not
                    significant) -- claim cannot be falsified yet.
       CONTRADICTS-- measured CI excludes 0 AND observed sign OPPOSITE
                    predicted sign (significant against the claim).
       UNCLAIMED  -- no `expected_effect` declared for this (metric, panel).

  3. Writes a `claim_validation` block onto each entry, validates the
     whole registry against the schema (must still pass), and emits a
     TSV + summary JSON with verdict counts per delta.

  4. Print headline: % SUPPORTS / NEUTRAL / CONTRADICTS / UNCLAIMED
     across the 22 measured rows, and per-delta counts.

Stdlib + jsonschema only.
"""
from __future__ import annotations

import csv
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRIES = ROOT / "registry" / "entries"
SCHEMA = ROOT / "registry" / "schema.json"
OUT = ROOT / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Expected-effect seeds: human-readable claim → predicted sign per
# (metric, panel) tuple. Seeding only the directional claims where the
# change text is unambiguous. Entries not in this map are UNCLAIMED for
# that (metric, panel).
# ---------------------------------------------------------------------------
# Each entry: delta_id -> list of (metric, panel, predicted_sign, rationale)
EXPECTED_EFFECTS = {
    "delta_aero": [
        ("zvf", "n2_same_stack_last10", "<0",
         "AERO inflates the effective group size via off-policy rollouts; "
         "fewer all-same groups should lower ZVF on the same stack."),
        ("reward_mean", "n2_same_stack_last10", ">=0",
         "AERO's off-policy rollouts reuse the same stack's signal -- "
         "no reason to expect a reward loss; we expect at least parity."),
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "Same logic on the 5-seed risk panel: lower risk of zero-variance groups."),
    ],
    "delta_gift": [
        ("zvf", "n2_same_stack_last10", ">0",
         "GIFT subtracts a gamma-style likelihood baseline -- a small "
         "increase in zero-variance fraction is plausible because the "
         "gamma prior regularises the group mean toward a constant."),
        ("reward_mean", "n2_same_stack_last10", ">=0",
         "GIFT's documented loss shift of +16,722 absolute loss should "
         "be at least reward-neutral on the same stack."),
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "On the 5-seed risk panel, GIFT's gamma prior reduces risk "
         "of zero-variance groups (anti-starvation)."),
    ],
    "delta_areal": [
        ("zvf", "n2_same_stack_last10", "<0",
         "AREAL's decoupled rollout budget acts as a larger effective "
         "G on this single-batch run; should lower ZVF."),
        ("reward_mean", "n2_same_stack_last10", ">=0",
         "AREAL's autoscaler is statically set to 8 in our N2 run; "
         "should be at least reward-neutral on the same stack."),
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "Decoupled rollout should reduce risk of zero-variance groups."),
    ],
    # Methods measured only on the zvf130 panel.
    "delta_cppo": [
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "CPPO's continuity penalty reduces abrupt policy updates; "
         "risk of ZVF collapse should be lower than vanilla GRPO."),
    ],
    "delta_ngrpo": [
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "NGRPO's noise-injection acts as exploration; should lower "
         "the risk of zero-variance starvation."),
    ],
    "delta_mcgrpo": [
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "MCGRPO's Monte-Carlo advantage uses a per-prompt baseline; "
         "should reduce ZVF risk vs vanilla GRPO."),
    ],
    "delta_es": [
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "Evolution Strategies as a policy-gradient surrogate; "
         "inherits lower ZVF risk via population-based evaluation."),
    ],
    "delta_scafgrpo": [
        ("zvf_risk_mean", "zvf130_5seed", "<0",
         "Scaffolded GRPO refactors the advantage; should be at least "
         "as ZVF-safe as vanilla GRPO."),
    ],
}


def sign_match(observed: float, predicted: str) -> bool:
    """True iff the sign of `observed` matches the operator in `predicted`."""
    if predicted == ">0":
        return observed > 0
    if predicted == "<0":
        return observed < 0
    if predicted == ">=0":
        return observed >= 0
    if predicted == "<=0":
        return observed <= 0
    if predicted in ("==0", "=0"):
        return observed == 0
    raise ValueError(f"unknown predicted_sign: {predicted!r}")


def classify(observed: float, ci_low: float, ci_high: float,
             predicted: str | None) -> tuple[str, str]:
    """Return (verdict, rationale)."""
    if predicted is None:
        return ("UNCLAIMED",
                "no expected_effect declared for this (metric, panel) pair")
    sign_ok = sign_match(observed, predicted)
    sig = (ci_low > 0) or (ci_high < 0)
    if not sig:
        return ("NEUTRAL",
                f"measured delta={observed:+.4f} CI=[{ci_low:+.4f},{ci_high:+.4f}] "
                f"includes 0; cannot falsify predicted {predicted}")
    if sign_ok:
        return ("SUPPORTS",
                f"measured delta={observed:+.4f} CI=[{ci_low:+.4f},{ci_high:+.4f}] "
                f"is significant and matches predicted {predicted}")
    return ("CONTRADICTS",
            f"measured delta={observed:+.4f} CI=[{ci_low:+.4f},{ci_high:+.4f}] "
            f"is significant but OPPOSITE to predicted {predicted}")


def main():
    # Stage 1: optionally extend entries with expected_effects (additive).
    n_extended = 0
    for delta_id, eff_list in EXPECTED_EFFECTS.items():
        path = ENTRIES / f"{delta_id}.json"
        if not path.exists():
            print(f"  [warn] {path.name} missing; skipping seed")
            continue
        rec = json.load(open(path))
        if "expected_effects" not in rec:
            rec["expected_effects"] = []
        # only add if not already present
        have_keys = {(e["metric"], e["panel"]) for e in rec["expected_effects"]}
        added = 0
        for metric, panel, pred, rationale in eff_list:
            if (metric, panel) in have_keys:
                continue
            rec["expected_effects"].append({
                "metric": metric, "panel": panel,
                "predicted_sign": pred, "rationale": rationale,
            })
            added += 1
        if added:
            path.write_text(json.dumps(rec, indent=2) + "\n")
            n_extended += 1
    print(f"[seed] extended {n_extended} entries with expected_effects")

    # Stage 2: validate that schema still passes after extension.
    import jsonschema
    schema = json.load(open(SCHEMA))
    V = jsonschema.Draft202012Validator(schema)
    # Stage 3: audit every delta_*.json.
    rows = []
    counts = Counter()
    per_delta_counts = defaultdict(Counter)
    n_skipped_unmeasured = 0
    for p in sorted(ENTRIES.glob("delta_*.json")):
        rec = json.load(open(p))
        delta_id = rec["id"]
        expected = {(e["metric"], e["panel"]): e
                    for e in rec.get("expected_effects", [])}
        measured = rec.get("measured", []) or []
        cv_list = []
        for m in measured:
            key = (m["metric"], m["panel"])
            exp = expected.get(key)
            pred = exp["predicted_sign"] if exp else None
            verdict, rationale = classify(m["delta"], m["ci_low"], m["ci_high"], pred)
            counts[verdict] += 1
            per_delta_counts[delta_id][verdict] += 1
            row = {
                "delta_id": delta_id,
                "metric": m["metric"],
                "panel": m["panel"],
                "predicted_sign": pred if pred is not None else "",
                "observed_delta": m["delta"],
                "ci_low": m["ci_low"],
                "ci_high": m["ci_high"],
                "significant": m["significant"],
                "verdict": verdict,
                "rationale": rationale,
            }
            rows.append(row)
            cv_list.append({
                "metric": m["metric"], "panel": m["panel"],
                "predicted_sign": pred,
                "observed_delta": m["delta"],
                "ci_low": m["ci_low"], "ci_high": m["ci_high"],
                "significant": m["significant"],
                "verdict": verdict,
                "rationale": rationale,
            })
        if measured and not cv_list:
            n_skipped_unmeasured += 1
        # Stage 4: write claim_validation block back if anything to record.
        if cv_list:
            rec["claim_validation"] = cv_list
            errs = list(V.iter_errors(rec))
            assert not errs, (delta_id, errs[0].message)
            p.write_text(json.dumps(rec, indent=2) + "\n")

    # Stage 5: full-registry re-validation.
    ok = bad = 0
    for p in sorted(ENTRIES.glob("*.json")):
        if list(V.iter_errors(json.load(open(p)))):
            bad += 1
        else:
            ok += 1

    # Stage 6: write outputs.
    tsv_path = OUT / "p6_measured_vs_claim.tsv"
    with open(tsv_path, "w", newline="") as fh:
        cols = ["delta_id", "metric", "panel", "predicted_sign", "observed_delta",
                "ci_low", "ci_high", "significant", "verdict", "rationale"]
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {tsv_path}")

    summ = {
        "iter": 46,
        "pillar": "P6",
        "vein": "(a) measured-vs-claimed validation",
        "n_measured_rows": len(rows),
        "verdict_counts": dict(counts),
        "verdict_pct": {k: round(100 * v / max(1, len(rows)), 1) for k, v in counts.items()},
        "per_delta_counts": {k: dict(v) for k, v in per_delta_counts.items()},
        "registry_validate": {"pass": ok, "fail": bad, "total": ok + bad},
        "extended_entries": n_extended,
    }
    summary_path = OUT / "p6_measured_vs_claim_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summ, fh, indent=2, sort_keys=True)
    print(f"[ok] wrote {summary_path}")
    print()
    print(f"=== Iter 46 P6 — measured-vs-claim validation ===")
    print(f"Total measured rows audited: {len(rows)}")
    print(f"  SUPPORTS    : {counts.get('SUPPORTS', 0)}")
    print(f"  NEUTRAL     : {counts.get('NEUTRAL', 0)}")
    print(f"  CONTRADICTS : {counts.get('CONTRADICTS', 0)}")
    print(f"  UNCLAIMED   : {counts.get('UNCLAIMED', 0)}")
    sig_share = (counts.get('SUPPORTS', 0) + counts.get('CONTRADICTS', 0)) / max(1, len(rows))
    print(f"  significant-share (SUPPORTS+CONTRADICTS) / total = {sig_share:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
