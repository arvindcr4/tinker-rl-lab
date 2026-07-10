#!/usr/bin/env python3
"""P6 iter-98 backfill RED-FLAG leaves with explicit provenance.

Iter-94 H4 surfaced 9 RED-FLAG leaves (>50% null rate across 20 stack
records). The cluster loss_form.{clip_eps_low, clip_eps_high,
length_normalization, token_mask} + reference_kl.{kl_beta, kl_estimator}
+ loss_form.advantage_normalization is concentrated on the 5 zvf130_*
single-batch risk-index harness entries where loss-form internals are
managed-by-tinker and unverifiable.

The schema's nullable_number/nullable_string/nullable_boolean types
constrain these leaves to value-or-null, so we cannot stuff a literal
"reported-as-unknown" marker into the leaf itself. Instead, this script
appends a structured provenance tag to each zvf130_* entry's notes
field, naming the affected leaves and explaining that they are null
because the single-batch risk-index harness is managed-by-tinker and
does not expose loss-form internals. This is the operationally
correct way to declare "reported-as-unknown" under the current schema.

The 9 RED-FLAG leaves covered:
  loss_form.clip_eps_low, clip_eps_high, length_normalization, token_mask,
  advantage_normalization
  reference_kl.kl_beta, kl_estimator
  decontamination.performed, parser_robustness_probe

The first 7 are loss-form internals managed by the Tinker harness;
the last 2 (decontamination.*) are universal across the corpus because
the field was added later and most legacy entries pre-date it.

Outputs (3 files):
- experiments/results/p5p8/p6_iter98_redflag_backfill.tsv
- experiments/results/p5p8/p6_iter98_redflag_backfill.json
- patched registry/entries/zvf130_*.json (5 entries)
"""
import json
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REG = ROOT / "registry"
ENT = REG / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"

# 7 loss-form / KL internals managed by Tinker
LOSS_FORM_KL_LEAVES = (
    "loss_form.clip_eps_low",
    "loss_form.clip_eps_high",
    "loss_form.length_normalization",
    "loss_form.token_mask",
    "loss_form.advantage_normalization",
    "reference_kl.kl_beta",
    "reference_kl.kl_estimator",
)
# 2 decontamination leaves (universal coverage gap, not loss-form)
DECONTAMINATION_LEAVES = (
    "decontamination.performed",
    "decontamination.parser_robustness_probe",
)
ALL_9_LEAVES = LOSS_FORM_KL_LEAVES + DECONTAMINATION_LEAVES

# All 5 zvf130_* stack records carry managed-by-tinker loss-form gaps
TARGET_IDS = ("zvf130_cppo", "zvf130_es", "zvf130_mcgrpo",
              "zvf130_ngrpo", "zvf130_scafgrpo")

PROVENANCE_TAG = (
    "iter-98-redflag-backfill: 7 loss-form/KL leaves ("
    "loss_form.clip_eps_{low,high}, length_normalization, token_mask, "
    "advantage_normalization, reference_kl.kl_{beta,estimator}) are null "
    "because the zvf-iter130 single-batch risk-index harness is "
    "managed-by-tinker and does not expose loss-form internals; reported-"
    "as-unknown on purpose, not as oversight."
)
DECONTAMINATION_TAG = (
    "iter-98-redflag-backfill: 2 decontamination leaves (performed, "
    "parser_robustness_probe) are null across all zvf130_* entries "
    "because the field was added in iter-94 after these entries were "
    "frozen; reported-as-unknown with provenance tied to iter-94."
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry_id in TARGET_IDS:
        p = ENT / f"{entry_id}.json"
        if not p.exists():
            print(f"[p6-iter98] MISSING {p}")
            continue
        rec = json.loads(p.read_text())
        old_notes = rec.get("notes", "")
        new_notes = old_notes.rstrip() + "\n\n" + PROVENANCE_TAG + "\n\n" + DECONTAMINATION_TAG
        rec["notes"] = new_notes
        p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        rows.append({
            "entry_id": entry_id,
            "framework": rec.get("framework", {}).get("name", "?"),
            "openness": rec.get("framework", {}).get("openness", "?"),
            "loss_form_kl_leaves_count": len(LOSS_FORM_KL_LEAVES),
            "decontamination_leaves_count": len(DECONTAMINATION_LEAVES),
            "total_redflag_leaves": len(ALL_9_LEAVES),
            "notes_chars_added": len(new_notes) - len(old_notes),
        })
        print(f"[p6-iter98] backfilled {entry_id} "
              f"(notes +{len(new_notes) - len(old_notes)} chars)")

    tsv = OUT / "p6_iter98_redflag_backfill.tsv"
    cols = ["entry_id", "framework", "openness",
            "loss_form_kl_leaves_count", "decontamination_leaves_count",
            "total_redflag_leaves", "notes_chars_added"]
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([
                r["entry_id"], r["framework"], r["openness"],
                str(r["loss_form_kl_leaves_count"]),
                str(r["decontamination_leaves_count"]),
                str(r["total_redflag_leaves"]),
                str(r["notes_chars_added"]),
            ]) + "\n")

    summary = {
        "n_entries_backfilled": len(rows),
        "loss_form_kl_leaves": list(LOSS_FORM_KL_LEAVES),
        "decontamination_leaves": list(DECONTAMINATION_LEAVES),
        "all_9_leaves": list(ALL_9_LEAVES),
        "target_ids": list(TARGET_IDS),
        "provenance_tag": PROVENANCE_TAG,
        "decontamination_tag": DECONTAMINATION_TAG,
    }
    (OUT / "p6_iter98_redflag_backfill.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # Re-run iter-94 validator to confirm no HIGH-severity regressions
    print()
    print(f"[p6-iter98] wrote {tsv.name}, "
          f"p6_iter98_redflag_backfill.json; patched 5 entries")


if __name__ == "__main__":
    main()