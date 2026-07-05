#!/usr/bin/env python3
"""P6 iter-94 gap-closure pass on the (dapo, gspo) surrogate entries.

Both `tinker_dapo_qwen3.5-4b_gsm8k.json` and `tinker_gspo_qwen3.5-4b_gsm8k.json`
have variant_deltas_applied entries with status="surrogate" or status="absent":
the tinker-managed stack only enforces a label-flip surrogate (asymmetric clip
for DAPO; sequence-level ratio for GSPO) and does NOT actually enforce the
other components (dynamic_sampling, overlong_reward_shaping, kl_removed for
DAPO; sequence-level clip for GSPO). Therefore the tinker stack is NOT a real
DAPO / GSPO arm and adding a measured row to delta_dapo.json / delta_gspo.json
sourced from those stack records would be misleading — it would advertise a
surrogate as the algorithm.

This script:
  1. Patches `registry/entries/delta_dapo.json` and `delta_gspo.json` to add
     an "intentionally null" note + provenance link to the tinker surrogate
     stack records, converting these entries from MISSING-by-omission
     (validator severity=MEDIUM) to INTENTIONAL-NULL on real evidence
     (validator severity=INFO).
  2. Re-runs the validator.
  3. Prints the updated gap list and a delta-vs-pre summary.

Stdlib only. The intent is operational: every MEDIUM gap on a variant_delta
record must resolve to either an actual measured row OR a documented reason
for null. This pass turns 2 of the 4 remaining MEDIUM gaps into legitimate
INTENTIONAL-NULL via the surrogate provenance path.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REG = ROOT / "registry" / "entries"

# (id, surrogate-stack-record-id, surrogate-note-from-stack-record)
SURROGATE_TARGETS = [
    ("delta_dapo",
     "tinker_dapo_qwen3.5-4b_gsm8k",
     ("The tinker_dapo_qwen3.5-4b_gsm8k stack record is a LABEL-FLIP "
      "SURROGATE only: variant_deltas_applied enumerates clip_higher as "
      "'surrogate' (asymmetric clip enforced via user config, managed loss "
      "internals unverifiable), dynamic_sampling as 'absent' (no group "
      "filter-and-resample hook), token_level_loss as 'unknown' "
      "(managed_by_tinker), overlong_reward_shaping as 'absent' (no length-"
      "aware penalty), and kl_removed as 'unknown' (kl_beta managed). Adding "
      "a measured row sourced from that stack record would advertise the "
      "surrogate as the real DAPO algorithm. Measured block intentionally "
      "null: same-stack DAPO arm (with all five components enforced) does not "
      "exist in the worktree. To be measured once a same-stack arm lands "
      "(criterion: same model + task + sampler + RLHF pipeline with all five "
      "DAPO components actually enforced, not just label-named).")),
    ("delta_gspo",
     "tinker_gspo_qwen3.5-4b_gsm8k",
     ("The tinker_gspo_qwen3.5-4b_gsm8k stack record is a LABEL-FLIP SURROGATE "
      "only: variant_deltas_applied enumerates sequence_level_ratio as "
      "'surrogate' (sequence-level ratio requested via user config; managed "
      "loss internals unverifiable) and sequence_level_clip as 'unknown' "
      "(managed_by_tinker). The Qwen team's full GSPO (Section 3 of arXiv:2507"
      ".18071) requires both sequence-level ratio AND sequence-level clip, "
      "neither of which is verifiably enforced on the managed stack. Adding "
      "a measured row sourced from that stack record would advertise the "
      "surrogate as the real GSPO algorithm. Measured block intentionally "
      "null: same-stack GSPO arm (with both sequence-level components "
      "verifiably enforced) does not exist in the worktree. To be measured "
      "once a same-stack arm lands (criterion: same model + task + sampler + "
      "RLHF pipeline with sequence-level ratio AND sequence-level clip both "
      "actually enforced, not just label-named).")),
]


def patch_entry(entry_id, surrogate_stack_id, new_note_addendum):
    p = REG / f"{entry_id}.json"
    rec = json.loads(p.read_text())
    notes = rec.get("notes") or ""
    addendum = (f" | iter-94 surrogate-marker: {new_note_addendum}")
    rec["notes"] = notes + addendum
    p.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main():
    for entry_id, stack_id, note in SURROGATE_TARGETS:
        rec = patch_entry(entry_id, stack_id, note)
        print(f"[p6-iter94] patched {entry_id}.json with surrogate-marker "
              f"({len(note)}-char note referencing {stack_id})")
    # re-run validator
    import subprocess
    r = subprocess.run(
        ["python3", str(HERE / "p6_iter94_schema_validator.py")],
        capture_output=True, text=True)
    print("\n[p6-iter94] validator re-run stdout:")
    print(r.stdout)


if __name__ == "__main__":
    main()