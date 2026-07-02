# experiments/_archive/ — INDEX

**Purpose:** Audit trail — result rows removed/superseded during the 2026-04-19 master_results consolidation. Kept for provenance; NOT part of the live corpus. See `../CHANGELOG.md`.

**Key files:**
- `removed_duplicates_2026-04-19.json` — the 17 low-value rows dropped from the v2 master: 10 unnamed Qwen2.5-0.5B "ghost rows" (no experiment_id/trace) from the old collab dump + 7 exact-name duplicates (first occurrence kept).
- `ppo_qwen3-8b_superseded_2026-04-19.json` — the prior `ppo_qwen3-8b` row (peak=1.0, last10=0.35) replaced by the canonical Modal trace (peak=0.75, last10=0.225) after the step-5 reproducibility check.

**Find it fast:**
- why a row vanished from master_results → here + `../CHANGELOG.md`
