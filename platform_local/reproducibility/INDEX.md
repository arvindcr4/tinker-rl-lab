# reproducibility/ — INDEX

**Purpose:** Cheap reviewer-facing checks that verify the paper's headline claims against stored experiment traces (no GPU retrain needed).

**Key files:**
- `check_qwen3_8b_claim.py` — recomputes last-10 GRPO/PPO reward on Qwen3-8B from `experiments/master_results.json`; exits 0 iff within ±2 pp of the published 34.4% / 22.5% claims.
- `qwen3_8b_claim_check.json` — recorded output/result of that check.
- `smoke_test_2026-04-19.log` — captured log from a reviewer smoke-test run.

**Find it fast:**
- to verify the headline Qwen3-8B numbers → `python reproducibility/check_qwen3_8b_claim.py`
