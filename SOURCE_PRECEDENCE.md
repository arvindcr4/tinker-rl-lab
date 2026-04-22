# Source Precedence for Discrepant Values

This document explains the source of truth for values that appear in multiple places with different numbers.

---

## Qwen PPO Last-10 Discrepancy

### The Two Values

| Source | Value | Context |
|--------|-------|---------|
| **Source of Truth (ledger)** | **0.225** | `experiments/master_results.json` row: `ppo_qwen3-8b`, field `last10_avg` |
| Alternative (statistics) | 0.350 | `experiments/statistical_analysis.md` |

### Why They Differ

**Ledger value (0.225):**
- Computed as the arithmetic mean of per-step reward values over the last 10 training steps
- Direct extraction from W\&B run logs and local CSV files
- The authoritative number used in the paper's tables

**Statistics value (0.350):**
- Uses a rolling-5-step window average before taking the mean of the last 10
- This smooths transient reward spikes
- Used in statistical summary comparisons but NOT in the paper's tables

### Resolution

The paper uses **0.225** (ledger value) as the authoritative number in:
- Table 7 (PPO vs GRPO comparison)
- The results narrative
- Any direct comparison claims

The **0.350** value is mentioned in the artifact-sensitivity discussion but NOT used as evidence.

### Claim Treatment

**This row is NOT used as directional evidence that GRPO > PPO.**

The paper explicitly states:
> "We therefore withhold a directional Qwen algorithm claim."
> "The appropriate conclusion is identifiability rather than ranking."

The Qwen row, combined with the Llama row (which favors PPO), demonstrates that:
1. Algorithm labels are under-specified treatments
2. Full stack details must be reported
3. No universal PPO/GRPO ranking is possible from this data

---

## Other Discrepancies

### GSM8K Held-Out Accuracy

**Paper claims:**
- Base model: 164/200 = 82.0%
- GRPO mean: 83.3%
- p-value: 0.26

**Verification files:**
- `supporting_data/reports/final/gsm8k_base_control_200.json`
- `supporting_data/reports/final/gsm8k_heldout_seed*.json`

**Status:** Values are consistent. The p-value computation uses Welch's t-test with appropriate degrees of freedom.

---

### Table References

**Anonymous paper (`main_anon.tex`):**
- Table 2 → Referenced in older drafts; current version uses different numbering
- Current tables: Table 7 (PPO vs GRPO), Table 8 (Held-out), Table 9 (ZVF validation)

**If you see "Table 2" in old docs:**
- These refer to draft versions
- The final anonymous paper uses the table numbering shown in `main_anon.tex`

---

## How to Verify

```bash
# Verify Qwen PPO values
python3 scripts/verify_claims_offline.py --claim qwen3_8b_headline_reward

# Verify held-out results
python3 scripts/verify_claims_offline.py --claim gsm8k_heldout_nonsignificant

# Check master_results.json directly
grep -A 20 "ppo_qwen3-8b" experiments/master_results.json
```

---

## Contact

For questions about source precedence or to report new discrepancies, see `REVIEWER_VERIFICATION.md`.
