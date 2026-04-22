# Reviewer Verification Guide

This document maps every major claim in the paper to its evidence files,
one-line verification commands, and expected outputs. Run these commands
from the extracted bundle root to verify the claims independently.

## Quick Start (5 minutes)

```bash
# Verify bundle integrity
sha256sum -c SHA256SUMS.txt

# Run all offline claim verification
python3 scripts/verify_claims_offline.py
```

Expected output:
```
PASS checksum manifest
PASS qwen3_8b_headline_reward
PASS gsm8k_heldout_nonsignificant  
PASS zvf_triage_claim
PASS framework_stack_sensitivity_claim
PASS figure_inputs_present
PASS table_numbers_match_sources
```

---

## Claim Verification Table

| Claim | Paper Location | Evidence Files | Command | Expected Output |
|-------|---------------|---------------|---------|----------------|
| **C1: Qwen3-8B GRPO last-10 = 34.4%, PPO = 22.5%** | Table 7, §Results | `experiments/master_results.json` | `python3 scripts/verify_claims_offline.py --claim qwen3_8b` | `GRPO last10=0.3438, PPO last10=0.225` |
| **C2: Held-out GSM8K 82.0% → 83.3%, p = 0.26** | Abstract, §Held-out | `gsm8k_base_control_200.json`, `gsm8k_heldout_seed*.json` | `python3 scripts/verify_claims_offline.py --claim gsm8k_heldout` | `base=164/200=82.0%, GRPO mean=83.3%, p≈0.26` |
| **C3: ZVF triage catches 2 collapsed runs, 0 false positives** | Claim 1, §ZVF | `zvf_predictive_validation_results.json`, `zvf_predictive_validation_runs.csv` | `python3 scripts/verify_claims_offline.py --claim zvf_triage` | `TP=2, FP=0, precision=1.0` |
| **C4: PPO/GRPO ordering reverses across model families** | Claim 2, §PPO-GRPO | `master_results.json`, `framework_comparison.json` | `python3 scripts/verify_claims_offline.py --claim stack_sensitivity` | Qwen favors GRPO, Llama favors PPO |
| **C5: Tool-use rewards are schema/JSON, not execution** | Claim 3, §Limitations | Tool-use result JSONs, reward parser source | `python3 scripts/verify_claims_offline.py --claim tool_proxy` | Confirms JSON/schema-only scoring |
| **C6: HumanEval/MATH are reward-environment probes** | Abstract, §Limitations | Reward harness source, result rows | `python3 scripts/verify_claims_offline.py --claim proxy_harnesses` | Confirms test-split usage |
| **C7: ZVF correlation r = -0.769, p = 0.0008** | §ZVF Analysis | `experiments/master_results.json` | `python3 scripts/verify_claims_offline.py --claim zvf_correlation` | r=-0.769, p<0.001 |
| **C8: Exponential saturation mean R² = 0.210** | §Scaling | `experiments/master_results.json` | `python3 scripts/verify_claims_offline.py --claim saturation_fit` | Mean R²≈0.21 |
| **C9: TRL GRPO 5-seed mean = 73.4%, SD = 7.0%** | §TRL Baseline | `experiments/master_results.json` | `python3 scripts/verify_claims_offline.py --claim trl_baseline` | mean≈0.734, sd≈0.07 |
| **C10: Group size G=8 peak at 84.4% last-10** | §Group Size | `experiments/master_results.json` | `python3 scripts/verify_claims_offline.py --claim group_size` | G=8 last10≈0.844 |

---

## Source Precedence for Discrepant Values

### Qwen PPO Last-10 Discrepancy

The Qwen PPO row has two reported values:

| Source | Value | File |
|--------|-------|------|
| **Source of Truth (ledger)** | **0.225** | `experiments/master_results.json` row: `ppo_qwen3-8b` |
| Alternative (statistics) | 0.350 | `experiments/statistical_analysis.md` |

**Explanation:** The ledger value (0.225) is the per-step reward trace mean over the last 10 steps. The statistics summary value (0.350) uses a different aggregation method (rolling window average before mean). The paper uses the ledger value as the authoritative number.

**Claim Treatment:** This row is NOT used as directional evidence that GRPO > PPO. The paper states this is artifact-sensitive evidence for stack-identifiability, not an algorithm ranking.

### Table References

The anonymous paper (main_anon.tex) uses:
- Table 7: PPO vs GRPO comparison
- Table 8: Held-out GSM8K evaluation
- Table 9: ZVF validation results

---

## Figure Provenance

See `FIGURE_PROVENANCE.md` for detailed figure-by-figure verification.

Quick check:
```bash
python3 scripts/regenerate_figures.py --check
```

---

## Evaluation Protocol

See `EVAL_PROTOCOL.md` for detailed protocol documentation.

---

## Schema Validation

To verify `master_results.json` schema consistency:
```bash
python3 scripts/validate_master_results_schema.py
```

Known issues documented:
- Field names vary: `name`, `experiment_id`, `run_id`, `experiment_name`
- Field aliases: `last_10_avg`, `last10_avg`, `last-10`
- Missing values encoded as `null` in some rows
- Method field: 81/95 rows have it; others use `algorithm`

This does NOT affect scientific claims; the verification script handles aliases.

---

## What NOT to Claim (Per Paper)

The paper explicitly disclaims these claims. Do not interpret the evidence as supporting:

1. GRPO universally improves reasoning
2. ZVF predicts final performance (it's a triage diagnostic)
3. PPO is inferior to GRPO (or vice versa)
4. Dense > MoE or MoE > Dense
5. Scaling laws established (exponential fits are descriptive only)
6. 30-step runs characterize asymptotic behavior
7. Verifiable rewards eliminate reward hacking
8. HumanEval/MATH results prove end-to-end capability
9. BrowserGym smoke test proves browser competence

---

## Troubleshooting

### Script not found
If `scripts/` is not in your PATH:
```bash
cd <extracted_bundle>
ls scripts/
python3 scripts/verify_claims_offline.py
```

### Missing dependencies
```bash
pip install -r requirements.txt  # or use venv
```

### SHA256 mismatch
If `sha256sum -c SHA256SUMS.txt` fails:
```bash
sha256sum <bundle_member>  # check individual file
# Compare against SHA256SUMS.txt
```

---

## Contact

For questions about verification, contact the authors through the venue's author response system.
