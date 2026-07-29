# W&B evidence audit for NeurIPS submission 36320

**Audit date:** 2026-07-27  
**Access mode:** read-only API inspection; the credential was read from an ephemeral environment variable and is not stored in this artifact.  
**Scope:** claims material to the OpenReview rebuttal and post-May evidence relevant to a resubmission.

## Decision summary

1. The reported Qwen3-8B `92.6%` versus `92.1%` calculation is arithmetically reproducible from W&B, but the five-seed comparison is not a five-seed auditable matched experiment. Seeds 42--44 are zero-runtime backfills; only seeds 45--46 are live W&B records. Those four live records strongly timestamp-align with four Tinker histories through step 30, and Tinker contains plausible earlier paired sources for the backfills. However, the W&B backfills contain no upstream Tinker IDs, and Tinker exposes no seed, arm, algorithm, W&B ID, source commit, or held-out predictions with which to disambiguate the candidates. Seed 42 also uses batch 4 while seeds 43--46 use batch 8. The five-seed result and its transfer inference must therefore be withdrawn.
2. The Qwen PPO `0.350` and `0.225` values are both genuine W&B summaries, but they belong to two distinct 30-step Modal runs with different run IDs and start dates. They are not two aggregations of one trace. The submitted row is a provenance error and must remain quarantined.
3. W&B plus the repository substantiates a post-May E1 campaign: 40 accepted Qwen3-8B/GSM8K units using a clipped, completion-masked TRL GRPO baseline with `beta=0` (five arms by eight paired seeds), each with 500 held-out examples. A later statistical re-audit found that the frozen aggregate's DAPO `DISAPPEARS` verdict used a normal-approximation MDE and omitted the preregistered multiplicity step. The conservative finite-sample MDE is 1.012 pp, just above the 1 pp margin, so all four comparisons must be treated as inconclusive until the pipeline is repaired. E1 remains useful feasibility evidence for a same-stack audit design; it does not validate the submitted early-collapse rule, establish runner transfer, or resolve reference-KL dependence.

## Claimed matched canonical-GRPO comparison

W&B project: `arvindcr4-pes-university/neurips36320-matched-grpo`.

| Seed | Canonical run | REINFORCE run | Canonical final | REINFORCE final | Provenance class |
|---:|---|---|---:|---:|---|
| 42 | `p5trk9gs` | `psn205wv` | .920 | .915 | zero-runtime `backfill_wandb.py`; batch 4 |
| 43 | `d74fkjqw` | `j1gj3z8x` | .920 | .930 | zero-runtime `backfill_wandb.py`; batch 8 |
| 44 | `t9iaxwig` | `hmtwgbzu` | .925 | .925 | zero-runtime `backfill_s44.py`; batch 8 |
| 45 | `hyk5shwc` | `c18pk9et` | .925 | .915 | live `train_matched_tinker.py`; runtimes 594s/643s; batch 8 |
| 46 | `hekerdfz` | `lwz6nhis` | .940 | .920 | live `train_matched_tinker.py`; runtimes 577s/739s; batch 8 |

Across all five summary pairs, the means are .926 and .921 and the paired differences are `[.005, -.010, .000, .010, .020]`. The paired t-test gives `p=.3739`; the mean-difference 95% interval is approximately `[-.00888, .01888]`, or `[-0.89, +1.89]` percentage points. This interval is not an equivalence result.

The two live pairs alone differ by +1 and +2 percentage points in favor of the canonical-labelled arm. Two pairs cannot support a transfer conclusion. Tinker confirms that all four reached step 30, but neither W&B nor Tinker captures the source and item-level evidence needed to verify that estimator semantics were the only treatment difference.

## Qwen PPO discrepancy

W&B project: `arvindcr4-pes-university/tinker-rl-lab-world-class`.

| W&B run | Started | `final/last10_avg` | Cumulative reward | Runtime |
|---|---|---:|---:|---:|
| `ri2pajjl` | 2026-04-18 15:42:49 UTC | .350 | .28333 | 644s |
| `vrb9zxql` | 2026-04-19 03:01:29 UTC | .225 | .24167 | 824s |

Both use the same visible high-level configuration (`Qwen/Qwen3-8B`, seed 42, 30 steps, Modal H100, `ppo_reinforce`) but are separate runs. Without an explicit selection rule or reconciled source identity, choosing either one for a model-level table is post-selection. No PPO/GRPO comparison should use this row.

## Post-May E1 campaign

The repository's frozen aggregate and remote-verification receipts report `COMPLETE`: five arms by eight paired seeds, Qwen3-8B, GSM8K, 30 training steps, one stack fingerprint, and a fixed 500-item held-out evaluation for every accepted unit. A fresh authenticated Hugging Face audit resolves all 40 frozen repository/commit pairs. Each contains six checkpoint trainer states, a final adapter, and a final 500-row manifest whose hash matches the local manifest and campaign receipt. Four GRPO units lack only the separate evaluation-resume sidecar; their final traces and scores are intact.

Paired seed-level held-out results against the clipped, completion-masked TRL GRPO baseline (`beta=0`):

| Arm | Delta | 95% CI | Verdict |
|---|---:|---:|---|
| DAPO | +0.10 pp | [-0.45, +0.675] pp | `INCONCLUSIVE` under conservative finite-sample audit: 90% CI [-0.35,+0.575] pp is inside +/-1 pp, but paired-t 80%-power MDE is 1.012 pp, above the margin |
| GSPO | +0.50 pp | [-0.125, +1.20] pp | `INCONCLUSIVE` |
| Dr.GRPO | -0.20 pp | [-0.95, +0.725] pp | `INCONCLUSIVE` |
| AERO | -0.075 pp | [-0.825, +0.675] pp | `INCONCLUSIVE` |

None of the 40 units collapsed. This campaign directly improves coherence, seed count, clipped/completion-masked runner disclosure, held-out size, and run-level provenance. It supports the feasibility of a same-stack audit, but the corrected finite-sample analysis leaves all four comparisons inconclusive. It supplies no positive failures for validating the submitted early-step triage rule, no untreated pre/post capability effect, no reference-KL result, and no matched comparison to the submitted runner.

## Rebuttal-safe wording

> Correction: our initial response over-described a five-seed Qwen3-8B comparison. The five W&B summaries reproduce 92.6% versus 92.1% and recomputed `p=.374`. The four live seed-45/46 W&B records strongly timestamp-align with four Tinker histories through step 30, and Tinker shows plausible earlier pairs. However, the three zero-runtime backfills contain no upstream IDs, while Tinker exposes no seed/arm/algorithm/W&B metadata with which to identify them. We therefore withdraw the five-seed result as transfer evidence.

> Separately, after submission we executed a preregistered same-stack audit using a clipped, completion-masked TRL GRPO baseline (`beta=0`): five GRPO-family arms by eight paired Qwen3-8B/GSM8K seeds, 30 steps, and a fixed 500-item held-out evaluation per unit. Our frozen private ledger records checkpoints, fingerprints, and per-item traces for all 40 units. A conservative finite-sample re-audit treats all four held-out comparisons as inconclusive; no unit collapsed. This is feasibility evidence for an audit workflow, not transfer of the submitted runner, reference-KL dependence, or the early-collapse rule, and independent verification requires an anonymized artifact.

## Security note

At least one historical W&B configuration contains a long-lived API credential as a logged configuration value. The credential supplied for this audit should be rotated. No credential is reproduced here.
