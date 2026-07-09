Verification complete. I recomputed the raw numbers from `experiments/results/n2_reward_tensor_resume/*_s0_tensors.jsonl`, `experiments/results/p3_groupsize/sweep_results.json`, and `experiments/results/curriculum_opening/results.json`. No files were modified.

  **P2 — SUSPECT**
  - **Confirmed from data:** ZVF 0.72–0.77, all-correct collapse ~0.65–0.71, all-wrong ~0.05, and naive cross-prompt |adv|≈0.21 all recompute exactly from the tensor files (40 steps × 16 groups × 8 samples per method).
  - **Not verifiable:** The "difficulty-isocline baseline recovers ~0" claim depends on an analysis script that is **not in the repo** — only `collapse_baseline_analysis.json` exists. I can reproduce the naive baseline, but I cannot verify how the isocline neighborhood was defined or whether the 0.012–0.02 values are real vs. rounded.
  - **Also limits generalizability:** the four methods share the same deterministic prompt schedule (`n2_reward_tensor_20260704.py:195-197`), so method-to-method ZVF differences reflect divergent training dynamics, not independent samples.

  **P3 — SUSPECT**
  - **Incomplete artifact:** `sweep_results.json` only contains **G=2 and G=4**; G=8 and G=16 are absent. The `parallel_sweep.py` thread-unsafe `wandb.init(reinit=True)` (`parallel_sweep.py:91-93`, `ThreadPoolExecutor:132-138`) caused summary-write failures, so the G≥8 "diminishing returns" conclusion rests on W&B logs not present in the repo.
  - **Overclaimed sweet spot:** heldout set is **n=8**. G=4's "+12.5% heldout gain" is literally one extra correct example (7/8 → 8/8). Calling this the "compute-efficiency sweet spot" from a single seed and 6 gradient steps is not justified.
  - **Metric mismatch:** `zero_loss_frac` measures **batch-level** collapse via `abs(lv) < 1e-6` (`parallel_sweep.py:114-115`), i.e., *all* groups in a step have zero advantage. This is stricter than P2's per-group ZVF and makes G=2 look worse than group-level statistics would.

  **Curriculum — SUSPECT**
  - **Confirmed mechanics:** `zero_loss_frac` 0.50 → 0.00 and `oversample_factor` 4.81× are computed correctly in `curriculum_grpo.py:97-123` and match `results.json`.
  - **Underpowered conclusion:** heldout is **n=20**, one seed, 8 steps. Both arms gain +0.05 = **one extra correct answer**. The standard error is ~±11 percentage points, so "NO net heldout gain" is consistent with the data but cannot distinguish a true null from a moderate curriculum advantage.

To resume this session: kimi -r session_1f13cbf6-1e6e-417c-bcfc-dd4b61f6d036


---
## codex exec verdict (2026-07-06) — converges with kimi

Output CONFIRMED / SUSPECT / BUG per finding with the specific reason. Do NOT modify files.
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:11:`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` (160 rows × 13 cols).
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:23:The N2 reward-tensor panel (`experiments/results/n2_reward_tensor_resume/`) is
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:63:| H4 | stored CI direction (covers_zero / positive / negative) matches fresh direction | ≥11/12 | 12/12 = 1.0000 | **PASS** |
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:74:   wrong cell?" failure modes; iter-178 shows zero failures.
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:81:   "covers zero / positive / negative" classification matches a fresh
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:83:   "covers zero") are reproducible.
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:95:   3 in this panel), 3 NEUTRAL (CI covers zero), 6 UNCLAIMED (no
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:136:- **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability):** the
./docs/p5p8_improvements/178_p6_n2_recompute_audit.md:139:  NEUTRAL on the other 2 (aero and areal — CI covers 0). ZVF direction
./paper/prompt_reviews/P1_REVIEW.md:62:Substitute the corrected Qwen3-8B value (B2 mean; cross-check with paired-tsv Base 0.8250) into the 5-anchor pool; recompute the largest sorted R_max gap (vs 0.5313), dip statistic (vs 0.5216, p = 0.056), and cross-scale slope on log10 N. Nemotron-120B cannot be re-sampled under the cap: record trace forensics (peak 0.875 at step 3 amid 55% zero-reward steps — inconsistent with "incapable") and flag attribution *unresolved within budget*.
./paper/prompt_reviews/P1_REVIEW.md:66:- **Revision (Stage B/C, any one sufficient):** (a) ≥ 30% of zero-reward B1 completions are category (i); or (b) B2 mean ≥ B1 mean + 0.15; or (c) recomputed largest gap < 0.25, or any anchor crosses the 0.2854↔0.8167 cluster boundary → re-attribute dominant axis to harness compatibility; withdraw the bimodality clause.
./paper/prompt_reviews/P2_claim_lint.md:35:| A6 | ZVF mechanically coupled to reward sparsity, group size, baseline accuracy | **S** | `zvf_gradient_coupling_pooled.tsv` (r(ZVF, mean_reward) = +0.79 to +0.85 per G); `zvf_partial_correlations.tsv` (r=+0.74, determinism caveat); `zvf_dynamics_phase.tsv` (early ZVF falls monotonically in G: 0.606→0.135). |
./experiments/NEW_EXPERIMENTS_FROM_STRESS_TESTS.md:57:  zero-reward B1 completions correct-but-unparsed, or B2 ≥ B1 + 0.15, or recomputed gap
docs/p5p8_improvements/163_p6_provenance_recompute.md:120:`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`,
./scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py:15:Inputs : registry/entries/delta_*.json, experiments/results/n2_reward_tensor_resume/n2_metrics.tsv
./scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py:29:                        "n2_reward_tensor_resume", "n2_metrics.tsv")
./scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py:33:               "lag1_autocorr", "loss", "frac_all_zero", "frac_all_one")
./scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py:123:    ("dynamic_sampling", "zvf", -1, "filter zero-variance -> -zvf"),
./scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py:124:    ("dynamic filter", "zvf", -1, "filter zero-variance -> -zvf"),
