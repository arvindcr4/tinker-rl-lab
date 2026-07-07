A. **GO / NO-GO per paper**

P1: [FIX-FIRST] Worst issue: overclaims “cross-library, cross-scale” scaling from heterogeneous, apparently single-seed traces and inconsistent frontier model range.

P2: [RISKY] Worst issue: safest conceptually, but claims correlations/collapse tracking without clear held-out causal design; ZVF may be tautological with reward sparsity.

P3: [FIX-FIRST] Worst issue: admits headline G=32 result is “illustrative, reconstructed” and “no G=32 cell was measured”; this is review-dangerous.

P4: [RISKY] Worst issue: “Qwen3-8B” / near-ceiling capped setting gives mostly null results; contribution may be judged too narrow, but not fatal if framed honestly.

P5: [FIX-FIRST] Worst issue: 17× backend swing is confounded by “managed backend also silently pinned a different base checkpoint,” so it is not a clean backend-only claim.

P6: [FIX-FIRST] Worst issue: repeats P5’s confounded 17× claim and contains suspicious registry entries using “qwen3.5,” which looks nonstandard/fabricated unless documented.

P7: [FIX-FIRST] Worst issue: escalates P2 from “not predictive/causal” to controller claims using small-n, toy-scale, single-task interventions; too much theory-to-practice leap.

P8: [FIX-FIRST] Worst issue: outside the ZVF/GRPO portfolio, uses suspicious “Qwen3.5-4B,” synthetic one-config fraud data, positive-enriched 500-row eval, and unmeasured deployment claims.

B. **INTEGRITY RED FLAGS across the portfolio**

“0.6B–∼685B” in P1 versus “0.6B–∼671B” in P2/P3/P4/P5/P6. This looks like inconsistent model accounting.

“We release all code, logs, and checkpoints.” Across frontier-scale “∼685B” or “∼671B” models, releasing checkpoints is implausible unless this means adapters/logs only.

“the managed backend also silently pinned a different base checkpoint” in P5. Then the abstract still says “swapping only the backend.” That is internally contradictory.

“moved final training reward from 5.0% to 84.4%” in P5 versus “from 85.6% to 5.0%” in P6. Same exhibit, different number/order.

“Qwen3.5-4B” / “tinker_grpo_qwen3 .5 -4 b_gsm8k .json” / “Qwen3.5-4B SFT.” Suspicious model name. If real, cite exact release/source. If not, this is a major credibility problem.

“368-run audit across seven models” in P7 conflicts in scale with the repeated “70+ RL runs” benchmark framing. Maybe W&B run count includes logged evals, but unexplained it looks inflated.

“DAPO-style dynamic sampling drives ZVF to 0.00” risks being mechanically true by construction if zero-variance groups are filtered, not an independent empirical success.

“agentic investigation of the alert queue, at roughly 85× lower cost than a human analyst” in P8 is not measured and sounds like product marketing.

“current Qwen3.5-4B SFT run reaches accuracy 0.792 but AUC 0.48268 on a 500-row positive-enriched held-out evaluation.” Accuracy on enriched data with AUC below random needs careful confusion-matrix/calibration explanation.

C. **CROSS-PAPER CONSISTENCY**

The central ZVF story is mostly coherent: P2 says ZVF diagnoses signal starvation but aliases mastery/incapacity; P7 proposes PCD/adaptive-G as repair. But P7’s tone “earns the reading it withheld” overreaches relative to P2’s caution.

P3 says measured equivalence only to G=16, while using an illustrative G=32 swing in the abstract/conclusion. That will look like laundering an unmeasured comparison into a headline.

P5/P6 duplicate heavily: same MinReport/registry/stackdiff narrative, same 17× and DAPO telemetry exhibits. They may be seen as one paper split into two.

P1/P2/P3/P4 repeat the same TinkerRL benchmark intro/figure/authors. Acceptable for a portfolio, but as separate submissions it will look salami-sliced unless each has a distinct experimental core.

P8 does not fit the portfolio except via “measure capability, don’t trust the label.” It weakens thematic coherence.

D. **TOP 5 FIXES before submission**

1. Remove or quarantine all unmeasured/reconstructed headline claims, especially P3’s G=32 result. Put them in appendix as exploratory only.

2. Fix all inconsistent numbers: 671B vs 685B, 84.4 vs 85.6, 70+ vs 368, retention/R² mismatches.

3. Audit every model name and citation. Replace “Qwen3.5” unless it is a real, citable artifact with exact checkpoint IDs.

4. Reframe P5/P6 backend claims: because checkpoint changed, say “stack bundle” not “backend-only.”

5. Reduce causal/controller language in P7 and deployment/economic claims in P8 unless backed by multi-seed, matched-budget, measured experiments.

E. **PORTFOLIO VERDICT**

**do-not-submit-yet**: the ideas are plausible, but the current digest contains enough inconsistent numbers, confounded claims, suspicious model names, and over-strong extrapolations to embarrass the student in review.
