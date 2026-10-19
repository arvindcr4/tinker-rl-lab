# Errata — ARS-corrected revision, 19 October 2026

Additive corrected copy of `reports/public_revision_2026-10-11/` thesis
sources. The frozen 11 October export is unchanged. No new training,
evaluation, number, or reference. Full text in thesis Appendix E §E.4
(`thesis/ch_back_errata.md`); roadmap log in
`reports/ars_review_2026-10-19/corrections/06_status.md`.

1. **Abstract to body tier.** Saturated PPO/GRPO: exact sign-flip p = 0.625
   before paired-t p = 0.374 (0.990 vs 0.992, five seeds) — pilot only.
   Unsaturated: exact p = 0.0625 (n = 5 floor) before paired-t p = 0.016;
   +5.0pt demoted to "minimal untuned-critic PPO shows no detectable gain;
   GRPO arms gained", with untuned critic (no warm-up/GAE/tuning), 8×
   distinct-prompt exposure, and 190–196-token completions under a 200 cap.
2. **Stored-evidence labelling.** All actor-dependent campaign results are
   "stored-evidence, not replayed". Dirty training source, exact consumed
   rows, sampler weights, and adapter revision remain unrecoverable/withheld.
3. **Audit disclosure.** Public-checkout audit is INCOMPLETE: 970 registered
   occurrences, 828 PASS, 22 CONTEXT (context-only), 120 ARTIFACT_MISSING;
   33 of the 828 are arithmetic from printed counts and 501 are
   token-presence checks against the project's own sources. The 120 missing
   occurrences are located by chapter in §1.8 (Chapter 8: 59; §9.1/Table 9.1:
   16; Table A.8: 17; §7.1–7.2: 24; Table A.10: 4). Disclosed in abstract,
   §1.8 (Table 1.1), §9.2. Affected claims are reported, not publicly
   verifiable from this package.
4. **Terminology box (Ch. 1).** Every headline maps to clipped-GRPO /
   REINFORCE group-mean / unknown (managed/unrecovered), assigned row by row
   from Table 4.2. Campaign actor loss is unknown; custom Tinker loops, the
   P3 group-size sweep and the four-method controller trace are REINFORCE
   group-mean (no clipping).
5. **ZVF P2 → instrumentation.** Mixed-pool correlations quarantined in
   §6.3.4 as withdrawn (17 finite cells, 9 synthetic; AUROC panel 41/52
   synthetic safe rows, 7 imputed collapse anchors; 45/60 range "runs" are
   variance-mitigation projection rows). No prognostic claim without measured-only held-out
   validation (not run).
6. **Controller unevaluated.** Plug-in zero-rescue is an estimator tautology
   (461/640 groups at p̂ ∈ {0,1}); no matched-budget bakeoff was run.
7. **E14 denominator erratum.** 0.5131043831902395 = 2,271/4,426 (native
   accepted-set), not 2,271/4,428 (= 51.29%, all-disposition diagnostic).
   The recovered terminal note is preserved unchanged; its reading is
   corrected. Campaign figures are harness scores with failure rates inline.
8. **Registry.** R0–R5 are declared-difference codes, not flip-risk
   probabilities; registry/badge presented as schema proposal (no external
   validation). C1 is a parser/reference audit case study (assistant-assisted,
   no independent human validation).

Author-only items are listed in `AUTHOR_TODO.md` and are not done here.
Checksums/quality receipts for this copy are pending (not regenerated).
