# Dual-model bounded manuscript review

- Date: 2026-07-14
- Kimi command/model: `kimi -p --model kimi-code/kimi-for-coding`
- Z.AI command/model: `zai -p --model glm-5.2`
- Mode: independent, read-only review with repository tools disabled
- Decision rule: report only high-confidence scientific or reproducibility
  contradictions; do not propose stylistic edits

## Targeted evidence boundaries

Both reviewers checked the five manuscript boundaries most likely to create a
submission-level contradiction:

1. whether PPO or SAO results are measured or only preregistered;
2. whether the G=32 group-size surface is measured or reconstructed;
3. whether P7 establishes adaptive-controller superiority;
4. whether the observed 17x stack swing can be assigned causally to the
   backend; and
5. whether MIN-REPORT-RL contains eight items or a conflicting seven-item
   standard.

The claim-bearing packet drew from N01, the P5--P7 abstracts/conclusion, the
iter-27 group-size synthesis, and `PROGRAM_AUDIT.md`. Z.AI reviewed the compact
seven-file packet. Kimi's first matched-packet invocation produced no output
within seven minutes and was terminated; Kimi then completed on a smaller
claim card containing the exact passages for the same five checks. This result
also agrees with the earlier broader Kimi pass recorded in `KIMI_REVIEW.md`.

## Kimi result

Kimi returned `NONE` and gave five reasons:

- PPO/SAO are consistently labeled as hypotheses and evaluation-contract
  proposals; the empirical section explicitly contains GRPO runs only.
- G=32 is consistently labeled reconstructed or extrapolated, with a direct
  matched-budget sweep still required.
- P7 calls the adaptive-G result feasibility evidence: it matches the best
  fixed-recipe gain while spending more rollouts and does not establish
  superiority.
- The 17x swing is descriptive and explicitly confounded by an undisclosed base
  checkpoint change in the managed stack.
- P5's eight reporting items agree with P6's seven run-manifest fields plus the
  eighth held-out pass@k evaluation item.

## Z.AI result

Z.AI independently returned `NONE` and verified the same boundaries:

- N01 repeatedly states that PPO/SAO generalization remains unverified.
- The measured group-size table contains G in {2,4,8,16}; G=32 occurs in the
  explicitly reconstructed/illustrative analysis.
- The P7 abstract and conclusion agree that adaptive G is not a controller win.
- The 85.6/5.0 approximately 17x span is not assigned to the backend because
  checkpoint identity changed simultaneously.
- The P5 list and P6 7+1 mapping both contain eight requirements.

## Reconciliation

The two independent reviews agree on all five high-risk boundaries and identify
no source-level contradiction. No manuscript edit was applied from this pass.
The remaining gates are experimental: execute the PPO/SAO preregistration,
directly train the G=32 matched-budget arm, and run P7's fixed-token adaptive-G
bakeoff.
