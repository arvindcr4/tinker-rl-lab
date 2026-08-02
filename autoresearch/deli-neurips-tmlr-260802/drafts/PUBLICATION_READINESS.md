# Publication readiness decision

Date: 2026-08-02

## Decision

Do not send a new archival submission today.

The best current paper is the narrow flagship manuscript, *Same Terminal
Signal, Different Action*. Its best venue is TMLR after the overlapping NeurIPS
review ends or is formally withdrawn. Sending it to TMLR while NeurIPS
submission 36320 is active would create a parallel-submission problem.

The current evidence is not enough for a NeurIPS empirical claim about better
learning, lower rollout cost, a better controller, or a better framework. The
future sampler study can become a NeurIPS paper only after its full paired
confirmatory matrix passes the frozen gate.

## What is true

1. Under the paper's stated two-state, two-action assumptions, the same
   all-failure history can make stopping optimal in one state and a prepaid retry
   batch optimal in another. The minimax-regret result is proved.
2. The S1 CPU artifact checks 14 intended cases for each of TRL and verl and a
   shared 36-case controller matrix. Those intended fixtures pass their declared
   reference.
3. Native framework traces differ from that reference on four tested TRL cases
   and one tested verl case. This is a semantic difference, not a framework-bug
   claim.
4. The r4-2 audit trail is intact. The exact executed objective source was
   recovered and matches SHA-256
   `980a56a1651299a5adbe7a0927c13b12d42d9d7e1a36205500a24d5eeba9b61b`.
5. The sole completed intended-full balanced cell fails the frozen mechanism
   gate: 69 of 100 steps meet the joint rule, below the required 95. High cosine
   by itself does not pass that rule.
6. The filtered positive-control construction is infeasible under the frozen
   model and decoding contract. The causal question is unanswered.
7. In the separate E1 audit, all four arm-versus-GRPO comparisons are
   `INCONCLUSIVE` after exact paired-t power and the registered four-test BH
   correction.

## What must not appear as a result

- “Training improved GSM8K from 82.0% to 83.3%.” The descriptive scores exist;
  improvement and equivalence are both inconclusive.
- “Canonical GRPO transfers at 92.6% versus 92.1%.” Three of five pairs are
  zero-runtime backfills without an auditable upstream mapping.
- “ZVF predicts failure.” The retrospective result has two positives in 22
  runs, and reward alone flags the same two cases.
- Any PPO-versus-GRPO, TRL-versus-verl, estimator, or framework ranking from the
  reviewed tables.
- The spectral draft's benchmark gains, ZVF reduction, or learning claims. Its
  checked implementation is a synthetic diagnostic, and the cited 77-case raw
  audit bundle is absent.
- A 20% sampler cost saving or held-out non-inferiority. The confirmatory result
  table has not been produced.
- A deployed, external-user, or use-inspired result. No external intervention
  receipt exists.

## Submission files prepared

| File | Status | SHA-256 |
|---|---|---|
| Canonical manuscript PDF | Compiles, 11 pages | `b782db35560dedcdacfe1b7ee8181604b767f5bbc21e75af13ad01896c4e5763` |
| Full internal review bundle | Clean extraction passes | `09293f0dd83137642afe3fbc56cb919db41b878d96d1586306e852eb594b153f` |
| Anonymous TMLR PDF | Official TMLR style, 12 pages | `2f18d499f3fe5226d24c1c759ce48b8376ddcdb2b7a73559688081393c1fdb60` |
| Anonymous TMLR supplement | 33 files, clean extraction passes | `44fd058fd96097640e04af66f0d33905e2d1beeebe85be16993bfa8ab8920fd1` |
| Three-week progress deck | 12 editable slides; 18 reviewed vs. 12 active made explicit | `2a878cad35ecd8a82bcb79122681b421678eb162a17ffd024a7cb7fbfac57ce2` |
| Progress deck PDF | 12-page preview | `d9b883fc37b29bcea0b3033c8e614e8631df2cc75476ded411b3854d6b0526fd` |
| 18-paper portfolio review | Historical 18-file audit reconciled to current 12+6 roster | `ee7395f6c551579226228aee3f9584c18c5aa73abf6e2fbca9ee0ac3d801f95c` |
| Machine-readable paper verdicts | 18 reconciled data rows | `311bbe1f9c9d2c51a6cdd2973cfd6dbab6fc02ced4ca86c3191cc0e1a5dcee0f` |
| Current paper manifest | 12 active + 6 absorbed paths, pages, and hashes | `01710772876797cb9086b5c0f0125c69d082951756a5bfa06400d6c7beba0356` |

The anonymous supplement includes all 600 stored gradient relations, both S1
receipt projections, unchanged S1 source and tests, the executed objective
snapshot, an internal manifest, and an offline verifier. It has no hits for the
author name, institution, email, home path, or public account handle.

## Eighteen-file review and current paper queue

The source audit covered the full pre-consolidation 18-file snapshot: 868 PDF
pages and 329 distinct included source files. That frozen corpus is preserved as
the review evidence. It is not the current submission queue.

The repository now has 12 active roots, P1-P12, and six absorbed archives. The
12 active PDFs were freshly rebuilt, pass structural PDF checks, and total 488
pages. The six archives are readable history, not extra paper slots. U01 was
repaired to 232 pages but still has unresolved citation warnings and remains a
thesis/evidence compendium only.

- Keep P11 (former R08) as a bounded audit case study. Its four comparisons
  remain `INCONCLUSIVE`.
- Cut P2, including the former R02 stratification result, to one question.
- Treat P9 (former R04) as an artifact candidate only after an anonymous
  clean-machine release check; use P8 as documentation.
- Build P5/P6 into one reporting and registry resource, then obtain external
  entries and a user decision study.
- Run prospective experiments before promoting P1, P4, P7, or P12.
- Keep P08_fraud, R01, R02, R06, R07, and U01 absorbed.

The paper-by-paper evidence, nearest prior art, fatal gate, and disposition are
recorded in `audits/18_PAPER_PORTFOLIO_REVIEW.md`; the 18-row reconciliation
ledger is `audits/18_PAPER_VERDICTS.tsv`, and current paths/hashes are in
`audits/paper_portfolio/current_manifest.tsv`. This review does not replace the
flagship TMLR route. It explains why none of the reviewed material should jump
ahead of it.

## OpenReview record

The exact reviewed artifact is the 17-page PDF tied to forum `CXbcYe69BQ` and
submission 36320 in the dated local inspection record. The local files named
`NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md` and
`NEURIPS_2026_REVIEWER_9KJK_FOLLOWUP.md` are drafts. There is no posting receipt.

On 2026-08-02, the Ego browser was redirected to OpenReview login, and both
OpenReview note APIs returned HTTP 403. The current live response text could not
be re-read. Do not describe the local drafts as reviewer-visible.

## Safe route to publication

### If NeurIPS rejects

1. Wait until the review is formally closed.
2. Freeze the TMLR PDF and anonymous supplement digests.
3. Check the final NeurIPS record once more and prepare an overlap table.
4. Submit the conformance/decision-insufficiency paper to TMLR as a
   methods/reproducibility and registered-feasibility paper.
5. Use the current title, abstract, failed 69/100 result, and non-claims. Do not
   restore the RLM proposal or the old empirical headlines.

### If NeurIPS is accepted

Do not send this overlapping manuscript to TMLR. Turn the conformance code into
a later, distinct evaluation resource with new coverage and independent use, or
finish the sampler study as a separate future paper without reusing the accepted
paper's text, figures, or results.

### If NeurIPS is withdrawn

Wait until the withdrawal is effective in OpenReview, then run the overlap and
anonymity checks again before submitting to TMLR.

## What would make a future NeurIPS paper credible

- every required live mixed-update seam observed before confirmatory execution;
- the complete frozen paired-seed matrix, or a prospectively amended count;
- held-out outcomes and charged-token accounting for every valid row;
- the registered non-inferiority and multiplicity rules applied without repair
  after seeing results;
- at least two tasks and more than one implementation boundary;
- an independent clean-machine reproduction of the final artifact; and
- no headline based only on tests, preflights, hashes, or remote receipts.

## Role of the public hypothesis index

`arvindcr4/llm-rl-posttrain-index` should be a companion resource, not the main
claim of this paper. Its useful parts are the source-cited semantic deltas and
the map of neighboring methods. The 10 causal hypotheses, 31 workshop notes,
and toy digit-addition runs do not establish novelty or a large-model result.

A separate survey/resource paper would need a fixed search boundary, duplicate
handling, screening rules, source-fidelity checks, extraction-error estimates,
version governance, and evidence that another researcher can use the registry
to make a better decision. Until then, call it an index, not a systematic
review.

## Bottom line

There is a publishable TMLR-shaped paper here, but it is a careful methods and
failure-analysis paper. It is not a successful controller paper. The failed gate
is part of the contribution because the audit caught the tempting wrong story.
The legal timing gate, not another round of wording, is now the main blocker.
