# ARS Academic-Paper-Reviewer Panel Report — Public Thesis Revision 2026-10-11

**Title reviewed:** *Tinker RL Lab: Post-Training of Large Language Models — A multi-framework training harness and studies of GRPO signal starvation* (Arvind C R, PES University, M.Tech Data Science and Machine Learning, UE20CS972 Project Phase-2)
**Manuscript path:** `reports/public_revision_2026-10-11/` (317-page corrected PDF `thesis/Thesis_Report_ArvindCR_2026-10-11_ESA_CORRECTED_REVIEW.pdf`, SHA-256 `83d8ee3a…80e0`)
**Date:** 2026-10-19
**Mode:** full (ARS academic-paper-reviewer v1.11.1, one execution seat panel, one manuscript only)
**Thesis scale used:** Pass / Minor corrections / Major corrections / Revise-and-resubmit, mapped to journal scale in §4.

## Disclosures

- Single-agent-per-paper role separation is NOT independent error processes; all five seats below were produced by one agent in one pass and share the same model family. Agreement across seats must not be read as independent corroboration.
- NOT_CALIBRATED: no calibrated score, percentile, or acceptance probability is offered. Severity labels are ordinal judgements tied to cited anchors.
- criteria_binding_unavailable: venue inferred only from `\documentclass[12pt,a4paper]{report}` (`thesis/preamble.tex:8`), title page (`thesis/frontmatter.tex:19-36`), and filename. No author-confirmed criteria binding, PES rubric, or examiner instructions were supplied. Thesis standards below are the panel's generic M.Tech expectations, not a PES-confirmed rubric.
- This review is READ-ONLY on the manuscript. No manuscript file was edited.

## Coverage statement — exactly what was opened

Opened in full as text: `README.md` (15 lines), `COMPLETION_ASSESSMENT.md`, `thesis/README.md`, `thesis/frontmatter.tex` (239 lines, incl. abstract ll.180-229), `thesis/preamble.tex`, `thesis/ch01_introduction.md` (161 lines), `thesis/ch02_literature.md` (88 lines), `thesis/ch04_methodology.md` (153 lines), `thesis/ch06_results_core.md` (199 lines), `thesis/ch07_results_infra.md` (151 lines), `thesis/ch08_results_fraud.md` (Appendix F, 66 lines), `thesis/ch09_results_campaign.md` (Chapter 8, 198 lines), `thesis/ch10_synthesis_conclusions.md` (Chapter 9, 116 lines), `thesis/ch_back_c1_evidence.md` (Appendix G, 146 lines), `thesis/ch_back_reproducibility.md` (Appendix B, 120 lines), `thesis/ch_back_errata.md` (Appendix E, 78 lines). Opened as structured data: `VALIDATION.json`, `thesis/public_quality_receipt.json`, `PUBLIC_CHECKOUT_NUMBER_AUDIT.json` (summary + sampled groups), `thesis/c1_case_ledger.json` (schema, 64 cases / 11 events counts, sampled cases). Inspected in part: `thesis/references.bib` (744 lines, 88 entries; head + citation-key census, not every entry), `thesis/ch_back_run_registry.md` (first 80 lines), `thesis/figures/fig_zvf_definition.tex` (head), directory listing (25 figure `.tex`/`.pdf` pairs).

Sampled or skipped, disclosed: `ch03_requirements.md`, `ch05_implementation.md`, `ch_back_campaign_detail.md`, `ch_back_coverage_audit.md`, `ch_back_notation.md`, `ch_back_provenance.md`, remainder of `ch_back_run_registry.md`, `build_thesis.py`, `compile_figures.py`, `provenance_audit.json` (238K), `PUBLICATION_MANIFEST.json`, and the rendered PDF binary (page count and hash taken from receipts, not re-extracted) were **not** read line-by-line. Findings that depend on them are marked unresolved. Grep/search was used only to locate; every cited claim below was opened as body text.

---

## 1. Phase 0 — Field analysis

**Primary discipline:** Machine learning — reinforcement learning for LLM post-training (RLVR/GRPO). **Secondary:** ML systems / reproducibility science; benchmark evaluation governance. **Paradigm:** Empirical measurement and negative-results science; explicitly *not* a new-algorithm paper. The thesis states this itself: "the scope … is measurement and diagnostics, not leaderboard performance" (`ch01_introduction.md:99`). **Methodology type:** Mixed — (a) controlled small-scale training experiments with multi-seed contrasts, (b) retrospective recomputation from stored tensors, (c) artifact/system building (harness, registry, `stackdiff`), (d) single-actor benchmark campaign with receipt governance, (e) frozen-policy sampling diagnostics (V2, C1). **Inferred target venue/tier:** Institutional M.Tech thesis (PES University), evaluation-review copy; as a journal analogue, a reproducibility / ML-systems venue (e.g., TMLR, ReScience-style, or a NeurIPS Datasets & Benchmarks / ML Reproducibility workshop) — not a main-track algorithms venue. **Maturity:** Late-stage thesis compendium aggregating eight internal reports (P1–P8) plus the E1–E14 campaign; scientifically mature in self-audit, immature in confirmatory power. The manuscript's own completion note is candid: "not complete as a fully reproduced research campaign or an approved institutional submission" (`COMPLETION_ASSESSMENT.md:3`).

### Reviewer Configuration Cards

**Journal-Fit (EIC).** *Lens:* Does a 317-page M.Tech thesis whose headline is mostly well-documented nulls meet the degree's contribution bar, and is it submittable in this exact form? *Must check:* contribution vs. compendium bloat, institutional readiness (declarations, approvals, independent review), proportion (fraud appendix, registry detail). *Blind spot to avoid:* confusing honesty about limits with having overcome them.

**R1 Methodology.** *Lens:* Identification, power, and inference. *Must check:* seed counts (n=3/5/8), saturated-task nulls, paired-by-seed-not-by-item designs, exact-test floors (n=5 → p≥0.0625), multiplicity, budget matching, mixed synthetic/measured pools. *Blind spot:* demanding large-N confirmatory trials the degree timeline cannot fund — judge whether claims are correctly downgraded instead.

**R2 Domain.** *Lens:* GRPO/RLVR correctness and literature placement. *Must check:* loss-form taxonomy (REINFORCE vs clipped surrogate vs managed unknown), ZVF novelty vs DAPO/GRESO/RL-ZVP prior art, Bernoulli-null derivation, controller logic. *Blind spot:* treating the label "GRPO" as one algorithm — the thesis's own point.

**R3 Perspective.** *Lens:* Reproducibility, governance, and external validity. *Must check:* receipt binding to actual runs, decontamination scope, public-checkout audit (120 missing), serving-stack confounds, who can actually reuse this. *Blind spot:* accepting hash consistency as scientific validity — the thesis warns against this itself (`COMPLETION_ASSESSMENT.md:27`).

Devil's Advocate is the fixed 5th seat (no card, by design).

---

## 2. Five seat reports

### 2.1 Journal-Fit (EIC)

**Summary judgement.** This is an unusually honest thesis that documents, at great length, why most of its motivating hypotheses could not be established. That honesty is its chief scholarly virtue and its chief examination risk: examiners must decide whether a measurement-and-governance contribution with no demonstrated training gain satisfies an M.Tech in Data Science and Machine Learning. In this panel's judgement it can — but not in the present 317-page compendium form without major corrections to framing, proportion, and submission readiness.

**Strengths (anchored).**
- *EIC-S1 — Canonical-numbers discipline.* Table 1.1 fixes one row per measurement with model/task/G/n/source (`ch01_introduction.md:135-161`), explicitly refusing to interchange the three Qwen3-8B GSM8K training-reward summaries (last-10 34.4%, trace mean 0.285, pooled 99.9% vs 73.4%). Few theses at any level do this.
- *EIC-S2 — Visible retractions.* Withdrawn claims stay visible with reasons: the MoE-vs-dense gap withdrawn after Nemotron misclassification (`ch06_results_core.md:43`), the "Dr. GRPO uniquely trap-free" claim withdrawn as unlocatable (`ch06_results_core.md:143`), mediation percentages withdrawn (`ch06_results_core.md:157`). Appendix E (`ch_back_errata.md`) itemises ~30 source corrections. This is exemplary scholarly conduct.

**Weaknesses.**

- **EIC-W1 — Compendium proportion defeats the thesis argument.** *Anchor:* eight objectives O1–O8 (`ch01_introduction.md:49-63`), four RQs, P1–P8, E1–E14, V2, C1, plus a synthetic-data fraud appendix that "sits outside the RL contribution" (`ch08_results_fraud.md:3`). *What is wrong:* at ~90k words of chapter Markdown alone, the contribution (ZVF diagnostic + reporting standard + governance protocol) is diluted by exhaustive lane-by-lane and iteration-by-iteration audit trails (e.g., the fifteen exploratory length-bias tests in `ch06_results_core.md:147-159`). An examiner cannot tell what would be lost if half were cut. *Fix:* add a 2-page "what this thesis claims, in one table" (claim → evidence tier → chapter) after the abstract, and move iteration-level audits (iterations 12/20/24/136 detail) to a supplementary audit volume. State explicitly which chapters are examinable contribution vs. archive.

- **EIC-W2 — Not submittable in this exact form, by its own receipts.** *Anchor:* `README.md:11` ("does **not** claim the strict independent-review gate passes; … `pending-independent-review`"), `thesis/public_quality_receipt.json` (`status: pending-independent-review`), `COMPLETION_ASSESSMENT.md:16` (guide approval, declarations, signatures, similarity clearance outstanding), `frontmatter.tex:110-113` ("NEEDS AUTHOR CONFIRMATION" on the AI-tools declaration). *What is wrong:* the public derivative correctly refuses to certify itself, but a thesis submitted for award cannot carry a pending-review receipt and an unconfirmed AI declaration. *Fix:* obtain the separate final-thesis review of these exact bytes, confirm the AI-tools list against PES policy wording, and replace the evaluation-review title page with the institutional submission front matter — or state plainly in the viva that this artefact is a review copy, not the submission.

- **EIC-W3 — Contribution of the candidate vs. group inheritance needs examiner-proofing.** *Anchor:* Declaration (`frontmatter.tex:85-92`) and §1.6 (`ch01_introduction.md:107-115`): Semester-3 group built the harness; candidate claims direction, P1–P8, campaign. Boundary is git tag `capstone-final-2026-04-25` (`ch_back_reproducibility.md:66-68`). `COMPLETION_ASSESSMENT.md:17` adds "Collaborator authorization remains unresolved." *What is wrong:* commit counts (584 commits, 5,050 files) are explicitly disclaimed as contribution measures, leaving the viva with no positive, checkable partition of code authorship; unresolved collaborator authorisation is an institutional risk, not a footnote. *Fix:* include a signed contribution matrix (component × author) countersigned by the guide, and resolve or formally record collaborator authorisation before submission.

### 2.2 R1 — Methodology

**Summary judgement.** Methodologically the most self-aware small-N RL study this panel has seen — and still underpowered for its central comparative question (RQ2), which the thesis correctly leaves "Unanswered at this budget" (`ch10_synthesis_conclusions.md:74`). The corrections in Appendix E fix many inference errors; the remaining problems are design-level, not arithmetic.

**Strengths.**
- *R1-S1 — Recomputation over log-trusting.* ZVF values 0.130/0.190/0.155 recomputed from stored tensors (`ch06_results_core.md:57`); per-step-average Bernoulli null replacing the pooled/held-out baselines with Jensen-gap reasoning (`ch06_results_core.md:73`); design-cluster bootstrap for E11's 156 paired designs replacing an iid-312 Wilson interval (`ch09_results_campaign.md:84`).
- *R1-S2 — Correct null language.* "Statistically indistinguishable … does not mean demonstrated equivalence" with TOST reported separately (`ch06_results_core.md:13`, `ch06_results_core.md:39`: TOST p=0.104 at ±0.005 fails, passes at ±0.010).

**Weaknesses.**

- **R1-W1 — Headline contrasts are at ceiling with broken pairing.** *Anchor:* same-stack PPO vs GRPO, 0.990 vs 0.992, paired Δ=−0.002, p=0.374 (`ch01_introduction.md:29`, `ch06_results_core.md:39`); evaluation sets share only 3–7 of 200 items per seed; train–eval overlap ~7% (GRPO) vs ~40% (PPO) by RNG-stream replay. Group-size sweep likewise at 0.978–0.990, ANOVA F(3,8)=1.82, p=0.22, MDE d=3.07 at n=3 (`ch06_results_core.md:95`). *What is wrong:* a null at 0.99 accuracy with non-identical, asymmetrically contaminated item sets cannot bound the estimator effect; the thesis says so, yet RQ2 is still framed as addressed by these rows. *Fix:* reframe the saturated contrasts as pilot/feasibility only; the unsaturated rerun (§6.5.5) should be the sole RQ2 evidence, with a pre-registered item-paired design (identical items, disjoint from training) listed as required future work with a power calculation.

- **R1-W2 — The one positive contrast rests on an untuned strawman and a test-sensitive p-value.** *Anchor:* unsaturated rerun GRPO G8 − PPO = +0.050 [0.015, 0.085], paired-t p=0.016, exact sign-flip p=0.0625 (`ch06_results_core.md:185-187`, abstract `frontmatter.tex:201-204`); PPO is "a linear value head trained from zero … no critic warm-up, no GAE, no tuning"; PPO sees 8× distinct prompts/step; completions 190–196 tokens under a 200 cap. *What is wrong:* at n=5 the exact two-sided test cannot reach 0.05, so "suggestive" hinges entirely on the parametric t; beating an untuned critic with unequal prompt exposure under a near-binding cap is not evidence about PPO vs GRPO as algorithms. The thesis discloses all of this — but the abstract still leads with the 5.0-point figure. *Fix:* demote the contrast in the abstract to "no detectable gain for a minimal critic baseline; GRPO arms gained"; report the exact p first; either tune/warm up the critic or rename the arm "batch-centred baseline with learned offset" throughout.

- **R1-W3 — ZVF predictive validity was built on synthetic pools; the correction guts P2's empirical core.** *Anchor:* historical ρ≈0.27 "n=23" actually 17 finite cells, 9 synthetic, mixed outcomes incl. training reward (`ch06_results_core.md:81-85`); 45 of 60 "runs" behind ZVF ranges are simulation rows (`ch06_results_core.md:65`); AUROC 0.929 panel has 41/52 synthetic safe rows and 7 imputed collapse anchors (`ch06_results_core.md:85`). *What is wrong:* P2's defensible remainder is "ZVF is recomputable and regime-dependent (0.16–0.84)" — a measurement, not a diagnostic with demonstrated prognostic value. The thesis concedes "predictive validity is unestablished" (abstract), yet Chapter 7 still derives a controller from it. *Fix:* retitle P2's contribution as *instrumentation*; move all mixed-pool correlations to a clearly-labelled "withdrawn analyses" subsection; require a measured-only, held-out-endpoint validation (even if null) before any controller claim.

- **R1-W4 — Budget matching is inconsistent across the very studies compared.** *Anchor:* `ch04_methodology.md:82` — G-sweep fixes prompts×steps (1,280–10,240 completions), PPO contrasts fix completions but change prompt exposure, Qwen3-8B panel fixes rollouts but changes optimiser steps. *What is wrong:* "same-stack" therefore means different equalities in different chapters; cross-study synthesis (§9.1) compares them as if commensurable. *Fix:* add a budget table (prompts, completions, generated tokens, optimiser steps, wall-clock where known) per contrast and forbid cross-budget comparisons in synthesis.

- **R1-W5 — Multiplicity and exploratory inflation in P4.** *Anchor:* ~15 tests across iterations 12/20/24, uncorrected p-values, seed-level n=3 bootstraps with 10 distinct configurations (`ch06_results_core.md:147`); BH leaves nothing below 0.05 (smallest adjusted 0.116–0.487 depending on family, `ch06_results_core.md:157-159`). *What is wrong:* the chapter commendably reports the correction, but still narrates individual "excluding zero" intervals (e.g., Δρ=+0.098 [+0.073,+0.119]) at length. *Fix:* collapse §6.5.3 to a single exploratory table with adjusted p-values only; delete mechanistic narration of non-surviving effects.

### 2.3 R2 — Domain (GRPO / RLVR)

**Summary judgement.** Domain treatment is careful, current through mid-2026, and correctly refuses priority claims. Its central technical insight — that several "GRPO" runners are REINFORCE with a group-mean baseline — is important and well-evidenced. The cost of that honesty is that large parts of the thesis are not about GRPO as published.

**Strengths.**
- *R2-S1 — Loss-form audit.* Table 4.2 (`ch04_methodology.md:61-80`) reads runner code row-by-row: custom Tinker loops have no ratio/clip; TRL dump has β=0.04 with on-device reference; managed settings "unknown remains unknown." The thesis insists zero-advantage ≠ zero optimiser step when KL/auxiliary losses exist (`ch04_methodology.md:38,57`). This corrects a widespread sloppiness in the field.
- *R2-S2 — Prior-art positioning.* Zero-variance interventions (DAPO dynamic sampling, GRESO, CPPO, NGRPO, RL-ZVP, AERO) are surveyed with an explicit no-priority disclaimer (`ch02_literature.md:29-33`); novelty for the registry is bounded as a search finding, "not proof of absence" (`ch01_introduction.md:41`, `ch02_literature.md:47`).

**Weaknesses.**

- **R2-W1 — If the runners are REINFORCE, the title overclaims.** *Anchor:* "the custom-loss Tinker reference loop is REINFORCE with a group-mean baseline and has no ratio or clipping" (`ch01_introduction.md:13`); campaign actor's executed loss "unverified because the dirty training source is missing" (`ch09_results_campaign.md:27`). *What is wrong:* a thesis titled around GRPO whose best-audited stacks are REINFORCE variants, whose frontier anchors are managed-API black boxes, and whose campaign actor's loss is unknown, studies *group-relative REINFORCE-style post-training* more than GRPO (Shao et al.) proper. External readers will mis-cite it. *Fix:* add a terminology box in Chapter 1 mapping each study to {clipped GRPO surrogate / REINFORCE group-mean / unknown managed} and use those labels in every headline, not only Table 4.2.

- **R2-W2 — ZVF/PCD theory is correct but thin, and its provenance needs viva care.** *Anchor:* closed form h_G(p)=p^G+(1−p)^G derived in one line (`ch06_results_core.md:73`); source document headed as ChatGPT Pro / Gemini Deep Think contributions, "NOT experiments" (same paragraph; declared at `frontmatter.tex:126-130`). PCD is symmetric and "cannot by itself distinguish mastery from incapacity" (`ch07_results_infra.md:55`); gradient link is corr=+0.71 on a 0.5B toy, "directional only" (`ch07_results_infra.md:53`). *What is wrong:* the mathematics is elementary Bernoulli bookkeeping; presenting S=p(1−p)(1−h_G) as a "signal-starvation theory" (P7 title) inflates it. The candidate must be able to derive every line unaided at the board, given the frontier-model provenance. *Fix:* retitle P7's theory as a *model* or *accounting identity*; add a viva-ready appendix deriving h_G, PCD expectation, and T3 (mean |advantage|=2p̂(1−p̂)) without appeal to synthesis documents.

- **R2-W3 — Controller "zero rescue" is a tautology of the plug-in estimator.** *Anchor:* 461/640 groups with p̂∈{0,1} force h_G=1 for all G; "Neither calculation identifies latent p as 0 or 1" (`ch07_results_infra.md:59`); 92.3% of escalations on all-correct groups (`ch07_results_infra.md:61`); fresh-rollout V2 then shows 33/461 groups do become mixed, 13 numeric (`ch07_results_infra.md:78-94`). *What is wrong:* the retrospective audit could never have found rescue by construction; pages of Pareto/Bayesian refinement elaborate a model known to be misspecified at the boundary, and the N10 "replication" reverses the threshold (m<τ) and unit (`ch07_results_infra.md:69`). *Fix:* present the plug-in result in one paragraph as a negative design lesson; delete or appendix the eight-controller posterior-predictive ranking; the honest controller test is the unrun matched-budget bakeoff — say the controller is *unevaluated as a training policy*.

- **R2-W4 — Framework comparison cannot support any framework claim.** *Anchor:* Tinker (Qwen3-8B-Base, last-10 0.856) vs TRL (Qwen3-8B Instruct, 0.050), verl/OpenRLHF rows are "dry-run placeholders" (`ch01_introduction.md:19`, `ch04_methodology.md:28`, `ch_back_run_registry.md:49-61`); marginal η² sum to 1.897 on collinear factors (`ch06_results_core.md:11`). *What is wrong:* O1 ("single harness … multiple back-ends") is delivered as software, but the 17× swing exhibit that motivates P5 is confounded by checkpoint identity — an under-specification exhibit, as the thesis says, not a backend effect. The synthesis occasionally drifts toward the stronger reading. *Fix:* audit Chapter 9 for any sentence implying a backend effect size; keep the exhibit strictly as "labels under-specify stacks."

### 2.4 R3 — Perspective (reproducibility & governance)

**Summary judgement.** The governance apparatus — receipts, terminal-state taxonomy, no-pooling rule, number audit — is the thesis's most transferable contribution. It is also where the gap between *consistency* and *validity* is widest, and where a public reader still cannot verify the headline campaign.

**Strengths.**
- *R3-S1 — Terminal-state honesty.* Seven-state lane taxonomy, blocked lanes recorded rather than dropped (`ch04_methodology.md:122`); original vs replacement scopes never pooled (`ch09_results_campaign.md:5`); E6's 90/812 reported as a lower bound with 108 judge-dependent tasks ungraded (`ch09_results_campaign.md:115,134`).
- *R3-S2 — Serving-stack audit.* Table 8.2 traces the missing-stop-sequence bridge defect lane-by-lane and refuses to quantify its causal share of E4's 0.0 (`ch09_results_campaign.md:49-74`, `ch09_results_campaign.md:138`). This is exactly the stack-conditioned reporting P5 preaches, practised on the author's own campaign.

**Weaknesses.**

- **R3-W1 — Public-checkout reproducibility is INCOMPLETE by the project's own audit.** *Anchor:* `PUBLIC_CHECKOUT_NUMBER_AUDIT.json`: 970 registered occurrences, 828 PASS, 22 CONTEXT, 120 ARTIFACT_MISSING, status INCOMPLETE; `COMPLETION_ASSESSMENT.md:23` ("`--check` exits 1 as intended"; 33 of the 828 are arithmetic from printed counts, not raw recomputation). *What is wrong:* one in eight registered numbers cannot be traced from the public package; the historical `provenance_audit.json` larger pass count "must not be attributed to this public checkout." A reproducibility thesis whose public artefact fails its own completeness gate needs that failure on the abstract page, not only in assessment files. *Fix:* state the 828/22/120 split in the thesis abstract/limitations; either deposit the withheld artefacts under controlled access with a named custodian, or downgrade affected claims to "reported, not publicly verifiable" in Table 1.1.

- **R3-W2 — Campaign actor provenance does not bind to the claimed run.** *Anchor:* adapter revision withheld (`ch09_results_campaign.md:13`); dirty source diff not uploaded; exact 80 consumed prompts unknown; sampler weights unavailable (`ch09_results_campaign.md:19,43`); decontamination is a bounded 8-gram envelope check over 3 of 14 suites, max Jaccard 0.134 for Omni-MATH, explicitly not global/semantic (`ch04_methodology.md:124`); E10 receipt itself declares `heldout_claim: false` (`ch04_methodology.md:112`). *What is wrong:* receipts bind file identity, not execution: "Hashes establish consistency, not source authenticity or scientific validity" (`COMPLETION_ASSESSMENT.md:27`). The campaign therefore demonstrates a *protocol*, not a verified evaluation of a known training run. *Fix:* every campaign headline should carry the suffix the thesis sometimes uses — "stored-evidence result, not replayed" (`COMPLETION_ASSESSMENT.md:18`) — including in the abstract, where VerilogEval/LAB-Bench/AgentDojo figures currently read as plain scores (`frontmatter.tex:220-223`).

- **R3-W3 — Headline campaign scores are harness scores with majority failure modes.** *Anchor:* E1: 593/713 patches corrupt for `git apply`, tests ran on unpatched base (`ch09_results_campaign.md:92`); E8: 1,259/1,967 unparsed, 1,086 truncated (`ch09_results_campaign.md:62,80`); E11: 150/183 failures are extraction failures under a 1,024-token thinking-on cap (`ch09_results_campaign.md:84`); E14: capped 34.5% vs uncapped 85.9% correct, confounded by difficulty (`ch09_results_campaign.md:86`). E14's recovered terminal note also mislabels 2271/4428 with the 2271/4426 decimal (same paragraph). *What is wrong:* quoting 41.35%, 0.2288, 51.31% without the failure-mode qualifier in the same breath invites mis-citation as capability measures. *Fix:* report each as "end-to-end harness score (X% parse/format failures)" at first mention everywhere, including tables; correct the E14 note's arithmetic in an erratum rather than only in prose.

- **R3-W4 — Registry/badge validity is weaker than its presentation.** *Anchor:* on 98 manifests item 2 validated in 0/98; nine subfields at 0.0% (`ch07_results_infra.md:19`); badge constant across cells, ρ=0.000 for seed-pair dispersion (`ch07_results_infra.md:23`); `stackdiff` R0 historically printed "identical" while unscored model/task/decontamination fields could differ (`ch07_results_infra.md:41-45`). *What is wrong:* RQ3 is answered by a single DAPO pair receiving R5 — a demonstration that the code runs, not that the ordinal code predicts label flips. The thesis concedes it is "not a calibrated probability," but "flip-risk" language persists. *Fix:* rename R0–R5 outputs to "declared-difference codes"; validate the badge against at least one external corpus or present the registry purely as a schema proposal.

- **R3-W5 — C1's human-validity gap.** *Anchor:* C1 11/64 frozen endpoint, 3 conservative witnesses (`ch07_results_infra.md:99-115`); review "assistant-assisted … do not provide independent human validation" (`ch04_methodology.md:145`); ledger `review_method: Assistant-assisted … not independent human validation` (`c1_case_ledger.json`); cohort selected on 8 parser-zeros then truncated to first 64/87 (`ch04_methodology.md:135`). *What is wrong:* the most quotable new result ("3 recoveries") depends on LLM judgements of reference correctness (16 wrong references, 19 ambiguous of 64) with no human adjudication and a selection mechanism guaranteed to enrich reference faults. The Wilson interval is a "working model," not population coverage (`ch04_methodology.md:143`). *Fix:* obtain blind human rating on at least the 11 event cases (small cost, decisive for credibility); report C1 strictly as a parser/reference audit case study, not a recovery-rate estimate.

### 2.5 Devil's Advocate (fixed 5th seat)

#### Strongest Counter-Argument (200–300 words)

The strongest case against this thesis is that it is a monument to measurement that never demonstrates the phenomenon it measures matters. ZVF is recomputable — but its predictive validity is withdrawn, its collapse association was synthetic, and its controller rescues nothing except by definition. The reporting standard is plausible — but its badge cannot discriminate anything on the only corpus tested, its key field validates 0/98 times, and its risk code is validated on one hand-picked pair. The campaign is governed — but its actor's training code is lost, its training rows are unknown, its decontamination covers three suites by 8-gram overlap, seven original contracts were never run, and its only trained-vs-base test is inconclusive in all fourteen lanes. Strip away the audit apparatus and the empirical residue is: on saturated toy tasks nothing separates; on one unsaturated task GRPO beats an untuned critic by a test-sensitive margin; larger groups lower a descriptive statistic without improving accuracy; and a 40-step adapter scores some numbers through harnesses where most failures are parsing, patching, or truncation artefacts. A sceptical examiner could therefore argue the project optimised the wrong thing: 206 analysis iterations and 317 pages spent proving, with increasing rigour, that the experiments as designed could not answer their questions — when a fraction of that effort spent on one adequately powered, item-paired, unsaturated comparison with a tuned baseline would have produced a smaller thesis with one real answer. The defence — that exposing exactly this is the contribution — is genuine, but it converts the thesis into a cautionary tale about itself, and cautionary tales must still show the counterfactual done right at least once. This one never quite does.

#### Issue List

**CRITICAL**

- **DA-CRITICAL-1 — Claims stronger than evidence tier in the abstract.** *Dimension:* claim calibration. *Location:* `frontmatter.tex:197-229`. The abstract foregrounds +5.0 points (GRPO vs untuned PPO), campaign scores, and C1's 11/64, each of which the body downgrades to suggestive / harness-bound / selection-bound. Abstract-first readers (and citation practice) will propagate the strong versions.
- **DA-CRITICAL-2 — Circular / self-referential validation.** *Dimension:* validation independence. *Location:* registry validated on its own 20 seed entries written to its own schema (`ch07_results_infra.md:33-37`); ZVF null correlation r=0.92 computed against a null derived from the same step's mean reward (`ch06_results_core.md:75`); number audit's 501 "P" passes check tokens against the project's own printed sources (`PUBLIC_CHECKOUT_NUMBER_AUDIT.json` summary). Consistency of a closed system is repeatedly at risk of being read as correctness.
- **DA-CRITICAL-3 — Reproducibility receipts do not bind to claimed runs.** *Dimension:* provenance. *Location:* `ch09_results_campaign.md:19,43`; `ch_back_reproducibility.md:96-100`. The campaign's central artefact (actor training) has no recoverable execution source, no consumed-row inventory, and withheld adapter revision; six recovered files live in a private archive absent from the frozen checkout. The receipt chain proves files match hashes, not that the described training produced the evaluated weights.

**MAJOR**

- **DA-MAJOR-1 — Null/negative results over-read as method insight.** *Location:* §9.1 synthesis (`ch10_synthesis_conclusions.md:7-17`). Repeated nulls at ceiling/underpowered designs are synthesised into "measurement layer is useful while algorithmic levers are noise-limited" — a general claim the designs cannot support beyond these stacks and budgets.
- **DA-MAJOR-2 — Salami/compendium overlap.** *Location:* P1–P8 internal reports repackaged as chapters with shared exhibits (17× swing and DAPO pair motivate both P5 and P6; `ch02_literature.md:35`, `ch07_results_infra.md:11,27`). From this text alone, the same evidence carries multiple "contributions"; the thesis should state the publication plan and de-duplicate exhibits, or examiners may read breadth as inflation. No sibling *published* papers are visible from this text, so misconduct is not alleged — only double-counting risk within the compendium.
- **DA-MAJOR-3 — Single-actor, single-institution, short-horizon external validity.** *Location:* `ch10_synthesis_conclusions.md:49-65`. 30–50-step horizons, one 40-step campaign actor, GSM8K/arithmetic-dominated RLVR, no preference-based RLHF. Conclusions must not travel to long-horizon reasoning or production post-training; the thesis mostly honours this, the title does not.

**MINOR**

- **DA-MINOR-1 — Fraud appendix (P8) is off-thesis.** Synthetic data, SFT not RL, AUCs on different splits (0.7955 vs 0.4827 chance-level; `ch08_results_fraud.md:22-31`). Harmless if clearly quarantined; currently it consumes examiner goodwill. Cut to one page or remove.
- **DA-MINOR-2 — Figure/table legibility risk (inferred, not fully verified).** 25 TikZ figures were not rendered-checked by this panel beyond one `.tex` head; the preamble's own comment records a history of off-page ink (`preamble.tex:80-83`). The quality receipt claims all 317 pages pass text-bounds checks — trust but verify at viva with the printed PDF.

#### Ignored Alternative Explanations

1. High ZVF may be *optimal* behaviour late in training (mastered distribution) rather than waste — the thesis notes the aliasing but still frames ZVF as "starvation/dark compute" (`ch01_introduction.md:23`). The normative framing outruns the unsigned statistic.
2. The unsaturated GRPO>PPO gap may reflect prompt-exposure (PPO's 64 distinct prompts dilute per-prompt learning) or cap interactions, not the baseline estimator.
3. Campaign score variation across lanes may be dominated by harness/serving quality (stop sequences, patch formatting, token caps) rather than suite difficulty or actor capability — Table 8.2 makes this plausible but no lane isolates it.

#### Missing Stakeholder Perspectives

- **Practitioners** who must choose G, framework, or reporting burden today: the thesis gives them diagnostics without a single validated decision rule (controller unevaluated, badge undiscriminating, G optimum unresolved).
- **Benchmark owners** whose reference errors C1 alleges (16/64 wrong): no upstream engagement or errata submission is recorded; GSM8K-Platinum is cited (`ch02_literature.md:41`) but C1's cases are not cross-checked against it in the opened text.
- **Human raters:** every judgement layer (C1 reconciliation, V2 audit, Jev triage) is model-assisted; no human annotation budget was spent even where 11 cases would have sufficed.

#### Observations (Non-Defects)

- The sheer errata table (Appendix E) and the decision to preserve the erroneous E14 note unchanged while correcting its reading are model research integrity, not defects.
- Reporting cost/carbon estimates with dominant-uncertainty disclosure (`ch10_synthesis_conclusions.md:67`) exceeds field norms.
- The no-pooling rule costs the author a headline number; that restraint should be credited in the viva.

#### So-what test

If every number in this thesis is taken at face value, what should the field do differently on Monday? Defensible answer: log per-group rewards, publish loss-form/stack manifests, never pool replacement scopes, and distrust single-seed saturated-task comparisons. That is a real, if modest, contribution to practice. What the field cannot take away: whether GRPO beats PPO, what G to use, whether ZVF predicts failure, or whether the campaign actor is any good. A thesis whose so-what is entirely about *how to report* must be examined as a metrology thesis — and judged on whether its instruments are validated. On that standard, ZVF-as-measurement passes; ZVF-as-diagnostic, the controller, and the badge do not yet.

---

## 3. Editorial Decision Letter

**Decision: Major Revision** (journal scale) — equivalently **Major corrections** on the thesis scale (not Pass, not Minor corrections, not Revise-and-resubmit/Reject).

Dear Candidate,

The panel is unanimous that the work is honest, technically literate, and substantially your Semester-4 contribution as declared, and unanimous that it cannot pass without major corrections. The disagreement was only about severity: R2 and R3 would accept the measurement/governance contribution once reframed; R1 and the Devil's Advocate stress that RQ2 remains unanswered and the controller/badge unvalidated; the EIC stresses that the artefact as packaged explicitly declines to certify itself for submission. All seats agree the defects are correctable without new large-scale training — hence Major corrections rather than Revise-and-resubmit.

**Adjudication of Devil's Advocate CRITICAL issues (each addressed visibly):**
- *DA-CRITICAL-1 (abstract overclaim) — VALIDATED.* The abstract must be rewritten to body-tier strength (exact p-values first, harness qualifiers attached, C1 as audit not rate). Blocks any pass until fixed; fix is editorial.
- *DA-CRITICAL-2 (circular validation) — VALIDATED in part.* Recomputation from stored tensors and the design-cluster bootstrap are genuinely independent checks and survive; registry-on-own-entries and same-step null correlations do not validate what their surrounding prose implies. Requires reframing, not new experiments, except for the registry where an external-corpus check or explicit "schema proposal only" downgrade is required.
- *DA-CRITICAL-3 (receipts do not bind to runs) — VALIDATED and unresolved on current evidence.* The dirty source and consumed rows may be unrecoverable; the panel does not require the impossible. It requires that every dependent claim be permanently labelled "stored-evidence, not replayed / not publicly verifiable," and that the 120/970 audit gap be disclosed wherever reproducibility is claimed. This blocks *Accept/Pass* for a reproducibility thesis until the labelling is complete; it does not require rerunning the campaign.

No CRITICAL issue is rejected; none, once honestly labelled, requires rejecting the thesis outright, because the core measurements (ZVF recomputation, loss-form audit, governance ledger) stand independent of the unrecoverable actor provenance.

**Three separate judgements, as required:**
- *Scientific validity:* Sound where claims are bounded as the body (not abstract) bounds them. No fabricated or irreproducible headline was found; several historical claims are correctly withdrawn. Invalid only if read at abstract strength.
- *Venue (degree) fit:* Fits an M.Tech as a measurement/systems thesis *if* reframed as such; as an algorithms thesis it would fail for lack of a demonstrated improvement. The fraud appendix does not fit and should be cut or quarantined.
- *Submission readiness:* Not ready. Independent review pending, AI declaration unconfirmed, collaborator authorisation unresolved, institutional front matter absent (EIC-W2/W3).

We look forward to a corrected thesis that claims exactly what its excellent audit trail supports — no more, and, importantly, no less.

---

## 4. Revision Roadmap (source-ordered, non-ranking)

Every weakness ID in manuscript order, with Must-fix / Should-fix and the minimum remedy. No work order is implied.

| ID | Tag | Minimum remedy |
|---|---|---|
| DA-CRITICAL-1 | Must-fix | Rewrite abstract to body-tier claims; exact tests first; harness/selection qualifiers inline. |
| EIC-W2 | Must-fix | Independent review of exact bytes; confirm AI declaration vs PES policy; institutional front matter or explicit review-copy status. |
| EIC-W3 | Must-fix | Countersigned contribution matrix; resolve/record collaborator authorisation. |
| R2-W1 | Must-fix | Terminology box mapping each study to clipped-GRPO / REINFORCE group-mean / unknown; use in all headlines. |
| R1-W3 | Must-fix | Demote P2 to instrumentation; quarantine mixed-pool correlations as withdrawn; no prognostic language without measured-only held-out validation. |
| R3-W1 | Must-fix | Disclose 828/22/120 audit split in abstract/limitations; controlled-access deposit or "not publicly verifiable" labels in Table 1.1. |
| DA-CRITICAL-3 | Must-fix | Label all actor-dependent results "stored-evidence, not replayed"; state unrecoverable items once, prominently. |
| R3-W2 | Must-fix | Abstract campaign scores carry stored-evidence + decontamination-scope qualifiers. |
| EIC-W1 | Should-fix | One-table claim map after abstract; move iteration-level audits to supplement; mark examinable vs archival chapters. |
| R1-W1 | Must-fix | Saturated contrasts → pilot only; RQ2 rests on unsaturated rerun + powered item-paired future design. |
| R1-W2 | Must-fix | Demote/rename untuned-PPO contrast; exact p first in abstract and §6.5.5. |
| R1-W4 | Should-fix | Per-contrast budget table (prompts/completions/tokens/steps); ban cross-budget synthesis comparisons. |
| R1-W5 | Should-fix | Collapse §6.5.3 to adjusted-p exploratory table; remove narration of non-surviving effects. |
| R2-W2 | Should-fix | Retitle "theory" → model/accounting; self-contained derivation appendix for viva. |
| R2-W3 | Must-fix | Controller stated as unevaluated training policy; plug-in tautology in one paragraph; ranking detail to appendix. |
| R2-W4 | Should-fix | Purge any backend-effect reading of 17× exhibit from synthesis. |
| R3-W3 | Must-fix | Harness-score labelling with failure rates at every first mention; E14 denominator erratum. |
| R3-W4 | Must-fix | Rename flip-risk → declared-difference codes; external validation or schema-only downgrade. |
| R3-W5 | Must-fix | Blind human rating of 11 C1 event cases; C1 framed as reference/parser audit. |
| DA-CRITICAL-2 | Must-fix | Distinguish independent recomputation from closed-system consistency wherever validation is claimed. |
| DA-MAJOR-1 | Should-fix | Scope §9.1 synthesis claims to tested stacks/budgets. |
| DA-MAJOR-2 | Should-fix | State P1–P8 publication/de-duplication plan; single-use each exhibit. |
| DA-MAJOR-3 | Should-fix | Title/abstract scope limiters (short-horizon, verifiable-reward, small/mid models). |
| DA-MINOR-1 | Should-fix | Cut Appendix F to ≤1 page summary or remove. |
| DA-MINOR-2 | Should-fix | Viva print check of all figures/tables in final PDF. |

---

Acronym check (advisory): did not run in this panel execution.
