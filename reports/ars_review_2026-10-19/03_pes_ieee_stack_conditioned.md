# ARS Academic Paper Review — Full Panel (v1.11.1, mode=full)

**Manuscript:** Stack-Conditioned Measurement of GRPO Post-Training: Signal Diagnostics and a Fourteen-Lane Evaluation Ledger
**Manuscript path:** `papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.tex`
**Date:** 2026-10-19 | **Mode:** full | **Panel:** Journal-Fit (EIC), R1 Methodology, R2 Domain, R3 Perspective, Devil's Advocate (fixed 5th seat)

## Disclosures

- Single-agent-per-paper role separation is **not** independent error processes. All five seats were executed by one agent in one panel execution; seat disagreement is structured role-play, not independent review.
- All seats use the **same model family**. Correlated blind spots (e.g., shared priors about GRPO literature, shared reading errors) cannot be ruled out.
- **NOT_CALIBRATED.** No calibration set, inter-rater history, or decision-threshold validation exists for this panel. Severity labels are ordinal judgments, not measured error rates.
- **criteria_binding_unavailable.** Target venue is inferred only from `\documentclass[conference]{IEEEtran}` (tex:1) and the directory name `pes_ieee_attribution`. No author-confirmed venue, page limit, or review criteria were bound to this execution. All venue-fit judgments are provisional on that inference.
- This review is read-only on the manuscript. No manuscript file was edited.

## Coverage statement

Files opened in full and cited below only from bodies actually read:

- `papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.tex` — full 421 lines. There are **no** `\input`/`\include` statements; the single figure (`fig:gu`, tex:239–258) is inline PGFPlots and the two tables (`tab:campaign`, tex:335–356; `tab:lanes`, tex:358–387) are inline. Nothing was sampled or skipped in the manuscript body.
- `papers/pes_ieee_attribution/refs.bib` — full 416 lines, 48 entries. All 48 cited keys resolve; there are no undefined citations and no uncited entries (verified by key set comparison).
- `papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.blg` / `.log` — build warnings checked (one BibTeX warning: empty booktitle in `schulman2017proximal`; font-shape substitutions and underfull boxes only otherwise). The compiled PDF was **not** visually inspected; layout claims below rest on source only.
- `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md` — full 285 lines (the 19 September ledger the manuscript's §IV-F reports).
- `outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json` — full file (11/11 PASS record).
- `platform_hybrid/experiments/results/groupsize_zvf_sweep.json` — summary block and run structure (source named in `fig:gu` caption, tex:256).
- `tools/review_ledger/PES_IEEE_Attribution_recompute.py` — full script read and **executed**; output `platform_hybrid/experiments/results/review_2026-10-18/PES_IEEE_Attribution/recompute.json` inspected.
- Sibling-paper titles only (for overlap scoping, bodies not opened): `papers/acl_arr_oct2026/paper_camera.tex`, `papers/icicc2027/paper_camera.tex`, `papers/rait2027/paper_camera.tex`, `papers/spectral_entropy_grpo/spectral_entropy_paper.tex` — title lines via grep. Overlap findings below are therefore flagged as title-level risk, not proven text reuse.

Not opened (Unresolved): the 368-run W&B export (tex:227–237), per-seed data for the variant (tex:285–290) and length (tex:310–315) contrasts, 26 Sept. rerun data (tex:393–398), actor training code/prompts (unrecovered, tex:200–203), and per-lane receipts behind LAB-Bench 450/1,967 and AgentDojo 88/97.

Traces that close (panel recomputation): sweep GU/ZVF + SDs (`fig:gu`) exact from `groupsize_zvf_sweep.json` (ZVF 0.838/0.764/0.691/0.631); Phase-1 ZVF 0.706–0.770, all-correct 0.647–0.711, trigger 1,867/1,723 exact from tensors; PPO-vs-GRPO Δ=−0.002, t₄=−1.0, p=0.374 exact (CI recomputed [−0.0076,+0.0036]); VerilogEval 129/312, SWE-bench 2/731, Omni numerator 2,271 via ledger + code check. All `tab:campaign` divisions correctly rounded.

---

## 1. Phase 0 — Field analysis

**Primary discipline:** Machine learning — reinforcement learning for LLM post-training (RLVR/GRPO). **Secondary discipline:** Empirical software/measurement science for ML — reproducibility, benchmark governance, and evaluation auditing. The paper's centre of gravity is the secondary discipline wearing primary-discipline clothes: no new estimator is proposed (tex:130–131, "We do not propose a new estimator"), and every capability claim is explicitly disclaimed (Abstract, tex:47–49).

**Paradigm:** Post-positivist measurement-and-audit.

**Methodology type:** Observational telemetry + small matched experiments + archival audit: tensor recomputation (§IV-A), 12-run sweep and 368-run log summary (§IV-B), offline replay (§IV-C), two paired contrasts (n=5, n=8; §IV-D), post-hoc length analysis (§IV-E), 14-lane ledger (§IV-F). Inference is confined to the paired contrasts (tex:185–193); the rest is descriptive — the paper's strongest discipline, and the reason most of it carries no inferential weight.

**Inferred target venue/tier:** IEEE conference (IEEEtran conference class, two-column, ~6 pages of source). The author block (PES University, M.Tech project stated at tex:90) and the directory name suggest an institutional/regional IEEE conference or a PES-affiliated venue rather than a selective IEEE flagship (e.g., ICML/NeurIPS-adjacent workshops would not use IEEEtran). Tier inference: **low-to-mid selectivity IEEE conference**.

**Maturity:** Late draft, pre-submission. The text is unusually self-critical for its tier — confounds, ceilings, missing records, and non-claims are stated inline rather than buried.

### Reviewer Configuration Cards

**Journal-Fit (EIC)** — IEEE conference editor. Lens: scope/contribution-type admissibility, page economy, practitioner takeaway. Penalises: honest-but-empty abstracts (tex:47–49), grey-source bibliography bulk. Rewards: reproducibility value.

**R1 Methodology** — ML statistician. Lens: replication unit, power/MDE, ceilings, selective bins, denominators. Penalises: n-less correlations. Rewards: seed-as-unit (tex:186), BH family (tex:189–190), non-equivalence language (tex:192–193).

**R2 Domain** — RLVR practitioner. Lens: estimator fidelity vs Eq. 1, ZVF placement vs DAPO/GRESO/RL-ZVP, harness-vs-model failures. Penalises: undefined labels. Rewards: tie-probability baseline; Eq. 3 caveat (tex:177–181).

**R3 Perspective** — evaluation-governance scholar. Lens: beneficiaries, incentives, absent stakeholders (owners, vendor, downstream users). Penalises: 14-lane breadth without grades. Rewards: no pooling of scopes (tex:203–205) or table rows (tex:336–339).

**Devil's Advocate (fixed 5th seat, no card).** Mandate: strongest counter-argument, evidence-tier inflation, circular validation, salami/overlap, receipt-to-run binding.

---

## 2. Seat reports

### 2.1 Journal-Fit (EIC)

**Summary judgment.** A well-written audit in search of the right track. As a research-track paper (four contributions, tex:92–103) it over-promises — the harness collapses to two back-ends (tex:93–96), most lanes never produced a grade. As a measurement report it is far stronger. Everything hinges on whether the venue admits negative/measurement results (criteria_binding_unavailable).

**Strengths (anchored).**
- S-EIC-1. The abstract models honest reporting: inconclusive contrasts, missing comparator, and non-claims in 15 lines (tex:34–49).
- S-EIC-2. Page economy is good: one figure, two tables, equations (1)–(3), no appendix bloat. `tab:lanes` (tex:358–387) is exactly the artefact a programme committee wishes more benchmark papers included.

**Weaknesses.**

- **EIC-W1 — Contribution/venue mismatch (contributions list, tex:92–103).** The four bullets promise a harness, a diagnostic + falsifiable model, contrasts + reporting standard, and a campaign audit. The body delivers: a harness with 2/4 back-ends executed (tex:142–144), a diagnostic whose model predictions are (i) supported only by selectively binned logs, (ii) mixed, (iii) a single scalar (tex:229–237), inconclusive contrasts (tex:288–290), and a ledger with 3 strictly complete scopes (tex:322–325). *Fix:* rewrite the bullets to claim the audit methodology and the minimum reportable stack as the contributions; demote the harness to infrastructure; state the paper type (measurement/audit) in the introduction's first paragraph.
- **EIC-W2 — No actionable residue for the practitioner audience (Conclusion, tex:409–416).** The conclusion is entirely negative ("No controller benefit or post-training capability gain is established"). The one forward recommendation — "an asymmetric, cost-aware trigger is the next design target" (tex:270–271) — is undeveloped. *Fix:* add 3–4 concrete, evidence-bound practice rules (e.g., minimum seeds given observed MDEs; when ZVF telemetry changes a decision; what to log before launch so a ledger like `tab:lanes` is possible).
- **EIC-W3 — Bibliography spend vs. evidence (refs.bib; `tab:lanes` citations tex:371–384).** 17 of 48 references exist solely to name lanes that are closed, blocked, or partial, including six 2026-dated grey sources (`lau2026bankertoolbench`, `vidgen2026apexagents`, `liu2026lifescibench`, `emulated2026sdab`, `quesma2026binaryaudit`, `proximal2026frontierswe`) with "accessed 18 October 2026" notes. *Fix:* move lane provenance to a footnote/URL list; keep archival citations for lanes with graded results.
- **EIC-W4 — Missing submission apparatus (whole manuscript).** No data/code availability statement, no compute/cost statement, no acknowledgements/funding, no conflict note re the closed Tinker platform. *Fix:* add an availability paragraph naming the exact artifacts/scripts/versions that reproduce each table/figure, or state plainly what will not be released.

### 2.2 R1 Methodology

**Summary judgment.** The statistical protocol (§III-D, tex:185–193) is better than most GRPO papers: seed as unit, paired tests with df, MDEs from the noncentral t, a declared BH family, and "inconclusive ≠ equivalent" throughout.

**Strengths (anchored).**
- S-R1-1. Confining inference to matched contrasts and labelling single-seed/logged/offline material descriptive (tex:191–193) is correct and rare.
- S-R1-2. `fig:gu` shows seed SDs, refuses a fitted law, and warns that G changes the rollout budget (tex:253–256). This panel verified every plotted value against `groupsize_zvf_sweep.json`.

**Weaknesses.**

- **R1-W1 — Identical Phase-1 arms (§IV-A, tex:210–216; §IV-C pooled counts tex:263–266).** The paper reports four method labels over 640 groups each. Independent recomputation from the stored tensors (this panel, executing the repo's recompute script) yields **bit-identical summaries for GRPO and AERO**: mean step ZVF 0.7203, 461 degenerate groups, 427 all-correct, 34 all-wrong — identical on every statistic, while GIFT (0.7703) and AReaL (0.7063) differ. *What is wrong:* the "four method labels" panel may contain only three distinct runs, and the pooled 2,560-observation trigger audit double-counts one run. *Fix (minimum):* disclose the dispatch mapping per label, hash-compare the tensor files, and either correct the panel to distinct runs or relabel it as repeated measurement of fewer implementations. Until resolved, no cross-method reading of §IV-A/C is possible.
- **R1-W2 — Ceiling-bound primary contrast (§IV-D, tex:278–284).** PPO vs GRPO is run at 0.98–1.00 held-out accuracy on synthetic arithmetic with a 0.5B model, n=5 seeds. The 95% CI [−0.008, +0.004] is narrow only because there is nowhere to move; the design could not have detected any plausible algorithm effect. The paper says "failure to detect … not evidence of equivalence" (tex:281–282) — correct, but then the contrast contributes nothing beyond illustrating the protocol. *Fix:* either rerun one contrast at interior accuracy (the paper already has an interior regime: 0.67–0.71 sampling cohort, tex:217–219) or explicitly retitle §IV-D a protocol demonstration. Do not leave it carrying "stack is part of the result" weight.
- **R1-W3 — Selective bins and missing runs in the U-shape exhibit (§IV-B, tex:229–233).** Of 368 logged runs (350 finished), the bins account for 207 + 52 + 10 = 269. The remaining ~81–99 runs (reward in (0, 0.1) ∪ [0.6, 0.9), plus unfinished runs) are unreported, and the "interior" evidence for prediction (i) rests on **10 runs** spanning ZVF 0.12–0.38. Bin edges are neither pre-registered nor justified. *Fix:* report the full binning (or a scatter/quantile table) over all finished runs with counts per bin; state how unfinished runs were handled.
- **R1-W4 — Correlation without n (prediction (iii), tex:234–237).** "Pearson r=+0.71 … the number of observations is not logged … directional only." A correlation with unknown n, over autocorrelated steps (which §III-D itself disqualifies as independent observations), should not appear as a numbered prediction test. *Fix:* remove prediction (iii) from the model test, or recompute it from tensors with n, CI, and step-dependence handling. As written it invites citation as "r=0.71" stripped of its caveat.
- **R1-W5 — Variant contrast reported without per-seed data or CIs (§IV-D, tex:285–290).** Means, paired differences, raw/BH p-value floors ("all raw p≥0.21; BH-adjusted p≥0.84"), and MDE range are given, but no per-contrast CI, no per-seed table, and no exact p-values. *Fix:* add a compact per-seed table (8×5) in text or supplement and exact statistics per contrast.
- **R1-W6 — Abstract/body count mismatch on completed lanes (Abstract tex:45–46 vs §IV-F tex:322–331).** Abstract: "only five lane scopes produced complete graded results." Body: three strictly complete + Omni-MATH "terminal-complete with two judge-excluded rows" + SWE-bench Pro "full-suite result whose 731-task denominator includes 18 tasks without a native evaluation." Five is reachable only by counting the latter two as "complete graded," which their own qualifiers contradict. `tab:campaign` lists 5 lanes (6 rows counting Omni twice). *Fix:* use one taxonomy everywhere (strictly complete / terminal-complete / full-suite-with-denominator-note) and make the abstract use it.
- **R1-W7 — Length-bias section tests nothing (§IV-E, tex:305–318).** By the paper's own account the flag is post hoc and tests reward not length, and generation is right-censored at 200 tokens with starts near the cap so "this configuration cannot test unconstrained length inflation." *Fix:* cut §IV-E to a threats paragraph; reclaim the space for R1-W5's table.

### 2.3 R2 — Subject-matter expert (cross-stack ZVF & contrast)

**Summary judgement.** A forensic ledger claiming no training benefit. "Cross-stack" overstates the archive: no crossed run replicates task/reward/seed across stacks with tensors intact, so stack contrasts are case studies. Major Revision if the venue accepts measurement audits.

**Strengths (anchored).**
- S-R2-1. Tensor-backed ZVF recomputation is real. Where Phase-1 tensors survive (4 policies × 640 groups), recomputed mean ZVF 0.706–0.770 and 1,867/2,560 zero-variance groups (1,723 all-correct, 144 all-fail) match the text to rounding (tex:213–215, 264–266; `variance_mitigation_*` files in `results/`).
- S-R2-2. Group-size sweep is the cleanest controlled contrast: same Qwen2.5-0.5B arithmetic stack, G∈{2,4,8,16} × 3 seeds, recomputed GU means 0.162/0.237/0.309/0.369 with ±1 SD matching Figure 1 (tex:223–226, 249–250). The direction — larger G recovers variance-bearing groups but does not eliminate zero-variance — is supported at the level of these 12 runs.
- S-R2-3. Null discipline in §IV-D: same-stack 8-seed contrasts (GRPO/DAPO/GSPO/Dr. GRPO/AERO within 0.631–0.638; raw p≥0.21, BH p≥0.84, MDE 0.010–0.015 vs design margin 0.010) are correctly reported as inconclusive rather than equivalence (tex:285–290).

**Weaknesses.**

- **R2-W1 — "Cross-stack" ZVF is an ecological comparison, not a controlled stack contrast (§IV-B/C/D).** Each stack bundle varies model, task, G, horizon and reward simultaneously: Phase-1 (Qwen3.5-4B GSM8K G=8), sweep (Qwen2.5-0.5B arithmetic G∈{2,4,8,16}), W&B zvf-audit (368 runs, 9 models, G∈{4,6,8,16}, 2–12 steps), same-stack GSM8K arm (Qwen3-8B G=4 30 steps), long-horizon replay (Qwen3-4B-Instruct). No task is replicated across two independent stacks. *What is wrong:* any "stack effect" is fully confounded with model/task/horizon; the W&B U-shape (ZVF 0.959 at R=0, 0.881 at R≥0.9, 0.12–0.38 at R∈(0.1,0.6)) largely restates the h_G(p̄) algebra for heterogeneous p. *Fix:* retitle the claim to "manifest-bound audit entries differ," add an explicit confounding table (stack × model × task × G × horizon), and remove causal verbs ("stack drives," "stack-conditioned measurement" as finding rather than method). The DAPO label-collision (0.00 vs 0.58) and Tinker-vs-TRL 0.856 vs 0.050 exhibits (tex:291–296) are suggestive but confounded by checkpoint (Qwen3-8B-Base vs Qwen3-8B) and closed internals — keep them as exhibits, not estimates.
- **R2-W2 — The headline E1–E14 ledger measures an undisclosed intervention (§III-E/§IV-F).** One Tinker LoRA actor (Qwen3.6-35B-A3B, rank-32, 40 steps, G=4, 2 prompts/step, 80 draws from 512 API-Bank/SWE-Gym, LR 1e-5, temp 0.7, seed 809; reward zero on 22/40 steps; 80 drawn prompts and executed training code "not recovered") is evaluated on 14 lanes, yet only 5 scopes yield complete graded results, 6 later replacements run under "protocol override" without closing original contracts, and 7 originals remain externally closed/blocked (tex:196–205, 322–333). No full-suite base-model counterpart exists; the Sept-26 paired rerun (5–151 items/lane, min McNemar p=0.25) is the only trained-vs-base test. *What is wrong:* pass rates (VerilogEval 129/312, Omni-MATH 2,271/4,426 judged rows, SWE-bench Pro 2/731) characterise an actor whose training difference from base is unestablished and whose training set cannot be audited for contamination against GSM8K/VerilogEval/Omni-MATH. *Fix:* foreground the paired rerun as the primary training-effect statement; publish the 512-pool composition and decontamination check; add contract-vs-replacement taxonomy so "14-lane" is not read as 14 completed benchmarks; report Omni-MATH consistently (manuscript Table I gives both 2,271/4,426=0.5131 judged and 2,271/4,428=0.5129 all-rows — correct — but the source ledger `E1_E14_FINAL_RESULTS_2026-09-19.md:133–135` mislabels 2271/4428 as 0.5131; state the judged denominator as canonical everywhere).
- **R2-W3 — ZVF as GRPO scalar is under-positioned against reward-design effects.** ZVF is invariant to whether zero-variance arises from saturated mastery, impossible prompts, binary-reward coarseness, low temperature, truncation at max length (e.g., VerilogEval 150/312 extraction failures under 1,024-token cap with thinking enabled, tex:400–402), or checker brittleness. GSM8K answer-format failures and SWE-bench denominator inflation (731 includes 18 tasks without native evaluation, tex:327–329) show harness design dominates several "signal" readings. *What is wrong:* without per-prompt p̂ trajectories, length/truncation covariates, or a comparison to distributional alternatives (advantage std, gradient norm, effective sample size), ZVF's incremental diagnostic value over mean reward is asserted, not shown. The r=+0.71 gradient-norm correlation rests on one run with unlogged n (tex:234–237). *Fix:* add at least one head-to-head where ZVF predicts a training outcome mean-reward alone does not, on the same stack; or downgrade ZVF to bookkeeping identity.
- **R2-W4 — Offline adaptive-G audit cannot support the asymmetric-trigger recommendation (§IV-C).** Replaying the trigger on Phase-1 tensors where 92.3% of zero-variance groups are all-correct guarantees the conclusion; plugging p̂∈{0,1} into h_G forces zero predicted rescue by construction (paper acknowledges this, tex:267–271). The interesting case (all-fail groups where larger G might rescue) is n=144 with no fresh rollouts. *Fix:* require a live A/B with matched rollout budget (including compute-matched larger-G baseline, cf. `p3_isocompute_groupsize.py` in repo) before any trigger recommendation, even a negative one.

### 2.4 R3 — Mechanism, theory and identifiability expert

**Summary judgement.** The iid-Bernoulli identity h_G(p)=p^G+(1−p)^G is correct and the symmetry argument (all-pass equally signal-free under mean-centering) is a useful corrective. Beyond that, the "signal theory" is not identified: S(p,G) is a heuristic product, ZVF adds no parameter beyond p (and prompt-mixture dispersion), and stack-vs-algorithm attribution is unidentified by design. Major Revision to either identify the mechanism or reframe as pure measurement.

**Strengths (anchored).**
- S-R3-1. Eq. 4 derivation and the (i) U-shape / (ii) G-monotonicity implications are clean, falsifiable in principle, and correctly note the estimator dependence (1/G vs 1/(G−1), group vs batch std) that makes "GRPO" stack-bound (tex:67–74, 166–171).
- S-R3-2. Refusal to report unrecoverable historical ANOVAs ("row-level membership … not recoverable," tex:274–277) and to pool original/replacement scopes is methodologically correct identifiability hygiene.

**Weaknesses.**

- **R3-W1 — Zero-variance ≠ zero learning under the paper's own Eq. 1 is overstated as "contributes no reward-driven gradient" (tex:77–82).** True for the centred advantage term with ε>0, but deployed GRPO variants retain KL/entropy/clipping, length penalties, and (in σ-normalised forms) cross-group normalisation through batch statistics; DAPO-style dynamic sampling, Dr. GRPO's removal of σ-scaling, and sequence-level GSPO each change what a uniform group contributes. The Phase-1 panel itself uses a "mean-centred runner without σ_g division" (tex:211–213), so measurements come from a different estimator than Eq. 1 defines. *Fix:* scope every ZVF claim to the exact estimator that produced it; add a table mapping estimator variant → what a uniform group contributes; do not generalise from the mean-centred runner to "GRPO."
- **R3-W2 — S(p,G)=p(1−p)(1−h_G(p)) double-counts and is untested where it matters (§III-C, tex:172–183).** Both factors peak at p=0.5 by construction; under normalised advantages E[Â²] is ≈1 regardless of p (paper notes this) and under mean-centering p(1−p) already *is* the signal. Prediction (iii) (gradient norm ∝ p(1−p)) is tested by a single logged scalar with unknown n. No comparison of S against alternatives ((1−h_G) alone, p(1−p) alone, E|Â|) on held-out outcomes is offered. *Fix:* either derive E‖∇‖ for the two estimators actually used and test it on tensor data with CIs, or present S explicitly as a plotting heuristic with no predictive claim.
- **R3-W3 — Mixture dispersion is the only degree of freedom, but it is never estimated.** With per-prompt p_i, batch ZVF ≈ mean_i h_G(p_i); observed deviations from h_G(p̄) (e.g., interior-cohort ZVF 0.130/0.190/0.155 pooled 0.158 at 0.67–0.71 accuracy, tex:217–220) encode Var(p_i) and within-group correlation (shared prefix, temperature, determinism of the checker). The theory therefore cannot distinguish "prompt too easy/hard" from "sampling too correlated" *Fix:* estimate effective G or intra-group correlation from tensors; report ZVF decomposition into location (p̄) vs dispersion; test whether deviations from iid predict anything.
- **R3-W4 — "Provenance- and manifest-bound attribution" (§VI, tex:409–416) declares non-identifiability and then re-introduces attribution language.** If the uncrossed corpus "cannot allocate effects to stack versus algorithm" (tex:130–131), the contribution is provenance discipline, not attribution. Title and §VI should say so.

## 3. Devil's Advocate (fixed 5th seat)

### Strongest counter-argument

Strip away the ledger scaffolding and this manuscript proves three things nobody disputes, using mostly data nobody can reuse. First, under mean-centered GRPO a constant-reward group has zero advantage — this is arithmetic, visible directly from Eq. 1, and ZVF is simply the frequency with which that triviality occurs. Calling the frequency a "diagnostic" does not make it informative when, by the paper's own Phase-1 numbers, ZVF is 0.706–0.770 for precisely the reason any practitioner would guess: a 4B model on GSM8K at G=8 gets most prompts uniformly right, so of course most groups are signal-free. The same scalar would be high for the opposite reason — a model that uniformly fails — and the paper concedes ZVF "require[s] mean reward for interpretation," which is to say ZVF adds nothing beyond the reward distribution it summarizes. Second, the only controlled training comparisons are null *and* underpowered by the authors' own MDE analysis, so they cannot distinguish "variants are equivalent" from "this harness at 30–40 steps on 0.5B–8B models cannot see variant differences" — yet the framing invites the first reading. Third, the flagship 14-lane campaign evaluates a 40-step, 80-prompt LoRA whose training data are unrecovered and whose only paired test finds no difference from base, on lanes where nine of fourteen never produced a complete grade and the headline verbal result is confounded by 150/312 answer-extraction failures. What remains is an exceptionally honest lab notebook: valuable as a cautionary artefact, but the venue is being asked to publish the *absence* of every effect the title promises to measure. Honesty about a null is not itself a finding, and "stack-conditioned measurement" without a single crossed stack × algorithm cell is a method in search of its first successful application.

### Issue list

**DA-CRITICAL**

- **DA-CRITICAL-1 — Circular/self-referential validation of ZVF (theory–evidence circularity; §III-C/§IV-A–B).** ZVF is validated against quantities computed from the *same* reward tensors it summarises: tensors define ZVF, tensors define p̂, and agreement of ZVF with h_G(p̂) / with the U-shape in final-reward bins is then offered as corroboration. The W&B binning (ZVF 0.959 at R=0 vs 0.12–0.38 at intermediate R) is near-tautological: final reward near 0 or 1 mechanically implies most groups were uniform. No out-of-sample prediction (e.g., ZVF at step t predicting held-out gain, collapse, or gradient magnitude on *future* steps/runs) is demonstrated. Prediction (iii) rests on one logged scalar, r=+0.71, n unlogged (tex:234–237). *Required to clear:* a pre-registered predictive test on held-out runs where ZVF (or S) forecasts an outcome not used in its construction, with CI; failing that, downgrade all "signal" language to identity/bookkeeping.
- **DA-CRITICAL-2 — Flagship campaign evaluates an ineffective, undisclosed intervention (internal validity; §III-E/§IV-F/§V).** 80 prompt draws, 55% zero-reward steps, unrecovered training rows/prompts, no full-suite base comparator, paired rerun null (min McNemar p=0.25, 5–151 items). Item-level training separation "not established" (tex:200–203) means leakage can neither be ruled in nor out for any lane sharing task families with API-Bank/SWE-Gym. The majority of the paper's page budget (Tables I–II, §IV-F, §V) is spent characterising this actor. *Required to clear:* either (a) demote E1–E14 to an appendix-scale governance case study explicitly framed as "ledger mechanics demonstrated on a null actor," with the abstract/title adjusted, or (b) supply a training intervention with a detectable paired effect and auditable training manifest. As written, the ledger's prominence exceeds its evidentiary tier by an order of magnitude.
- **DA-CRITICAL-3 — Null/negative results over-read as stack-attribution evidence (claim–evidence tier mismatch; Title/Abstract/§IV-D/§VI).** Title promises "Stack-Conditioned Measurement of GRPO"; text concedes "the uncrossed corpus cannot allocate effects to stack versus algorithm" (tex:130–131) and "No … independent stack attribution is established" (Abstract). The two same-stack nulls are inconclusive at MDE ≥ design margin; cross-stack exhibits are confounded (different checkpoints, closed internals, dry-run plans for veRL/OpenRLHF). There is therefore *no* cell in the paper where stack is varied holding algorithm/model/task fixed with raw data intact. *Required to clear:* retitle/reframe as a provenance/ledger method paper, or add at least one fully crossed stack × algorithm block. The current middle position — attribution branding, provenance substance — will mislead citation practice even with caveats.

**DA-MAJOR**

- **DA-MAJOR-1 — Reproducibility receipts do not bind to claimed runs (reproducibility; global).** Figure 1 cites `groupsize_zvf_sweep.json` without path/hash/version; Phase-1 tensors, W&B export, per-lane receipts, Tinker run IDs, adapter checkpoint hash, and analysis code are not identified in the manuscript at a level that would let an independent reader re-derive Table I or §IV-D numbers. Managed-platform internals are closed and the VerilogEval sampler path "no longer exists" (tex:403–404). This panel *could* recompute several numbers only because it had repo access — the manuscript alone does not support recomputation. *Fix:* artifact appendix with repo commit, file hashes, run IDs, and a one-command recompute entry point per table/figure; or state explicitly which claims are non-reproducible by design.
- **DA-MAJOR-2 — Potential salami/overlap with sibling papers in this repo (publication ethics/scope; repo-visible).** Sibling manuscripts in the same repository carry titles *The Stack Is Part of the Result: Negative Results on Attributing GRPO-Family Differences* (ACL ARR), *Signal Starvation in GRPO … Zero-Variance Groups* (RAIT), and *Receipt-Bound Evaluation Governance: A Fourteen-Lane … Case Study* (ICICC) — i.e., the three pillars of the present paper (stack attribution, ZVF, 14-lane ledger) each headline a sibling submission. Bodies were not opened, so text/data overlap is *unconfirmed*, but title-level overlap obliges disclosure. *Fix:* include an overlap statement enumerating shared data/tables/text with each sibling and what is unique here; editors should request the siblings for comparison before acceptance.
- **DA-MAJOR-3 — Phase-1 estimator mismatch (construct validity).** Headline ZVF measurements come from a mean-centred runner *without* σ_g division while Eq. 1 / title claim GRPO generally. Variants named in the panel (GIFT, AERO, AReaL — tex:211–212) are never defined in the manuscript. *Fix:* define every label, scope claims per estimator (see R3-W1, R1-W1 on the identical GRPO/AERO arms).

**DA-MINOR**

- **DA-MINOR-1 — Bibliography future-dating and grey-source load** (see EIC-W3): six 2026 vendor/web sources, access dates 18 Oct 2026, year mismatches (`thinkingmachines2024tinker` year 2025; `pineau2020improving` year 2021; `rafailov2024direct` year 2023; `le2025rlzvp` year 2026; `khandpurswemultilingual` no year). Verify or replace before submission; a checker will flag these.
- **DA-MINOR-2 — Undefined/orphan formalism:** PCD (tex:156–163) is defined, distinguished from pairwise disagreement, and never used in any result; GU=1−ZVF is a rename, not a metric. Cut or use.
- **DA-MINOR-3 — Ledger date incoherence:** manuscript ledger is "19 September" with 26/27 September addenda and bibliography accessed 18 October 2026, while benchmark citations claim 2026 venues. Add a campaign timeline so readers can order contracts, overrides, and reruns.

### Ignored alternative explanations

1. **Saturation, not GRPO pathology:** high ZVF is expected under any group-relative method (incl. RLOO, tex:113–115); no non-GRPO control shows specificity. 2. **Harness failure:** extraction failures (150/312), truncation, checker strictness manufacture uniform groups; ZVF cannot separate "model cannot" from "parser will not." 3. **Sampling correlation** shrinks effective G — never estimated. 4. **Short-horizon artefact:** 30–40 steps reflect initial p_i; long-horizon dynamics untested (§IV-E censored). 5. **Archive selection:** crashed/deleted W&B runs are invisible.

### Missing stakeholder perspectives

- **Benchmark owners:** seven "externally closed" lanes reported evaluator-side only; closure reason (policy/contamination/non-response) uncharacterised. **Platform vendor:** closed internals, defunct sampler path, no version pin/escrow — portability unknowable. **Practitioners:** no decision rule with error bars; G cost never quantified per recovered signal (tex:226–227). **Training-pool contributors:** API-Bank/SWE-Gym composition undisclosed; licence/representativeness unassessable (low severity, total gap).

### Observations (non-defects)

- ZVF's cheapness (computable from logged rewards without extra rollouts) is a genuine practical virtue even if its incremental predictive value is unproven; as a *logging standard* proposal it is reasonable.

### So-what test

*If every number in this paper is exactly right, what should a GRPO practitioner do differently on Monday?* Current answer: log ZVF and mean reward, distrust method labels without manifests, prefer larger G if rollouts are free, and do not trust sub-1-point variant differences at ≤8 seeds. The last two are already folk wisdom; the first two are worthwhile but need (a) a demonstrated decision ZVF changes and (b) the 8-item manifest (tex:299–303) validated on an external corpus, not just the authors' archive. The 14-lane ledger, as a governance template, is the most transferable artefact — but the paper buries that contribution under GRPO-branding. Reframed as "how to ledger a heterogeneous evaluation campaign so null actors and closed lanes remain auditable," the so-what is clear. As "stack-conditioned measurement of GRPO," it is not yet earned.

## 4. Editorial decision letter

Dear Authors,

**Decision: Major Revision.**

**Consensus vs. disagreement.** All seats agree that the statistical non-claims are correctly stated, that the tensor-backed ZVF recomputations and the group-size sweep are the strongest evidence, and that the E1–E14 ledger cannot support training-effect or stack-attribution claims. Seats disagree on remedy scale: Journal-Fit and R2 would accept a reframed measurement/governance paper at this tier after major revision; R3 requires either a derived, tested signal prediction or explicit demotion of S(p,G) to heuristic; R1 requires resolution of the identical GRPO/AERO Phase-1 arms before any further consideration. No seat supports Accept or Minor Revision in the current form. No seat recommends outright Reject *if* the venue admits audit/negative-result contributions — but see venue fit below: if it does not, this becomes a Reject on scope, not on execution.

**Devil's Advocate CRITICAL adjudication (visible, per protocol).**
- **DA-CRITICAL-1 (circular ZVF validation): VALIDATED — blocks Accept.** No seat identified an out-of-sample predictive test; R3-W2/W3 and R1-W4 corroborate. Cleared only by a predictive test on held-out runs or by downgrading ZVF/S to bookkeeping identities throughout (title, abstract, §III-C, §VI).
- **DA-CRITICAL-2 (null undisclosed actor carries flagship): VALIDATED — blocks Accept.** Corroborated by R2-W2 and conceded in tex:200–203, 391–398. Cleared by demoting/reframing the ledger as governance mechanics on a null actor (with title/abstract change) or by a training intervention with auditable manifest and detectable paired effect.
- **DA-CRITICAL-3 (attribution branding without a crossed cell): VALIDATED — blocks Accept.** Corroborated by R2-W1, R3-W4, and the manuscript's own tex:130–131. Cleared by retitling/reframing to provenance/ledger method or by adding a crossed stack × algorithm block. No DA CRITICAL is rejected; none is left merely "noted."

**Scientific validity (separate from readiness).** *Partially valid within narrow scope.* Verified by this panel's recomputation: Phase-1 ZVF levels/composition, sweep GU trend, PPO-vs-GRPO null statistics, VerilogEval/SWE-bench numerators, Omni-MATH numerator. Not valid as stated: any stack-vs-algorithm allocation (unidentified), any training benefit (null/undisclosed actor), prediction (iii) (n-less scalar), Phase-1 cross-method reading (identical arms, R1-W1). The paper's own caveats cover most of these; validity fails mainly where title/contribution framing exceeds the caveated body.

**Venue fit (separate).** *Uncertain — criteria_binding_unavailable.* Inferred IEEE-conference tier: fit is plausible **only** as a measurement/reproducibility contribution and only after reframing (EIC-W1). As a GRPO research contribution it does not fit any selective venue, since it proposes no estimator and establishes no effect. Authors must confirm the target track accepts audit/negative-result papers; this panel cannot.

**Submission readiness (separate).** *Not ready.* Independent of the scientific reframing: unresolved identical arms (R1-W1), undefined method labels (GIFT/AReaL; AERO defined only as a citation), missing per-seed data for §IV-D variants, no artifact binding (DA-MAJOR-1), no overlap disclosure for sibling submissions (DA-MAJOR-2), future-dated/grey bibliography items, and no availability statement. Each is fixable without new training runs except where the roadmap says otherwise.

We encourage resubmission after major revision along either of the two coherent paths above — (A) provenance/ledger method paper (no new training required; reframing + artifact binding + R1-W1 resolution), or (B) GRPO measurement paper with a crossed block and a predictive ZVF test (new runs required). Path A is the panel's recommended route for the inferred venue.

## 5. Revision roadmap

Non-ranking; source order (manuscript order, then cross-cutting). Tags indicate the minimum for re-review, not importance order.

| ID | Tag | Minimum remedy |
|---|---|---|
| DA-CRITICAL-3 / R3-W4 (Title) | Must-fix | Retitle/reframe to provenance/ledger method, or add a crossed stack×algorithm block. |
| R1-W6 (Abstract) | Must-fix | One lane-completion taxonomy in abstract, §IV-F and tables; correct "five … complete." |
| EIC-W1 (§I) | Must-fix | State paper type (measurement/audit); rewrite contribution bullets to match delivered evidence. |
| R2-W1 (§I/§IV) | Must-fix | Confounding table (stack×model×task×G×horizon); remove causal stack language. |
| R3-W1 (§I/Eq.1) | Must-fix | Per-estimator table: what a uniform group contributes; scope claims to the estimator used. |
| DA-MINOR-2 (§III-B) | Should-fix | Use PCD in a result or delete it; drop GU as a named metric or justify it. |
| R3-W2 (§III-C) | Must-fix | Derive+test a gradient prediction with CI on tensors, or demote S(p,G) to heuristic with no predictive claim. |
| R3-W3 (§III-C) | Should-fix | Decompose ZVF into p̄ vs dispersion/correlation (effective G) from tensors. |
| R2-W2 (§III-E) | Must-fix | Publish 512-pool composition + decontamination check; contract-vs-replacement taxonomy; foreground paired rerun. |
| R1-W1 / DA-MAJOR-3 (§IV-A) | Must-fix | Explain/hash-verify identical GRPO/AERO tensors; define GIFT/AERO/AReaL; correct or relabel panel. |
| R1-W3 (§IV-B) | Must-fix | Full accounting of all 350 finished W&B runs; justify bins; report the missing ~81 runs. |
| R1-W4 / DA-CRITICAL-1 (§IV-B) | Must-fix | Remove n-less r=0.71 as a prediction test, or recompute with n/CI/dependence handling + held-out predictive test. |
| R2-W3 (§IV-B) | Should-fix | Head-to-head: ZVF predicts an outcome beyond mean reward on same stack, or downgrade to bookkeeping. |
| R2-W4 (§IV-C) | Should-fix | No trigger recommendation (even negative) without live, compute-matched A/B; label replay as arithmetic only. |
| R1-W2 (§IV-D) | Should-fix | Move ceiling contrast to interior regime or retitle §IV-D a protocol demonstration. |
| R1-W5 (§IV-D) | Must-fix | Per-seed table + exact p/CIs for all four variant contrasts; identify source artifacts. |
| R1-W7 (§IV-E) | Should-fix | Cut length-bias subsection to a threats paragraph (censored design cannot test its question). |
| R2-W2 (Tables I–II) | Must-fix | Consistent Omni-MATH denominator (judged 4,426 canonical); fix source-ledger conflict note; SWE denominator footnote retained. |
| DA-CRITICAL-2 (§IV-F/§V) | Must-fix | Demote ledger to governance case study on a null actor, or supply auditable intervention with detectable paired effect. |
| DA-MINOR-3 (§IV-F) | Should-fix | Campaign timeline (contracts → overrides → reruns → ledger freeze). |
| EIC-W2 (§VI) | Should-fix | Add 3–4 evidence-bound practice rules (seeds/MDE, logging manifest, G cost rule). |
| EIC-W3 / DA-MINOR-1 (refs) | Must-fix | Verify/replace 2026 grey sources; fix year mismatches; move ungraded-lane provenance out of archival refs. |
| EIC-W4 (back matter) | Must-fix | Data/code availability + compute/cost + platform-dependence statements; cite recompute entry points. |
| DA-MAJOR-1 (global) | Must-fix | Artifact appendix: commit, paths, hashes, run IDs, one-command recompute per figure/table; mark non-reproducible claims. |
| DA-MAJOR-2 (global) | Must-fix | Overlap statement vs. ACL ARR / RAIT / ICICC siblings (shared data/tables/text; unique contribution here). |
| DA-CRITICAL-1 (global) | Must-fix | Every "signal" claim either backed by held-out prediction or rewritten as identity/bookkeeping. |

## 6. Attachment

Acronym check (advisory): did not run in this panel execution.
