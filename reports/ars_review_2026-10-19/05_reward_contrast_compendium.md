# ARS Full-Panel Review — Reward Contrast, Not Algorithm Labels

- **Manuscript:** `platform_hybrid/paper/paper.tex` — "Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs"
- **Date:** 2026-10-19
- **Mode:** full (ARS academic-paper-reviewer v1.11.1, one execution seat panel, one manuscript)
- **Editorial decision:** Reject (see Editorial Decision Letter; scientific validity, venue fit, and submission readiness are assessed separately there)

## Disclosures

- Single-agent-per-paper role separation is **not** independent error processes. The five seats below (Journal-Fit/EIC, R1 Methodology, R2 Domain, R3 Perspective, Devil's Advocate) were produced by one reviewer process adopting distinct stances; agreement across seats therefore does not carry the evidentiary weight of independent reviewers.
- All seats share the same model family. Correlated blind spots (e.g., shared misreading of an artifact or shared unfamiliarity with a 2026 preprint) cannot be ruled out.
- **NOT_CALIBRATED.** Scores, severities, and the decision are uncalibrated judgments, not predictions of any venue's outcome.
- **criteria_binding_unavailable.** No author-confirmed criteria binding was supplied. Target venue is inferred only from the document class/style (`neurips_2026`, `paper.tex:12-21`) and in-file comments ("Track: Evaluations & Datasets (recommended) or Main Track", `paper.tex:5`). Venue-fit judgments below are conditional on that inference.

## Coverage statement

Files opened and read as bodies (not merely located by search):

- `platform_hybrid/paper/paper.tex` — read in full, lines 1–1813, in four sequential windows.
- Every direct `\input` of that root, read in full: `sections/abstract.tex` (23 lines), `sections/intro.tex` (27), `sections/related_work_v2.tex` (275), `sections/diagnostic_method.tex` (64), `sections/statistical_evidence_policy.tex` (44), `sections/evidence_identity_audit.tex` (53), `sections/conclusion.tex` (10), `sections/audit_regression_cases.tex` (45), `sections/checklist.tex` (83), and `ethics_statement.tex` (427). No further nesting was found in these files beyond the root's own includes.
- `references.bib` — machine-checked in full: 261 entries; 153 distinct keys are actually cited from the files above; 0 cited keys are missing from the bibliography; 108 entries are never cited. Individual entries were read for the load-bearing 2026 diagnostic citations (e.g., `he2026advantage`, `nie2026gradstarvation`, `zhang2026aero`, `kaddour2026tpo`, `li2026oxrl`); the remaining entries were not each read as bodies.
- Figures/tables as text: all LaTeX tables and figure captions in the files above were read. Rendered figure binaries (`tikz/*.pdf`, `figures/wave6_sensitivity.png`, `figures/v2/kl_proxy.pdf`, `figures/v2/zvf_heatmap`, `figures/v2/old_trl_seeds.png`) were **not** visually inspected; figure-content claims are assessed via captions and underlying data files only.
- Repo artifacts opened to trace headline numbers: `platform_tinker/reports/final/gsm8k_heldout_seed{042,137,256,512,999}.json` and `gsm8k_base_control_200.json` (summaries recomputed); `platform_hybrid/experiments/zvf_lagged_regression_results.json`; `platform_hybrid/experiments/tinker-runs/results/wave6_ablations.json` (all three sweeps); `platform_hybrid/experiments/master_results.json` (campaign_v2_w2 rows and spot rows); `platform_hybrid/experiments/tinker-runs/results/cross_tool_qwen3-32b.json`, `cross_tool_llama-8b-inst.json`, `frontier_gsm8k_{nemotron-120b,deepseek-v3.1,qwen3-235b}.json`; the loss-function region of `platform_hybrid/experiments/tinker-runs/scripts/grpo_exp_a_baseline.py` (lines around `grpo_loss_fn`, inspected via targeted search, not a full-file read).

Sampling / skipped material, disclosed: the ~200 additional files under `platform_hybrid/paper/sections/` that `paper.tex` does **not** input were listed but not read; they are not part of this manuscript. Sibling manuscripts `paper_P1*.tex`–`paper_P7*.tex` were not read beyond their `\title` lines (used only for the overlap observation, which is framed accordingly). Ledger reconciliation against `master_results.json` was a targeted spot-check (group-size block, Wave 6, frontier rows, one scale row), not an exhaustive row-by-row audit of all 95 experiment records. No compiled `paper.pdf` for this root exists in the workspace, so page count and typeset figure legibility could not be verified. External existence of cited 2025–2026 preprints was not verified against publisher/arXiv records in this panel.

---

## Phase 0 — Field analysis

- **Primary discipline:** Machine learning — reinforcement-learning post-training of large language models (group-relative / critic-free policy gradient, verifiable rewards).
- **Secondary disciplines:** Empirical ML methodology and reproducibility/artifact auditing; research-on-research (evidence tiers, claim quarantine, reporting contracts).
- **Paradigm:** Diagnostic audit and failed-identifiability report. The manuscript explicitly positions itself as "an engineering diagnostic, not a benchmark result, not a new algorithm, and not a clean causal study" (`paper.tex:1295`).
- **Methodology type:** (i) A scoped algebraic identity for one specified noncanonical runner; (ii) retrospective descriptive audit of a heterogeneous, mostly single-seed run corpus; (iii) a prospective evaluation protocol that is specified but not executed. No controlled experiment, no validated predictor, and no intervention comparison is reported.
- **Inferred target venue/tier:** NeurIPS 2026 (style file `neurips_2026`; comment suggests Evaluations & Datasets track or Main Track). Top-tier, highly selective venue. Inference only — see `criteria_binding_unavailable` above.
- **Maturity:** Late-stage corrective descendant of a prior submission. The abstract states the text is "a descendant of NeurIPS submission #36320, not its authenticated or byte-identical reviewed PDF" (`sections/abstract.tex:9-10`). Substantial self-audit apparatus (withdrawals, quarantines, regression cases) has been layered onto a large historical compendium; the result reads as a correction manifest wrapped around a retained results ledger rather than a submission-shaped paper.

### Reviewer Configuration Cards

**Journal-Fit (EIC).** Stance: venue fit, contribution significance, and submission shape for a top-tier ML venue. Lens: does the validated core clear the bar for a full paper in the inferred track, and is the manuscript structured so a reader can find that core? Will penalize compendium structure, duplicated displays, and overlap with sibling submissions; will credit unusual candor about withdrawn claims.

**R1 — Methodology.** Stance: statistical and experimental validity. Lens: independent units, estimands, window definitions, artifact-to-number binding, and whether each display's label matches its computation. Will trace headline numbers to repo artifacts and treat any untraced number as unreported; sympathetic to descriptive work only when windows and cohorts are exactly bound.

**R2 — Domain.** Stance: RL-for-LLMs subject-matter correctness and novelty against the 2024–2026 group-relative RL literature. Lens: whether ZVF/GU adds anything beyond known zero-advantage-group diagnostics, whether reward environments measure what their task labels claim, and whether related-work characterizations are accurate and load-bearing.

**R3 — Perspective.** Stance: practitioner, reproducer, and downstream-user perspective, including ethics/compute accounting and artifact usability. Lens: what decision a reader can take from this paper, whether receipts bind claims to executions, and whether costs/risks are stated in a form a third party can act on.

**Devil's Advocate (fixed 5th seat, no card).** Charged with the strongest good-faith case against acceptance, including claim–evidence-tier mismatch, circular validation, and the "so-what" test.

---

## Seat report 1 — Journal-Fit (EIC)

### Strengths (anchored)

- **Exceptional claim discipline.** The same three contributions are stated identically in abstract, introduction, and conclusion (`sections/abstract.tex:6-22`; `sections/intro.tex:9-19`; `sections/conclusion.tex:3-10`), and a full section enumerates claims explicitly *not* made (`paper.tex:1292-1311`). Few submissions quarantine their own historical results this explicitly (e.g., the withdrawn 15-run ZVF correlation, `paper.tex:963-971`; the withdrawn saturation fits, `paper.tex:532-542`).
- **The prospective operator-evaluation protocol is genuinely well specified** — matched budgets charging discarded/resampled completions, whole-run holdout, nonlinear reward/history baselines, predefined failure-event coverage (`paper.tex:1592-1606`). As a design contribution this is the strongest forward-looking material in the manuscript.

### Weaknesses

**EIC-W1 — Validated core is below full-paper significance for the inferred venue.**
*Anchor:* `sections/abstract.tex:13-22` ("two descriptive all-wrong cases, not a validated predictor or stopping policy… not a deployment benefit or algorithm-ranking claim"); `sections/intro.tex:18-19` (contributions "do not include a new diagnostic definition, a controlled cross-library benchmark, a validated stopping rule, or an algorithm ranking").
*What is wrong:* By the authors' own correct accounting, what survives audit is an elementary identity for one noncanonical runner plus two all-zero descriptive cells plus a reporting contract. That is a valuable correction note, but it is not a NeurIPS-scale empirical, algorithmic, or datasets contribution. The Datasets/Evaluations reading also fails: no reusable benchmark, dataset, or evaluated artifact with bound provenance is delivered (checklist answers reproducibility "No", `sections/checklist.tex:23-26`).
*Fix:* Recast and retarget as a short reproducibility/audit note (workshop, journal reproducibility track, or archival correction record), **or** execute the already-specified prospective evaluation (`paper.tex:1592-1606`) so the paper contains a validated result. Textual revision alone cannot close this gap.

**EIC-W2 — Compendium structure buries the contribution and duplicates displays.**
*Anchor:* Root file is 1,813 lines with the same rows repeated across `tab:main_results` (`paper.tex:372-441`), `tab:cross_arch_scaling` (`paper.tex:686-713`, caption admits "These rows repeat entries of Table main_results"), `tab:dense_moe` (`paper.tex:727-743`), `tab:frontier` (`paper.tex:777-797`), and `tab:ppo_grpo` (`paper.tex:817-837`). Nearly every results subsection narrates a removed historical table or figure (e.g., `paper.tex:495-501`, `517-521`, `546-549`, `1069-1075`).
*What is wrong:* A new reader must excavate the two-cell reviewed record from dozens of pages of withdrawn-claim archaeology and repeated ledgers. No compiled PDF for this root exists in the workspace to confirm the typeset length, but the source volume — including a 427-line ethics input placed before references (`paper.tex:1689`) — is incompatible with a standard conference page budget in its current organization.
*Fix:* Restructure around the reviewed record: identity, two cells, evidence-tier contract, prospective protocol. Move the historical ledger to a clearly labeled archival appendix or external audit log; delete duplicate regroupings of identical rows; compile and report the final page count.

**EIC-W3 — Overlap with sibling manuscripts and prior submission is undisclosed in venue terms.**
*Anchor (this text alone):* descendant-of-#36320 statement (`sections/abstract.tex:9-11`); "The associated thesis Table 6.2 remains provisional" (`paper.tex:1322-1323`). *Workspace context (titles only, bodies not read):* the same directory contains sibling manuscripts on ZVF, group size, scaling identifiability, length bias, minimum reporting, a GRPO registry, and a ZVF controller (`paper_P1_scaling.tex` through `paper_P7_zvf_controller.tex`), whose topics visibly partition this compendium's sections.
*What is wrong:* Whether or not any sibling is under review, submitting this compendium alongside spin-offs of the same corpus creates salami/dual-submission exposure that the manuscript never addresses in disclosure terms. A venue cannot tell which results are claimed where.
*Fix:* Add an explicit overlap disclosure: relationship to submission #36320, to the thesis, and to each sibling manuscript derived from this corpus; state which displays are unique to this submission. If siblings are or will be under review concurrently, consolidate or clearly partition the claims before submission.

## Seat report 2 — R1 Methodology

### Strengths (anchored, independently traced)

- **Several headline numbers trace exactly to artifacts.** The five-seed held-out GSM8K record (`paper.tex:1399-1409`: 83.0/82.5/80.5/84.0/86.5%, mean 83.3%, SD 2.2; base 164/200 = 82.0%) reproduces exactly from `platform_tinker/reports/final/gsm8k_heldout_seed*.json` and `gsm8k_base_control_200.json` (recomputed mean 83.3, SD 2.20). The ZVF lagged table (`paper.tex:1002-1018`: N = 927/861/795/630; partial r = 0.050/−0.006/0.059/0.080) reproduces exactly from `platform_hybrid/experiments/zvf_lagged_regression_results.json`. The Wave 6 sweep (`paper.tex:573-591`) reproduces from `wave6_ablations.json`, including the temperature peak range [0.688, 0.812], the T=0.4 last-10 value 0.425, and the batch peak sequence 0.875/0.688/0.594/0.547. The two reviewed tool-use cells trace to `cross_tool_qwen3-32b.json` / `cross_tool_llama-8b-inst.json` (30-step all-zero traces, `zero_reward_pct` = 100).
- **The statistical evidence policy is exemplary** (`sections/statistical_evidence_policy.tex` in full): step-iid inference, reconstructed-seed bootstraps, and inconsistent multiplicity families are correctly withdrawn rather than repaired cosmetically.

### Weaknesses

**R1-W1 — Group-size and frontier window labels contradict their own artifacts (violates the paper's own contract).**
*Anchor:* `tab:main_results` group-size block labels runs "26 logged" / "11 logged" with Peak and Last-10 columns (`paper.tex:412-417`); text at `paper.tex:1204-1210`. The paper's own rule: "A training-step count is not a logged-observation count" (`sections/evidence_identity_audit.tex:28`). *Artifact:* in `master_results.json`, `campaign_v2_w2_qwen3-8b_G{2,4,16}` have `steps_completed` = 26 but `reward_trace` length 6 (G32: 11 vs 3), and each `last10_avg` equals the mean of **all** logged points (e.g., G16: mean of 6-point trace = 0.3803 = reported last-10). Similarly Qwen3-235B-A22B is tabled at 15 steps with last-10 = 100% (`paper.tex:396`) while `frontier_gsm8k_qwen3-235b.json` records `steps` = 20 with a 4-point trace.
*What is wrong:* "Last-10" over 3–6 observations and "26 logged" for 6 observations are mislabeled windows in the paper's flagship ledger — precisely the failure mode the audit exists to catch. Peak over 6 points vs peak over 30 points are also not comparable summaries, yet they share one table.
*Fix:* Relabel every ledger row with (steps planned, steps completed, logged observations, window actually computed). Recompute or rename "last-10" where fewer than 10 observations exist. Add a machine-checked table-generation gate so window labels cannot diverge from trace lengths.

**R1-W2 — The 28-run SI/PTD analysis is untraced and mechanically coupled, yet retained as quantitative results.**
*Anchor:* `paper.tex:888-913` (r = −0.436 and −0.517 vs last-10 reward; instability bins 19/5/4; GRPO vs classic-RL SI means 0.162 ± 0.157 vs 0.814 ± 0.370; classic-RL mean PTD 0.619 ± 0.139); Nemotron SI = 1.180, PTD = 0.762 (`paper.tex:915-923`).
*What is wrong:* No released per-run table identifying the 28 experiments, their windows, or denominator handling was found in this panel (targeted search of `platform_hybrid/experiments/*.json|*.md` located no source for these coefficients). The text itself notes PTD = 1 − r̄_tail/r_max shares the tail mean with the outcome (`paper.tex:891-894`), so the correlations cannot validate a warning signal; SI/PTD are additionally undefined at zero denominators (`paper.tex:881-883`), and the bins are flagged as needing a "denominator-validity audit" (`paper.tex:905`) that has not been done. Untraced, self-coupled summaries should not survive in a paper whose standard elsewhere is withdrawal.
*Fix:* Either publish the bound per-run SI/PTD table with cohort IDs and denominator rules and reframe the correlations as definitional, or withdraw §"Reward-Trajectory Stability Proxies" numbers to the same quarantine standard applied to the 15-run ZVF summary.

**R1-W3 — ZVF lagged analysis cannot support even a descriptive predictive reading in its current form.**
*Anchor:* `paper.tex:985-1000` and `tab:zvf_lagged` (33 runs; pooled overlapping pairs N up to 927; lag-10 partial r = 0.080).
*What is wrong:* The manuscript correctly explains why a linear reward control cannot establish incremental information (the G=2 counterexample, `paper.tex:988-990`) and withholds p-values — but then retains the table under a "Disentangling Prediction from Tautology" heading. The 33-run cohort is not bound to run IDs in the text, within-run dependence is unmodeled, and no nonlinear-baseline or whole-run-holdout comparison is performed. A reader is left with coefficients the paper itself proves uninterpretable.
*Fix:* Demote the table to an explicitly negative methodological illustration (or remove it), bind the 33 run IDs, and reserve any predictive language for the prospective whole-run-holdout design already specified at `paper.tex:1588-1606`.

**R1-W4 — The specified runner's loss itself undermines training-reward dynamics as evidence.**
*Anchor:* `sections/diagnostic_method.tex:46-54`: the callback "sums all returned token log-probabilities," applies no completion-only mask and no length normalization, and "shifts the concatenated prompt/response token sequence and appends a dummy zero target" — so ℓ is "not a claimed correctly masked conditional-completion likelihood." The identity is anchored to this implementation (`grpo_exp_a_baseline.py`, `grpo_loss_fn`; population-SD + 1e-8 form confirmed in the workspace copy).
*What is wrong:* Two separable issues are conflated throughout the ledger. (a) The centered-coefficient identity is exact for this callback, but (b) every "GRPO" training-reward trajectory in `tab:main_results` was produced by optimizing prompt tokens as well as completions under a shifted-target construction. Reward dynamics under a misaligned likelihood cannot be read as evidence about group-relative RL behavior generally, yet tables continue to label rows "GRPO" (e.g., `paper.tex:387-417`) with the noncanonical caveat located sections away.
*Fix:* Rename ledger rows to the actual treatment (e.g., "audited group-relative runner (noncanonical)") wherever they appear, not only in caveat sections; state in the ledger caption that prompt-token loss and target shifting are part of the treatment. Commit binding (`ddcbb868`) to executions remains admittedly incomplete (`sections/diagnostic_method.tex:56-59`) and must be closed or the rows further demoted.

**R1-W5 — Held-out GSM8K design is correctly hedged but analytically underused and over-relied upon elsewhere.**
*Anchor:* `paper.tex:1399-1413` (one-sample seed-level t(4) = 1.32, p = 0.256 against a single base pass; "not an equivalence or no-effect result"); ethics §M1 reuses the same +1.3pp figure to bound dual-use uplift (`ethics_statement.tex:45-50`).
*What is wrong:* The single base evaluation supplies no baseline variance, and the same 200 test prompts are used for every seed — a prompt-paired analysis (e.g., per-item agreement/disagreement against base) is the natural estimand and is not reported even though per-seed artifacts exist. Meanwhile the inconclusive estimate does load-bearing work in the dual-use argument, where "inconclusive" and "small uplift" are not interchangeable.
*Fix:* Report prompt-paired base-vs-adapter agreement from the existing artifacts (or state why per-item records cannot support it); in the ethics statement, replace the point-estimate uplift bound with the correct statement that capability change is unmeasured within wide uncertainty.

**R1-W6 — Wave 6 headlines use peak statistics from a single seed.**
*Anchor:* `paper.tex:573-591` ("per-step batch size shows a clear monotone decay in peak reward").
*What is wrong:* The numbers trace (see Strengths), but peak-over-30-steps from one seed is a max-statistic dominated by trace noise; the fixed-step design also spends unequal rollout budgets across batch sizes, as the text notes (`paper.tex:589-591`). "Clear monotone decay" overstates a single-seed descriptive sweep, and first-5/last-10/peak are three post hoc windows on the same traces.
*Fix:* Retain the sweep as descriptive telemetry with window definitions tied to trace length; remove ordering language ("clear," "monotone decay" as a finding) or add seeds under a declared token budget.

---

## Seat report 3 — R2 Domain

### Strengths (anchored)

- **The Bernoulli reference theory is correct and unusually careful.** Expected homogeneity h_G(p) = p^G + (1−p)^G, its U-shape, the prompt-heterogeneity point (E[h_G(p_q)] ≠ h_G(p̄)), and the G=8 example (mean reward 0.5 with expected ZVF 1 vs 1/128) are all stated correctly (`paper.tex:1041-1059`), as is the G=8→16 expected-GU gain of ~13.96pp at p=0.2, which correctly refutes any "diminishing returns beyond 8 are negligible" shortcut. The deterministic regression cases (`sections/audit_regression_cases.tex`) are valid checks of these interpretations.
- **No priority is claimed where none exists.** The manuscript states plainly that 2026 work "pre-empts this diagnostic" and that "no priority for the ZVF definition is claimed" (`sections/related_work_v2.tex:59-76`; `paper.tex:1320`). This honesty is correct — and, as R2-W1 notes, fatal to a novelty-based case for acceptance.

### Weaknesses

**R2-W1 — The diagnostic is prior art under other names; the domain contribution is terminological.**
*Anchor:* Closest-prior table (`sections/related_work_v2.tex:78-92`): ZVF (exact equality) vs AVSPO's Advantage Collapse Rate (SD below tolerance) — "matching thresholds and prompt weights makes the basic measured object coincide" (`sections/related_work_v2.tex:59-63`). Dynamic sampling that discards homogeneous groups is a headline component of DAPO, described at `sections/related_work_v2.tex:97-100`; prompt selection by group variance/pass-rate (VCRL, online difficulty filtering, GRESO, CPPO) is catalogued at lines 66-69 and 106-109.
*What is wrong:* "Fraction of groups with zero advantage variance" is a known, already-operationalized quantity in the group-relative RL literature. Renaming it ZVF/GU and proving the centered-coefficient identity for a noncanonical runner does not give the domain a new measurable, a new estimator, or a new intervention. The paper concedes this; a top venue cannot accept on the remainder (an audit of the authors' own corpus) without a domain-transferable result.
*Fix:* Position the work explicitly as a reproducibility/measurement note about *reporting* homogeneous-group fractions (signed: all-wrong vs all-correct vs homogeneous-intermediate, which `sections/diagnostic_method.tex:60-64` correctly insists on), and demonstrate value over ACR-style reporting on at least one external or prospective cohort — or target a venue where audit notes are in scope.

**R2-W2 — Task labels exceed what the reward environments measure; untraced team percentages remain in Results.**
*Anchor:* Tool-use reward "scores JSON well-formedness, tool-name selection, and argument-key presence without executing tools"; HumanEval-style sandbox "rejects legitimate completions that use `len`, `range`, or `sum`" (`paper.tex:629-634`). Despite this, `tab:task_grpo` (`paper.tex:636-655`) reports Code (HumanEval) 32%→86% and 67%→100%, Tools 52%→70%, and non-numeric "Func."/"Impr." entries, and §Auxiliary dynamics asserts a two-phase pattern "observed consistently across multiple seeds and tasks" (`paper.tex:657-666`).
*What is wrong:* (a) A sandbox that bans `len`/`range`/`sum` is not HumanEval in any domain-meaningful sense; 86%/100% under it will be cited as HumanEval gains regardless of caveats. (b) These team-experiment percentages were not traced to artifacts in this panel and carry no run IDs, windows, or evaluator revisions in the text. (c) The "multiple seeds" consistency claim at `paper.tex:661-662` contradicts the paper's own repeated statement that most runs are single-seed.
*Fix:* Remove `tab:task_grpo` percentages from Results (or bind each cell to a run ID + evaluator and rename tasks to what was measured, e.g., "restricted-sandbox code-format reward"). Delete or evidence the multi-seed consistency sentence.

**R2-W3 — Related work is a broad survey pasted around a narrow audit, with secondary numbers reproduced uncritically.**
*Anchor:* `sections/related_work_v2.tex` (275 lines) surveys PRMs, tool-use agents, scaling laws, and frameworks largely disconnected from the two-cell reviewed record; specific external effect sizes are asserted as facts, e.g., CPPO "up to 7.98× speedup on GSM8K" (line 109), Scaf-GRPO "boosting AIME 2024 by 44.3% relative" (lines 111-113), SSPO "46.72 versus 43.01 (GRPO)" (lines 116-119). Bibliography: 261 entries, 108 never cited.
*What is wrong:* The audit's own standard — bind every number to its cohort and evaluator — is not applied to literature numbers, several of which come from very recent (2026) preprints whose existence and claims this panel could not verify from the workspace. The uncited-entry bloat and survey breadth (e.g., vendor "Deep Research" systems, lines 198-200) dilute the one related-work subsection that matters (zero-variance diagnostics, lines 59-76).
*Fix:* Cut related work to the zero-advantage-diagnostic lineage, stack-identifiability precedents (Implementation Matters; Henderson et al.; rliable), and direct comparators; either verify each quoted external effect size against its source or delete the numbers; remove uncited bibliography entries.

**R2-W4 — No same-stack comparator gives the domain an actionable lesson.**
*Anchor:* Missing-baselines limitation (`paper.tex:1424-1430`): no SFT-only warm-start, best-of-N/rejection sampling, DAPO/Dr. GRPO correction, or longer-budget matched comparison was run.
*What is wrong:* The title promises a lesson — reward contrast over algorithm labels — but the corpus contains no comparison in which reward contrast is manipulated while the stack is held fixed (the fixed-completions reward-intervention check is proposed only as future work, `paper.tex:1579-1584`). A domain reader finishes knowing that homogeneous groups give zero centered coefficients (definitional) but not whether any known remedy (dynamic sampling, resampling, warm-start) helps under this stack.
*Fix:* Run the cheap decisive check the paper itself designs: hold policy/completions/masks fixed and intervene on reward vectors to verify coefficient behavior in the *deployed* runner, and/or one matched-budget dynamic-sampling vs fixed-schedule arm. Without at least one such result, retitle to reflect an audit/correction scope.

## Seat report 4 — R3 Perspective

### Strengths (anchored)

- **The immutable-cohort contract is a reusable practitioner contribution**: run IDs and source hashes bound to checkpoint/tokenizer revisions, objective version, split, evaluator, metric window, logging cadence, aggregation, seeds, completion status, evidence tier (`sections/evidence_identity_audit.tex:24-30`), plus release gates (`sections/audit_regression_cases.tex:36-45`). If adopted, this checklist would prevent most failures the paper documents.
- **Availability is stated without inflation**: local source vs managed configuration vs undisclosed server internals are separated, and blanket public checkpoint/log availability is explicitly *not* claimed (`paper.tex:1217-1244`).

### Weaknesses

**R3-W1 — No usable operator decision is delivered.**
*Anchor:* "Unsigned ZVF does not choose an action: retire/distill and retry/resample/warm-start require opposite choices" (`paper.tex:976-978`); "A lower ZVF alone is not operator benefit" (`paper.tex:1589-1590`); prospective hypotheses only (`paper.tex:1262-1286`).
*What is wrong:* A practitioner monitoring training learns to compute one more fraction but receives no validated threshold, no false-stop/recovery rates, and no cost analysis (wasted rollouts vs premature stops). Since reward = 0 is already directly visible in the two reviewed cells, the incremental operational value over "watch the reward" is asserted as "triage" but never demonstrated, even retrospectively with a non-overlapping early-window/late-endpoint analysis.
*Fix:* Provide at minimum a retrospective decision analysis on held-out whole runs: freeze an early signed-ZVF rule (all-wrong vs all-correct separated), compare against reward-mean/slope baselines on a later non-overlapping endpoint, and report false stops and recoveries. Otherwise present ZVF strictly as a reporting field, not a triage diagnostic.

**R3-W2 — Reproducibility receipts do not bind to the claimed runs.**
*Anchor:* Reviewed cells' "exact checkpoint revisions, authenticated run IDs, evaluator revisions, and complete rollout hashes have not been reconciled" (`sections/evidence_identity_audit.tex:5-7`); local source snapshot "is not proof that a particular reviewed execution used that exact commit" (`sections/diagnostic_method.tex:56-59`); adapters referenced by wildcard (`arvindcr4/tinker-rl-bench-*`, `ethics_statement.tex:30-31`); checklist reproducibility = "No" (`sections/checklist.tex:23-26`).
*What is wrong:* This panel *could* trace several numbers because it had full workspace access (see R1 Strengths) — but that is not the position of an external reproducer. From the manuscript alone, no reviewed cell can be reconstructed: the execution that produced reward 0 / ZVF 1 cannot be re-run or even uniquely identified, and the managed backend is closed (`ethics_statement.tex:310-318`). For a paper whose thesis is identifiability, failing its own cohort contract on its only two primary cells is a central, not peripheral, gap.
*Fix:* Publish a minimal immutable manifest for the two reviewed cells (checkpoint revision, runner commit, prompt set hash, reward code version, raw group rewards) sufficient for exact ZVF recomputation. This is achievable without new training and should be the first revision item.

**R3-W3 — Compute, cost, and carbon accounting exhibits discrepancies instead of resolving them.**
*Anchor:* Spend entries "sum to USD 147–177, inconsistent with the earlier USD 130–140 total… no project spend total is certified" (`ethics_statement.tex:104-109`); Modal hours conflict 18 vs 250 GPU-hours (`ethics_statement.tex:184-186` vs `paper.tex:1723-1748`, "~1,200" total); carbon table is a "scenario sum (not project total)" under an unverified 1×H100-per-Tinker-run assumption (`ethics_statement.tex:188-217`).
*What is wrong:* Candor about unreconciled ledgers is preferable to false precision, but a submission should not ask reviewers to accept three mutually incompatible accountings as a Transparency contribution. The carbon scenario (~296 kg) will be quoted as the project's footprint despite its labels.
*Fix:* Replace all historical allocation tables with one cohort-bound ledger (dates, run IDs, measured vs assumed hours, cash vs credits, failures included/excluded) or remove quantitative totals and state only that accounting is unavailable. Do not retain a summable carbon table whose total the text disowns.

**R3-W4 — Audience and framing: a correction manifest addressed to prior reviewers, not to new readers.**
*Anchor:* Abstract opens with submission-genealogy ("descendant of NeurIPS submission #36320… We audit that submitted record rather than claiming that local edits change it," `sections/abstract.tex:9-11`); results sections repeatedly narrate what was removed/withdrawn/quarantined (e.g., `paper.tex:495-501`, `532-542`, `963-971`); "Gradient Utilization" terminology is retained while being disowned as measuring gradient usefulness (`sections/intro.tex:21-24`).
*What is wrong:* A reader without the #36320 review history cannot tell what is new science vs point-by-point rebuttal residue. Retaining the misleading name "Gradient Utilization" for historical continuity — in tables (`tab:audited-tool-cells`) where GU = 0 will be read as "no gradient" — actively invites the misinterpretation the text warns against. The 8-author front matter with consumer email addresses (`paper.tex:51-83`) and unanonymized preprint mode also signal a lab report rather than a submission-ready manuscript (anonymization instructions exist only as comments, `paper.tex:7-16`).
*Fix:* Write the resubmission for a cold reader: problem, identity, evidence, contract. Move genealogy and withdrawal narration to a separate audit log cited once. Rename GU to a non-causal label (e.g., mixed-group fraction) with a one-line historical alias note.

---

## Seat report 5 — Devil's Advocate

### Strongest Counter-Argument

This manuscript asks a top venue to publish the withdrawal of its own prior claims, dressed as a positive thesis its evidence cannot carry. The title asserts a general lesson — reward contrast, not algorithm labels, is what matters — yet no comparison in the corpus identifies that lesson: the framework-swap cohort identity is unresolved, the Qwen PPO row is quarantined for provenance conflict, the Llama comparison window conflicts across summaries, and two of four Task 4 backends are deterministic dry runs. Stack sensitivity is therefore inferred from gaps the paper itself declares unidentified, which is circular: unexplained heterogeneity is relabeled as evidence that labels are insufficient, when it is equally consistent with logging errors, mismatched checkpoints, or plain noise. What remains as primary evidence is an algebraic tautology — centering a constant vector yields zeros — proven only for a deliberately noncanonical runner that omits clipping, reference KL, and completion masking and that optimizes prompt tokens under a shifted target, together with two all-zero tool-schema cells with no held-out outcome. Two zeros cannot validate a triage diagnostic, estimate its false-stop rate, or distinguish a dead reward environment from a broken parser, a chat-template mismatch, or an impossible prompt set. Meanwhile the retained 30-plus-row ledger, stability-proxy correlations, and team-experiment percentages present exactly the extractable numbers the disclaimers disown, including demonstrably mislabeled windows. A reader who believes the caveats learns almost nothing actionable; a reader who ignores them will cite numbers the authors themselves do not stand behind. Either way, acceptance would reward the appearance of rigor — exhaustive self-quarantine — over the production of a single identified, reproducible, decision-relevant result. The honest core deserves an archival correction record, not a research slot.

### Issue List

**DA-CRITICAL-1 — Headline thesis stronger than evidence tier [Dimension: claim calibration; Location: title `paper.tex:45-46`, Task 4 `paper.tex:448-479`, PPO/GRPO `paper.tex:806-847`].**
"Reward Contrast, Not Algorithm Labels" is a comparative causal framing. The supporting comparisons are, by the text's own disposition, unresolved (Task 4 Tinker tail 85.6% vs 84.4%, `sections/evidence_identity_audit.tex:45`), quarantined (Qwen PPO, `paper.tex:830-847`), or dry-run fallbacks (verl/OpenRLHF, `paper.tex:460-464`). No identified contrast manipulation exists. The title should not survive in its current form without a same-stack identified comparison.

**DA-CRITICAL-2 — Primary empirical record cannot validate the proposed diagnostic [Dimension: evidence sufficiency; Location: `sections/evidence_identity_audit.tex:9-22`, `sections/abstract.tex:13-15`].**
n = 2 all-wrong cells, both with held-out "not evaluated," zero positive/mixed reviewed cases, zero recovery cases. Even hypothetical perfect detection of two positives gives a 95% sensitivity interval of [0.158, 1] (the paper's own calculation, `sections/audit_regression_cases.tex:26-29`). Triage utility, prediction, and stopping value are entirely unmeasured — yet "triage diagnostic" language persists throughout (e.g., `paper.tex:973-978`, `1432-1440`).

**DA-CRITICAL-3 — Retained Results invite the exact misreading the disclaimers forbid, and contain window errors [Dimension: reporting integrity; Location: `tab:main_results` `paper.tex:372-441`, §KL proxies `paper.tex:888-913`, `tab:task_grpo` `paper.tex:636-655`].**
Quarantine-by-caption does not work in practice: tables are extracted, caveats are not. This panel verified that "26 logged / Last-10" group-size rows rest on 6-point traces whose overall mean was relabeled "last-10" (see R1-W1). A paper arguing that labels mislead cannot itself publish GRPO-labeled rows from a noncanonical prompt-token-optimizing runner, untraced SI/PTD coefficients, and restricted-sandbox percentages under a "HumanEval" heading, however many footnotes it attaches.

**DA-MAJOR-1 — Circular / self-referential validation [Dimension: inference; Location: `paper.tex:888-894`, `985-1000`].**
PTD is an algebraic function of the tail mean it is correlated with; ZVF is a deterministic function of the reward distribution whose future values it is asked to predict beyond a linear control the paper proves inadequate. The manuscript discloses both couplings — disclosure does not convert them into evidence. These analyses should be presented as negative examples or removed, not retained as "Quantitative Results."

**DA-MAJOR-2 — Reproducibility receipts do not bind to claimed runs [Dimension: reproducibility; Location: `sections/diagnostic_method.tex:56-59`, `paper.tex:1217-1244`, `sections/checklist.tex:23-31`].**
Neither the #36320 reviewed record (unauthenticated descendant), nor commit `ddcbb868` (execution mapping incomplete), nor wildcard adapter URLs bind any reviewed number to a re-runnable execution. The audit's central demand on others — immutable cohort identity — is unmet for its own two primary cells.

**DA-MAJOR-3 — Salami/overlap exposure across the corpus [Dimension: publication ethics (disclosure, not misconduct); Location: `sections/abstract.tex:9-11`, `paper.tex:1322-1323`; sibling titles in same directory].**
The same run corpus visibly supports this compendium, a thesis table, a prior submission, and at least seven sibling manuscripts partitioning ZVF/scaling/group-size/length-bias/reporting topics. From this text alone, duplicated submission cannot be proven and none is alleged; but the absence of any overlap map makes double-counting of the same rows across venues unauditable.

**DA-MINOR-1 — Misleading retained terminology [Dimension: communication; Location: `sections/intro.tex:21-24`].**
"Gradient Utilization" is kept "as terminology" while the text insists it measures neither gradient magnitude nor usefulness. In tables, GU = 0 will be read as "no gradient." Historical continuity is not worth a systematically misleading name.

**DA-MINOR-2 — Padding and production immaturity [Dimension: presentation; Location: BrowserGym smoke `paper.tex:1109-1139`; artifact-scope table `paper.tex:599-617`; bibliography].**
A one-step 0.0-reward browser integration smoke test, a Gymnasium/SB3-Zoo feature-checklist table, and 108 uncited references add length without evidence. No compiled PDF for this root was present to verify typesetting.

### Ignored Alternative Explanations

- **Verifier/parser failure as the cause of the two all-zero cells.** A strict schema scorer returning 0 for every completion over 30 steps is at least as consistent with a parser, template, or tool-specification bug as with a policy that samples no valid calls. No raw completion inspection, parser unit test, or positive-control (e.g., scoring gold/reference completions through the same reward) is reported for these cells. ZVF = 1 cannot distinguish "no contrast to learn from" from "reward channel disconnected."
- **Optimization-side failure independent of contrast.** Prompt-token loss, shifted targets with a dummy zero target, LoRA configuration, or managed-backend execution issues could suppress learning even with mixed rewards. Because the reviewed record contains no mixed-reward control cell under the same runner, "no contrast" and "broken optimization" are fully confounded.
- **Checkpoint/template mismatch.** Base-vs-instruct chat-template or tokenizer mismatches produce uniformly unparseable outputs (hence homogeneous rewards) for reasons unrelated to group-relative mechanics. The dense/MoE and base/instruct discussions acknowledge entanglement but no template audit is shown.
- **Task impossibility / prompt-set degeneracy.** If the 35-prompt synthetic tool set (cf. `ethics_statement.tex:253-255` for the xLAM track; the reviewed cells' prompt set is not bound in the text) contains unsolvable or misspecified items, homogeneous failure reflects the dataset, not a general RL phenomenon.

### Missing Stakeholder Perspectives

- **Prior reviewers/readers of submission #36320**, whose reviewed record this descendant corrects without an authenticated diff — they cannot tell what changed.
- **The managed-platform operator (Tinker/Thinking Machines):** claims about undisclosed server internals, interruptions, and credential expiry (`paper.tex:1344-1352`) are single-sided; a platform response or status record is neither sought nor cited.
- **Downstream adapter users:** LoRA adapters are released via the Hub (`ethics_statement.tex:30-31`) with model-card caveats, but no safety evaluation perspective from potential deployers of tool-calling adapters is represented beyond author assertion.
- **Benchmark/dataset maintainers** (GSM8K, HumanEval, xLAM): test-split reuse, sandbox modification, and contamination are discussed from the authors' side only; maintainer guidance on split use is not engaged.

### Observations (Non-Defects)

- The null held-out result (82.0% → 83.3%, p = 0.26) is **not** over-read as equivalence; the text explicitly denies that reading (`paper.tex:1407-1410`). This panel verified the underlying per-seed numbers trace exactly. The deficiency is analytic underuse (R1-W5), not misstatement.
- Withdrawals of the saturation fits, 15-run correlation, scale scatter, and significance families are appropriate corrections, not defects. The defect (DA-CRITICAL-3) is that other displays meeting the same standard were retained.
- Short horizons and single seeds are understandable under student-budget managed-API constraints (disclosed at `ethics_statement.tex:136-141`, `352-366`). They limit conclusions; they are not, by themselves, grounds for rejection — the mismatch between that evidence tier and the title/venue is.

### So-what test

Delete ZVF from the two reviewed cells: an operator watching reward stuck at exactly 0.0 for 30 steps already knows the run is dead and already faces the same retire-vs-retry decision. ZVF adds value only if it fires *earlier*, *more selectively*, or *more actionably* than reward level/slope on runs that later diverge — none of which is measured here (no early-window/late-endpoint split, no false-stop count, no recovery case). As presented, the diagnostic renames an already-visible symptom. The paper's own prospective protocol (`paper.tex:1592-1606`) is the correct test of whether the renaming buys anything; until that test exists, the "so-what" answer is: a reporting field worth one column in a run log, not a paper-length claim.

---

## Editorial Decision Letter

**Decision: Reject.**

Dear Authors,

Thank you for a manuscript whose candor about its own evidence is, in places, exemplary. All five seats agree that the self-audit apparatus — evidence tiers, quarantines, withdrawn significance decisions, and the immutable-cohort contract — is methodologically correct and, in this panel's spot-checks, several surviving numbers (the five-seed held-out GSM8K record, the ZVF lagged table, the Wave 6 sweep, the two all-zero tool-use traces) trace exactly to repository artifacts. The disagreement across seats is not about your honesty; it is about what, after your own audit is applied, remains for a top venue to publish.

**Consensus across seats.** (1) The scoped centered-reward identity for the specified runner is correct. (2) The diagnostic quantity itself is prior/concurrent art under other names (notably advantage-collapse reporting), as the manuscript concedes. (3) The primary reviewed evidence — two all-wrong synthetic tool-use cells without held-out evaluation — cannot validate ZVF as a predictor, stopping rule, or triage policy. (4) The retained descriptive ledger, stability-proxy correlations, and auxiliary team tables do not meet the binding standard the paper itself sets, and include window labels ("26 logged," "last-10") that this panel verified contradict their underlying traces. (5) No identified comparison isolates "reward contrast" from stack, checkpoint, sampler, optimizer, or evaluator effects.

**Disagreement / nuance across seats.** R1 and R3 emphasize that parts of the record *do* trace exactly and that the cohort contract plus prospective protocol are reusable contributions — on that basis, a substantially shortened audit note would be welcome elsewhere. R2 and the Devil's Advocate weight pre-emption by existing zero-advantage diagnostics more heavily and doubt that even a shortened version clears a novelty bar without a prospective result. The EIC seat regards the decisive issue as venue/contribution scale rather than any single technical flaw. No seat recommends acceptance at the inferred venue in current form; seats differ only on the most promising resubmission shape (short correction/audit note vs new prospective study).

**Adjudication of every Devil's Advocate CRITICAL issue (visible, as required):**

- **DA-CRITICAL-1 (title/thesis stronger than evidence tier): VALIDATED.** The comparative title claim has no identified supporting comparison; its candidate evidence is unresolved, quarantined, or dry-run. A validated CRITICAL of this kind blocks Accept. Remedy requires either retitling to audit scope or a new same-stack identified experiment — beyond textual revision.
- **DA-CRITICAL-2 (n = 2 record cannot validate the diagnostic): VALIDATED.** With two all-wrong cells, no held-out outcomes, no positive/mixed/recovery cases, operating characteristics are not merely imprecise but unidentified. Blocks Accept for any version that presents ZVF as a validated triage/predictive diagnostic.
- **DA-CRITICAL-3 (retained Results contradict the quarantine standard, incl. verified window mislabels): VALIDATED.** Independently confirmed by R1's artifact traces (R1-W1). Blocks Accept; unlike CRITICAL-1/2, this one is remediable by withdrawal/recomputation without new training, and must be remedied in any future version anywhere.

No DA CRITICAL is rejected; therefore no rejection rationale for a CRITICAL is required. Because validated CRITICAL issues remain, Accept and Minor Revision are excluded. Major Revision was considered and not selected: closing CRITICAL-1/2 requires new prospective experimentation (the protocol at `paper.tex:1592-1606` is well designed but unexecuted), which exceeds a revision cycle; and the contribution-scale problem (EIC-W1) would remain even if CRITICAL-3 were fixed by editing alone.

**Scientific validity (separate from venue fit):** The narrow claims the manuscript actually stands behind — the centered-coefficient identity for the specified callback, the Bernoulli h_G reference calculations, the two descriptive all-zero cells, and the correctly hedged null held-out comparison — are, within their stated scope, valid; several were independently traced to artifacts by this panel. Validity fails only where retained displays exceed that scope (window mislabels, untraced SI/PTD summaries, task-labeled proxy percentages).

**Venue fit (separate from validity):** Poor for the inferred NeurIPS Main / Evaluations & Datasets full-paper tracks. The validated contribution (correction + reporting contract) matches a workshop, reproducibility/audit track, or journal correction record. Note `criteria_binding_unavailable`: if the actual target is such a venue, EIC-W1 should be re-weighted accordingly — DA-CRITICAL-3 and R1-W1 would still require repair.

**Submission readiness (separate from both):** Not ready. Compendium structure with duplicated ledgers, correction-manifest framing, unanonymized front matter as the default mode, unreconciled compute/carbon tables, 108 uncited references, and no compiled PDF for this root in the workspace all require production work before any submission.

We encourage the authors to (a) publish the corrected audit record promptly in an archival venue where correction is the contribution, after repairing R1-W1/R1-W2/R2-W2 and binding the two reviewed cells (R3-W2); and (b) treat the prospective operator-evaluation protocol as the basis of a future empirical paper. The field needs more audits conducted this honestly; it also needs them to culminate in at least one identified result.

## Revision Roadmap

Non-ranking list in manuscript source order. Tags indicate whether the item must be addressed for any future submission (Must-fix) or should be addressed to strengthen it (Should-fix). Remedies state the minimum acceptable change.

1. **DA-CRITICAL-1** (title, `paper.tex:45-46`) — **Must-fix.** Retitle to audit/correction scope, or support the comparative claim with a same-stack identified contrast manipulation.
2. **R3-W4** (front matter/abstract framing, `paper.tex:51-83`; `sections/abstract.tex:9-11`) — **Must-fix.** Rewrite for a cold reader; move #36320 genealogy and withdrawal narration to a separate audit log; prepare a properly anonymized submission mode.
3. **EIC-W3** (overlap disclosure) — **Must-fix.** Disclose relationships to submission #36320, the thesis, and all sibling corpus manuscripts; mark which displays are unique to this submission.
4. **DA-MAJOR-3** (salami/overlap auditability) — **Must-fix.** Provide the overlap map required under EIC-W3 so reuse of identical run rows across manuscripts is auditable. (Companion to item 3; no separate experiment needed.)
5. **EIC-W1** (contribution scale, abstract/introduction) — **Must-fix.** Retarget to an audit/correction venue or add a validated prospective result; do not submit the current core as a top-venue full paper.
6. **DA-CRITICAL-2** (two-cell record vs diagnostic claims, abstract) — **Must-fix.** Remove validated-triage/predictor language everywhere, or validate operating characteristics on held-out whole runs including positive, mixed, and recovery cases.
7. **DA-MINOR-1** ("Gradient Utilization," `sections/intro.tex:21-24`) — **Should-fix.** Rename to a non-causal label (mixed-group fraction); keep at most a one-line historical alias.
8. **R2-W1** (prior art, `sections/related_work_v2.tex:59-92`) — **Must-fix.** Reposition as reporting/audit note relative to ACR/dynamic-sampling lineage; claim no new measurable without a comparative demonstration.
9. **R2-W3** (related-work breadth and secondary numbers) — **Should-fix.** Cut to load-bearing lineage; verify or delete each quoted external effect size; delete uncited bibliography entries (108).
10. **R1-W4** (runner loss/masking, `sections/diagnostic_method.tex:46-54`) — **Must-fix.** Label all ledger rows with the actual noncanonical treatment including prompt-token loss and shifted targets, in table bodies/captions — not only in caveat sections.
11. **R3-W2** (receipt binding, `sections/diagnostic_method.tex:56-59`; `sections/evidence_identity_audit.tex:5-7`) — **Must-fix.** Publish an immutable manifest for the two reviewed cells (checkpoint revision, runner commit, prompt hash, reward version, raw group rewards) enabling exact ZVF recomputation.
12. **DA-MAJOR-2** (unbound #36320/commit/checkpoint receipts) — **Must-fix.** Satisfy item 11 and replace wildcard artifact references with exact URLs, revisions, and hashes; state plainly what remains unbindable. (Companion to item 11.)
13. **EIC-W2** (duplicated ledgers/structure, `tab:main_results` and its regroupings) — **Must-fix.** Keep one ledger; delete duplicate regrouping tables; move historical material to a labeled appendix/audit log; compile the PDF and meet the target page budget.
14. **R1-W1** (window mislabels, `paper.tex:412-417`, `396`, `1204-1210`) — **Must-fix.** Correct "logged" counts and "last-10" windows to actual trace lengths (6/6/6/3; Qwen3-235B 4-point trace) or recompute; add a generation gate tying labels to trace length.
15. **DA-CRITICAL-3** (retained Results vs quarantine standard) — **Must-fix.** Apply one standard: every retained number is bound and correctly windowed, or it is withdrawn. Encompasses items 14, 16, 18; listed separately because it was adjudicated as a blocking CRITICAL.
16. **R2-W2** (proxy task labels and team percentages, `tab:task_grpo`) — **Must-fix.** Remove or fully bind each cell; rename tasks to measured quantities; delete or evidence the "multiple seeds" consistency claim (`paper.tex:661-662`).
17. **R1-W6** (Wave 6 peak headlines, `paper.tex:573-591`) — **Should-fix.** Soften ordering claims to single-seed descriptive telemetry under unequal budgets, or add seeds at a declared token budget.
18. **R1-W2** (untraced SI/PTD 28-run analysis, `paper.tex:888-913`) — **Must-fix.** Publish the bound per-run table with denominator rules, or withdraw these numbers to quarantine.
19. **R1-W3** (ZVF lagged table, `paper.tex:985-1018`) — **Must-fix.** Bind the 33 run IDs; demote to a negative methodological illustration or remove; no predictive framing without whole-run holdout and nonlinear baselines.
20. **DA-MAJOR-1** (circular PTD/ZVF validation) — **Must-fix.** Reframe mechanically coupled correlations as definitional (or remove with items 18–19); never as validation. (Companion to items 18–19.)
21. **DA-MINOR-2** (BrowserGym smoke, artifact-scope table, bibliography padding) — **Should-fix.** Remove or move to an engineering appendix; each retained display must carry evidence, not integration existence.
22. **R3-W1** (no operator decision, `paper.tex:976-978`, `1262-1286`) — **Must-fix** for any "triage" claim. Provide early-window/late-endpoint decision analysis vs reward baselines with false-stop/recovery rates, or present signed ZVF strictly as a reporting field.
23. **R1-W5** (held-out GSM8K analysis and reuse, `paper.tex:1399-1413`; `ethics_statement.tex:45-50`) — **Should-fix.** Add prompt-paired base-vs-adapter analysis from existing artifacts; correct the dual-use uplift bound to reflect unmeasured change under wide uncertainty.
24. **R2-W4** (no same-stack comparator, `paper.tex:1424-1430`) — **Must-fix** for the current title; otherwise subsumed by item 1. Minimum: fixed-completions reward-intervention verification in the deployed runner and/or one matched-budget remedy arm.
25. **R3-W3** (compute/cost/carbon ledgers, `ethics_statement.tex:100-217`; `paper.tex:1720-1754`) — **Should-fix.** Replace incompatible historical tables with one cohort-bound ledger, or remove quantitative totals; do not retain a summable carbon total the text disowns.

Acronym check (advisory): did not run in this panel execution.

