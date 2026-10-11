# Tinker RL Lab: defense preparation

Prepared 11 October 2026 for the PES project evaluation. Use the [current thesis review copy](thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf) alongside the [corrected twelve-slide deck](defense/TinkerRL_PES_Defense_2026-10-11.pptx) and its [full source-aligned brief](defense/DEFENSE_BRIEF.md). This is a speaking and rehearsal aid, not a new experimental result. The review copy incorporates the [4 October draft's five reporting corrections](../public_revision_2026-10-04/thesis/README.md); it is not a certified institutional submission.

## The opening to rehearse

“An RL result depends on more than an algorithm's name. Model choice, sampling, reward parsing, and evaluation coverage can change the outcome. I built a measurement workflow that makes those conditions visible, tested what its diagnostics can support, and used it to report a fourteen-lane evaluation without turning missing evidence into a score.”

Then state the ownership boundary: “Semester 3 established the shared environment as a six-person group. My Semester 4 continuation developed the diagnostic and reporting studies, expanded the experiments, and produced the campaign analysis and this thesis.” Point to [PROJECT_HISTORY.md](../../PROJECT_HISTORY.md) if asked. Do not claim sole authorship of the inherited infrastructure.

## Twelve-minute talk

| Time | Point to make | Evidence to show |
|---|---|---|
| 0:00–0:45 | State the attribution problem and the contribution in the opening above. | Title, research question, ownership boundary. |
| 0:45–2:00 | Separate the two instruments: controlled P1–P8 studies ask what changes when a factor changes; E1–E14 asks what one checkpoint scores under named contracts. | The experiment/campaign distinction. Only Tinker and TRL completed the compared framework runs; this is not a completed seven-library benchmark. |
| 2:00–3:30 | Explain GRPO's within-group comparison. Equal rewards give no relative advantage whether every answer is right or wrong. That motivates ZVF. | Eight equal rewards; then the stored-tensor values 0.130, 0.190, 0.155. ZVF measures signal availability, not accuracy. |
| 3:30–5:00 | Show the implementation contribution: an eight-item minimum report, registry comparison, and named benchmark terminal states. | One stack difference and one receipt. Describe what each field prevents an evaluator from confusing. |
| 5:00–7:00 | Give the strongest controlled result with its limitation in the same sentence. | Five paired seeds: GRPO G8 − untuned PPO = +5.0 pp, paired-t CI [+1.5, +8.5], p = 0.016; exact sign-flip p = 0.0625. No general algorithm ranking. |
| 7:00–8:15 | Demonstrate the actual registry comparison on slide 6. Explain the difference between an explicitly zero KL coefficient and an unreported value. | Open DAPO record: 0.0; managed DAPO record: null. The implemented CLI returns R5. See [the reproducible example](defense/REGISTRY_EXAMPLE.md). |
| 8:15–9:30 | Use two campaign rows to explain why the harness matters. | VerilogEval 129/312 with 150 extraction failures among 183 failures. WebArena 90/812 with 108 ungraded tasks. No average across suites and no causal training claim. |
| 9:30–10:15 | Introduce C1 as a diagnostic of sampling and reference quality. | Frozen 11/64 and three separate numerical witnesses. The selected cohort is deterministic and review was assistant-assisted. |
| 10:15–11:15 | Restate what the candidate delivered and what another researcher can inspect. | Recomputable diagnostic, reporting standard, registry, terminal-state ledger, and artifact checks. P1–P8 are internal reports, not eight accepted papers. |
| 11:15–12:00 | Name the next experiment that would answer the main open question, then transition to the demo. | A tuned/warmed-up PPO critic, matched prompt exposure and token budget, more seeds, and a locked comparison protocol. This is proposed work, not a measured result. |

Aim to explain one mechanism, one controlled comparison, and one benchmark failure in depth. Keep extra decimal places, the full lane list, and C1 case detail for questions. The project is easier to assess when each claimed contribution has an inspectable example.

## Ninety-second demonstration

Before the room, run these from the repository root and open the generated HTML:

```sh
python3 submission/demo/run_demo.py --self-test
python3 submission/demo/run_demo.py
```

The page is `submission/demo/output/demo_report.html`. It opens directly without a server or network. Follow the [existing runbook](../../submission/demo/DEFENSE_RUNBOOK.md) for a live local server and failure handling.

| Time | Narration |
|---|---|
| 0–20 s | “This is a deterministic demonstration of the mechanism and artifact integrity. It uses synthetic reward groups plus a separately checked recorded artifact.” |
| 20–50 s | “Subtract the group mean and divide by its population standard deviation. Mixed rewards produce positive and negative advantages; both equal-reward examples produce zero relative signal.” |
| 50–75 s | “This part recomputes 80 recorded rewards and checks their input SHA-256. The displayed mean is 68.75% for this artifact.” |
| 75–90 s | “That verifies the displayed mechanism and arithmetic. It does not retrain the model, rerun a benchmark, or demonstrate generalization.” |

If integrity validation fails, show the failure and inspect the input difference; do not bypass the hash. Keep the HTML and deck open before presenting. A recorded output is a fallback, not evidence of a new successful execution.

## Questions to answer in one or two sentences

**What is original here?** The integration of a recomputable group-signal diagnostic, stack-conditioned reporting and registry, and an evaluation ledger with explicit scope and terminal states. The novelty claim for the GRPO-specific datasheet is bounded by the recorded literature search through July 2026; it is not proof that no related work exists.

**Did GRPO beat PPO?** Only a suggestive, baseline-specific gap on the unsaturated 1.5B study: +5.0 pp against a minimal untuned critic. Five paired seeds give t-test p = 0.016 but exact sign-flip p = 0.0625, so I do not claim general superiority.

**Why can two valid tests disagree?** The parametric test uses assumptions about the paired-difference distribution. With five pairs, an exact two-sided sign-flip test cannot give less than 2/32 = 0.0625 even when all differences point the same way.

**Why are the null results useful?** They identify what this design cannot resolve and prevent a saturated task or small sample from being presented as proof of equivalence. They also identify the concrete change needed in the next experiment.

**Does high ZVF mean the model has failed?** No. All-correct and all-wrong groups both have zero within-group variance. ZVF identifies unavailable group-relative signal; it needs the reward level and task context to be interpreted.

**Does ZVF predict final accuracy?** Predictive validity remains unestablished. The historical analysis mixes synthetic cells and heterogeneous targets, including late training reward. It cannot validate held-out prediction or prove a universal absence of association.

**What does 11/64 mean?** C1's frozen endpoint on the deterministic first 64 eligible cases. The three numerical witnesses are a separate assistant-reviewed classification, not a replacement score, training gain, or whole-benchmark recovery rate.

**Which Omni-MATH result is correct?** Both have explicit denominators: 2,271/4,426 = 51.31% over judged rows, and 2,271/4,428 = 51.29% over all dispositions. The latter includes two unjudged rows as failures.

**Why are some benchmark scores low?** They are outcomes of a complete actor-and-harness configuration. For example, VerilogEval had 150 extraction failures; its 1,024-token cap with thinking enabled is a plausible cause, but missing per-sample finish reasons prevent a definitive causal attribution.

**Can you average the fourteen scores or infer a training improvement?** No. Their tasks, scopes and denominators differ. The campaign measures one actor, and the later small paired base-versus-trained comparison is inconclusive in every lane.

**Can a reviewer reproduce everything offline?** No. The offline package demonstrates mechanisms, selected arithmetic, and artifact identity. Training, private benchmark payloads, missing historical execution sources and provider services require separate access or cannot be replayed in their original form.

**Is the project unfinished because some native benchmarks are blocked?** Those contracts are incomplete, and the ledger says so. The demonstrated project contribution is the measurement and reporting system, supported by completed experiments and explicitly scoped partial results; unavailable contracts are limitations, not invented successes.

**What would you improve first with another experiment?** Fix the controlled comparison: tune and warm up the PPO critic, equalize exposure and token budgets, preserve identical evaluation items, and increase the seed count. Freeze that protocol before examining results.

## Questions prompted by the evaluator review

**Why is the historical kappa not a valid reliability result?** Both classifiers gave the same label to every trace. Raw agreement is complete, but chance agreement is also one, making Cohen’s kappa undefined. The old helper returned one by convention; the thesis now states that error explicitly.

**Which way does the Jensen inequality go?** The Bernoulli all-equal probability is convex in success probability. Evaluating it at a pooled mean understates or equals the average of the per-step plug-in values. This is a model-based baseline, not an exact latent probability for each prompt.

**Why remove the E12 interval?** The 151 rubric items belong to six applications. An independent-item Wilson interval treats those correlated items as independent evidence. The score remains descriptive, and the paired comparison uses the six application clusters.

**What changed after review?** The experiment scores stayed fixed. The report corrected statistical interpretations, evidence bindings and scope descriptions. A checksum verifies the delivered files; it does not prove the scientific claims.

## Final preparation

- Rehearse to twelve minutes plus ninety seconds; practice the five-seed test explanation without reading.
- Locate Table 1.1, Table 6.1, Table 8.B and Table C.3 in the chosen thesis PDF before presenting.
- State the shared-versus-individual contribution and the AI-tool assistance accurately.
- Complete the author approval, required signatures, any institutional similarity certificate, portal format/deadline confirmation and upload receipt listed in [SUBMISSION.md](../../SUBMISSION.md). Preparing this defense does not complete those institutional steps.

## Three questions to practise without notes

**Why is ZVF a contribution if its formula is simple?** The algebra explains the mechanism. The implementation contribution is logging it consistently, recomputing it from stored rewards, relating it to other recorded quantities, and demonstrating where its interpretation fails. I do not claim to have invented group normalization.

**Do equal generations imply a fair compute comparison?** No. The unsaturated study uses 8 prompts times 8 completions for GRPO and 64 prompts times 1 for PPO. Both generate 64 completions per step, but prompt exposure, response length and optimizer work can differ. The result remains configuration-specific.

**Does the C1 Wilson interval estimate a population recovery rate?** No. It summarizes the frozen proportion under a binomial calculation. The deterministic first-64 selection and uncertain references prevent a general population claim. The 11 frozen events and three conservative numerical witnesses remain separate.

**What did the Colab check improve?** It traced the pooled correlation to its source rows: six missing-ZVF cells were dropped, nine retained cells came from simulation, and the target was not a common held-out metric. The report now withdraws that interpretation while retaining the historical output. This is a reproducible source audit, not a new training result.
