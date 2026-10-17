# Review acceptance record

> **Author approval required — AI-tool declaration.** On 16 October 2026 the thesis's "Declaration on the Use of AI Tools" was redrafted by an AI agent to name the agentic tools the repository records (Claude Code agent teams, Codex, Grok CLI, Muse CLI, Antigravity/Gemini, `agy`, `squad`, Kimi, Jev, ChatGPT Pro, Gemini Deep Think) and agents launching cloud jobs. It is a draft. Arvind C R must check it against the actual workflow and PES policy wording, and approve or correct it, before any submission.

> **Provisional numbers.** The v2 same-stack GRPO-vs-PPO figures (thesis Table 6.2) use 7 of 10 PPO seeds; the exposure-matched PPO arm is pending. See `validation/samestack_v2/README.md` for the update path.

This is an assistant-checked evaluation review package prepared on 11 October 2026. It is not an institutional acceptance, a grade prediction, or independent authentication of the historical experiments. The deliverable includes the current thesis PDF and sources, an editable twelve-slide defense, speaking and viva preparation, selected evidence, an offline demonstration, and file-integrity receipts.

## Review coverage

The thirty areas below record what was inspected. They are review areas, not a claim that thirty agents ran. The host limited the team to four concurrent agents including the coordinator. Separate agents audited evidence, prepared evaluation material, tested software, and independently reviewed the scientific claims, numerical changes and rendered documents.

| # | Area | Resolution or explicit boundary |
|---|---|---|
| 1 | Thesis identity | Current PDF is explicitly marked as an evaluation review copy. |
| 2 | Revision date | Thesis and deck identify 11 October 2026. |
| 3 | Scientific scope | Controlled studies, actor evaluation and C1 remain separate. |
| 4 | Inherited ownership | Semester 3 group contribution retained. |
| 5 | Individual contribution | Semester 4 work mapped to implementation and evidence in the reviewer guide. |
| 6 | AI declaration | Redrafted 16 October 2026 to name the agentic tools recorded in the repository; **author approval required**. |
| 7 | Five reporting corrections | October 4 corrections incorporated without changing measured results. |
| 8 | Audit failure semantics | Unbound occurrences fail the check, including inconsistent aggregate status cases. |
| 9 | Audit counts | 967 registered occurrences: 945 passing and 22 context-only; 34 passing checks are printed-count calculations. |
| 10 | Source availability | Fresh inventory covers this thesis tree; historical evidence may be outside the ZIP. |
| 11 | Frozen releases | Original October releases and their top-level publication manifest remain unchanged. |
| 12 | PDF text | All 312 pages pass required-text, replacement-character and physical-bounds checks. |
| 13 | PDF layout | Earlier 312-page version inspected as layout thumbnails; changed pages and selected full-size spots reviewed after the Colab correction. Not every page read at full size. |
| 14 | References | Citation audit and selected provenance checked; not every reference independently authenticated. |
| 15 | Figures | Existing plots retained; the research-question schematic corrects the historical association claim. No new plot is presented as measured evidence. |
| 16 | Controlled comparison | v1 PPO gap superseded by the v2 rerun (512-token cap, warm-started critic, 1,319 items): no detectable GRPO advantage, equivalent within ±0.02. Provisional: 3 PPO seeds and the exposure-matched arm pending. |
| 17 | Inferential limits | Prediction, decontamination and causal serving-defect statements narrowed. Colab confirms 17 finite historical pairs, nine synthetic; predictive interpretation withdrawn. |
| 18 | Benchmark denominators | Judged rows, all dispositions, replacements and partial scopes remain explicit. |
| 19 | Missing outcomes | Missing grades and externally blocked contracts remain incomplete. |
| 20 | C1 endpoints | Frozen 11/64 kept separate from three assistant-reviewed numerical witnesses. |
| 21 | Deck alignment | Study budget and signal/causality language corrected; brief aligned with the deck. |
| 22 | Deck inclusion | Final editable PPTX included, hashed and successfully imported; all twelve rendered slides reviewed. |
| 23 | Talk timing | Twelve-minute route and ninety-second demo supplied; human rehearsal still needed. |
| 24 | Viva accuracy | Answers distinguish superseded, inconclusive and unmeasured results; the GRPO-vs-PPO answer now states the v2 null. |
| 25 | Demo execution | Default offline path and seven self-tests checked; live provider mode is outside acceptance. |
| 26 | Demo labeling | Four illustrative groups separate from the fixed-hash recorded reward artifact. |
| 27 | Software gates | Results and exact commands recorded in validation/SUMMARY.md and delivery_checks.json. |
| 28 | Numerical edge cases | Degenerate Student-t/TOST and Mann–Whitney cases corrected; nonfinite inputs fail explicitly. |
| 29 | Package integrity | Manifest binds included bytes; standard-library verifier included. Hashes do not authenticate original execution. |
| 30 | Institutional remainder | Author approval, signatures/certificates, portal requirements and actual submission remain human/institutional actions. |

## Independent reviews and corrections

Fresh assistant reviews found overstated prediction/causality language, a wrong study budget in the inherited deck, an Appendix J layout orphan, a stale aggregate audit status, and commands that were unavailable in the extracted package. The current files correct those findings. The Colab follow-up reproduced the frozen source audit in a separate Linux environment, corrected both historical correlation denominators and projection membership, and separated heterogeneous training outcomes from held-out accuracy. Its four acceptance tests pass; see colab_validation/README.md. The subsequent evaluator review also corrected the Jensen argument, undefined kappa, bootstrap identity, mediation interpretation and clustered E12 uncertainty; see EVALUATOR_REVIEW.md. A separate numerical reviewer compared the statistical boundaries to SciPy and analytic limits. Later nonfinite-input guards received another read-only review.

The attempted cross-family code-review agent could not start on this account. A separate supported-model reviewer completed that review instead. This is independent assistant review in separate contexts, not external human scientific review or a successfully completed cross-family review.

The PDF checks cover all-page layout thumbnails, independent full-size spot checks, and automated text/bounds inspection. The slides were reviewed as rendered images and imported with the presentation tooling; native PowerPoint rendering was not exercised. Full validation logs remain in the local checkout. The ZIP includes compact receipts and summaries.

## 16 October 2026 correction

An assistant-run publication-readiness audit recomputed the three headline claims from raw records and found the v1 PPO contrast refuted by its own corrected rerun, the ZVF U-shape largely explained by the binomial identity, and the stack exhibits confounded. The thesis, deck and guides were corrected accordingly (see README "What changed" and EVALUATOR_REVIEW.md). The number audit, PDF text/bounds check and manifest verification were rerun; changed PDF pages and the five edited slides were inspected as rendered images. This remains assistant review, not external human scientific review.

## Delivery boundary

No benchmark score was invented or improved by editing this package. No model training or native benchmark scoring was launched. A bounded ephemeral Colab CPU runtime reanalysed selected numeric records; the CLI confirmed cleanup. Original benchmark completion and external infrastructure access remain governed by their existing evidence contracts. The package supplies a concrete basis for evaluating the work that was implemented and measured; it cannot guarantee the grade.

Before official submission, the author must review and approve the final claims, contribution and AI-tool declarations; obtain required approvals and certificates; confirm the exact portal format and deadline; and submit the requested documents. See SUBMISSION.md in the repository or ZIP root.
