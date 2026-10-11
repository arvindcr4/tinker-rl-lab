# Evaluator review and corrections

This review asks whether an M.Tech examiner can identify the individual contribution, follow the experimental design, inspect the implementation, and distinguish the measured result from its interpretation. It is an assistant review, not a PES mark allocation. No applicable official marking rubric was available in the inspected project materials. The review does not substitute a different institution's rubric or predict a grade.

## Issues corrected

| Risk to the evaluation | Correction in the delivered documents |
|---|---|
| A reversed Jensen-inequality explanation undermines the theoretical argument. | Chapter 6 now states that the convex Bernoulli null evaluated at the pooled mean understates the mean of the per-step plug-in values. |
| Perfect agreement was reported as valid Cohen's kappa despite constant labels. | The source helper's `1` is identified as a convention/error. Kappa is undefined in this case; only raw agreement is complete. |
| A bootstrap reanalysis was described as a new run. | The report identifies 4,000 resamples of the same measured sweep, not independent training evidence. |
| Layer-overlap and mediation claims exceeded what the scripts calculated. | The top-nine chance reference is 0.25, the all-step target includes step 1, and the correlation-product output is no longer described as causal mediation. |
| Matched-generation or proposed-control designs were presented as achieved controls. | Actual prompt exposure and sampling costs are explicit; design diagrams are labelled as targets, with executed departures stated. |
| Sampling filters were said to eliminate all wasted generation. | The report distinguishes zero ties in accepted batches from rejected and resampled completions. |
| Observational E11/E14 patterns were presented as causal effects of a generation cap. | Causal attribution is withdrawn. Finish-reason absence, difficulty and configuration differences remain visible. |
| E12 treated correlated rubric items as independent observations. | The independent-item Wilson interval is removed. The descriptive score and application-cluster comparison remain. |
| Entropy, small-sample inference and replacement-wave scope were overstated. | Marginal entropy sums are separated from joint information and unique identifiers; E4/TOST explanations and E1/E13 reporting are corrected. |
| C1 scientific references appeared to resolve when exact bindings were unavailable. | Appendix G includes a readable source key. Unavailable S10/S12/S13 exact-byte bindings are explicit nulls. |
| Objectives did not give the examiner a quick completion map. | Chapter 1 now maps O1–O8 to implementation, evidence and attained/partial status. |
| The deck emphasized limitations without enough concrete implementation or ownership. | Slide 6 demonstrates the actual registry comparison; slide 9 separates shared Semester 3 foundations from the individual continuation. |
| Some visible slide text contradicted its qualified notes. | Scope names, E11 causal wording, the mixed-provenance correlation and C1 event definition are aligned with the thesis. |
| The closing slide did not clearly state the delivered contribution or next experiment. | The deck closes on the implemented workflow and a tuned, exposure-matched, larger paired comparison as future work. |
| A two-page revision record distracted from the research and contained an orphan paragraph. | The revision record is shortened while preserving the publication and certification boundaries. |

The underlying model scores and original experiment files remain unchanged. Historical outputs that are mathematically invalid are retained only as identified historical outputs, with the unsupported interpretation withdrawn.

## Independent checks

Separate agents reviewed methods, results and contribution/presentation. A reviewer then cross-checked the mathematical corrections against the original analysis code and data. The independent slide review inspected every rendered slide at full size and prompted the final mixed-provenance and endpoint clarifications. Automated checks bind the rebuilt files to their receipts; the number audit covers registered claims rather than every numerical statement.

The package now includes [the actual registry example](defense/REGISTRY_EXAMPLE.md), which runs offline on two selected records. Its expected R5 result demonstrates a reporting conflict. It does not claim a predictive model or a controlled experiment.

## Colab follow-up

The subsequent executable source audit resolves the previously uncertain correlation membership. The reported 23 cells reduce to 17 finite cells, nine synthetic; the residual target mixes outcomes including late training reward. The thesis, abstract, synthesis and slides now withdraw the measured-held-out interpretation. See `colab_validation/README.md` for the exact inputs, runtime receipt and replay instructions.

## Remaining questions to prepare for

The historical studies still have small seed counts, residual confounding, an untuned PPO critic, missing execution sources and some assistant-reviewed references. Those limitations require better experiments or new evidence, not stronger wording. Rehearse why they narrow each conclusion and what concrete follow-up would address them.

The current PDF is a review copy. Required personal submission fields, author approval, signatures, certificates and portal requirements remain subject to the institution's actual process. The project materials do not establish that those steps have been completed.

The updated [defense preparation](DEFENSE_PREP.md) includes answers about kappa, Jensen's inequality and clustered evaluation. Use those answers to explain the corrections candidly if an examiner compares versions.

The same correction was propagated through the literature and methods chapters, contribution summary, research-question schematic, errata and viva answers. The synthetic nine-method panel is labelled as such at first mention. A contradictory plateau-timing sentence was replaced by the historical short-lead observation with its unvalidated monitoring scope. The objective map remains unnumbered so the canonical table and existing Table 1.1 references agree.
