# Prior-art and venue-fit audit

**Audit date:** 2026-08-02  
**Scope:** the live NeurIPS submission record, July flagship artifacts, E1 re-audit,
S1 conformance package, next-submission design, and current primary literature and
venue policies. This is a publication-positioning audit, not evidence that a learning
claim has been established.

## Decision

The only paper story with credible venue fit today is:

> **Terminal homogeneity is decision-insufficient, and implementation claims need
> executable conformance:** a fail-closed S1 certificate, with the registered r4-2
> gate failure as the case study.

The best route is **TMLR after the current NeurIPS review ends or the submission is
withdrawn**. The story is venue-aligned today, but it must not be submitted today:
the current NeurIPS paper overlaps it. The local anonymous S1+r4-2 package now passes;
timing and venue overlap are the remaining blockers. It must not be described as a
public artifact until it is released after review.

A standalone zero-variance/reward-contrast mechanism paper, a controller-efficiency
paper, and a literature-hypothesis-index paper are no-go as headline submissions on the
current evidence. A future NeurIPS Evaluations & Datasets paper becomes credible only
after the conformance artifact is public, versioned, independently runnable, and shown
to change at least one empirical interpretation.

## Evidence snapshot from the repository

| Artifact | What is actually supported | Publication consequence |
|---|---|---|
| `zvf-program/flagship/paper/CLAIM_AUDIT.md` | S1 intended integrations pass; r4-2 has only six scientific acceptances and no positive control. | Supports methods/reproducibility and a feasibility postmortem, not a causal training or controller claim. |
| `zvf-program/flagship/s1/` | Float64 reference traces, frozen TRL and verl fixtures, intended adapters, native-semantics probes, and content-addressed receipts. No generation, model training, or optimizer update is performed. | Strongest distinctive artifact; it certifies implementation semantics, not learning quality. |
| `autoresearch/reason-260728-0744/E1_STATISTICAL_REAUDIT.md` | The earlier DAPO `DISAPPEARS` verdict was unsafe: exact finite-sample MDE exceeds the 0.01 margin and the locked multiplicity procedure was not used. | The historical positive-looking audit claim must not be reused. |
| Current E1 R08 revision | The correction replaces the normal MDE with exact paired-t power, executes BH over the four comparisons, regenerates all four verdicts as `INCONCLUSIVE`, and compiles the revised manuscript. The focused nine-test statistics suite passes locally. | This repairs inference hygiene; it does not create equivalence, a winner, or a stack-causality result. Freeze the complete release artifact before submission. |
| `zvf-program/audit/reproducibility_audit.tex` | Forty reconciled arm-seed units on one model, one task, one stack, and one short horizon; all comparisons are now inconclusive. | A useful protocol case study, but too narrow for a general algorithm ranking. |
| `zvf-program/next-submission/DESIGN.md` | A matched maximum-group controller design and explicit cost/held-out gates exist. Required confirmatory cells have not produced a complete claim-bearing result. | Design/preflight evidence only; no efficiency-quality claim is available. |
| `platform_hybrid/registry/` and `zvf-program/registry/` | A source-provenanced machine-readable stack/variant catalog exists. The separate 50-idea catalog is generated ideation. | Registry can be a companion resource; the idea catalog is not research evidence. |

## Candidate contribution audit

### 1. Zero-variance group or reward-contrast mechanism — severe collision

The broad mechanism is already occupied from several directions:

- DAPO dynamically filters groups with accuracy 0 or 1 and oversamples until a batch
  contains informative groups ([DAPO](https://arxiv.org/abs/2503.14476)).
- GRESO predicts and skips uninformative prompts before rollout and reports rollout and
  end-to-end speedups ([GRESO](https://arxiv.org/abs/2506.02177)).
- AERO explicitly treats all-correct/all-wrong groups as zero-advantage and allocates
  rollout budget adaptively ([AERO](https://arxiv.org/abs/2602.14338)).
- RL-ZVP shows that zero reward variance need not imply zero useful learning signal once
  the objective supplies an entropy-guided advantage
  ([RL-ZVP](https://arxiv.org/abs/2509.21880)).
- Gradient Starvation gives an exact group-mean analysis, a Jensen heterogeneity result,
  and multi-seed experiments ([Gradient Starvation](https://arxiv.org/abs/2605.07689)).
- A June 2026 analysis connects GRPO, Dr.GRPO, and DAPO through the group standard
  deviation and makes unanimous-group silence explicit
  ([GRPO, Dr.GRPO, and DAPO in One Number](https://arxiv.org/abs/2607.00152)).
- Finite-sample GRPO theory already studies estimator error, asymptotics, and group-size
  scaling ([Demystifying GRPO](https://arxiv.org/abs/2603.01162)).

Therefore none of these is a credible novelty claim by itself: the Bernoulli homogeneous
group probability, “no reward contrast means no centered advantage,” heterogeneous
prompt Jensen effects, group-size tradeoffs, dynamic resampling, or early skipping.

The narrow residual question is **objective-conditional boundary action**: when should a
system skip, resample, enlarge the group, or preserve a zero-variance group because a
different objective still extracts signal? That question could support a later empirical
paper only with complete matched-cost trials, held-out outcomes, reward-cardinality
strata, and more than one task/stack. Those results do not exist here today.

### 2. Implementation conformance plus provenance — strongest residual

Prior work already establishes that implementation details are scientifically material:
Dr.GRPO isolates normalization and length biases
([Dr.GRPO](https://arxiv.org/abs/2503.20783)); group-relative REINFORCE analysis separates
importance sampling, clipping, and regularization
([Group-Relative REINFORCE](https://arxiv.org/abs/2509.24203)); and controlled reasoning
evaluation shows that decoding, seed, prompt, hardware, and software choices can change
conclusions ([A Sober Look at Progress in Language Model Reasoning](https://openreview.net/forum?id=90UrTTxp5O)).

In this scoped primary-source search, I found no direct equivalent of S1's combined
package: exact float64 losses, masks, ratios, actions, and flattened gradients across
frozen TRL and verl fixtures, plus intended/native semantic separation and a
content-addressed receipt. That is a defensible residual, but it must be worded as a
scoped nearest-neighbor finding rather than a literature-exhaustive novelty theorem.

The publishable unit is the **executable semantic certificate and its failure modes**.
Hashes and provenance are supporting infrastructure, not an independent scientific
contribution. The paper must demonstrate that the certificate catches a material
mismatch and changes an empirical interpretation or prevents an invalid comparison.

### 3. Same-stack algorithm-claim audit — promising protocol, blocked result

Holding a stack fixed is useful, but it does not identify the original published gain as
a stack effect. E1 can say only what survives under its declared model, task, horizon,
and implementation. Its current 40-run case study has insufficient scope for a family
ranking, and all four corrected comparisons are inconclusive. Non-significance is not
equivalence.

The residual contribution is a **power-aware, fail-closed survival protocol** that joins
treatment fingerprints, stack fingerprints, public traces, exact small-sample analysis,
and executable conformance. To become a credible TMLR paper, it needs the corrected R08
analysis frozen into a public anonymized reproduction package and at least one comparison
between a faithful-recipe stratum and the common-stack stratum. Broader claims need more
models/tasks/horizons; the protocol paper itself can remain modest if its lessons are
generalizable and actionable.

This is a separate candidate paper. The final Route-1 spine below does not merge
E1 with the flagship conformance and registered-feasibility study.

### 4. Literature hypothesis index — companion only

A machine-readable, source-cited variant/stack-delta registry could be useful if it has a
declared search boundary, extraction schema, source-fidelity checks, version governance,
and a utility evaluation. A raw list of generated hypotheses or expected effects has no
empirical standing. It cannot establish novelty, priority, or likely impact, and it
should not be described as a systematic review without a review protocol.

TMLR allows surveys when they expose new connections, trends, or open problems, but its
AI-tool policy makes authors responsible for the work and expects ideas, claims, and
results to be human-sourced ([TMLR FAQ](https://jmlr.org/tmlr/faq.html)). The index is
therefore best used as a transparent companion to the conformance paper, not its main
claim.

## Narrow paper that remains credible

**Working title:** *Same Terminal Signal, Different Action: Fail-Closed Conformance and
Decision Insufficiency in Group-Relative RL*.

Credible contribution set:

1. Define a framework-neutral objective trace and certificate covering the exact
   quantities required to identify a group-relative update.
2. Instantiate it against at least two real framework semantics, explicitly separating
   native behavior from the intended controlled treatment.
3. Make the receipt fail closed under code, fixture, adapter, or treatment drift.
4. Use r4-2 as a registered failure case: high cosine looks reassuring, but the frozen
   joint predicate fails at 69/100, below 95/100, and blocks an equivalence claim.
5. Prove the restricted decision-insufficiency result without turning it into a claim
   about an untested controller.

Required nonclaims:

- no new optimizer or universal zero-variance mechanism;
- no algorithm winner, equivalence, or proof that prior gains were stack effects;
- no controller efficiency/quality improvement without completed matched-cost cells;
- no learning claim from tests, receipts, preflights, hashes, or accepted jobs;
- no exhaustive-survey claim for the literature registry.

## Current venue-policy audit

### NeurIPS 2026

- The main-track deadline was **May 6, 2026**, so no new 2026 submission route exists
  today ([Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)).
- The conference is double blind; the paper, supplement, and linked code/data must be
  anonymized. Preprints are allowed, including non-anonymous ones, but authors must not
  advertise them as under review at NeurIPS
  ([Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)).
- Parallel archival work with overlapping authors counts as prior work. If publishing
  either submission would make the other incremental, both can be rejected; thin
  slicing and overlapping main/E&D submissions are also barred throughout review.
- Papers made public after **March 1, 2026** are contemporaneous rather than a rejection
  basis, but must still be cited and compared. This protects priority treatment for the
  current 2026 review; it does not erase those papers as prior art for TMLR or a future
  NeurIPS submission.
- Main-track review accepts general, theoretical, use-inspired, concept/feasibility, and
  negative-result contributions, judged on quality, clarity, significance, and
  originality. The current artifact-only story lacks the evaluation needed for a strong
  main-track significance case.
- E&D explicitly welcomes rigorous reproduction, auditing, stress testing, and tools or
  frameworks that improve evaluation claims
  ([E&D call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)). Its 2026
  deadline was also May 6 and code/data had to be final at submission
  ([E&D FAQ](https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ)).
- The current handbook does not require a rejected-paper resubmission dossier; reviewers
  are told not to seek old reviews actively. Resubmission is nevertheless possible only
  after the prior archival review has ended and the new paper obeys overlap rules.

### TMLR

- TMLR explicitly covers reproducibility studies, analysis methods, and surveys. It
  forbids reuse of text, figures, or results from a published, accepted, or parallel
  archival paper; preprints and nonarchival workshops are allowed
  ([Editorial Policies](https://jmlr.org/tmlr/editorial-policies.html)).
- Acceptance asks whether claims are accurate, convincing, and clear and whether some
  audience would be interested. Novelty and significance are not acceptance criteria;
  a proper reproducibility report with generalizable, actionable lessons can qualify
  ([Acceptance Criteria](https://jmlr.org/tmlr/acceptance-criteria.html)).
- Review is double blind and rolling. The submission and supplement must be anonymized;
  a preprint can exist under either identity, but the anonymous submission must not link
  to the named version ([Author Guide](https://jmlr.org/tmlr/author-guide.html)).
- A rejected TMLR resubmission must link the prior submission and explain changes.
  Revealing author identity after rejection prevents a revised TMLR resubmission.

### Overlap consequence on 2026-08-02

The live NeurIPS submission already combines reward contrast, algorithm-label
skepticism, and diagnostic auditing. Submitting the conformance/audit story to TMLR while
that paper remains under review would create prohibited parallel archival overlap under
both venues' rules. The safe choices are to wait for the September 24 NeurIPS decision
or withdraw before submitting elsewhere.

If NeurIPS accepts, a TMLR paper must be genuinely distinct and cannot be an expanded
version reusing the accepted text, figures, or results. If NeurIPS rejects, a substantially
reframed conformance/reproducibility paper can go to TMLR after review ends; the rejected
submission may remain a preprint. No post-submission artifact can retroactively turn the
reviewed NeurIPS manuscript into a different paper.

## Ranked publication routes

| Rank | Route | Credibility today | Gate |
|---:|---|---|---|
| 1 | **TMLR after NeurIPS resolution: conformance method + fail-closed audit case study** | Best fit; modest novelty is acceptable and reproducibility is in scope. Do not submit while the current NeurIPS paper remains live. | Wait for the NeurIPS decision or withdraw; keep the failed joint-mechanism result frozen; use the verified anonymous artifact. |
| 2 | **Future NeurIPS E&D: executable GRPO conformance benchmark** | Strong topical fit once it is a maintained evaluation object rather than a code appendix. | Public final code/data, versioned framework fixtures, independent reproduction, coverage/utility evidence. |
| 3 | **Future NeurIPS main: formula-to-gradient-to-outcome paper** | Potentially strong, but no claim-bearing outcome evidence exists today. | Complete multi-task/multi-stack causal chain or matched-cost controller campaign with held-out gates. |
| 4 | **TMLR survey/resource: semantic-delta registry** | Plausible separate resource, weaker than Route 1. | Systematic boundary, extraction reliability, governance, new synthesis, and user evaluation. |
| 5 | **Zero-variance theory/controller paper** | No-go: direct novelty collisions and missing confirmatory evidence. | A genuinely new objective-conditional result plus complete learning/cost evidence. |
| 6 | **New NeurIPS 2026 submission** | Impossible; deadline closed. The existing paper can only complete its current review. | Future cycle or a different venue after overlap clears. |

## Minimum submission gate for Route 1

1. Preserve the frozen 95/100 joint mechanism rule and report the observed 69/100 as a
   failed gate in the abstract, results, limitations, and verifier.
2. Remove every implication that high cosine, non-significance, or an incomplete matrix
   proves equivalence or locates a gain in a framework.
3. Package S1 and the r4-2 numerical projection anonymously with exact commands, hashes,
   and a clean-extraction check.
4. Show the interpretive consequence: cosine alone suggests near-collinearity, while the
   registered cosine/relative-L2 rule blocks the equivalence claim.
5. Produce a claim-to-artifact table separating proofs, implementation conformance,
   stored gradient relations, held-out quality, charged cost, and non-claims.
6. Freeze an overlap map against the final NeurIPS record before TMLR submission.

Items 1--5 now pass in the local package. Item 6 cannot pass while the overlapping
NeurIPS review remains active. Until that legal timing gate clears, the output is a
methods/reproducibility preprint and registered-feasibility postmortem, not a new
algorithm, winner, controller improvement, or proof of learning.

## Search boundary

The novelty check used primary paper records on arXiv/OpenReview and official venue
policies available on 2026-08-02. It targeted the nearest mechanism, implementation,
audit, and policy neighbors; it is not a systematic-review completeness claim. All 18
external links below the citation threshold were checked in batches of at most 20.
