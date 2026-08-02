# Review of the 18-paper portfolio

Date: 2026-08-02

## Decision

I would not submit any of these 18 manuscripts unchanged.

The 18 files compress into about six contribution families, several short
derivatives, and one 239-page internal compendium. The most useful current
pieces are:

1. **R08**, as a bounded reproducibility and statistical-audit case study;
2. **R02**, after a severe cut, as a workshop note on why pooled ZVF summaries
   can mislead;
3. **R04**, as an artifact or benchmark package once every public and anonymous
   release claim is checked; and
4. the reporting and registry work in **P05/P06/R06/R07**, combined into one
   resource rather than four papers.

The best archival submission in the repository is still the separate flagship
paper, *Same Terminal Signal, Different Action*. That manuscript is outside this
18-file roster. Its TMLR route remains blocked while the overlapping NeurIPS
submission is active.

## What I checked

- The roster is the current `platform_hybrid/paper/PAPERS_README.md`, with IDs
  P01-P08, R01-R08, U01, and N01.
- Every source file and every recursively included source was read into the
  review corpus: 414 include-closure mentions, 329 distinct source files,
  57,339 source lines, and 3,056,844 bytes.
- All 18 manuscripts were rebuilt from the live checkout. All 18 PDFs compile,
  all input hooks resolve, and the compile logs contain no unresolved citations
  or references.
- The set totals 868 compiled pages. That is a sign of accumulated program
  documentation, not evidence of 18 publication-sized contributions.
- The source audit found no active rendered figure fallbacks, no duplicate
  labels, and no TODO markers. Sixteen source files contain the word
  `placeholder`, but these are dormant fallback branches or declared registry
  backlog entries, not missing figures in the current PDFs.
- The seven-gram overlap scan finds substantial reuse among R02/R03/R04
  (Jaccard 0.427, 0.308, and 0.306) and between U01 and P01/P04 (0.245 and
  0.247). This confirms that those files cannot be presented as independent
  contributions without an explicit overlap account.

The machine-readable inventory, hashes, expanded source, extracted text, and
similarity matrix are in `audits/paper_portfolio/`.

## Portfolio map

| Bucket | Manuscripts | Publication treatment |
|---|---|---|
| Bounded audit | R08 | Keep as a case study or companion artifact. Do not turn inconclusive comparisons into a ranking. |
| ZVF and group-size analysis | P02, P03, R02, R03, R05 | Keep R02 as a short note; merge useful theory and group-size material; retire the rest as separate submissions. |
| Scaling and benchmarking | P01, R01, R04 | R04 may become an artifact. P01 needs matched evidence. Retire R01. |
| Length and rollout control | P04, P07, N01 | Park until prospective, matched-cost outcomes exist. |
| Reporting and registry | P05, P06, R06, R07 | Combine into one public resource with external entries and a user study. |
| Fraud detection | P08 | Separate project. Park until the evaluation is like-for-like and uses credible data. |
| Internal compendium | U01 | Archive as a thesis and evidence reference, not a venue submission. |

## Paper-by-paper verdicts

### P01 - Scaling laws

The useful result is negative: model size, stack, recipe, and budget move
together in the available anchors, so the differences cannot be assigned to
scale. The frontier anchors are single-seed and confounded. Recent direct work
also occupies the broad scaling claim, including
[Scaling Behaviors of LLM RL Post-Training](https://arxiv.org/abs/2509.25300),
[Predictive Scaling Laws for Efficient GRPO](https://arxiv.org/abs/2507.18014),
and [Where Should RL Post-Training Compute Go?](https://arxiv.org/abs/2607.13389).

**Verdict:** Do not send the 45-page scaling paper. Cut it to a workshop note
about failed identifiability or keep it as a thesis chapter.

### P02 - Zero-variance fraction

The exact relation `pass@G - p^G = 1 - ZVF` is a useful accounting identity.
ZVF remains descriptive under the stated sampling model; it is not a causal
mechanism or a calibrated predictor of training failure. The held-out
association is weak, and the same value can describe mastery or incapacity.
Existing systems already react to homogeneous groups, including
[DAPO](https://arxiv.org/abs/2503.14476),
[GRESO](https://arxiv.org/abs/2506.02177), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Merge the identity and caveats into R02 or the flagship appendix.

### P03 - Group size

Direct measurements stop at G=16. G=32 is reconstructed, while G=4 appears only
under a declared `(1-ZVF)/sqrt(G)` proxy and cohort. The DPO bridge is an
interpretation rather than a measured gradient equality. Small- versus
large-group behavior also has direct recent treatment in
[It Takes Two: GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977).

**Verdict:** Merge the observed-versus-proxy table into R02. A separate paper
needs a prospective, fixed-cost sweep over multiple tasks.

### P04 - Length bias

Under the 200-token cap and short horizon, the small GRPO/Dr.GRPO comparison did
not show a clear length effect. The arms have five and three seeds, and the cap
suppresses the regime studied by
[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783).

**Verdict:** Park it. A negative-results note needs an uncapped, matched-seed
replication and a claim bounded to that setting.

### P05 - MIN-REPORT-RL

The checklist is useful, but the headline 17x exhibit changes both backend and
base checkpoint. It shows under-specification rather than backend causality.
Reporting standards and experiment metadata also have long prior art, from
[Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560)
to [SLM Lab](https://arxiv.org/abs/1912.12482),
[OpenRLHF](https://arxiv.org/abs/2405.11143), and
[automatic ML provenance systems](https://www.amazon.science/publications/automatically-tracking-metadata-and-provenance-of-machine-learning-experiments).

**Verdict:** Merge it with P06/R06/R07. Require external entries, extraction
agreement, and a user decision study before writing a resource paper.

### P06 - GRPO registry

The source-provenanced table and explicit unknown states are useful. The current
entries still come mainly from this program, several fields are backlog items,
and there is no extraction-reliability or user-utility result.

**Verdict:** Merge with P05 and R07. Release it, recruit outside entries, freeze
the search boundary, and measure extraction errors before a resource paper.

### P07 - ZVF controller

The fixed-token comparison is a reasonable plan, but no prospective controller
advantage exists in the repository. Retrospective Pareto language overstates the
evidence. Nearby methods already report adaptive-rollout or pruning results:
[GRESO](https://arxiv.org/abs/2506.02177),
[CPPO](https://arxiv.org/abs/2503.22342),
[DARS](https://arxiv.org/abs/2508.13755), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Park it until the frozen bakeoff measures quality, charged tokens,
wall time, and failure cases against the closest methods.

### P08 - Fraud detection

The 50,000-row exercise is synthetic. XGBoost and LLM AUC come from different
held-out splits, so the 85x cost comparison and hybrid recommendation are not
like-for-like tests. Tabular LLM comparisons already include
[TabLLM](https://arxiv.org/abs/2210.10723).

**Verdict:** Keep it outside the RL portfolio. Reopen it with one frozen split,
a credible fraud benchmark, repeated uncertainty, and a tested hybrid router.

### R01 - ACM benchmark variant

The comparison mixes LLM-native stacks with classic RL systems using small MLPs
and discrete environments. Its gaps combine task encoding, model class,
implementation, and hardware.

**Verdict:** Retire this derivative. Keep its comparison-boundary warning in
R04.

### R02 - NeurIPS ZVF variant

This is the cleanest ZVF result: a pooled association collapses or reverses
within GSM8K strata. It remains retrospective and cannot show prediction or a
training benefit.

**Verdict:** Cut it to an 8-10 page workshop note, publish the exact strata and
code, and remove controller language.

### R03 - NeurIPS workshop artifact

The walkthrough is readable, but seven-gram overlap is 0.427 with R02 and 0.306
with R04. Its evidence already lives in those papers.

**Verdict:** Merge it into R04's documentation or use it as a nonarchival
handout.

### R04 - NeurIPS DNB benchmark

The treatment fingerprints, 79-run scope, seven-library map, and reproduction
commands could form a useful artifact. Several rows and release statements are
still incomplete. Pinned containers, anonymous code, and `make reproduce-main`
need a clean-machine public test.

**Verdict:** Make this the artifact paper, merge R03 into it, and publish an
exact runnable/documented/missing coverage table.

### R05 - ZVF theory

T1 is a binomial interval result. T2 counts rolls to a nonzero-gradient event,
not learning progress. T3 is a proxy tie at G in {2,3}. The word `universal`
exceeds the stated objective and sampling assumptions. The PDF also discloses
drafting assistance.

**Verdict:** Move the correct statements to an appendix and keep the disclosure
truthful. The current contribution belongs inside a larger paper.

### R06 - MIN-REPORT position

This readable P05 condensation repeats the confounded 17x exhibit and adds no
external adoption. The NeurIPS 2026 Position Paper track required substantial
human writing and a declaration.

**Verdict:** Fold its clearest prose into the reporting/registry resource. Keep
all truthful assistance disclosures.

### R07 - Living GRPO registry

This is the clearest resource statement for P06, but it has no independent
evidence. Its compile log also has 167 underfull boxes.

**Verdict:** Use it as P06's README and fix the tables after external-entry and
content gates pass.

### R08 - Reproducibility audit

This is the strongest bounded paper in the roster. It reconciles 40 arm-seed
units and applies exact paired-t power with the registered four-test BH
correction. All four verdicts are `INCONCLUSIVE`; the old DAPO label remains
only in the correction history. One model, task, stack, and 30-step horizon
cannot support a ranking or equivalence claim.

**Verdict:** Keep it as an audit case study. Add a faithful-recipe stratum and a
second task for an archival version, and keep every inconclusive label.

### U01 - Umbrella benchmark

The 239-page file preserves the program but reuses large blocks from P01 and P04
and mixes several questions and contribution types.

**Verdict:** Archive it as the thesis/evidence compendium. Do not upload it as a
venue paper.

### N01 - Unified signal starvation

The diagnostic contract is coherent, but no PPO or SAO outcome and no controller
test exist. [SAO](https://arxiv.org/abs/2607.07508) is direct recent prior art.

**Verdict:** Keep it as a prospective methods note. An archival paper needs the
frozen PPO/GRPO/SAO matrix, matched cost, and held-out outcomes.

## Ranked publication portfolio

| Rank | Action | Files | Exact gate |
|---:|---|---|---|
| 1 | Preserve the current flagship TMLR route | Separate flagship package, not one of the 18 | Resolve the live NeurIPS overlap; keep the failed 69/100 mechanism gate and all nonclaims frozen. |
| 2 | Publish a bounded audit package | R08, with selected R04 artifact machinery | Add a faithful-recipe comparison and second task for an archival version; a workshop version may stay narrower. |
| 3 | Publish a short stratification note | R02, with selected P02/P03/R05 material | Cut to one question, publish the exact strata, and remove prediction/controller implications. |
| 4 | Release a benchmark artifact | R04 plus R03 documentation | Anonymous clean-machine reproduction, exact release digests, and an honest coverage table. |
| 5 | Build one reporting and registry resource | P05 + P06 + R06 + R07 | External entries, extraction agreement, version governance, and a user decision study. |
| 6 | Run before writing another empirical paper | P01, P04, P07, N01 | Prospective matched experiments, more than one task, fixed cost, held-out outcomes, and declared stopping rules. |
| 7 | Park or archive | P08, R01, U01 | P08 needs a separate credible evaluation; R01 should be retired; U01 remains the internal compendium. |

## Venue and authorship constraints

- TMLR permits assistive language-model use, but the authors remain responsible
  for every claim and for disclosure required by its policy. It also bars
  overlapping parallel archival submissions. See the
  [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html) and
  [author guide](https://jmlr.org/tmlr/author-guide.html).
- NeurIPS 2026's general policy places responsibility on authors and distinguishes
  routine editing or coding aid from important nonstandard method use. The 2026
  Position Paper call applied a stricter substantial-human-writing rule. See the
  [NeurIPS 2026 policy](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines),
  [Position Paper call](https://neurips.cc/Conferences/2026/CallForPositionPapers),
  and the conference's
  [2026 policy report](https://blog.neurips.cc/category/2026-conference/).
- A truthful disclosure must not be deleted merely to make a manuscript look
  eligible. Venue fit should follow the work that was actually done.

## Bottom line

The portfolio has publishable material, but publication requires consolidation:
one current flagship route, two or three bounded companion candidates, one
reporting/registry resource that still needs outside use, several experiment
contracts, and an internal compendium. The next concrete action is to keep the
flagship evidence frozen until the live NeurIPS overlap clears.
