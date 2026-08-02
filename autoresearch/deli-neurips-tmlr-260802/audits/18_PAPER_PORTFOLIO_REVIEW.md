# Review of the 18-paper portfolio

Date: 2026-08-02

## Decision

I would not submit any of the 18 reviewed manuscripts unchanged.

The review began from an 18-file snapshot. During the audit, the repository
consolidated that history into 12 active roots, P1-P12, and six absorbed
archives. The underlying verdict did not change: the material is about six
contribution families, not 18 independent papers. The best current pieces are:

1. **P11** (former R08), as a bounded reproducibility and statistical audit;
2. **P2**, including the stratified result absorbed from R02, after a severe cut;
3. **P9** (former R04), as an artifact package once every release claim is
   checked; and
4. **P5/P6**, with former R06/R07 absorbed, as one reporting/registry resource.

The best archival submission in the repository is still the separate flagship
paper, *Same Terminal Signal, Different Action*. That manuscript is outside this
18-file roster. Its TMLR route remains blocked while the overlapping NeurIPS
submission is active.

## What I checked

- The frozen review snapshot used the former IDs P01-P08, R01-R08, U01, and
  N01. Every source file and recursively included source was read: 414
  include-closure mentions, 329 distinct files, 57,339 lines, and 3,056,844
  bytes. Its PDFs totalled 868 pages.
- The post-review queue is the machine-checked P1-P12 roster in
  `platform_hybrid/paper/scripts/publication_worthiness_check.py`. All 12 active
  PDFs were freshly rebuilt and pass structural PDF checks; they total 488 pages.
- The six absorbed roots are retained only as history. Their present archive
  PDFs are readable. U01 was repaired to 232 pages but still produces undefined
  citation warnings, which is another reason to keep it out of the venue queue.
- The source audit found no active rendered figure fallbacks, no duplicate
  labels, and no TODO markers. Sixteen source files contain the word
  `placeholder`, but these are dormant fallback branches or declared registry
  backlog entries, not missing figures in the current PDFs.
- The seven-gram overlap scan finds substantial reuse among R02/R03/R04
  (Jaccard 0.427, 0.308, and 0.306) and between U01 and P01/P04 (0.245 and
  0.247). This confirms that those files cannot be presented as independent
  contributions without an explicit overlap account.

The current reconciliation manifest and the frozen review-snapshot inventory,
expanded source, extracted text, and similarity matrix are in
`audits/paper_portfolio/`.

## Portfolio map

| Bucket | Manuscripts | Publication treatment |
|---|---|---|
| Bounded audit | P11 (former R08) | Keep as a case study. Do not turn inconclusive comparisons into a ranking. |
| ZVF and group-size analysis | P2, P3, P10; absorbed R02 | Cut P2 to one question; merge useful theory and group-size material. |
| Scaling and benchmarking | P1, P8, P9; absorbed R01 | P9 may become an artifact. P1 needs matched evidence; P8 is documentation. |
| Length and rollout control | P4, P7, P12 | Park until prospective, matched-cost outcomes exist. |
| Reporting and registry | P5, P6; absorbed R06/R07 | Build one public resource with external entries and a user study. |
| Fraud detection | absorbed P08_fraud | Separate project; reopen only with a like-for-like credible evaluation. |
| Internal compendium | absorbed U01 | Thesis/evidence reference, not a venue submission. |

## Paper-by-paper verdicts

### P01 / current P1 - Scaling laws

The useful result is negative: model size, stack, recipe, and budget move
together in the available anchors, so the differences cannot be assigned to
scale. The frontier anchors are single-seed and confounded. Recent direct work
also occupies the broad scaling claim, including
[Scaling Behaviors of LLM RL Post-Training](https://arxiv.org/abs/2509.25300),
[Predictive Scaling Laws for Efficient GRPO](https://arxiv.org/abs/2507.18014),
and [Where Should RL Post-Training Compute Go?](https://arxiv.org/abs/2607.13389).

**Verdict:** Do not send the 45-page scaling paper. Cut it to a workshop note
about failed identifiability or keep it as a thesis chapter.

### P02 / current P2 - Zero-variance fraction

The exact relation `pass@G - p^G = 1 - ZVF` is a useful accounting identity.
ZVF remains descriptive under the stated sampling model; it is not a causal
mechanism or a calibrated predictor of training failure. The held-out
association is weak, and the same value can describe mastery or incapacity.
Existing systems already react to homogeneous groups, including
[DAPO](https://arxiv.org/abs/2503.14476),
[GRESO](https://arxiv.org/abs/2506.02177), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Keep the identity, caveats, and R02 stratification result together
in a much shorter P2 or the flagship appendix.

### P03 / current P3 - Group size

Direct measurements stop at G=16. G=32 is reconstructed, while G=4 appears only
under a declared `(1-ZVF)/sqrt(G)` proxy and cohort. The DPO bridge is an
interpretation rather than a measured gradient equality. Small- versus
large-group behavior also has direct recent treatment in
[It Takes Two: GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977).

**Verdict:** Merge the observed-versus-proxy table into P2. A separate paper
needs a prospective, fixed-cost sweep over multiple tasks.

### P04 / current P4 - Length bias

Under the 200-token cap and short horizon, the small GRPO/Dr.GRPO comparison did
not show a clear length effect. The arms have five and three seeds, and the cap
suppresses the regime studied by
[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783).

**Verdict:** Park it. A negative-results note needs an uncapped, matched-seed
replication and a claim bounded to that setting.

### P05 / current P5 - MIN-REPORT-RL

The checklist is useful, but the headline 17x exhibit changes both backend and
base checkpoint. It shows under-specification rather than backend causality.
Reporting standards and experiment metadata also have long prior art, from
[Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560)
to [SLM Lab](https://arxiv.org/abs/1912.12482),
[OpenRLHF](https://arxiv.org/abs/2405.11143), and
[automatic ML provenance systems](https://www.amazon.science/publications/automatically-tracking-metadata-and-provenance-of-machine-learning-experiments).

**Verdict:** Merge it with P6 and the archived R06/R07 material. Require external entries, extraction
agreement, and a user decision study before writing a resource paper.

### P06 / current P6 - GRPO registry

The source-provenanced table and explicit unknown states are useful. The current
entries still come mainly from this program, several fields are backlog items,
and there is no extraction-reliability or user-utility result.

**Verdict:** Merge with P5 and the archived R07 material. Release it, recruit outside entries, freeze
the search boundary, and measure extraction errors before a resource paper.

### P07 / current P7 - ZVF controller

The fixed-token comparison is a reasonable plan, but no prospective controller
advantage exists in the repository. Retrospective Pareto language overstates the
evidence. Nearby methods already report adaptive-rollout or pruning results:
[GRESO](https://arxiv.org/abs/2506.02177),
[CPPO](https://arxiv.org/abs/2503.22342),
[DARS](https://arxiv.org/abs/2508.13755), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Park it until the frozen bakeoff measures quality, charged tokens,
wall time, and failure cases against the closest methods.

### P08_fraud / absorbed - Fraud detection

The 50,000-row exercise is synthetic. XGBoost and LLM AUC come from different
held-out splits, so the 85x cost comparison and hybrid recommendation are not
like-for-like tests. Tabular LLM comparisons already include
[TabLLM](https://arxiv.org/abs/2210.10723).

**Verdict:** Keep it outside the RL portfolio. Reopen it with one frozen split,
a credible fraud benchmark, repeated uncertainty, and a tested hybrid router.

### R01 / absorbed into P9 - ACM benchmark variant

The comparison mixes LLM-native stacks with classic RL systems using small MLPs
and discrete environments. Its gaps combine task encoding, model class,
implementation, and hardware.

**Verdict:** Retire this derivative. Keep its comparison-boundary warning in
P9.

### R02 / absorbed into P2 - NeurIPS ZVF variant

This is the cleanest ZVF result: a pooled association collapses or reverses
within GSM8K strata. It remains retrospective and cannot show prediction or a
training benefit.

**Verdict:** Cut it to an 8-10 page workshop note, publish the exact strata and
code, and remove controller language.

### R03 / current P8 - NeurIPS workshop artifact

The walkthrough is readable, but seven-gram overlap is 0.427 with R02 and 0.306
with R04. Its evidence already lives in the current P2/P9 papers.

**Verdict:** Use P8 as P9's documentation or as a nonarchival
handout.

### R04 / current P9 - NeurIPS DNB benchmark

The treatment fingerprints, 79-run scope, seven-library map, and reproduction
commands could form a useful artifact. Several rows and release statements are
still incomplete. Pinned containers, anonymous code, and `make reproduce-main`
need a clean-machine public test.

**Verdict:** Make P9 the artifact paper, use P8 as its documentation, and publish an
exact runnable/documented/missing coverage table.

### R05 / current P10 - ZVF theory

T1 is a binomial interval result. T2 counts rolls to a nonzero-gradient event,
not learning progress. T3 is a proxy tie at G in {2,3}. The word `universal`
exceeds the stated objective and sampling assumptions. The PDF also discloses
drafting assistance.

**Verdict:** Move the correct statements to an appendix and keep the disclosure
truthful. The current contribution belongs inside a larger paper.

### R06 / absorbed into P5 - MIN-REPORT position

This readable P05 condensation repeats the confounded 17x exhibit and adds no
external adoption. The NeurIPS 2026 Position Paper track required substantial
human writing and a declaration.

**Verdict:** Fold its clearest prose into the reporting/registry resource. Keep
all truthful assistance disclosures.

### R07 / absorbed into P6 - Living GRPO registry

This is the clearest resource statement for P06, but it has no independent
evidence. Its compile log also has 167 underfull boxes.

**Verdict:** Use it as P6's README and fix the tables after external-entry and
content gates pass.

### R08 / current P11 - Reproducibility audit

This is the strongest bounded paper in the roster. It reconciles 40 arm-seed
units and applies exact paired-t power with the registered four-test BH
correction. All four verdicts are `INCONCLUSIVE`; the old DAPO label remains
only in the correction history. One model, task, stack, and 30-step horizon
cannot support a ranking or equivalence claim.

**Verdict:** Keep it as an audit case study. Add a faithful-recipe stratum and a
second task for an archival version, and keep every inconclusive label.

### U01 / absorbed into thesis and P9 - Umbrella benchmark

The original 239-page snapshot preserved the program but reused large blocks
from P01 and P04 and mixed several questions and contribution types. The repaired
archive is now 232 pages and still has unresolved citation warnings.

**Verdict:** Archive it as the thesis/evidence compendium. Do not upload it as a
venue paper.

### N01 / current P12 - Unified signal starvation

The diagnostic contract is coherent, but no PPO or SAO outcome and no controller
test exist. [SAO](https://arxiv.org/abs/2607.07508) is direct recent prior art.

**Verdict:** Keep it as a prospective methods note. An archival paper needs the
frozen PPO/GRPO/SAO matrix, matched cost, and held-out outcomes.

## Ranked publication portfolio

| Rank | Action | Files | Exact gate |
|---:|---|---|---|
| 1 | Preserve the current flagship TMLR route | Separate flagship package, not one of the 18 | Resolve the live NeurIPS overlap; keep the failed 69/100 mechanism gate and all nonclaims frozen. |
| 2 | Publish a bounded audit package | P11, with selected P9 artifact machinery | Add a faithful-recipe comparison and second task for an archival version; a workshop version may stay narrower. |
| 3 | Publish a short stratification note | P2, with selected P3/P10 material | Cut to one question, publish the exact strata, and remove prediction/controller implications. |
| 4 | Release a benchmark artifact | P9 plus P8 documentation | Anonymous clean-machine reproduction, exact release digests, and an honest coverage table. |
| 5 | Build one reporting and registry resource | P5 + P6 | External entries, extraction agreement, version governance, and a user decision study. |
| 6 | Run before writing another empirical paper | P1, P4, P7, P12 | Prospective matched experiments, more than one task, fixed cost, held-out outcomes, and declared stopping rules. |
| 7 | Keep as history | P08_fraud, R01, R02, R06, R07, U01 | These are absorbed roots, not extra submission vehicles. |

## Venue and authorship constraints

- TMLR permits assistive language-model use, but the authors remain responsible
  for every claim and for disclosure required by its policy. It also bars
  overlapping parallel archival submissions. See the
  [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html) and
  [author guide](https://jmlr.org/tmlr/author-guide.html).
- NeurIPS 2026's general policy places responsibility on authors and distinguishes
  routine editing or coding aid from important nonstandard method use. The 2026
  Position Paper call applied a stricter substantial-human-writing rule. See the
  [NeurIPS 2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook),
  [Position Paper call](https://neurips.cc/Conferences/2026/CallForPositionPapers),
  and the conference's
  [2026 policy report](https://blog.neurips.cc/category/2026-conference/).
- A truthful disclosure must not be deleted merely to make a manuscript look
  eligible. Venue fit should follow the work that was actually done.

## Bottom line

The consolidation is now explicit: 12 live roots, six absorbed archives, and no
unchanged submission-ready paper. The practical queue is the flagship after its
NeurIPS overlap clears, then P11, a sharply cut P2, and the P9 artifact. P5/P6
need outside use; P1/P4/P7/P12 need prospective experiments.
