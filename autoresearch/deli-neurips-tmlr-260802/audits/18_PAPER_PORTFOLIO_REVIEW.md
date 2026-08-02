# Review of the current 12-paper portfolio

Date: 2026-08-02

## Decision

I would not submit any of the 12 active manuscripts unchanged.

The review began from an 18-file snapshot. During the audit, the repository
consolidated that history into 12 active roots, P1-P12, and six absorbed
archives. The underlying verdict did not change: the 12 active papers contain a
smaller set of potentially publishable kernels, not 12 venue-ready papers. The
best surviving pieces are:

1. **P11** (former R08), after its replay discrepancy, effective sample size,
   timing, and missing estimand are stated plainly;
2. the matched two-seed group-size panel in **P3**, together with the bounded
   theorem in **P10**;
3. the algebraic ZVF identity and three recomputable tensors in **P2**, after the
   unsupported empirical claims and simulation-to-training leap are removed;
4. **P9/P8**, only after the artifact ledger, coverage claims, and clean-machine
   reproduction are rebuilt; and
5. **P5/P6**, only after the confounded headline exhibits are removed and the
   reporting resource has outside use.

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
  PDFs were freshly rebuilt and pass structural PDF checks; they total 486 pages.
- The six absorbed roots are retained only as history. Their present archive
  PDFs are readable. U01 was repaired to 232 pages but still produces undefined
  citation warnings, which is another reason to keep it out of the venue queue.
- The final active builds have no undefined citations, undefined references, or
  multiply defined labels. P10 nevertheless renders ten placeholder three-box
  diagrams, including duplicate pairs. Those figures are not evidence and must
  be removed or replaced before P10 is circulated as a paper.
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
| Bounded audit | P11 (former R08) | Repair the disclosure first. Keep all four comparisons inconclusive and state the effective n=1, replay discrepancy, timing limit, and absent estimand. |
| ZVF and group-size analysis | P2, P3, P10; absorbed R02 | Repair P2; retain the verified P3 matched panel and P10 theorem; delete P10's invalid empirical and placeholder material. |
| Scaling and benchmarking | P1, P8, P9; absorbed R01 | Rebuild P9's ledger and coverage accounting. Treat P8 as documentation only after its manifest and evidence tiers agree. P1 still needs matched evidence. |
| Length and rollout control | P4, P7, P12 | Park until prospective, matched-cost outcomes exist. |
| Reporting and registry | P5, P6; absorbed R06/R07 | Remove P5's confounded 17x and forced eta-squared exhibits, then build one public resource with external entries and a user study. |
| Fraud detection | absorbed P08_fraud | Separate project; reopen only with a like-for-like credible evaluation. |
| Internal compendium | absorbed U01 | Thesis/evidence reference, not a venue submission. |

## Paper-by-paper verdicts

### P01 / current P1 - Scaling laws

The headline architecture result fails a basic label audit. The exact checkpoint
used for Nemotron-3-Super is a hybrid LatentMoE model, according to its
[NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16),
but P1 assigns it to the dense arm. Reclassifying that anchor changes the
MoE-minus-dense gap from +0.3376 to +0.1875 and the exact one-sided permutation
value from 0.0238 to 0.1780. It also falsifies the draft's statement that no MoE
anchor collapses: Nemotron is both MoE and the panel's collapse case.

The remaining useful result is negative: model size, stack, recipe, budget, and
trace length move together in the available anchors, so the differences cannot
be assigned to scale or architecture. The frontier anchors are single-seed and
confounded, and some three-step probes are weighted like 30-step runs. Recent
direct work also occupies the broad scaling claim, including
[Scaling Behaviors of LLM RL Post-Training](https://arxiv.org/abs/2509.25300),
[Predictive Scaling Laws for Efficient GRPO](https://arxiv.org/abs/2507.18014),
and [Where Should RL Post-Training Compute Go?](https://arxiv.org/abs/2607.13389).

**Verdict:** Do not send the 46-page scaling paper. Delete the architecture
headline. At most, cut it to a workshop note about failed identifiability or
keep it as a thesis chapter.

### P02 / current P2 - Zero-variance fraction

The relation `pass@G - p^G = 1 - ZVF` is an algebraic accounting identity under
the stated iid-Bernoulli model, not an empirical validation result. The draft's
“505 tasks, 1.11e-16” claim has no named raw audit artifact; 505 is the size of a
filtered 600-row proxy cohort elsewhere in the repository. Three raw group
tensors do recompute their reported ZVF values exactly (pooled 0.1583), but that
does not establish prediction or causation. The large variance-mitigation table
is labeled synthetic in one caption and presented as measured elsewhere, the
per-seed ZVF/accuracy mapping swaps seeds 42 and 123, and one AERO sentence cites
the unrelated RL-ZVP paper. The held-out association is weak, and the same ZVF
can describe mastery or incapacity.
Existing systems already react to homogeneous groups, including
[DAPO](https://arxiv.org/abs/2503.14476),
[GRESO](https://arxiv.org/abs/2506.02177), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Hold the current draft. Separate synthetic projections from
measurements, repair the seed mapping and citation, delete the unsupported
505-task validation claim, then keep only the identity, caveats, and R02
stratification result in a much shorter P2 or flagship appendix.

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

The checklist is useful, but the headline 17x exhibit changes backend, algorithm
label, base checkpoint, clipping behavior, and termination behavior. Its low
arm is fully clipped with zero terminated completions. The comparison therefore
shows under-specification rather than a backend effect. A separate $\eta^2=1$
group-size exhibit is forced by one observation per group, despite a caption
that says three seeds per arm; it is not a measured effect size.
Reporting standards and experiment metadata also have long prior art, from
[Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560)
to [SLM Lab](https://arxiv.org/abs/1912.12482),
[OpenRLHF](https://arxiv.org/abs/2405.11143), and
[automatic ML provenance systems](https://www.amazon.science/publications/automatically-tracking-metadata-and-provenance-of-machine-learning-experiments).

**Verdict:** Remove or replace both invalid exhibits, then merge the checklist
with P6 and the archived R06/R07 material. Require external entries, extraction
agreement, and a user decision study before writing a resource paper.

### P06 / current P6 - GRPO registry

The source-provenanced table and explicit unknown states are useful. The current
entries still come mainly from this program, several fields are backlog items,
and there is no extraction-reliability or user-utility result.

**Verdict:** Merge with P5 and the archived R07 material. Release it, recruit outside entries, freeze
the search boundary, and measure extraction errors before a resource paper.

### P07 / current P7 - ZVF controller

The fixed-token comparison is a reasonable plan, but no prospective controller
advantage exists in the repository. The main controller result is an offline
replay of reward tensors, not an executed controller: 1,867 cells called
“escalation events” in the abstract are de-escalations in the body. The open E3
artifact is two-digit addition with one stored arm-level aggregate per arm, has
no cost-matched fixed-G control, and does not implement the paper's PCD guard.
Several tables also omit a model or pool learning rates while claiming only G
varies. Retrospective Pareto language therefore overstates the evidence. Nearby
methods already report adaptive-rollout or pruning results:
[GRESO](https://arxiv.org/abs/2506.02177),
[CPPO](https://arxiv.org/abs/2503.22342),
[DARS](https://arxiv.org/abs/2508.13755), and
[AERO](https://arxiv.org/abs/2602.14338).

**Verdict:** Park it, repair the replay and provenance tables, and do not call
the current artifact a controller evaluation. Promotion requires a prospective
frozen bakeoff measuring quality, charged tokens, wall time, and failure cases
against the closest methods.

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
with R04. Its evidence already lives in the current P2/P9 papers. Its headline
Modal row says GSM8K while the manifest says generated synthetic math, a Tier-C
partial run is promoted despite the stated exclusion rule, and the claimed
seven-library coverage has results for only four. The matched-budget G=2 versus
G=16 two-seed panel is the one central empirical result that fully recomputes.

**Verdict:** Rebuild the headline table from the manifest and keep the verified
group-size panel. Only then use P8 as P9's documentation or a nonarchival
handout.

### R04 / current P9 - NeurIPS DNB benchmark

The treatment fingerprints and cross-stack ambition could form a useful
artifact, but the present evidence ledger does not support the headline scope.
The manifest sums to 75 rather than 79 runs; only four of the claimed seven
libraries have checked-in results; MATH-500 has no manifest row; HumanEval is a
zero-step failed cell; and the 96 GPU-hour reproduction card is roughly
60--180x above the stored run time. Several frontier rows disagree across three
named sources, one cell appears twice with different values, and the advertised
`make reproduce-main`/smoke targets do not exist.

**Verdict:** Hold P9 as a candidate, rebuild every table from one source ledger,
and reconcile run counts, steps, compute, task coverage, and duplicate cells.
Only then use P8 as documentation and run a public clean-machine reproduction.

### R05 / current P10 - ZVF theory

T1 is a binomial interval result. T2 counts rolls to a nonzero-gradient event,
not learning progress. T3's G in {2,3} tie and extremal bound are the clean
result and recompute algebraically. The empirical sections are not reliable:
the purported Dr.GRPO arm removes only standard-deviation normalization rather
than the full Dr.GRPO length term, several numeric bounds do not recompute, and
the “uncapped” length panel begins near its 1,024-token cap. All ten figures are
placeholder three-box TikZ diagrams, including duplicates, rather than the plots
their captions describe. The PDF also discloses drafting assistance.

**Verdict:** Delete the placeholder figures and invalid empirical sections;
retain the correct T3 theorem and bounded T1/T2 statements as an appendix or
short theory note. Keep the disclosure truthful.

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

This has the strongest bounded dataset in the roster. It reconciles 40 arm-seed
units and applies exact paired-t power with the registered four-test BH
correction. All four verdicts are `INCONCLUSIVE`. The audited draft nevertheless
contained a stale DAPO `DISAPPEARS` diagram, described a pilot as two-seed even
though only one arm-level aggregate per arm is stored, and left all registered
published-effect values null. The diagram and pilot disclosure were corrected;
the executed study still cannot estimate a fraction of a published gain. One
model, task, stack, and 30-step horizon cannot support a ranking or equivalence
claim, and one legacy evaluation replay shifted by 0.004, larger than the 0.001
DAPO point difference.

**Verdict:** Keep it as an audit case study after the estimand and replay
limitations are made prominent. Add a faithful-recipe stratum and a second task
for an archival version, and keep every inconclusive label.

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
| 1 | Preserve the current flagship TMLR route | Separate flagship package, not one of the 12 active papers | Resolve the live NeurIPS overlap; keep the failed 69/100 mechanism gate and all nonclaims frozen. |
| 2 | Repair, then scope a bounded audit | P11, with selected P9 artifact machinery | Disclose the n=1 arm aggregates, replay shift, date-only amendment timing, and absent published-gain estimand; add a faithful-recipe comparison and second task for an archival version. |
| 3 | Build a short verified diagnostic note | Repaired P2, verified P3 panel, and P10 theorem | Ask one question, publish exact strata, keep only recomputable results, and remove controller implications, invalid empirics, and placeholder figures. |
| 4 | Rebuild, then release a benchmark artifact | P9 plus repaired P8 documentation | Reconcile the ledger, coverage, task failures, and compute card before anonymous clean-machine reproduction and exact release digests. |
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
unchanged submission-ready paper. The separate flagship can take the TMLR route
only after its NeurIPS overlap clears. Inside this portfolio, repair P11 first;
then build a short note from the verified P2/P3/P10 kernel; then rebuild the
P9/P8 artifact ledger. P5/P6 need invalid exhibits removed and outside use.
P1/P4/P7/P12 need prospective experiments.
