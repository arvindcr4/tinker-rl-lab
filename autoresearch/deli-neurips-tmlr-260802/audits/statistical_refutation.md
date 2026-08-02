# Statistical and experimental refutation audit

Date: 2026-08-02  
Scope: May GSM8K held-out comparison, E1 five-arm audit, retrospective ZVF triage,
S1 conformance, r4-2 flagship pilot, and spectral/entropy flagship headlines.

## Bottom line

No checked-in result currently establishes improved policy learning, held-out
capability, cross-stack objective equivalence, or a validated early-stop policy.
The strongest surviving results are narrower:

- the May and E1 raw held-out scores are reproducible descriptive measurements;
- every E1 arm-versus-GRPO effect is **INCONCLUSIVE**;
- ZVF's 2/22 result is descriptive retrospective concordance, not prediction;
- S1 **SURVIVES** only as deterministic conformance of the *intended adapters* on
  injected canonical fixtures; native framework conformance does not survive; and
- r4-2 receipt integrity is now verifiable, but its sole completed
  intended-full/balanced mechanism cell fails the frozen 95/100 gate (69/100),
  while the causal and confirmatory designs remain incomplete.

`SURVIVES` below means the exact bounded statement is supported. `INCONCLUSIVE`
means the numbers exist but do not decide the scientific estimand. `INVALID`
means the stated inference contradicts the design, frozen rule, or available
provenance.

## Methods and independence checks

I recomputed binary outcomes from raw item rows, aligned records by item and seed,
used exact small-sample t distributions/noncentral-t power, exact McNemar tests,
Benjamini--Hochberg (BH) over the four E1 contrasts, and a post-hoc crossed
seed-by-item bootstrap (50,000 resamples, RNG seed 20260802). The crossed bootstrap
is a generalization sensitivity analysis, not a replacement for the locked E1
seed-bootstrap estimand. I also recomputed exact Clopper--Pearson intervals for the
ZVF confusion matrix and re-evaluated every r4-2 receipt with the frozen predicate.

Validation executed in this checkout:

- 9/9 `audit.test_aggregate_audit` tests pass after the statistical repair;
- 24/24 common S1 reference/receipt tests pass;
- 16/16 spectral prototype tests pass;
- `verify_claims.py --repo-root .` passes; and
- a fresh extraction of `review_bundle.zip` independently reports S1 `PASS`, r4-2
  `PASS`, 17 frozen source files, and the failed 69/95 mechanism cell.

## 1. May held-out GSM8K: 82.0% versus 83.3%

**Sources.** `platform_tinker/reports/final/gsm8k_base_control_200.json` and
`gsm8k_heldout_seed{042,137,256,512,999}.json`. SHA-256 values, in that order:
`3e74b05bb3f4098950b43d1ae27729749f5bc7c458ed763507df643addf0e2a8`,
`41f6923687dc5f4588aaa8833f586f8191807d006162cabba0112aef91811eab`,
`b4c65e5b05f38618c63a07dda0e114582205344787cbe00e411fab27e78ef279`,
`9a714eb02bdc2793e610768ef9aa17c0dce919f49c76331d36f2cacc5a449a6c`,
`ab7172f509acc49a47cddf4f810e654e4a15e357ecb3fee022be8a9bcc0b3592`,
`a101bc294035439e524eb5bf79a0dcbfd46bb0a59c554413941cc42d6256d7fa`.

**Raw recomputation.** The six files contain the same 200 indices, truncated
question strings, and targets. Correct counts are base 164/200 and trained
`[166,165,161,168,173]/200`; trained-seed accuracies are
`[.830,.825,.805,.840,.865]`, mean `.833`, sample SD `.02197`. Seed-level
differences from the fixed base are `[+.010,+.005,-.015,+.020,+.045]`.

- One-sample seed t test: `t(4)=1.3234`, two-sided `p=.2563`, 95% t CI for the
  mean delta `[-.01427,+.04027]` (-1.43 to +4.03 percentage points).
- Exact 80%-power two-sided MDE at alpha .05: `.03695` (3.70 points).
- Per-seed exact McNemar p-values: `.851, 1.000, .736, .572, .093`; none rejects.
- Crossed seed-by-item bootstrap 95% CI: `[-.034,+.060]` (-3.4 to +6.0 points).

**Verdict.** The descriptive `82.0% -> 83.3%` and `p=.26` **SURVIVE**. Any
statement that GRPO improved capability, had no effect, was equivalent to base, or
yielded a “negative/null result” is **INCONCLUSIVE**. The data have power only for
large effects: failure to reject is not evidence of absence.

**Design failures.** The evaluator selects `range(limit)`, so these are the first
200 GSM8K test rows, not a random 50-row sample. Dataset revision, base-model
revision, evaluator commit, full completions, and stable dataset item IDs are not
recorded. Five training seeds share one deterministic base evaluation; there is no
independent confirmatory cohort. `p0_gsm8k_paired.md`'s per-prompt mean-indicator
`p=.539` treats prompts, after averaging the same five seeds, as the inferential
units and does not capture training-seed uncertainty. Its post-hoc five-model
majority-vote `p=.0347` concerns an ensemble on this item slice, not a single-model
training effect, and was neither preregistered nor multiplicity-controlled.
Manuscript text calling `.539` a “random 50-problem subset” is factually invalid.

**Allowed wording.** “On the first 200 GSM8K test rows, five post-training
checkpoints averaged 83.3% versus one greedy base evaluation at 82.0%; the observed
+1.3-point seed-level difference was imprecise (`t(4)=1.32`, `p=.256`, 95% CI
-1.43 to +4.03 points) and does not establish improvement or equivalence.”

**Minimum decisive experiment.** Freeze model/dataset/evaluator hashes and a
prospective seed-by-item estimand; evaluate pre/post checkpoints on the full held-out
set with raw completions. At the observed seed SD, exact t power requires 25
independent training seeds for the observed 1.3-point effect (40 for a 1-point
effect), followed by a crossed/hierarchical seed-by-item analysis on a separate
confirmatory cohort.

## 2. E1 same-stack audit: five arms x eight seeds

**Sources.** Locked analysis: `zvf-program/audit/preregistration.json` SHA-256
`1df4e39ec93d7c6f660ded574b477395b6bfe44f8cbe257434e3761b92e96656`.
Raw set: 40 files under
`zvf-program/audit/results/full/manifests/`; the SHA-256 of the sorted per-file hash
ledger is `c30643a7994f70c585d313e2a550ea9798a2fe17d76fc47492a204e9f56ceb4b`.
Campaign verification SHA-256 is
`f83fcdcf8575a8f52461d72f8926d1eb3fba6f1908465064e27fdfe3e0e3749a`
and records 40/40 locally valid and 40/40 remotely verified.

All traces have 500 unique common indices and targets. Recomputed `correct` equals
`prediction == target` and reproduces `audit_record.heldout_score`. The contract
pins GSM8K revision `740312add88f781978c0658806c59bc2815b9866`, Qwen3-8B
revision `b968826d9c46dd6066d109eabc6255188de91218`, 30 steps, and stack
fingerprint `4737be74fdc97dd400b846e93c6b0c03f443ee65da2cc89697bf5b078d16aa0d`.

| Arm vs GRPO | Delta | Locked 95% seed-bootstrap CI | paired-t p | BH q | exact MDE80 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| DAPO | +.00100 | [-.00450,+.00675] | .756 | .858 | .010116 | INCONCLUSIVE |
| GSPO | +.00500 | [-.00125,+.01200] | .210 | .841 | .011854 | INCONCLUSIVE |
| Dr.GRPO | -.00200 | [-.00950,+.00725] | .673 | .858 | .014830 | INCONCLUSIVE |
| AERO | -.00075 | [-.00825,+.00675] | .858 | .858 | .013192 | INCONCLUSIVE |

**Refuted implementation result.** The pre-audit aggregator used
`(z_.975+z_.80)s/sqrt(8)` and reported DAPO MDE `.008667`, allowing
`DISAPPEARS`. Exact noncentral paired-t power gives `.010115893 > .010`, so the
locked power condition fails and DAPO is **INCONCLUSIVE**. The preregistered BH
function also had no call site. The current working tree now executes exact power
and BH, gates directional verdicts on the BH decision, emits the method, and passes
9/9 tests. This repairs the analysis; historical `DISAPPEARS` text is **INVALID**.

As a population-generalization sensitivity check, crossed seed-by-item 95% CIs are
AERO `[-.01625,.01450]`, DAPO `[-.01350,.01575]`, Dr.GRPO
`[-.01775,.01400]`, and GSPO `[-.01050,.02050]`. DAPO's crossed 90% CI
`[-.01100,.01325]` also exceeds the +/-1-point equivalence region. The locked
seed-bootstrap result may describe these 500 fixed items; it cannot establish
GSM8K-population equivalence.

**Verdict and allowed wording.** The complete same-stack execution and observed
score differences **SURVIVE** as descriptive evidence. All scientific comparisons
are **INCONCLUSIVE**: “Across eight paired seeds, no arm produced a BH-significant
held-out difference from GRPO, and none met the locked exact-power equivalence
rule.” Do not say the methods are the same, disappear, improve learning, or fail to
improve learning.

**Minimum decisive experiment.** Choose one primary contrast and one estimand.
For DAPO's +/-1-point fixed-item gate, the plug-in minimum is nine paired seeds, but
the observed variance is too uncertain: using its one-sided 95% upper SD bound
requires about 22 paired seeds. Preregister that prospective cohort, preserve item
rows/completions, use exact noncentral-t power plus TOST or an explicitly powered
equivalence model, and use crossed inference if the claim generalizes beyond the
fixed 500 items. Power and multiplicity must be executable before outcomes exist.

## 3. Early ZVF triage: 2 collapsed runs among 22

**Sources.** `platform_hybrid/experiments/zvf_predictive_validation.py` SHA-256
`c52f8188178007ea9c9efdf07ed71013dd53640d935354319b12e1dec7d553c5`;
output JSON SHA-256
`134cca1e2bc4ef574c19a5b304f4c60737348bc8dd99232f8117a2a0599ab8c0`.
The two positive raw files are
`platform_hybrid/experiments/tinker-runs/results/cross_tool_{llama-8b-inst,qwen3-32b}.json`,
hashes `d4babb2f92318c0bfa6b3b4c6ecffaaf4e234864a4c0a737f83b6f23cf622ea3`
and `55f0b6b237cab0ed0f80c280ba1ad282f1664668ea3499f890574d04acb8f4f8`.

The script is explicitly retrospective. It scans changing glob sets, finds 52 raw
records, de-duplicates to 22, pools 8 GSM8K, 2 tool-use, and 12 `unknown` runs, and
uses only seed labels 42 and 20260422. Both collapsed cases are tool-use runs with
early reward 0, early ZVF 1, and late reward 0. The joint rule flags 2/2 with 0/20
false positives, but `early_reward_mean <= .05` alone produces the identical 22
predictions. Early ZVF versus late reward has Spearman `-.158`, bootstrap CI
`[-.634,.430]`; adding ZVF changes leave-one-run-out R2 only `.8848 -> .8882`.

The apparent recall/precision 2/2 has exact 95% interval `[.158,1]`; specificity
20/20 has `[.832,1]` (FPR 0/20 upper bound `.168`). The reported AUC bootstrap
`[1,1]` is degenerate because only two positives are resampled and both are
perfectly separated; it does not express event-sampling uncertainty.

**Verdict.** “The frozen-looking rule retrospectively agrees with the two observed
tool-use collapses” **SURVIVES**. A claim that ZVF predicts collapse, adds value
over early reward, transfers across tasks, or justifies an early-stop policy is
**INCONCLUSIVE** (and **INVALID** if presented as prospective validation).

**Minimum decisive experiment.** Freeze the rule before a held-out run cohort;
hold model, task, backend, runner, evaluator, horizon, and seed structure fixed;
compare against reward-only in nested out-of-sample evaluation. Even with perfect
classification, at least 17 independent collapse events and 36 non-collapse runs
are needed for two-sided 95% lower bounds of .80 sensitivity and .90 specificity.
Event enrichment is acceptable only if prevalence-dependent metrics are corrected.

## 4. S1 objective conformance

**Sources.** `zvf-program/flagship/s1/results/implementation_freeze.json` and its
TRL/verl receipts. Freeze SHA-256 is
`a785ceadc143ff7449070a64f0cecd8bd858b62c561bb19260e51e9f40a69ace`;
receipt hashes are `6b9b62ae68a44303d6900f58e9764966380368cb73d82b335397124bbf0b74dc`
and `3221155ea599bb29d3ae8cd98ede82fc589d9f23dcd62537e1b85b4c9ca9427f`;
fixture digest is `c35916cf7db0b6c7ff6d0e35925a165b304fc78ff3d63845b9a853ca8af8ae9b`.

The combined receipt reports `S1_PASS`, 14 intended cases per stack, 36 controller
cases, float64 (`rtol=1e-6`, `atol=1e-8`), TRL 1.2.0 and verl 0.3.0.post1 on CPU.
This **SURVIVES** as deterministic fixture-level integration conformance. It is not
end-to-end training: no model is loaded, no completions are generated, and no
optimizer runs. Formula notes state that canonical advantages are injected before
the pinned loss kernels.

Native conformance is **INVALID**: TRL has four `MATERIAL_DIFFERENCE` and one
`NOT_TESTED`; verl has one `MATERIAL_DIFFERENCE` and four `NOT_TESTED`. The 36
controller cases are deterministic policy fixtures, not evidence that a controller
improves reward or capability. S1 does not authorize GPU screening or learning
claims.

**Minimum broader experiment.** Exercise native and intended advantage, loss,
gradient, optimizer, masking, and distributed-reduction paths on hashed real model
batches on CPU and GPU across pinned versions; then run separately preregistered,
matched, multi-seed learning comparisons. Keep conformance and learning as distinct
endpoints.

## 5. r4-2 flagship pilot

**Sources.** The decisive accepted record is
`zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/acceptance/fpilot__intended_full__balanced_equal_length__s23.json`,
SHA-256 `4164c8b356c12a38f6d04ea49576ca560d7b04522a0875ef80b78db39e68bf2e`.
The predicate in `zvf-program/flagship/pilot/analysis.py:122-131,154-175`
(SHA-256 `0d40f218e4e159ac1f4875ffcb67eba658c5a9e2ce73231a079640584255d17f`)
counts joint-zero as equivalent and requires each
nonzero receipt to have cosine `>=.999` **and** relative-L2 `<=.01`; balanced
intended-full requires 95/100.

Recomputation of `.full_record.manifest.gradient_receipts` gives 100 receipts: 65
joint-zero, 35 nonzero, and only four nonzero receipts meet both bounds, hence
**69/100 < 95/100**. Therefore “high agreement” or
intended/native equivalence under the registered gate is **INVALID**. The high
minimum cosine (`.999881` in this cell) is descriptive near-collinearity; 31/35
nonzero steps fail the relative-L2 half of the joint criterion.

The six accepted scientific units and their endpoint/token/receipt counts
**SURVIVE** as receipt-backed descriptions. The source-provenance blocker discovered
during this audit is repaired: exact executed `objective.py` bytes were recovered
as `pilot/provenance/r4-2-objective.py`, SHA-256
`980a56a1651299a5adbe7a0927c13b12d42d9d7e1a36205500a24d5eeba9b61b`,
hard-bound into the review bundle, and both live and fresh
clean-extraction verifiers pass all 17 frozen source bindings. The verifier still
states that it checks receipt integrity/internal invariants and does not regenerate
gradients or predictions from private checkpoints.

The campaign-level causal result remains **INCONCLUSIVE**: only seed 23 has the
four-condition balanced block; seed 11 has two scientific conditions; seed 37 has
none; the filtered positive-control regime is absent; and no confirmatory seeds
ran. Missing cells cannot reverse the accepted 69/100 cell failure. Endpoint
accuracy differences among six incomplete cells are not causal effects.

**Allowed wording.** “Receipt integrity and descriptive near-collinearity were
verified, but the sole completed intended-full/balanced cell failed the registered
mechanism gate (69/100 versus 95/100); the incomplete matrix answers neither the
causal nor confirmatory question.”

**Minimum decisive experiment.** Preserve the repaired frozen source. Because the
registered cell already fails, complete data cannot convert this campaign into GO.
Any altered threshold or positive-control construction requires a new preregistered
campaign. Run all 24 screening cells (2 regimes x 3 seeds x 4 conditions) with a
feasible positive control, then launch untouched confirmatory seeds only after a
frozen GO. Regenerate gradients/evaluations from accessible checkpoints for an
independent computational audit.

## 6. Spectral/entropy flagship headlines

`spectral_entropy_paper.tex` is marked “SUPERSEDED--DO NOT SUBMIT,” yet asserts ZVF
`64.2% -> 1.1%`, GSM8K `+7.9` points, MATH `+7.8` points, and improved sample
efficiency without traceable raw experiments. Every such empirical claim is
**INVALID** and must remain quarantined.

The evidence-safe `spectral_entropy_paper_kdense_revision.tex` correctly makes no
learning claim. Its 16 focused synthetic/prototype tests independently pass, so
that narrow code-check statement **SURVIVES**. Its referenced 77-case independent
audit files (`results/audit_findings_summary.json`, CSV/JSON raw sweeps) are absent
from this checkout; the printed bundle hash
`dbd1386a29d4f413f7a5da56ba6191891915117efa5d13ff6e7eb9ad2c47e5e9` is not the
bundle, so the 77-case quantitative headline is **INCONCLUSIVE** here. Norm
preservation, nonzero synthetic
gradients, and tests are not mitigation, sample-efficiency, or policy-learning
evidence.

**Minimum learning experiment.** Supply and verify the 77-case bundle first. Then
preregister matched equal-compute multi-seed language-model runs with reward-only,
random-score/placebo, larger-group/resampling, and ablation controls; evaluate
held-out capability, online reward, KL, token/FLOP cost, and failures separately.

## Submission decision from this track

**No-go for a NeurIPS/TMLR empirical improvement or equivalence paper on current
evidence.** A narrower reproducibility/negative-results manuscript can be honest if
its spine is: executable claim-to-run auditing exposed (i) an E1 small-sample power
verdict error, (ii) no decisive E1 arm effect, and (iii) an r4-2 registered gate
failure hidden by cosine-only prose. It must present May and ZVF as underpowered
descriptive cases, S1 as fixture conformance, r4-2 as incomplete receipt evidence,
and explicitly exclude learning, capability, equivalence, and production-triage
claims.
