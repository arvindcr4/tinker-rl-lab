# Review 19 resolution

Protocol: corrected-review19-v2. Historical result artifacts are unchanged and were not rerun. Changed output-directory declarations use a new versioned subtree. These repairs do not validate historical numerical claims or establish training effects.

## Fixed
- R1: synthetic controller reward adjustment removed; preservation verdict and ratio are null, status NOT_EVALUATED
- R2: cost-efficiency restoration now means newly restored contrast after appending target-minus-pilot samples; already mixed groups contribute zero
- R3 (identity): baseline and intervention use the same IID estimator in controller gains; unsupported anti-herding verdict unavailable
- R4: iter111/131 ordinary with-replacement bootstrap; iter159 standard RNG replaces pathological low-bit LCG
- R5 (levels): iter151/163 use actual replicate counts at 2.5/97.5 percentiles
- R6: inversion direction and admissible homogeneous-model domain
- R7: C4 applies the same guarded trigger/bump as C3 to its caller-supplied smoothed statistic
- R8 (endpoints): iter111/131 preserve exact probability endpoints
- R17 (subset): iter171 sample count is observations, bootstrap count separate; iter199 method identity fixed
- R18: named repository-root off-by-one paths corrected
- R19: copied literal recomputations are null with UNVERIFIED_SOURCE

## Partly addressed / remaining
- R3: signature retains observed versus predicted diagnostics, but cannot establish dependence; independent evaluation unavailable
- R5: iter139 difference-of-means and hybrid pooled interval scale still require migration
- R8–R16: target feasibility/status, observed trigger substitution, Pareto dominance, aggregate retention, cross-iteration policy version alignment, degenerate concordance proxies, entropy sign, objective-aligned regret, and feedback simulation remain unverified except fixes explicitly listed above
- R16: iter199 is relabeled open-loop fixed-initial-probability; its hybrid selection and complete evidence export still need repair
- R17: remaining TSV alignment and missing CI artifacts are not rebuilt
- R20–R21: full evidence-panel validation, identifier pairing, stable seed migration and complete provenance manifest remain open
- Design concerns A–E remain limitations: outcome-informed retention is not realizable generation savings; aggregate inversion is a homogeneous scenario; conditional bootstrap is not independent training-seed uncertainty; rollout proxies are not dollar costs

## Validation
CPU-only function regressions cover append probabilities (8/17), already-mixed groups, small/four-method bootstrap variation, paired identity, inverse roundtrips/domain rejection, identity-controller zero gain, and actual sample count. No training, paid jobs, provider calls, or stored-result edits.

## Follow-up repairs (supersede matching open items above)
- R5: iter139 now bootstraps conditional FIRE/NO-FIRE group means, preserving the [1,1] constant-group fixture; hybrid pooled mean intervals divide by actual pooled sample count
- R9–R11: observed ZVF dispatch restored, Pareto dominance directions/ties corrected, iter159 retention and paired retention differences use ratios of aggregate contrast in each replicate (zero baseline is unavailable)
- R12/R14: iter187 high trigger is >= TAU, posterior entropy subtracts information gain
- R16/R17: iter199 hybrid evaluates supplied candidates and requires an explicit prompt probability model; missing trigger values fail rather than default to zero; all CI rows precede CI serialization; iter147 TSV follows header order
- R21: stable digest seeds in granularity replay and iter163; remaining scripts still need migration
- Exact finite-pool fallback computes actual full-pool contrast instead of returning one
Additional CPU regressions cover ratio estimands, conditional constant-group gaps, dominance/tradeoff/tie, observed-trigger reproduction and candidate-grid/unsupported-model behavior.

## Closure pass: corrected-review19-v3

These changes supersede the specific open code paths below; no historical numerical results or paper conclusions are validated. All seven changed analyses write to `corrected-review19-v3`, leaving stored evidence untouched.

- R8 (iter115): one target-attainment predicate is shared across decisions and replay, including G64. Exact probability endpoints remain exact. Explicit no-feasible-target status and controller-independent salvage are exported with replay and seed summaries.
- R13 (iter143): always-on/off identical vectors have undefined kappa; undefined pairs are counted and excluded from threshold selection. Raw agreement and both firing prevalences accompany every pair in JSON. All-undefined panels have no selected threshold or passing hypothesis. SD replaces the invalid signed-statistic CV criterion; overlapping pair SE remains descriptive, not independent-seed inference.
- R13 (iter195): the impossible AP inequality remains only as a clearly named always-off synthetic diagnostic, and DF as a mixed-group indicator. No replacement controller is invented. Cross-system verdicts are null/NOT_EVALUATED, with rule identities in the output. Useful controller concordance remains unestablished.
- R15 (iter167/175): a versioned objective change makes the oracle maximize raw incremental yield on the shared bounded action menu, matching raw-yield capture and regret consumers, including bootstrap. Cost-efficiency comparisons remain descriptive and are not an efficiency-optimal bound. This is a model replay, not independent training validation or a fixed-dollar-budget optimum.
- R17 (iter175): the fourth decision slot always records the actual Dualformer target, including C6 fallback and no-fire branches.
- R20 (iter143 only): exact panel length, unique aligned step IDs and finite unit-interval ZVF are required. Rows sort by explicit step ID; padding/truncation is removed. Other R20 loaders are not certified by this closure pass.
- R21 (iter123/127): stable digest seeds replace process-randomized Python hashes. Together with prior migrations this fixes the four specifically named scripts; a complete provenance manifest remains open.

Validation: `tests/test_review19_closure.py` exercises synthetic replay/aggregation, both oracle summary/bootstrap consumers, C6 target fields, full iter143 and iter195 output entry points using temporary fixtures, malformed seed logs, and subprocesses with distinct PYTHONHASHSEED. Historical results were not loaded or regenerated; no training, provider calls or paid jobs were run.

## Output and policy closure: corrected-review19-v4

- R12: iter119/127 share the iter119 target policy (floor .50, strict predicted-target comparator, baseline below floor); iter135/155 share high-ZVF `>= tau` firing, including equality. The largest candidate remains a fallback, not a feasibility certificate.
- R17: iter107 pooled failure shares now use a within-method stratified step bootstrap rather than point-estimate CI placeholders; zero exclusion means lower endpoint above zero. This is conditional step uncertainty, not independent training-seed inference. Iter115 replay and seed-summary writes were already fixed in v3; a temporary-fixture main-path regression now verifies both advertised artifacts and the replay target status.
- R20: hybrid-unification rejects missing/nonfinite/out-of-range ZVF. Iter171 joins explicit step identities before jointly excluding missing/nonfinite metric pairs, rejects duplicate IDs, and records the conditional step pairing. Matching step IDs does not establish independent training replication or a shared randomized prompt design.

All newly changed analyses target a separate corrected-review19-v4 output subtree. Synthetic tests cover boundaries, policy parity, heterogeneous pooled intervals, missingness/reordered identities, duplicate rejection, and actual iter115 output files. Historical artifacts were not loaded, regenerated, or certified; no training/provider/paid jobs were run.
