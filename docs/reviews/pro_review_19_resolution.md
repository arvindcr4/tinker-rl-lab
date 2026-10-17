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
