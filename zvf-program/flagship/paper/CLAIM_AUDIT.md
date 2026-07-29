# Claim-to-evidence audit

Date: 2026-07-27  
Scope: `zvf-program/flagship/paper/main.tex`  
Policy: a claim is publishable here only when it is proved under explicit assumptions, reproduced from a content-addressed repository artifact, supported by a primary source, or labeled as proposed/unrun.

## Claim ledger

| ID | Manuscript claim | Status | Evidence or falsifier |
|---|---|---|---|
| C1 | A conditionally i.i.d. binary group is homogeneous with probability `p^G + (1-p)^G`. | Sourced background | Analytic identity and arXiv:2605.07689. It is not claimed as novel and does not cover correlated or nonbinary outcomes. |
| C2 | Under the same model, prompt heterogeneity increases expected degeneracy relative to plugging in the mean success probability. | Sourced background | Strict convexity and Jensen; arXiv:2605.07689. |
| C3 | Under within-group i.i.d. sampling, independent groups, and stationary `p`, the first mixed group has a geometric waiting time and the displayed reliability formula. | Proved | Proposition 1. `verify_claims.py` provides stable floating-point sanity checks; the manuscript proof is authoritative. |
| C4 | For fixed `G`, boundary rollout cost is `1/epsilon + O_G(1)`; for every `G`, complete-group sampling obeys `G/q_G(epsilon) >= 1/epsilon`. | Proved | Proposition 2, including the fixed-`G` qualification and union-bound proof. The paper makes no adaptive-stopping claim. |
| C5 | In the two-action problem {stop, commit to all `m` batch rollouts}, compatible latent states reverse the optimal action when `Delta>0`, yielding positive two-point minimax regret. | Proved | Proposition 3 and its numerical sanity-check witness. It is not a theorem about arbitrary sequential controllers. |
| C6 | For a finite action set, the conditional gross value of side information is nonnegative. | Proved, standard | Conditional value-of-information inequality. Net value subtracts acquisition/action cost; the equality condition is stated. |
| C7 | The frozen S1 receipt is `S1_PASS`, with 14 intended cases per stack, a shared 36-case controller matrix, exact tolerances, and no receipt errors. | Artifact-verified | `implementation_freeze.json`; receipt hashes, source hashes, counts, policy/case sets, and the S1 `1e-8` threshold are asserted by `verify_claims.py`. The campaign's separate `1e-12` floor and TRL `1e-4` epsilon are also source-asserted. |
| C8 | Native verdicts contain 4 material and 1 not-tested case for TRL, and 1 material and 4 not-tested cases for verl. | Artifact-verified | Both receipt verdict vectors and exact count maps are asserted. These are reference-difference labels, not bug claims. |
| C9 | r4-2 contains exactly 31 manifest jobs with status counts 10 accepted, 14 contract-infeasible, 2 infrastructure-failed, 1 validation-failed, and 4 quota-pending. | Artifact-verified | `launch_manifest.json` and `supervisor_state.json` are parsed; manifest/state ID equality, manifest counts/fingerprint, every exact disposition set, and both file digests are asserted. Appendix D gives the exhaustive crosswalk. |
| C10 | Exactly six final scientific acceptance records exist, all balanced equal-length units with the six stated condition/seed IDs. | Artifact-verified | Exact accepted-ID set, acceptance filenames, regime, package pins, source-manifest shape, and corpus/step-0 matching invariants are asserted. |
| C11 | The six records contain 62 or 65 stored joint-zero relations, no one-sided-zero relation, minimum nonzero cosine at least 0.999844, and maximum nonzero relative L2 at most 0.017681. | Receipt-verified | All 600 stored records are inspected. Joint-zero requires both stored norms to equal zero and both angle/distance fields to be null. Nonzero bounds are range-checked. Gradients are not recomputed from private checkpoints. |
| C12 | Evaluation uses six checkpoints over the same 128-example held-out set; all starts are 20/128; the table's final counts and charged-token totals match the ledgers. | Receipt-verified | Evaluation step grid, denominator, unique-row count, accuracy arithmetic, shared step-0 evidence hash, and positive token/FLOP fields are asserted. Per-example predictions and FLOPs are not regenerated. |
| C13 | The filtered positive-control construction failed its frozen CV gate: observed CV 0 was below 0.35 under the registered Qwen3-1.7B/512-token contract. | Artifact-verified feasibility outcome | Original seed-11 `ReplayContractError`, preregistered threshold/model/cap, and exact filtered job disposition are checked. Descendant summaries were corrected to remove an erroneous Qwen2.5-0.5B label; no gate, status, or scientific artifact changed. |
| C14 | The preregistered causal training hypothesis and confirmatory study are not established. | Required non-claim | The positive-control regime was never constructed, the balanced screening matrix is incomplete, and no confirmatory seeds ran. |
| C15 | RLM traces may resolve some terminal-outcome ambiguity. | Proposed | Prospective protocol only. It is falsified by failed root-held-out calibration/action reversal, loss under matched cost, or leakage/conformance explanations. No RLM result is reported. |

## Verification scope

`verify_claims.py` has three explicit scopes:

1. executable sanity checks for the displayed formulas, using stable `log1p`/`expm1` calculations;
2. cryptographic integrity and internal invariants of S1 receipts and their source files;
3. cryptographic integrity and internal invariants of campaign state, acceptance records, stored gradient diagnostics, evaluations, and compute ledgers.

It does **not** symbolically prove the propositions, recompute gradients from model checkpoints, regenerate evaluation predictions, or rerun training. “Receipt-verified” is used where that distinction matters.

## Rejected or narrowed statements

- Removed the expected-gradient-norm bound, “double exponential” language, absorbing-prompt claims, entropy/mutual-information theorems, and automatic process-reward claims from earlier drafts.
- Recast first-contrast and boundary results as elementary cost-accounting corollaries of known starvation analysis.
- Restricted the boundary asymptotic to fixed `G` and added an exact all-`G` lower bound; removed the adaptive-controller extrapolation.
- Restricted the decision result to two actions with a prepaid batch and `Delta>0`; removed “every outcome-only policy.”
- Replaced the generic optimizer no-op claim with a protocol-specific statement: beta/weight decay are zero, accumulation is one, and the loop explicitly skips `optimizer.step` while advancing the scheduler.
- Defined exact-zero semantics and replaced “maximal semantic divergence” with “categorical material divergence.”
- Removed the unsupported 59/3/38 corpus decomposition and partial step-60 claim from the manuscript.
- Reframed the empirical campaign as a registered feasibility outcome/protocol postmortem, not a scientific negative result.
- Corrected the value-of-information conditioning and exact group-size endpoint domain.

## Review bundle

`build_review_bundle.py` creates deterministic `review_bundle.zip` with:

- manuscript source, PDF, bibliography, this audit, verifier, and review disposition;
- S1 source, tests, amendment, freeze, and both stack receipts;
- root `pyproject.toml` and `uv.lock` for the pinned S1 environment check;
- the frozen preregistration;
- r4-2 state, manifest, execution notes, acceptance/results/recovery evidence; and
- frozen source-provenance archives.

The internal `MANIFEST.sha256` covers every payload. `REVIEW_BUNDLE.sha256` records the outer archive hash.

After extraction:

```bash
python3 verify_claims.py --repo-root repository
```

## Verification commands

From the repository root:

```bash
python3 zvf-program/flagship/paper/verify_claims.py
python3 -m pytest -q platform_hybrid/experiments/signal_starvation/test_metrics.py
PYTHONPATH=zvf-program/zvf-triage/src python3 -m pytest -q zvf-program/zvf-triage/tests/test_controller.py
python3 zvf-program/flagship/paper/build_review_bundle.py
python3 /Users/arvind/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  /Users/arvind/Developer/tinker-rl-lab/zvf-program/flagship/paper/main.tex
```

Expected focused-test baseline: 5 signal-metric tests and 16 controller tests pass. Those tests establish checked implementation behavior, not improved learning.

Clean-extraction S1 reproduction was also run from the generated bundle:

```bash
cd /absolute/path/to/review_bundle/repository
PYTHONPATH=zvf-program uv run --isolated --no-project --python 3.12 \
  --with trl==1.2.0 --with transformers==5.5.4 \
  python -m unittest flagship.s1.test_reference \
  flagship.s1.test_receipts flagship.s1.test_trl_adapter -v

cd /tmp
PYTHONPATH=/absolute/path/to/review_bundle/repository/zvf-program/flagship \
  /tmp/tinker-s1-verl-py311/bin/python -m unittest s1.test_verl_adapter -v
```

Observed baseline: **35/35** common+TRL tests and **10/10** verl tests passed.
The verl environment was freshly installed with Python 3.11,
`verl==0.3.0.post1`, `torch==2.4.0`, and `transformers==4.45.2` as documented in
the bundled S1 README.

## Primary-source bibliography check

- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- Gradient starvation: <https://arxiv.org/abs/2605.07689>
- Advantage collapse / AVSPO: <https://arxiv.org/abs/2605.21125>
- DAPO: <https://arxiv.org/abs/2503.14476>
- HybridFlow / verl: <https://arxiv.org/abs/2409.19256>
- TinyV: <https://arxiv.org/abs/2505.14625>
- VerifyBench: <https://arxiv.org/abs/2505.15801>
- Recursive Language Models: <https://arxiv.org/abs/2512.24601>
- RLM harness study: <https://alexzhang13.github.io/blog/2026/harness/>

## Audit conclusion

After the listed corrections and bundle verification, the manuscript is suitable as an evidence-bounded methods/reproducibility preprint and registered-feasibility postmortem. It is not supportable as a causal-training, new-controller, or RLM-results paper. Promotion to those claims requires a redesigned feasible positive control, fresh preregistration, complete multi-seed matched-cost execution, and confirmation.
