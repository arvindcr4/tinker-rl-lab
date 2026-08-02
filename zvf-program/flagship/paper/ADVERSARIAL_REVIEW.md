# Independent adversarial review and disposition

Date: 2026-07-27  
Reviewer: external ChatGPT web review, Extra High reasoning tier (the requested
Pro tier was unavailable).  
Materials initially supplied: compiled PDF, claim audit, and verifier.

## Initial verdict

Major revision; reject in the supplied form. The reviewer judged the mathematical
core mostly correct after narrowing, but found that the attachment set did not
substantiate the artifact claims and that the incomplete campaign could only
support a feasibility/protocol postmortem.

## Findings and disposition

| Priority | Finding | Disposition |
|---|---|---|
| P0 | Evidence was not self-contained; verifier assumed repository layout. | Added deterministic content-addressed review bundle, portable `--repo-root`, explicit missing-evidence errors, source/receipt hashes, and bundle boundary. |
| P0 | Full scalar objective, interventions, controller rule, and case inventory were absent. | Added Eq. 4, exact four-condition definitions, full-triage rule, and complete case/native-verdict appendix. |
| P0 | Campaign cannot test the preregistered causal hypothesis. | Reframed throughout as a registered feasibility outcome/protocol postmortem; explicitly not a scientific negative result. |
| P1 | Zero gradient was overgeneralized to optimizer no-op. | Defined the audited gradient object and exact-zero semantics; limited no-op wording to the frozen skip-step protocol and listed beta/weight-decay/accumulation assumptions. |
| P1 | Boundary asymptotic was non-uniform in changing `G`. | Restricted the expansion to fixed `G`, added `O_G` notation and exact all-`G` lower bound, and removed adaptive-policy inference. |
| P1 | Decision theorem overclaimed arbitrary outcome-only controllers. | Renamed and restricted it to {stop, prepaid batch}; stated `Delta>0` and excluded sequential/variable-`m` policies. |
| P1 | Ledger and verifier scopes disagreed. | Rewrote the ledger; verifier now asserts native counts, exact IDs/regimes, status crosswalk, source hashes, stored zero norms/null metrics, evaluation grid, protocol, and endpoints. It explicitly does not recompute gradients. |
| P1 | Evaluation denominator/protocol/budgets were missing. | Added N=128, exact split, decoding/parser, start/final counts, step grid, and charged tokens; ledgers retain intermediate curves and FLOPs. |
| P2 | VOI conditioning/equality, Appendix endpoint, and Proposition 1 assumptions were incomplete. | Corrected all three in manuscript and executable sanity checks. |
| P2 | Infrastructure anecdotes lacked review evidence. | Removed them as scientific support and labeled development logs as anecdotes. |
| P3 | Harness author, verifier citations, table ambiguity, appendix title, and RLM rendering. | Corrected attribution and positioning, separated TinyV/VerifyBench support, removed the ambiguous table column, retitled appendix, and repaired control characters. |

## Additional integrity correction

The review pass exposed metadata drift: fourteen descoped descendant summaries
named Qwen2.5-0.5B, while the preregistration and executed frozen runtime identify
Qwen/Qwen3-1.7B. The summaries are now model-neutral and a dated correction is
stored in supervisor state and execution notes. The original failed-validation
error, threshold, cap, job states, and scientific artifacts are unchanged.

## Remaining limitations

The artifact supports a methods/reproducibility preprint and registered
feasibility postmortem, not a causal-training, controller-benefit, or RLM-results
paper. Private checkpoints and raw corpora remain outside the public review
bundle; promotion to stronger claims requires a fresh feasible positive control,
complete screening, and confirmation.

## Final P1 follow-up

A separate final audit found no remaining P0 issue and identified three P1
reproducibility mismatches. All were corrected before release:

- S1's `1e-8` comparison tolerance is now explicitly separated from the
  campaign's `1e-12` zero floor and TRL's `1e-4` reward epsilon; the verifier
  reads all three constants from source.
- The verifier now parses `launch_manifest.json`, requires manifest/state ID
  equality, and asserts every exact disposition set and manifest digest.
- The bundle now includes `uv.lock` and `pyproject.toml`, the S1 README splits
  incompatible TRL/verl environments, and clean-extraction reruns passed 35/35
  common+TRL tests plus 10/10 verl tests.

The follow-up audit found no remaining P0/P1 issue and judged the revised
package publishable in its stated scope: a methods/reproducibility preprint and
registered-feasibility postmortem.

## 2026-08-02 truth-audit correction

A later audit supersedes the final sentence above on two points:

- The sole completed intended-full balanced cell satisfies the frozen joint
  cosine/relative-L2 predicate on 69/100 steps, below the preregistered 95/100
  threshold. High cosine alone was descriptive and cannot be called registered
  equivalence. The paper and claim ledger now report a failed mechanism cell.
- The live `pilot/objective.py` had changed after execution, while accepted
  receipts bound the executed file to SHA-256 `980a56a...`. The exact file was
  recovered from the previous content-addressed bundle, frozen under
  `pilot/provenance/r4-2-objective.py`, and made a hard input to the bundle
  builder. Clean extraction now verifies the manifest, exact source hash, and
  the 69/100 failed gate.

These corrections preserve the methods/postmortem scope but further narrow the
empirical claim: the campaign establishes receipt integrity, a failed balanced
mechanism cell, and an infeasible positive-control construction. It does not
establish implementation equivalence or a causal training effect.
