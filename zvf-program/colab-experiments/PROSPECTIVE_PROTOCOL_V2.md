# Colab exploratory protocol v2

Applies only to future executions of the revised scripts. Prior logs, result JSON,
manuscripts and claims are historical evidence; no result is recomputed here.
Ship `protocol_stats.py` alongside scripts that import it (single-file upload is
no longer sufficient). E2 variants are legacy pilots/smokes, not interchangeable
replications of a final study; script/config/package hashes distinguish them.

- E1: evolving-policy, constructed-difficulty association. Seed-cluster bootstrap,
  raw per-seed correlation and difficulty summaries; three independent seeds
  cannot support strong inference. The measured summed standardized policy-gradient
  norm is not an unbiased estimator of binary reward variance.
- E3: `threshold_adaptiveG` names the actual threshold treatment. All arms have a
  fixed outer-step budget, explicitly unmatched compute, with rollout/token/update
  accounting. This is not a packaged-controller efficacy or equal-compute claim.
  A future equal-budget experiment requires a separately frozen rollout limit,
  whole-group final-step rule, and source-hashed controller parity tests.
- E4: hierarchical prompt/original-rollout bootstrap, conditional on calibration.
  Groups are sampled with replacement. Plug-in p-hat prediction is exactly the
  conditional resampling expectation; agreement cannot independently validate the
  true-p law. True p is unknown. Bootstrap intervals do not undo pilot selection.
- E5: exploratory partial-credit geometry only. Pilot filtering, live-group
  selection, population standardization, length normalization and layer slice
  are interventions. A separate, not-yet-run theory-conformance protocol must use
  binary rewards, declared per-group normalization, all groups (including zeros),
  and the same disjoint heldout prompt distribution as the geometry arm.
- E6: shared initial pool, common disjoint mixed-difficulty heldout, whole-group
  rollout cap without overshoot. Early stopping and remainder mean unequal used
  budgets; never claim matched compute. Drop-disabled arms must not silently
  remove prompts from controller observations. Controller is experimental until
  parity with a source-hashed canonical controller is demonstrated.
- E7: the frozen matrix includes bf16/fp32 × LR {1e-6,2e-6}, plus separate attention
  and temperature contrasts. Paired per-seed deltas are exploratory (two seeds).
  A zero-update baseline does not establish post-training distribution equivalence.

## Claim authority
The audit's preregistration plus exact aggregate analysis determine current verdicts;
LaTeX is a rendering, not another analysis. DAPO MDE80 0.0101158932289 exceeds the
0.01 margin: INCONCLUSIVE. Historical DISAPPEARS notes remain superseded history.
Do not infer publication readiness from historical execution status alone.
