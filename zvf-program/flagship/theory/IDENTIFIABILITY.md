# Provisional identifiability result: same failure signal, different action

Status: supporting proposition only; rejected as a standalone novelty claim on
2026-07-20 after the prior-art audit in `NOVELTY_AUDIT.md`.

## Restricted setting

Consider binary-reward group-relative training after a group of `G >= 2`
completions all receives reward zero. An outcome-only controller observes the
reward vector, empirical success rate, group size, homogeneous-group flag, and
the same finite primary-reward history. It does not observe reward-channel
health. There are two realistic latent regimes:

1. **Clean-hard.** The verifier is correct and each additional completion has
   independent success probability `q in (0,1)`.
2. **Broken-verifier.** The reward channel deterministically maps both correct
   and incorrect completions to zero until repaired.

The available actions are `retry` with `m` additional completions or `recheck`
with one independent known-correct calibration completion. Values and costs are
bounded and nonnegative; the construction does not use adversarial reward
functions or inaccessible latent observations. The broken verifier is silent:
runtime error state, latency bucket, and reward-code version are matched, so
ordinary telemetry does not resolve the latent regime.

## Proposition

Let `Delta_H > 0` be the clean-hard utility advantage of retry over recheck and
`Delta_B > 0` the broken-verifier utility advantage of recheck over retry. The
two regimes induce the same primary observable state, but their optimal actions
are opposite. Any randomized policy measurable only in that state has positive
worst-case regret at least

`Delta_H * Delta_B / (Delta_H + Delta_B)`.

A perfect known-correct calibration probe resolves the action ambiguity but is
not free. If its charged cost is `d` and it misclassifies reward-channel health
with probability at most `epsilon`, its worst-case regret is at most
`d + epsilon * max(Delta_H, Delta_B)`.

## Proof

Because the primary observation is identical, an outcome-only policy retries
with one probability `alpha` in both regimes. Its regret is
`(1-alpha) Delta_H` in clean-hard and `alpha Delta_B` in broken-verifier.
Minimizing the maximum equalizes the two terms, giving
`alpha = Delta_H / (Delta_H + Delta_B)` and the stated strictly positive lower
bound. A perfect calibration completion returns one under the clean verifier
and zero under the broken verifier, so the probe policy selects the correct
action in both regimes. The enriched policy still pays the probe cost in the
clean regime; misclassification adds at most the larger action gap.

The reversal holds on an open parameter region, not at a knife-edge. Writing
`U_R^H` for clean retry, its strict conditions are
`U_R^H + d > 0` and `B - d + m c > 0`, where `d` is probe cost, `B` repaired
signal value, and `m c` retry cost. Continuity preserves both inequalities in
a neighborhood. The executable test checks all 81 combinations in a bounded
neighborhood of the default witness.

## Executable witness

`action_reversal.py` fixes `G=8`, `m=8`, `q=0.10`, success value `1.0`, sample
cost `0.02`, probe cost `0.03`, and repaired-signal value `0.50`. It emits the
matched observable state, both utility tables, opposite optimal actions, the
positive minimax regret, and imperfect-probe bounds. The tests verify the action
reversal and bound algebra.

`reward_channel_regimes.py` grounds the latent regimes in the frozen E1 GSM8K
reward contract without modifying E1. That reward path returns `0.0` both for
a wrong answer and for a completion whose final integer is unparseable after
the configured marker. The executable pair uses the same all-zero primary
outcomes and matched silent telemetry under a clean `####` parser and a
misconfigured `FINAL:` parser. A known-correct `#### 42` control traverses the
same configurable reward function and yields one versus zero. The script first
verifies the complete E1 source hash
`986811e3e78fe86ffcbede4a98599ada167ff1975b3341eef391eb2b2e7fe8c6`.

## Practical resolving observation

The named extra observation is reward-channel health measured with a
known-correct calibration completion evaluated by the same reward path. This
is available before another training update and is charged explicitly. It is
not equivalent to looking at more primary failures: under a broken verifier,
additional ordinary completions remain observationally zero.

## Falsifiers and kill criteria

Drop the theorem contribution if any of these survives review:

- real reward pipelines cannot supply a known-correct calibration completion;
- the calibration path bypasses the failure mode rather than traversing the
  same reward channel;
- verifier failure can be ruled out by already-observed runtime evidence, so
  the latent regimes are not observationally matched in practice;
- retry remains optimal after charging all generation and probe costs in the
  broken-verifier regime, or recheck remains optimal in clean-hard;
- the result reduces to a generic no-free-lunch statement without predicting a
  measurable action reversal in executable task regimes.

The next theory task is adversarial: enrich the observable state with ordinary
runtime telemetry and determine whether the two regimes can still be matched.

## Submission role after novelty audit

The construction remains useful for explaining why an all-zero reward vector
does not identify whether to spend on more samples or inspect the reward path.
It is not the flagship theorem. Prior work already establishes verifier false
negatives, their damage to RL training, dynamic secondary verification, and
format-sensitive verifier failures. The minimax lower bound is mathematically
correct but too close to a generic hidden-state decision argument to satisfy the
prespecified nontriviality threshold. It may appear as a short motivating
proposition or appendix sanity check, with those limitations stated explicitly.
