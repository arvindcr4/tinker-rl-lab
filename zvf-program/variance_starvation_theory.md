# Variance starvation: corrected mathematical contract, v2

Revision: 2026-10-17. Supersedes the earlier working framework's Theorems 2–8.
The withdrawn claims are not supporting evidence for a controller or historical
run. This document changes no stored results and authorizes no reruns.

## 0. Setup and estimators

For a fixed prompt x, draw G >= 2 independent completions from a differentiable
policy. The verifier R(x,y) is binary, and p = Pr(R=1 | x). Let rbar be the
sample mean and sigma² = G^-1 sum_i (r_i-rbar)² (population-divisor sample
variance, not the unbiased estimator used in the diagnostic package).

Freeze sampled advantages A_i = (r_i-rbar)/(sigma+epsilon) when differentiating.
A scalar local policy-gradient surrogate is
L(theta) = G^-1 sum_i A_i log pi_theta(y_i | x).
Its gradient estimator at the sampling policy is
g = G^-1 sum_i A_i s_i, s_i = grad_theta log pi_theta(y_i | x).
A clipped/off-policy objective needs separate ratio, clipping and support
assumptions; the following statements concern this on-policy estimator only.

## 1. Exact binary degeneracy probability

Pr(sigma=0 | x) = p^G + (1-p)^G.
Pr(reward contrast | x) = 1-p^G-(1-p)^G.

This is the probability of a sampled all-correct or all-wrong group. A finite
uniform sample does not establish population p=0 or p=1. GU = 1-ZVF measures
the observed fraction of groups with reward contrast, not optimizer activity.
Generation rejection/resampling changes which population the statistic describes.

## 2. Correct gradient bound

Assume ||s_i|| <= L almost surely. Jensen's inequality, conditional on the full
sample, gives

||g||² <= G^-1 sum_i A_i² ||s_i||²
       <= L² sigma²/(sigma+epsilon)².

Taking expectation, without moving the random denominator outside it:

E[||g||² | x] <= L² E[sigma²/(sigma+epsilon)² | x]
              <= L² [1-p^G-(1-p)^G].

The original variance-dependent bound is withdrawn. For a Bernoulli-logit
policy, G=2, p=.5, epsilon=.5: mixed samples have |g|=.25 and probability .5,
so E[g²]=.03125. The earlier claimed .0078125 is false. The new samplewise
bound with L=.5 gives exactly .03125 in this example.

## 3. Current-group zero-contribution lemma; no absorption claim

If all rewards in this sampled group coincide, A_i=0 and its on-policy
surrogate contribution is zero. That does not freeze shared parameters:
other prompts, other losses, KL terms, optimizer momentum or weight decay can
change them. Permanent absorption would require population endpoint support
that is invariant under every update, no other parameter-changing term, and
an optimizer state preserving that invariant. Those assumptions have not been
established for these experiments. Prior “absorbing” claims and weak-coupling
corollaries are withdrawn; empirical observations are sampled degeneracies.

## 4. Reward and trajectory equivalence

Equality of completion distributions alone does not imply equality of reward
laws across prompts: the gold answers/grading maps may differ. Include the
map R(x,.) in the equivalence identity, or assume a common grading map.
For discrete T and R with R a deterministic function of T under one fixed
joint law, H(R) <= H(T), and the resolution gap is H(T)-H(R)=H(T|R).
Without that deterministic coarsening, no containment or entropy ordering is
asserted. Two reward-law classes, for example Bernoulli(.2) and Bernoulli(.8),
do not imply starvation. Population-wide endpoint degeneracy requires the
prompt-conditioned reward laws to be supported on {delta_0, delta_1}.

## 5. Epsilon and fixed-size binary groups

With k successes, sigma² = k(G-k)/G². The smallest positive sigma is
sqrt(G-1)/G, attained at k=1 or G-1. For G=8 this is .3307189139.
Thus epsilon=1e-4 or 1e-6 cannot dominate a positive binary sample standard
deviation at this G. Approaching a population endpoint increases the
probability of exactly zero variance; it does not create arbitrarily small
positive sample variances at fixed G. Continuous/shaped rewards require a
separate analysis. The earlier epsilon-paradox explanation is withdrawn.

## 6. Group size: exact and sufficient guarantees

For target contrast probability delta in (0,1), choose the smallest integer
G >= 2 satisfying p^G+(1-p)^G <= 1-delta. The earlier bound that discarded
the positive second term was necessary, not sufficient. Example: p=.6,
delta=.5, G=2 has degeneracy .52 and fails.

Let q=max(p,1-p)<1. Since p^G+(1-p)^G <= 2q^G, a sufficient choice is

G >= max(2, ceil(log((1-delta)/2)/log(q))).

A uniform guarantee requires both tails bounded away from endpoints, e.g.
p in [a,1-a] with a>0. An upper bound on p alone permits p=0 and cannot
ensure contrast. Half-degeneracy upper thresholds, computed from the exact
formula (the lower thresholds are their reflections):

| G | p with p^G+(1-p)^G=.5 |
|---|---|
| 2 | .5000000000 |
| 4 | .8406250193 |
| 8 | .9170040427 |
| 16 | .9576032807 |
| 32 | .9785720621 |

Near an endpoint p=1-h, contrast is G h + O(h²) for fixed G>1; products
of fixed powers of endpoint distances are polynomial, not double-exponential.

## 7. Information identity and withdrawn improvement bound

For a deterministic verifier, I(Y;R | x)=H(R | x), because H(R | Y,x)=0.
This identity does not bound parameter-space movement or learning improvement
without score/Fisher geometry, smoothness and step-size assumptions.
Counterexample: p(theta)=sigmoid(10 theta), theta=0, G=2, epsilon=.5,
learning rate .1. Half the groups are mixed, produce g=2.5, and move theta
to .25. Expected accuracy gain is .5*(sigmoid(2.5)-.5)=.2120709100,
whereas eta I=.1 log(2)=.0693147181. The old eta-I improvement theorem is
withdrawn. No replacement optimization-rate theorem is claimed.

## 8. Diagnostic and compute scope

Always record reward source (raw verifier versus shaped training reward),
reduction, scale, variance divisor and tolerance. Raw-verifier contrast and
training-reward contrast may disagree; measure loss tokens, gradients and
parameter changes separately. “Matched compute” must identify a prospective
resource: all actual generated tokens including rejected/resampled samples is
one possible choice. Rollouts times token cap is only an upper bound. Report
attempted/accepted groups, generated/training tokens, optimizer steps and GPU
seconds separately. Three unanimous nonzero paired signs have two-sided
sign-test p=.25, so they do not justify a conventional significance claim.
