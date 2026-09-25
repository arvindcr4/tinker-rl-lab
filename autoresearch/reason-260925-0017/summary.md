# Adversarial theory review — M.Tech thesis

**Goal:** "improve my thesis so it is theoretically accurate"
**Mode:** convergent · **Domain:** research (testability, evidence support, explanatory power)
**Agents:** 15 cold-start (6 critics → 6 defenders → 3 blind judges) · 2.1M tokens
**Target:** `outputs/PES_Phase2_Third_Review_2026-09-24/thesis/` (216pp)

## Majority verdicts

| Claim | Verdict | Load-bearing |
|---|---|---|
| C1 ZVF formalisation / "descriptive identity" defence | WEAKENED (3/3) | yes |
| C2 Attribution design (one factor varied ⇒ attributable) | WEAKENED (2), FALLS (1) | yes |
| C3 Cross-scale null (P1) | WEAKENED (2), FALLS (1) | split |
| C4 ZVF–G correlation and confound handling | WEAKENED (3/3) | yes |
| **C5 Length bias (4 of 11 runs)** | **FALLS (3/3, unanimous)** | yes |
| C6 P7 signal-starvation theory / controller | WEAKENED (3/3) | yes |

Critic self-verdicts were harsher than the judges': C2, C3, C5 rated FALLS by their
own assigned attacker. Defenders conceded heavily — across six defences, 20
conceded or partially conceded versus a single refuted.

## The findings that matter most

Two were **independently verified by judges re-running the analysis**, not merely
argued. Those are listed first.

### 1. C5 — the length-bias claim does not survive (unanimous)

A judge applied the thesis's own stated rule (peak before 65% of training, t/p < 0.90)
directly to `platform_hybrid/experiments/master_results.json`:

- it fires on **18 of 29 GRPO runs (62%)** and **12 of 15 PPO runs**
- the thesis reports **4 of 11** (36%) — i.e. *below* the base rate the rule produces
  on traces with no decay at all
- condition 1 alone fires with probability ≈ 0.65 under an exchangeable null

Neither 65% nor 0.90 is pre-registered anywhere in the repository. The project's own
source for the flag reads it as "a generic instability detector", and the thesis cites
that same source as run-level proof that length bias is real. The Dr. GRPO control is
unmatched (0.5B easy-arithmetic against flagged 8B and 120B runs), and in the one
matched comparison the control is flagged too.

**Consequence:** the chapter's run-level length-bias evidence must be withdrawn or
restated as "the rule does not discriminate decay from noise; it fires on most runs
including the control."

### 2. C3 — 5 of 12 scaling anchors are interrupted runs, undisclosed

Verified against the anchor configs: exactly five of the twelve anchors are partial
runs whose regressed values are means of 3–5 logged steps rather than 20–30. The
dependent variable is not measured on a consistent footing across anchors.

Compounding it: "scale" is **perfectly collinear with the training recipe** — four
hyperparameters change together and only at the scale boundary — so the null cannot
separate "no scale effect" from "recipe changes offsetting scale changes". The
figure's noise floor is 3.3× too small, and large between-model spread is offered as
evidence *for* the null when it is evidence of model-family heterogeneity.

### 3. C2 — the headline PPO/GRPO null is pseudo-replicated, and the scale label is invented

- The n = 10 is **the autocorrelated last-10 steps of one run per arm**, not ten runs.
  ch01 states "ten runs per arm". Nominal n is inflated tenfold and the CI is thereby
  manufactured. The project's own rigour documentation declares these statistics
  uncomputable.
- The file cited as backing the Pillar-1 number is **a different model at 1/16th the
  scale**; the "Qwen3-8B" label appears in neither cited source. Chapter 4 invents
  the scale attribution.
- The η² decomposition that "motivates the whole attribution commitment" is four
  **marginal** one-way ANOVAs whose η² sum to **1.897** across 42 experiments. A
  partition sums to 1; these cannot be read as shares of variance.

### 4. C1 — the "descriptive identity" caveat is non-actual for the runner it protects

ch04 disclaims KL, auxiliary losses, clipping and token masking, quoting a source
that has them. The measured runner has none — its own driver is annotated "no KL/clip"
and the loss is a single advantage-weighted term. So on the system that produced every
ZVF number in the thesis, ZVF = 1 *does* entail a zero step gradient. The defence
pre-empts the only mechanism that would make the metric falsifiable by invoking
objective features the measured objective does not contain. No refutation condition is
stated for GU anywhere in the thesis.

### 5. Cross-cutting — the same four ZVF values are compared against two baselines with opposite signs

The thesis's own source states the correct baseline is the per-step average of the
closed form (0.60 at G = 8, not 0.33 at the pooled mean — Jensen convexity), that
measured ZVF then modestly **exceeds** the null by ≈0.09, and explicitly declines to
read that as evidence the independence null is a lower bound. ch06 instead reports
residuals of −0.126 to −0.232 against the other baseline and reads "consistently falls
below" as support. The sign of the result is an artefact of baseline choice, and only
the flattering choice is printed. The same contradiction appears in C4 and C6.

## What survives

The formal core holds up and was the judges' highest-scoring content: `h_G(p) = p^G +
(1−p)^G` is a genuinely refutable null, the ZVF definition is precise and
reproducible, and the descriptive runner-level identity is the thesis's soundest
material. Judges scored testability 6–9 on the formalisation even while rejecting the
inferences drawn from it.

The pattern across all six claims is consistent: **the mathematics is right; the
evidential claims built on it are not.** Several failures are the same failure —
a quantity compared against the wrong baseline, or a rule applied without checking its
base rate.

## Recommended disposition

| Finding | Fix type | Owner |
|---|---|---|
| C5 length-bias claim | withdraw or restate as a negative about the rule | author's call |
| C3 five interrupted anchors | disclose; re-run or drop those anchors | author's call |
| C3 collinearity of scale and recipe | restate the null's scope | author's call |
| C2 pseudo-replication | correct "ten runs per arm"; restate or drop the CI | author's call |
| C2 invented scale label | factual correction | mechanical |
| C2 η² summing to 1.897 | restate as marginal, not a partition | mechanical |
| C1 non-actual caveat | re-derive against the actual runner | mechanical |
| Sign-flip vs. cited source | align to the source's baseline and its caution | mechanical |

Nothing here was invented by the reviewers: every finding cites a thesis line, a
repository file, or an independent re-run.
