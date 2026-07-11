# Viva Voce Script — 20 minutes (14 slides + Q&A buffer)

Pacing: ~85 seconds per slide ≈ 19½ minutes of speech; practice at a pace
that leaves 30–60 s slack. Bold = say with emphasis. (Numbers in brackets are
cumulative time targets.)

---

## Slide 1 — The problem in one picture [0:00 → 1:30]

Good morning. My thesis is about a blind spot in how we train language models
with reinforcement learning.

The dominant method today, GRPO, trains on *groups*: for every prompt we
sample G answers, and each answer is scored *relative to its own group*. The
consequence is structural: **if every answer in a group gets the same reward,
that group teaches the model nothing** — the advantages are identically zero.

That happens in two opposite situations: the model fails every attempt — the
task is too hard — or it solves every attempt — the task is mastered. And the
second case is the dangerous one, because the reward curve reads *success*
precisely at the moment learning has stopped. My thesis names, measures, and
budgets that phenomenon.

## Slide 2 — The instrument [1:30 → 3:00]

The instrument is one statistic: the **Zero-Variance Fraction** — the fraction
of groups in a step whose rewards have zero variance. Its complement is
gradient utilisation: how much of your batch is actually teaching the model.

Three properties matter. It's free — computed from rewards you already have.
Under binary rewards it has a closed population form, p to the G plus
one-minus-p to the G, which is a **U-shape in difficulty** — starvation at
both ends. And because of that U-shape, ZVF alone cannot tell mastery from
incapacity — it must always be read **together with mean reward**. That pair
is the diagnostic.

## Slide 3 — Claim contract [3:00 → 4:30]

I want to be precise about what I'm defending, so my thesis is organised as a
claim contract.

**Claim one:** ZVF plus reward is a cheap, *calibrated* online diagnostic of
zero-advantage regimes. **Claim two:** at a matched rollout budget, group size
selects *which end* of training starves.

And four explicit non-claims: I claim no predictive or causal power for ZVF;
no data-dependent optimal group size — in fact I'll show you why that's
impossible under my own objective; no controller efficacy; and no
generalisation beyond the stack I actually measured: one 8-billion-parameter
model, GSM8K-family tasks, one managed training API, mostly one to three
seeds. Everything I show you lives inside those fences.

## Slide 4 — Scope and infrastructure [4:30 → 5:45]

Briefly, the infrastructure. Training runs on a managed LoRA API; evaluation
is vLLM across three GPU backends; and there's an observational corpus of
logged runs across seven libraries up to 671B parameters.

One number shaped the whole thesis: the base model already solves GSM8K at
**91% pass@32**. There's almost no capability headroom — so nothing I show
you claims capability gains; the claims are about *training dynamics*. And
every result is seeded, manifested, mirrored to Weights & Biases, and
resumable from checkpoints — that discipline gets a chapter of its own.

## Slide 5 — Claim 1, the key exhibit [5:45 → 7:15]

Here is the single most important table in the thesis. Same budget — 2,560
rollouts per arm, two seeds each. Left: G=2 for 160 steps. Right: G=16 for 20
steps.

Both reach comparable final training reward. But look at the late-run ZVF:
the G=2 arms end at **0.75 to 1.0** — nearly every group is all-correct,
contributing zero gradient — while G=16 never exceeds 0.25.

If you monitored only reward, the G=2 run looks like your best experiment —
reward one-point-oh. ZVF tells you the truth: **training ended some time ago;
you're burning compute sampling answers you already know**. Reward tells you
where the policy is; ZVF tells you whether training is still moving.

## Slide 6 — Calibration [7:15 → 8:30]

For a diagnostic to be trustworthy it needs error bars. ZVF is a
binomial-proportion estimator, and we validated its confidence intervals on
512-prompt pools: the **Wilson interval covers at 0.95 to 0.98 in every
setting we tested**, and it's what we use.

One subtlety we found: under curriculum ordering — sorting prompts by
difficulty — global coverage collapses. But the interval remains calibrated
for the *local*, curriculum-stage ZVF. So that's a labelling requirement —
know which quantity your interval covers — not an invalidation.

## Slide 7 — The reliability budget [8:30 → 9:45]

The second theorem turns ZVF into a planning tool. If ZVF is high, how many
rollouts must I *budget* to guarantee one informative group? The answer is
geometric: G times log-delta over log-ZVF — and I want to be careful, because
we initially over-claimed this. It is a **quantile, not a minimum**: signal
can arrive in your very first group. What it gives you is a guarantee level.

Empirically it is essentially exact — the model quantile matched the observed
quantile at **ratio 1.00 in all six difficulty strata**. In the hardest
stratum, guaranteeing one usable gradient at ninety-percent confidence costs
160 rollouts. That's what "wasted compute at high ZVF" means, precisely.

## Slide 8 — An honest negative result [9:45 → 11:15]

The third theorem was supposed to give an optimal group size that depends on
your data. It doesn't — and I think this is the most instructive slide in the
deck.

There's a two-line algebraic identity: one minus p-squared minus
one-minus-p-squared is exactly 2p(1−p), and the same at G=3 with a 3. Divide
by G and **J(2) equals J(3) for every difficulty prior** — and everything
above G=3 is worse. The optimum is always {2,3}, *no matter the data*.

Which means our earlier "theory agrees with experiment" validation was
guaranteed by algebra — it tested nothing. An external adversarial review
caught this in my own theory paper; I verified it, corrected the paper the
same day, and report it here as a negative result: **per-rollout accounting
can never justify adaptive group sizes. You need a richer objective.**

## Slide 9 — Claim 2 [11:15 → 12:30]

So what *does* the data say about group size? Static sweeps are genuinely
inconclusive — the ranking changes depending on whether you hold steps or
tokens constant.

The matched-rollout-budget experiment resolves it as a **schedule** question:
small G converts your budget into more optimiser steps early — consistent
with the per-rollout theory — and then starves in the endgame; large G keeps
signal alive to the end. This also qualifies the "GRPO is secretly
contrastive" equivalence claims: they hold on final reward but *not* on
signal availability, and they decay as training scale grows.

The obvious next step is a ZVF-triggered controller. I've designed it —
measure with the Wilson bound, escalate when the reliability budget exceeds
the step budget — but I make **no efficacy claim**: the decisive
compute-matched trial against static G=16 is future work.

## Slide 10 — Loss form [12:30 → 13:45]

A parallel question: does the loss form matter? Dr.GRPO exists because GRPO's
normalisation allegedly inflates response length.

Six uncapped arms, three seeds per loss, 1,024-token budget: **no length
inflation in either loss** — every arm's completions *shrink* six to twelve
percent — and no ZVF separation either. At this scale, the loss-form choice
has no observable footprint. That's not a superiority claim in either
direction; it's a reporting lesson — at this scale such comparisons measure
stack noise unless everything else is pinned down.

## Slide 11 — The incident [13:45 → 15:15]

And here's why I believe that lesson viscerally. The first version of that
panel was **invalid**: the loss flag was documented — the commit message even
claimed the mapping — but it was never wired. Both arms silently trained the
same objective. And nothing in the outputs revealed it. Rewards looked
plausible, lengths looked plausible, ZVF looked plausible. We found it only
by reading the runner.

The response is the protocol I now defend: invalidate loudly, preserve the
invalid artifacts under explicit names, fix, rerun the same day — and note
that the corrected panel *reversed* one of the invalid panel's conclusions.
The lesson: **stack identity cannot be certified from plausible-looking
outputs.**

## Slide 12 — From incident to standard [15:15 → 16:45]

That incident, plus measured cross-stack evidence, became a standard: eight
reportable items, each earning its place because it *flipped a result in our
own corpus* — a backend swap that silently bundled a different base
checkpoint moved final reward across a seventeen-fold span; the same "DAPO"
label produced ZVF 0.00 on one stack and 0.58 on another; a reward jitter
below the parser's resolution collapsed ZVF from 0.158 to zero.

The standard ships as tooling: a machine-readable registry of stacks and
variant deltas, and a stackdiff tool that grades a pair of runs for
label-flip risk from manifests alone.

## Slide 13 — Limitations and roadmap [16:45 → 18:00]

Limitations, stated plainly: one model, one task family, one closed stack,
one to three seeds; the theory has declared proof gaps; ZVF as defined needs
discrete rewards.

The publication plan was itself externally reviewed and is gated: the
diagnostic paper — claims one and two — is submittable without any controller
story; the controller paper waits for a compute-matched win over static G=16
*and* reward-only schedules at three-plus seeds; the survival audit of other
methods waits until we can run it on an open stack with tooling that would
have caught our own bug automatically. Credibility first, audit second.

## Slide 14 — Conclusion [18:00 → 19:00]

To conclude in three lines. Under verifiable rewards, group-relative training
starves at **both ends** of difficulty. ZVF, read with reward, sees both
walls — including the one the reward curve is structurally unable to show.
And group size decides which wall your budget hits.

The thesis, the seventeen consolidated working papers, and every artifact
behind every number are public in the repository. Thank you — I'm happy to
take questions.

---

## Anticipated questions (with answers)

**Q: Why not just use dynamic sampling (DAPO-style) and skip degenerate groups?**
A: Complementary, not competing. Dynamic sampling *buys* ZVF=0 at ~45% extra
rollout cost — a purchase, not a prize. ZVF is the meter that tells you
whether the purchase is worth it; our offline bandit pilot suggests targeted
reallocation recovers 97% of an oracle's informative groups.

**Q: Isn't ZVF just 1 − accuracy-variance — trivially derivable?**
A: Yes, it's simple — that's the point. The contribution isn't the formula,
it's the calibration (Wilson coverage), the budget theorem, the two-walls
characterisation, and the demonstration that reward alone is structurally
blind to the mastered wall.

**Q: Your G=2 arms hit reward 1.0 — isn't that just "training worked"?**
A: On the 256-prompt training pool, yes — that's mastery of the sampled set,
not capability. The point is diagnostic: reward can't distinguish "converged
productively" from "burning budget on zero-gradient groups". Held-out
capability barely moves (82.0→83.3, p=0.26), consistent with sharpening.

**Q: Why should we trust your numbers after the loss-flag bug?**
A: Because of what we did next: preserved the invalid artifacts, re-ran,
reported the reversal, then commissioned three independent audit rounds that
found and fixed eight further defects — and a fourth round found only
fabrications, which we refuted against source. Every claim has a named
artifact; the trail is public.

**Q: What's the single next experiment?**
A: The pre-registered compute-matched bakeoff: ZVF-triggered G-escalation vs
static G=16 vs a reward-only schedule, ≥3 seeds, held-out time-to-threshold
as the endpoint. It decides whether the controller paper exists.

**Q: Does any of this hold off GSM8K / off Qwen?**
A: Unknown, and stated as a non-claim. The MATH-500 partial result says
GSM8K-trained gains do not replicate on hard math — supporting the
sharpening interpretation and the fenced scope.
