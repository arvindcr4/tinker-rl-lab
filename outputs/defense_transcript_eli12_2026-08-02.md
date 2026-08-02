# Faculty Review — Read-Aloud Transcript (ELI12, Indian English)

**Tinker RL Lab · Arvind C R · 02 Aug 2026 · 14 slides**
Pacing target: ~11 min spoken. Speak slowly and clearly, ~120 words/min. Times are cumulative. Keep sentences short. Pause after each bullet.

---

## Slide 1 — Title · target 0:45 (0:00 → 0:45)

Good morning everyone, and thank you for this time.

My name is Arvind, and today I will tell you about my work — teaching large language models to think better, and more importantly, catching when that teaching secretly stops working.

Let me give you the one-picture idea first. Imagine a basketball coach. He only compares your shots inside one drill. If all eight shots go in — or all eight miss — there is nothing to compare. The drill taught you zero.

The same thing happens inside GRPO training. The scoreboard shows a perfect score, but the model is learning nothing.

I built one number that counts those empty drills. I call it ZVF. This talk is about that number, and about two years of work around it.

---

## Slide 2 — The problem · target 0:45 (0:45 → 1:30)

So here is the problem in one line: the scoreboard says everything is fine, but actually, the coach is asleep.

Let me explain how GRPO works. For one question, the model writes eight answers. We grade each answer right or wrong. Then we compare the answers with each other.

Better than the group average? Push up. Worse? Push down. That push is the learning.

Now the catch: if all eight answers are right, or all eight are wrong — there is nothing to compare. The group contributes zero learning.

And the sad part: as the model gets better, this happens more and more. So when the scoreboard looks the best, the learning has quietly stopped.

One more thing: the same recipe on different software gives different results. I measured a seventeen-times difference from one small change. So we need a second number — and we need to report what actually ran.

---

## Slide 3 — GRPO in plain words · target 0:40 (1:30 → 2:10)

Let me explain GRPO in four simple steps.

Step one: one question, eight answers. We call that a group.

Step two: grade each answer right or wrong. Simple check — no expensive second AI needed.

Step three: compare. Better than the group average? Push up. Worse? Push down.

Step four: repeat — next question, next group, again and again.

The catch is here: when all answers agree, the group gives zero push. Training continues, GPU money keeps getting spent — but learning is zero. That is what I call signal starvation. And it is invisible on the reward curve.

---

## Slide 4 — The measuring stick: ZVF · target 0:50 (2:10 → 3:00)

So I built a simple measuring stick. I call it ZVF — Zero-Variance Fraction.

What does it do? It simply counts — out of all the drills, how many are empty? High ZVF means most drills teach nothing. Low ZVF means the model is actually learning.

The best part: this is not an approximation. The math is exact. We verified it to machine precision — 1.11 times ten to the power minus sixteen — on a 505-question audit.

It is cheap — no extra runs needed. It is trustworthy — the confidence interval works in every setting I tested. And you can watch it live, during training, not after.

One honest note: I am not claiming a controller yet. We test before we sell.

---

## Slide 5 — The program · target 0:45 (3:00 → 3:45)

This work happened over two semesters.

Semester three, with my group of six, we built the test ground — TinkerRL-Bench. Four frameworks, six backends, math tasks from simple addition to grade-school problems. We submitted it to NeurIPS 2026, with a full reproducible artifact.

Semester four, solo, I found the discovery — ZVF, the group-size dial, a reporting standard called MIN-REPORT-RL, and a registry.

In total: one thousand and one commits, eighteen documents, four hundred and eighty pages of papers — and one honest big test, which you will see next.

---

## Slide 6 — The findings · target 1:00 (3:45 → 4:45)

Four findings, in plain words.

First: starvation is real. Small groups — G equals two — end in a wall where every drill is empty. ZVF is close to one, while the reward reads one point zero. Large groups, G equals sixteen, keep learning till the end.

Second: group size is the dial. G equals four wins our 505-task utility audit. But there is no single best G. I proved the math always favours small groups — and I report that negative result openly. That is important. Negative results matter.

Third: same recipe, different kitchen. One undisclosed change moved the final reward from eighty-six percent down to five percent. Same label, totally different result. That is why I insist on reporting what actually ran.

Fourth: Dr.GRPO's criticism — real math, but no footprint at my scale. I ran six versions, three seeds each. Answer lengths actually shrank by four to twelve percent. No rambling problem here.

## Slide 7 — The honest big test · target 0:50 (4:45 → 5:35)

Now the honest big test. I ran forty runs on Google's free A100 GPUs — eight seeds, five methods: GRPO, DAPO, GSPO, Dr.GRPO, and AERO.

Same model, same questions, same thirty steps, same LoRA. Only the method changed.

Every run finished. Six older records had missing checksums — I repaired all six by replaying the exact checkpoints, before counting them.

Result: scores are nearly tied — around sixty-three to sixty-five percent. And the correct statistics say: we cannot tell the methods apart at eight seeds. All four verdicts: inconclusive.

No method won — and we say it out loud. That is what fail-closed preregistration means: you write the stop rule before spending GPU hours, and then you obey it.

---

## Slide 8 — We caught our own mistake · target 0:50 (5:35 → 6:25)

And now the part I am actually proud of.

Our first analysis said DAPO "disappears" — a strong claim. But when we re-checked the statistics exactly as our preregistration promised, we found two problems.

One: the old math used a shortcut — a large-sample approximation. It overestimates power when you only have eight seeds.

Two: the multiplicity check we promised was never actually executed in the code.

With the exact math, the minimum detectable effect came to zero point zero one zero one two — just above our zero point zero one margin.

So the verdict flipped: all four methods, inconclusive. Same runs, same scores. Only the statistics changed — and with them, the honest answer.

We corrected our own over-claim, before anyone else could.

---

## Slide 9 — The NeurIPS review · target 1:00 (6:25 → 7:25)

Now about the NeurIPS review. This is a story I want to tell you honestly.

The reviewers said: good idea, but the paper over-claims. They listed seventeen weaknesses, five critical.

Examples: "ZVF might just be reward in disguise." "Your G-equals-32 claim contradicts your own G-equals-8 table." "Single-seed runs are powering your headline claims."

So what did we do? The hard part: we audited our own rebuttal claims, before responding.

And we found problems in our own numbers. We withdrew the 92.6 versus 92.1 five-seed claim — three of the five W&B seeds were zero-runtime backfills, with no upstream IDs. We quarantined the PPO row — 0.225 and 0.350 were two different runs. We removed an AUROC built on synthetic anchors.

We replaced all three responses and the AC comment. The lesson: the review killed an over-claimed paper — not the diagnostic. Concede scope, correct the record, preserve the methodology.



## Slide 10 — The rules · target 0:45 (7:25 → 8:10)

So now, every claim in this lab has to survive five checkpoints.

One: report what actually ran — MIN-REPORT-RL, eight items. Each item earned its place by flipping a result in my own data.

Two: preregister before running — the protocol is written and hash-locked before spending GPU money.

Three: fail closed — if a gate cannot pass, we say blocked. Not "almost there."

Four: hash every receipt — checkpoints, manifests, everything. An independent verifier re-downloads and re-checks all of it.

Five: check it, then explain it — no claim goes into a talk or paper until the evidence file exists. That rule caught our own DAPO mistake.

Please understand: these are not bureaucracy. Every rule exists because a real mistake taught it to me.

---

## Slide 11 — Verification · target 0:35 (8:10 → 8:45)

And you can check my work yourself — in under ten minutes.

There is a smoke test with seven checks, one command. Eighty-eight focused tests passing. Seven out of seven integration checks. The headline number re-checked within two percentage points.

Everything is public: GitHub, W&B with one hundred and fifty-three public runs, Hugging Face checkpoints, and the 983-run audit workbook.

A reviewer who cannot run it is not reviewing — they are guessing. So I made running it the easy part.

---

## Slide 12 — What's next · target 0:50 (8:45 → 9:35)

Where am I now? Three gates remain.

Gate one: authorize one small amendment. Right now the flagship gate is fail-closed on a mathematical contradiction we proved — fifty-nine all-wrong and three all-correct groups, out of one hundred. It is a paperwork plus math fix — the only blocker before any GPU spend.

Gate two: run the twenty-four-unit pilot. Four conditions, two regimes, three seeds, on A100. Replay-locked corpora, hash receipts on every step. The dry-run plans already exist — execution is the remaining step.

Gate three: then earn the flagship claim — matched multi-seed evidence, gradient geometry, token-matched group sweep, uncapped mediation. And only then, the ZVF-aware controller bakeoff.

My rule is simple: no claim, until the pilot runs and the receipts verify.


## Slide 13 — Remember · target 0:40 (9:35 → 10:15)

Five things to remember from this talk.

One: a real discovery — ZVF counts the empty drills, verified to machine precision on five hundred and five questions.

Two: a serious benchmark — TinkerRL-Bench, NeurIPS 2026 submitted, nine hundred and eighty-three runs audited.

Three: standards that stick — MIN-REPORT-RL, built from mistakes I actually made.

Four: honest statistics — we corrected our own over-claim, before anyone else could.

Five: fail-closed discipline — no claim until the evidence clears, on every gate.

The result is still to be earned. But everything that earns it — is now checkable.

---

## Slide 14 — Thank you · target 0:30 (10:15 → 10:45)

Thank you very much for your time.

In one sentence: I found a hidden failure in how we train thinking models, built a number that sees it, proved the number is trustworthy, and set up rules so no claim outruns the evidence.

I am very happy to take questions. And I will answer with the evidence — not the hope.

Thank you.
