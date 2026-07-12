# M.Tech Phase-1 Defense — Read-Aloud Transcript (ELI12)
**Signal Starvation and Stack-Conditioned GRPO · Arvind C R**
Pacing target: ~18 min spoken + ~2 min live demo = 20-min hard cap. Speak ~135 words/min. Times are cumulative.

---

## Slide 1 — Title  ·  target 1:00  (0:00 → 1:00)

Good morning, and thank you. My project is about a hidden failure in how we train large language models with reinforcement learning — and about why the very same experiment can give completely different answers depending on where you run it.

Let me decode the title in one breath. The method is called GRPO. It teaches a model by making it write several answers to the same question, then comparing those answers against each other. The problem: when all the answers agree — all right, or all wrong — there is nothing to compare, so the model learns nothing, even though the scoreboard looks perfect. I call that **signal starvation**. I built a simple number that measures it, the **Zero-Variance Fraction**, or **ZVF**. And **stack-conditioned** is my second finding: the exact same method gives different results on different software — so you cannot trust a comparison unless you know exactly what ran. One measuring discipline, two claims, every number traceable to an artifact.

---

## Slide 2 — Base Paper & My Understanding  ·  target 2:00  (1:00 → 3:00)

Let me start with the paper my work is built on. GRPO comes from DeepSeekMath — it is the algorithm behind DeepSeek-R1. Here is the idea in plain terms.

Older methods used a second AI, a "critic," to judge how good each answer was. That is expensive. GRPO throws the critic away. Instead, for each question it samples a group of answers — say eight — grades each one right or wrong, and scores each answer by how it did compared to the group average. Better than its groupmates, positive push; worse, negative push. Cheap, stable, and it is now the default way to train these models.

Now here is my contribution, and it is a picture. Think of each question as a shooting drill in basketball, and the eight answers as eight shots at the same hoop. GRPO's coach has no absolute standard — he can only compare your shots to each other in that drill. Some go in and some miss, he learns something. But if all eight go in — or all eight miss — there is nothing to compare, and the drill taught zero. And here is the sting: as the player gets better, more and more drills end in all-makes. So practice silently stops teaching you exactly when the scoreboard looks its best. On a math run, the reward reads a perfect one-point-zero while learning has quietly stopped.

That is the blind spot. The reward curve — the one number everyone watches — literally cannot show it. My thesis is: measure that blind spot, prove the measurement is trustworthy, budget for it, and show what changes once you can see it. One more anchor: a 2025 paper called Dr.GRPO criticizes GRPO from another angle — I test that critique directly later, in Result 4.

---

## Slide 3 — Problem & Research Questions  ·  target 0:45  (3:00 → 3:45)

So two facts drive everything. First: starvation is invisible — the reward number says "success" at the exact moment training dies, so you need a second number beside it. Second: the same training, run by different people on different setups, can swing wildly — in my own data, a seventeen-times difference in final reward from one undisclosed change. Unless you report what actually ran, comparisons are meaningless.

That gives four questions. One: can I isolate a single knob by holding the software fixed? Two: is ZVF a useful, practical diagnostic? Three: is group size the dial that controls starvation? And four: do the training gains survive on held-out tests?

---

## Slide 4 — Overall Architecture  ·  target 1:15  (3:45 → 5:00)

Here is the machine that produces every number, in four layers. Training runs on a managed API called Tinker — the loss code is sealed, which is actually an audit constraint I use on purpose. Telemetry: next to every reward point I log ZVF and a second signal, so I am always watching the pair, not just reward. Evaluation runs on three independent backends, so no single tool's quirks own my results. And underneath it all, an audit trail: every run recorded, checkpointed, and traceable in a workbook.

The design rule is one sentence: no number enters a paper without a path back through this stack — from the run, to the telemetry, to the saved artifact, to an audit row.

---

## Slide 5 — Not Only GRPO  ·  target 0:45  (5:00 → 5:45)

I did not only run GRPO. Each other method answers one specific question. PPO keeps the critic — so it tells me whether dropping the critic was actually free. GSPO changes exactly one knob on the same setup — a clean control. Dr.GRPO tests the base paper's strongest criticism. DPO trains with no groups at all — so by construction it has no starvation, a useful counterfactual. REINFORCE and plain fine-tuning are floors. And one method, "DAPO," I ran under one label on two different stacks and got completely different behavior — which becomes evidence for my reproducibility claim. Every row has runnable artifacts behind it.

---

## Slide 6 — What I Implemented (Sem-4 Solo)  ·  target 1:30  (5:45 → 7:15)

Quick attribution, because it matters. Semester 3 was group work — a multi-framework benchmark of seventy-nine runs across seven libraries, now submitted to NeurIPS 2026's main track. Everything I show from here is my solo Semester-4 work, built on that foundation.

What did I train on? For the RL work, the GSM8K grade-school math set, with a simple right-or-wrong reward on the boxed answer, plus some synthetic arithmetic for the small models. Crucially, my test sets — the harder MATH-500, and the coding benchmarks — I never trained on. Held-out means held-out.

What did I build? The measurement stack — the per-step ZVF telemetry, the confidence intervals, the budgeting math. The experiment infrastructure — a runner that can be killed and resume exactly, down to the random state. A packaged, installable library called zvf-triage with eighty-two passing tests. The reporting standard and its enforcement tools. And the theory — including two errors that outside reviewers caught, which I corrected and reported openly rather than hiding.

---

## Slide 7 — Result 1: ZVF Sees What the Reward Curve Cannot  ·  target 2:00  (7:15 → 9:15)

Now the core result. I took two setups on the exact same compute budget. One writes just two answers per question; the other writes sixteen. I let them run.

Look at what the reward curve says: the small-group runs finish at basically a perfect score — one-point-zero. By that number alone, they win. But now look at ZVF, my second coordinate: in those same runs, three-quarters to all of the groups are all-correct — every one of them contributing zero learning. The run is burning compute and teaching nothing. Meanwhile, the sixteen-answer runs are still at a modest score, still mid-learning, ZVF low, signal fully intact.

So read as a pair, reward-and-ZVF cleanly separates two things the reward curve smashes together: "the policy is good" versus "training is still moving." The small group looked like the winner, but its lead ended in dead compute.

Two honest caveats up front. First, ZVF is a diagnostic, not a predictor — in every collapse I measured, ZVF rose after the reward plateau, so it is a cheap alarm, not a crystal ball. Second, you must always read the pair: ZVF alone cannot tell "the model has mastered this" from "the model cannot do it at all" — both look identical, because both are ties. Reward breaks the tie.

---

## Slide 8 — Result 2: Group Size Is a Schedule Variable  ·  target 2:00  (9:15 → 11:15)

That raises the obvious question: should we use small groups or big groups? And my answer is: that is the wrong question.

The trick in the experiment is that I hold the total budget fixed, not the number of steps. Under that fair comparison, group size does not pick a winner — it picks which end of training starves.

Small groups spend the budget on lots of fast feedback early, so they learn quickly. But remember the basketball drill: as the model gets good, a group of just two answers almost always comes out both-correct — a tie — so the small group runs out of signal at the end, right as the reward hits its peak. Big groups do the opposite: early on they over-pay for contrast they do not need, but even late in training, sixteen answers rarely all agree, so every update still carries signal.

So group size is not a constant you tune once — it is a schedule. Start small while answers still disagree; grow it as they stop. The anticipated attack here is: "the big group only avoids the wall because it has not learned the task yet." And my answer is: exactly — that is the claim. At a fixed budget, group size chooses which end starves. I declare no winner; the pair just tells you where your budget went.

---

## Slide 9 — Result 3: The Estimator Is Calibrated, the Budget Is Exact  ·  target 1:30  (11:15 → 12:45)

Result 2 is only trustworthy if the ruler is trustworthy, so I proved three things about it.

One: ZVF is a properly calibrated estimate — its confidence intervals actually cover the truth ninety-five to ninety-eight percent of the time, in every setting I tested.

Two: I can turn it into a budget. There is a clean formula for "how many answers do I need to roll before I get one informative, non-tied group?" On my hardest batch of problems, that is a hundred and sixty rollouts for a ninety-percent guarantee. That is a planning number, not a wall.

Three — and this is the honest one — I looked for a formula that would hand me the single best group size, and I proved it does not exist: the per-rollout math always favors the smallest groups, for every difficulty. That is a negative result, and outside reviewers found it in my own theory. I report it as a result, not a footnote — and it is exactly why Result 2 treats group size as a schedule instead of a magic constant.

---

## Slide 10 — Result 4: GRPO vs Dr.GRPO, and the Incident  ·  target 2:00  (12:45 → 14:45)

Result 4 tests that outside critique, Dr.GRPO. Its claim is that GRPO's math secretly rewards rambling — because a wrong answer's penalty gets divided by its length, a long wrong answer is punished less per word than a short one, like splitting a parking fine among more passengers. So the model supposedly learns to pad its answers. Dr.GRPO deletes that division.

I ran six versions side by side, three seeds each, with generous length limits. The result: answer lengths shrank — by four to twelve percent — in every single version. No rambling, no verbosity trap at my scale, and no late difference in ZVF between GRPO and Dr.GRPO. So the critique is real math, but it left no footprint here.

And now the part I am proud of, not embarrassed by. My first version of this panel was invalid — a documented setting that was supposed to switch the loss function was never actually wired up, so all six arms were secretly running the same thing. No output would ever have shown me that; I only caught it by reading the code line by line. So I did the right thing loudly: I flagged it, preserved the broken files under an "invalid" name, reran the same day — and one of my conclusions actually reversed. That failure became a protocol, and the protocol feeds my reproducibility standard.

---

## Slide 11 — Result 5: Setting the Standard  ·  target 1:30  (14:45 → 16:15)

Which brings me to the reproducibility finding — "stack-conditioned." The idea is simple: same recipe, different kitchen, different dish. Same GRPO label, different software, different result. And I have three measured examples from my own data, each of which flipped a result.

One: an undisclosed backend swap — which also quietly changed a starting checkpoint — moved final reward across a seventeen-times span, from eighty-six percent down to five, under the same label. Two: the same "DAPO" label gave near-zero ZVF on one trainer and fifty-five-plus percent on another. Three: reward noise too tiny for a human to notice collapsed a batch's ZVF from sixteen percent to zero — so you even have to report which grader you used.

Out of these I built MIN-REPORT-RL: an eight-item checklist of what every RL training report must state. And the point I want to land is that every one of those eight items earned its place by flipping a result in my own data — not from taste. Plus two tools to enforce it mechanically.

---

## Slide 12 — Result 6: Held-Out Evaluation  ·  target 1:15  (16:15 → 17:30)

Result 6 asks the honest question: after all this training, is the model actually smarter, or just more confident about what it already knew?

The base model, given thirty-two tries, already solves ninety-one percent of the grade-school math. So there is only about nine points of real headroom — which means this benchmark simply cannot demonstrate big new ability, and recognizing that is itself a finding. After training, I see no forgetting and a small improvement on coding transfer, but it is within noise at one seed — which is exactly why my standard demands proper pass@k curves with error bars, not a single number. And on genuinely hard math, the gains do not carry at all. So what I am seeing is sharpening — the model learning to lead with answers it could already find — not brand-new capability. I state that boundary myself, before anyone has to ask.

---

## Slide 13 — Implementation Scale & Evidence Trail  ·  target 0:45  (17:30 → 18:15)

Briefly, the scale behind this. I enumerated and audited nine hundred and eighty-three training runs on the Tinker account this morning; the thesis rests on nineteen claim-critical ones, each linked to its dashboard, its checkpoint, and its result file. Plus external runs on three other platforms, a thousand-plus logged experiments, forty-nine published model repos, and the packaged library with all eighty-two tests green. If you ask me about any number, there is a workbook row that takes you to it.

---

## Slide 14 — Demo (Live)  ·  target 1:00 + live  (18:15 → 19:15+)

Let me show it working. The safe version runs fully offline with one command — it recomputes ZVF on a small fixture, verifies a recorded run against its checksum, and opens a dashboard. I verified it passing this morning. If connectivity allows, I will then show the live dashboard where you can watch reward and ZVF diverge on the real runs, and the audit workbook where every key run is two clicks away. And if the network fails, there is an eighty-six-second recording embedded right here that plays offline.

*(Run ./submission/demo/demo.sh — narrate: fixture ZVF = 0.500, recorded artifact ZVF = 0.30, checksum PASS. Then W&B panel if time allows.)*

---

## Slide 15 — Scope, Limitations & Roadmap  ·  target 0:45  (→ ~20:00)

Finally, the honest boundaries. One model, one family of tasks, one managed API, one-to-three seeds — every claim is stated at that level and nowhere above it. I claim no predictive power for ZVF beyond diagnosis; my controller's value is explicitly gated on a future pre-registered, compute-matched test before I will claim it works. The roadmap is staged accordingly: the diagnostic paper is publishable now, the controller paper waits for that win, the survival audit waits for an open stack.

And the one transferable lesson: at this scale, the thing that limits you is not compute, and it is not novelty — it is certainty about what actually ran.

Thank you. I am happy to take questions, and I can open the audit workbook or any run on request.

---

### Trim knobs if running long
- Slide 3: cut the second RQ sentence, keep only the four questions.
- Slide 5: drop DPO/REINFORCE examples, keep PPO + DAPO only.
- Slide 13: compress to one sentence ("983 runs audited, 19 claim-critical, all traceable").
- Slide 14: skip the live W&B tour, rely on the offline demo + embedded recording.
