# Talking Script — ZVF Research Status for Ramesh
**Date:** 4 July 2026 · **Slides:** `reports/ZVF_status_Ramesh_2026-07-04.pptx` · **Length:** ~8 minutes

---

## Slide 1 — Title

"Good morning, Ramesh. Quick background in two sentences. We train language models with a method called GRPO: the model tries the same question several times, each answer gets a score, and the model moves toward what scored well. Our work centers on one simple number — the zero-variance fraction, ZVF — which tells us when that learning has quietly stopped. Today: what it is, what we've shown, what's running, and what I need from you."

## Slide 2 — Executive Summary

"Here's the whole idea. GRPO learns by comparing answers. If all the answers to a question get the same score, there's nothing to compare — the model learns nothing from that question. ZVF just counts how often that happens.

Four numbers for this slide. Five hundred plus: we've completed more than five hundred training runs on Tinker — the inventory actually lists 844 run IDs — and the papers use seventy-plus carefully chosen ones. Zero point nine three: our ZVF warning score, when shown one failing run and one healthy run, puts the failing one on top 93 percent of the time. Eight: the number of papers, and all of them build to PDF today. Three: the experiments running right now, adding more evidence while we talk.

The plan is to aim all of this at ICLR 2027."

## Slide 3 — The ZVF Story

"The story has three steps: measure, predict, act.

Measure. At every training step we ask a simple question: for how many of the training questions did every answer get exactly the same score? That fraction is ZVF. It costs nothing extra to compute — the scores are already there. And it solved a real mystery for us: one of our tasks scored zero percent the entire time. ZVF showed why — it was stuck at one hundred percent. Every answer scored the same, so the model never had a signal to learn from. Nothing was broken; the training was starving.

Predict. We built a warning score out of ZVF. It flags a collapsing run before the score curve visibly breaks — like a smoke alarm that goes off before you see flames. Given a failing run and a healthy one, it picks out the failing one 93 percent of the time. We tried making it fancier by adding answer-length information — no improvement. The simple alarm already works.

Act. The newest paper turns the alarm into a controller. When ZVF climbs, it automatically changes how many answers the model samples per question, which brings back the contrast GRPO needs. So we went from noticing failure, to predicting it, to preventing it."

## Slide 4 — Paper Portfolio

"Eight papers, organized as four pillars plus one side study.

Pillar one is the foundation — four papers. Paper one: bigger models don't automatically learn more from this training; what matters is whether the model could nearly do the task before training. Paper two: the ZVF warning light itself. Paper three: how many answers per question is best? There's no single best number — and this also tests a claim that GRPO is secretly the same as another method called DPO. Paper four: this training does not teach models to ramble on short math problems — our checks agreed eight out of eight times.

Pillar two: the same algorithm name behaves differently in different software libraries, so papers should report the whole stack they used, not just the name.

Pillar three: a catalog, readable by machines, of what each library actually implements and how the variants differ.

Pillar four: the ZVF controller I just described — from diagnostic to automatic fix.

And the side study: on credit-card fraud, the language model is most useful for reading and explaining, while a classic method, XGBoost, keeps the scoring job.

ZVF runs through the middle of this: it's paper two in the foundation, and it drives the controller in pillar four."

## Slide 5 — Running Right Now

"Three experiments are live as we speak — the numbers on the slide are read straight from the run logs when the deck is generated.

N2 keeps the full gradebook: every score for every answer, across four related training methods, instead of just class averages. That means we compute ZVF exactly — no estimating.

N10 re-runs the length experiment from eight random starting points instead of three. If eight different starts give the same answer, it's very hard to call it luck.

And the mega sweep is filling in a coverage map — 506 combinations of model, task, and settings — showing where a learning signal exists at all.

Everything saves its progress continuously, so if a run stops, nothing is lost."

## Slide 6 — From Runs to Papers

"Why these three experiments? Each one fixes the exact thing a reviewer would poke at.

The full gradebook feeds the warning-light paper — exact numbers instead of estimates — and gives the controller paper the data to tune when to act.

The eight seeds turn 'we saw it once' into 'we see it every time, with error bars.' That was the biggest weakness in the length paper.

And the coverage map gives the scaling and group-size papers breadth, while every cell writes a standard report card — real worked examples for the report-the-stack paper."

## Slide 7 — Keeping Ourselves Honest

"A few habits keep us honest. Every headline claim gets double-checked against the actual data files on disk. A citation check flagged two references that need fixing — that's tracked and will be done before submission. All eight papers build with zero errors, though three still show reference warnings we're cleaning up. And anyone can re-run everything: fixed random seeds, logged dashboards, checksums, and a step-by-step guide.

We also say our weak spots out loud. Most headline results came from a single seed — N10 is fixing exactly that, right now. The training runs are short on purpose, so we present them as early snapshots, not final outcomes. One health metric was missing for part of the campaign — we disclose it. And the zero-percent task became a finding instead of an embarrassment, thanks to ZVF.

Reviewers consistently reward this kind of honesty."

## Slide 8 — Next Steps & Decisions

"This week: let the three experiments finish, fold their results into the papers, double-check any numbers we touch, and lock the ICLR 2027 plan. Going by last cycle, the deadlines should land around mid-September — roughly ten weeks away.

Over the next month: error bars everywhere, connecting our results to the wider literature, strengthening the tool-use evaluation, and finishing the reference cleanup.

Two questions for you.

One — for ICLR 2027, one big unified ZVF paper, or the ZVF pair leading with companion papers?

Two — anything you'd want addressed before we commit to the ICLR timeline?

Thank you — happy to take questions."

---

**Venue reality check (as of 4 July 2026):**
- **Target: ICLR 2027 — deadlines expected ~mid-Sept 2026 (2026 cycle: abstract 19 Sept, paper 24 Sept); CFP not yet posted.**
- NeurIPS 2026 E&D — already past (abstract 4 May, full paper 6 May 2026).
- Fallbacks if ICLR timing slips: ICML 2027 D&B (~Jan 2027), NeurIPS 2027 E&D (~May 2027), ACL/EMNLP/NAACL 2027 via ACL Rolling Review.
