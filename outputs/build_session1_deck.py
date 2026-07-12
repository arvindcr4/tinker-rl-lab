#!/usr/bin/env python3
"""Phase 1 First Review deck — DEFENSE FORMAT (examiner rubric):
title -> base paper & understanding -> architecture -> what I implemented
-> results achieved -> demo. 14 slides, ~20 minutes, timed speaker notes."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BLUE = RGBColor(0x1F, 0x4E, 0x79); INK = RGBColor(0x21, 0x21, 0x21)
MUTED = RGBColor(0x5A, 0x5A, 0x5A); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0x8A, 0x6D, 0x00)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def para(tf, text, size=16, color=INK, bold=False, bullet=False, align=PP_ALIGN.LEFT, first=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    r = p.add_run(); r.text = ('•  ' if bullet else '') + text
    f = r.font; f.size = Pt(size); f.color.rgb = color; f.bold = bold; f.name = 'Calibri'
    p.alignment = align; p.space_after = Pt(7)
    return p

def slide(title=None, page=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.8))
        para(tb.text_frame, title, size=27, color=BLUE, bold=True, first=True)
    if page:
        pb = s.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.6), Inches(0.35))
        para(pb.text_frame, str(page), size=11, color=MUTED, first=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s

def bullets(s, items, top=1.25, size=15):
    tb = s.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.8), Inches(5.7))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (t, kw) in enumerate(items):
        para(tf, t, size=size, bullet=True, bold=kw, first=(i == 0))
    return s

def table(s, rows, top=1.35, left=0.9, width=11.5, col_widths=None, size=13):
    n_r, n_c = len(rows), len(rows[0])
    shp = s.shapes.add_table(n_r, n_c, Inches(left), Inches(top), Inches(width), Inches(0.36*n_r))
    t = shp.table
    if col_widths:
        for i, w in enumerate(col_widths): t.columns[i].width = Inches(w)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci); cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.font.name = 'Calibri'
                    r.font.bold = (ri == 0); r.font.color.rgb = INK if ri else WHITE
    return t

# ---------------------------------------------------------------- 1 Title
s = slide(notes="(1 min) Read the title, then decode it in one breath: GRPO trains on groups; "
                "when every completion in a group gets the same reward the group carries zero gradient — "
                "that is signal starvation; ZVF is the statistic that measures it; 'stack-conditioned' is "
                "the reproducibility finding that the same algorithm label gives different results on "
                "different software stacks. One measurement discipline, two claims, all artifacts audited.")
tb = s.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(4.2))
tf = tb.text_frame; tf.word_wrap = True
para(tf, 'M.Tech Project — Phase 1 Review (Defense Format)', size=30, color=BLUE, bold=True, align=PP_ALIGN.CENTER, first=True)
para(tf, 'RL Post-Training of LLMs: Signal Starvation and Stack-Conditioned GRPO', size=21, color=INK, bold=True, align=PP_ALIGN.CENTER)
para(tf, 'Title decoded: GRPO learns from reward contrast inside groups of completions; when a group is all-correct or all-wrong it teaches nothing (signal starvation, measured by the Zero-Variance Fraction), and what a run "shows" depends on the software stack it ran on (stack-conditioned).', size=14, color=MUTED, align=PP_ALIGN.CENTER)
para(tf, 'Arvind C R (Arvind Chitra Rajasekaran)  ·  SRN: PES2PGE24DS140', size=15, color=MUTED, align=PP_ALIGN.CENTER)
para(tf, 'Guide: Ramesh Prakash Guledgudd  ·  Dept. of CSE, PES University  ·  M.Tech Data Science & AI', size=14, color=MUTED, align=PP_ALIGN.CENTER)

# ------------------------------------------------- 2 Base paper & understanding
bullets(slide('Base Paper & My Understanding of It', 2, notes=
  "(2 min) Base paper: GRPO from DeepSeekMath (Shao et al., 2024), the algorithm behind DeepSeek-R1. "
  "Explain the mechanism from first principles: no critic; sample G completions per prompt; each advantage "
  "is its reward minus the group mean. My understanding goes one step further than the paper: that subtraction "
  "has a structural blind spot — identical rewards zero out every advantage. Secondary anchor: Dr.GRPO "
  "(Liu et al., 2025) which critiques GRPO's normalisation terms — my Result 4 tests it head-on."), [
    ('Base paper — GRPO, from "DeepSeekMath" (Shao et al., 2024; basis of DeepSeek-R1): replaces PPO\'s learned critic with a group-relative baseline — sample G completions per prompt, advantage = own reward − group mean.', True),
    ('Why it matters: critic-free means cheap and stable at LLM scale with verifiable (binary) rewards; it is now the default RL post-training family (DAPO, GSPO, Dr.GRPO, GRESO...).', False),
    ('My understanding — the structural blind spot: if all G completions earn the SAME reward (all-correct or all-wrong), every centred advantage is exactly zero: the group consumes compute but contributes zero gradient. The reward curve cannot show this; it can read "success" precisely while learning has stopped.', True),
    ('Secondary anchor — Dr.GRPO (Liu et al., 2025): claims GRPO\'s per-length/std normalisation biases updates toward verbosity. I test this claim under controlled conditions (Result 4).', False),
    ('Thesis position: measure the blind spot (Zero-Variance Fraction, ZVF), calibrate it, budget it, and show what it changes in practice.', True),
])

# ---------------------------------------------------------- 3 Problem & RQs
bullets(slide('Problem & Research Questions', 3, notes=
  "(1 min) Compress. Two facts motivate everything: starvation is invisible in reward curves, and "
  "published comparisons are stack-conditioned. Then read the four RQs quickly."), [
    ('Signal starvation is silent: mean reward reads "success" exactly when the all-correct wall starves training. You need a second coordinate.', True),
    ('Published results are stack-conditioned: same label, different stacks, different outcomes (measured: a 17× final-reward span from an undisclosed backend+checkpoint swap).', True),
    ('RQ1 same-stack control · RQ2 ZVF as practical diagnostic · RQ3 group size G as the starvation dial · RQ4 do training gains survive held-out evaluation?', False),
])

# ---------------------------------------------------------- 4 Architecture
bullets(slide('Overall Architecture', 4, notes=
  "(1.5 min) Walk the four layers left to right: training on the managed Tinker API (LoRA, closed loss "
  "kernel — an audit constraint I exploit deliberately); evaluation on three vLLM backends so no single "
  "backend's quirks own the numbers; telemetry: per-step ZVF/GU next to reward, mirrored to W&B; and the "
  "audit layer: run manifests, checkpoint/resume, the runs-audit workbook. Everything downstream cites this."), [
    ('Training layer — Tinker managed API: LoRA rank-4 GRPO/Dr.GRPO fleets with full state checkpointing and kill-and-resume (built after two mid-programme credit exhaustions; verified live).', True),
    ('Evaluation layer — vLLM pass@k harness on three independent backends: Modal, Lightning AI, Colab; problem-clustered bootstrap CIs; seeded per problem.', False),
    ('Telemetry layer — per-step (reward, ZVF, GU) traces for every training run, mirrored to W&B (zvf-training project); reward-parser v2 with false-positive audit.', True),
    ('Library — zvf-triage (Apache-2.0, 82 tests): drop-in triage callback — classifies starvation regime, adapts G, drops dead prompts, auto-stops doomed runs; veRL / OpenRLHF / NeMo-RL adapters.', False),
    ('Audit layer — 983 Tinker runs enumerated and classified; 19 claim-critical runs identified, each linked to its W&B page, checkpoint, and result JSON (workbook in outputs/).', False),
])

# ------------------------------------------- 5 What I implemented (attribution)
bullets(slide('What I Implemented (Sem-4 Solo, on the Sem-3 Foundation)', 5, notes=
  "(1.5 min) Attribution first: Sem 3 was the group capstone — the multi-framework bench and survey, frozen at "
  "tag capstone-final-2026-04-25. Everything on this slide is Sem-4 solo work. Be specific: these are files "
  "and packages he can open, not concepts."), [
    ('Inherited (Sem 3, Group 6): multi-framework benchmark scaffold (TinkerRL-Bench), literature survey, baseline GRPO runs. Frozen at tag capstone-final-2026-04-25.', False),
    ('ZVF measurement stack: per-step ZVF/GU telemetry in the trainer, calibrated confidence intervals (Wilson), waiting-time reliability budget, stratified batch analysis.', True),
    ('Experiment infrastructure: matched-budget runner with --resume (state + optimiser + RNG fast-forward), per-(step,prompt) seeding, W&B resume; loss-form panel runner (GRPO vs Dr.GRPO).', True),
    ('zvf-triage: packaged library (callback, controller, regime classifier, framework adapters, 82-test suite) — publication to PyPI staged.', True),
    ('Standards & tooling: MIN-REPORT-RL 8-item reporting standard, GRPO-Registry (machine-readable stack catalog), stackdiff flip-risk grader, run-audit workbook.', False),
    ('Theory: T1 estimator calibration, T2 reliability budget, T3 optimal-G analysis — plus two corrections found by external adversarial review, adopted and reported openly.', False),
])

# ------------------------------------------------------- 6 Result 1: Claim 1
s = slide('Result 1 — ZVF Sees What the Reward Curve Cannot (Claim 1)', 6, notes=
  "(2 min) THE core result. Walk the table: late in training the G=2 arms read reward ~1.0 — by the reward "
  "axis, perfect. ZVF says 75-100% of groups are all-correct: zero gradient. Same budget, G=16 arms are "
  "mid-learning with ZVF under 0.25 and signal intact. Read as a pair, (reward, ZVF) separates 'policy is good' "
  "from 'training is still moving'. ZVF alone aliases mastery with incapacity — always read the pair.")
table(s, [
    ['', 'late-run mean reward', 'late-run ZVF', 'gradient signal'],
    ['G=2 × 160 steps', '≈ 0.9–1.0 (pool mastered)', '0.75–1.0 (all-correct wall)', 'effectively zero'],
    ['G=16 × 20 steps', '≈ 0.3–0.5 (mid-learning)', '0.00–0.25', 'sustained'],
], col_widths=[2.6, 3.4, 3.2, 2.3])
bullets(s, [
    ('Same rollout budget (2,560/arm), seeds 123/456: reward alone declares G=2 the winner; the (reward, ZVF) pair shows its lead ended in zero-gradient compute.', True),
    ('ZVF is a diagnostic, not a predictor: in every collapse we measured, ZVF rose AFTER the reward plateau — a cheap alarm, not a cause. Pooled cross-task correlations do not survive stratification and are never used as claims.', False),
    ('Population form is the U-shaped kernel h_G(p)=p^G+(1−p)^G: starvation at both walls (too hard / mastered); larger G narrows both.', False),
], top=3.15)

# ------------------------------------------------------- 7 Result 2: Claim 2
bullets(slide('Result 2 — Group Size Is a Schedule Variable (Claim 2)', 7, notes=
  "(2 min) The decisive experiment design point: hold the ROLLOUT BUDGET fixed, not the step count. "
  "Small G converts the budget into more optimiser steps early — then exhausts its own signal as accuracy "
  "rises (the p->1 wall of the kernel). Large G pays for contrast it doesn't need early and retains signal late. "
  "So group size controls WHICH END of training starves — a schedule question, not a constant. The naive "
  "static sweep is confounded by what the budget is held in — I show both views."), [
    ('Design: matched budget of 2,560 rollouts per arm — G=2×160 steps vs G=16×20 steps (batch 8, 512-token completions, LoRA rank 4, seeds 123/456).', True),
    ('Finding: G=2 races to reward ≈1.0 on the sampled pool, then terminates inside the all-correct zero-variance wall; G=16 ends mid-learning with ZVF ≤ 0.25 and signal intact.', True),
    ('Interpretation: small G buys early optimiser steps and starves the endgame; large G holds signal throughout. Group size selects which end of training starves — it is a schedule variable.', True),
    ('Honesty check: static fixed-step sweeps are non-monotone and confounded by what the budget is held in; the controller that would exploit this is designed but its efficacy is NOT claimed (pre-registered test is future work).', False),
])

# ------------------------------------------------- 8 Result 3: theory calibrated
bullets(slide('Result 3 — The Estimator Is Calibrated, the Budget Is Exact', 8, notes=
  "(2 min) Three theory results, each validated on real 512-prompt pools. T1: Wilson interval covers 0.95-0.98 "
  "in every tested setting — report which ZVF the interval covers under curriculum ordering. T2: geometric "
  "waiting-time budget N = G ln(delta)/ln(ZVF) matched observed quantiles at ratio 1.00 in all six difficulty "
  "strata — hardest stratum needs 160 rollouts for a 90%-guaranteed informative group. T3 is the honest one: "
  "our signal-per-rollout objective turns out to have a UNIVERSAL argmax G* in {2,3} for every prior — an "
  "algebraic identity, found by external review of our own theory. We report it as a negative result."), [
    ('T1 (calibration): ZVF_t is an unbiased binomial-proportion estimator; Wilson CI covers 0.95–0.98 in every tested setting (Wald marginal at m=32). Curriculum ordering is an estimand-labelling requirement, not an invalidation.', True),
    ('T2 (reliability budget): rollouts-to-next-informative-group is geometric; N(ZVF)=G⌈ln δ/ln ZVF⌉ matched observed quantiles at ratio 1.00 across all six difficulty strata (hardest: ZVF=0.886 ⇒ 160 rollouts). A budget, NOT an impossibility bound.', True),
    ('T3 (honest negative): the signal-per-rollout objective satisfies J(2)=J(3) for EVERY difficulty prior — its argmax is universally {2,3}, so it cannot yield a data-adaptive G. Found by adversarial review; reported as a result, not buried.', True),
    ('Two earlier statement errors (a quantifier confusion in T2; a GU sign slip) were caught by external review, corrected, re-validated — and are documented in the thesis as part of the method.', False),
])

# --------------------------------------- 9 Result 4: loss panel + the incident
bullets(slide('Result 4 — GRPO vs Dr.GRPO: No Footprint at This Scale (+ the Incident)', 9, notes=
  "(2 min) Tests the base-paper critique directly. Six uncapped arms (1,024 tokens), 3 seeds per loss: "
  "completion lengths SHRINK 6-12% in all six arms — no verbosity trap at this scale — and no late-ZVF "
  "separation. Then own the incident proudly: the first panel was invalid because a documented --loss flag "
  "was never wired to the loss; no output-level trace revealed it — only reading the runner did. We "
  "invalidated loudly, preserved artifacts under .invalid names, reran same-day, and one conclusion REVERSED. "
  "That incident is now a case study and the seed of the reporting standard.", ), [
    ('Six-arm uncapped panel (Qwen3-8B, 1,024-token cap, 3 seeds/loss): GRPO lengths 1004→905, 981→944, 996→900; Dr.GRPO 999→931, 972→902, 1000→878 — lengths SHRINK 6–12% in every arm; no length inflation, no late-ZVF separation between losses.', True),
    ('Reading: at this scale the loss-form choice has no observable footprint on length or ZVF — evidence about reporting, not superiority. Comparisons between these losses measure stack noise unless controlled far more tightly than the labels suggest.', False),
    ('The incident: the first panel ran with a documented --loss drgrpo flag that was never wired in — both "arms" silently trained identical GRPO. Caught only by reading the runner; artifacts preserved under .invalid_actually_grpo names; corrected rerun REVERSED one conclusion.', True),
    ('Response became protocol: invalidate loudly → preserve → rerun → record. This failure is a first-class result feeding the reproducibility standard (Result 5).', False),
])

# ------------------------------------------- 10 Result 5: reproducibility results
bullets(slide('Result 5 — Reproducibility: Measured Flips, and the Standard They Justify', 10, notes=
  "(1.5 min) Three measured instances, each a lever that flipped a result: 17x reward span from an undisclosed "
  "backend swap that also bundled a checkpoint change; the same 'DAPO' label yielding ZVF 0.00 on an open "
  "trainer vs 0.55-0.58 on a closed stack; and reward micro-jitter below the verifier's resolution collapsing "
  "batch ZVF 0.158 to 0. Every item of the 8-item MIN-REPORT-RL standard exists because one of these levers "
  "moved a result in OUR OWN data — not from taste."), [
    ('Backend swap (undisclosed, bundled a base-checkpoint change): final training reward moved across a 17× span — 85.6% vs 5.0% — under the same label.', True),
    ('Same "DAPO" label: mean ZVF 0.00 on an open trainer with true dynamic sampling vs 0.55–0.58 on a closed stack running an asymmetric-clip surrogate.', True),
    ('Reward-parser sensitivity: micro-jitter ε~U(0,1e-4) below verifier resolution collapses batch ZVF 0.158 → 0.000 — reported ZVF must name its verifier.', False),
    ('Deliverables: MIN-REPORT-RL (8-item minimum reportable stack — every item justified by a measured flip), GRPO-Registry (machine-readable catalog, 20 seed entries), stackdiff (pairwise flip-risk verdicts R0–R5).', True),
])

# ------------------------------------------- 11 Result 6: held-out evaluation
bullets(slide('Result 6 — Held-Out Evaluation: Gains, Transfer, and an Honest Boundary', 11, notes=
  "(1.5 min) RQ4. Base Qwen3-8B on GSM8K: pass@1 30.4% but pass@32 91% — the base model already solves "
  "almost everything at k=32, so GSM8K alone cannot demonstrate capability expansion; that scope discipline "
  "is itself a finding. Post-RL adapters: zero forgetting on MBPP and a +1.5-2.5 point pass@32 frontier "
  "improvement, within noise at single-seed — exactly why the standard mandates pass@k curves with CIs. "
  "MATH-500 partial: GSM8K-trained gains do NOT replicate on hard math — distribution sharpening, "
  "not capability expansion."), [
    ('Baseline capability (200 problems, n=32, clustered bootstrap): pass@1 30.4% [27.5, 33.1] but pass@32 91.0% — GSM8K is nearly saturated at k=32; ~9 points of headroom bounds what training can claim here.', True),
    ('Transfer: post-RL adapters show zero forgetting and mild positive transfer on MBPP (pass@32 within noise of or above base for all G); pass@1-only reporting would misread the G=2 arm as a regression.', False),
    ('Hard-task boundary (MATH-500, partial): GSM8K-trained frontier gains do not carry — consistent with distribution sharpening rather than capability expansion. Stated as a non-claim in the thesis.', True),
    ('Cross-scale observations (70+ curated runs, 7 libraries, 0.6B–671B): scale does not uniformly reduce ZVF; starvation is (difficulty × G × phase) geometry, not something scale buys you out of.', False),
])

# ---------------------------------------------------- 12 Scale & evidence trail
bullets(slide('Implementation Scale & Evidence Trail (audited 12 Jul)', 12, notes=
  "(1 min) Fast slide. 983 runs on the Tinker account — audited and classified this morning; the thesis "
  "claims rest on 19 identified claim-critical runs, each linked to W&B and its artifact. If asked about "
  "any number: the workbook key_runs sheet has the run id, checkpoint, W&B link, and result JSON."), [
    ('983 Tinker training runs enumerated via REST API — 26 base models, 0.6B → 1T; all 65 corrupted runs predate June 8; every thesis-supporting run is clean and checkpointed.', True),
    ('External backends: Modal / Lightning AI / Colab pass@k panels (26 runs) + 4 cross-library H100 baselines; 1,034 W&B runs across 17 projects; 49 HuggingFace artifact repos.', False),
    ('zvf-triage: 82/82 tests green; wheel + sdist built, twine-checked; PyPI publication staged.', False),
    ('Traceability: 19 claim-critical runs highlighted in the audit workbook, each with W&B link + checkpoint + result artifact (outputs/tinker_runs_audit_2026-07-12.xlsx).', True),
])

# ------------------------------------------------------------------ 13 Demo
bullets(slide('Demo (Live)', 13, notes=
  "(1.5 min + live demo) Run the one-command offline demo FIRST — it cannot fail on the network. "
  "./submission/demo/demo.sh: mechanism fixture (4 groups, ZVF=0.5, GU=0.5), recorded artifact (80 rewards, "
  "mean 0.6875, ZVF 0.30), SHA-256 integrity check, HTML dashboard. Then if time and connectivity allow: "
  "the W&B zvf-training panel with live E-R2b curves, and the audit workbook key_runs sheet. "
  "Fallback: the 86-second recorded walkthrough. Verified PASS this morning."), [
    ('One command, fully offline: ./submission/demo/demo.sh — mechanism fixture (4 groups → ZVF=0.500, GU=0.500), recorded artifact check (80 rewards, mean 0.6875, ZVF 0.3000), SHA-256 integrity, JSON + HTML dashboard. Status: PASS (re-verified today).', True),
    ('Live artifact tour: W&B zvf-training — the (reward, ZVF) pair diverging on the real E-R2b arms; audit workbook key_runs sheet — every claim-critical run traceable in two clicks.', False),
    ('zvf-triage quickstart: pip-installable package, examples/quickstart.py — the diagnostic as a reusable library, not a one-off script.', False),
    ('Fallback if connectivity fails: 86-second recorded walkthrough (thesis/viva/demo_walkthrough.mp4).', False),
])

# ------------------------------------------------- 14 Limitations & close
bullets(slide('Scope, Limitations & Roadmap', 14, notes=
  "(1 min) Close with the honesty that survives cross-examination: one model, one task family, one managed "
  "API, 1-3 seeds — the claims are stated at the stack level and nowhere above it. Roadmap is gated: "
  "diagnostic paper publishable now; controller paper gated on a pre-registered compute-matched win; "
  "survival audit gated on an open stack. Then stop talking and invite questions."), [
    ('Declared scope: Qwen3-8B, GSM8K-family binary rewards, Tinker managed API (LoRA rank 4), 1–3 seeds per result — claims are stated at the stack level and nowhere above it.', True),
    ('No causal or predictive power is claimed for ZVF beyond diagnosis; controller efficacy is explicitly gated on a pre-registered, compute-matched comparison (≥3 seeds, held-out metrics).', False),
    ('Roadmap (gated): bounded diagnostic paper (Claims 1–2) → controller paper (needs the win) → survival audit (needs an open stack). Thesis consolidates all 17 working documents.', False),
    ('The transferable lesson: at this scale the binding constraint is not compute or novelty — it is certainty about what actually ran.', True),
])

s = slide(notes="Thank the panel. Repo and email on screen. Offer the audit workbook or any W&B run on request.")
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.7), Inches(11.3), Inches(2.0))
tf = tb.text_frame
para(tf, 'Thank you — Questions', size=30, color=BLUE, bold=True, align=PP_ALIGN.CENTER, first=True)
para(tf, 'github.com/arvindcr4/tinker-rl-lab  ·  arvindcr4@gmail.com', size=15, color=MUTED, align=PP_ALIGN.CENTER)

prs.save('outputs/PESU_MTech_Phase1_Session1_Review_ArvindCR.pptx')
print('slides:', len(prs.slides._sldIdLst))
