#!/usr/bin/env python3
"""Session-1-scoped Phase 1 First Review deck (first guide session: problem,
RQs, inherited baseline, PROPOSED method & plan — no final results)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BLUE = RGBColor(0x1F, 0x4E, 0x79); INK = RGBColor(0x21, 0x21, 0x21)
MUTED = RGBColor(0x5A, 0x5A, 0x5A); WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def para(tf, text, size=16, color=INK, bold=False, bullet=False, align=PP_ALIGN.LEFT, first=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    r = p.add_run(); r.text = ('•  ' if bullet else '') + text
    f = r.font; f.size = Pt(size); f.color.rgb = color; f.bold = bold; f.name = 'Calibri'
    p.alignment = align; p.space_after = Pt(8)
    return p

def slide(title=None, page=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.8))
        para(tb.text_frame, title, size=28, color=BLUE, bold=True, first=True)
        ln = s.shapes.add_textbox(Inches(0.6), Inches(1.1), Inches(12.1), Inches(0.05))
    if page:
        pb = s.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.6), Inches(0.35))
        para(pb.text_frame, str(page), size=11, color=MUTED, first=True)
    return s

def bullets(s, items, top=1.4, size=16):
    tb = s.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.7), Inches(5.4))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (t, kw) in enumerate(items):
        para(tf, t, size=size, bullet=True, bold=kw, first=(i == 0))
    return s

# 1 Title
s = slide()
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.6))
tf = tb.text_frame; tf.word_wrap = True
para(tf, 'M.Tech Project — Phase 1 First Review (Session 1)', size=32, color=BLUE, bold=True, align=PP_ALIGN.CENTER, first=True)
para(tf, 'RL Post-Training of LLMs: Signal Starvation and Stack-Conditioned GRPO', size=20, color=INK, align=PP_ALIGN.CENTER)
para(tf, 'Arvind C R (Arvind Chitra Rajasekaran)  ·  SRN: PES2PGE24DS140', size=16, color=MUTED, align=PP_ALIGN.CENTER)
para(tf, 'Guide: Ramesh Prakash Guledgudd  ·  Dept. of CSE, PES University', size=16, color=MUTED, align=PP_ALIGN.CENTER)
para(tf, 'M.Tech Data Science & AI  ·  UE20CS971', size=14, color=MUTED, align=PP_ALIGN.CENTER)

# 2 Context
bullets(slide('Context: From Group Capstone to Solo Continuation', 2), [
    ('Semester 3 (Group 6): multi-framework RL post-training benchmark, literature survey, initial GRPO/SFT/DPO experiments, NeurIPS main-track submission.', False),
    ('Frozen boundary: tag capstone-final-2026-04-25 — group deliverables archived.', False),
    ('Semester 4 (this phase): individual continuation under guide Ramesh Prakash Guledgudd.', True),
    ('This session: agree problem scope, research questions, methodology, and the semester work plan.', False),
])

# 3 Problem
bullets(slide('Problem Statement & Motivation', 3), [
    ('Published RL-for-LLM results are stack-conditioned: the same algorithm yields different outcomes across frameworks, backends, and defaults.', True),
    ('Group-relative methods (GRPO) suffer signal starvation: when all rollouts in a group agree, the advantage is zero and no learning signal flows.', True),
    ('Capstone evidence: heterogeneous results across 7 libraries; zero-variance groups frequent on sparse binary rewards.', False),
    ('Without controlled de-confounding, algorithm comparisons and "scaling" claims are unreliable.', False),
])

# 4 RQs
bullets(slide('Research Questions', 4), [
    ('RQ1 — Same-stack control: does PPO vs GRPO differ when every other pipeline element is held fixed?', True),
    ('RQ2 — Can a Zero-Variance Fraction (ZVF) statistic serve as a practical signal-starvation diagnostic?', True),
    ('RQ3 — How does group size G trade off contrast density, ZVF, and trainability?', True),
    ('RQ4 — Do training-reward gains survive held-out evaluation (length bias, generalization)?', True),
])

# 5 Literature anchors
bullets(slide('Literature Anchors', 5), [
    ('RLHF → PPO fine-tuning (Ouyang et al., 2022): value-based post-training baseline.', False),
    ('DPO (Rafailov et al., 2023): preference optimization without rollouts.', False),
    ('GRPO / DeepSeek-R1 (Shao et al., 2024; DeepSeek, 2025): group-relative advantages, verifiable rewards.', False),
    ('Dr.GRPO and length-bias corrections (Liu et al., 2025).', False),
    ('Gap: no controlled cross-stack study isolating estimator choice from infrastructure.', True),
])

# 6 Inherited system
bullets(slide('Inherited Platform (Capstone Baseline)', 6), [
    ('tinker-rl-lab: one repo integrating Tinker API, SkyRL, verl, OpenRLHF, TRL, Atropos environments.', False),
    ('Baseline result to build on: GRPO on Qwen3-8B GSM8K — peak 62.5%, last-10 34.4% (training reward).', False),
    ('Existing scaffolding: evaluation pipeline, W&B tracking, reproducibility audits.', False),
    ('Solo phase reuses this infrastructure; new work is experiments, diagnostics, and analysis.', True),
])

# 7 Proposed methodology
bullets(slide('Proposed Methodology: Four De-Confound Pillars', 7), [
    ('Pillar 1 — Same-stack PPO vs GRPO: only the advantage estimator differs.', True),
    ('Pillar 2 — Measure ZVF directly; theoretical form vs observed, across tasks.', True),
    ('Pillar 3 — Group-size sweep G ∈ {2,4,8,16} × seeds with held-out checks.', True),
    ('Pillar 4 — Held-out generalization: GRPO vs Dr.GRPO on GSM8K-CoT, pre/post McNemar.', True),
    ('All runs seeded, logged, and audited; statistics with CIs, not single-seed claims.', False),
])

# 8 Work plan
bullets(slide('Work Plan & Deliverables (Semester 4)', 8), [
    ('Phase A (weeks 1–4): pillar experiments + ZVF telemetry implementation.', False),
    ('Phase B (weeks 5–8): analysis papers P1–P4 (scaling, ZVF, group size, length bias).', False),
    ('Phase C (weeks 9–12): standards & tooling papers P5–P7 (reporting standard, registry, adaptive controller).', False),
    ('Phase D: applied case study P8 (LLM vs XGBoost on fraud) + workshop paper submission.', False),
    ('Reviews: Session 1 (this) → Session 2 (mid results) → Session 3 (complete results + demo).', True),
])

# 9 Expected outcomes
bullets(slide('Expected Outcomes & Evaluation Criteria', 9), [
    ('A defensible answer to PPO-vs-GRPO under matched conditions (effect size + CI).', False),
    ('ZVF validated (or refuted) as a cheap online diagnostic, with a group-size guidance rule.', False),
    ('Eight short papers (P1–P8) + one workshop submission; reproducible artifact.', False),
    ('Success = claims that survive held-out evaluation and independent audit, not peak training reward.', True),
])

# 10 Close
s = slide()
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.7), Inches(11.3), Inches(2.0))
tf = tb.text_frame
para(tf, 'Thank you — Discussion & Guide Inputs', size=30, color=BLUE, bold=True, align=PP_ALIGN.CENTER, first=True)
para(tf, 'github.com/arvindcr4/tinker-rl-lab  ·  arvindcr4@gmail.com', size=15, color=MUTED, align=PP_ALIGN.CENTER)

prs.save('outputs/PESU_MTech_Phase1_Session1_Review_ArvindCR.pptx')
print('slides:', len(prs.slides._sldIdLst))
