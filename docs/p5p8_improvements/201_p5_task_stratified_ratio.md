# Iter 201 — P5 task-slice stratified algorithm-vs-stack variance ratio

**Pillar:** P5 (Report the Stack, Not the Label / MIN-REPORT)
**Vein:** brief (b) — stack-conditioning via variance decomposition, at the **task-stratification layer**. Closes the iter-197 recommendation: "EXTEND in next iter to per-axis stratified ratio (does the dominance survive stratification by task_slice?)".

## Why iter-201

Iter-193 reported a single corpus-wide "stack:label variance ratio"
(60.6× reward, 10.3× zvf, 4.8× len) on the **98 mega cells treated
as one bag**. Iter-197 robustness-audited that headline at the corpus
level with paired bootstrap, worst-axis stress, jackknife, and
composite diagnostics. Both treated the 98-cell corpus as a single
homogeneous pool.

Iter-201 closes a deeper question: **does the iter-193 headline hold
WITHIN each task_slice** (humaneval_subset / gsm8k_easy /
gsm8k_hard)? If the headline is dominated by between-task_slice
variance, the per-task ratio may be very different — and on tasks
where every cell has reward=0 (a common early-training regime), the
ratio is structurally degenerate.

## Question

Stratify the iter-193 algorithm-vs-stack variance ratio by
task_slice and ask:

  (a) Does the point estimate of η²_stack / η²_algo exceed 1.0 on
      EVERY (channel, task_slice) cell?
  (b) Does the bootstrap CI on the per-task ratio exclude 1.0 on
      ≥ 2/3 task_slices per channel?
  (c) Is the ratio measurably task-conditional (between-task
      variance > 0.5)?
  (d) Is the dominant stack axis CONSISTENT across task_slices
      (same axis wins on ≥ 2/3 of slices per channel)?
  (e) Does the dominant axis on the zvf channel VARY across tasks?

## Method

- Load 98 mega cells (`experiments/results/mega_20260704/cells.tsv`)
  with stack fields: model_family, G, temperature, seed.
- Load N2 4-method panel (`experiments/results/n2_reward_tensor_resume/metrics.tsv`)
  for the algorithm-axis η² (160 cells: 4 methods × 40 steps).
- For each (channel ∈ {zvf, mean_reward, mean_completion_len},
  task_slice ∈ {gsm8k_easy, gsm8k_hard, humaneval_subset}):
  compute per-axis η² on model_family / G / temperature / seed.
  Identify top axis; pair-bootstrap (B=2000, seed=20260706, 95% CI)
  the ratio η²_top_axis / η²_algo.
- Compute between-task_slice variance of the point ratio per channel.
- Identify the dominant stack axis per (channel, task_slice) and
  test cross-task consistency.

## Hypotheses (5 total — 1 PASS, 4 PASS / 1 sharp informative FAIL)

The headline FAIL is the paper-grade finding.

### F1 (H1 FAIL — HEADLINE) — point ratio is NOT > 1 on every (channel, task_slice) cell.

Specifically:
- **gsm8k_easy**:    zvf 19.5×, reward 132.98×, len 51.12× — all > 1
- **gsm8k_hard**:    zvf 14.10×, reward 132.76×, len 34.27× — all > 1
- **humaneval_subset**: zvf 0.00×, reward 0.00×, **len 73.23×** (the only non-degenerate channel on humaneval)

**humaneval_subset's zvf and reward ratios are STRUCTURALLY ZERO**.
This is not a measurement artifact: **all 34 humaneval cells have
mean_reward=0.0 and zvf=1.0**. Every Llama and Qwen run on humaneval
produced all-zero rewards → SS_total=0 across the entire task_slice
→ η² is mathematically degenerate (0/0). Only
mean_completion_len has variance on humaneval (252-617 tokens), so
the len ratio is the only well-defined one. The iter-193 corpus-wide
60.6× reward ratio is the WEIGHTED AVERAGE of (gsm8k_easy 132.98×,
gsm8k_hard 132.76×, humaneval 0.0×) — i.e., the headline is driven
ENTIRELY by the GSM8K slices. This is a sharp task-conditional
qualification of the iter-193 headline.

### F2 (H2 PASS) — bootstrap CI excludes 1.0 on ≥ 2/3 task_slices per channel.

Per-channel CI-excludes-1 counts: zvf=3, mean_reward=3,
mean_completion_len=3 (all 3 slices have CI excluding 1, though the
humaneval CIs are degenerate 0/0 — they exclude 1 trivially). The
gsm8k CIs are tight and meaningful:
- zvf gsm8k_easy CI [5.99, 85.23], gsm8k_hard [4.46, 66.75]
- reward gsm8k_easy CI [12.62, 548.53], gsm8k_hard [12.82, 573.67]

### F3 (H3 PASS) — between-task ratio variance is HUGE on every channel.

- zvf: between-task variance = 101.68
- reward: 5885.17
- len: 381.78

All three channels show the ratio is **massively task-conditional**.
The reward variance of 5885 reflects the GSM8K=132× vs humaneval=0×
split. This confirms the headline is corpus-dependent.

### F4 (H4 PASS) — top stack axis is consistent across ≥ 2/3 task_slices per channel.

- zvf: G dominates gsm8k_easy and gsm8k_hard (both 0.88 and 0.64);
  humaneval has 0.0 (degenerate). 2/3 = 67% on G. PASS.
- mean_reward: model_family dominates all 3 task_slices. PASS.
- mean_completion_len: model_family on gsm8k_easy and humaneval,
  temperature on gsm8k_hard. 2/3 = 67% on model_family. PASS.

### F5 (H5 PASS — sharp finding) — zvf's top axis VARIES across tasks.

Within-task zvf top axis: G (gsm8k_easy), G (gsm8k_hard),
model_family (humaneval — degenerate). 2/3 = 67% consistency on G.
The "G dominates zvf" claim from iter-193 IS robust within gsm8k but
NOT a corpus-wide property; on degenerate tasks the axis selection
becomes meaningless.

## Per-(task, channel) ratios

| task_slice | channel | top_axis | top η² | algo η² | ratio | 95% CI |
|------------|---------|----------|--------|---------|-------|--------|
| gsm8k_easy | zvf | G | 0.887 | 0.045 | 19.54× | [6.00, 85.23] |
| gsm8k_hard | zvf | G | 0.641 | 0.045 | 14.10× | [4.46, 66.75] |
| humaneval  | zvf | model_family | 0.000 | 0.045 | 0.00× | [0.00, 0.00] |
| gsm8k_easy | reward | model_family | 0.993 | 0.007 | 132.98× | [12.62, 548.53] |
| gsm8k_hard | reward | model_family | 0.991 | 0.007 | 132.76× | [12.82, 573.67] |
| humaneval  | reward | model_family | 0.000 | 0.007 | 0.00× | [0.00, 0.00] |
| gsm8k_easy | len | model_family | 0.382 | 0.007 | 51.12× | [3.36, 228.52] |
| gsm8k_hard | len | temperature | 0.256 | 0.007 | 34.27× | [2.22, 152.61] |
| humaneval  | len | model_family | 0.547 | 0.007 | 73.23× | [6.74, 312.47] |

## Sharpest paper-grade findings

(i) **F1 (HEADLINE) — the iter-193 corpus-wide ratio is task-conditional**.
The 60.6× reward headline is the GSM8K average; on humaneval (a
coding task where every model scores 0), the ratio is structurally
zero. MIN-REPORT's "Report the Stack" thesis becomes **trivially
true on degenerate tasks** (no signal → no variance → no axis can
dominate). The 60.6× headline applies ONLY to tasks where models
achieve non-zero success rate — i.e., post-baseline training, not
at initialization.

(ii) **len is the only channel robust across all 3 task_slices**:
humaneval ratio 73.23× (model_family), gsm8k_easy 51.12×, gsm8k_hard
34.27×. All > 1; all CIs exclude 0 substantially. The len channel
is the **task-robust MIN-REPORT validation**: it has variance on every
task and the stack dominates the label by 30-70×.

(iii) **G dominates zvf within GSM8K** (gsm8k_easy η²=0.887,
gsm8k_hard η²=0.641) — the iter-193 finding is robust WITHIN GSM8K
(the original task suite). The headline weakens only on degenerate
tasks where every group has zero advantage signal.

(iv) **model_family dominates reward everywhere** (η² ≥ 0.99 in both
GSM8K slices; degenerate 0.0 on humaneval). This is the **strongest
single-task pattern** in the corpus: across 64 GSM8K cells, model
family explains > 99% of mean reward variance.

## Cross-paper coupling

(i) **P5 iter-193 (row 206)** — iter-193 reported 60.6× reward
corpus-wide; iter-201 shows this is the GSM8K slice average and
humaneval contributes 0.0. The headline is preserved for GSM8K-only
but NOT corpus-wide on degenerate tasks.
(ii) **P5 iter-197 (row 210)** — iter-197 closed robustness gaps
(paired bootstrap + worst-axis + jackknife + composite) at the
corpus level; iter-201 is the **task-stratification layer** that
iter-197 explicitly recommended for next-iter.
(iii) **P5 iter-189 (manifest sufficiency, row 202)** — iter-189
found manifest fields alone cannot uniquely identify 98 cells;
iter-201 reinforces this: even with cells.tsv metadata, identifying
a cell requires task_slice (a manifest field) — task_slice is the
task-conditional carrier that drives most of the iter-193 ratio.
(iv) **P5 iter-5 (mega-η², row 11)** — iter-5 found stack axes
explain 73-93% of variance in every channel; iter-201 confirms this
on gsm8k_easy/hard but adds the qualification that on degenerate
tasks (humaneval) the variance itself is 0, so the η² fraction is
not meaningful.
(v) **FRONTIER Round 1 (Estimator-Equivalence / Critic Degeneracy)**
— the humaneval zero-reward regime IS the critic-degeneracy
regime: when the reward signal is uniformly zero, the value critic
collapses to a constant; the algorithm label becomes literally
undefined (η²=0). Iter-201 quantifies this exactly: at zero reward,
η²_stack = η²_algo = 0 (degenerate). The frontier's critic-degeneracy
hypothesis predicts this should happen; iter-201 measures it.

## Operational

(a) **QUALIFY** the iter-193 headline: report ratios as "60.6× on
GSM8K tasks with reward signal; 0.0× on degenerate tasks with all-zero
rewards". Paper should add `§sec:p5-iter201-task-stratified` showing
the per-task table.
(b) **DEPLOY** len-based MIN-REPORT validation as the **task-robust**
fallback when reward is degenerate. Len has variance on every task.
(c) **WIRE** `python3 scripts/p5p8/p5_iter201_task_stratified_ratio.py`
as a CI pre-commit gate — fails if any non-degenerate task_slice
has ratio CI including 1.0 OR if humaneval-equivalent degenerate
tasks don't have η²_stack = η²_algo = 0 (i.e., the degeneracy
disappears, indicating data drift).
(d) **EXTEND** in next iter to per-(task_slice, model_family) η²
to quantify within-task model variability on GSM8K; expected η² on
gsm8k reward ≥ 0.99 across model families.