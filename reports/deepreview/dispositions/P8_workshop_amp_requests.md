# P8_workshop change requests for the Amp session (from the P8/P12 deep-review cluster)

The P8 files are owned by the concurrent Amp session, so this cluster did not edit them. Its exploratory edits were reverted. Below are the verified facts and the exact changes requested. Paths are relative to the repo root. `P` is `platform_hybrid/paper/neurips_2026_variants/`.

Protocol rule 2 applies: data with no artifact behind it is **removed**, not tidied up. Every number below was traced to an artifact; the path is given with it.

---

## 0. The open question: Qwen3-8B training reward 34.4% vs held-out base 82.0%

**Verdict: the gap is most likely explained by generation truncation, compounded by template and sampling differences. It is not explained by a stricter reward format or by the train/test split. Truncation was not measured for the run itself.**

**Training run** (`platform_hybrid/experiments/tinker-runs/results/scale_gsm8k_qwen3-8b.json`, peak 0.625, last10_avg 0.34375, 30 steps, Tinker run 3154dee5):
- Produced by `platform_hybrid/experiments/tinker-runs/scripts/tinker_parallel_runner.py` at git rev `711e87b18`/`02cf6d9ed`.
- Sampling: `SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)`.
- Prompt: raw ChatML with a system line asking for `\boxed{}`. The Qwen3 thinking toggle is never disabled, so the model runs in thinking mode.
- Train split: GSM8K `train`.
- Reward: lenient. It accepts any `\boxed{}` match, otherwise the last number in the full decoded text.
- Batch: 2 prompts per step, so a "peak" is one step's mean over 16 samples.

**Held-out evaluation** (`platform_tinker/reports/final/evaluate_gsm8k_test.py`, outputs `gsm8k_base_control_200.json` = 164/200 and `gsm8k_heldout_seed{042,137,256,512,999}.json`):
- `max_tokens=2048`, greedy (T=0).
- No system prompt.
- Uses the **first 200** GSM8K test items.
- Parser prefers text after `</think>` and falls back to several patterns inside the thinking text, so it is more lenient than the training reward.

**Direct truncation evidence** (same model, prompt family and parser, 512-token cap):
- `zvf-program/experiments-next/results/pool_qwen3-8b_train_n512_r32_s42.json`:
  - mean reward 0.300;
  - 99.0% of the 5,184 length-logged samples hit the 512 cap (mean length 511.4);
  - the 51 samples that finished under the cap all scored 1.0; truncated samples scored 0.287.
- `passk_modal_qwen3-8b_base_gsm8k_p200_n32_s42.json`: GSM8K test pass@1 is 0.306 at 512 tokens.
- MATH-500 pass@1 is 0.0003 at 512 tokens vs 0.598 at 4096 (`passk_modal_qwen3-8b_base_math500_p500_n32_s42.json` and `..._mt4096.json`).
- Contrast: Qwen3-8B-**Base** (no thinking mode) at the same 512 cap trains to last-10 of 0.86–0.99 with mean completion length 263 (`results/heldout_gsm8k.json`).

**Not measured:**
- Run 3154dee5 itself logged no completions or lengths, so its truncation rate is inferred, not measured.
- The separate effects of temperature and system prompt are not isolated.
- The five evaluated checkpoints (899d909e, 5db4e965, aabb48cb, 99971b26, b3ba8df6) come from a **different March campaign** (G=4, 50 steps, training last-10 of 27.5–35%), not from the Table-2 run.

**Requested text.** Insert after the "Held-out negative control" paragraph in `P/paper_P8_workshop.tex`:
```
\paragraph{Why Qwen3-8B training reward is far below its held-out accuracy.}
The Tinker case-study run samples at most 512 tokens at temperature 0.8 with a
\texttt{\textbackslash boxed\{\}} system prompt and Qwen3's default thinking mode;
the held-out evaluation uses 2048 tokens, greedy decoding, no system prompt and a
parser that can read answers inside the thinking text. In a separate 512-token
sampling pool with the same model, prompt and reward (5{,}184 samples), 99.0\% of
completions hit the cap, truncated completions scored 0.287 and the 51 untruncated
ones all scored 1.0. The training-reward gap is therefore consistent with
truncation of thinking-mode output; the reward is not stricter than the evaluation
parser. Completion lengths were not logged for the case-study run itself, so its
truncation rate is inferred rather than measured, and the separate contributions of
temperature and system prompt are unmeasured. The held-out checkpoints come from a
different 50-step, $G{=}4$ training campaign, not from the Table~\ref{tab:headline} row.
```

---

## 1. Remove the unsourced token-budget reanalysis (findings 665, 673; rule 2)

`Table tab:groupsize-tokennorm` in `platform_hybrid/paper/sections/group_size_reconcile.tex` is the hand-entered `FALLBACK_ROWS` dict.
- Source: `platform_hybrid/experiments/group_size_token_normalized.py`, lines 41–62.
- The dict's own header says it was copied *from* the paper table.
- Same precedent as P2/P3, where this table was removed.

Remove the table and every claim built on it:
- `P/paper_P8_workshop.tex`, paragraph "Group size: brief": delete the sentence starting `A 3-seed token-budget reanalysis at $T{\ge}16$M tokens places a fitted apex near $G{\approx}32$ ...` through `... we deliberately avoid promoting a group-size law.`
- `P/paper_P8_workshop.tex`, Table `tab:evidence-grade`: delete the row `Token-budget fitted apex near $G{\approx}32$ at $T{\ge}16$M & descriptive & 3-seed appendix reanalysis; not a universal optimum. \\`.
- `P/sections/conclusion_workshop.tex`, question 2: delete `; a separate 3-seed reanalysis places a fitted apex near $G{\approx}32$ for one stack and budget`.
- `P/sections/intro_workshop.tex`: unchanged.
- `P/ethics_statement_zvf.tex`, M4:
  - old: `Our work argues that critic-free RL with budget-dependent group-size selection (the optimal $G$ depends on total token budget; see Appendix~\ref{sec:appendix:group-size-reconcile}) can be run on modest budgets.`
  - new: `Our release shows that critic-free GRPO can be run on modest budgets.`
- `P/sections/run_manifest.tex`:
  - delete the `C$^\S$ ... qwen3-8b (G=64)$^\S$ ...` row and the `$^\S$~G=64 ...` footnote;
  - in the caption, delete `the additional $\dagger$ on the G=32 group-size row marks it as the single-seed cell that anchors the upper end of the within-sweep token-budget reanalysis in Appendix~\ref{sec:appendix:group-size-reconcile}.`;
  - change `B$^\dagger$ & Tinker & qwen3-8b (G=32)` to `B & Tinker & qwen3-8b (G=32)`.
- `platform_hybrid/paper/sections/group_size_reconcile.tex`:
  - delete the subsections "Formal Definition of Token-Budget Gradient Efficiency" (its `GE_tok` estimator exists only to fill the fallback table; the 505-task paragraph can stay, see §3), "Token-Budget-Normalized Sweep" (table and paragraph "Accuracy-optimal $G$ shifts with budget") and "Reconciliation Statement";
  - retitle the section `Group-Size Evidence`.
- **Shared with P9:** fix-P9 said it forks its own `*_dnb.tex` copies, so editing these originals is safe for P9.

---

## 2. Matched-budget panel is sourced; add it to the manifest (665 part 2)

**Artifacts:**
- `platform_hybrid/experiments/tinker-runs/results/er2b_g{2,16}_s{123,456}.json` (fields `reward_trace`, `step_log[].zvf`).
- W&B `zvf-training` runs `pob7nd05`, `k5m9tq4h`, `tiicy3km`, `woym360v`.

**Setup:**
- Qwen/Qwen3-8B on GSM8K via Tinker.
- LoRA rank 4, lr 3e-5, T=1.0, 8 prompts per step.
- 8×2×160 = 8×16×20 = 2,560 rollouts.
- The 512-token cap is recorded only in the W&B config.

**Measured values:**
- G=2: last-10 reward 0.900 / 0.963; last-10 ZVF ranges 0.875–1.0 and 0.75–1.0.
- G=16: last-10 reward 0.322 / 0.389; final-step ZVF ≤ 0.25. However, seed 456 reaches 0.375 within its last 10 steps.

**Requested changes:**
- `P/paper_P8_workshop.tex`:
  - old: `while $G{=}16$ ends mid-learning ($\approx0.3$--$0.5$) with $\mathrm{ZVF}\le0.25$`
  - new: `while $G{=}16$ ends mid-learning (last-10 reward $0.32$--$0.39$) with final-step $\mathrm{ZVF}\le0.25$ (Qwen3-8B, Tinker, LoRA rank 4, 8 prompts/step)`
- `P/sections/run_manifest.tex`, new Tier-B block row:
  - `B & Tinker & qwen3-8b (matched budget, G=2$\times$160 / G=16$\times$20) & gsm8k & train & 4 & 2 per arm & 160/20 & BV & --- & \checkmark & --- \\`

---

## 3. Group-size measured table / 0.9837 (674)

`tab:groupsize-measured` matches `platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv`, recomputed from the per-seed `groupsize_zvf/G*_s{42,123,456}.json`. The text quotes last-10 reward `0.9837` (G=16), which exists only in the TSV's `last10_mean` column (0.9771 / 0.9714 / 0.9732 / 0.9837) and not in the table. The text also frames the comparison as if it concerned Qwen3-8B.

Requested: in `group_size_reconcile.tex`, add a row `Last-10 train reward & $0.977$ & $0.971$ & $0.973$ & $0.984$ \\` to `tab:groupsize-measured`, and make the "Apparent Contradiction" paragraph say these are Qwen2.5-0.5B two-operand-addition numbers.

The 505-task audit numbers in that file recompute exactly:
- U(4)=0.32914, U(5)=0.32737, difference 0.00177;
- problem-clustered bootstrap [0.00083, 0.00272];
- Y32−Y4 strata +0.29 / +0.15 / +0.45.

Add these disclosures there:
- p_x is k/8 from one G=8 group;
- the cohort keeps only 1≤k≤7 (505 of 600 groups);
- the unbiased estimator gives U(4)=0.37784 ≈ U(5)=0.37789, so the G=4 selection is a plug-in artifact. See `tools/review_ledger/P12_signal_starvation_recompute.py`.

---

## 4. ZVF correlation analyses are unsourced (671, 677, 691, 696, 697; rule 2)

`P/sections/stratified_diagnostics.tex` cites `experiments/analysis/v5/{bootstrap,partial_corr}_table.csv`:
- those files never existed in the tree or in git history (checked with `git log -S`);
- the cohort contract `verification_v3/zvf_correlation.json` is also missing;
- the N=15/19/29/7 pools cannot be rebuilt, because the frontier GSM8K rows have no step-level ZVF traces in the repo.

Figure `figures/v2/workshop_killer.png` has no generator in the tree or history (added in `b0ac85bfb`). Its panel (c) value `r_within=+0.06` matches no table, and its grouping variable is unknown.

The reported p-values are internally consistent with r, N and df. But "+ task FE" costs one df, so it was encoded as a single column; that is valid only if the N=29 pool has exactly 2 tasks, which is unverifiable.

**Requested (rule 2):**
- remove Figure `fig:workshop-killer`, Table `tab:zvf-stat-ablation`, and the `r=-0.769 / -0.443 / -0.026 / +0.401 / +0.13` sentences in Section 2, the abstract (iii), the intro claim hierarchy, and the evidence-grade row;
- remove `\input{sections/stratified_diagnostics}`;
- in `conclusion_workshop.tex` Q1, delete `Our residualized within-GSM8K scatter (Figure~\ref{fig:workshop-killer}c) finds $r_{\mathrm{within}}$ close to zero; is this a backend artifact or a general property?`
- the restoration path is to commit the cohort IDs and the per-run ZVF traces and to regenerate.

---

## 5. Headline table (Table `tab:headline`) and run claims (669, 670, 690)

**Qwen3-235B-A22B row (100% / 100%): remove (rule 2).**
- `frontier_gsm8k_qwen3-235b.json` has `run_id "partial-from-console"`, `checkpoint "training_interrupted"`, and a 4-entry `reward_trace` entered by hand, with `steps_completed` 15/20.
- W&B for Tinker run eb01734d shows cumulative reward 0.958, which contradicts last-10 = 1.0.
- The manifest already lists it as Tier C partial.

**Other Tinker rows MATCH their result JSONs:**
- Qwen3-8B 0.625 / 0.34375;
- Llama-8B-Inst 1.0 / 0.84375;
- DeepSeek-V3.1 1.0 / 0.85 (G=4, 20 steps);
- Nemotron 0.875 / 0.1625;
- tool-use 0 / 0.

Add to the caption: `Each step averages 2 prompts (16 samples at G=8), so peaks are single-step values; all models share one ChatML \boxed{} prompt; tool-use reward is json.loads of the full response, so 0\% reflects output format, not capability.`

**TRL row:**
- Artifact: `platform_hybrid/experiments/results/modal_results_all.json` `trl_grpo_math[].final_accuracy`: 0.735, 0.81, 0.62, 0.74, 0.765; mean 0.734, SD 0.0703.
- Task: synthetic two-operand addition ("What is a + b?"), 500 training problems, not GSM8K.
- Metric: held-out accuracy on 200 fresh problems.
- **The training reward is `1.0 if numbers else 0.0`, so any number scores. This is a reward bug** (`platform_modal/scripts/modal_run_experiments.py::run_grpo_math`).
- Requested: change `unresolved` to `synth.\ addition`, `mean 0.734` to `73.4\%` (690), and add a caption note on the reward bug.
- Also: `compute_statistics.py` `TRL_SEEDS` pads the list with five invented 0.734 entries. Flag it; do not use it.

**Held-out control (670, 682):**
- Tinker backend, first 200 GSM8K test items, greedy, 2048 tokens.
- Per-seed accuracy 0.830 / 0.825 / 0.805 / 0.840 / 0.865; SD 2.20 pp; one-sample t=1.32, df=4, p=0.256 vs base 164/200.
- Requested in `paper_P8_workshop.tex`:
  - old: `Its $p{=}0.26$ is a one-sample seed-level test against that fixed baseline, not a paired-seed test;`
  - new: `Its $p{=}0.26$ is a one-sample $t$-test across five seeds against that fixed baseline ($t{=}1.32$, df${=}4$; seed SD $2.2$\,pp; Tinker backend; first 200 GSM8K test items, greedy, 2048 tokens), not a paired-seed test;`
- `ethics_statement_zvf.tex`:
  - old: `Only the GSM8K track was followed up with a separate 200-example held-out evaluation per seed.`
  - new: `Only the GSM8K track was followed up with a held-out evaluation on the first 200 GSM8K test items per seed (not the full 1{,}319-item test set).`
  - Also change `Our held-out evaluation respects the test/train split.` to `Our held-out evaluation uses the first 200 items of the test split.`

**Scope list (669, 681, 704).** The libraries PufferLib, rl_games and d3rlpy never trained:
- `modal_registry/.../{pufferlib,rl_games,d3rlpy}/seed_*/result.json`: about 2 s config prints; d3rlpy returned code 1.
- veRL/OpenRLHF entries are `"mode": "dryrun"` synthetic traces (`framework_comparison.json`).
- Requested in `abstract_workshop.tex` and the `paper_P8_workshop.tex` Setup:
  - old: `7 RL libraries (TRL, Stable Baselines3, CleanRL, Tianshou, PufferLib, rl\_games, d3rlpy)`
  - new: `4 RL libraries with completed runs (TRL, Stable Baselines3, CleanRL, Tianshou; the latter three run PPO on synthetic math)`
  - old: `5 model families (total parameters 0.6B--671B; active 0.6B--37B ...)`
  - new: `the Qwen2.5, Qwen3, Qwen3.5, Llama-3.x, DeepSeek-V3.1, Nemotron, gpt-oss and Kimi-K2 families (total parameters 0.5B to about 1T)`
- In the acknowledgments, delete `OpenRLHF, veRL,`.

---

## 6. Ethics / compute / carbon (678, 679, 699, 700; rule 2)

There is no billing, cost, GPU-hour or codecarbon artifact in the repo. Repo searches found only prose restatements.

The artifacts contradict the tables in two places:
- L4 per-seed training times are 136–184 s (about 0.2 GPU-h total, vs "~10");
- the Tinker inventory lists 1,054 training runs, vs "25/32/13".

Requested: replace Tables `tab:compute_spend` and `tab:carbon2` and the paragraphs "Failed-run accounting", "Assumed constants", "GPU-hour estimates" and "Interpretation and uncertainty" with:
```
We did not retain billing exports, GPU-hour telemetry, or emissions logs for the
platforms used (Tinker, Modal, Colab, an institutional A100 node), so we do not
report itemised spend, energy, or CO$_2$ estimates. Runs are short LoRA jobs
(10--50 optimizer steps on Tinker; 125 steps for the TRL baseline).
```
Keep the "Hidden costs" paragraph, with its parenthesis fix below.

---

## 7. Mechanical text fixes (exact old -> new)

Generated from this cluster's reverted edit script. Each `old` string occurs exactly once in the current file.

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
We use the public release as-is; specifically, we use the first 35 prompts
$\times$ 10 rollouts in the 10x Structural Ceiling tool-use track and the
full 60k split for the xlam-60k real-data run in Section~\ref{sec:results}.
```
new:
```
We use the public release as-is in the exploratory tool-use track; no
xLAM-60k real-data run is reported in this note.
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
Attribution to Salesforce is required by the
license and is provided in Section~\ref{sec:acknowledgments} and in the
xLAM-derived model cards.
```
new:
```
Attribution to Salesforce is required by the
license and is provided in this paragraph and in the xLAM-derived model cards.
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
We therefore draw \emph{quantitative} conclusions only
from the open-source side (TRL, veRL on Modal H100, OpenRLHF) where every
hyperparameter is auditable,
```
new:
```
We therefore draw \emph{quantitative} conclusions only
from the open-source side (TRL on Modal L4/H100 and the classic-RL PPO stacks
in the run manifest) where every hyperparameter is auditable,
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
All $p$-values are
un-corrected for the number of comparisons actually made across the
paper; the Bonferroni-surviving comparisons in
Section~\ref{sec:results} are so flagged, but most of our secondary
analyses are descriptive.
```
new:
```
All $p$-values are
uncorrected for the number of comparisons actually made across the
paper, and most of our secondary analyses are descriptive.
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
infrastructure generously subsidised by an academic research group with
shared access to one A100 node used for TRL baselines and
pre-submission sanity checks).
```
new:
```
infrastructure generously subsidised by an academic research group (with
shared access to one A100 node used for TRL baselines and
pre-submission sanity checks).
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
8{,}500 grade-school math word problems (7{,}473 train / 1{,}319 test)
```
new:
```
8{,}792 (``8.5K'') grade-school math word problems (7{,}473 train / 1{,}319 test)
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
  \item Every figure and every summary statistic derived solely from
        Tinker data is marked with ``\dag'' and a footnote stating that
        independent replication requires Tinker API access.
```
new:
```
  \item Summary statistics in the headline tables derived from Tinker
        data are marked with ``\dag'' or grouped under an explicit Tinker
        subheader; independent replication requires Tinker API access.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
each group $g$ containing $K$ rollouts with scalar outcome rewards
$r_{g,1}, \ldots, r_{g,K} \in \mathbb{R}$.
```
new:
```
each group $g$ containing $G$ rollouts (the group size used throughout the
paper) with scalar outcome rewards $r_{g,1}, \ldots, r_{g,G} \in \mathbb{R}$.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
\mathbb{1}\!\left[\;\widehat{\mathrm{Var}}(r_{g,1}, \ldots, r_{g,K}) = 0\;\right],
```
new:
```
\mathbb{1}\!\left[\;\widehat{\mathrm{Var}}(r_{g,1}, \ldots, r_{g,G}) = 0\;\right],
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
where $\widehat{\mathrm{Var}}$ is the unbiased sample variance and $K\ge2$.
```
new:
```
where $\widehat{\mathrm{Var}}$ is the unbiased sample variance and $G\ge2$.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
A separate near-constant telemetry statistic uses variance $\le10^{-6}$;
```
new:
```
A separate near-constant telemetry statistic, used only in post-hoc logging
and never as a training-time gradient mask, uses variance $\le10^{-6}$;
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
$1/K$; hence the tolerance agrees with exact collapse only when $1/K>10^{-6}$,
```
new:
```
$1/G$; hence the tolerance agrees with exact collapse only when $1/G>10^{-6}$,
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
For $K$ i.i.d.~Bernoulli rollouts with prompt-specific success probability
$p_g$, the expected ZVF under a null independence model is
\begin{equation}
\mathbb{E}[\mathrm{ZVF}_t] \;=\; \mathbb{E}_{p_g}\!\left[\,p_g^{K} + (1-p_g)^{K}\,\right].
\end{equation}
```
new:
```
For $G$ i.i.d.~Bernoulli rollouts with prompt-specific success probability
$p_g$, the expected ZVF under a null independence model is
\begin{equation}
\mathbb{E}[\mathrm{ZVF}_t] \;=\; \mathbb{E}_{g\sim\mathcal{D}}\!\left[\,p_g^{G} + (1-p_g)^{G}\,\right],
\end{equation}
where $\mathcal{D}$ is the prompt distribution.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
while a uniform-hard population with all $p_g = 0.5$ and $K=8$ yields
```
new:
```
while a uniform-hard population with all $p_g = 0.5$ and $G=8$ yields
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
  \item \textbf{$K=1$.} The unbiased sample variance and this statistic are undefined; exclude such groups.
```
new:
```
  \item \textbf{$G=1$.} The unbiased sample variance and this statistic are undefined; exclude such groups.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`
old:
```
outcome-reward GRPO on verifiable tasks with
$K \ge 4$ and
```
new:
```
outcome-reward GRPO on verifiable tasks with
$G \ge 4$ and
```

### `platform_hybrid/paper/neurips_2026_variants/sections/stratified_diagnostics.tex`
old:
```
outcome. Adding task FE alone collapses the pooled $r$ from $-0.44$ to
$-0.03$; with any combination of $\log G$ and base accuracy the partial $r$
is between $+0.02$ and $+0.14$ and never significant.
```
new:
```
outcome. Adding task FE alone collapses the pooled $r$ from $-0.443$ to
$-0.026$; in the pooled sample with task FE and any combination of $\log G$ and
base accuracy the partial $r$ is between $+0.021$ and $+0.136$ and never
significant (within GSM8K, the partial $r$ with $\log G$ and base accuracy is
$+0.420$, $p{=}0.094$).
```

### `platform_hybrid/paper/neurips_2026_variants/sections/stratified_diagnostics.tex`
old:
```
(\texttt{pingouin.partial\_corr}). Last-10 reward is the outcome and is never
on the right-hand side.
```
new:
```
(\texttt{pingouin.partial\_corr}). The $p$-values here are parametric
two-sided $t$-test values, unlike the permutation $p_{\text{perm}}$ of
Table~\ref{tab:zvf-bootstrap}. Last-10 reward is the outcome and is never
on the right-hand side.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/abstract_workshop.tex`
old:
```
The rest of this artifact note is a candid evidence-grade table and three
discussion prompts for the workshop.
```
new:
```
The rest of this artifact note is a candid evidence-grade table and five
discussion prompts for the workshop.
```

### `platform_hybrid/paper/neurips_2026_variants/sections/abstract_workshop.tex`
old:
```
\paragraph{August 2026 evidence boundary.} This workshop artifact preserves
```
new:
```
\paragraph{Evidence boundary.} This workshop artifact preserves
```

### `platform_hybrid/paper/neurips_2026_variants/sections/run_manifest.tex`
old:
```
Section~\ref{sec:zvf-analysis} of the ZVF variant); the additional
```
new:
```
Section~\ref{sec:zvf-analysis}); the additional
```

### `platform_hybrid/paper/neurips_2026_variants/sections/run_manifest.tex`
old:
```
the single-seed Tinker frontier and group-size evidence underlying
Table~4 of the DNB variant; the Tier-A
```
new:
```
the single-seed Tinker frontier and group-size evidence; the Tier-A
```

### `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`
old:
```
We evaluate four domains: GSM8K, MATH-500 (exploratory), HumanEval
(subset), and synthetic/xLAM tool-use.
```
new:
```
We report GSM8K, synthetic arithmetic, and synthetic tool-use (an attempted
HumanEval run is listed as a failed Tier-C cell).
```

Also requested, without exact text:
- remove the NuminaMath and Open-Platypus provenance paragraphs from `ethics_statement_zvf.tex`, since no runs use them (704);
- Section 5 artifact tier (iii) and the intro artifact card: the manifest does not enumerate a raw-JSONL subset or the hardware/hyperparameter mapping. Either add a `Raw JSONL` column or reword to "step-level aggregate traces" only (675).

Bibliography items (684, 705, 706) are in `reports/deepreview/dispositions/P8_P12_shared_requests.md`. For the related work (688, 689), cite the existing shared-bib keys `yu2025dapo`, `tong2025drgrpo`, `nie2026gradstarvation`, `zhang2026greso`, `ahmadian2024backtobasics`, `wu2025grpodpo` and `mahrooghi2026goldilocks` in one sentence after the ZVF definition, e.g.:
```
Flat-group starvation and its fixes are established: DAPO filters flat groups by
dynamic sampling~\citep{yu2025dapo}, GRESO skips predicted zero-variance prompts
before rollout~\citep{zhang2026greso}, Dr.~GRPO removes standard-deviation and
length normalization~\citep{tong2025drgrpo}, and the binary-reward identity is
formalized by \citet{nie2026gradstarvation}; on group size and critic-free
baselines see \citet{ahmadian2024backtobasics,wu2025grpodpo,mahrooghi2026goldilocks}.
ZVF is offered only as a logging diagnostic in that setting.
```
