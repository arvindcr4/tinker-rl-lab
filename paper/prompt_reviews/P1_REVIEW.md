# P1 (Scaling) — Synthesized Review

**Inputs:** `P1_stress_test.md`, `P1_ablation_gap.md`, `P1_claim_lint.md`, `P1_decisive_experiment.md` (all in this directory, all dated 2026-07-04).
**Target:** `paper/paper_P1_scaling.tex` and its `\input` sections.

---

## 1. Verdict on the central claim

The central claim — GRPO reward gain shows no cross-scale slope over "five orders of magnitude," capability class (instruct-vs-base) is the dominant axis, and the saturation model is structurally misspecified — is **not robust as stated**. The flat-slope null is baked in by construction: 4/5 anchors start at or near reward ceiling at step 1, the repo's own power calibration (`scaling_law_iter121_synthetic_recovery.tsv`) shows ≤26% detection power against realistic slopes, and no experiment anywhere measures the *gain* (Δ held-out) at more than one scale point. The capability-class clause rests on an apparently inverted label — the "base" Qwen3-8B anchor's trace records the Instruct checkpoint's exact model ID and score — and is contradicted by the paper's own iter129 Bayes factor (log BF = −9.53 against capability adding predictive power). On top of this, 8 of 58 quantitative claims are contradicted by the repo's data files (including "five orders of magnitude," which is actually ≈2.4 decades, and the entire Δ₁T paragraph). The paper is survivable only if rescoped to "no slope measurable in the saturated GSM8K regime at T ≤ 30, on verified labels," with the numeric errata corrected and a headroom-controlled positive control added or explicitly flagged as future work.

---

## 2. Top 3 prioritized actions

### Action 1 — RUN EXPERIMENT: harness-vs-capability audit of the "incapable" cluster
- **What:** The three-stage zero-training-compute audit specified in §3 below (label audit → transcript regeneration → cluster recomputation). The Stage-A pre-check already indicates the hard-kill trigger fires: the R̄ = 0.2854 anchor labeled `dense-base` is `Qwen/Qwen3-8B` — the **Instruct** `model_id` in `base_instruct_paired.tsv` (Base = `Qwen/Qwen3-8B-Base`, scoring 0.8250) — and the Nemotron anchor is an A12B MoE mislabeled `dense`. If confirmed, the bimodality/capability-class clause is void as stated and must be re-derived from verified labels.
- **Why first:** Near-zero cost, and its outcome gates what Actions 2–3 should say — there is no point polishing prose around a clustering built on inverted labels. It attacks the only affirmative causal clause in the paper (the other two are nulls).
- **Effort:** ~0.5 day total (Stage A ~30 min scripting; Stage B one sampling-only Tinker job, ≤600 completions on one ≤8B model; Stage C ~1 h re-analysis). Zero training steps.
- **Touches:** new artifacts under `experiments/results/` (iter-style meta JSON + verification table); then, depending on branch: `paper/sections/scaling_laws.tex` (bimodality paragraph `par:scaling-iter125`, iter129 capability table), `paper/sections/p1_abstract.tex` (claims A5/A6), `paper/sections/p1_conclusion.tex` (C3/C4). Regardless of branch, two label errata (Qwen3-8B class; Nemotron family = MoE) must be corrected in `scaling_laws.tex` and `scaling_law_iter129_capability_scaling.tsv`.

### Action 2 — FIX PROSE: correct the 8 unsupported claims and 5 highest-risk mislabels
- **What:** Apply the claim-lint rewrite list (semantics-preserving; conclusions survive, numbers don't). Minimum set:
  1. "five orders of magnitude" → "≈2.4 orders (4B–1T)" — `p1_abstract.tex` (A2b), `frontier_synthesis_scaling.tex`, `p1_conclusion.tex`.
  2. Δ₁T paragraph, `scaling_laws.tex` lines 481–492 — every load-bearing number contradicts `scaling_law_extended_frontier.tsv` (Nemotron Δ₁T is −0.50 not +0.50; 7/12 not 9/12; outlier is Qwen3.5-27B +0.562). Use the linter's drop-in rewrite.
  3. PPO control numbers in `frontier_synthesis_scaling.tex` (F2): "Δ = +0.001, p = 0.75" → "Δ = −0.002, p = 0.374" per `samestack_ppo_grpo.json` (already correct in `scaling_law_iter29.tex`).
  4. Tab. `scaling-cross` CI in `scaling_laws.tex` (S2): re-type from `scaling_law_cross_scale.tsv` — CI [−0.796, +0.313], n_boot ≈ 990, not [−1.119, +0.313]/5000 (transposed from another file).
  5. iter-133 "0/7 three-phase" → "2/7" (S19b, `scaling_laws.tex`); iter-65 "Nemotron unique collapse anchor" → classifier says *valley*, P2 FAILED (I65b, `scaling_law_iter65.tex`); "reward **gain**" → "mean training reward" in `p1_abstract.tex`/`p1_conclusion.tex` (A2c/C1) unless Action 3 Stage 0 adds the Δ column; "zero-variance fraction" → "zero-reward step fraction" (A5, abstract); "only negative slope" (S3b); iter-21 unfilled `$\hat\delta$` placeholder; iter-29 F1 "sustained" → "divergent"; commit or drop the missing `frontier_calls/digests/frontier_P1.md` pointer (F4).
- **Why second:** These are reproducible-by-reviewer factual errors ("we release all code and logs" invites exactly that check); they must be fixed under every branch of Action 1. Certain payoff, no compute.
- **Effort:** ~1 day of edits, no experiments.
- **Touches:** `paper/sections/p1_abstract.tex`, `scaling_laws.tex`, `frontier_synthesis_scaling.tex`, `p1_conclusion.tex`, `scaling_law_iter65.tex`, `scaling_law_iter29.tex`.

### Action 3 — ADD ABLATION: headroom-controlled cross-scale gain (positive control)
- **What:** Close the highest-risk reviewer gap — no experiment shows the benchmark *could* detect a gain slope if one existed. Two stages:
  - **Stage 0 (free, ~1 day, do before submission):** promote the existing iter121 artifacts into the P1 build — the late−early gain-proxy regression (`scaling_law_iter121_late_early.tsv`) and the synthetic-recovery power table — as an explicit "detection floor" paragraph; add the level-vs-gain decomposition (R(1) slope +0.060/decade ≈ mean-reward slope +0.081/decade, i.e. cross-scale variance is base-model level, not RL). This converts a silent flaw into a scoped claim.
  - **Stage 1 (≈40–80 H100-hours, post-audit or camera-ready):** minimal slice of backlog item C2 — 3–4 dense anchors (Qwen3-0.6B, Qwen3.5-4B, Qwen3-8B, opt. Qwen3.5-27B), per-anchor prompt subsets filtered to base pass rate p0 ∈ [0.2, 0.6], GRPO G=8, T=150, 3 seeds ≤8B; endpoint H = (A_post − A_pre)/(1 − A_pre) regressed on log10 N with bootstrap CIs. "H > 0 with slope ≈ 0, tight CI" is the best outcome: it upgrades the null from vacuous to substantive.
- **Why third:** Biggest reviewer-rejection risk ("the central negative result is unfalsifiable as measured"), but Stage 1 is the only costly item on this list and its design should incorporate Action 1's verified labels.
- **Effort:** Stage 0 ~1 day, no GPU; Stage 1 ≈40–80 H100-hours + minor API cost, reusing the `base_instruct_paired.tsv` pre/post eval path and the Tier-A seed machinery.
- **Touches:** `paper/paper_P1_scaling.tex` (add iter121 inputs), new paragraph in `paper/sections/scaling_laws.tex`, `paper/sections/p1_abstract.tex` (rescope the headline), `experiments/results/` (new H-estimand TSVs); Stage 1 uses the existing TRL LoRA r=32 pipeline / 5-seed manager per `sections/_shared_methods.tex`.

---

## 3. The single decisive experiment (launcher-ready)

**Name:** `p1_harness_vs_capability_audit`
**Question:** Is the low-R̄ "incapable" cluster (Qwen3-8B anchor R̄ = 0.2854; Nemotron-120B R̄ = 0.1750) a capability-class effect, or a harness artifact (mislabeled checkpoints + hardcoded ChatML template + 512-token cap on a thinking-mode model)?
**Decision:** Keep the bimodality/capability-class clause (gap 0.531, `scaling_law_iter125_bimodality.tsv`) vs re-attribute the dominant axis to harness compatibility and withdraw the clause.
**Budget cap:** 0 training steps; one sampling-only Tinker job, one model ≤ 8B, ≤ 600 completions; ~0.5 day total.

### Stage A — Label audit (pure re-analysis, ~30 min)
For each of the 5 anchors in `experiments/results/scaling_law_iter117_meta.json`, join: `anchors[i].trace_file` → trace JSON `model` field (`experiments/tinker-runs/results/*.json`) → matching `model_id` row in `experiments/results/base_instruct_paired.tsv` → class label in `experiments/results/scaling_law_iter129_capability_scaling.tsv`. Emit a 5-row table (anchor, trace model ID, paired-tsv label, iter129 label, agree/contradict).
*Pre-checked facts the script must formally reproduce:* `scale_gsm8k_qwen3-8b.json` has `model = "Qwen/Qwen3-8B"` = the **Inst** row (pre-RL 0.2925 ≈ anchor's 0.2854; Base = `Qwen/Qwen3-8B-Base` at 0.8250); Nemotron trace is `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (MoE) but labeled `dense`.

### Stage B — Transcript regeneration + re-score (Tinker, ≤ 600 completions)
No transcripts exist in the repo (traces store per-step scalars only), so regenerate at step 0 (anchor is flat from step 1, so step-0 sampling is a faithful stand-in; no checkpoint needed). Sample `Qwen/Qwen3-8B` on 150 GSM8K train prompts (seed=42, same loader as `experiments/tinker-runs/scripts/tinker_parallel_runner.py`), 2 samples/prompt, two conditions:
- **B1 (harness replica):** the run's exact hardcoded ChatML `<|im_start|>` template + `SYSTEM_PROMPT_MATH`, `max_tokens=512`, `temperature=0.8`, `top_p=0.95`.
- **B2 (harness fixed):** same prompts/decoding, model's own chat template with thinking disabled (fallback: `max_tokens=2048`).

Score all completions with the original `reward_math` (note: it already includes the last-number lenient fallback). Classify each zero-reward B1 completion: (i) correct-but-unparsed; (ii-a) truncated-before-answer at the 512 cap; (ii-b) genuinely wrong; (iii) degenerate. Report (i) alone and (i)+(ii-a) separately.

### Stage C — Cluster recomputation (re-analysis, ~1 h)
Substitute the corrected Qwen3-8B value (B2 mean; cross-check with paired-tsv Base 0.8250) into the 5-anchor pool; recompute the largest sorted R_max gap (vs 0.5313), dip statistic (vs 0.5216, p = 0.056), and cross-scale slope on log10 N. Nemotron-120B cannot be re-sampled under the cap: record trace forensics (peak 0.875 at step 3 amid 55% zero-reward steps — inconsistent with "incapable") and flag attribution *unresolved within budget*.

### Pre-registered decision rule
- **Hard-kill (Stage A):** the 0.2854 anchor's trace model ID equals the Instruct `model_id` while labeled base → instruct-vs-base clustering **void as stated**; all class claims re-derived from verified labels before any capability-class sentence survives.
- **Revision (Stage B/C, any one sufficient):** (a) ≥ 30% of zero-reward B1 completions are category (i); or (b) B2 mean ≥ B1 mean + 0.15; or (c) recomputed largest gap < 0.25, or any anchor crosses the 0.2854↔0.8167 cluster boundary → re-attribute dominant axis to harness compatibility; withdraw the bimodality clause.
- **Survival:** < 10% category (i), |B2 − B1| ≤ 0.05, all 5 labels verified → clause stands (label errata still required); next test = matched base-vs-instruct pair, same family, ≥ 3 seeds, step-1 solve rate stratified to [0.1, 0.9], trigger: gain slope ≥ +0.05/decade with bootstrap 95% CI excluding 0.
- **Indeterminate:** category (i) ∈ [10%, 30%) and B2 − B1 ∈ (0.05, 0.15) → downgrade to descriptive wording ("pre-RL harness-measured accuracy separates the anchors") pending the matched pair.
- **If Stage B cannot run** (renderer/model unavailable): Stages A + C still execute; Stage A alone can void the clustering; apply the descriptive downgrade and mark Nemotron a single-seed case study with a family-label erratum.

### Deliverables
Archive under `experiments/results/` with an iter-style meta JSON: Stage-A verification table, B1/B2 per-completion scores + classifications, Stage-C recomputed gap/dip/slope. These feed the Section Drafter as `{{raw_notes}}`; thresholds 30% / 0.15 / 0.25 / ±0.05 go into `{{must_keep_points}}`.

---
*Synthesized 2026-07-04 from the four P1 contract outputs; every number above traces to a file cited in those reviews.*
