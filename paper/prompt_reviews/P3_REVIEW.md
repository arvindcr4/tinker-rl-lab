# P3 Review Synthesis — Group Size (`paper_P3_group_size.tex`)

**Synthesized from:** `P3_stress_test.md`, `P3_claim_lint.md`, `P3_decisive_experiment.md`.
**Missing input:** `P3_ablation_gap.md` was never produced (P1/P2/P4 each have one; no P3 ablation-gap file exists anywhere in the worktree as of 2026-07-04 11:25). Its core content is recoverable from the claim lint, which explicitly routes the A7/R3/C4 gap ("identity validated against measured gradients" — no artifact) to the Ablation Gap Finder; that gap is Action 3 below. This synthesis therefore rests on 3 of 4 contracts.
**Date:** 2026-07-04.

---

## 1) Verdict on the central claim

The central claim splits in two, and the halves fare very differently. The **non-monotonicity half** (trainability vs G is non-monotone with no universal optimum) is robust: it reproduces exactly from measured artifacts (`groupsize_zvf_sweep.json`, interior apex at G=8 with overlapping SEs) and is hedged appropriately. The **equivalence half** ("G=4 ≈ G=32 token-normalized on capable models") is **not robust**: no measured G=32 cell exists anywhere in the benchmark, the only bridging dataset is an illustrative reconstruction (`FALLBACK_ROWS`) that itself contradicts equivalence at every budget T ≥ 4M (retention 97.6% → 83.3% → 75.0% → 72.7%), and the two regimes where equivalence *is* observed are both metric-compressed (near-ceiling arithmetic, range ~1.3 pp; under-trained T=1M floor, CI spanning zero). Worse, the paper's own iter115 linkage analysis falsifies the proposed preference-density mechanism in exactly the regime the claim extends into (GU ratio stays >4× while retention collapses — gradient noise σ²_R/G, not contrast starvation, is binding). The quantitative core is otherwise in good shape (21/38 claims reproduce exactly), but 5 unsupported claims — including a thrice-repeated gradient-validation claim with no artifact, one hard numeric mismatch (~2.7×), and one impossible statistic (p<10⁻²⁴ at n=4) — mean the paper is not submission-safe until the prose is fixed and one measured G=4-vs-G=32 cell exists.

## 2) Top 3 prioritized actions

### Action 1 — FIX PROSE: remove unsupported claims, hard numeric errors, and qualifier-shedding
**Type:** fix prose. **Effort:** ~half a day of LaTeX edits, zero compute. **Do first — required regardless of any experiment outcome.**

| Fix | Exact location | Change (per lint rewrite suggestions) |
|-----|----------------|----------------------------------------|
| A7/R3/C4: "validated against our measured gradients" — **no artifact exists** (only scalar `grad_norm`); appears 3× | `sections/p3_abstract.tex:16`, `sections/p3_results_intro.tex:22–24`, `sections/p3_conclusion.tex:8–9` | → "consistent with our measured training diagnostics (advantage variance, ZVF, gradient norms), with the decisive gradient-level test specified but not yet run" |
| M20: impossible statistic p<10⁻²⁴ (and p<10⁻⁴ variant) at n=4 | `sections/group_size.tex:375–379` and `:366` | → "Spearman ρ=−1.000 (n=4; exact permutation p=0.083 — reported descriptively)" |
| M13: half-saturation column off ~2.6–2.7× from its own cited TSV | `sections/group_size.tex:1331–1351` (Table `tab:iter95-ceilings`) | → replace with artifact values 0.593/0.705/0.858/1.208/1.750 M from `group_size_iter95_ceilings.tsv`, or fix `scripts/group_size_iter95.py` if the TSV is stale |
| A3/M3/M19: illustrative reconstruction presented as measured "sweep"; imperative practitioner rule built on it | `sections/p3_abstract.tex:10–11`, `sections/group_size.tex:44–53, 282–287, 316–323, 416–426` | → tag every abstract/headline use of the G=4-vs-G=32 grid "illustrative, reconstructed from ablation logs"; soften "never deploy G=4 at T≥16M" to the reconcile section's own "qualitative argmax pattern only" register |
| A4/R1/C2: "reward retention ≈100.3%" is held-out **accuracy**, not reward (reward retention = 99.3%) | `sections/p3_abstract.tex:11–13`, `sections/p3_results_intro.tex:6–13`, `sections/p3_conclusion.tex:21–22` | → "held-out accuracy retention"; conclusion adds "on the measured near-ceiling task; the GSM8K reanalysis instead shows retention falling to ≈73%" |
| M15: "HOLDS NATIVELY" headline vs artifact flag `wu_native_claim_holds=False` | `sections/group_size_iter135.tex:29–40` + `scripts/group_size_iter135.py` | → reconcile the flag logic or soften the headline |
| M16: linear fit mislabeled as log-log (true log-log slope is −0.137, not −0.230) | `sections/group_size.tex:354–373` | → relabel as linear fit; fix the "−0.230 vs −0.5 per decade" units mix |
| A1/C5: "70+ runs" (61 completed of 95), "release … gradients" (only scalar norms) | `sections/p3_abstract.tex:5–7, 18–19`, `sections/p3_conclusion.tex:25–27` | → honest counts per `experiments/master_results.csv`; "per-step gradient-norm logs" |

### Action 2 — RUN EXPERIMENT: the measured token-matched G=4 vs G=32 pair (the decisive experiment, Section 3)
**Type:** run experiment. **Effort:** Stage 0 ≈ hours, **zero compute** (re-slice of existing logs); Stage 1 = 6 runs × 4M tokens = **24M sampled tokens** on Tinker (≈1–2 days wall clock). **Touches:** new script under `scripts/` + results into `experiments/results/`; outcome then rewrites `sections/group_size_reconcile.tex` (caption + downstream of `tab:groupsize-tokennorm`) and the iter107/111/115/135 subsections of `sections/group_size.tex` per triggers R1–R3.

This is the single highest-information action for the paper's thesis: every version of the equivalence claim currently rests on a synthetic grid, and this one cell either falsifies the claim (R1), falsifies its mechanism (R2), or forces withdrawal of the reconstruction and all downstream artifacts (R3). Full launcher spec in Section 3.

### Action 3 — ADD ABLATION: gradient-vector residual-isomorphism check (the missing ablation-gap item)
**Type:** add ablation. **Effort:** ~1 day engineering + a small rerun of the existing 0.5B/arithmetic setup with gradient-vector (or per-rollout score-projection) logging; trivial compute. **Touches:** the proposed-but-never-run test at `sections/frontier_synthesis_group_size.tex:114–125`; new script + TSV in `scripts/` / `experiments/results/`; then upgrades the Action-1 hedged prose at `sections/p3_abstract.tex:16`, `sections/p3_results_intro.tex:22–24`, `sections/p3_conclusion.tex:8–13` back to a genuine validation claim.

Run the paper's own specified check — per-batch cos(V_GRPO − V_mDPO, V_KL) ≈ 1 — to validate Eq. (fs-contrast) on real gradients. This is the exact artifact whose absence makes A7/R3/C4 unsupported, and the item the claim lint routed to the (missing) Ablation Gap Finder contract. If not run before submission, the Action-1 rewrite stands and the claim stays demoted. Secondary ablation from the lint's missing-evidence list, same bucket: produce a repo artifact for the "+0.253 at p≥0.75 vs +0.010 at p<0.5" stratified-gain claim (R2), currently backed only by an external frontier-model exchange.

## 3) The single decisive experiment (launcher-ready)

**Question:** Does token-normalized equivalence G=4 ≈ G=32 survive on a *measured*, token-matched pair at T = 4M optimizer-visible tokens per arm, on a capable model at mid-difficulty (p₀ ∈ [0.3, 0.6]) — the smallest budget where the reconstruction predicts non-equivalence (Δ = +0.11)? Same run identifies the mechanism (contrast starvation via GU/ZVF vs gradient noise σ²_R/G).

**Resource caps:** Tinker API only; model ≤ 8B; ≤ 40 steps/run; ≤ 600 unique prompts; Stage 1 hard cap 24M sampled tokens.

**Stage 0 — gate, zero compute (run first; can kill the hypothesis alone):**
- Data: `experiments/results/groupsize_zvf_sweep.json` (Qwen2.5-0.5B, arithmetic, G∈{2,4,8,16}, seeds {42,123,456}, 40 steps, per-step logs; verified present).
- Slice: token-matched pairs G=4@step 4s vs G=16@step s, restricted to G=16 train mean_reward ∈ [0.3, 0.8] (window s∈{2..6}; seed 456: {2..7}); ~16 paired points split into sub-windows W1 = s∈{2,3,4}, W2 = s∈{5,6(,7)}.
- Statistic: per-window mean paired difference d = reward(G=16@s) − reward(G=4@4s), paired within seed; sign consistency across W1/W2.

**Stage 1 — decisive measured pair (only if Stage 0 passes):**
- Model: Qwen/Qwen3.5-4B via Tinker (fallback Qwen3-8B-Base).
- Prompts: calibrate with k=4 samples at T=1.0 on 600 GSM8K train prompts → keep 300 with per-prompt pass ∈ [0.25, 0.6] as frontier pool (aggregate p₀ must land in [0.3, 0.6]; tighten band to [0.2, 0.5] if p₀ > 0.6). Eval: 300 fixed held-out GSM8K prompts, greedy, once per finished arm per seed.
- Arms (identical LR/schedule/max-len, both ≤40 steps, step- and token-matched):
  - G=4: batch ≈ ⌈4M/(40·4·L̂)⌉ prompts/step (≈84 at L̂=300), sampled with replacement from the pool (log epoch count).
  - G=32: batch ≈ ⌈4M/(40·32·L̂)⌉ prompts/step (≈10 at L̂=300).
  - Stop each arm at first step where cumulative optimizer-visible tokens ≥ 4M; record exact T (arms within 5% of each other).
- Seeds: {42, 123, 456} per arm → 6 runs, ≈24M sampled tokens.
- Primary metric: held-out accuracy; Δ = acc(G=32) − acc(G=4), prompt-level paired within seed (900 paired obs), 10k-resample paired bootstrap 95% CI + TOST at ε ∈ {0.024, 0.05}. Power: paired SE ≈ 0.018 → ≈4.8 SD margin at predicted Δ=+0.11; TOST@0.05 well-powered, TOST@0.024 directional only.
- Mechanism telemetry (free): per-step per-group advantage variance, ZVF, GU — identical schema to `groupsize_zvf_sweep` / iter115.

**Decision rules (first firing rule decides):**
- **R0 (Stage 0):** |mean paired d| > 0.02, sign-consistent in ≥2/2 sub-windows across all 3 seeds → equivalence fails within measured G range; revise before any new compute; re-scope Stage 1 to a T_equiv bisection (1M vs 4M, single seed pair).
- **R1 (primary):** Δ 95% CI lower bound > +0.024 (retention < 90% with CI upper bound < 0.976) → demote claim to "equivalence is budget-conditional; holds only under-trained, T ≤ T_equiv ≈ 1M."
- **R2 (mechanism kill-shot):** GU(G=4)/GU(G=32) ≥ 2 while retention < 90% → preference-density mechanism falsified for this regime; demote "preference-density dial" to the contrast-saturated near-ceiling regime only.
- **R3 (symmetric):** TOST@ε=0.05 passes (p < 0.05) → withdraw or re-derive FALLBACK_ROWS table and all downstream artifacts (iter107/111/115/135 retention curve, TOST table, compute-cost projection, T* extrapolations).
- **R4 (inconclusive):** CI straddles band and TOST fails → do NOT add runs; re-eval the 3 existing checkpoint pairs on the full GSM8K test split (sampling-only) to shrink SE ~2×, re-apply R1/R3 once; if still in band, downgrade claim to "not distinguishable from a ≤5-point gap at T=4M" and gate all T* artifacts behind that caveat.

**Validity preconditions (violation voids the readout, fires no trigger):** aggregate p₀ ∈ [0.3, 0.6]; realized T = 4M ± 5% per arm, arms within 5%; all 6 runs ≥ 35/40 steps; identical 300 fixed eval prompts, greedy, every run. Design, thresholds, and triggers are frozen — no post-hoc revision.

**Known discrepancy carried through all contracts:** the upstream "iter138 contrast-yield analysis" does not exist in this worktree (max iter135); the 4.15–5.03× GU figure is iter115's (`group_size_iter115_zvf_linkage.tsv`).

---

### Bottom line

Non-monotonicity: keep, it is measured and hedged. Equivalence: currently an extrapolation across three axes at once (G 16→32, scale 0.5B→capable, task ceiling→frontier) bridged only by a synthetic table that contradicts the claim at every T ≥ 4M. Fix the prose today (Action 1), run Stage 0 this week for free, and let the 24M-token Stage 1 cell decide which of the three demotions (R1/R2/R3) the paper takes — or, if R3 fires, which tables it withdraws.
