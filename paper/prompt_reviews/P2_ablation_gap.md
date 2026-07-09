# P2 (ZVF) — Ablation Gap Finder

Contract executed: `research_prompts/revision/ablation-gap-finder.md` against
`paper/paper_P2_zvf.tex` ("The Zero-Variance Fraction: A Descriptive Diagnostic for
Signal Starvation in GRPO"). Cross-checked against `experiments/results/` and
`experiments/FRONTIER_EXPERIMENT_BACKLOG.md` before claiming the gap.

**Main claim (one sentence):** ZVF is a cheap, mechanistically grounded diagnostic
worth reporting alongside mean reward — it tracks signal starvation and collapse
(rho ~ 0.56–0.62; ZVF Risk Index max-fusion AUROC 0.929 / 0.805), even though it
only weakly predicts final held-out outcome (rho ~ 0.27).

---

## 1) Missing ablation

**A matched-protocol baseline-control + out-of-sample ablation for the ZVF Risk
Index: compute the identical three early-warning channels (level, rolling lag-1
"critical slowing down", drift slope) from the *mean-reward trace* (and the
*entropy trace* where logged), max-fuse them identically, and compare AUROC against
the ZVF version under leave-one-method-out cross-validation on the same 52-row
panel.**

Two fused deficiencies make this one ablation:

- **(a) No non-ZVF comparator anywhere in the collapse-prediction pipeline.** Every
  predictor ever benchmarked for the alarm/AUROC task is a ZVF transform:
  `zvf_iter30_feature_importance.tsv` (current_zvf, trailing_mean_25,
  trailing_auc_25, trailing_slope_10 — all ZVF), `zvf_iter78_single_channel.tsv`
  (AR1/CUSUM/H-run, all on the ZVF trace), `zvf_iter110/118/130/134_axis_aurocs.tsv`
  (magnitude/CSD/drift, all on the ZVF trace). A grep across `scripts/zvf_*.py`
  finds no reward-trace or entropy-trace alarm baseline. Yet the raw inputs exist:
  `experiments/results/variance_mitigation.tsv` logs per-step `reward_mean` for all
  45 method-seed trajectories, and `groupsize_zvf_sweep.json` `step_log` carries
  per-step `mean_reward`, `entropy`, `advantage_variance`, `grad_norm`.
- **(b) The headline AUROC is tuned and evaluated on the same 52 rows.** The
  iter130 subsection of `sections/zvf.tex` states in the text that the CSD channel
  is "logistic-centred at 0.45 **so that** the GRPO-collapse signature of iter126
  ... is the discriminator" and the drift channel is "centred at 0.004/step **so
  that** GRPO's slope ... registers as risk"; the max-fusion itself was chosen over
  the weighted blend after observing that the blend "washes the discrimination
  out" on this panel. The reported 0.929 [0.83, 1.00] is therefore an in-sample
  statistic. Worse, a held-out transfer that was already computed
  (`zvf_iter134_heldout.tsv`, *not* in the paper) shows the index
  **false-alarming on every real converged Qwen3-8B/GSM8K run** (all `converged`
  tinker_gsm8k rows get `predicted_collapse=True` at last10 ~ 0.67–0.71).

## 2) Why reviewers will ask for it (highest-risk gap)

Likely reviewer wording:

> "The paper's practitioner recommendation is to 'report ZVF alongside mean
> reward', its flagship quantitative result is a risk-index AUROC of 0.929, and
> Section 3 calls ZVF 'the earliest cheap diagnostic we have' for collapse. But
> the index's logistic centers and fusion rule are admittedly chosen on the same
> 52 runs it is scored on, and no comparison is made against the identical alarm
> computed from mean training reward or policy entropy — quantities every
> practitioner already logs for free. The authors' own Table
> `tab:zvf-gradient-coupling` shows per-step ZVF correlates with entropy at
> r = −0.84 to −0.91 and with mean reward at +0.79 to +0.85, so the null
> hypothesis that ZVF is a redundant transform of already-monitored signals is
> highly plausible and untested. Without this control I cannot assess the
> paper's central contribution."

Why this is the *most dangerous* gap rather than a nice-to-have:

1. **It is claim-load-bearing.** The pillar's entire positive contribution is
   incremental diagnostic value ("earns its place", "report alongside mean
   reward", the 0.929/0.805 numbers, the 0.30/0.55 operating thresholds). The
   negative results (rho = 0.27, mastery/incapacity aliasing) survive without the
   ablation, but they alone do not carry a paper.
2. **The paper itself concedes the control is missing.** The Discussion says
   "multivariate tests against reward mean, entropy, and divergence proxies are
   needed before any causal reading" — a reviewer reads that as an admission that
   the decisive experiment was skipped despite requiring no new training.
3. **The repo's own data already points the wrong way.** The executed backlog item
   A1 (`FRONTIER_EXPERIMENT_BACKLOG.md`, `pcd_vs_zvf_summary.tsv`) measured
   Spearman(mean_reward, outcome) = **+0.95** vs Spearman(mean_zvf, outcome) =
   +0.56 on the 80 anchors — the free baseline dominates ZVF on the outcome axis.
   And `zvf_partial_correlations.tsv` reports raw r(ZVF, batch-mean reward) =
   +0.74 while explicitly *withdrawing* the earlier entropy/advantage-variance
   "controlled" partials as circular/pseudoreplicated — so the current draft has
   strictly less baseline control than an earlier revision claimed.
4. **The existing defenses will not survive review.** The two places that gesture
   at incremental validity are (i) the n = 14 residualized regression in
   `sections/zvf.tex` (sign-only claim, bootstrap CI on the raw correlation spans
   [−0.29, +0.94]) and (ii) iter38's 3-NN "+14.3pp over reward-only" on n = 14
   rows — where the "reward-only" features are *end-of-run* peak/last10 values,
   i.e., the same quantities the failure label is computed from, not a
   training-time alarm. Neither addresses the risk index or the alarm task.
5. **The already-computed held-out check contradicts the in-sample number.** A
   reviewer who runs the released code will find `zvf_iter134_heldout.tsv` and see
   the 100% false-positive rate on the only real-model GSM8K runs. Having the
   file in the repo but not in the paper is worse than not having it.

## 3) Minimal way to run it (cheapest credible version)

**Pure re-analysis. Zero GPU, zero new training; one script (~1 day), stdlib-only
like `scripts/pcd_vs_zvf.py`.** Compute cap: trivially within any budget.

1. **Reward-trace risk index (full 52-row panel).** For each trajectory in
   `variance_mitigation.tsv` (+ the tool-use and scaling anchors used in iter130),
   compute from the per-step `reward_mean` series the same three channels:
   level (logistic of trailing mean, direction-adjusted), rolling lag-1
   autocorrelation (w = 15), first-half drift slope. Max-fuse. Report
   Mann-Whitney AUROC with B = 2000 bootstrap CI on both the cross-experiment
   (52-row) and within-methods (45-row) panels, side by side with
   `zvf_risk_max`. Also report a 2-line trivial baseline: trailing-mean reward
   with a fixed threshold.
2. **Entropy-trace arm (G-sweep cells only).** Same three channels on the
   per-step `entropy` from `groupsize_zvf_sweep.json` `step_log`; report AUROC on
   the 12 G-sweep trajectories. Scope-limit honestly: entropy is not logged for
   the variance-mitigation panel.
3. **Leave-one-method-out validation.** Re-fit the logistic centers on the panel
   minus one method (or minus one experiment family), score the held-out method,
   pool out-of-fold scores; report LOMO AUROC for both the ZVF index and the
   reward index. This converts the 0.929 from an in-sample fit into a defensible
   number (or exposes it).
4. **Splice `zvf_iter134_heldout.tsv` into the paper** as the transfer check,
   including the tinker_gsm8k false positives, with a re-calibrated operating
   point if needed.
5. **Pre-commit the decision rule** (mirroring the style already used in
   `zvf_counterfactual_appendix.tex`): ZVF retains its claimed value iff
   LOMO AUROC(ZVF max-fusion) − LOMO AUROC(reward max-fusion) > 0 with a
   bootstrap CI excluding 0 on the cross-experiment panel.

## What result would change the paper's conclusion

If the reward-trace max-fusion matches or beats the ZVF max-fusion under
leave-one-method-out CV (difference CI covering 0 or negative), then ZVF adds no
measurable diagnostic value over a signal every dashboard already shows. The
conclusion "ZVF earns its place as a descriptive diagnostic", the recommendation
to "report ZVF alongside mean reward", the "earliest cheap diagnostic" sentence,
and the iter130 Risk Index thresholds (0.30/0.55) would all have to be retracted
or reframed; the pillar would collapse from "useful new diagnostic + caveats" to
"negative result: an unsigned saturation statistic that is a redundant transform
of training reward" — a different paper. Conversely, a surviving positive margin
would give the pillar its first genuine incremental-validity evidence and
directly neutralize the most probable rejection argument.

## Cross-check: ablations that already exist (why this gap is real)

Verified in `experiments/results/` and the P2 sections — the ZVF-*internal*
ablation coverage is thorough, so a reviewer cannot be answered with "we ablated
a lot":

- Group size G in {2,4,8,16} (`groupsize_zvf_sweep.*`, `zvf_dynamics.tex`).
- ZVF threshold sensitivity (`zvf_iter74_threshold_sensitivity.tsv`,
  `zvf_iter86_threshold_curve.tsv`, `zvf_iter86_k_persist_sensitivity.tsv`).
- Shuffle-null / stability controls (`zvf_iter94_shuffle_null.tsv`).
- Calibration and operating-point sweeps (`zvf_iter102_calibration.tsv`,
  `zvf_iter118_calibration.tsv`, `zvf_iter122_op_sweep.tsv`).
- Feature-variant comparisons *within* ZVF (iter30, iter34, iter42, iter78).
- Difficulty strata (`zvf_iter62_difficulty_strata.tsv`), survival/hazard
  (`zvf_iter82_*`), lead-time (`zvf_leadtime_*`, iter126).
- PCD/LARQ vs ZVF structural comparison — backlog item A1, executed 2026-07-03
  (`pcd_vs_zvf_{shape,summary}.tsv`); note this compares *replacement* statistics
  on the outcome axis, not baseline *alarms* on the collapse-prediction task.
- Reward-shape counterfactual: pre-registered but **unrun** (12 empty cells in
  `zvf_counterfactual_appendix.tex`; `experiments/tool_use_zvf_sweep/`) — a real
  secondary gap, but openly disclosed and framed as protocol, hence lower risk.

What does **not** exist anywhere: a reward- or entropy-trace alarm comparator, or
any out-of-fold evaluation of the iter130 risk index. The backlog (A1–A5, B1–B3,
C1–C3) does not contain this ablation either — closest is A1, which explicitly
flags its own confound ("contemporaneous mean_reward-outcome is partly
mechanical") and stops short of the alarm-task head-to-head.

## Runner-up gaps (lower risk, for completeness)

1. **Reward-shape counterfactual left as an empty pre-registered table** — a
   reviewer may ask why a 12-run, 50-step LoRA sweep was not run before
   submission; mitigated by the pre-registration framing.
2. **Task/scale diversity of the dynamics claims** — the sticky-drift and
   late-phase-saturation results rest on Qwen2.5-0.5B synthetic arithmetic
   (disclosed in `zvf_dynamics.tex` scope paragraph); disclosed, so grumble-level.
3. **Temperature dependence** — ZVF is mechanically temperature-dependent; all
   runs use temp 1.0 and the anti-herding sign-reversal already shows
   regime-dependence; no sweep exists. Secondary because the paper never claims
   temperature invariance.
