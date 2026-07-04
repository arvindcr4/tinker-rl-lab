# P2 Minimal Decisive Experiment — Can the ZVF Critical-Slowing-Down Claim Survive on Real Data?

Executed per `research_prompts/design/minimal-decisive-experiment.md` (Ready-to-Copy
Prompt contract). Role: experiment planner optimizing decision value per cost.
Date: 2026-07-04. Worktree: `/home/claude/tinker-rl-lab-minimax`.

Feeds from: `paper/prompt_reviews/P2_stress_test.md` (Hypothesis Stress Test). Per the
handoff convention, the stress test's Disconfirming check D2 is the input
`{{research_question}}` and its "Result pattern forcing revision" is the input
`{{decision_needed}}`.

---

## Input (placeholder fills)

- **`{{research_question}}`** (from D2): On *real* GRPO runs of the paper's
  verifiable-arithmetic task, does the linearly detrended, matched-window rolling
  lag-1 autocorrelation (w=15) of the per-step ZVF trace separate collapse seeds from
  safe seeds — with collapse defined *externally* (held-out accuracy ≥20% relative
  below its running peak for ≥5 consecutive steps, never via any ZVF threshold) — and
  does the ZVF τ=0.4 crossing lead the external collapse step by ≥10 rollout steps?
- **`{{decision_needed}}`**: Whether to (i) withdraw/demote the CSD early-warning
  claim from "shows" to at most "untested hypothesis", (ii) cut or re-label
  `tab:zvf-by-library` as "simulation projection", and (iii) re-mark every number
  sourced from `variance_mitigation.tsv` as synthetic/dry-run projection — per the
  pre-registered triggers (a)–(d) in the stress test.
- **`{{resource_limit}}`**: Tinker API only (no local GPU); model ≤ 8B; ≤ 40 training
  steps or sampling-only; ≤ 600 prompts — OR pure re-analysis of data already under
  `experiments/results/`. Prefer reuse of existing checkpoints/logs.

**Budget conflict resolved by design.** D2 as written (5 seeds × ≥150 steps = ≥750
training steps) exceeds the 40-step budget ~19×. The design below extracts the same
*decision* at ~5% of that cost by exploiting an asset the stress test's V6 missed: the
repo already contains one **real** per-step ZVF trajectory. `experiments/results/
arithmetic_metrics.jsonl` is a genuine 100-step Tinker run (`tinker_cookbook.recipes.
math_rl.train`, Llama-3.2-1B, G=4, 100 groups/batch, lr=1e-4 per
`experiments/run_tinker.sh`; run id `39aa5eb2-…:train:0`) logging per step
`env/all/by_group/frac_all_good` and `frac_all_bad`, so
ZVF_t = frac_all_good_t + frac_all_bad_t exactly, plus per-step accuracy on the fresh
100-prompt batch and six saved sampler checkpoints
(`arithmetic_checkpoints.jsonl`, batches 20–100 + final).

---

## 1) Setup

**One dataset (verifiable arithmetic), one metric family (detrended rolling lag-1 of
ZVF vs external collapse), two arms.**

**Arm A — pure re-analysis (zero Tinker cost; ALREADY EXECUTED, read-only).**
Parse `arithmetic_metrics.jsonl`; compute ZVF_t, accuracy_t, the pre-registered
external collapse detector, first ZVF>τ crossings for τ ∈ {0.4, 0.5, 0.6, 0.7, 0.9},
and rolling lag-1 (w=15) of the *linearly detrended* ZVF trace. Run the identical
statistic on freshly regenerated simulator traces
(`variance_mitigation_integration.synthesize_rows`, method=grpo, seeds 0–4) to test
whether the simulator that generated 100% of the published CSD evidence reproduces
real dynamics. Measured results (script logic preserved in this doc's appendix;
reproducible in ~1 min):

| quantity | real run (safe) | simulator grpo seeds 0–4 |
|---|---|---|
| external collapse (pre-registered rule) | **none** (max drawdown 1.75%) | 3 of 5 by hard-coded quota |
| ZVF trajectory | 0.190 → 0.990 by step 10 → ~1.0 | 0.0 → 0.86–0.93 over 100 steps (sigmoid) |
| first step ZVF > 0.4 | **step 2** | steps 43–46 |
| raw rolling lag-1 (w=15), iter126 protocol | **0.045** | 0.38–0.46 |
| detrended rolling lag-1 (w=15), mean | **−0.068** | −0.158 … −0.055 |

Three Arm-A facts are already decision-relevant: (1) the simulator's timescale is
wrong by ~20× (τ=0.4 crossing at step 2 real vs 43–46 simulated) and its raw lag-1 is
inflated ~9× vs real — it is disqualified as a stand-in for real GRPO dynamics on this
task; (2) linear detrending *erases the entire "CSD" signal inside the simulator
itself* (detrended means ≤ 0 for all five seeds, collapse and safe alike) — the
published 0.609-vs-0.415 gap is the sigmoid trend, not slowing-down; (3) the τ=0.4
alarm fires at **step 2 of a healthy run** that ends at 100% accuracy — a real-data
false positive: on this task ZVF saturates benignly via mastery within ~10 steps.

**Arm B — one new destabilized run (the only new Tinker spend).**
Because the paper's smallest real config masters arithmetic in ~10 steps, a vanilla
150-step 0.5B run per D2 would mostly log a flat ZVF≈1 plateau and (per Arm A) almost
surely zero collapse events — trigger (a) would fire trivially. The cheapest run that
can produce a *real* collapse trace, i.e. give the CSD claim its best possible shot:

- `tinker_cookbook.recipes.math_rl.train`, same arithmetic task, same model family as
  the existing safe trace (Llama-3.2-1B; ≤ 8B ✓) so safe-vs-collapse is a
  within-model contrast against the existing 100-step log.
- G=4, groups_per_batch=100 (matching the safe run), **lr=1e-3 (10× the safe run)** —
  the standard cheapest destabilizer for small-model GRPO — **40 training steps** (≤40 ✓).
- Fixed prompt pool: 480 unique training prompts + 120 held-out prompts = 600 (≤600 ✓).
- Per step, log ZVF_t from `by_group/frac_*` and evaluate held-out accuracy by
  sampling the 120 held-out prompts (sampling-only; not training steps).
- Checkpoint sampler weights every 10 steps (reuse pattern of
  `arithmetic_checkpoints.jsonl`) so any post-hoc probe is sampling-only.

Cost: one run, ≈15–30 min wall-clock at the safe run's observed 8–18 s/step plus
eval sampling. Everything else is CPU re-analysis.

**Analysis protocol (pre-registered, identical to D2's statistics at reduced n).**
Detect external collapse on the *held-out* accuracy curve of Arm B with the
pre-registered rule. If collapse at step s_c: compute mean detrended rolling lag-1
(w=15) on [0, s_c) of the Arm-B trace, and on *all length-matched windows*
(every contiguous window of length s_c) of the real safe trace (Arm A), giving a
matched-window safe distribution (mean μ_safe, SD σ_safe, max). Compute τ-crossing
lead times relative to s_c for τ ∈ {0.4, 0.5, 0.6, 0.7} with a k=5-step persistence
requirement, and record whether the same τ also fires persistently on the safe run
(alarm precision).

## 2) Decision rule

Pre-registered mapping to the stress test's triggers, adapted to n = 1 collapse
candidate + 1 real safe reference (the minimum that can decide the *next step*):

- **T-a (existence).** If Arm B produces **no** external collapse within 40 steps
  even at 10× lr → real collapse traces are not obtainable at any budget this paper
  can spend → trigger (a) fires by construction: **demote the CSD claim to "untested
  hypothesis" immediately**; do not schedule further runs.
- **T-b/c (separation).** If collapse occurs: the claim survives only if the Arm-B
  pre-collapse detrended rolling lag-1 exceeds μ_safe + 2σ_safe **and** exceeds the
  matched-window **maximum** of the safe trace (the band-overlap test that already
  failed in D1 on synthetic data). Otherwise triggers (b)/(c) fire → demote.
- **T-d (actionability).** The claim survives only if some single τ ∈ {0.4…0.7}
  yields a persistent crossing ≥10 steps before s_c on the collapse run **and** no
  persistent crossing on the safe run before its step 60. Arm A already shows every
  τ ≤ 0.9 fires by step 4 on the safe run, so T-d can only pass if the collapse-run
  ZVF *starts* low and the safe-run comparison is re-scoped to matched early windows
  — this must be reported as-is, not post-hoc re-thresholded.
- **Cross-library claim (no experiment needed).** `tab:zvf-by-library` cannot be
  regenerated from real runs within any budget here (AERO is not implemented on this
  stack); by the pre-registered pattern it is **re-labeled "simulation projection"
  unconditionally**. Likewise, D0/D1 + Arm-A facts (1)–(2) force the mandatory
  revision NOW, independent of Arm B: every number sourced from
  `variance_mitigation.tsv` (iter126 H1/H2/H3, `tab:zvf-by-library`, zvf-dynamics
  pooled table, iter130 risk-index rows) is re-marked synthetic/dry-run projection or
  deleted. Arm B decides only whether the CSD idea survives as a *hypothesis with
  preliminary real-data support* versus *untested/disconfirmed hypothesis*.

## 3) Success threshold

The CSD early-warning claim survives (as "preliminary real-data support", never as
"shows", pending the full 5-seed D2) **iff ALL of**:

1. Arm B exhibits an external collapse at some s_c ≥ 20 (so [0, s_c) contains ≥ 6
   detrended w=15 windows);
2. Arm-B pre-collapse mean detrended rolling lag-1 > μ_safe + 2σ_safe **and**
   > max over length-matched safe windows (no band overlap);
3. at least one τ ∈ {0.4…0.7} gives k=5-persistent lead time ≥ 10 steps on Arm B
   with zero persistent false alarm on the matched early windows of the safe run.

Any single failure ⇒ the pre-registered demotion executes. (Note the asymmetry, which
is what makes one run decisive: passing all three at n=1 only *permits* further
investment in the full D2; failing any one at n=1, on the claim's most favorable
achievable conditions, already satisfies the stress test's "ANY of (a)–(d)"
withdrawal pattern on real data.)

## 4) Failure interpretation

- **If T-a fires (no collapse at 10× lr in 40 steps):** on this task binary-reward
  GRPO ends in benign mastery-saturation (ZVF→1 with accuracy→1, as the real safe
  trace already shows), so "collapse early warning" has no referent here. Fallback
  action (one): rewrite the ZVF story as a *saturation/compute-waste* diagnostic
  (mechanism steps 1–2, which the real trace genuinely supports: 99% of groups
  produce zero gradient signal after step 10 while accuracy is perfect) and demote
  all CSD/collapse language to future work.
- **If T-b/c fires (no separation / band overlap):** matches D1's synthetic-data
  result on real data; interpretation: rolling lag-1 of a bounded, trend-dominated
  fraction is a windowing artifact, not resilience loss. Same fallback action.
- **If T-d fires (no actionable lead / false alarm on safe run):** ZVF level
  thresholds cannot distinguish mastery from pre-collapse (the aliasing
  counterexample, now on real trajectories). Same fallback action.
- **If all three pass:** keep the claim only as "consistent with CSD in one real
  destabilized run"; the mandatory synthetic re-labeling still applies; the full
  5-seed D2 becomes justified follow-up spend.
- **In every branch**, the revision already forced by D0/D1 and Arm A executes now.

---

## Workflow handoff (→ Section Drafter from Notes)

- `{{raw_notes}}` := the Setup above + Arm-A measured table + Arm-B logs
  (`zvf_t`, held-out acc_t, s_c, matched-window lag-1 stats, lead-time grid) once run.
- `{{must_keep_points}}` := the Decision rule (T-a…T-d thresholds verbatim) and the
  unconditional re-labeling of `variance_mitigation.tsv`-derived numbers and
  `tab:zvf-by-library`.

## Appendix: Arm-A reproduction (read-only, ~1 min, run from repo root)

```python
import json, numpy as np
rows = [json.loads(l) for l in open("experiments/results/arithmetic_metrics.jsonl")]
zvf = np.array([r["env/all/by_group/frac_all_good"] + r["env/all/by_group/frac_all_bad"] for r in rows])
acc = np.array([r["env/all/correct"] for r in rows])
peak = np.maximum.accumulate(acc)                      # external collapse rule:
run = 0; s_c = None                                    # >=20% rel below peak, >=5 steps
for t, b in enumerate(acc <= 0.8 * peak):
    run = run + 1 if b else 0
    if run >= 5: s_c = t - 4; break
def rl1(x, w=15, detrend=True):
    out = []
    for i in range(len(x) - w + 1):
        s = np.array(x[i:i+w], float); t = np.arange(w)
        if detrend: s -= np.polyval(np.polyfit(t, s, 1), t)
        out.append(np.corrcoef(s[:-1], s[1:])[0, 1] if s.std() > 1e-12 else np.nan)
    return np.array(out)
print(s_c, zvf[0], zvf[-1], int(np.argmax(zvf > 0.4)),
      np.nanmean(rl1(zvf)), np.nanmean(rl1(zvf, detrend=False)))
# -> None 0.19 1.0 2 -0.0677 0.0451
# Simulator comparison: import experiments/variance_mitigation_integration,
# synthesize_rows(MethodConfig.for_method("grpo"), seed) for seed in 0..4, same rl1:
# first ZVF>0.4 at steps 43-46; raw rolling lag-1 0.38-0.46; detrended -0.158..-0.055.
```

Notes and caveats: `env/all/correct` in the safe run is accuracy on the fresh
100-prompt batch sampled each step (on-policy, pre-update), an adequate held-out
proxy for the *safe/no-collapse* determination; Arm B uses a true fixed 120-prompt
held-out set for the collapse determination, as the pre-registration requires. Model
is Llama-3.2-1B (the config actually behind the existing log), substituted for D2's
Qwen2.5-0.5B to keep the safe-vs-collapse contrast within one model; both satisfy
the ≤8B constraint.
