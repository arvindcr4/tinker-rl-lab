# P2 Hypothesis Stress Test — ZVF as Diagnostic + Critical-Slowing-Down Early Warning

Executed per `research_prompts/design/hypothesis-stress-test.md` (Ready-to-Copy Prompt contract).
Role: skeptical reviewer testing causal logic.
Date: 2026-07-04. Worktree: `/home/claude/tinker-rl-lab-minimax`.

---

## Input

**Hypothesis.** Zero-Variance Fraction (ZVF) is a useful cross-library descriptive
diagnostic of when binary verifiable rewards stop teaching, and its trajectory shows
critical-slowing-down (CSD) early warning before GRPO collapse (rolling lag-1
autocorrelation 0.61 vs 0.42, n=3 collapse vs 2 safe seeds).

**Proposed mechanism (stated explicitly, derived from the paper's own sections).**

1. *Gradient-nullity step* (`p2_abstract.tex`, `zvf_gradient.tex`): in GRPO a prompt
   group whose G completions all receive the same binary reward has zero
   group-centered advantages, so it contributes no policy gradient. ZVF = the share
   of groups in this state = the share of compute producing no learning signal.
2. *Saturation-drift step* (`frontier_synthesis_zvf.tex` Eq. 1, `zvf_dynamics.tex`):
   under binary rewards with latent per-prompt success probability p_x,
   E[ZVF] = p_x^G + (1−p_x)^G. Training drives p_x → 1 on mastered prompts, so ZVF
   drifts monotonically upward (GRPO early-phase 0.044 → late-phase 0.870) and is
   temporally sticky (lag-1 ρ₁ ≈ 0.94 method-pooled), i.e. it tracks when rewards
   "stop teaching."
3. *Cross-library step* (`zvf.tex` Table `tab:zvf-by-library`): because step 1 is a
   mechanical property of group-relative advantage estimation, ZVF is comparable
   across mitigation libraries run on one stack; vanilla GRPO 0.48 vs AERO 0.22
   (halved) with tied last-10 accuracy.
4. *CSD step* (`zvf_iter126.tex` H1–H2, citing Scheffer 2009): as the policy
   approaches collapse, the ZVF trajectory loses resilience — perturbations decay
   more slowly — so the rolling (w=15) lag-1 autocorrelation rises before the
   collapse event. Measured: collapse seeds {0.574, 0.607, 0.647} (mean 0.609) vs
   safe seeds {0.409, 0.421} (mean 0.415), Cohen d = 7.3; ZVF crosses τ=0.4 a mean
   of 35.7 steps before collapse. Hence ZVF is claimed as an early-warning signal.

**Known counterexamples (input to this test).**

- `scripts/pcd_vs_zvf.py` on the *real* GSM8K per-group reward tensors
  (`experiments/results/pcd_vs_zvf_summary.tsv`, 600 groups): (a) mastery–incapacity
  aliasing — ZVF-indicator = 1.0 for both p̂=1 (mastered, n=76) and p̂=0 (incapable,
  n=19) groups, opposite outcome implications; (b) micro-jitter falsification — a
  U(0, 1e-4) reward jitter flatlines batch ZVF 0.1583 → 0.0000 while PCD is invariant
  (0.153802 → 0.153802): ZVF reports a fully "healthy" batch under any non-binary
  reward component (length penalty, KL-shaped reward).
- AERO (arXiv 2602.14338) already treats ZVF-style zero-variance groups as its
  control variable — and its *adaptive group size* changes G online, while
  E[ZVF] = p^G + (1−p)^G depends on G, so cross-library ZVF comparisons involving
  AERO are not measurements on a common scale.
- The EWS claim rests on n=3 vs n=2 seeds of a single method on a single task.

---

## 1) Weakest link

**The critical-slowing-down early-warning claim (mechanism step 4) has no real
data behind it, and even on its own data the 0.61-vs-0.42 gap is a
window-truncation artifact computed on synthetic trajectories with circular,
quota-assigned collapse labels.**

Verified during this stress test (all reproducible, commands below):

- **V1 — the trajectories are simulator output, not training runs.**
  `experiments/results/variance_mitigation.tsv` — the sole source for iter126's H1/H2
  and for the cross-library dynamics tables — is byte-identical to the output of the
  `--dry-run` synthesizer `synthesize_rows()` in
  `experiments/variance_mitigation_integration.py` (100/100 exact matches on
  (zvf, reward_mean) for grpo seed 0; docstring: "emit synthetic per-step rows whose
  aggregate statistics match the projections in Table variance-head2head"). The
  per-method ZVF plateaus (grpo 0.88, aero 0.72, …) and sigmoid midpoints are
  *hard-coded constants* in `zvf_curve()`.
- **V2 — the paper's exact H1 numbers regenerate from the simulator.** Running
  iter126's own `lag1`/`rolling_lag1` (w=15) on freshly regenerated
  `synthesize_rows` traces reproduces the published values to 4 decimals:
  collapse 0.5741 / 0.6072 / 0.6469, safe 0.4094 / 0.4213.
- **V3 — the gap vanishes under matched windows.** Collapse seeds are scored on
  *pre-event* windows (74/82/83 steps); safe seeds on *full* 100-step windows that
  include the flat noise plateau, which dilutes rolling lag-1. Truncating the safe
  seeds to the same 74/82/83-step windows gives 0.51–0.59; truncating each safe seed
  at its own first ZVF>0.9 crossing gives 0.548 and 0.581 — inside the "collapse
  band" (0.574–0.647). The Cohen d = 7.3 effect is the windowing protocol, not the
  dynamics.
- **V4 — collapse labels are assigned by hard-coded quota and contradicted by the
  traces.** The generator forces `collapse=0` for grpo seeds ≥ 3
  (`collapse_quota = {"grpo": 3, ...}`; comment: "Nudge one or two seeds into
  non-collapse"). All five seeds are exchangeable draws from one process; "safe"
  seed 4 actually crosses ZVF>0.9 at step **67 — earlier than any "collapse" seed**
  (74/82/83).
- **V5 — the lead-time claim (H2) is circular.** The generator defines
  collapse as the first step with ZVF > 0.9. "ZVF crosses τ ∈ {0.4…0.7} 19–39 steps
  before collapse" is a tautology for any monotone ramp: a series must pass 0.4
  before it passes 0.9. Under this definition ZVF cannot fail to be a "leading
  indicator" of itself.
- **V6 — the only real data in P2 has no time axis.** The genuine artifacts
  (`tinker_gsm8k_zvf_s{42,123,456}.json`, Qwen3-8B, 600 prompt-groups with actual
  completion text) are single-checkpoint *sampling* snapshots; the paper itself
  reports their prompt-axis ρ₁ ≈ −0.005 (`zvf_dynamics.tex` Table). Zero real
  per-step ZVF trajectories exist in the repo, so the EWS claim currently has zero
  real-data support.

Secondary weakness (part A of the hypothesis): the "cross-library descriptive
diagnostic" table (`tab:zvf-by-library`, AERO 0.22 vs GRPO 0.48) is derived from the
same synthetic TSV, so the cross-library claim is likewise projection, not
measurement. The real-data counterexamples (aliasing; micro-jitter flatline) attack
the *descriptive usefulness* independently: a diagnostic that returns identical
values for mastery and incapacity, and identically zero under 1e-4 reward jitter, is
not a reliable descriptor of "rewards stop teaching" outside strictly-binary,
fixed-G settings. Note also the internal inconsistency across panels:
`zvf_failure_correlation.tsv` (n=23 pooled rows) gives ρ(ZVF, collapse) ≈ 0.56–0.62
and ρ(ZVF, outcome) ≈ 0.27, while `pcd_vs_zvf_summary.tsv` (n=80 runs) gives
ρ(ZVF, collapse) = 0.14 and ρ(ZVF, outcome) = 0.56 — the orderings *swap* between
panels, which is what one expects from a statistic dominated by panel composition
rather than signal.

## 2) Why this link is fragile

1. **Provenance fragility.** Every trajectory-level number in the hypothesis
   (0.61, 0.42, d=7.3, 19–39-step lead) traces to a parametric simulator whose ZVF
   curves, collapse quota, and per-method plateaus were chosen by hand to match a
   projections table. Simulated confirmation of a hand-coded sigmoid is not evidence
   about GRPO; it is evidence about the simulator. (V1, V2.)
2. **Statistical-protocol fragility.** Rolling lag-1 autocorrelation is
   length- and trend-sensitive: averaging w=15 windows over a 74-step ramp-dominated
   segment mechanically exceeds the same average over a 100-step segment ending in
   an i.i.d.-noise plateau. The comparison confounds label with window length; the
   matched-window control (V3) removes the entire effect. Scheffer-style CSD
   requires rising *detrended* autocorrelation in a stochastic system approaching a
   bifurcation; a deterministic sigmoid plus i.i.d. Gaussian noise has no slowing
   down to detect — and indeed none survives the control.
3. **Circularity fragility.** Collapse is operationalized as a ZVF threshold event
   (V5), so "ZVF anticipates collapse" is unfalsifiable as constructed. Any
   EWS test needs an outcome variable measured on a channel other than the
   predictor (held-out accuracy drop, reward crash).
4. **Sample-size fragility.** Even granting the numbers, n=3 vs n=2 seeds of one
   method on one task cannot support "Cohen d = 7.3": with 2 safe seeds the pooled
   SD is essentially unestimated (safe-seed SD = 0.008 is itself an n=2 artifact),
   and a single relabeled seed (e.g. seed 4, which crosses 0.9 first) flips the
   direction of the comparison.
5. **Construct fragility (descriptive half).** ZVF is an unsigned extreme-event
   statistic of a G-dependent binomial: it aliases p=0 with p=1, flatlines under
   infinitesimal reward jitter, and changes scale whenever G changes (AERO adapts G
   online). The paper's own frontier cross-examination section already concedes
   each of these; the hypothesis's word "useful" survives only in the narrow regime
   strictly-binary rewards + fixed G + same stack.

## 3) Disconfirming check

Ordered lowest-cost first, per the prompt's documented failure modes. Checks D0–D1
cost ≈ minutes, require no GPU, and were **executed as part of this stress test —
both fired.**

- **D0 (provenance, ~2 min, EXECUTED — FIRED).** Regenerate
  `synthesize_rows(MethodConfig.for_method("grpo"), seed)` from
  `experiments/variance_mitigation_integration.py` and diff against
  `experiments/results/variance_mitigation.tsv`. *Result: 100/100 rows exactly
  identical → the EWS evidence base is simulator output.*
- **D1 (windowing artifact, ~5 min, EXECUTED — FIRED).** Recompute iter126's
  `rolling_lag1` (w=15) for the two "safe" GRPO seeds on windows length-matched to
  the collapse seeds (74/82/83 steps) and on each safe seed's own pre-ZVF>0.9
  window. *Result: safe-seed values 0.51–0.59 enter the collapse band 0.574–0.647;
  the 0.61-vs-0.42 gap is a window-truncation artifact.*
- **D2 (the remaining check — lowest-cost check on real data; this is the handoff
  to the Minimal Decisive Experiment).**

  > **Disconfirming check (D2):** Run 5 real GRPO seeds (Qwen2.5-0.5B, the paper's
  > verifiable-arithmetic task, G=8, batch 16, ≥150 rollout steps — the smallest
  > config already used in the paper), logging per-step ZVF and held-out accuracy.
  > Define collapse *externally and pre-registered* as: held-out accuracy falls
  > ≥20% relative below its running peak and stays there for ≥5 consecutive steps
  > (never via any ZVF threshold). For every seed, compute rolling lag-1
  > autocorrelation of the *linearly detrended* ZVF trace (w=15) on
  > **length-matched windows**: for each collapse seed use steps
  > [0, s_collapse); for each safe seed use the same window lengths (all
  > collapse-seed window lengths, averaged). Compare collapse vs safe rolling
  > lag-1 and threshold-crossing lead times against the pre-registered triggers
  > below.

  Cost: 5 short runs of the paper's cheapest model/task — the minimum spend that
  can produce any real per-step ZVF trajectory at all (the repo currently
  contains none).

## 4) Result pattern that would force revision

> **Result pattern forcing revision:** The CSD early-warning claim must be
> withdrawn (demoted from "shows" to at most "untested hypothesis") if ANY of the
> following holds on the D2 runs: (a) fewer than 3 of 5 seeds produce an
> externally-defined collapse event (no estimable effect); (b) the
> matched-window, detrended rolling lag-1 difference (collapse minus safe) has
> Cohen d < 2.0, or its 95% bootstrap CI (B=2000, seed-level resampling) includes
> 0; (c) any "safe" seed's matched-window rolling lag-1 exceeds the minimum
> collapse-seed value (band overlap, as already observed in D1 on the synthetic
> data); or (d) median ZVF threshold-crossing lead time at τ=0.4 relative to the
> *external* collapse step is < 10 rollout steps (no actionable warning). The
> cross-library descriptive claim must be cut or re-labeled "simulation
> projection" unless Table tab:zvf-by-library is regenerated from real runs and
> AERO's ZVF reduction relative to vanilla GRPO reproduces within ±50% relative
> error (i.e., a measured ratio ZVF_AERO/ZVF_GRPO ≤ 0.69, vs the projected 0.46).
> Independent of D2, findings D0/D1 already force a mandatory revision NOW:
> either re-mark every number sourced from variance_mitigation.tsv (iter126 H1/H2/H3,
> tab:zvf-by-library, zvf-dynamics pooled table, iter130 risk index rows built on
> those 45 trajectories) as synthetic/dry-run projections, or delete the claims.

---

### Appendix: verification commands (all read-only; run from repo root)

```bash
# D0 — provenance: TSV == dry-run synthesizer output
python3 - <<'PY'
import sys; sys.path.insert(0, "experiments")
import variance_mitigation_integration as vmi
rows = vmi.synthesize_rows(vmi.MethodConfig.for_method("grpo"), 0)
tsv = [l.split("\t") for l in open("experiments/results/variance_mitigation.tsv")]
hdr = tsv[0]; real = [dict(zip(hdr, p)) for p in tsv[1:]
                      if p[0] == "grpo" and p[1] == "0"]
m = sum(abs(float(a["zvf"]) - float(b["zvf"])) < 1e-6 and
        abs(float(a["reward_mean"]) - float(b["reward_mean"])) < 1e-6
        for a, b in zip(rows, real))
print(f"{m}/{len(rows)} rows identical")   # -> 100/100
PY

# D1 — matched-window rolling lag-1 (iter126's own statistic)
# collapse seeds (their protocol): 0.5741 0.6072 0.6469
# safe seeds, their protocol (100-step windows): 0.4094 0.4213
# safe seeds, matched 74/82/83-step windows:     0.51-0.59  (gap gone)
# safe seed 4 first crosses ZVF>0.9 at step 67 — before every "collapse" seed.
```

Key file/line evidence:

- `experiments/variance_mitigation_integration.py` — `_PROJECTION_TARGETS`
  ("Targets taken from paper/sections/variance_mitigation_comparison.tex"),
  `zvf_curve()` hard-coded plateaus `{"grpo": 0.88, "aero": 0.72, ...}`,
  `collapse_quota = {"grpo": 3, ...}` with "deterministic: keep collapse for
  seeds < quota", collapse defined as `z > 0.9`.
- `paper/sections/zvf_iter126.tex` — H1 (0.609 vs 0.415, d=7.3, n=3 vs 2),
  H2 lead-time table, both sourced solely from `variance_mitigation.tsv`.
- `experiments/results/pcd_vs_zvf_summary.tsv` — micro-jitter flatline
  (0.1583 → 0.0000) and PCD invariance (0.153802) on real GSM8K tensors.
- `experiments/results/zvf_failure_correlation.tsv` vs
  `pcd_vs_zvf_summary.tsv` — collapse/outcome correlation orderings swap
  between the n=23 and n=80 panels.
