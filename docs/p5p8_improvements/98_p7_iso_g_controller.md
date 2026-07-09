# 98 — Iter-83 P7 Iso-Yield Dynamic Grouping (Iso-G) controller prototype

**Pillar:** P7 (Pillar 3 — adaptive-G controller for GRPO group-size starvation)

**Vein (fresh, not in 97 prior rows):** the **frontier synthesis**
(`FRONTIER_INSIGHTS.md` Round 2 — Gemini Deep Think) proposed "**Iso-Yield
Dynamic Grouping (Iso-G)** — Abandon [static G]". The earlier P7 controllers
(\texttt{zvf-triage}, Dualformer-Auto, Hybrid, Bayesian) all act on a
*step-level signal* (per-step zvf, per-prompt $\hat p$, etc.) and pick
$G'$ from a fixed menu. **Iso-G acts directly on the per-prompt
contrastive yield** $Y(\hat p, G') = 1 - \mathrm{ZVF}_{\text{iid}}(\hat p, G')$
and chooses the smallest $G' \in \{2, 4, 6, 8, 10, 12, 16, 24, 32\}$ that
achieves $Y \ge \tau_y$ for that specific prompt. This is the **exact
operationalisation** of the frontier synthesis's "Concrete Invention".

## Setup

N2 four-method same-stack corpus
(`experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl`):
40 steps × 16 prompts × $G{=}8$ binary rewards = **2560 prompt-step
decisions**. For each (method, step, prompt) with $k$ successes at
$G_{\text{base}}{=}8$, the Iso-G controller picks:
$$G^\star = \min\{G' \in \{2,4,6,8,10,12,16,24,32\} : Y(k/8, G') \ge \tau_y\}$$
with $\tau_y \in \{0.50, 0.70, 0.90, 0.95\}$. Default to $G_{\text{base}}$
if no $G'$ achieves the target.

Five reference controllers on the same 2560 obs:
- **C0 fixed-G=8** (baseline)
- **C1 zvf-triage@$\tau{=}0.70$** (escalate to $G{=}16$ when step-zvf $\ge$ 0.70)
- **C2 Dualformer-Auto** (Berkeley row-01 per-prompt $\hat p$-gated $G \in \{2,4,8,16\}$)
- **C3 Hybrid** ($\tau{=}0.70$, $\delta{=}0.20$: escalate boundary band $[0.70, 0.90)$, de-escalate saturation band $\ge 0.90$)
- **C4 Bayesian@$\tau_{\text{post}}{=}0.60$** (per-prompt Beta-Binomial mid-range probability)
- **C5 Iso-G** (this iteration, four $\tau_y$ values)

**Yield-restored metric**: $\Delta Y_{\text{prompt}} = Y(\hat p, G_{\text{ctrl}}) - Y(\hat p, G_{\text{base}})$.
Total $\Delta Y$ across 640 prompts (per method) is the contrast-restored
sum; **yield-per-1000-extra-rollouts** is the cost-efficiency metric.

## Headline (validated, this iteration)

### H1 — Iso-G@0.90 Pareto-dominates every other controller on yield-per-1000-extra-rollouts

| method | C1 zvf-triage | C3 Hybrid | C4 Bayesian | **C5 Iso-G@0.90** | Iso-G / C1 ratio |
| --- | --- | --- | --- | --- | --- |
| grpo | 3.24 | 3.61 | 4.38 | **19.42** | **6.0×** |
| aero | 2.89 | 3.07 | 4.00 | **19.61** | **6.8×** |
| gift | 1.99 | 2.47 | 3.47 | **19.49** | **9.8×** |
| areal | 2.77 | 2.89 | 4.63 | **19.33** | **7.0×** |

**At $\tau_y = 0.90$ the Iso-G controller restores 6–10× more contrast
per extra rollout than the best existing controller (zvf-triage) on
every method**. This is the **first falsifiable evidence** that
**per-prompt yield-targeted grouping** strictly Pareto-dominates
per-step signal-targeted grouping on the cost-efficiency axis.

### H2 — Bootstrap CIs on the Iso-G / zvf-triage ratio (95% percentile, $B{=}2000$, seed 20260705)

Step-level resample of the 40-step trajectories:

| method | Iso-G@0.90 yield/1k | zvf-triage yield/1k | **Ratio (Iso-G / zvf)** | 95% CI excludes 1.0? |
| --- | --- | --- | --- | --- |
| grpo | 19.43 [19.28, 19.60] | 3.24 [2.60, 3.89] | **6.06×** [4.99×, 7.50×] | YES |
| aero | 19.63 [19.37, 20.00] | 2.89 [2.29, 3.53] | **6.88×** [5.54×, 8.63×] | YES |
| gift | 19.51 [19.27, 19.84] | 2.00 [1.49, 2.57] | **9.97×** [7.57×, 13.21×] | YES |
| areal | 19.33 [19.19, 19.53] | 2.76 [2.14, 3.34] | **7.10×** [5.78×, 9.09×] | YES |

**All four 95% CIs on the ratio exclude 1.0 with lower bounds > 4.9×** —
the Pareto dominance is **statistically detectable** on every method.

### H3 — Iso-G@0.50 (aggressive de-escalation) is the cheapest controller but signal-harmful

| method | cost ratio | n_deescalated | total $\Delta Y$ |
| --- | --- | --- | --- |
| grpo | **0.881** | 179 | **−34.21** |
| aero | **0.874** | 179 | **−37.94** |
| gift | **0.900** | 147 | **−29.52** |
| areal | **0.878** | 188 | **−34.38** |

The aggressive Iso-G@0.50 (target yield 0.50, smallest $G'$ that achieves
it) **saves 10–13% rollouts** vs fixed-G=8 baseline (lowest cost ratio
in the controller bank), but **destroys ~30–38 units of total contrast
across the 640 prompts**. This is the **clean negative finding**: at
$\tau_y = 0.50$ the controller de-escalates too aggressively (G=2 on
mid-range prompts $k \in \{2,3,4,5,6\}$ has yield $\le 0.5$ but loses
$G{=}8$'s near-perfect yield), so the saving is paid in contrast.

### H4 — Iso-G@0.95 ≈ Iso-G@0.90 in cost-efficiency; $\Delta Y$ monotonically increases in $\tau_y$

| method | Iso-G@0.90 | Iso-G@0.95 | $\Delta$ |
| --- | --- | --- | --- |
| grpo | 19.42 / 1k, $\Delta Y=23.23$ | 18.89 / 1k, $\Delta Y=25.73$ | slight regression on ratio, gain on $\Delta Y$ |
| aero | 19.61 / 1k, $\Delta Y=20.59$ | 19.04 / 1k, $\Delta Y=23.01$ | same pattern |
| gift | 19.49 / 1k, $\Delta Y=18.28$ | 18.98 / 1k, $\Delta Y=20.27$ | same pattern |
| areal | 19.33 / 1k, $\Delta Y=25.05$ | 18.95 / 1k, $\Delta Y=27.70$ | same pattern |

**$\tau_y = 0.90$ is the knee**: it is the lowest $\tau_y$ where the
controller's de-escalation branch is *not* contrast-harmful on the
mid-range prompts ($k \in \{3,4,5\}$ at $G' \in \{4, 6\}$ achieve
$Y \ge 0.90$ without going to $G{=}2$).

## Mechanism: why Iso-G dominates

The existing controllers act on **step-level or per-prompt signal**
(zvf, $\hat p$, mid-range prob) and pick $G'$ from a **fixed binary or
ternary menu** (escalate or don't, de-escalate or don't). They ignore
the **continuous trade-off** in $G'$: a prompt with $k=4$ at $G{=}8$
(i.e., $\hat p = 0.5$) has yield $1 - 2 \cdot 0.5^8 = 0.992$ at
$G{=}8$ and yield $1 - 2 \cdot 0.5^4 = 0.875$ at $G{=}4$. The existing
controllers would either keep $G{=}8$ (Dualformer) or escalate to
$G{=}16$ (zvf-triage). Iso-G@0.90 picks $G' = 6$ (yield = 0.984, just
above the 0.90 target) — saving 2 rollouts vs $G{=}8$ while preserving
near-perfect contrast. **This is the per-prompt granularity the existing
controllers structurally cannot access.**

## Cross-paper coupling

(i) **P6 iter-82 row 97** — the iter-82 window-sensitivity audit
established that registry-claimed deltas are sensitive to the
window; the Iso-G controller's per-prompt $G'$ choice is window-free
(deterministic from observed $k$). This is the natural extension of
iter-82's measurement-vs-claim audit into the **per-prompt decision**
domain.

(ii) **P7 iter-71 row 83 (per-prompt Dualformer reproduce)** —
Dualformer's $G' \in \{2, 4, 8, 16\}$ menu is **a strict subset** of
Iso-G's $\{2, 4, 6, 8, 10, 12, 16, 24, 32\}$ menu, and Dualformer's
$\hat p$-gated rule is **a single iso-yield curve** (yield = 0.75 at
the threshold). Iso-G's continuous-$\tau_y$ family strictly subsumes
Dualformer-Auto; the iter-83 row 98 6–10× improvement on
yield-per-1k-extra is the cost-efficiency version of iter-71's
0.21-vs-0.562 saving discrepancy.

(iii) **P7 iter-83 frontier synthesis connection** — the
frontier synthesis (Gemini Deep Think) wrote: *"Concrete Invention:
Iso-Yield Dynamic Grouping (Iso-G). Mechanism: Abandon [static G]"*.
Iter-83 row 98 is the **first prototype** of this invention on real
N2 reward tensors, with measured 6–10× cost-efficiency gain over
step-level controllers.

## Operational recommendation

For prompt distributions that carry a mix of boundary and mid-range
$k$ values (i.e., where some prompts are not fully saturated),
**Iso-G@0.90 is the Pareto-dominant controller on
yield-per-1000-extra-rollouts**. Onthe N2 four-method corpus
(72.9% boundary prompts, 27.1% mid-range) the gain is 6–10× over
zvf-triage and 5–7× over Hybrid, with 95% CIs that exclude 1.0 on
every method. **For fully-saturated regimes** (e.g., iter-31 sat-band
panels) Iso-G's de-escalation branch fires on every prompt and the
controller degenerates to Dualformer-Auto.

## Reproduction

```bash
python3 scripts/p5p8/p7_iter83_iso_g.py
```

Outputs:
- `experiments/results/p5p8/p7_iter83_iso_g_per_prompt.tsv` (23040 rows = 9 controllers × 2560 obs)
- `experiments/results/p5p8/p7_iter83_iso_g_per_method.tsv` (36 rows = 9 controllers × 4 methods)
- `experiments/results/p5p8/p7_iter83_iso_g_summary.json`

## Status

validated — n=4 methods × 9 controllers × 2560 prompt-step decisions,
±300 LoC pure-stdlib script, paired-step bootstrap $B{=}2000$, seed 20260705.
All four Pareto-dominance 95% CIs exclude 1.0.