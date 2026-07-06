# Improvement 88 — P7 exact finite-pool (hypergeometric) contrast-preservation: the binomial G* ceiling is not the ceiling

| field | value |
| --- | --- |
| pillar | **P7** (ZVF theory → adaptive-G controller) |
| target | `paper/sections/p7_controller.tex` §4.10bis "Exact finite-pool correction" (NEW) + inline correction to Table~\ref{tab:p7-per-prompt-optimal} caption |
| class | **T2** fresh-data evidence (exact combinatorics on real N2 rollout pools) + **T1** statistical rigor (bootstrap CIs on the correction) + **T3** cross-paper coupling (grounds the frontier δ_div in exact enumeration) |
| status | **validated** (N2 four-method, 40 steps × 16 prompts × 4 methods = 2,560 prompt-steps; exact formula checked vs brute-force enumeration, max err 1.1e-16) |
| artifact | `scripts/p5p8/p7_exact_finite_pool_g.py` (≤230 LoC, **stdlib only**) |
| evidence | `experiments/results/p5p8/p7_exact_finite_pool_{per_prompt.tsv, summary.tsv, summary.json}` |

## 1. Question (fresh vein — not in prior 87 rows)

The in-paper per-prompt $G^*$ analysis (iter-47, §4.10,
`p7_per_prompt_optimal_g`) scores each candidate reduced group size
$G'$ with the **i.i.d. binomial** collision model
$\mathrm{CP}_\text{binom}(G'\!\mid p)=1-(p^{G'}+(1-p)^{G'})$, $p=k/8$,
and its Table caption calls the result the *"strict efficiency ceiling
**under the i.i.d. binomial model**."*

But the honest counterfactual for *"what if we had only rolled out $G'$
of the 8 samples"* is to **subsample $G'$ of the actual 8 rewards** —
sampling **without replacement** from a finite pool. Its exact
contrast-preservation probability is **hypergeometric**, not binomial:

$$\mathrm{CP}_\text{exact}(G'\mid k, N) = 1 - \frac{\binom{k}{G'}+\binom{N-k}{G'}}{\binom{N}{G'}},\quad \binom{k}{G'}=0\ \text{for}\ k<G'.$$

The frontier synthesis (Round 2, Gemini/ChatGPT) argues observed ZVF
under-predicts the i.i.d. baseline because finite-pool sampling
**anti-herds** ($\delta_\text{div}>0$). This iteration makes that exact:
$\mathrm{CP}_\text{exact}\ge\mathrm{CP}_\text{binom}$ always (a finite pool
can never collide *more* than i.i.d.), so the binomial model
**over-predicts starvation** and the iter-47 "ceiling" is loose.

## 2. Method

- Load the 4 N2 tensor files (each 40 steps × 16 prompts × 8 binary rollouts).
- For each prompt-step: $k=\sum$ rewards out of $N=8$.
- **Validation gate:** for the first 8 real prompts, brute-force
  $\mathrm{CP}$ by enumerating all $\binom{8}{G'}$ subsets and assert
  it equals the closed form. Max error $=1.1\times10^{-16}$ (pass).
- Two per-prompt Iso-G allocators, both = *smallest $G'\in\{2..7\}$ with
  $\mathrm{CP}\ge\tau_c$, else keep $G{=}8$*: one scored with
  $\mathrm{CP}_\text{binom}$ (iter-47's model), one with
  $\mathrm{CP}_\text{exact}$. Ground-truth preserved contrast is always
  scored with $\mathrm{CP}_\text{exact}$ (the model-free truth).
- Bootstrap over the 40 steps ($B=2000$, seed 20260705) for CIs on the
  extra rollouts the exact allocator saves.

## 3. Headline results (measured 2026-07-05)

**(a) Exact per-prompt anti-herding bonus.** Over the 693 non-degenerate
(mixed) prompt-steps, at $G'{=}2$:
$$\mathrm{CP}_\text{exact}-\mathrm{CP}_\text{binom}=\mathbf{+0.0494\ [+0.0481,+0.0506]}$$
(pooled mean, bootstrap 95% CI). This is $\delta_\text{div}$ measured by
**exact enumeration**, not observed-vs-model residual — a per-prompt,
model-free confirmation of the frontier's finite-pool anti-herding claim.
Consequence: iter-47 reports these 693 prompts' economized contrast with
the binomial model, understating the true preserved contrast by
$693\times0.0494\approx\mathbf{34.2}$ ZVF-units — contrast it books as
"lost to $G^*$" that the exact accounting shows is **preserved**.

**(b) The binomial ceiling leaves rollouts on the table.** Rebuilding the
allocator with exact scoring at a meaningful target contrast $\tau_c$:

| $\tau_c$ | cost ratio (binom) | cost ratio (exact) | extra saving | phantom-starved recovered |
| --- | --- | --- | --- | --- |
| 0.3 | 0.811 | 0.811 | 0.0% (0) | 0 |
| **0.5** | 0.868 | **0.834** | **3.46% (709 rollouts)** | 0 |
| **0.7** | 0.918 | **0.881** | **3.66% (749 rollouts)** | **287** |

At $\tau_c{=}0.5$ the extra saving is from the exact model choosing a
*smaller* $G'$ for the same prompts; at $\tau_c{=}0.7$ it additionally
**recovers 287 "phantom-starved" prompts** — prompts the binomial model
wrongly holds at $G{=}8$ (believing no $G'{<}8$ preserves $\ge0.7$
contrast) that the exact accounting can safely drop. Per-method the
extra-saving bootstrap CI **excludes zero for all four methods** at both
$\tau_c$ (grpo +201 [175,229], aero +182 [154,211], gift +158 [123,193],
areal +208 [173,246] at $\tau_c{=}0.7$).

## 4. What this changes about the paper

The iter-47 Table~\ref{tab:p7-per-prompt-optimal} caption should not call
the binomial $G^*$ the "strict efficiency ceiling" without qualification:
the true (exact finite-pool) ceiling is **tighter** ($0.834$ vs $0.868$ at
$\tau_c{=}0.5$). The correction is small in absolute compute (3–4%) but
**directional and exact**, and it converts the frontier's qualitative
"anti-herding" argument into a per-prompt number with a CI. It also
sharpens the honesty note: the "zero contrast restored" headline is a
property of $G{=}8$ saturation, *not* of the scoring model — but the
"contrast lost to $G^*$" number (0.130 ZVF/prompt) is model-dependent and
biased high by the binomial approximation.

## 5. Verdict

The exact hypergeometric accounting is the correct counterfactual for
subsampling a fixed rollout pool. It **strengthens** (does not overturn)
the iter-47 result: the per-prompt controller is a genuine Pareto
economy, and the achievable economy is slightly larger and the
anti-herding bonus is exactly quantified. No fabricated citation used
(reuses `su2024dualformer`, `alphaproof2025nature` already in
`references.bib`; frontier synthesis attributed inline).

## 6. Reproduction

```bash
python3 scripts/p5p8/p7_exact_finite_pool_g.py
# stdlib only, ~1 min; validates exact formula vs brute-force enumeration,
# then computes both allocators over 2,560 prompt-steps with B=2000 bootstrap.
```
Outputs: `experiments/results/p5p8/p7_exact_finite_pool_per_prompt.tsv`
(30,720 rows: 2,560 prompt-steps × 3 τ_c × ... method/step/prompt/k/g_exact/g_binom/cp),
`p7_exact_finite_pool_summary.tsv` (12 rows: 4 methods × 3 τ_c),
`p7_exact_finite_pool_summary.json` (full machine-readable summary).
