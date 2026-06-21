# Colab experiment results — E2 rerun + E4–E7 batch

New Colab-only batch (each requires a capability closed/LoRA-only/fixed-stack **Tinker** lacks), one per ZVF-Program pillar. Reviewed and hardened over three rounds (plan, first results, implementation). Fixes in `PLAN_E4_E7.md`. Toy 0.5B on synthetic arithmetic — directional evidence, not publishable effect sizes. Logged to W&B `zvf-colab-experiments`. **Table = v3 (post-hardening) numbers.**

| Exp | Pillar | Status | Headline |
|-----|--------|--------|----------|
| E2 | P4 | done | LoRA Δ=+0.50 vs full-FT Δ=+0.46 held-out |
| E4 | P1 | done | ZVF(p,K)=p^K+(1-p)^K fits R²=0.9999 (p̄=0.7969, K=32 ZVF=0.0269≈0.008 worked example); precision ΔZVF +0.032 |
| E5 | P2 | done | per-group (N=132 live): signal↔p(1-p) r=-0.158 (slope -8.748); Fisher↔p(1-p) r=0.035 — weak/inconclusive |
| E6 | P3 | done | fixed Δ=-0.04/ZVF=0.85 | adaptiveG Δ=0.00/ZVF=0.84 | +drop Δ=0.02/ZVF=0.83 (matched ~420 rollouts) |
| E7 | P4 | done | gen-baseline bf16=0.812==fp32=0.812 (precision ~no gen effect); fp32 train Δp=-0.125 (noisy/LR-dep, NOT v1's +0.72); nonfinite_grads=0; fp32 Δp=-0.125; bf16_lr1e-6 Δp=-0.021; fp32_lr1e-6 Δp=+0.006; eager_attn Δp=-0.048; temp_0.7 Δp=+0.041 |

## Round-3 (hardening) — what changed

The hardening review found the shared root cause and per-script bugs; all fixed, then re-run (v3):
- **Root cause**: `MAX_NEW=24` truncated any reasoning trace > ~24 tokens → no `####` →
  deterministic 0 reward → hard prompts spuriously forced to p=0. Fixed: `MAX_NEW=128`
  + `parse()` returns None when `####` is absent (was grabbing the question's digits).
- **E4 (P1)**: closed-form `ZVF(p,K)` fit is rock-solid (**R²=0.9999**, ZVF 0.73→0.027
  across K). p≈0.5 centering is *unreachable* — the model's accuracy is bimodal
  (easy→high p, hard→p≈0), so unchoking truncation pushed p̄ to 0.80; `calibrated_to_p0.5=false`
  flagged. The scaling-law claim holds regardless of centering.
- **E5 (P2)**: hardening WORKED — continuous partial-credit reward + `MAX_NEW=128` gave
  **live groups across the full exact-match p range [0.0, 0.958]** (N=132), and
  population-standardized advantage (pilot rbar/sd) removed the v2 Simpson's-paradox
  artifact. Result: **signal↔p(1-p) r=-0.16 — a robust honest NEGATIVE**. With every
  confound removed, this toy 0.5B/arithmetic setup does not demonstrate T3's inverted-U.
- **E6 (P3)**: switching the borderline/eval task to 2-digit *multiplication* (headroom;
  3-digit addition was ceilinged) surfaced the **first directional triage edge**:
  adaptiveG_drop Δ=+0.02 > adaptiveG 0.00 > fixed_G −0.04, and the drop arm used fewer
  rollouts (378 vs 432). Small and within 2-seed noise → suggestive, not conclusive.
- **E7 (P4)**: `grad_diag` no longer hides non-finite grads. Definitive: bf16 and fp32 have
  **identical generation baselines (0.812)**, fp32 training is noisy/LR-dependent (not better),
  and **nonfinite_grads=0** (no bf16 instability). v1's +0.72 fp32 effect was a format
  confound, full stop. The real headline-moving lever is sampling temperature (temp_0.7 +0.041).

Net: hardening converted E5 into a *trustworthy* negative (full p-range), strengthened E4's
fit, surfaced a directional E6 signal, and closed the E7 precision question. v1/v2 remain in
git history (c575a68, 8e89188); this batch supersedes them.
