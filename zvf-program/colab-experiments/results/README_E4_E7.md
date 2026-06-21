# Colab experiment results — E2 rerun + E4–E7 batch

New Colab-only batch (each requires a capability closed/LoRA-only/fixed-stack **Tinker** lacks), one per ZVF-Program pillar. Codex (gpt-5.5) reviewed the plan before launch; fixes in `PLAN_E4_E7.md`. Toy 0.5B on synthetic arithmetic — directional evidence, not publishable effect sizes. Logged to W&B `zvf-colab-experiments`.

| Exp | Pillar | Status | Headline |
|-----|--------|--------|----------|
| E2 | P4 | done | LoRA Δ=+0.50 vs full-FT Δ=+0.46 held-out |
| E4 | P1 | done | ZVF(p,K)=p^K+(1-p)^K fits R²=0.9999 (p̄=0.7199, K=32 ZVF=0.0082≈0.008 worked example); precision ΔZVF +0.009 |
| E5 | P2 | done | signal↔p(1-p) r=-0.658, ↔GU r=-0.641, Fisher↔p(1-p) r=0.001 — INCONCLUSIVE — p-tails have ~no live groups to measure |
| E6 | P3 | done | fixed Δ=0.00/ZVF=0.87 | adaptiveG Δ=0.00/ZVF=0.78 | +drop Δ=0.00/ZVF=0.78 (matched ~420 rollouts) |
| E7 | P4 | done | fp32 Δp_vs_ref=-0.007 (v1's +0.72 was a format confound, now ~0); nonfinite_grads=0; fp32 Δp=-0.007; bf16_lr1e-6 Δp=-0.048; fp32_lr1e-6 Δp=+0.014; eager_attn Δp=-0.014; temp_0.7 Δp=+0.014 |

## Round-2 (Codex re-review of the *results*) — what changed

Codex reviewed the first-run scripts+results and flagged all four as untrustworthy
(format-gated p, confounded effects). The v2 fixes (few-shot scaffold + empirical
p-calibration + decoupling diagnostics) **overturned two of the four first-run
"wins" — which is the point of the review:**

- **E4 (P1): strengthened.** With a meaningful p-range the closed form fits R²=0.9999
  and empirical ZVF reaches the 0.008 worked-example scale at K=32. (Calibration
  overshot to p̄=0.72, not 0.5; `calibrated_to_p0.5=false` flagged.)
- **E5 (P2): corrected to INCONCLUSIVE.** v1's +0.68 signal↔p(1-p) was an artifact of
  a compressed p-range. Properly calibrated, the correlation is not robust because
  the p-tails (small p(1-p)) have ~no live groups to measure — the inverted-U is
  largely *unmeasurable* with live-group gradients (which is itself the ZVF story).
  Next step: per-group regression of signal on local p̂(1-p̂) instead of 5 bin means.
- **E6 (P3): mechanism confirmed, no held-out payoff.** With a genuinely triage-relevant
  pool (75% dead 3-digit *multiplication*, 25% learnable addition) the controller
  drops dead prompts and lowers ZVF (0.87→0.78), but no arm gains held-out accuracy
  at 420 rollouts — triage efficiency does not convert to accuracy at this toy scale.
- **E7 (P4): corrected.** v1's dramatic bf16→fp32 jump (+0.72 train reward) was a
  FORMAT-COMPLIANCE confound. With the few-shot scaffold fixing format, fp32 Δp≈-0.007
  and nonfinite_grads=0 (no bf16 instability). The real headline-moving unreported
  lever was the prompt scaffold/parser, not precision — which still supports
  MIN-REPORT-RL, just relocates the lever.

v1 results remain in git history (commit c575a68); this batch supersedes them.
