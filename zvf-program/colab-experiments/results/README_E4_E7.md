# Colab experiment results — E2 rerun + E4–E7 batch

New Colab-only batch (each requires a capability closed/LoRA-only/fixed-stack **Tinker** lacks), one per ZVF-Program pillar. Codex (gpt-5.5) reviewed the plan before launch; fixes in `PLAN_E4_E7.md`. Toy 0.5B on synthetic arithmetic — directional evidence, not publishable effect sizes. Logged to W&B `zvf-colab-experiments`.

| Exp | Pillar | Status | Headline |
|-----|--------|--------|----------|
| E2 | P4 | done | LoRA Δ=+0.50 vs full-FT Δ=+0.46 held-out |
| E4 | P1 | done | ZVF(p,K)=p^K+(1-p)^K fits R²=0.9995; fp32 moves ZVF by -0.035 |
| E5 | P2 | done | signal↔p(1-p) r=0.681 > signal↔GU r=0.435; Fisher↔p(1-p) r=0.889 |
| E6 | P3 | done | fixed Δ=0.33/ZVF=0.29 | adaptiveG Δ=0.33/ZVF=0.30 | +drop Δ=0.30/ZVF=0.33 (matched ~600 rollouts) |
| E7 | P4 | done | stack-lever ΔZVF vs ref — fp32: ΔZVF=+0.29±0.16; eager_attn: ΔZVF=-0.05±0.03; temp_0.7: ΔZVF=-0.14±0.01 |
