# Colab experiment results

Persisted from `colab run` stdout (job b58dzsrjs). Raw log: `colab_run_b58dzsrjs.log`.

| Exp | Status | Headline |
|-----|--------|----------|
| E1 grad-signal | done | corr(grad_norm, p(1-p)) = 0.71 (validates Theory T3) |
| E2 LoRA vs full-FT | FAILED | torchao 0.10.0 incompatible (needs >0.16); rerun pending |
| E3 open audit | done | DAPO drove ZVF->0 (+45% rollouts); adaptive-G + Dr.GRPO best held-out |

Toy 0.5B model on synthetic arithmetic -- directional evidence, not publishable effect sizes.
Also logged to W&B project `zvf-colab-experiments`.
