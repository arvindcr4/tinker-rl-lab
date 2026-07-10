# Semester 4 Paper and Evidence Map

The PDF copies in `papers/` are review artifacts. The LaTeX sources and evidence remain in their canonical repository paths so there is only one editable source of truth.

| Paper | Research focus | Canonical source | Principal evidence area |
|---|---|---|---|
| P1 | Cross-library and cross-scale GRPO behavior | [`../paper/paper_P1_scaling.tex`](../paper/paper_P1_scaling.tex) | [`../experiments/results/`](../experiments/results/) and scaling sections under [`../paper/sections/`](../paper/sections/) |
| P2 | Zero-Variance Fraction as a signal-starvation diagnostic | [`../paper/paper_P2_zvf.tex`](../paper/paper_P2_zvf.tex) | ZVF result tables in [`../experiments/results/`](../experiments/results/) |
| P3 | Group size, contrast density, and relationship to DPO | [`../paper/paper_P3_group_size.tex`](../paper/paper_P3_group_size.tex) | Group-size sweeps in [`../experiments/results/`](../experiments/results/) |
| P4 | Length bias and held-out generalization | [`../paper/paper_P4_length_bias.tex`](../paper/paper_P4_length_bias.tex) | Length-bias and held-out outputs in [`../experiments/results/`](../experiments/results/) |
| P5 | Minimum reporting standard for stack-conditioned results | [`../paper/paper_P5_minreport.tex`](../paper/paper_P5_minreport.tex) | P5 analysis notes under [`../docs/p5p8_improvements/`](../docs/p5p8_improvements/) and P5 outputs under [`../experiments/results/p5p8/`](../experiments/results/p5p8/) |
| P6 | Machine-readable GRPO stack registry | [`../paper/paper_P6_registry.tex`](../paper/paper_P6_registry.tex) | Registry schemas, validators, and P6 outputs under [`../experiments/results/p5p8/`](../experiments/results/p5p8/) |
| P7 | Adaptive group-size controller based on signal starvation | [`../paper/paper_P7_zvf_controller.tex`](../paper/paper_P7_zvf_controller.tex) | Controller simulations and audits under [`../experiments/results/p5p8/`](../experiments/results/p5p8/) |
| P8 | LLM evidence extraction versus XGBoost for fraud decisions | [`../paper/paper_P8_fraud.tex`](../paper/paper_P8_fraud.tex) | Fraud data/scripts at repository root and P8 outputs under [`../experiments/results/p5p8/`](../experiments/results/p5p8/) |

## Inherited versus new

Inherited from Semester 3:

- The multi-framework repository and baseline integrations.
- Initial Tinker/SkyRL/GRPO experiments and the group literature foundation.
- The early benchmark, evaluation, and reproducibility scaffolding.

Added in Semester 4:

- The eight standalone P1–P8 research papers and their current authorship.
- Post-capstone experiment, audit, synthesis, and paper iterations.
- Expanded scaling, ZVF, group-size, length-bias, reporting, registry, controller, and fraud analyses.
- Berkeley-course-derived research audits and prototypes.

This distinction is about academic contribution. It does not duplicate or relocate the shared runtime code.
