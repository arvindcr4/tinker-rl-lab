# Limitations and Broader Impact

> Required by NeurIPS Paper Checklist Items 2, 10, and 11.

## Limitations

### Model Scale

All experiments are conducted on models ranging from 0.6B to 30B parameters. Results may not generalize to larger models (70B+) where training dynamics, LoRA effectiveness, and optimization landscapes differ significantly. The scaling analysis (3B → 30B) provides partial evidence of trends but cannot be reliably extrapolated.

### Task Coverage

The benchmark focuses primarily on:
- **Mathematical reasoning**: Arithmetic, GSM8K
- **Response style**: Preference learning for conciseness
- **Knowledge transfer**: Distillation from larger models

This does not cover important RL post-training domains such as:
- Open-ended creative writing
- Multi-turn dialogue
- Code generation
- Safety alignment (RLHF with human preferences)
- Tool use and agentic behavior

### Platform Dependency

The Tinker API experiments require access to the Tinker platform (https://thinkingmachines.ai/tinker). While we provide TRL-based standalone implementations for all core experiments, the API-based training loop (Atropos + Tinker) cannot be independently reproduced without Tinker API access. This limits full reproducibility for the platform-specific claims.

### Fine-Tuning Method

All experiments use LoRA (Low-Rank Adaptation) rather than full fine-tuning. While LoRA is the dominant method for practical LLM training, our results do not speak to whether full fine-tuning would yield different relative comparisons between methods or libraries.

### Statistical Limitations

- We use 5 seeds per experiment. While this exceeds the typical 3 seeds in most published RL work, Patterson et al. (2024) recommend 10+ seeds for reliable comparisons. Compute constraints limited us to 5.
- Bootstrap confidence intervals assume i.i.d. samples across seeds. Correlations due to shared initialization or data ordering could narrow true confidence intervals.
- For the scaling analysis (Qwen 30B MoE), only 3 seeds were used due to compute constraints.

### Hyperparameter Sensitivity

We did not conduct exhaustive hyperparameter sweeps. Hyperparameters were selected based on defaults from the Tinker cookbook and TRL documentation. A more thorough ablation study of learning rate, LoRA rank, batch size, and group size would strengthen the empirical contribution.

---

## Broader Impact

### Positive Impacts

- **Democratizing RL post-training research**: By providing cross-library implementations with standardized evaluation, this work lowers the barrier to entry for researchers exploring RL fine-tuning of language models.
- **Improving reproducibility standards**: The statistical analysis toolkit and multi-seed evaluation protocol directly address the RL reproducibility crisis documented by Henderson et al. (2018) and Pineau et al. (2020).
- **Enabling fair comparison**: Standardized reward functions and hyperparameter mappings reduce confounding variables in cross-library comparisons.

### Potential Negative Impacts

- **Improved fine-tuning could be misused**: More effective RL post-training methods could be used to fine-tune models for harmful purposes (generating misinformation, bypassing safety training, producing harmful content). This risk applies broadly to all RL post-training research.
- **Alignment circumvention**: Knowledge distillation techniques could potentially be used to transfer capabilities while weakening safety guardrails. Our distillation experiments do not specifically investigate this risk but the methodology could be adapted for such purposes.
- **Compute inequality**: The compute requirements (~446 A100 GPU-hours) make this research accessible primarily to well-resourced institutions, potentially widening the gap between resource-rich and resource-constrained research groups.

### Mitigations

- All released model checkpoints are fine-tuned versions of already-released base models (Meta Llama 3, Qwen 3) and do not introduce new safety risks beyond those inherent in the base models.
- We provide model cards documenting intended use and known limitations for all released checkpoints.
- The benchmark focuses on mathematical reasoning and style preference — domains with low misuse potential compared to open-ended generation.
- We encourage the community to apply these evaluation methods to safety-critical domains (alignment, harmlessness, helpfulness) in future work.

---

## Responsible Use Guidelines

1. **Do not use** the released models or methods to intentionally bypass safety training of aligned models.
2. **Do not use** distillation methods to transfer harmful capabilities between models.
3. **Cite and credit** the base model creators (Meta, Alibaba) and respect their respective licenses.
4. **Report** any discovered vulnerabilities or misuse potential to the maintainers via GitHub issues.
