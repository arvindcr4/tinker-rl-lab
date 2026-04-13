# NeurIPS 2026 Paper Checklist

> This checklist must be included in the LaTeX submission.
> Reference: https://neurips.cc/public/guides/PaperChecklist

## 1. Claims
**Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?**
- [x] Yes
- Justification: The paper clearly states contributions in Section 1 (Introduction), including the unified benchmark scope, cross-library evaluation methodology, and empirical findings. Limitations on model scale and task diversity are explicitly noted.

## 2. Limitations
**Did you discuss the limitations of your work?**
- [x] Yes
- Justification: See Section 7 (Limitations). We discuss: (a) experiments limited to 1B–30B parameter models, (b) task coverage limited to math reasoning and preference learning, (c) Tinker API dependency for some experiments, (d) LoRA-only fine-tuning without full fine-tuning baselines.

## 3. Theory, Assumptions and Proofs
**Did you state the full set of assumptions of all theoretical results?**
- [x] Yes
- Justification: The GRPO objective and importance-sampling loss formulations are stated with assumptions in Section 3 (Method). The on-policy distillation derivation with KL divergence assumptions is in Appendix A.

## 4. Experimental Result Reproducibility
**What steps did you take to make your results reproducible?**
- [x] Yes
- Justification: We provide:
  - Complete source code with pinned dependencies (requirements.txt)
  - Docker container for exact environment reproduction
  - Seed management across Python, NumPy, PyTorch, CUDA (utils/seed.py)
  - All experiments run with 5 seeds (42, 123, 456, 789, 1024)
  - REPRODUCE.md with exact commands for every experiment
  - YAML configs for all hyperparameters

## 5. Open Access to Data and Code
**Did you include code, data, and instructions needed to reproduce results?**
- [x] Yes
- Justification: Code is released as a supplementary anonymized repository. Model checkpoints and datasets are hosted on Hugging Face Hub. See REPRODUCE.md for complete instructions. An anonymized version is provided for review.

## 6. Experimental Setting/Details
**Did you specify all training details?**
- [x] Yes
- Justification: All hyperparameters are documented in YAML config files (atropos/configs/) and in the paper's Appendix B (Hyperparameter Tables). Data splits follow standard GSM8K train/test. Hyperparameter selection methodology is described in Section 4.

## 7. Experiment Statistical Significance
**Does the paper report error bars and statistical significance?**
- [x] Yes
- Justification: All results report mean ± standard error across 5 seeds. Learning curves include 95% confidence bands. Pairwise algorithm comparisons use Welch's t-test. We follow the methodology of Colas et al. (2019) and use rliable (Agarwal et al., 2021) for aggregate metrics.

## 8. Experiments Compute Resources
**Did you provide sufficient compute resource information?**
- [x] Yes
- Justification: See COMPUTE.md and Section 4.4 of the paper. We report GPU type (NVIDIA A100 40GB/80GB), total GPU-hours per experiment and for the full project (~200 A100 GPU-hours), cloud provider details, and Tinker API compute costs.

## 9. Code of Ethics
**Have you read the NeurIPS Code of Ethics?**
- [x] Yes
- Justification: We have reviewed the Code of Ethics and confirm compliance. No human subjects research or sensitive data collection is involved.

## 10. Broader Impacts
**Did you discuss potential negative societal impacts?**
- [x] Yes
- Justification: See LIMITATIONS_AND_IMPACT.md and Section 7.2 of the paper. We discuss potential risks of improved RL fine-tuning (misuse for harmful content generation, alignment bypassing) and mitigations (responsible disclosure, evaluation frameworks).

## 11. Safeguards
**Do you have safeguards for responsible release?**
- [x] Yes
- Justification: Model checkpoints are released with model cards documenting intended use and limitations. All models are fine-tuned versions of already-released base models (Llama 3, Qwen 3) with their respective licenses. No novel safety risks beyond the base models are introduced.

## 12. Licenses
**Did you cite creators and respect licenses of existing assets?**
- [x] Yes
- Justification: All base models, datasets, and libraries are cited with their licenses:
  - Meta Llama 3: Llama 3 Community License
  - Qwen 3: Apache 2.0
  - GSM8K: MIT License
  - NoRobots: Apache 2.0
  - TRL, Transformers: Apache 2.0
  - Our code: Apache 2.0

## 13. Assets
**Did you document new assets released?**
- [x] Yes
- Justification: All released model checkpoints include Hugging Face model cards with training details, evaluation results, intended use, and limitations. The benchmark dataset release includes a datasheet. See huggingface/ directory for templates.

## 14. Crowdsourcing and Research with Human Subjects
**N/A** — No crowdsourcing or human subjects research.

## 15. IRB Approvals
**N/A** — No human subjects research.

## 16. Declaration of LLM Usage
- [x] Yes
- Justification: LLMs were used for code formatting assistance and documentation drafting. They were not used as a component of the core research methodology. All scientific claims, experimental design, and analysis are human-authored.
