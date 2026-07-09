### Open Training Recipes for Reasoning in Language Models
*Speaker:* Hanna Hajishirzi

#### Key claims / techniques
- Open, reproducible, fully-open language-model ecosystems (OLMo, Tülu) are prerequisites for accelerating the science of LMs, rather than relying on closed APIs.
- A modern LLM post-training pipeline has four stages: supervised instruction tuning (SFT), preference tuning, RL with verifiable rewards (RLVR), and verifier/reward-model components.
- Data quality and targeted data mixing dominate post-training performance; persona-driven synthetic data is especially useful for scaling reasoning, math, code, and precise-instruction-following skills.
- Adding more persona-driven synthetic math data consistently improves MATH performance, while adding grade-school math helps GSM8K; self-consistency voting lets the authors prune ~40% of synthetic CoT data with no loss.
- PPO generally outperforms DPO by roughly 1%, but at higher implementation complexity, memory cost, and throughput cost; DPO’s cheapness makes it more practical for development.
- Scaling reward models does not always translate to better downstream models, and using in-domain prompts during preference tuning can yield further gains.
- Tülu 3 uses length-normalized DPO after experimenting with SimPO; preference tuning primarily improves style/chat while continuing SFT capability gains with smaller absolute magnitude.
- RL with verifiable rewards uses binary correctness signals (e.g., ground-truth answers for GSM8K, MATH, IFEval) instead of neural reward models, avoiding over-optimization and working “out of the box” with PPO.
- OLMo 2 chains multiple RLVR stages and is reported on par or better than Llama 3 and Qwen 2.5, rivaling DeepSeek and GPT-4o.
- Simple test-time scaling (s1) shows that a 1,000-sample high-quality, hard, diverse reasoning dataset plus inference-time budget forcing (e.g., appending “Wait”) can match or exceed o1-level MATH performance.
- Two-stage base-model training—pre-training on trillions of tokens followed by mid-training on ~50B high-quality in-domain/synthetic tokens—is presented as an efficient way to upgrade base-model quality.

#### Relevance hooks
- RL post-training benchmarking: the talk directly compares PPO, DPO, SimPO, and length-normalized DPO, and reports RLVR training curves and final results across GSM8K, MATH, IFEval, and BBH.
- Length bias: the discussion of length-normalized DPO and the decision to avoid SimPO signals attention to response-length and reward-hacking effects in preference optimization.
- RL reproducibility standards: the OLMo/Tülu recipes are framed as fully open, documented, and reproducible alternatives to closed models, emphasizing decontamination, license checks, and public data/evaluation ingredients.

#### Cited paper titles (verbatim only)
- Tülu 1
- Tülu 3
- OLMo
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- SimPO
- s1: Simple test-time scaling

Index row: sp25 | OLMo-Tulu-Reasoning-Hanna.pdf | Hanna Hajishirzi | Open, reproducible post-training recipe covering SFT, preference tuning, RLVR, and test-time budget forcing for reasoning models | ok
