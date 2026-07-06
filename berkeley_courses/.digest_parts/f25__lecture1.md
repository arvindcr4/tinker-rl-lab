### Introduction to training LLMs for AI agents
*Speaker:* Yann Dubois

#### Key claims / techniques
- Pretraining is next-token prediction on >10T tokens of filtered/deduplicated web data; the practical bottleneck is data quality and compute, and scaling laws allow small-scale experiments to predict large-model performance.
- Compute-optimal training (Chinchilla) suggests ~20 tokens per parameter for fixed training FLOPs, but production models often use much larger ratios (>150:1) because inference cost is ignored by that rule.
- A current flagship pretraining run (LLaMA 3 400B) is estimated at ~3.8×10²⁵ FLOPs, ~26M GPU-hours, ~$52M, and ~4.4ktCO₂eq, illustrating the cost and carbon scale of frontier pretraining.
- Supervised fine-tuning (SFT) is behavior cloning on desired outputs; it can learn style and instruction following from ~10k examples but is bounded by human ability and can reward hallucination when the model clones answers it does not understand.
- Reinforcement learning post-training (RLHF/RLAIF) maximizes desired behavior rather than cloning it, using rule-based rewards, learned preference models, or LLM-as-judge rewards.
- DeepSeek-R1 uses GRPO with Monte-Carlo advantage estimates; Kimi K1.5/K2 use a similar loss, and both require heavy sampling infrastructure.
- RL infrastructure is a first-class bottleneck: sampling multiple outputs per problem, long agent rollouts, and slow environment feedback require techniques such as pausing long-tail rollouts, concurrent rollouts, and colocated engines to keep inter-GPU communication under ~30 seconds.
- Close-ended evaluation (e.g., MMLU) works when answers are automatically verifiable; open-ended evaluation relies on pairwise human preference (ChatBot Arena) or LLM judges (AlpacaEval), with length and other spurious correlations as known failure modes.
- Systems optimizations for training include automatic mixed precision/bf16, operator fusion via `torch.compile`, tiling/FlashAttention (~1.7× end-to-end speedup), data/model/pipeline/tensor parallelism (ZeRO, GPipe, Megatron-LM), and sparse Mixture-of-Experts architectures.
- Tokenization via Byte Pair Encoding (BPE) merges common character subsequences to obtain a vocabulary that is more general than words yet shorter than character sequences.

#### Relevance hooks
- GRPO/RL post-training benchmarking: the lecture explicitly describes DeepSeek-R1’s GRPO objective with MC advantages, Kimi K2/K1.5’s similar loss, and RL infrastructure bottlenecks that matter for reproducible RL agent training.
- Evals-with-error-bars / length bias: notes that automated LLM judges and open-ended benchmarks can exhibit spurious correlations (e.g., length preference) and that causal/regression controls for length are needed.
- Agent evaluation methodology: discusses Kimi K2’s complex SFT pipeline with simulated users, tools, and rubric-based rejection sampling, plus RL environment latency as a key agent-training constraint.

#### Cited paper titles (verbatim only)
- The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

Index row: f25 | lecture1.pdf | Yann Dubois | Broad survey of LLM pretraining, SFT, RL/RLHF, evaluation, and systems, emphasizing scaling laws, GRPO, and RL infra bottlenecks. | ok
