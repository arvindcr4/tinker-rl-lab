# 10x Structural Ceiling — Experiment Matrix

**Goal:** Systematically map when RL fine-tuning helps vs. doesn't, establishing a "structural ceiling" thesis with NeurIPS-caliber evidence.

**Total runs:** ~50 | **Estimated cost:** ~$95-120 on Tinker (~$1.90/run) | **Timeline:** 4 weeks

## Block Summary

| Block | Experiment | Configs | Purpose |
|-------|-----------|---------|---------|
| **A** | Multi-seed GSM8K | 2 (seeds 4-5) | Statistical power for null finding (seeds 1-3 exist) |
| **B** | Model family isolation | 3 (Gemma-2-9B, Phi-3-medium, Mistral-7B) | Break Qwen architecture confound |
| **C** | Size ladder | 4 (0.6B, 1.7B, 4B, 14B) | Scaling curve — where does the null break? |
| **D** | PPO baseline | 1 (Qwen3-8B) | Algorithm isolation — critic-based RL |
| **E** | DPO baseline | 1 (Qwen3-8B) | Offline RL alternative |
| **F** | Constrained decoding | 2 (constrained vs unconstrained) | Decoder confound — is GRPO's gain real? |
| **G** | Group size ablation | 3 (G=4, 32, 64) | Group saturation onset mapping |
| **H** | Benchmark transfer | 2 (MATH, HumanEval) | Beyond GSM8K — harder benchmarks |
| **I** | LR sweep | 3 (1e-5, 1e-4, 3e-4) | Hyperparameter sensitivity |
| **J** | Tool-use cross-family | 2 (Gemma-2-9B, Llama-8B) | Replicate 0%→92% JSON on non-Qwen |

**Total new configs: 23** (+ existing 5 from capstone = 28 unique experiments)

## Key Hypotheses

1. **Structural ceiling:** RL reliably learns format compliance (JSON validity, answer boxing) but NOT semantic competence (mathematical reasoning, code logic)
2. **Group saturation:** Zero-variance gradient fraction increases with training, causing learning stalls — measurable via diagnostic
3. **Architecture independence:** The null result on GSM8K generalizes across Qwen, Llama, Gemma, Phi, Mistral families
4. **Size threshold:** There may exist a model size above which RL breaks through the ceiling (14B test)

## Novel Contributions

1. **Group Saturation Diagnostic** — first systematic measurement of zero-variance fraction in GRPO training
2. **Structural vs. Semantic decomposition** — constrained decoding ablation isolates format learning from reasoning
3. **Cross-family scaling law** — "Chinchilla for RL fine-tuning" across 4+ model families
4. **Practical prescription:** When to use RL vs. SFT vs. constrained decoding

## Running

```bash
# All experiments
./run_all.sh

# Specific block
./run_all.sh block_b

# Dry run
./run_all.sh --dry-run

# Single config
python grpo_10x_runner.py --config configs/block_b_gsm8k_gemma2_9b.yaml

# Analysis
python analyze_results.py
```

## Priority Order (impact per dollar)

1. **Block B** — Family isolation is the single strongest evidence upgrade
2. **Block G** — Group saturation diagnostic is the novel contribution
3. **Block C** — Size ladder gives the scaling law figure
4. **Block F** — Constrained decoding directly addresses reviewer criticism
5. Blocks H, D, E, I, J, A — diminishing but still valuable
