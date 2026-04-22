# Dissertation benchmark comparison

## Benchmark set

I used a compact set of representative, high-quality dissertations rather than a generic web list. The set covers classical deep RL, meta-learning/RL experimental organization, LLM reasoning, LLM agents with reward shaping, and LLM alignment.

| File | Why it was used as a benchmark |
| --- | --- |
| `schulman_2016_optimizing_expectations.pdf` | Strongest reference for RL dissertation structure: theory, algorithm, experiments, proofs, and implementation details are tied together tightly. |
| `finn_2018_learning_to_learn_with_gradients.pdf` | Strong reference for chapter-level experimental rhythm: problem framing, method, related work, experiments, discussion, and future work repeat cleanly. |
| `fu_2025_improving_complex_reasoning_llms.pdf` | Closest benchmark for LLM reasoning: clear roadmap from LLM training paradigms to reasoning methods, evaluation, and chapter-specific summaries. |
| `zhuang_2025_reasoning_planning_reward_shaping.pdf` | Closest benchmark for LLM agents and tool use: includes explicit error categories, evaluation framing, and recommendations after each empirical block. |
| `sun_2026_scalable_alignment_llms.pdf` | Strong recent LLM alignment benchmark: organizes work by large research themes and repeatedly connects SFT, RLHF/RLAIF, reasoning, and evaluation. |

## Comparison against the current capstone report

### What the report already does well

The current report is unusually strong for a capstone because it does not present a single happy-path result. It has a broad experiment inventory, a clear evidence hierarchy, figures that summarize major conclusions, held-out evaluation checks, and an original diagnostic framing around Zero-Variance Fraction (ZVF) and Gradient Utilization (GU). Its central claim is also appropriately conditional: GRPO is useful when a base model and reward function produce within-group reward variance.

### Gaps found from the dissertation benchmark

1. The report had a strong headline thesis, but the practical diagnostic logic was spread across several paragraphs. Schulman-style dissertations make the theory-to-action bridge explicit.
2. The Results chapter gave verdicts, but it did not yet include a compact error/failure taxonomy. Zhuang's agent dissertation shows how useful this is for tool-use and planning work.
3. The Limitations chapter was too short for dissertation-level scrutiny. The benchmark theses separate internal, construct, external, statistical, and reproducibility threats more explicitly.
4. Future work was directionally good, but the causal chain from failure mode to next experiment needed to be easier to audit.
5. The artifact story was present, but the report should more clearly tell future readers what logs, diagnostics, and scripts are needed to reproduce the claims.

## Changes made to improve the report

1. Added a new Results section, `Failure Taxonomy and Diagnostic Action Map`, that converts the empirical findings into a practitioner-facing diagnostic table.
2. Replaced the short `Limitations` chapter with `Limitations and Threats to Validity`, split into internal validity, construct validity, external validity, statistical conclusion validity, and reproducibility limits.
3. Strengthened the report's dissertation-like contribution by making the negative results actionable instead of merely cautionary.
4. Preserved the existing conservative claims. No new empirical claims were added beyond what the current logs and validation support.

