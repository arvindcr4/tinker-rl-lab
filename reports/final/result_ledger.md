# Source-of-Truth Result Ledger

This ledger reconciles headline numbers used in the final capstone report. It separates online training reward, held-out accuracy, partial runs, failed runs, and supporting external/team experiments so the report cannot be read as cherry-picking among result files.

## Included Headline Results

| Experiment ID | Dataset / split | Model | Training / method | Reward or metric | Decoding / selection rule | Final number | Main-claim role |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `gsm8k-base-control-200` | GSM8K test, same 200 examples as GRPO held-out eval | Qwen3-8B, no LoRA adapter | None | Held-out greedy exact-match accuracy | Temperature 0, same parser and prompt as GRPO held-out eval | 164/200 = 82.0% | Base reference for held-out math only |
| `gsm8k-grpo-heldout-5seed` | GSM8K test, 200 examples per seed | Qwen3-8B + GRPO LoRA | GRPO checkpoints from five seeds | Held-out greedy exact-match accuracy | Final reported checkpoints, temperature 0, same parser as base control | Mean 83.3%, SD 2.2, p=0.26 vs 82.0% base | Non-significant held-out lift; no broad math-improvement claim |
| `structural-gsm8k-qwen8b` | GSM8K rollout prompts | Qwen3-8B | GRPO | Online training reward | Training-time stochastic rollout metric, last-10 summary | Last-10 near 1.0 in selected completed runs | Training-dynamics and saturation evidence only |
| `sandhya-tool-sft-grpo` | Custom function-calling data | Qwen2.5-3B | SFT then GRPO | Structured tool-call score | Custom schema-validity/task rubric | 0.72 SFT to 0.91 SFT+GRPO | Structured-output refinement after warm-up |
| `strict-tool-no-warmup` | Strict Tinker tool-use tasks | Qwen3-32B / Llama-8B variants | GRPO without SFT warm-up | Online rule-based reward | Training-time rollout reward | 0.000 | Sparse schema reward dead without warm-up |
| `central-humaneval-binary` | HumanEval-style verifier in central structural campaign | Qwen3-8B | GRPO | Online binary functional reward | Training-time binary pass/fail reward | Last-10 0.024, initial ZVF 1.00 | No usable signal in this sparse-reward setup |
| `madhu-humaneval-external` | HumanEval external/team pipeline | Qwen3-8B | External GRPO pipeline | Functional pass count | Different pipeline and reward design from central run | 141/164 | Supporting evidence only; not pooled with central HumanEval |
| `ppo-grpo-qwen` | GSM8K rollout prompts | Qwen3-8B | PPO vs GRPO | Last-10 online reward | Recorded training reward comparison | GRPO 0.344; PPO 0.225 in this ledger, 0.350 in `experiments/statistical_analysis.md` | Artifact-sensitive comparison; not used as a GRPO-over-PPO claim |
| `ppo-grpo-llama` | GSM8K rollout prompts | Llama-3.1-8B-Instruct | PPO vs GRPO | Last-10 online reward | Recorded training reward comparison | PPO 0.975 vs GRPO 0.844 | Direction flip showing confounding |

## Excluded, Partial, or Supporting-Only Artifacts

| Artifact / run ID | Status | Reason excluded from headline claims | How it is used |
| --- | --- | --- | --- |
| `reports/final/gsm8k_test_results.json` | Failed | Model/tokenizer dependencies did not load; no valid full-GSM8K examples were completed | Disclosed as an invalid full-GSM8K attempt |
| `humaneval_qwen3-8b` | Partial | Timed out after 40/164 examples | Reproducibility warning only |
| `heldout_qwen3.5-27b` | Partial | Timed out at 100/200 examples | Hypothesis-generating larger-model held-out probe |
| `heldout_qwen3-32b` | Partial | Timed out at 100/200 examples | Hypothesis-generating larger-model held-out probe |
| `kl_qwen3-8b` | Failed | PPO+KL variant hit a gradient error before producing a valid comparison | Implementation failure, not an algorithmic result |
| Interrupted frontier / MoE runs | Partial | Too few steps or interrupted checkpoints | Reported only as frontier probes, not scaling-law evidence |
| Eight-example keyword/preference pilots | Supporting-only | Data too small and metric too narrow for causal claims | Appendix or breadth evidence only |
| `browser-control-smoke-agent-browser` | Supporting-only | Three local synthetic browser tasks are too small and non-standardized for benchmark claims | Agentic systems evidence for real browser observation/action/screenshot control |
| `browsergym-tinker-smoke-configs` | Added scaffold | BrowserGym MiniWoB/WebArena configs and wrapper are runnable infrastructure; full benchmark runs still require MiniWoB/WebArena service setup | Reproducible path from Tinker GRPO to browser benchmark rewards |

## Claim Rules

- Online training reward is never reported as held-out benchmark accuracy.
- Held-out GSM8K improvement is described as small and statistically non-significant.
- Tool-call results are described as structured-output/schema emission unless execution-level success is measured.
- HumanEval evidence is split by setup: the central sparse binary setup had no signal; the external pipeline shows code is not intrinsically impossible under richer rewards.
- PPO/GRPO comparisons are treated as model/backend interaction evidence, not as an algorithm leaderboard.
- The Qwen PPO/GRPO row is not directional evidence because artifact summaries disagree over the PPO last-10 value (0.225 here versus 0.350 in `experiments/statistical_analysis.md`).
- Browser-control smoke results are reported only as agentic capability coverage unless replaced by standardized BrowserGym/WebArena runs.
