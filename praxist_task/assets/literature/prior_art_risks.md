# E9 prior-art, leakage, and comparability risks

Recorded: 2026-08-29T04:16:16Z

## Leakage and contamination

- Other competitors' Kaggle solutions, write-ups containing solution logic, or leaked private-test information must not enter prompts, repairs, or task assets. The official MLE-bench plagiarism tooling is a check, not permission to ingest such material.
- Iterative repair after seeing native or private-test feedback changes the experimental condition. Repaired artifacts must be a pre-registered secondary arm with the exact repair list and receipt chain; they cannot replace the original agent outcome.
- Public samples and fixtures establish plumbing only. They are not held-out evaluation evidence.

## Runtime and artifact risks

- The Tinker sampler checkpoint referenced by the bridge returns provider `404`, while the training run currently lists no checkpoints. The Hugging Face PEFT export proves artifact availability, not recoverability of the deleted provider checkpoint.
- vLLM's Qwen MoE LoRA path has layout-sensitive serving flags. Header inspection confirms this adapter uses fused 3D expert tensors; any dynamic mixed-format serving must explicitly declare `is_3d_lora_weight: true`. The existing merged checkpoint avoids that declaration path but still requires shard verification and deterministic canaries.
- An immutable adapter commit is necessary but insufficient: base-model revision, tokenizer, runtime, dtype, and decoding parameters also affect reproducibility.

## Benchmark-version risks

- MLE-bench v1 has documented known issues and its leaderboard is not currently accepting new submissions. Results must name the exact source commit and cannot be represented as an accepted leaderboard submission.
- The pandas 3 text-normalization issue was confirmed for the hash-locked Russian submission: pandas 3 produced an invalid/null grader outcome, while a native-only regrade under pinned `pandas==2.2.2` returned a valid `0.97348`. Earlier program-execution failures remain separate failures and are not relabeled.
- Resource-envelope deviations from the benchmark defaults can materially affect performance. Record CPU, GPU, memory, time, data access, and model-call budgets for every run.

## Statistical and reporting risks

- The current 37/75 unique native grades are incomplete coverage, not an E9 suite score.
- A single seed is not benchmark-comparable evidence for the recommended mean and SEM. Any interim single-seed statistics must be labeled exploratory.
- Selecting only competitions with runnable data or successful programs creates survivorship bias. The denominator and blocker taxonomy must retain all 75 contracted competitions.
- Operational metrics such as coverage, cost, or valid-submission rate are diagnostic; they must not be substituted for Any Medal (%).

## Governance controls

- Fail closed on missing licenses, task splits, native graders, online W&B-before-paid-work, or immutable Hugging Face receipts.
- Preserve all failed receipts and hashes. Never overwrite a null or failed outcome with a repaired run.
- Maintain the user's $50.00 additional cap as a hard cumulative limit; the exact persistent ceiling is $55.91445263 from the recorded baseline.
