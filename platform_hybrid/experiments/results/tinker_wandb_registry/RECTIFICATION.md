# Correlation rectification with Hugging Face evidence

Generated: 2026-07-10T20:06:04.576338+00:00

- HF repos scanned: 33
- Unique Tinker run IDs found on HF: 19
- ...of which present in the 949-run registry: 19
- ...NOT in the registry (orphans): []

## Resolution distribution

- `unmatched`: 882
- `confirmed_wandb`: 28
- `candidate`: 17
- `hf_only`: 5
- `confirmed_wandb_hf`: 5
- `hf_arbitrated_wandb_mislink`: 5
- `candidate_hf`: 4
- `conflict_unresolved`: 3

## Conflicts and arbitrations

- **hf_arbitrated_wandb_mislink** `73ff186a-d8e3-50c3-afb8-2b863cd09579:train:0` Tinker=`Qwen/Qwen3-32B` W&B=['kimik2thinking'] HF=['Qwen/Qwen3-32B'] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/r8uwff0t)
- **hf_arbitrated_wandb_mislink** `657a920a-9e74-55d2-9354-71a6ec2f1f61:train:0` Tinker=`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` W&B=['gptoss20b'] HF=['nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16'] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/19y4lpcc)
- **hf_arbitrated_wandb_mislink** `ca2e3a24-7401-5770-af34-a0d27177aeaa:train:0` Tinker=`meta-llama/Llama-3.1-8B-Instruct` W&B=['kimik2thinking'] HF=['meta-llama/Llama-3.1-8B-Instruct'] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/d3bvb4db)
- **hf_arbitrated_wandb_mislink** `56b99b24-03bd-5d00-ae23-831933ef53b2:train:0` Tinker=`deepseek-ai/DeepSeek-V3.1` W&B=['gptoss20b'] HF=['deepseek-ai/DeepSeek-V3.1'] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/8tg4wv64)
- **hf_arbitrated_wandb_mislink** `3154dee5-4fd7-59b6-ba3b-721eef675bfc:train:0` Tinker=`Qwen/Qwen3-8B` W&B=['kimik2thinking', 'qwen3527b'] HF=['Qwen/Qwen3-8B'] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/d3bvb4db)
- **conflict_unresolved** `0ef59237-82ae-5e22-b789-61da6a6a85c8:train:0` Tinker=`Qwen/Qwen3.5-4B` W&B=['nvidianemotron3super120ba12bbf16'] HF=[] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/ax59u2zl)
- **conflict_unresolved** `78e2d35b-d7ae-5b92-81e8-7a1a2ff294d3:train:0` Tinker=`Qwen/Qwen3.5-27B` W&B=['qwen330ba3b'] HF=[] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/8nhyfstb)
- **conflict_unresolved** `0dfee749-2c83-50b8-95bf-b96e04ca51eb:train:0` Tinker=`meta-llama/Llama-3.1-8B-Instruct` W&B=['deepseekv31'] HF=[] (https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/r28pjkud)

## Files

- `hf_runs.jsonl`: per-repo harvest (run IDs + model labels).
- `tinker_wandb_hf_correlation.csv`: v1 correlation + 4 HF columns.

Provenance: read-only HF listing; no token stored; nothing written
to HF. The v1 CSV is left untouched.
