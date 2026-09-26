# E1 paired vLLM arm: trained (seed809) on SWE-bench Pro (2026-09-26)

**Result: resolved 0/10 = 0.0 (Wilson95 0.000–0.278), native evaluator.** Paired vs vLLM base: 0.0 vs 0.0, b=0 c=0, McNemar p=1.0 (`../paired.json`).

- Actor: `pavlov-public-portfolio-bf16` at `https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run`
  (Qwen3.6-35B-A3B + seed809 stepfinal LoRA, pre-merged BF16, vLLM 0.28 on 1× H200).
- Same 10 instances, stored source contexts, prompt, non-thinking chat template, temperature 0, diff extraction and native
  grader as the Tinker-base run (`../result.json`). Only the endpoint and model id change. Raw `/v1/completions` with token ids.
- **max_tokens clamp:** the server's max-model-len is 32,768, so max_tokens = min(8192, 32768 − prompt). 9/10 prompts
  (25.7k–30.7k tokens) were clamped to 2,112–7,098 tokens. The clamp is identical in the base arm, so the pair is
  like-for-like but not comparable to the Tinker run's 8,192. One generation here (flipt-967855) stopped at its cap.
- Outcome: 10/10 extractable diffs; `git apply --check` passes on 1/10 (flipt-aebaecd). The zero is diff-format failure.
- protonmail-815695 cannot be resolved under this harness (Karma ChromeHeadless fails to start).
- Cost: 262,616 prompt + 17,968 sample tokens; sampling 94.8 s; native eval 203.6 s. The shared GPU bill is reconciled by the lead.
- Compare only with `../vllm_base/`, never with the Tinker-base numbers.

## Rerun (from outputs/e1_e14_small_scale_2026-09-26)
```bash
set -a; source trained_actor/.env.local; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python E1/code/run_e1_vllm.py sample trained
python3 E1/code/run_e1_vllm.py eval trained
uv run --no-project --with modal==1.5.4 python E1/code/run_e1_vllm.py apply trained   # diagnostic
python3 E1/code/run_e1_vllm.py score trained && python3 E1/code/run_e1_vllm.py paired
```
