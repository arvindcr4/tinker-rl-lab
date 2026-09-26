# E1 paired vLLM arm: base control on SWE-bench Pro (2026-09-26)

**Result: resolved 0/10 = 0.0 (Wilson95 0.000–0.278), native evaluator.** Paired vs vLLM trained: 0.0 vs 0.0, b=0 c=0, McNemar p=1.0 (`../paired.json`).

- Actor: `qwen36-base-bf16` at `https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run`
  (Qwen3.6-35B-A3B base BF16, same volume snapshot and vLLM image/flags as the trained arm, 1× H200).
- Same 10 instances, stored source contexts, prompt, non-thinking chat template, temperature 0, diff extraction and native
  grader as the Tinker-base run (`../result.json`). Only the endpoint and model id change. Raw `/v1/completions` with token ids.
- **max_tokens clamp:** the server's max-model-len is 32,768, so max_tokens = min(8192, 32768 − prompt). 9/10 prompts
  (25.7k–30.7k tokens) were clamped to 2,112–7,098 tokens. The clamp is identical in the trained arm, so the pair is
  like-for-like but not comparable to the Tinker run's 8,192. No generation here hit its cap.
- Outcome: 10/10 extractable diffs; `git apply --check` passes on 1/10 (flipt-aebaecd), and 9 are corrupt patches. protonmail-815695 cannot be resolved here (Karma ChromeHeadless fails to start).
- Grading: pass 1 lost the teleport output and `eval_results.json` to a full disk (`raw/eval_attempt1_enospc.log`). Pass 2 was
  SIGTERMed. Pass 3 reused the 9 saved per-instance outputs and graded only teleport.
- Cost: 262,616 prompt + 9,266 sample tokens; sampling 53.9 s; the final eval pass took 74.0 s. The shared GPU bill is reconciled by the lead.
- Compare only with `../vllm_trained/`, never with the Tinker-base numbers.

## Rerun (from outputs/e1_e14_small_scale_2026-09-26)
```bash
set -a; source trained_actor/.env.local; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python E1/code/run_e1_vllm.py sample base
python3 E1/code/run_e1_vllm.py eval base
uv run --no-project --with modal==1.5.4 python E1/code/run_e1_vllm.py apply base   # diagnostic
python3 E1/code/run_e1_vllm.py score base && python3 E1/code/run_e1_vllm.py paired
```
