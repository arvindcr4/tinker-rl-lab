# E2 paired vLLM arm: trained (seed809) on EffiBench substitute (2026-09-26)

**Result: pass@1 = 21/30 = 0.700 (Wilson95 0.521–0.833).** Paired vs vLLM base: 0.700 vs 0.800, b=0 c=3, McNemar p=0.25 (`../paired.json`).

- Actor: `pavlov-public-portfolio-bf16` at `https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run`
  (Qwen3.6-35B-A3B + seed809 stepfinal LoRA, pre-merged BF16, vLLM on 1× H200).
- Same 30 problems, upstream prompt, non-thinking chat template, temperature 0, max_tokens 2048, and Docker grader as the
  Tinker-base run (`../result.json`). Only the endpoint and model id change. No context clamp was needed.
- Substitute for FrontierSWE; it does not approximate or bound a FrontierSWE score.
- 2 items (1120, 1842) have broken upstream tests and count as failures. 2 completions hit the 2048 cap (2513, 2533); both failed.
- Secondary: NET geomean 0.791 over the passed items. The paired log-NET difference is wall-clock noise, not a claim.
- Cost: 25,520 prompt + 18,184 sample tokens; sampling 77.5 s; grading 5.7 s. The shared GPU bill is reconciled by the lead.
- Compare only with `../vllm_base/`, never with the Tinker-base numbers.

## Rerun (from outputs/e1_e14_small_scale_2026-09-26)
```bash
set -a; source trained_actor/.env.local; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python E2/code/run_e2_vllm.py sample trained
python3 E2/code/run_e2_vllm.py grade trained && python3 E2/code/run_e2_vllm.py paired
```
