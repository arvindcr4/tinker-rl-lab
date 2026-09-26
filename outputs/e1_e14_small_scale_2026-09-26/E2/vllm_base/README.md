# E2 paired vLLM arm: base control on EffiBench substitute (2026-09-26)

**Result: pass@1 = 24/30 = 0.800 (Wilson95 0.627–0.905).** Paired vs vLLM trained: 0.700 vs 0.800, b=0 c=3, McNemar p=0.25 (`../paired.json`).

- Actor: `qwen36-base-bf16` at `https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run`
  (Qwen3.6-35B-A3B base BF16, same volume snapshot and vLLM image/flags as the trained arm, 1× H200).
- Same 30 problems, upstream prompt, non-thinking chat template, temperature 0, max_tokens 2048, and Docker grader as the
  Tinker-base run (`../result.json`). Only the endpoint and model id change. No context clamp was needed.
- Substitute for FrontierSWE; it does not approximate or bound a FrontierSWE score.
- 2 items (1120, 1842) have broken upstream tests and count as failures. 1 completion hit the 2048 cap (2533); it failed.
- Secondary: NET geomean 0.944 over the passed items. The paired log-NET difference is wall-clock noise, not a claim.
- Cost: 25,520 prompt + 17,923 sample tokens; sampling 75.0 s; grading 3.4 s. The shared GPU bill is reconciled by the lead.
- Compare only with `../vllm_trained/`, never with the Tinker-base numbers.

## Rerun (from outputs/e1_e14_small_scale_2026-09-26)
```bash
set -a; source trained_actor/.env.local; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python E2/code/run_e2_vllm.py sample base
python3 E2/code/run_e2_vllm.py grade base && python3 E2/code/run_e2_vllm.py paired
```
