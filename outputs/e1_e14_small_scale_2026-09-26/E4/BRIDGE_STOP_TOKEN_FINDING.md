# E4 2026-09-21 base-model rerun: the 0.0 traces to the serving bridge

**Finding (2026-09-26).** The 100-trial E4 rerun of 2026-09-21 (mean reward 0.0) was served through the Modal
bridge's `/v1/responses` route. That route passes no stop sequences to the sampler and decodes with special
tokens stripped, so generation is not cut at the end-of-turn token `<|im_end|>`.

**Evidence**
- `zvf-program/flagship/modal_tinker_openai_bridge.py`: the `/v1/responses` handler calls the shared sampler with
  `stop=None` (line 524), and decoding uses `skip_special_tokens=True` (line 314).
- The 09-21 trajectories show generation running past `<|im_end|>` into invented `user` / `<tool_response>` /
  `assistant` turns. This is the "role-token degeneration" (`user user assistant …`) that the 09-19 ledger read as
  the base model failing to sustain the tool loop (recorded in `E4/result.json` → `prior_base_run.caveat`).
- The same base model, served through `E4/code/tinker_shim.py` with `<|im_end|>` as a stop sequence, scored 0.4087 on
  task `btb-19b3361c` in the 2026-09-26 small-scale rerun (`E4/result.json`).

**Consequence.** The 09-21 zero is best read as a defect of the serving bridge, not as a measurement of the base
model's tool use or of finance reasoning. It stays in the record as an infrastructure finding. The two runs are not
pooled: they differ in serving path, task count and caps.
