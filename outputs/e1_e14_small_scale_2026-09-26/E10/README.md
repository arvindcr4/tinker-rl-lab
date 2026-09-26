# E10 small-scale — AgentHarm public split (original benchmark; private split blocked)

**Harmful (30): refusal rate 20/30 = 66.7%** (Wilson 95% 48.8–80.8%); harm score (mean native score) 0.221; 6/30 fully completed.
**Benign (30): score 0.814**; 19/30 fully completed; 2/30 refused.
The actor and both judges are base Qwen3.6-35B-A3B (no adapter) on Tinker: non-thinking, temp 0, max_tokens 4096.

What ran: native `inspect_evals.agentharm` / `agentharm_benign` at pinned b935c0e5 with inspect_ai 0.3.269.
The split is test_public, dataset @e23b3fe6. Defaults: message_limit 20, 0 irrelevant tools, no hint/detailed filter.
Behaviour ids: `random.Random(20260926).sample(30)` per split; the ids are in `raw/selected_ids.json`.
The model is served through `code/shim.py`, a localhost OpenAI chat.completions shim over Tinker. The Qwen3.6 template renders
the tools, and the shim parses its XML `<tool_call>` output back into OpenAI tool_calls.
The default GPT-4o refusal and semantic judges were replaced by the same Tinker model.

Rerun (venv A: tinker/transformers/pillow; venv B: `pip install -e inspect_evals@b935c0e5 openai`):
```
set -a; source .env; set +a; cd outputs/e1_e14_small_scale_2026-09-26/E10/code
A/bin/python shim.py 18765 ../raw/shim_log.jsonl &
OPENAI_BASE_URL=http://127.0.0.1:18765/v1 OPENAI_API_KEY=local-shim B/bin/python run_e10.py; kill %1
python ../../E8/code/finalize.py e10
```
Caveats: the judges grade their own model's outputs (self-judging), so scores may be biased and are not leaderboard-comparable.
The lost-adapter lane figure (AgentDojo benign 97/97) is a different benchmark. Two earlier launches scored nothing: one hit inspect's Responses API default, the other the unsupported 'developer' role. See raw/failed_attempts/.

Paired vLLM arm (trained vs base, same Modal H200 engine, same items/prompts/caps/graders): `vllm_trained/`, `vllm_base/`, `paired.json`.
Rerun: `ARM=trained|base ACTOR_PORT=.. JUDGE_PORT=.. bash ../E8/code/run_arm.sh` (+ Tinker judge shim for E10; Omni-Judge `modal run` for E14), then `python3 ../E8/code/paired.py E10`.
