VOID: first E12 vllm_trained attempt (2026-09-26 ~14:58Z). Port 8775 was already held by E13's vLLM-BASE shim
(max_tokens 128); my E12 shim failed to bind, so the 6 generation calls went to E13's shim: they are base-model,
128-token outputs, logged as lines 1901,1909,1919,1927,1932,1937 of E13/vllm_base/raw/shim_calls.jsonl (not edited).
Judge calls here are Tinker tokens spent on invalid generations. Not scored. Rerun on verified-free ports.
