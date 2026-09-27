# E13 — BALROG replacement scope (255 native episodes)

Status: RUNNING (final numbers go in result.json; this file is updated at the end).

## Setup
- BALROG balrog-ai/BALROG@b7afe79, native `Evaluator.run_episode`, `NaiveAgent`, native `vllm` OpenAI client, native
  `config.yaml` episode counts (babyai 5x10, babaisai 40x3, textworld 3x10, crafter 1x10, minihack 8x5, nle 1x5 = 255).
- Pinned forks as in the 2026-09-26 small-scale rerun (minihack 3ecb6da, TextWorld 1d56f47, Minigrid cf73dd1,
  baba-is-ai 33c2ca1, balrog-nle 0.9.0). Runs locally on macOS arm64, NetHack included (NLE reset/step verified).
- Actor: shared endpoint `pavlov-public-portfolio-bf16` through `code/actor_proxy.py` (adds bearer key,
  `enable_thinking=false`, caps in-flight requests at 4, logs every call to raw/proxy_calls.jsonl).
- Sampling as recorded for E13-native-20260912-01: temperature 1.0, max_tokens 8192, max_text_history 16, native seeds.
- Code: `code/run_full.py` (plan/run/summarize), `code/make_result.py` (result.json).

## Deviations
1. Non-thinking template (`enable_thinking=false`). The 2026-09-12 original run evidently ran with thinking on (~2.5k
   output tokens/step); on the full 255-episode suite (tens of thousands of steps) that is far outside budget.
2. NetHack step cap (lead decision, 2026-09-27 01:40 UTC). The recorded protocol is 5 NLE episodes with native
   `max_episode_steps=100000` (+ `no_progress_timeout=150`) and no per-episode wall-clock bound (the old hosted
   supervisor's 3600 s lease covered the whole run, not individual episodes). At ~2.4 s/step on the shared actor
   (~$5.24/h, $150 cap across all lanes), one episode that runs to the maximum could take ~65 h. Decision: keep all
   native settings, run NLE last, and cap NLE at **2000 agent steps** through BALROG's native
   `eval.max_steps_per_episode` knob. When the cap is hit, the native evaluator writes the episode JSON with
   progression-at-truncation. A truncated episode has no `"done": true` key, and result.json labels it as truncated.
   Three NLE episodes had already ended naturally before the cap existed (214, 306 and 599 steps, all with 0 progression).
   The cap does not bind for any of them, so they stand as native.
   The in-flight uncapped episode-01 (~818 steps, not finished) was killed and rerun from scratch with a new native
   seed. The babaisai episodes in flight at the stop were rerun too. Both events are logged in raw/attempts.jsonl as
   `killed_for_restart`.
3. macOS arm64 instead of the Linux x86 image (same pinned sources; dynamics not bit-verified).

## Evidence rules
Replacement-scope numbers only. They are never pooled with the original-contract 13-episode babyai receipt
(E13-native-20260912-01). Errored or missing episodes score 0 in the denominator.
