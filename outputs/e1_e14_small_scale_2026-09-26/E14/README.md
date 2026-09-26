# E14 small-scale — Omni-MATH (substitute for blocked FrontierMath)

**Result: 38/100 = 38.0%** (Wilson 95% 29.1–47.8%), graded by the official open judge **KbsdJames/Omni-Judge @de5bdca1**.
Actor: base Qwen3.6-35B-A3B (no adapter) on Tinker, non-thinking, temp 0, max_tokens 2048.

What ran: 100 rows `random.Random(20260926).sample(range(4428))` of the pinned test.jsonl (sha256 7c87be8e…).
The native actor prompt is the system line "You are an experienced educator in the field of MATHEMATICS." plus the problem.
Judge: the native `OmniJudgeTokenizer.get_context` prompt, vLLM 0.6.3 on a Modal A10G, temp 0, max 300 tokens, stop at eos/eot.
The Modal app stopped on completion and cost $0.0807. The score is the native `get_result.parse_report` 'Equivalence Judgement' == TRUE.
Unparsed reports count as wrong; there were 0.

Rerun:
```
set -a; source .env; set +a; cd outputs/e1_e14_small_scale_2026-09-26/E14
python code/run_actor.py                     # resumes from raw/actor/
modal run code/modal_omnijudge.py --inp raw/actor_generations.jsonl --out raw/omnijudge_raw.json
python3 code/score.py && python3 ../E8/code/finalize.py e14
```
Caveats: a new arm, not comparable to the 2271/4428 (51.31%) adapter figure.
73/100 answers hit the native 2048-token cap. The adapter actor hit that cap too, on ~87% of rows.
Omni-Judge marked 15 of the truncated answers correct from partial work. Of the 27 answers that finished, 23 were marked correct.
This is Omni-Judge, not the GPT-4o leaderboard judge.

Paired vLLM arm (trained vs base, same Modal H200 engine, same items/prompts/caps/graders): `vllm_trained/`, `vllm_base/`, `paired.json`.
Rerun: `ARM=trained|base ACTOR_PORT=.. JUDGE_PORT=.. bash ../E8/code/run_arm.sh` (+ Tinker judge shim for E10; Omni-Judge `modal run` for E14), then `python3 ../E8/code/paired.py E14`.
