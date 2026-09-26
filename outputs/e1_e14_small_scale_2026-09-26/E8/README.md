# E8 small-scale — LAB-Bench public (substitute for private LifeSciBench)

**Result: 26/80 = 32.5% accuracy** (Wilson 95% 23.2–43.4%), base Qwen3.6-35B-A3B (no adapter) on Tinker, non-thinking, temp 0, max_tokens 4096.
Per category (of 10): CloningScenarios 0, DbQA 1, FigQA 5, LitQA2 3, ProtocolQA 5, SeqQA 1, SuppQA 3, TableQA 8.

What ran: 10 questions per category (80), `random.Random(20260926).sample` over the prepared manifest's sorted task order
(`outputs/public_portfolio_2026-09-05/native_setup/prepared/`, pinned LAB-Bench @998a8e0, native choice shuffle seed 809).
Prompts, the chembench MCQ parser and `Evaluator.compute_metrics` are the native ones, loaded through
`zvf-program/flagship/public_portfolio_native.py`. FigQA/TableQA images are sent as Tinker image chunks.

Rerun (needs a venv with tinker==0.30.4, transformers, pillow, pydantic, python-dotenv, tqdm):
```
set -a; source .env; set +a
cd outputs/e1_e14_small_scale_2026-09-26/E8/code && python run_e8.py   # resumes from raw/*.json; rescoring is deterministic
python finalize.py e8                                                    # rewrites ../result.json from raw/
```
Files: `raw/<cat>__<id>.json` (raw completions), `raw/graded.jsonl`, `raw/summary.json`, `raw/transport_errors_first_attempt/`.

Caveats: a new arm, not comparable to the lost-adapter 1967/1967 (22.88%) run. 22/80 answers hit the token cap.
Two FigQA images were over Tinker's 2 MiB asset limit and were re-encoded as JPEG before being re-sent. Their first attempts
returned HTTP 400 with no model output. A ~5.7k-token smoke test is not included in the cost.
`code/tk.py` is the shared Tinker sampler. `code/finalize.py` writes the result.json files for E8, E10, E11 and E14.

Paired vLLM arm (trained vs base, same Modal H200 engine, same items/prompts/caps/graders): `vllm_trained/`, `vllm_base/`, `paired.json`.
Rerun: `ARM=trained|base ACTOR_PORT=.. JUDGE_PORT=.. bash ../E8/code/run_arm.sh` (+ Tinker judge shim for E10; Omni-Judge `modal run` for E14), then `python3 ../E8/code/paired.py E8`.
