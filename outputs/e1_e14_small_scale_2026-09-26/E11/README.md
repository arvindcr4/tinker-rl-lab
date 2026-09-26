# E11 small-scale — VerilogEval (original benchmark, public subset)

**Result: 35/50 = 70.0% pass@1** (Wilson 95% 56.3–80.9%). code-complete-iccad2023: 20/25; spec-to-rtl: 15/25.
Actor: base Qwen3.6-35B-A3B (no adapter) on Tinker, non-thinking, temp 0, max_tokens 4096, 1 sample/problem, no retries.

What ran: 25 problems per framing, `random.Random(20260926).sample` over the sorted `*_prompt.txt` list of
NVlabs verilog-eval @c498220d (`outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d`). The raw prompt file is the single
user message, as in the retained driver. `e11_model_run.extract_module` and `write_sample` lay out the samples, and
`e11_paid_run_driver.configure_build/run_harness` run the native `gmake sv-iv-test` with the repo-pinned iverilog-12.
Pass means the native test-bench log reads `Mismatches: 0 in N samples`. Prob099 (unscoreable) was not selected, so both denominators are 50.

Rerun:
```
set -a; source .env; set +a
cd outputs/e1_e14_small_scale_2026-09-26/E11/code && python run_e11.py   # reuses raw/samples/*; rebuilds raw/build/
python ../../E8/code/finalize.py e11
```
Caveats: a new arm. The retained receipt (129/312 = 41.35%) used thinking, temp 0.2 and all 312 items, so it is not comparable.
`sv-iv-analyze` (summary.csv) fails locally because langchain is missing. Verdicts therefore come from the per-problem test logs, as in the retained driver.
The first harness pass scored 0/50 only because `python` was missing from PATH, so make aborted before simulation.
The fix is the `code/bin/python` shim. The samples were identical, and the failed log is in raw/.

Paired vLLM arm (trained vs base, same Modal H200 engine, same items/prompts/caps/graders): `vllm_trained/`, `vllm_base/`, `paired.json`.
Rerun: `ARM=trained|base ACTOR_PORT=.. JUDGE_PORT=.. bash ../E8/code/run_arm.sh` (+ Tinker judge shim for E10; Omni-Judge `modal run` for E14), then `python3 ../E8/code/paired.py E11`.
