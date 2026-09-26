# E12 (AppBench): public tasks, not held-out. Rubric pass rate 122/151 = 0.808 (judge v2). Judge v1 gave 81/151 = 0.536

**What ran:** the 6 public AppBench tasks from `outputs/e12_appbench/hf_dataset/AppBench vExternal.csv` (151 rubric items: 24/33/22/25/23/24).
The actor is the `Qwen/Qwen3.6-35B-A3B` base model (no adapter) on Tinker, with thinking off and temperature 0. For each task it received the Prompt plus the CLI addendum and produced the whole Next.js + Supabase app as source files in a single response (max 16384 tokens) → `raw/gen_<n>.txt`.
**Judge:** the same base model on Tinker (self-judge, since `.env` holds no Gemini key), with thinking off, temperature 0 and a 6144-token limit. It did a static review of the generated source against each rubric item (PASS only for concrete end-to-end implementation). Its outputs are `raw/judge_v2_<n>.json` (primary) and `raw/judge_<n>.json` (v1).
There is no build, no deploy and no browser testing. AppBench's own protocol grades a running app with two human graders over 3 attempts, so these numbers are not comparable to it.

| task | items | v2 pass | v1 pass |
|---|---|---|---|
| 1 Financial Dashboard | 24 | 18 | 15 |
| 2 Hospital Dashboard | 33 | 29 | 27 |
| 3 Legal Assistant | 22 | 19 | 19 |
| 4 Pharmacy System | 25 | 25 | 0 |
| 5 Drawing Game | 23 | 16 | 20 |
| 6 Rental Booking (gen truncated at cap) | 24 | 15 | 0 |

**Why v2 is primary:** v1 ignored the requested per-item reasons and returned bare PASS/FAIL. v2 requires evidence first, as one `n | evidence | PASS/FAIL` line per item. Treat the 0.54 to 0.81 gap as the real uncertainty; the self-judge is lenient and unstable.
**Cost:** Tinker used 270,566 tokens (`raw/calls.jsonl`). Modal and Colab were not used.
**Rerun:** `set -a; source ../../../.env; set +a; cd code; /Users/arvind/.local/share/uv/tools/tinker/bin/python run_e12.py && /Users/arvind/.local/share/uv/tools/tinker/bin/python judge_v2.py`. The scripts resume from existing raw files; delete those files to resample.
