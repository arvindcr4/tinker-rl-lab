# E1 small-scale base-model arm — SWE-bench Pro (2026-09-26)

**Result: resolved 0/10 = 0.0 (Wilson95 0.000–0.278), native evaluator.**

- Benchmark: original SWE-bench Pro public test split (pinned dataset @7ab51149, evaluator @ca10a60a), 10 instances
  chosen by `seed=20260926` over `dataset_test_731.jsonl` order (`raw/selection.json`).
- Actor: Qwen/Qwen3.6-35B-A3B base, no adapter, Tinker, non-thinking, temperature 0, max_tokens 8192.
- Flow (agentless, same code as the original lane): in the native x86 image at base commit, pick files by
  path mentions + `git grep` of issue identifiers (<=24 files, <=96k chars), prompt for a unified diff, extract it.
- Grading: `zvf-program/flagship/e1_swe_bench_pro_full_eval.py` (pinned upstream evaluator, digest-pinned images,
  Modal sandboxes, `--block_network`). `raw/evaluation/eval_results.json` is the source of the number.
- Outcome: 9/10 valid-looking diffs, 1 hit the token cap. Diagnostic `git apply --check`: 0/9 apply (8 bad hunk
  counts, 1 context mismatch). The zero reflects diff-format failure more than test logic.
- Cost: 286,553 Tinker tokens; ~$0.08 Modal; 0 Colab units.

## Rerun
```bash
cd outputs/e1_e14_small_scale_2026-09-26/E1
python3 code/run_e1.py select
uv run --no-project --with modal==1.5.4 python code/run_e1.py sources
set -a; source ../../../.env; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python code/run_e1.py sample
python3 code/run_e1.py eval && python3 code/run_e1.py score
uv run --no-project --with modal==1.5.4 python code/apply_check.py   # optional diagnostic
```
