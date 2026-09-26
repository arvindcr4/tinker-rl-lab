# E2 small-scale base-model arm — EffiBench substitute (2026-09-26)

**Result: pass@1 = 23/30 = 0.767 (Wilson95 0.591–0.882).** Secondary: wall-clock NET geomean 0.967 over the 23 passed (18 faster than canonical).

- Original lane: FrontierSWE (17 repo-level perf-optimization tasks). Not runnable here: no licence, x86-64 8-CPU/32 GB
  containers with multi-hour agent rollouts, and the host is aarch64 (Chrome crashes under QEMU).
- Substitute: EffiBench (alternatives map entry for E2), 30 problems chosen by `seed=20260926` (`raw/selection.json`).
- Gap: function-level LeetCode correctness + efficiency, not long-horizon repo optimization; does not approximate or bound FrontierSWE.
- Actor: Qwen/Qwen3.6-35B-A3B base, no adapter, Tinker, non-thinking, temperature 0, max_tokens 2048.
- Prompt: upstream `prompts/prompt.txt` + markdown description + small test cases (upstream format).
- Grader: upstream correctness rule (code + large `test_case` asserts, exit 0, empty stderr, 5 s) in `python:3.11-slim`, `--network none`.
- 2 items have broken upstream tests (canonical also fails); counted as failures. 23/28 on valid items.
- Cost: 42,805 Tinker tokens; $0 Modal; 0 Colab units.

## Rerun
```bash
cd outputs/e1_e14_small_scale_2026-09-26/E2
python3 code/run_e2.py select
set -a; source ../../../.env; set +a
/Users/arvind/.local/share/uv/tools/tinker/bin/python code/run_e2.py sample
python3 code/run_e2.py build && python3 code/run_e2.py exec && python3 code/run_e2.py score
```
Data: `raw/effibench_dataset_with_difficulty_and_algorithm.json` (upstream @d29e43bc; the HF mirror lacks `small_test_cases`).
