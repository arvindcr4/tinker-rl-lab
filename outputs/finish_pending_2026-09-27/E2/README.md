# E2 — CORE-Bench (codeocean_hard), replacement scope

**Result: 27/45 tasks fully correct = 0.600 (Wilson 95% CI 0.455–0.730).** Scored with the native grader (`benchmark.evaluations.eval_result_json` / `score_results`). Questions: 53/79 correct (written 28/33, vision 25/46). This is a replacement-scope number. Don't pool it with original-contract numbers.

- Actor: `pavlov-public-portfolio-bf16` (seed809 merged), temperature 0, non-thinking. Authorization: `../AUTHORIZATION.json` (E2 cap $20).
- Compute: one GCP VM per task (e2-highmem-2 for CPU tasks, n1-highmem-4 + T4 for GPU tasks; Ubuntu Pro 20.04, us-central1). Spot VMs first, with one STANDARD rerun if a spot VM was preempted.
- Runs: every task had one attempt, with the native 8100 s budget and a 150-turn cap. Outcomes: 32 called finish, 10 hit max_turns, 3 timed out. Tasks with errors, timeouts or no report count as failures.
- Spend: about $8.83 in GCP VMs (list-price estimate from `raw/*vm_ledger*.jsonl`, including all aborted attempts). Actor inference is billed to the shared endpoint cap.
- Infra restarts (the aborted attempts are kept in `raw/aborted_*`):
  - utf8 decode bug
  - driver bug
  - local DNS outage (30 tasks never got a VM)
  - driver-process death at ~07:44Z (10 tasks were in flight; their agent state was lost and their 10 VMs were deleted)
  
  The affected tasks were rerun from scratch. `raw/rerun2_run.log` is the 30-task rerun. No task was rerun because of its score.
- Leak check: the agent's commands contained no corebench, princeton, huggingface or codeocean URLs. `prep_task.py` strips `results/` and deletes the unstripped capsule before the agent starts.
- Cleanup: all `e2cb-*` VMs and disks were deleted and their absence confirmed with gcloud. The lane service account `e2-corebench-vm` was deleted.
- Files:
  - `result.json` (summary and per-task status)
  - `per_task.json`
  - `native_results_codeocean_hard.json`
  - `raw/<capsule>/` (trace, task, fetched report, meta)
  - `code/` (`run_corebench.py`, `prep_task.py`, `finalize.py`)
  
  `dataset/` holds the answer keys and is git-ignored.
