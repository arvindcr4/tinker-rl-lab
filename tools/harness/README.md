# Unattended harness

Outer-loop setup for multi-hour `muse` runs: fresh context per cycle, verify
gate per cycle, local commits, no pushes.

## Quick start

```bash
# 1. Optional: isolated worktree so the run never dirties your checkout
tools/harness/worktree-new.sh grind1 main

# 2. Fill in goal + tasks
$EDITOR tools/harness/handoff.json   # set "goal"
$EDITOR backlog.md                   # top item = next task (or use task_ledger.md)

# 3. Launch detached (survives terminal drops)
tools/harness/run-tmux.sh grind1 --max-iters 30
# with worktree: tools/harness/run-tmux.sh grind1 --workdir ../worktrees/grind1 --max-iters 30

# 4. Monitor / stop
tmux attach -t grind1
touch tools/harness/STOP   # graceful stop between iterations
```

## How a cycle works

1. `ralph.sh` runs one `muse exec --yolo` with the prompt in `PROMPT.md`
   (capped by `--max-model-steps` and a wall-clock timeout, default 30 min).
2. Changes? Run the verify gate (default `make lint-ruff && make test`).
3. Pass -> commit locally. Fail/timeout -> `git stash` (recoverable, never
   lost; see `git stash list`). No-change -> next task.
4. 3 consecutive failures stop the run for human review.

Logs: `tools/harness/logs/ralph-<iter>-<ts>.jsonl` (`muse exec --json`).

## Metric loops

For benchmark-driven iteration (loss/latency/pass-rate), use the
`autoresearch` skill instead (already installed): `/autoresearch` with
`Metric:` / `Verify:` — it adds plateau detection, checkpoints, rollback.

## Files

| File | Purpose |
| ---- | ------- |
| `ralph.sh` | outer loop driver |
| `run-tmux.sh` | detached tmux launcher |
| `worktree-new.sh` | isolated git worktree helper |
| `PROMPT.md` | per-cycle prompt template |
| `handoff.json` | cross-cycle state (agent-maintained) |
| `task_ledger.md` | completed-task log (agent-maintained) |

Safety notes: the loop never pushes, never rewrites history, and stashes
(rather than deletes) failed iterations. `--yolo` is scoped to the loop's
own `muse exec` calls on this dev checkout.
