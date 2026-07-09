# MiniMax M3 Autoresearch Runner

Runs **MiniMax M3** as a long-horizon autonomous agent (Claude Agent SDK →
MiniMax's Anthropic-compatible endpoint) that turns TinkerRL-Bench's existing data
into paper contributions across 4 pillars, under the deli-autoresearch protocol.

## Layout
- `orchestrator.py` — the deli loop: fresh session/iteration, stall→pivot, budget,
  and a hard `can_use_tool` guardrail (worktree-only writes; blocks push/gh/rm -rf/
  secret access/external uploads).
- `run.sh` — loads `../.env.minimax`, launches the orchestrator.
- `watchdog.sh` — L0 resident guard; restarts a dead/stalled loop within budget.
- `monitor.sh` / `stop.sh` — status snapshot / graceful stop.
- `state/` — `task_spec.md`, `progress.json`, `iteration_log.jsonl`,
  `directions_tried.json`, `heartbeat.json`, `STOP` (gitignored).
- `.venv/` — SDK env (gitignored).

## Isolation
All work happens in the git worktree `/home/claude/tinker-rl-lab-minimax`
(branch `minimax-autoresearch`). Your primary checkout is never touched.

## Smoke (bounded)
```bash
cd minimax_autoresearch
set -a; source ../.env.minimax; set +a
.venv/bin/python orchestrator.py --worktree /home/claude/tinker-rl-lab-minimax \
  --state-dir ./state --max-hours 0.4 --max-iterations 1 --pillars 1 --round-cap 12
```

## Full 35h run (unattended)
```bash
cd minimax_autoresearch
nohup ./run.sh 35 all        > logs/run.$(date +%s).out      2>&1 &
nohup ./watchdog.sh 35       > logs/watchdog.$(date +%s).out 2>&1 &
```
Watch: `./monitor.sh` · Stop: `./stop.sh` (graceful) or `./stop.sh --now`.

## Notes
- **Cost:** 35h of continuous MiniMax tool-use consumes a large chunk of the Token
  Plan quota. `progress.json.total_cost_usd` tracks reported spend; cap with
  `orchestrator.py --max-cost-usd N`.
- **Papers via MCP:** local firecrawl (arXiv research tools) + brave/serper + context7,
  reused from `~/.claude.json` at runtime (keys never copied). Hosted PubMed/Scholar
  are disabled under MiniMax auth — arXiv/firecrawl is used instead.
- **Key hygiene:** `.env.minimax` is gitignored/`chmod 600`. Rotate it after the run
  (it was pasted in chat once).
