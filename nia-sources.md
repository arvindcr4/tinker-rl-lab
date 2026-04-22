# Nia Indexed Sources — tinker-rl-lab

Track Nia-indexed sources used in this project so we don't re-list every time.
Last updated: 2026-04-22.

## Canonical references for BrowserGym × Tinker work

| Source | Type | Status | Use for |
|---|---|---|---|
| `ServiceNow/BrowserGym` | repo | ✅ indexed | env registration, action/observation API, MiniWoB/WebArena/WorkArena configs |
| `thinking-machines-lab/tinker-cookbook` | repo | ✅ indexed | `Env`/`EnvGroupBuilder`/`TrajectoryGroup` abstractions, RL loop, `verifiers_rl` override pattern |
| `tinker-docs.thinkingmachines.ai` (ID `4691bed4-0f6e-4273-84e4-650b15bf08cf`) | docs | ✅ indexed | RL loop tutorial, `Env` abstraction reference, quickstart |
| `web-arena-x/webarena` | repo | ✅ indexed | WebArena task set, reward validators |
| `Farama-Foundation/miniwob-plusplus` | repo | ⏳ indexing | MiniWoB static HTML corpus, task HTML files |

## Usage pattern

```python
# Check before web fetch
manage_resource(action='list', query='<keyword>')

# Targeted search on indexed repo
search(query='...', repositories=['ServiceNow/BrowserGym'])
nia_grep(source_type='repository', repository='ServiceNow/BrowserGym', pattern='...')
nia_read(source_type='repository', source_identifier='ServiceNow/BrowserGym:path/to/file.py')
```

## Research docs derived from these sources

- `docs/research/browsergym_tinker_plan.md` — recommended integration approach (2026-04-22)
