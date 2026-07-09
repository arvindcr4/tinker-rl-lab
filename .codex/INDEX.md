# .codex/ — INDEX

**Purpose:** Config for Codex (OpenAI) parallel subagent workflows on this repo.

**Key files:**
- `README.md` — subagent workflow patterns (Explore→Review→Implement, parallel dev, quality gate) + usage examples.
- `config.toml` — `max_threads = 6`, `max_depth = 1` (agents cannot spawn sub-agents).

**Subfolders:**
- `agents/` — per-role agent definitions: explorer, reviewer, worker (see its INDEX.md).

**Find it fast:**
- to change parallelism → `config.toml`
- to edit an agent role → `agents/`
