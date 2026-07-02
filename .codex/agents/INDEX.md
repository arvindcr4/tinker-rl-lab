# .codex/agents/ — INDEX

**Purpose:** Role definitions (name, system prompt, temperature/max_tokens) for Codex subagents.

**Key files:**
- `explorer.toml` — maps codebase structure/dependencies (temp 0.3, 4k tokens).
- `reviewer.toml` — security + correctness review (temp 0.2, 4k tokens).
- `worker.toml` — implements features/fixes per module (temp 0.5, 8k tokens).

**Find it fast:**
- to tweak an agent persona/params → the matching `*.toml`
