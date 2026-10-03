# Cycle {{ITER}} — unattended implementation turn

Working directory: {{WORKDIR}}
Task list: {{TASK_FILE}} (if missing, use tools/harness/task_ledger.md)
State: tools/harness/handoff.json and tools/harness/task_ledger.md

Do exactly ONE task this turn:

1. Read handoff.json, task_ledger.md, and {{TASK_FILE}}.
2. Pick the highest-priority incomplete task. If none remain, reply DONE and
   make no changes.
3. Implement it with the smallest correct change. Follow repo conventions
   (AGENTS.md, ruff, pytest).
4. Self-verify with: {{VERIFY_CMD}} — fix failures before finishing.
5. Update handoff.json (iteration, current_task, done list) and append a row
   to task_ledger.md. Keep both SHORT — they are the next cycle's memory.

Hard rules: never `git push`, never publish or deploy, never rewrite history,
never touch files outside {{WORKDIR}}. If blocked, write the blocker into
handoff.json `blocked` field and stop with no other changes.
