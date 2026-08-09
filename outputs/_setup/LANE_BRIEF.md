# Lane brief — shared rules for the E1–E14 lanes

Every lane is an independent session working in
`/Users/arvind/Developer/agentic_repos/tinker-rl-lab`. Read
`outputs/_setup/E1_E14_ASSET_INVENTORY.md` first — it is the verified local state
as of 2026-08-09, and it supersedes the "next actions" in
`outputs/PAVLOV_E1_E14_LOCAL_SPRINT_2026-08-09.md`.

## Scientific integrity (non-negotiable)

1. **Never fabricate a score.** If the suite cannot execute, `score` stays `null`
   and `status` stays `BLOCKED`. A blocked receipt is a valid deliverable.
2. **Never substitute a related benchmark.** GSM8K, MATH, HumanEval, BFCL,
   WebArena, xLAM, Glaive and friends are not the exact suite. A related
   benchmark passing is not this suite passing.
3. **Harness validation is not a model score.** Running the harness against a
   gold/reference answer proves the plumbing works. Label it
   `harness_validation` with `is_model_score: false`. Never promote it.
4. **Report what actually happened.** If a step fails, say so with the output.

## Cost and side-effect boundary

- No paid Tinker calls. `TINKER_API_KEY` exists in `.env` — do not use it.
- No W&B runs, no Hugging Face checkpoint pushes, no PRs.
- No paid model API calls. Stop at the model-artifact boundary and report the
  estimated cost of crossing it.
- No `git commit`, `git push`, `git checkout`, or branch operations. Write files.

## Shared-resource discipline

Docker is one shared **Colima** VM (8 CPU / 16.5 GiB). 63 GiB host disk free,
shared across all 14 lanes.

- Do not change Colima config. Do not run `docker system prune`.
- Do not delete `ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt:v4` — that
  is lane E2's 14.8 GB asset.
- Run `df -h /` before any download or image pull over 2 GB. **If free space is
  under 15 GiB, stop and report instead of pulling.**
- Before a pull/build over 2 GB, take the shared mutex:
  `mkdir outputs/_setup/docker.lock` (atomic — fails if held). Release with
  `rmdir outputs/_setup/docker.lock` when done, including on failure. If the
  lock is held, do other work and retry later; if it is still held after ~20
  minutes, report rather than forcing it.

## File ownership

Confine writes to your own `outputs/e<N>_*/**` plus the flagship files named in
your lane prompt. Do not edit another lane's files. Where an adapter is shared
(E10/E14, E12/E13), the named owner edits it and the other lane creates new
files.

## Credentials nobody has

`HF_TOKEN`, `GEMINI_API_KEY`, OpenRouter/Anthropic agent keys. Do not invent
them, do not prompt for them. Record them as required-credential blockers with
the exact command the user needs to run.

Present: Kaggle (`~/.kaggle/kaggle.json`), W&B (`~/.netrc`), Tinker (`.env`).

## Deliverables (always, even when blocked)

1. `outputs/e<N>_<name>/lane_status_2026-08-09.md` — at most one page: what now
   runs, the exact commands that work, what remains, and the single next action.
2. A receipt JSON beside it: `status` ∈ `RUNNING` / `PARTIAL` / `BLOCKED`, a
   null-or-real score, evidence paths, and blockers with the external receipt
   each one needs.

Final message back: status, what executed, what is still missing.
