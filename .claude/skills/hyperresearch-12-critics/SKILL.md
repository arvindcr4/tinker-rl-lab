---
name: hyperresearch-12-critics
description: >
  Step 12 of the hyperresearch V8 pipeline. Spawns 4 adversarial critics
  in parallel against the synthesized final report from step 11. Each
  critic produces an independent findings JSON that the patcher (step 14)
  consumes. Critics never modify the draft directly. Invoked via Skill
  tool from the entry skill (full tier only).
---

# Step 12 — Adversarial critique (parallel critics)

**Tier gate:** SKIP entirely for `light` tier — proceed directly to step 15 (polish). For `full` tier: spawn all 4 critics.

**Goal:** independent findings lists against the synthesized final report, each from a different adversarial angle. Critics complement rather than duplicate.

---

## Recover state

Read these inputs:
- `research/scaffold.md` — vault_tag
- `research/prompt-decomposition.json` — pipeline_tier, atomic items
- `research/notes/final_report_<vault_tag>.md` — merged draft from step 10
- `research/query-<vault_tag>.md` — canonical research query

---

## Procedure

1. **Spawn all 4 critics in parallel.** In ONE message:
   - `hyperresearch-dialectic-critic` → `research/critic-findings-dialectic.json` (counter-evidence the draft missed or straw-manned)
   - `hyperresearch-depth-critic` → `research/critic-findings-depth.json` (shallow spots where interim notes could fill substance)
   - `hyperresearch-width-critic` → `research/critic-findings-width.json` (corpus clusters the draft ignores despite evidence)
   - `hyperresearch-instruction-critic` → `research/critic-findings-instruction.json` (atomic items from the decomposition that the draft missed, under-covered, reordered, or reformatted)

2. **Pass each critic** (standard 3-piece contract):
   ```
   subagent_type: hyperresearch-<critic-name>-critic
   prompt: |
     RESEARCH QUERY (verbatim, gospel):
     > {{paste research/query-<vault_tag>.md body}}

     QUERY FILE: research/query-<vault_tag>.md

     PIPELINE POSITION: You are step 12 (<critic-name> critic) of the
     hyperresearch V8 pipeline. Step 11 (synthesizer) produced the final report at
     research/notes/final_report_<vault_tag>.md. After you return, step 13 may run a
     gap-fetch wave, then step 14 (patcher) applies findings as Edit hunks.

     YOUR INPUTS:
     - draft_path: research/notes/final_report_<vault_tag>.md
     - output_path: research/critic-findings-<critic-name>.json
     - vault_tag: <vault_tag>
     - decomposition_path: research/prompt-decomposition.json   (instruction-critic only)
   ```

3. **Wait for all critics.** If one fails, preserve its failed envelope and mark review_status: incomplete. Partial analysis may continue but cannot pass the final review gate. **Do NOT skip the instruction-critic specifically** — it's the only critic measuring prompt adherence, which is the dimension with the widest variance.

4. **Do not read the findings yourself and apply them.** The patcher (step 14) reads the findings. Your job is to hand them to the patcher — AFTER step 13 (gap-fetch) runs.

---

## Exit criterion

- All 4 critic findings JSONs exist (`research/critic-findings-<name>.json`)
- Each is a valid v2 JSON envelope: `status: completed` requires a `findings` array; `status: failed` requires `findings: null` and a nonempty `error`
- Failed envelopes satisfy artifact collection only: mark `review_status: incomplete`; they never satisfy completed review or the final gate

---

## Next step

Return to the entry skill (`hyperresearch`). Invoke step 13:

```
Skill(skill: "hyperresearch-13-gap-fetch")
```

## Final review contract v2

All artifacts belong under research/runs/<run_id>/; legacy paths above are logical names relative to that run. Carry run_id, query_sha256 and corpus_sha256 in each artifact. Critic envelopes also carry critic, status (completed/failed), draft_sha256, findings (array only for completed), and error for failures. Never accept legacy bare findings as completed review.

After ALL readability edits in step 16, rerun the integrity and lint checks and required critics against the delivered bytes. Write a schema_version: 2 manifest with tier, run_id, query_sha256, corpus_sha256, final_sha256 and lint_status: passed. Run `python tools/research_review_gate.py --report <final-report> --manifest <run-manifest> --reviews <required-critic-files>` and preserve its receipt. Any failure means review_status: incomplete and prevents a shippable designation. Light tier requires the final hash/lint manifest but no critics. No edit may follow a successful receipt without invalidating it.
