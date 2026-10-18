---
name: hyperresearch-8-corpus-critic
description: >
  Step 8 of the hyperresearch V8 pipeline. Spawns one corpus-critic subagent
  to identify "what source, if found, would overturn the current
  direction?" gaps, then runs a targeted fetch wave to fill them.
  Highest-leverage intervention point: corrections applied before drafting
  cost nothing; corrections applied after drafting require patches.
  Invoked via Skill tool from the entry skill (full tier only).
---

# Step 8 — Pre-draft corpus critic (targeted gap-fill)

**Tier gate:** `full` tier ONLY. Skip for `light`.

**Goal:** before drafting, ask "what source, if found, would overturn the current direction?" and run a targeted fetch wave to fill the most dangerous gaps. This is the highest-leverage intervention point in the pipeline — corrections applied before drafting cost nothing; corrections applied after drafting require patches and risk structural drift.

---

## Recover state

Read these inputs:
- `research/scaffold.md` — vault_tag
- `research/comparisons.md` — cross-locus tensions
- `research/loci.json` — scored loci
- `research/temp/source-tensions.json` — expert disagreements
- `research/prompt-decomposition.json` — specifically the `time_periods` array

---

## Pre-flight: period-pinned primary-source coverage check

**Run this BEFORE spawning the corpus-critic subagent.** If `prompt-decomposition.json -> time_periods` is non-empty, walk every entry and verify the vault contains a primary source filed *for that exact period* — not "most recent", not narrative commentary, not earnings-call transcripts standing in for tabular filings.

For each `time_period` entry:

1. Search the vault for a primary source matching the `primary_source` description and the `issuer`:
   ```bash
   PYTHONIOENCODING=utf-8 {hpr_path} search "<period> <issuer>" --tag <vault_tag> --include-body -j
   ```
2. Open the candidate notes (`note show <id> -j`) and verify the document's actual reporting period — the filing must cover the SPECIFIC period named in the prompt, not an adjacent one. A Q1 2025 10-Q does NOT satisfy "Q3 2024" — different period, different tabular data.
3. **If the period-pinned filing is missing, add it to `research/corpus-preflight-gaps.json` as a `priority: critical` gap of type `period-pinned-primary` BEFORE spawning the corpus-critic subagent.** Schema:
   ```json
   {{
     "id": "preflight-<run_id>-<unique ordinal>",
     "type": "period-pinned-primary",
     "target_position": "<period> exact figures for <issuer>",
     "search_queries": [
       "site:sec.gov 10-Q \"period ended September 30, 2024\" <issuer>",
       "<issuer> Q3 2024 10-Q filing PDF"
     ],
     "source_type": "primary-filing",
     "priority": "critical",
     "rationale": "Prompt names <period>; vault has no filing covering that exact period. Tabular line items only exist in the period-pinned filing. Without it, the draft will paraphrase rounded numbers from earnings calls and miss the rubric's exact figures."
   }}
   ```

Assign every gap a nonempty unique `id` when created, persist it, and never regenerate it during retries. Preflight IDs use `preflight-<run_id>-<ordinal>`; agent IDs use `agent-<run_id>-<ordinal>` to prevent collisions. Save an empty array when no preflight gaps exist.

**Before spawning the corpus critic, dispatch and await the targeted fetchers for all preflight gaps** using the fetcher template below. Persist results keyed by `gap_id` in `research/corpus-preflight-results.json`, including source note IDs, verified reporting period, and `fetched` or `unfindable` status with attempted queries and failure reason. Newly fetched filings must be indexed and available to the critic. Unfindable filings remain explicit critical gaps.

This preflight fetch wave attempts these filings BEFORE the corpus-critic starts its broader gap analysis. This ordering matters: numerical-precision misses are the largest single category of avoidable factual-accuracy failures.

---

## Procedure

1. **Spawn ONE `hyperresearch-corpus-critic` subagent** (Sonnet).

   **Spawn template:**
   ```
   subagent_type: hyperresearch-corpus-critic
   prompt: |
     RESEARCH QUERY (verbatim, gospel):
     > {{paste research/query-<vault_tag>.md body}}

     QUERY FILE: research/query-<vault_tag>.md

     PIPELINE POSITION: You are step 8 of the hyperresearch V8 pipeline.
     Step 6 produced research/comparisons.md. After you return, the
     orchestrator runs a targeted fetch wave, then step 10 drafts.

     YOUR INPUTS:
     - corpus_tag: <vault_tag>
     - comparisons_path: research/comparisons.md
     - loci_path: research/loci.json
     - output_path: research/corpus-agent-gaps.json
     - preflight_gaps_path: research/corpus-preflight-gaps.json
     - preflight_results_path: research/corpus-preflight-results.json
     OUTPUT CONTRACT: Emit a JSON array of gap objects. Every object requires
     a nonempty unique stable id (agent-<run_id>-<ordinal>), type
     (overturning/strengthening/independent-verification), target_position,
     search_queries (array), source_type, priority (critical/high), and rationale.
     Persist IDs across retries. Do not duplicate or rename preflight IDs;
     refer to existing preflight gaps through related_gap_ids instead.
   ```

2. **Contract v2: merge separate preflight and agent gap outputs** by stable gap ID into `research/corpus-critic-gaps.json`. Assert every mandatory preflight ID survives, preserving its critical priority. Reject missing, duplicate or conflicting IDs before continuing. Join preflight fetch results by `gap_id` without dropping gaps or lowering their critical priority. Read the merged gaps. Each gap has a `priority` (critical / high) and a `type` (period-pinned-primary / overturning / strengthening / independent-verification).

3. **Remaining targeted fetch wave.** Reuse the recorded preflight results; fetch only unattempted agent gaps or explicitly justified retries of unresolved preflight gaps. Spawn **2–4 fetcher subagents** (Sonnet) to search for and fetch the sources identified in the gaps.

   **Spawn template:**
   ```
   subagent_type: hyperresearch-fetcher
   prompt: |
     RESEARCH QUERY (verbatim, gospel):
     > {{paste research/query-<vault_tag>.md body}}

     QUERY FILE: research/query-<vault_tag>.md

     PIPELINE POSITION: You are a step 8 fetcher (corpus-critic gap-fill)
     of the hyperresearch V8 pipeline. The corpus critic identified specific
     gaps; you fetch sources targeting those gaps. After you return, the
     orchestrator updates comparisons.md based on what you found.

     YOUR INPUTS:
     - vault_tag: <vault_tag>
     - search_queries: [<gap.search_queries>]
     - source_type: <gap.source_type>
     - gap_id: <gap.id>
   ```

4. **Assess results.**
   - **Overturning source found:** re-read the relevant committed position from the interim note. If the new source genuinely undercuts it, update `research/comparisons.md` to note the weakened position — the draft will handle it with appropriate calibration. Do NOT re-run the full depth investigation; adjust the position's confidence level.
   - **Overturning source NOT found:** record only the bounded search outcome; confidence does not automatically increase. Note this in `comparisons.md` — "adversarial search for counter-evidence to [position] returned no substantive challenges."
   - **Strengthening/verification source found:** note the additional support in `comparisons.md`. The draft can assert more confidently.

5. **Log results** to `research/temp/corpus-critic-results.md`:
   - Each gap: what was searched, what was found (or not), how it affects the committed positions
   - Updated confidence levels for any positions that changed

---

## Exit criterion

- Preflight fetch results exist, with attempts completed and available before critic dispatch
- Every producer emits stable unique IDs; all mandatory preflight IDs survive the merged gap list and result join
- `research/corpus-critic-gaps.json` exists
- All critical gaps attempted (fetched or documented as unfindable)
- `research/temp/corpus-critic-results.md` exists
- `research/comparisons.md` updated with confidence/strengthening/overturning notes

---

## Next step

Return to the entry skill (`hyperresearch`). Invoke step 9:

```
Skill(skill: "hyperresearch-9-evidence-digest")
```
