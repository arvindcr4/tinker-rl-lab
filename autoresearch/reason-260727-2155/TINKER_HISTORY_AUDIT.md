# Tinker run-history audit for NeurIPS submission 36320

**Audit date:** 2026-07-28  
**Access mode:** read-only Tinker SDK `0.22.7`; the API key was read through a no-echo prompt, held only in a child process, removed after each query, and is not stored here.  
**Endpoints:** owned training-run inventory, user-checkpoint inventory, and session listing.

## Decision

Tinker history corroborates that the two directly logged W&B seed pairs, 45 and 46, are real training runs: four non-corrupt Qwen3-8B LoRA-rank-32 runs align one-to-one by time and each has 31 sampler checkpoints (`s0`--`s30`). Tinker also contains several earlier complete 30-step paired runs that are plausible sources for W&B's backfilled seeds 42--44.

It does **not** complete the five-seed provenance chain. The three W&B backfill pairs record no upstream Tinker IDs, and the Tinker training-run API exposes no seed, arm, algorithm, task, group size, batch size, learning rate, source commit, or W&B ID. Multiple complete and partial candidate pairs occur before the backfills. Seeds 42--44 therefore cannot be mapped uniquely to source runs or verified as the claimed treatments.

The correct conclusion is: the .926/.921 arithmetic is partially corroborated by real Tinker history, not fabricated, but remains inadmissible as a five-seed matched transfer experiment.

## Live W&B--Tinker alignment

| Claimed unit | W&B run/start | Tinker run | Checkpoint span | Tinker last request | Status |
|---|---|---|---|---|---|
| seed 45, canonical-labelled | `hyk5shwc`, 09:15:45 UTC | `79cb43b6-da9b-5904-a3df-e287a2e8c8ef:train:0` | `s0` 09:15:54 to `s30` 09:25:12; 31 checkpoints | 09:25:12 | non-corrupt |
| seed 45, REINFORCE-labelled | `c18pk9et`, 09:27:02 UTC | `79cb43b6-da9b-5904-a3df-e287a2e8c8ef:train:1` | `s0` 09:27:17 to `s30` 09:37:09; 31 checkpoints | 09:37:08 | non-corrupt |
| seed 46, canonical-labelled | `hekerdfz`, 09:39:26 UTC | `221b9765-4e2f-5582-b629-7973fe8907d9:train:0` | `s0` 09:39:41 to `s30` 09:48:51; 31 checkpoints | 09:48:50 | non-corrupt |
| seed 46, REINFORCE-labelled | `lwz6nhis`, 09:49:16 UTC | `221b9765-4e2f-5582-b629-7973fe8907d9:train:1` | `s0` 09:49:26 to `s30` 10:01:25; 31 checkpoints | 10:01:24 | non-corrupt |

The W&B runtimes end within seconds of the corresponding Tinker `s30` records. The shared base UUID and sequential `train:0`/`train:1` members also match the claimed paired execution structure. This is strong timestamp-based linkage for seeds 45--46, although Tinker still does not expose the estimator semantics or held-out predictions.

## Candidate sources for backfilled seeds 42--44

On 2026-07-24, the account has 19 Qwen3-8B training-run records. Before the directly linked seed-45 pair, the complete paired base UUIDs include:

| Base UUID | `train:0` | `train:1` | Interpretation limit |
|---|---:|---:|---|
| `8e1e681b-9aea-50de-a030-700202eeb2a9` | 31 checkpoints, complete 05:28 UTC | 31 checkpoints, complete 05:55 UTC | plausible paired source; no seed/arm link |
| `8f5f0b09-ad77-5331-8129-c58691d321bc` | 31 checkpoints, complete 07:29 UTC | 31 checkpoints, complete 07:43 UTC | plausible paired source; no seed/arm link |
| `8907c29b-1202-59b5-b6c3-bfe474fef11c` | 31 checkpoints, complete 07:54 UTC | 31 checkpoints, complete 08:05 UTC | plausible paired source; no seed/arm link |

There are also partial/retry structures: `1719375f-...` has 31/10 checkpoints, `15244ee0-...` has 31/7, and earlier attempts have only 1--9 checkpoints. Because W&B's backfill records contain neither a Tinker path nor an upstream run ID, assigning the three complete pairs to seeds 42, 43, and 44 would be inference from ordering, not provenance.

## Inventory refresh

- Owned training runs: **1,002**.
- User checkpoints: **1,657**.
- Qwen3-8B runs: **264**.
- Qwen3-8B runs on 2026-07-24: **19**.
- The older repository inventory reported 844 runs and 279 checkpoints through 2026-07-04; it is stale and should not be used for current counts.

## Rebuttal-safe correction

> Correction: our initial response over-described a five-seed Qwen3-8B comparison. Five W&B summaries reproduce 92.6% versus 92.1% and recomputed `p=.374`. The four live seed-45/46 W&B records strongly timestamp-align with four Tinker histories through step 30, and Tinker shows plausible earlier pairs. However, the three zero-runtime backfills contain no upstream IDs, while Tinker exposes no seed, arm, algorithm, or W&B metadata with which to identify them. We therefore withdraw the five-seed result as transfer evidence.

## Security note

The credential was supplied in chat and should be rotated. It is not reproduced in this artifact.
