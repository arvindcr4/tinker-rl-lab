# MIN-REPORT-RL / 1.0 — a verifiable provenance protocol for GRPO/RLVR runs (P5)

**Problem.** GRPO/RLVR results are reported inconsistently and are hard to reproduce or trust.
The GRPO-Registry (P6) *catalogs* the 7-item MIN-REPORT-RL block per stack, but cataloging is
descriptive — nothing checks that a given run actually satisfies it. P5 makes the standard
**enforceable**: every run emits a signed provenance record, and an automated verifier grades it.

## The record (`*.provenance.json`)
1. `min_report_rl` — the seven reportable items (schema `$defs.min_report`): loss form, reference-KL,
   sampler backend, telemetry, group-size schedule, held-out split, decontamination.
2. `provenance` — anchors: `git_sha`, `code_file` + `code_hash` (sha256 of the file, 16-hex prefix), `reward_fn_hash` (full sha256 of the reward function source, recomputed by `verify`), `data_ref` (a dataset label, not a content hash),
   `config_hash`. These let a third party recompute and confirm nothing drifted. (Before 2026-10-16 `reward_fn_hash` was a copy of `code_hash` and `data_hash` hashed the label string; both placeholders were removed and the two records are no longer covered by their signatures until re-signed.)
3. `rigor` — `n_seeds`, `seeds`, `reports_heldout`, `heldout_disjoint`, `reports_ci`.
4. `results` — the reported effect(s).

## The verifier (`minreport.py verify`)
Grades three axes and returns a badge **A–F**:
- **Completeness** — all 7 MIN-REPORT items present & non-null.
- **Integrity** — provenance hashes present; `code_hash` **recomputed against the live file** (catches
  code drift since the run).
- **Rigor** — `n_seeds >= 3` (hard FAIL if not — the single-seed trap), held-out reported & disjoint,
  uncertainty reported (WARN if absent).

## Demonstrated (2026-07-06)
| run | grade | why |
|---|---|---|
| `campaign-multiseed` (this repo's powered re-test) | **A (100%)** | 3 seeds, held-out disjoint, code_hash verified |
| `p3-sweep-singleseed` (the original threaded sweep) | **B, FAIL on rigor** | n_seeds=1 → underpowered check fails |

The verifier **automatically flags** exactly the weakness kimi/codex caught by hand — turning a manual
review into a one-command gate. Novelty vs P6: P6 is a static catalog; P5 is an executable protocol +
integrity check (the `code_hash`-recompute and the seed-power gate do not exist in prior GRPO reporting work).

## Phase-2
Sign records (ed25519) so provenance is tamper-evident; wire `minreport.py verify --strict` into CI so a
run cannot be published below grade B; extend the rigor axis with bootstrap-CI presence and effect-size vs noise.
