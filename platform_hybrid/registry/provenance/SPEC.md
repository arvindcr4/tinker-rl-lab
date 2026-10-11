# MIN-REPORT-RL / 1.0 — a verifiable provenance protocol for GRPO/RLVR runs (P5)

**Problem.** GRPO/RLVR results are reported inconsistently and are hard to reproduce or trust.
The GRPO-Registry (P6) *catalogs* the 7-item MIN-REPORT-RL block per stack, but cataloging is
descriptive — nothing checks that a given run actually satisfies it. P5 makes the standard
**enforceable**: every run emits a signed provenance record, and an automated verifier grades it.

## The record (`*.provenance.json`)
1. `min_report_rl` — the seven reportable items (schema `$defs.min_report`): loss form, reference-KL,
   sampler backend, telemetry, group-size schedule, held-out split, decontamination.
2. `provenance` — declared anchors: `git_sha`, `code_file` + `code_hash`, `reward_fn_hash`,
   `data_hash`, `config_hash`. This verifier recomputes only `code_hash`; presence of
   other anchors does not establish that their underlying artifacts were checked.
   The generator's `data_hash` hashes the configured dataset descriptor, not the
   downloaded dataset bytes.
3. `rigor` — `n_seeds`, `seeds`, `reports_heldout`, `heldout_disjoint`, `reports_ci`.
4. `results` — the reported effect(s).

## The verifier (`minreport.py verify`)
Grades three axes and returns a badge **A–F**:
- **Completeness** — all 7 MIN-REPORT items have their expected nonempty top-level
  shape. This is not full nested-schema or result-evidence validation.
- **Integrity** — provenance hashes present; `code_hash` **recomputed against the live file** (catches
  code drift since the run).
- **Rigor** — integer `n_seeds >= 3` with the same number of distinct integer
  seed identifiers, explicit Boolean assertions for held-out reporting/disjointness
  and decontamination, and uncertainty reporting (WARN if not explicitly asserted).
  Strings such as `"false"`, numeric truthiness, and duplicate seeds cannot pass.
  These assertions are not independently verified result-level evidence.

Generation without evidence leaves disjointness unknown and decontamination
unverified. Missing named configuration files fail rather than silently generating
an apparently configured record. Frozen records are not automatically migrated.
Malformed JSON field types produce failed checks instead of comparison/access
crashes. `--strict` rejects every WARN/FAIL; it does not change the score or grade.

## Current checked-in records (rechecked 2026-10-11)

| run | grade | strict status and remaining limitation |
|---|---|---|
| `campaign-multiseed` | **A (95.7%)** | **FAIL**: uncertainty reporting is not asserted in `rigor.reports_ci` |
| `p3-sweep-singleseed` | **B (82.6%)** | **FAIL**: one seed and no explicit uncertainty-reporting assertion |

The campaign has a literal `results.sd`, but its scope and evidentiary connection
are not verified by this tool. It must not be promoted to verified uncertainty
merely because a number is present. The historical 100% description is not the
current strict outcome. No measured results or signed records were changed to
produce these scores. The verifier checks declared metadata and the live source
hash; a grade alone is not a scientific-validity or publishability verdict.

## Phase-2
Detached Ed25519 verification and strict MIN-REPORT checks are wired into CI as
separate operations. Remaining work includes independently binding reported
uncertainty/effects to result artifacts and checking effect size against noise.
Neither signatures nor the current badge establish those missing scientific claims.
