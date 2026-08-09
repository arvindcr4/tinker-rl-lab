# E8 — LifeSciBench — lane status 2026-08-09

**Status: BLOCKED. Score: null.** All six blockers are access-gated and none is
reachable locally. Everything on the local side of the boundary is now done.

## 1. The public record (established from primary sources)

OpenAI published LifeSciBench on **2026-06-17**
(`https://openai.com/index/introducing-life-sci-bench/`) with a preprint
(`https://cdn.openai.com/pdf/b4299379-0a97-4ffa-8b9b-c3fbb299caa9/lifescibench_preprint.pdf`,
OpenAI + Tacit Labs, 19 authors).

**Documented:** 750 expert-authored tasks; 7 workflow categories x 7 biological
domains; 19,020 rubric criteria (~25/task); 1,062 attached artifacts; 53% of
tasks need >=1 artifact; 37 tasks carry prompt-provided URLs; 173 Ph.D. task
authors; 453 independent validation reviewers. Scoring is **normalized rubric
score** (awarded points / total points, problem-weighted) and **task pass rate**
(fraction of tasks at or above a **70%** threshold). Protocol is **single-turn**
with unrestricted internet browsing. Provider-reported results: GPT-Rosalind
0.576 / 36.1%, GPT-5.5 0.519 / 25.7%, Gemini 3.1 Pro 0.515 / 23.6%, GPT-5.4
0.479 / 20.7%, Grok 4.3 0.399 / 13.0%.

**Announced but not published — the answer to "is there a public slice?":**
**No. There is no public subset of any size.** No tasks, no rubrics, no
artifacts, no harness, no grader, no task-ID list, no repository, no dataset
card. Preprint appendix A.5 is the only release statement and it is a
restriction: release "may be limited by licensing, privacy, proprietary
information, or biological safety considerations," and content was excluded
where dissemination "could create biological safety risks."

**Two further gaps worth recording.** The preprint contains **zero** mentions of
contamination, decontamination, leakage, or held-out controls — checked by
full-text search. And OpenAI publishes **no request route for benchmark
materials**: the announcement's "Request access" button goes to
`openai.com/form/life-sciences-access/`, which is **GPT-Rosalind model access**,
and "Join as a contributor" goes to a Ph.D. recruitment form. Neither asks for
data. The preprint lists no correspondence address.

## 2. What executed

```bash
tmp=$(mktemp -d /tmp/e8-lifescibench.XXXXXX); python3 -m venv "$tmp/venv"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program
"$tmp/venv/bin/python" -m unittest flagship.test_pavlov_lifescibench_eval_adapter -v   # 25 pass, exit 0
"$tmp/venv/bin/python" zvf-program/flagship/pavlov_lifescibench_eval_adapter.py --json # exit 1, fails closed
```

Preflight emits 7 errors mapping onto the 6 blockers (licence accounts for two):

| Preflight error | Blocker |
|---|---|
| `dataset.revision must be an immutable 40-hex or sha256` | `E8_DATASET_REVISION` |
| `dataset license must be explicitly approved` + `license_id must be pinned` | `E8_LICENSE_RECEIPT` |
| `native_environment.revision must be an immutable 40-hex or sha256` | `E8_NATIVE_ENVIRONMENT_REVISION` |
| `native_verifier.revision must be an immutable 40-hex or sha256` | `E8_NATIVE_VERIFIER_REVISION` |
| `task_manifest must contain immutable evaluation task rows` | `E8_TASK_MANIFEST` |
| `train_split_manifest_hash must be a lowercase SHA-256 digest` | `E8_HELDOUT_DISJOINTNESS` |

## 3. What is now complete locally

Added to `zvf-program/flagship/pavlov_lifescibench_eval_adapter.py` (**purely
additive**: 498 insertions, 0 deletions vs HEAD):

- `task_id_hash` — deterministic, revision-bound task hashing
- `build_task_row` / `build_split_manifest` — validated rows, sealed manifest hash
- `build_heldout_proof` — raises on task-ID **or family** overlap, emits `proof_hash`
- `build_pinned_boundary` — the single call the campaign makes when access lands
- `build_e8_result_receipt` — canonical receipt emission with `receipt_hash`
- `build_synthetic_fixture` — end-to-end fixture, every value marked

Published taxonomy is now carried on task rows as `workflow` and `bio_domain`
and validated against the provider's 7+7 categories. `ALLOWED_DOMAINS` is left
alone: it is the campaign capability tag set from
`pavlovs_domain_contract.json`, a different axis.

**The synthetic fixture cannot become a result.** Every identifier carries
`SYNTHETIC-NOT-LIFESCIBENCH`; both validators reject any payload containing that
marker; `task_success` is `False` on every row and every metric is `0.0`. The
demo log shows the marker is the **only** substantive error — every hashing,
manifest, disjointness and receipt-sealing check passes underneath it, which is
what proves the plumbing is complete. `test_builder_chain_is_schema_complete_without_the_synthetic_marker`
runs the same chain with neutral identifiers and validates clean.

Tests: 13 pre-existing + 12 new = **25 pass**.

## 4. Schema reconciliation needed before any real run

Recorded, not silently fixed — these touch a shared contract.

1. The boundary models a **stateful agentic environment** (`stateful=true`,
   `reset_per_task=true`, observation-action-artifact schema). LifeSciBench is
   **single-turn free-response, rubric-graded**. `pavlovs_domain_contract.json`
   declares `stateful: true` for this suite, so this is a cross-lane decision.
2. `WANDB_REQUIRED_METRICS` are `success_rate` / `reward_mean` /
   `action_count_mean`. The real metrics are normalized rubric score and pass
   rate at 0.70. **These keys must be remapped or the logged numbers will not
   mean what their names say.**

## 5. Deliverables

| File | What |
|---|---|
| `lane_receipt_2026-08-09.json` | BLOCKED, score null, 6 blockers, artifact digests |
| `ACCESS_REQUEST_lifescibench_2026-08-09.md` | Send-as-is request + routing analysis |
| `focused_tests_2026-08-09.log` | 25 passing tests |
| `adapter_preflight_2026-08-09.json` | Fail-closed preflight, exit 1 |
| `synthetic_fixture_demo_2026-08-09.log` | Fixture hashes + sole-error proof |

## 6. Single next action

Send `ACCESS_REQUEST_lifescibench_2026-08-09.md` via
`https://openai.com/contact-sales/`. Expect refusal or silence given appendix
A.5 — **a documented refusal is still a win**, converting an open blocker into a
closed one. Item 5 (task-ID list) and item 6 (disjointness statement) are the
cheapest asks and worth pressing even if items 1–4 are declined.

**Do not** submit the GPT-Rosalind form solely to reach the benchmark team; it
is scoped to model access and triggers government-ID verification.
