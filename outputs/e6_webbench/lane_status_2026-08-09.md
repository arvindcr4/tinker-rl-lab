# E6 — `webbench_eval` (Halluminate WebBench) — lane status 2026-08-09

**Status: BLOCKED. Score: `null`. 0 tasks executed, 0 paid calls, no network.**

The blocker is unchanged and will not move locally: the Halluminate live browser
environment and its native verifier are not public. No local browser harness was
substituted, and none will be — a Playwright/BrowserGym/WebArena run is not
`webbench_eval`.

What did move: the public 2,647-task set now has an authoritative, reproducible,
unit-tested identity. That was the real missing piece, and it is done.

## What now runs

```bash
cd /Users/arvind/Developer/agentic_repos/tinker-rl-lab

# 1. Derive task index, aggregate hashes, split manifest, disjointness proof
python3 zvf-program/flagship/pavlov_webbench_eval_adapter.py --build-split \
  --dataset outputs/e6_webbench/webbenchfinal.csv \
  --training-task-manifest outputs/e6_webbench/eval_only_training_task_manifest.json \
  --license outputs/e6_webbench/WEBBENCH_LICENSE \
  --out-dir outputs/e6_webbench/split                       # exit 0

# 2. Boundary check over the full 2,647-task evaluation manifest
python3 zvf-program/flagship/pavlov_webbench_eval_adapter.py \
  --manifest outputs/e6_webbench/boundary_manifest_full_eval.json   # exit 1, BLOCKED

# 3. Runner preflight (zero-cost, offline)
python3 zvf-program/flagship/webbench_eval.py \
  --dataset outputs/e6_webbench/webbenchfinal.csv \
  --training-task-manifest outputs/e6_webbench/eval_only_training_task_manifest.json \
  --env outputs/e6_webbench/lane.w3we70/venv \
  --receipt outputs/e6_webbench/preflight_receipt_2026-08-09.json   # exit 2, BLOCKED

# 4. Tests
cd zvf-program && python3 -m pytest flagship/test_pavlov_webbench_eval_adapter.py -v
# 35 passed, 6 subtests passed
```

## Split-manifest artifacts (new)

| Artifact | Contents |
|---|---|
| `split/webbench_task_index.jsonl` | 2,647 rows: `task_id`, `task_uid`, `csv_id`, category, URL, registrable domain, per-task digest |
| `split/webbench_eval_split_manifest.json` | The evaluation split: all 2,647 IDs + aggregate hashes |
| `split/webbench_split_derivation.json` | Full derivation record, including the hash definitions |
| `split/webbench_disjointness_proof.json` | Train/eval disjointness against the eval-only training manifest |
| `split/webbench_task_characterization.json` | Factual profile of the task set |

Identity scheme: `task_id = webbench-task-<csv_id zero-padded to 4>` (order-defining,
lexicographic order equals numeric order); `task_uid = webbench-uid-<task_digest[:16]>`
(content-addressed, row-order independent); `task_digest = sha256(canonical_json({category, csv_id, starting_url, task}))`.

| Aggregate | Value |
|---|---|
| `task_id_hash` | `e677af69aa5d1dc7137e54c18c41d99343a41b3e5e77377f7512f42c1a34d2b5` |
| `task_digest_hash` | `061fd64b235505d3218f087c01b9b5946c6a7ef2baa5893c549b0ce8ad68a32a` |
| `task_index_hash` | `84921790fdd4cf20af876324311ea2d6399fc6032cd4ce4ca44fa5b7fe6f85e8` |
| `split_manifest_hash` | `feb368884a41c994567cd7067cf22c47fc67e27158dd635df7ab4a594cf93f0c` |
| `derivation_hash` | `6d93f512ac1ffdcf609603878cf86c9592fc0adb028e616854dd62ec8014150d` |

Cross-check: the derivation also recomputes `webbench_eval.py`'s independent
hashing scheme and reproduces both of its pinned constants exactly —
`22afbdd3cc47e6dba1e3c57ddbe5f762b54be5d2af6ac76bbd206c19eb83b12e` (ID hash) and
`66da44a04ec48fe356b3b0d1c420c40679faa1a7ac650728e254b625bb674a07` (row-content
hash). Two independent implementations agree on the same 2,647 rows.

The derivation is byte-reproducible: re-running `--build-split` into a fresh
directory, with relative instead of absolute input paths, produces all five
artifacts byte-identically (`cmp` clean).

**Disjointness: proven.** Evaluation = 2,647 IDs; training manifest = 0 IDs
(explicit empty list, and its declared hash `e3b0c442…` recomputes); overlap = 0.
Every derived artifact is labelled `receipt_class: local_derivation` with
`authenticated_receipt: null` so it can never be mistaken for a provider receipt.

## Task-set characterization (from the CSV, not the paper)

- **2,647 tasks**, all with distinct task text, each exactly 2 lines (instruction +
  a single-site constraint sentence). Length 160–453 chars, median 265.
- **Categories:** READ 1,637 (61.8%), CREATE 594 (22.4%), UPDATE 206 (7.8%),
  DELETE 166 (6.3%), FILE_MANIPULATION 44 (1.7%). **1,010 tasks (38.2%) are
  write-class** — they mutate live third-party sites.
- **449 hostnames / 448 registrable domains.** 2–40 tasks per domain, median 3;
  no domain has fewer than 2. Heaviest: stackexchange.com 40; grubhub, linkedin,
  stackoverflow, streeteasy 28 each; open.spotify.com 27; github.com 24.
- Every domain has at least one READ task (448/448); write categories are far
  narrower — CREATE 110 domains, DELETE 93, UPDATE 91, FILE_MANIPULATION 23.
- **2,633 https, 14 http** (`http://indeed.com`, `http://www.barnesandnoble.com`).
- **IDs are sparse, not contiguous:** 2,647 rows spanning `0..2724` with **78 IDs
  absent**. Rows are also not stored in ID order in the file. Any ID-ordered hash
  must sort explicitly — this is why the derivation sorts rather than trusting
  file order.
- Heuristic keyword probes over task prose: 536 tasks mention login/sign-in/account,
  20 mention checkout/payment/purchase, 0 mention captcha.
- **Fields the native verifier would need that the public CSV does not contain:**
  `success_criteria`, `expected_final_state`, `answer_key_or_rubric`,
  `allowed_side_effects`, `credential_scope`, `reset_procedure`. The CSV gives
  only `csv_id`, `starting_url`, `category`, `task`. There is no gold answer of
  any kind in the public release — that, not the browser, is why no score exists.

## What the boundary check rejects

Against the full 2,647-task manifest, `validate_webbench_manifest` returns
`BLOCKED` with exactly five blockers — and none of them are about the split:

| Blocker | Meaning |
|---|---|
| `source_receipt_invalid` (`revision_receipt`) | no authenticated receipt binding revision `ea7a1628…` |
| `source_receipt_invalid` (`license_receipt`) | no authenticated receipt binding the MIT license |
| `task_receipt_invalid` | no authenticated receipt binding `task_id_hash` |
| `split_receipt_invalid` | no authenticated receipt binding `split_manifest_hash` |
| `environment_contract_missing` | no native environment / artifact / verifier contract |

No `task_id_hash_mismatch` and no `split_manifest_hash_mismatch` appear: the
derived hashes are internally consistent and would satisfy `tasks_and_split` the
moment authenticated receipts exist. A unit test
(`test_built_split_manifest_satisfies_the_boundary_task_and_split_checks`) pins
that, and a second test
(`test_derived_split_alone_never_unblocks_the_environment_boundary`) pins that a
local split can never on its own make WebBench look runnable.

Runner preflight agrees: **7 of 8 gates PASS** (authoritative dataset, disjoint
task IDs, isolated environment, model binding, budget cap, W&B ordering, agent
command), with one BLOCKED gate — `official_native_environment_and_verifier`.

## What remains

1. **Halluminate live environment** with `environment_revision` (40-hex) and
   `container_image_digest` (`sha256:<64 hex>`), plus screenshot/DOM/reset
   capability and a credential scope.
2. **Halluminate native verifier** — `verifier_id`, 40-hex `verifier_revision`,
   `verifier_sha256`, argv `command`, HTTPS `receipt_url`, and
   `ground_truth_available=true`.
3. **Task authorization** — write-class tasks touch 448 live third-party domains;
   MIT covers the repository, not automated writes to those services.
4. **Held-out split confirmation** — whether the 78 absent IDs are withheld.
5. **Model artifact** — no completed Tinker sampler checkpoint; preflight falls
   back to base-model binding
   `Qwen/Qwen3-VL-30B-A3B-Instruct@9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
   Moot until 1–3 land.

Items 1–4 are provider-side. Nothing local unblocks them.

## Single next action

Send `outputs/e6_webbench/ACCESS_REQUEST_HALLUMINATE_2026-08-09.md` to Halluminate.
It is written to be sent as-is (fill in the current contact address; it does not
guess one). It carries our pinned hashes so they can verify our copy matches
theirs, and it specifies the exact JSON our runner accepts — which drops straight
into `webbench_eval.py --native-receipt`.

## Files

- Receipt: `outputs/e6_webbench/e6_lane_receipt_2026-08-09.json` (BLOCKED, score `null`)
- Access request: `outputs/e6_webbench/ACCESS_REQUEST_HALLUMINATE_2026-08-09.md`
- Boundary manifest: `outputs/e6_webbench/boundary_manifest_full_eval.json`
- Logs: `outputs/e6_webbench/logs/`
- Code: `zvf-program/flagship/pavlov_webbench_eval_adapter.py` (+ its test file)

Note: `adapter_manifest.json` and `adapter_report.json` are the older single-task
probe (`webbench-task-0`) and are left untouched for history. The full-suite
manifest is `boundary_manifest_full_eval.json`.
