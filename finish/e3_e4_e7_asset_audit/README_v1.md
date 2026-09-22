# E3/E4/E7 bounded local asset audit, 2026-09-12

Read-only inspection of existing lane sources and receipts. Only this report and
`.codex-run/finish_20260912/e3_e4_e7_asset_audit/audit_v1.json` are new. No external
availability refresh, provider inbox access, build, benchmark replay, or model call.
No source receipt, manifest, checkout, or another agent's edit was changed.

| Lane | Current local evidence | Missing or unverified | Parent next action |
|---|---|---|---|
| E3 | Existing receipts and synthetic/shape-only fixture directories; receipt specifies 80 private tasks | Provider-issued immutable bundle and permission, live runtime/reset/traffic generator, native grader and split proof; no new official package established by this bounded audit | Reconcile provider response if any; ingest exact licensed assets only when supplied |
| E4 | Checkout `ff6db552a44632643df20393065056e5f1f0092c`; 100 tasks; 600/600 generated task file hashes match current split manifest; tasks.jsonl matches pinned source hash; LICENSE exists | Full input/shared SEC/VDR/logo payload integrity, ready runtime image and native grader service not exhaustively revalidated; no current parent allocation | Reconcile attempts/allocation; revalidate remaining payload/runtime before selecting a supported unstarted attempt |
| E7 | Checkout `cbd86c7cd8519f01ae6b7ad7db7fdb653ea54f23`; 46 task directories and task/instruction/verifier interfaces present | No top-level LICENSE/COPYING artifact; selected dnsmasq Dockerfile references `build/build-assessment.sh`, absent locally; runtime image/build payload not validated | Preserve prior errored attempt; resolve permission and build/runtime assets, then select an explicitly unstarted task only under parent allocation |

E7 split is **lane-constructed, not upstream official**: 28 primary-eval, 10
heldout-labelled and 8 train. Do not call 46 the primary-eval denominator. Its
`dnsmasq-backdoor-detect-negative` attempt on August 22 exited with a budget-related
agent error and native reward 0.0; that is attempt-level evidence, not a lane score.
The audit records current presence and hash comparison of each referenced attempt
artifact; missing local files do not erase the immutable historical receipt.

Historical E4 dollar estimates and old budget caps are not current allocation.
This audit does not make any lane runnable, replace original scope, or authorize
unknown/completed replay. Runtime image availability was not probed by starting
Docker or contacting a paid provider.
