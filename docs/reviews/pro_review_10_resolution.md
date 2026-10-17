# Review 10: backends and infrastructure resolution

Base: `16cde307`; branch: `fix/pro-review-slice10-work`.
Review: `review_10_backends_infra.md`, source conversation
<https://chatgpt.com/c/6ad365e3-cd60-83ee-8bb5-e2270f13acca>.

## Scope and evidence

All changes and checks were local. No provider API, training, paid inference,
cloud resource creation, live credential use, push, merge, or stored research
result modification was performed. Infrastructure commands in the regression
suite target temporary fixture executables only. New manifests describe code
contracts, not newly obtained experimental evidence.

“Fixed” below means the reproduced code defect is corrected and covered offline;
it does **not** mean a GPU or cloud deployment was validated. “Partly fixed”
identifies an implemented containment/fix plus the missing end-to-end contract.

## Finding-by-finding disposition

| # | Status | Change and regression evidence | Remaining limitation / required follow-up |
|---|---|---|---|
| 1 | Fixed | Removed MIG xtrace; separated required secret retrieval/export; GCP metadata contains versioned Secret Manager references only and rejects raw forwarding; startup temp files are removed. Echoverse uses a positive public-config allowlist and an environment-variable-name CLI, redacts captured credential echoes, and never passes the value to its child argv. Vast Python reads inherited environment values; removed `--wandb_key`. Tests cover synthetic secret rejection, dry-run and W&B config, and shell source/argument contracts. | Historical exposure was not established. If these paths ran with real credentials, the owner should rotate the affected Tinker/W&B/HF/OpenAI credentials and audit old logs, metadata, W&B configurations and process records. No credential was rotated or used here. |
| 2 | Fixed | Failed GCP create never authorizes deletion. Cleanup requires successful create, recorded instance ID and matching run label, plus durable receipt collection. Echoverse fails updates while preserving existing checkouts; no delete/reclone or wrong-branch continuation. Tests reproduce create collision, ownership mismatch, and fetch/checkout/pull failures. | Ambiguous create responses require operator reconciliation; provider-side DELETE deadline limits standalone VM retention. |
| 3 | Fixed for active TRL path | Replaced keyword reward with strict, versioned numeric outcome correctness. Requires gold answers and unique prompt IDs; collapse is measured within verified generation groups. Legacy random metrics require explicit simulator mode and are labeled. Seed precedes adapter initialization; generated receipts use observed trainer global steps. Tests reproduce correct `2` versus keyword-filled wrong answer and two homogeneous groups with opposite rewards. | This is an outcome grader, not a PRM. `trl-gsm8k-numeric-v2` is a new protocol; no cross-backend equivalence or historical comparability is asserted. Full GPU/TRL-version validation remains required. |
| 4 | Partly fixed | CLI/launcher scalar settings resolve to a versioned hashed spec; local/Colab TRL use step budgets and consistent rank/batch/accumulation defaults. Requested optimizer/KL/seed/completion settings reach the generated trainer; prompt-length overflow fails explicitly. GCP propagates model/algorithm/task/seed and rejects unsupported overrides; frozen TRL GCP rejects non-frozen overrides. Modal fixed sweeps and the incomplete Vast adapter are blocked before provisioning. Dry-run matrix records argv, spec hash, blocked reasons, and explicitly denies verified scientific equivalence. | The represented spec is not the full preregistered runtime/decoder/dataset contract. Modal/Vast/SkyRL still need reviewed parameterized adapters; non-TRL GCP requires explicit revision/secret references through its own CLI. No fabricated adapter parity. |
| 5 | Fixed | Generated header is one serialized Python string literal. Every LaunchPlan stores argv directly and formats with shlex.join only for display. Reproductions cover executable-header injection and model paths containing spaces/option-looking text. | Downstream provider adapters must maintain these boundaries when they are implemented. |
| 6 | Partly fixed | Vast and Echoverse propagate failures. GCP no longer accepts serial markers; it collects an identity-bound JSON receipt over authenticated SSH, saves it atomically before owned deletion, and exits with the child code. MIG publishes versioned terminal shard receipts with phase/run/task-manifest/revision identity. Tests cover marker false positives, wrong receipts, failure exit and collection-before-delete. | Standalone GCP receipt is explicitly a **process receipt**, with unknown completed steps and empty artifact hashes. Scientific success still requires validated trainer/artifact receipts. SSH/Secret Manager IAM and preemption behavior were not exercised. |
| 7 | Partly fixed | Standalone GCP expiry uses DELETE rather than STOP and defaults to waiting. MIG has unique ownership records, immutable-ID cleanup, partial-deployment cleanup, a foreground deadline, terminal-receipt collection and no sleep-infinity tail. README corrects concurrency and billing/lifecycle claims. A fake-cloud integration test exercises controller cleanup. | Host loss/SIGKILL or ambiguous provider create results still require an independent authorized expiry controller. None was provisioned. Vast's external provisioner teardown contract remains unimplemented. Do not deploy MIG unattended. |
| 8 | Partly fixed | SkyRL server-only execution is rejected before launch. TRL accepts only actual completed-step receipts; verl requires an actual final_step result instead of substituting requested epochs. Tests reject the idle SkyRL server path and require mocked actual TRL receipts. | SkyRL training-client/readiness adapter is absent. Other framework-specific receipt semantics and simulator isolation require their own audited runtime adapters. |
| 9 | Fixed for reproduced boundaries | TRL uses unique protocol/spec/run directories, explicit resume only, and config/data-hash manifest matching. No get_last_checkpoint auto-resume. Colab smoke/full use distinct versioned directories; cached results must match requested configuration and include completion companions. Tests reject smoke-as-full and missing companions. | Model/dataset/runtime immutable revisions belong to the outstanding full runtime contract in #21. Legacy result paths are deliberately not silently reused. |
| 10 | Partly fixed | Merge gate rejects filename/payload mismatches, group mismatches, empty logs/environment/text, nonbinary grades and inconsistent summaries. Requires pinned item identities and independent completion regrading; publishes a whole verified bundle directory atomically and requires an explicit stratified hardware policy. Regressions cover the supplied forged result and a minimally valid regraded fixture. | No trusted historical item manifest was supplied and no historical outputs were regraded or merged. Full per-arm critic/optimizer/runtime schema and a reviewed pooling calibration remain necessary for scientific acceptance. |
| 11 | Partly fixed | Only a hash-verified completion marker permits skipping. Downloads stage both files before publishing; incomplete transfers preserve remote sessions and are not retrained blindly. Setup failures clean up owned sessions; existing sessions are never stopped. Tests cover JSON-only staging, failed text transfer, exact existing session matching, and preserving a newly created session on transfer failure. | Colab CLI session-list format and immutable session identity need live validation. A retained recovery session can incur charges; operator transfer/release and a separately authorized lifetime controller are required. |
| 12 | Fixed | Entire trusted shell job spec is validated before launch: positive concurrency, unique safe names, noncolliding declared destinations, positive timeouts. Handles are never overwritten; process groups are terminated on deadline/interruption, logs are closed, and required job failures exit nonzero. Tests cover duplicate names, zero concurrency, traversal, and local harmless failed/successful jobs. | Commands are intentionally trusted shell input; arbitrary output paths hidden inside shell strings cannot be inferred by the validator. Bare done-file skipping was removed. |
| 13 | Partly fixed | MIG writes per-instance stateful shard metadata as well as initial live metadata, restores same-run JSONL with --resume, syncs progress every minute and publishes terminal receipts. Fake-cloud test checks preserved metadata and generated startup syntax. | Actual gcloud/MIG replacement behavior needs staging validation. Up to one sync interval may be repeated/lost; there is no durable task lease/exactly-once guarantee or full task-coverage/hash aggregation gate. |
| 14 | Fixed | Corrected builder and three local helper repository roots. Builder validates local inputs before provisioning; worker executes from repository root. Tests cover root resolution, preflight ordering and generated module working directory. | Clean-image execution itself was not performed. |
| 15 | Fixed for shell parsing | Quoted peft/accelerate constraints; synthetic installer receives the constraints without creating redirection files. All modified shell files pass bash -n. | Complete tested runtime locks and import/version preflight remain part of #21; broad ranges were not falsely labeled reproducible. |
| 16 | Partly fixed | Echoverse requires a harness explicitly advertising --task-ids-file and --agent-api-key-env, passes an ordered manifest, hashes tasks/config/endpoints into run identity, and fails rather than using count-only execution. Test demonstrates rejection of the omitted old harness interface. | External harness source was not supplied. Its exact returned-ID contract, code revision, and end-to-end selected-task verification cannot be certified; this path is intentionally blocked until compatible support exists. |
| 17 | Partly fixed | HF verifier requires a separately trusted manifest digest, compares the exact artifact hash set, checks date/run identity/PNG signature, and exits nonzero on failure. Tests cover missing anchors and artifact tampering. Default UI can no longer claim PASS merely by hashing itself. | No trusted release manifest/signature or reconciled audit inventory was supplied. The verifier explicitly limits its claim to release integrity; derived scientific values and audit counts are not newly certified. |
| 18 | Partly fixed | Live fetch requires a reviewed per-cell index; HF/GCS fetched identities must match framework/task/model revision/seed/protocol. W&B is filtered and bounded with islice; history is labeled sampled_history. UI reports complete/partial/empty/unavailable without raw exceptions. Tests ensure an OpenRLHF request cannot fetch a TRL entry and mismatches are unavailable. | No reviewed source index was supplied; default is unavailable. W&B exact run IDs must come from that index; full source artifact-hash anchoring remains deployment work. |
| 19 | Fixed / fail closed | Export requires an explicit full reviewed commit and reads Git blobs, rejects symlinks, excludes credential filenames/binaries/unknown extensions, removes working-tree overlays, fails on residual scan problems, and requires a redacted gitleaks pass before packaging. Tests cover .env/unknown/keys, symlinks, missing commit, and missing/failing scanner. | No export was produced. Any binary inclusion needs separate verified handling. Scanner installation and actual full-repository scan were not performed, and secret values were never printed. |
| 20 | Fixed for reproduced helper costs | Matched evaluator saves each completed result atomically, releases model/wrapper references before reallocating and skips final reload. Mock tensor/model regression verifies zero overlap and only two allocations for two adapters. Paid ungrounded 45-call sweep is disabled; analyze_part requires explicit file evidence and caps output. | A reviewed, bounded paid-analysis orchestration is not implemented; the helper fails before calls when evidence is absent. No inference calls occurred. |
| 21 | Partly fixed | Canonical loading now validates the actual nested preregistration schema and fails on missing/malformed inputs. GCP dispatch and image recipe require full repository revisions; workers refuse a mismatched image checkout instead of pulling opportunistically. Image loading requires independently anchored archive hashes, removes HTTP fallback, and requires a digest-pinned Kiwix image. Hash-mismatch regression fails before docker load. | No reviewed archive manifest/digest, image revision set, per-backend runtime lock or complete model/tokenizer/dataset manifest was supplied. Base image families and remaining framework setup still need a full reviewed runtime contract; no expensive allocation is authorized by these tests. |
| 22 | Partly fixed / unsupported cells blocked | Added pre-allocation, digest-anchored task/site contract; only explicitly enumerated supported-site tasks are allowed. Unknown/forum/Wikipedia/map/judge-dependent tasks fail before gcloud. Worker passes those exact task IDs, shares reset_urls.sh, propagates required reset failures and checks readiness/local-origin redirects. Infrastructure setup failure emits a phase-labeled receipt. Tests reject unavailable services and verify ordered manifest tasks and early validation. | Full WebArena is deliberately unavailable: forum reset implementation, Wikipedia ZIM, valid map service, judge credentials and a reviewed task-to-service manifest were not supplied. No guessed benchmark subset was generated. Evaluation-side infrastructure error classification/aggregation still needs end-to-end validation. |
| 23 | Partly fixed | Analysis hardware labels come only from substantive per-run metadata, never directory names. Merge requires explicit hardware-stratified policy. CLI labels token budget as an estimate and uses configured accumulation. Tests verify L4 metadata and reject missing provenance. | No scientific calibration/pooling policy was invented. Observed sampled/optimization token counters and stopping remain unimplemented; an estimated token option is not evidence of equal realized work. Existing stored analyses were not recalculated. |
| 24 | Fixed for default/bounded-public policy | Public default is static evidence, even if a server token exists. Explicit bounded_public policy requires a persistent quota DB and a shared lifetime cap <=100 calls, atomic reservation, input <=4096 characters and 30-second inference timeout. Provider output tokens remain capped; errors are concise public codes. Tests cover default denial, input limit and persistent shared exhaustion. | Deployment must supply persistent storage and one shared quota authority. Multiple independent replicas must use a central quota service; separate SQLite files do not provide a global cap. No live policy was enabled. |

## Protocol and compatibility changes

- `trl-gsm8k-numeric-v2`: strict numeric outcome grader, explicit answer/prompt_id
  columns, group-conditional diagnostics, seeded adapter initialization, isolated
  run directories, explicit matching resume, and observed-step receipt
- `unified-resolved-v2`: resolved scalar dispatch spec and hash; not a claim that
  all preregistration fields or backend semantics are equivalent
- `unified-dispatch-v2`: secret-reference GCP bootstrap and identity-bound process
  receipt; the frozen next-submission GCP launcher remains untouched
- `samestack-colab-v2-validated-reuse`: distinct smoke/full namespaces; old results
  stay untouched and are not implicitly upgraded
- `colab-staging-v2` and `colab-verification-v2`: complete-bundle marker and stricter
  verification gates; missing evidence is rejected rather than guessed
- `webarena-restricted-services-v2` / `webarena-shard-v2`: explicit task/site
  manifest, recorded code revision, infrastructure/evaluation phase receipts,
  stateful shard assignments and bounded foreground controller
- Unified CLI's legacy `--epochs` now denotes its explicit optimizer-step budget;
  generated TRL uses max_steps and one fallback epoch. Defaults align to 30 steps,
  rank 16, batch 2 and accumulation 8. The token-budget flag remains an estimate
- iDPO is rejected for the scientific generated runner until a dedicated reward
  and trainer-configuration adapter exists; old GRPO callback assumptions must
  not be reused silently

## Offline validation

Interpreter: `/tmp/tinker-review-venv/bin/python`.

```sh
/tmp/tinker-review-venv/bin/python -m pytest -q \
  tests/reviews/test_pro_review_10_backends_infra.py \
  tests/test_trl_import.py tests/test_unified_matrix.py \
  tests/test_launcher_forwarding.py tests/test_peft_support.py \
  tests/test_pydantic_support.py tests/test_next_submission_trl_adapter.py
```

Final gate: **146 passed** (59 focused regressions plus 87 compatibility/
configuration/import tests). Existing tests were updated only for intentional
versioned behavior: actual receipt requirement, no automatic checkpoint resume,
step-budget defaults, and unsupported iDPO rejection.

Additional checks: Python compilation for every changed/new Python source;
`ruff check --select F` for changed/new Python sources; `bash -n` for every changed
shell script and the fixture-rendered MIG startup script; `git diff --check`.
Tests use temporary synthetic manifests/results and fixture CLIs only. No
pre-existing result or research artifact was rewritten.

## Deployment hold points

Do not treat the dry-run matrix as permission or proof to run a new scientific
campaign. Resolve the partial items above first, especially independent cleanup,
full runtime/adapter equivalence, trusted evidence manifests, missing external
harness contracts and unsupported WebArena services. Do not upgrade historical
protocol labels to the new versions without a new validated run.
