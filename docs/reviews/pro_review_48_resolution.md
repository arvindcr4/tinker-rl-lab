# Review 48: security and cost safety

Base: `6407255d`; branch `fix/pro-review-slice48-work`.
Source: `review_48_lens_security.md` (security audit of backend/infra and Modal).

## Scope

Local code changes and offline fixtures only. No training, paid API calls,
provider resource operations, credential reads, rotation, history rewriting,
publishing, or stored-result modifications. Cloud command tests use temporary
fixture executables. No live-looking credentials were discovered in this review;
this is not a new repository/history secret-scan certification. Review 15's
sanitizers and tool hardening are separate and were not duplicated.

Most original findings were already addressed by integrated reviews 10 and 02
and the native OpenRLHF contract. This patch closes remaining reproduced
injection, subprocess-deadline and resource-admission gaps. Existing intentionally
blocked execution paths remain blocked.

## New fixes

- GCP standalone create, ownership inspection and deletion now have explicit
  120-second subprocess deadlines. Runtime input is capped at 24 hours before
  provisioning. An ambiguous create timeout does not authorize deletion and its
  local startup file is removed.
- MIG admission caps workers at 10, disk at 1000 GB per worker, runtime at 24
  hours, and machine type at the reviewed `e2-standard-4`. Cloud subprocesses
  have 120-second deadlines, including polling and cleanup.
- Builder cache URIs are validated as data before their generated-shell use.
  Shell substitutions/quotes are rejected before cloud access. Builder uses a
  unique name, records immutable ID for cleanup, a private temporary setup file,
  and a provider four-hour DELETE deadline; local commands have finite limits.
  Disk/machine limits are enforced before allocation. Replaced `curl | sh` with
  distribution Docker packages. This is not a claim of a fully pinned runtime.
- Colab controller validates concurrency (1–16), serializes launch parameters,
  uses filesystem APIs rather than a shell for status paths, bounds tail reads,
  and propagates CLI failures. Upload and remote subprocess calls are bounded.
  The queue independently enforces the same concurrency ceiling. The explicit
  operator `sh` interface remains intentionally executable trusted input.

## Finding-by-finding disposition

| Finding | Disposition and evidence |
|---|---|
| A1 GCP startup credential tracing | Existing review10 fix retained: no xtrace; required fetch checked separately before export. Covered by review10 synthetic checks. |
| A2 Echoverse secret logging/argv | Existing public-config allowlist, environment-name auth interface, and redaction retained. Old harness lacking environment auth remains blocked. |
| A3 OpenRLHF W&B command credential | Existing environment-based authentication command and regression coverage retained; no credential value passed in argv. |
| A4 HF anonymous paid calls | Existing default-deny policy, persistent shared call quota, input cap and request deadline retained. A call quota is not a currency budget; multi-replica deployment needs a shared authority. |
| A5 deletion after failed create | Existing run ID/immutable resource identity and receipt-before-delete retained; new timeout reproduction verifies no deletion after ambiguous create. |
| A6 unbounded MIG lifecycle | Existing foreground owned cleanup and terminal receipts retained; resource admission and individual CLI deadlines added. Independent controller-loss reaper remains a deployment blocker. |
| A7 unchecked image downloads | Existing independently anchored archive hashes, HTTPS-only transport and digest-pinned Kiwix retained. Offline mismatch fixture rejects before Docker load. |
| A8 metadata credentials | Existing versioned Secret Manager references, raw-forwarding rejection and temp cleanup retained. IAM/workload isolation still requires operator review. |
| A9 duplicate queue names | Existing full preflight, unique names, process-group cleanup and terminal status retained; added maximum concurrency 16. |
| A10 Colab cancellation/timeouts | Existing same-stack core replaces unsafe Modal stub; bounded queue/session paths retained. Controller upload/exec failure/deadline gaps closed. Recovery sessions intentionally retained after incomplete transfer still need operator release/independent TTL. |
| A11 Echoverse checkout deletion | Existing fail-without-deleting behavior and branch validation retained. |
| A12 stopped VM disk cost | Existing provider DELETE runtime action retained; standalone runtime ceiling and cloud subprocess deadlines added. Builder now also has provider DELETE deadline. |
| A13 sweep shell injection | Existing argv generation plus shlex quoting for command/progress retained. Pilot/attempt manifests do not certify a total dollar budget. |
| A14 TRL header injection | Existing serialized header and argv contract retained. |
| A15 Vast credential-as-source | Existing inherited-environment authentication and removed credential CLI retained. No real credential read or exercised. |
| A16 false successful exit | Existing Vast/GCP status propagation retained; Colab controller now also propagates checked CLI failures. |
| A17 stale OpenRLHF publication | Existing unique private attempt directories and failure gate retained; native route does not publish checkpoints automatically. |
| A18 ignored OpenRLHF max_steps | Existing explicit effective optimizer-update budget/native contract retained; targeted budget/command tests pass. |
| A19 shell version redirections | Existing quoted requirement constraints retained; full tested hash locks remain outstanding. |
| A20 builder allocation before path failure | Existing local path preflight retained; now private setup staging, runtime/deletion guard and resource caps. |
| A21 unbounded W&B enumeration | Existing filtered exact-source index and bounded iteration retained; absent reviewed source index remains unavailable. |
| B1 total budget/cancellation | Partly addressed by existing attempt/receipt/timeouts and new ceilings. Monetary reservations, external Vast lifecycle contract, and independent cloud reapers remain unimplemented. Do not equate resource ceilings with a spending authorization. |
| B2 mutable supply chain | Removed builder curl-pipe-shell; existing repository revisions/archive anchors retained. Complete immutable OS/Python/model/dataset releases still require reviewed artifacts; no hashes invented. |
| B3 IAM | Not verifiable offline; no claim that cloud-platform scope proves broad IAM. Review real bindings and separate builder/worker/controller identities before deployment. |
| B4 generated-code boundaries | Colab launch concurrency/status paths and builder cache URI reproduced and fixed with offline tests. |
| B5 publication/ownership | Existing unique OpenRLHF directories, reviewed-source demo policy and commit/blob-based secret-scanned export retained. No export or private publication performed. |
| C negative findings | No unsafe deserializer introduced; AST/safe YAML, loopback WebArena binding, argv backend execution and intentional runtime blocks preserved. |

## Validation

Interpreter: `/tmp/tinker-review-venv/bin/python`.

- Review48: 22 focused regression cases (including an end-to-end fake builder
  render/cleanup run; generated setup syntax checked, never executed)
- Review10: 75 offline backend/infra regressions
- OpenRLHF budget, command, dryrun, input-artifact and metrics regressions
- Review02 Modal analysis regressions
- Combined run: 279 passed (one existing constant-input Spearman warning)
- `ruff check --select F` on changed Python; `bash -n` on modified/generated
  shell scripts; `git diff --check`

Historical exposure was not established. If affected launch paths ran before
these protections, the owner should rotate the relevant providers and restrict
old logs/metadata: Tinker/W&B/HF/OpenAI (`deploy_mig.sh` and GCP dispatch), agent
API provider (Echoverse), W&B (OpenRLHF and Vast). No secret values are needed in
that report and none are included here.
