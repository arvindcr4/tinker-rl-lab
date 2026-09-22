# Unblock carry-out — 2026-09-21 (~00:45 UTC)

Follow-up to UNBLOCK_ALL_2026-09-20.md (drafts-only) and the 2026-09-21
unblock prep (drivers, drafts, re-verifications). User instruction for this
pass: "unblock blocked and carry out". Spend authorization on record:
outputs/e1_e14_paid_authorization_unlimited_2026-09-20.json (unbounded;
per-run checks + all gates retained; paid work started by that receipt: false).

## Sends carried out

| Lane | Channel | Disposition | Receipt |
|---|---|---|---|
| E7 BinaryAudit | GitHub issue, QuesmaOrg/BinaryAudit | OPENED #22, verified live (author arvindcr4, 2026-09-21T00:42:10Z) | https://github.com/QuesmaOrg/BinaryAudit/issues/22 |
| E3 SDAB | email founders@emulated.so | accepted by local postfix queue 028231390A111 | remote delivery unverified; bounce-watch Gmail inbox |
| E14 FrontierMath | email math_evals@epoch.ai | accepted by local postfix queue 02CF91390A112 | same caveat |
| E12 AppBench | email support@afterquery.com (route 1) | accepted by local postfix queue F19151390A110 | same caveat; HF discussion-2 follow-up (route 3) held to avoid same-day duplicate |
| E10 AgentHarm | HF dataset discussion #9 | ALREADY SENT by lead 2026-09-20T16:00Z, verified live; no duplicate posted | https://huggingface.co/datasets/ai-safety-institute/AgentHarm/discussions/9 |

Mail caveats: `gog` (Gmail API) auth is stale (`invalid_grant`; needs
browser `gog auth login` by user), so sends went via local postfix
(direct-to-MX; SPF unaligned → may spam-folder; bounces return to the
Gmail inbox). Queue verified ACTIVE via mailq at send time.

## Launches: none executed — each fails a concrete gate (all checked live)

| Lane | Gate status |
|---|---|
| E1 $16 Modal | BLOCKED: no W&B login (no WANDB_API_KEY; `wandb verify` fails), sealed reservation absent (only v12 DRAFT), no live actor endpoint. Modal CLI itself is authed. |
| E2 ~$4 GCP | BLOCKED on HARD USER GATE: no fresh owner IAM binding exists; v19 proposal is stale (Sept-12 names/epochs) and watchdog-only, while the new driver needs create+delete (flagged in v14 draft `lead_review_notes`). User is project owner so can mint it; needs v14 seal + $3+$1 holds after. Driver + v14 draft ready. |
| E5 $80 | BLOCKED: no lead-issued bound authorization receipt (assess_launch_authorization refuses; will not self-mint) AND no real NativeRuntimeAdapter exists in-repo (zero subclasses found) — nothing to bind even with a receipt. `verify`+`plan` green. |
| E13 $8 | BLOCKED: v11 is DRAFT (PROPOSED_NOT_ALLOCATED, launch_authorized false); needs actor05 reservation, connectivity re-proof, supported_route gate change, supervisor deployment — all lead. Modal authed. |
| E4 rerun | Keys all live 2026-09-21; step-0 recheck clean (evidence 14/14, archive absent); intent/cost/cleanup pre-registered. BLOCKED on scope sign-off ($59.45+ new experiment). NOTE: Modal sampler bridge is DOWN (/health timeout 2026-09-21, app absent) — redeploy or Colab-Pro fallback required before trials; colab-pro added to launch-intent compute_resources (UNVERIFIED, needs user GPU confirmation). |
| E6/E9 | BLOCKED: quota re-checked live 00:10 UTC — still NO-GO both regions (cases open). E9 local build deferred (31 GiB free vs ≥30 GiB builder need). |
| E8-original | No send route drafted (private package, no contact on record). Lane stays CLOSED_EXTERNAL. |

## Exact user-only next actions (nothing else preppable)

1. E2: create the four-permission (create+get+delete instances/disks per
   driver gate, or split owner-creates/watchdog-deletes) exact-resource
   time-conditioned binding + receipt file; seal v14; issue $3+$1 holds.
2. E5: issue bound authorization receipt (request-sha-bound JSON); provide or
   point at the real actor/native adapter (retained E5 runtime).
3. E1: `wandb login`; seal v12 hold; bind live actor endpoint; authorize
   driver --execute.
4. E13: seal v11 (actor05, re-proof, gate change, deploy supervisor).
5. E4: export GEMINI_API_KEY + approve ≥$59.45 grader spend.
6. Sends: `gog auth login` (browser) to restore Gmail-API sending + bounce
   visibility; watch Gmail inbox for the three MTA bounces/replies; reply
   to E7 #22 if maintainers respond.
7. E6/E9: escalate AWS cases 178921342800925 / 178982528000009, or free
   ≥30 GiB and reconstruct the E9 runtime recipe for a local build attempt.
