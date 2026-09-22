# Unblock-all plan — 2026-09-20 (DRAFTS ONLY, nothing sent/spent/signed)

Status: 5/14 lanes hold a complete result (E1, E11 exact; E8, E10, E14 replacement/proxy-complete).
The 9 blocked exact lanes and their single next action:

## Needs YOUR send (outbound, drafted below — not sent)
- **E3 SDAB**: 3rd follow-up to founders@emulated.so (sent 08-22 + 08-29, zero replies). §E3 draft.
- **E10 AgentHarm**: 2nd follow-up on HF discussion #9 (zero replies; rechecked today). §E10 draft.
- **E12 AppBench**: FIRST contact via appbench.ai channel (prior email + HF thread dead; draft exists, never sent). §E12 draft.
- **E14 FrontierMath**: follow-up to math_evals@epoch.ai. §E14 draft.
- **E7 BinaryAudit**: NEW GitHub issue on QuesmaOrg/BinaryAudit (license 404 + split manifest + verifier). §E7 draft.

## Needs YOUR authorization (spend/keys/quota — no external party)
- **E2 CORE-Bench**: GCP IAM binding (owner-only) + $3+$1 reservation. Amendment accepted; driver rebuild + v14 draft are free and queued.
- **E4 BankerToolBench**: GEMINI_API_KEY (native verifier, hard) + agent key + spend (full-run est. $700–4500; grader floor ~$59 > $44 cap — needs subset or new cap).
- **E5 APEX**: $80 successor27 subset authorization (full 480 = ~$122 Tinker — does not fit; propose ~50-task stratified subset).
- **E13 BALROG**: sealed v11 grant + $8 H200 reservation (actor04 expired; supervisor written, undeployed).
- **E9 MLE-bench**: AWS us-east-1 quota 1→4 vCPU (or local Docker image build, $0 but slow); Kaggle 75/75 acceptance may need token refresh (browser, user-only).

## Free work queued (no approval needed, doing next)
E2 driver rebuild + v14 draft; E4 attempt-history verification; E9 local image-build attempt.

## §E3 draft (email, founders@emulated.so)
"Following up on our SDAB evaluation-access request of Aug 22 (plus Aug 29 note) — a single reproducible third-party run of an open-weights model against the frozen 80-task bundle with your native grader. We remain happy to run inside your infrastructure, sign an evaluation agreement, and hold publication for your review. If SDAB access is closed to outside evaluators, a one-line confirmation will let us record the lane accordingly. Thank you, Arvind (arvindcr4@gmail.com)."

## §E10 draft (HF discussion #9 follow-up)
"Hello AgentHarm team — following up on the held-out (test_private) access request above for an independent refusal-robustness evaluation (open-weights model Qwen/Qwen3.6-35B-A3B, defensive safety purpose, no-training/no-redistribution commitments as stated). If releasing the three held-out files isn't appropriate, we'd equally welcome a provider-run or supervised evaluation with a signed score. A short yes/no on whether external hosted evaluation is available would let us record the lane accurately. Thank you."

## §E12 draft (appbench.ai contact — full text in outputs/E12_AFTERQUERY_CONTACT_DRAFT_2026-09-20.md)
"I'm running an independent reproducibility campaign and would like to evaluate on AppBench. I need the held-out task slice with pinned revisions, the deployment/reset contract, an evaluation-use licence, and your grader protocol (or a provider-run evaluation of our submissions). We can run inside your infrastructure, avoid redistribution, and report only receipt-backed aggregates. A short reply on availability and terms would be enough to proceed."

## §E14 draft (email, math_evals@epoch.ai)
"Following up on the hosted FrontierMath Tiers 1–4 evaluation request for open-weights model Qwen/Qwen3.6-35B-A3B. To be explicit: I'm not requesting the problem set, only an Epoch-run evaluation under your holdout discipline, including fully blind. Could you confirm whether externally-initiated evaluations are accepted, and if so your preferred access mode, cost, and reporting terms — or a direct no if the eligible set is limited to models you select? If Tiers 1–4 isn't available, I'd appreciate a pointer to Open Problems verifier terms instead."

## §E7 draft (GitHub issue, QuesmaOrg/BinaryAudit)
"We are evaluating BinaryAudit as a primary suite with a fail-closed harness (pinned rev cbd86c7c, 46 task dirs inventoried). Three items block us that only you can supply: (1) the README claims Apache-2.0 but the revision has no LICENSE file and the GitHub license endpoint returns 404 — could you add the license text at a pinned commit or confirm the authoritative SPDX identifier and digest? (2) Is there an authoritative train/eval split manifest, or should all 46 tasks be treated as one evaluation set? (3) Can you confirm the native verifier entry point (we see per-task tests/test.sh) and the reference base image / Harbor setup for isolated execution? We will report BLOCKED rather than substitute another benchmark. Thank you."

## Honest boundary
Five of nine lanes need a third party to say yes; no action today guarantees unblock. What today CAN do: send all five messages (your call), clear the four spend/key/quota gates (your call), finish the free prep. If every provider stays silent, the lanes stay honestly BLOCKED with null scores — that is a valid submission state, not a failure.
