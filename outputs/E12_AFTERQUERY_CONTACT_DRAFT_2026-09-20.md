# E12 AppBench exact-lane recovery — DRAFT first-contact message (NOT SENT)

Requires user approval before sending. Proposed routes (in order):
1. `https://appbench.ai/` contact channel (live benchmark site, lineage from AfterQuery UI-Bench)
2. `https://afterquery.com` corporate contact
3. Hugging Face discussion on `AfterQuery/App-Bench` (dataset live; prior discussion 2 got no reply)

---

Subject: Request for AppBench evaluation access for independent reproducibility run

Hello AfterQuery team,

I am running an independent reproducibility campaign over agentic benchmarks
and would like to evaluate on AppBench (app-building with deployment).
Your `appbench.ai-docs` repository appears to be gone, so I am writing via
this channel instead.

What I need for an exact, admissible run:
- the 6-task held-out `appbench_eval` slice (or your current equivalent),
  with pinned revision hashes;
- the immutable deployment/reset contract and pinned runtime;
- the applicable license for evaluation use;
- a provider-signed result grant (our harness verifies Ed25519 trust roots);
- the two-human-grader plus re-adjudication protocol specification.

We can run inside maintainer-controlled infrastructure, avoid redistribution,
and report only receipt-backed aggregate results. Our lane score stays null
until a provider-issued immutable receipt exists.

Prior request: Hugging Face discussion on `AfterQuery/App-Bench` discussion 2
(August 2026, no reply received).

Thank you,
arvindcr4 (tinker-rl-lab reproducibility campaign)
