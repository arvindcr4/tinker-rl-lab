# E1–E14 final consolidated results — 2026-09-19

This is the completion-facing results document for the Tinker RL Lab
capstone campaign, superseding the working tables of 2026-09-05 and
2026-09-12. Every number below is bound to a surviving receipt and was
re-verified by deterministic code checks on 2026-09-19
(`outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`, 11/11 PASS).
Original-contract and replacement-scope numbers are never pooled
(project evidence rule).

## 1. What the campaign established

A single Tinker-trained actor — base `Qwen/Qwen3.6-35B-A3B`
(commit `995ad96e…`) with adapter
`arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1…`
(commit `64444133…`), bfloat16 — was evaluated against fourteen
held-out agent/RL benchmark suites under strict receipt, budget and
decontamination governance. Three replacement scopes are strictly
complete (E8 public, E10 benign, E11 native), one is 99.95% complete
(E14 Omni-MATH), and the original-contract E1 and E11 suites carry full
verified scores. The remainder are honestly recorded partial or
externally blocked lanes with immutable evidence trails.

## 2. Strictly completed replacement scopes

| Lane | Suite | Verified coverage | Result |
|---|---|---|---|
| E8 | LAB-Bench (public split) | 1967/1967, 8 categories, all COMPLETE | per-category accuracy: TableQA 0.787, FigQA 0.514, ProtocolQA 0.250, SuppQA 0.183, LitQA2 0.176, DbQA 0.131, SeqQA 0.033, CloningScenarios 0.000 |
| E10 | AgentDojo (benign utility scope) | 97/97 | complete benign-utility evaluation |
| E11 | VerilogEval (two native framings) | 312/312 | **129/312 = 41.35% pass@1 canonical** (67/156 code-completion, 62/156 spec-to-RTL; 129/311 sensitivity retained non-canonically) |

## 3. Original-contract suite results (where they exist)

| Lane | Suite | Verified result | Coverage note |
|---|---|---|---|
| E1 | SWE-bench Pro | **2/731 = 0.274% pass@1** | 731 terminal generations; 713 native evaluations; 14 generation failures + 4 artifact losses remain in denominator |
| E2 | FrontierSWE | 1/17 tasks; replay normalized 0.8628 | partial; authorization absent for remaining 16 |
| E4 | BankerToolBench | 1/100 tasks; recovery metric 0.3115 | partial; public artifacts restored, 1175 manifest checks pass |
| E5 | APEX-Agents | 7/480 native-scored; prefix mean 0.050505 | partial |
| E7 | BinaryAudit | 1/46 attempted; verifier reward 0.0 after agent error | partial, no grade |
| E9 | MLE-bench | 40/75 competitions natively graded (53.33% coverage); suite score null | modal_streaming arm only; merged-vLLM arm separate (1 valid grade, H&M 0.02132) |
| E11 | VerilogEval | 129/312 = 41.35% pass@1 | full-suite result |

## 4. Replacement-scope progress (incomplete lanes)

| Lane | Replacement suite | State |
|---|---|---|
| E1 | SWE-bench Multilingual | 35/300 graded; wave10 recovery (16 tasks, $16) sealed and locally validated but source lost; see §6 |
| E2 | CORE-Bench | 45/45 capsule setups complete; 0 graded; direct-VM HARD adapter built |
| E5 | Tau3 | 20/97 (20.62%) cleaned; successor27 ($80) reviewed, unreserved |
| E6 | WebArena | 0/812; canonical inventory verified (65 files, ordered IDs 0–811); AWS quota 1/16 |
| E9 | MLDevBench | 0/34 graded; runtime image incomplete; AWS quota 1<4 vCPU |
| E13 | BALROG | 13/255 episodes; 21 never-started receipt candidates mapped |
| E14 | Omni-MATH | **4426/4428 accepted (99.95%); official accuracy 51.31% reproduced**; 2 parser failures recorded |

## 5. Externally blocked lanes (no local or paid path)

E3 (SDAB private bundle), E7 (BinaryAudit private payload), E8-original
(LifeSciBench package), E10-original (AgentHarm private tasks), E12
(AppBench deployment), E13-original (OpenReward held-out games), E14-original
(FrontierMath hosted evaluation). Access requests sent; none answered.

## 6. Verification and integrity

- Deterministic checks (2026-09-19): 11/11 PASS — exact divisions, receipt
  presence, sealed-artifact hash equality, lost-source absence probes.
- The 2026-09-12 halt was clean: no orphan processes, deployments or
  sessions; disk recovered.
- Execution-source loss (`.codex-run` deletion) is recorded in
  `finish/STATE_RECONCILIATION_2026-09-19.md`; it is confined to the
  finish-era recovery/build code (E1 wave10 controller, E9 restored
  dependencies, E6 validator) — the replacement-lane native runners
  survive under `zvf-program/flagship/`. Sealed requests, validation
  receipts and test logs pin any faithful re-implementation.
- Budget: $5.6301 of the $50 additional cap counted; E5 $80 separately
  unreserved. No spend occurred on 2026-09-19.

## 7. What completion requires (decision package)

See `outputs/PES_Phase2_Review_2026-09-12/finish/DECISION_PACKAGE_2026-09-19.md`
for the six user-level decisions (E2/E13 lifecycle amendment, E5 $80,
E1 re-implementation + $16, E4 cap/scope, AWS quota requests, external
follow-ups). If all recommendations are accepted, ~$108 total new spend
completes every lane that is not externally blocked.

## 8. Jev (TypeSafe) judgment layer

`zvf-program/jev_lab/` adds a receipt-recording judgment harness for the
campaign (triage classification, claim-faithfulness, value ordering).

**Completed before degradation — research-value ordering (14 lanes, one
ask battery, receipt `outputs/jev_receipts/TRIAGE_SUMMARY_2026-09-19.json`):**
normalized 0–1 over four ordered levels (marginal / supporting / notable /
headline). Result: E6 0.39 and E14 0.34 highest (a full WebArena suite
result and the private FrontierMath number are headline-scale), then
E1 0.33 and E9 0.33, E8 0.31, E2 0.28, E10 0.27, E13 0.26, E11 0.20
(complete lane, low marginal value), E5 0.19, E4 0.15, E12/E7 0.12,
E3 0.08 lowest (no result path without the private bundle). This ordering
informs the decision package's recommendations.

**Invalidated by the incident:** both blocker-classification batteries
returned degenerate distributions (near-uniform or a constant class across
all lanes) after ~13:10Z when both `jev-latest` and `jev-preview` stopped
reading state (controlled yes-probe fell from ~1.0 to ~0.3). Their receipts
are retained as incident evidence; the batteries must be re-run once
`healthcheck()` passes. The per-lane "first uncleared blocker" column in
`finish/Pending_Experiments.md` therefore stands on document evidence,
not on jev judgment.

Design rule banked from the healthy window: arithmetic and string lookups
stay in code (Jev mis-answered a parity probe even while healthy); Jev is
reserved for classification, faithfulness and value ordering, exactly as
its docs intend.

## 9. Terminal-state actions (2026-09-19, finish-all directive)

The user (root) issued "finish all expts" on 2026-09-19; the verbatim
directive and its interpretation are recorded in
`outputs/PES_Phase2_Review_2026-09-12/finish/ROOT_DIRECTIVE_FINISH_ALL_2026-09-19.json`
(13:50Z). It is treated as acceptance of the DECISION_PACKAGE_2026-09-19
recommendations D1–D6. The per-lane ledger of record for the terminal
states is `finish/Pending_Experiments.md` (terminal-state table). The
actions below were taken under the directive.

### 9.1 E14 terminal note (replacement scope closed terminal-complete)

`finish/e14_terminal_note_2026-09-19.json` (13:40Z) resolves the 2
parser-exclusion rows (omni-01195-a6cbdc4ee3c8694b row 1195,
omni-02044-612ddacaa39d91fe row 2044): both Omni-Judge outputs are
truncated before any "## Equivalence Judgement" section, so the judge
emitted no verdict; the native scorer's omission of
missing-Equivalence-Judgement rows is correct official behavior, not a
parser bug. Score impact: none — official accuracy
2271/4428 = 0.5131043831902395 already counts both rows in the
denominator as not-correct; the 4426/4428 figure is parser coverage, not
the score denominator. E14 closes as terminal-complete under the native
protocol: 4428/4428 dispositions recorded (4426 accepted + 2 recorded
skip_reason). No re-judge: it would deviate from the official scorer and
generate new evidence with no score effect.

### 9.2 Unlimited-budget discovery

`finish/authorization_unlimited_v1.json` (2026-09-12) records
`budget_mode: unlimited` (`total_cap_usd: null`), superseding the prior
`authorization_live_v1.json` $75 cap; bounded per-run resource and
timeout controls remain in force. Read together with the finish-all
directive's recommended envelope ($108 components: E1 $16, E5 $80,
E2 $4, E13 $8), this supersedes the cap framing in §6 — spend is now
gated per-launch by technical-chain existence and preflight, not by a
cumulative cap. E4's ~$59 grader path remains outside the envelope
components and stays closed unless separately authorized. Paid launches
remain lead-only authority under the squad protocol.

### 9.3 Quota filings

- AWS us-east-1 Standard On-Demand vCPUs → 8: increase request filed,
  status **PENDING**, id `2eebd37d2a4447ceb81d0a45d1dffbe5fbVtQSM7`
  (E9 MLDevBench runtime builder).
- us-east-2 remains the pre-existing open request (E6 WebArena); no new
  E6 request was possible under the one-open-request rule.
*Integrated 2026-09-20 (squad e6e9-status):* both lanes live-checked
read-only 2026-09-19T14:00Z and recorded NO-GO — E6
(`e6_continuation/status_2026-09-19.json`): us-east-2 quota 1/16 vCPU,
request c67d89b0 CASE_OPENED since 09-12, one-open-request rule holds;
E9 (`e9_completion/status_2026-09-19.json`): us-east-1 quota 1/8 vCPU,
request 2eebd37d filed 09-19 now CASE_OPENED 178982528000009. Read-only
watcher `zvf-program/e6e9/check_quota_status.py`. JEV faithfulness checks
routed FALLBACK_OUTAGE (TypeSafe outage persisting;
`outputs/jev_receipts/E6E9_STATUS_JEV_2026-09-19.json`).

### 9.4 E2/E13 amendment acceptances

The root directive is recorded as acceptance of the E2/E13 lifecycle
amendment (DECISION_PACKAGE D1). Status: amendment accepted, launches
pending — E2 CORE-Bench (~$4 target-helper chain, 45/45 setups already
complete) and E13 BALROG ($8 actor chain, 21 never-started receipt
candidates mapped). Each launch still requires its technical chain to
exist (fresh sealed request where the prior expired, reviewable code in
a tracked path) and per-launch preflight.
*Integrated 2026-09-20:* E2 offline launch gate PASSES
(`zvf-program/e2_core/launch_gate.py` exit 0 — amendment accepted, harness
present); ~$4 launch pending, lead-only (fresh IAM + reservation remain
launch-time prerequisites). E13 hosted-supervisor chain verified 17/17
offline (`zvf-program/e13_balrog/test_hosted_supervisor.py`); $8 actor
launch pending, lead-only.

*Unblocked 2026-09-21 (no spend, no sends):* E2 orchestration driver
rebuilt (`zvf-program/e2_core/driver.py`, 12/12 offline tests: schedule
math, exact early DELETEs, cutoff, UNVERIFIED/LATE reporting, no-retry,
IAM gate) + v14 draft (`e2_completion/resource_request_v14_DRAFT.json`,
DRAFT_NOT_SEALED) — remaining blocker is the HARD USER GATE only (owner
GCP IAM binding). E13 v11 draft
(`e13_continuation/resource_request_v11_DRAFT.json`, PROPOSED_NOT_ALLOCATED,
launch_authorized false) — remaining: actor05 reservation, connectivity
re-proof, supported_route gate change, supervisor deployment (all lead).
E1 hold re-drafted (`zvf-program/e1_wave10/reservation_wave10_v12_DRAFT.json`;
prior expired 2026-09-20T14:06:55Z); driver validation-only mode green.
E5 `verify` + `plan` green live (27 tasks, PLAN_READY_NOT_EXECUTED);
`gate`/`launch` correctly refuse without the lead-issued bound
authorization receipt. E4 attempt-history re-verified 99/1
(`e4_completion/attempt_history_reverify_2026-09-21.json`), verdict stands.
E6/E9 quota re-checked live 2026-09-21: still NO-GO both regions
(`e6e9_quota_watch_2026-09-21.json`); E9 local Docker build deferred
(31 GiB free vs ≥30 GiB builder need + recipe reconstruction).

### 9.5 E1/E5 rebuilds in flight

- **E1 wave10** ($16): the sealed wave10 v6 request survives hash-matched
  but its execution source was lost with `.codex-run`; a faithful
  re-implementation is being built under tracked `zvf-program/` paths
  (per the lost-source policy) before launch.
*Integrated 2026-09-20 (squad e1-runtime/e1-driver/e1-tests + rebuild
completion):* wave10 re-implementation landed at `zvf-program/e1_wave10/`
(`bounded_procgroup.py`, `driver.py`, recovered sources) and verifies
10/10 targeted regressions offline, matching the lost-suite record
(`test_process_group.py` 4/4, `test_cleanup_regressions.py` 6/6). Fixes
vs the 09-19 snapshot: `process.json`/`worker.log` artifacts restored per
sealed fixtures, v6 call arity aligned, unresolved-verification wait seam
extracted, exact-identity reconcile implemented, caller phase runs the full
timeout per sealed `exec_controller_bound`. $16 launch pending, lead-only
(fresh authorization).
- **E5 successor27** ($80): reviewed and unreserved; the launch chain is
  being re-verified and rebuilt for admission, within the actor-ceiling
  constraint.
*Integrated 2026-09-20 (squad e5-tau3 + chain repair):* successor27 chain
re-verified offline (`zvf-program/e5_successor27/tests_offline.py`).
Fixed: duplicate-kwarg TypeError that failed every dispatch
(`native_recorded` journal append), plus three test-side wiring bugs
(unanchored wall-clock on the sealed 24h request, unwired halt-test clock,
strict key filter on journal scan) — controller safety properties
unchanged. *Fixed-all 2026-09-21:* the halt test now passes 52/52 — the
advance-only test double could never reach the halt path with zero intent
files (any task passing the first assert writes intent before the second
assert can fire), so `halt_after_ready` now raises DeadlineExceeded from
the readiness probe, exercising the genuine halt→stop→skip machinery with
the controller untouched. $80 admission launch pending, lead-only (actor
ceiling + fresh authorization).

### 9.6 External closures

Lanes with no local or paid path are being issued terminal close-out
records under the directive: E3 (SDAB private bundle), E7 (BinaryAudit
private payload), E8-original (LifeSciBench package), E10-original
(AgentHarm private tasks), E12 (AppBench deployment), E14-original
(FrontierMath hosted evaluation). Access requests previously sent remain
unanswered; the closures record the external block as terminal rather
than leaving the lanes pending.
*Integrated 2026-09-20 (squad external-closures, absorbed by lead):* six
closure records + INDEX + drafts at
`outputs/PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/`
(E3_sdab, E7_binaryaudit, E8_original_private, E10_agentharm_private,
E12_appbench, E14_frontiermath_private — all "no provider response" with
reopen conditions). Draft follow-up comments in `drafts.md` were NOT
posted (lead review pending).

### 9.7 Lanes awaiting verdict or judgment re-run

- **E4**: verdict RECEIVED 2026-09-19 — **CLOSED_PARTIAL**
  (`e4_completion/terminal_state_2026-09-19.json`): pass3/pass16 archive
  genuinely missing (0/6 artifacts; only TINKER_BRIDGE.json verifies), the
  0.3115 single-task recovery metric stands as receipt-only evidence; the
  only funded path is a fresh full rerun (~$59.45 grader-only lower bound),
  outside all envelopes unless separately authorized. JEV re-run routed
  FALLBACK_OUTAGE (`outputs/jev_receipts/E4_CLOSEOUT_JEV_2026-09-19.json`).
*Integrated 2026-09-21 (base-model full rerun, separately authorized):*
  user authorized a fresh 100-trial BankerToolBench rerun on the BASE model
  (zero-init LoRA snapshot == base sampling; no training steps) via the
  Modal bridge `/v1/responses` path, `--n-concurrent 2`, cap $55.91445263.
  Result: 100/100 trials completed (1 NonZeroAgentExitCodeError), MEAN
  REWARD 0.0 — all 100 in the 0.0 bucket (3h29m runtime; receipts at
  `outputs/e4_banker_toolbench/official_repo_ff6db552/jobs/btb-banking-tasks-tinker-bridge/result.json`).
  Mechanism (audited across all 100 trajectories): agents die after 2–14
  steps (median ~6) with no `deliverables/` output, so the verifier
  short-circuits 0.0 ("No deliverables found"); 27+ trajectories end in
  role-token degeneration (`user user assistant ...` repetition) — the
  3.6B base model cannot sustain the opencode tool-use loop, so this 0.0
  measures tool-dialogue collapse, not finance reasoning. Spend: $15.07
  charged of the $55.91 cap (16.35M prompt + 2.18M completion tokens).
- **jev_lab**: the two invalidated blocker-classification batteries ran
  2026-09-19 with all routings FALLBACK_OUTAGE (TypeSafe outage persisting;
  receipts at `outputs/jev_receipts/` incl. `TRIAGE_SUMMARY_2026-09-19.json`,
  `LEDGER_KEEPER_TERMINAL_STATE_JEV_2026-09-19.json`); re-run once
  `healthcheck()` passes. Design rule banked: arithmetic/string lookups
  stay in code, Jev reserved for classification/faithfulness/value
  ordering.
