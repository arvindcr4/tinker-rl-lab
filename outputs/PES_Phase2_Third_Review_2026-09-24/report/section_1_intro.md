## 1. Introduction and scope

### 1.1 The question

This report documents an evaluation campaign, not a training result. It asks
whether one open-weights actor — `Qwen/Qwen3.6-35B-A3B` (base commit
`995ad96e`) with the Tinker-trained adapter
`arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1`
(commit `64444133`), served in bfloat16 — can be characterised across a
heterogeneous portfolio of held-out agentic and reasoning benchmarks such that
every figure is produced by the benchmark's own native evaluator at a pinned
revision, and remains defensible independently of the training pipeline. A
second question runs alongside it: what does such a campaign establish when a
substantial share of its lanes cannot be scored at all?

### 1.2 Why held-out suites rather than reward curves

Training-reward traces were recorded throughout the campaign and are not used as
evidence here. A training reward is computed by the campaign's own verifier over
the tasks the policy was optimised on, so it measures progress against that
objective and cannot separate a change in capability from increased pressure on
it. The campaign's protocol states the point: a completed ten-step training arm
carrying a recorded reward trace "is not held-out or portfolio evidence"
(source: zvf-program/flagship/PAVLOV_EXPERIMENT_PROTOCOL_2026-08-09.md §1).

Held-out suites were chosen instead because their graders are authored upstream,
pinned at a revision, and outside the training loop. Where a lane's original
benchmark was unobtainable, a public replacement scope was used and is named as
such. Those public inventories were also audited for content overlap against the
pinned training source envelope: no field of eight or more normalised tokens
matched exactly, near, or by containment, and the audit records that short exact
matches do occur and declines an unqualified "zero overlap" claim
(source: outputs/public_portfolio_2026-09-05/decontamination/README.md).

### 1.3 Governance

**Receipts.** Each figure traces to a surviving evidence file. Sealed requests
and validation receipts are hash-matched even where the execution sources that
produced them were lost; all structural claims were re-verified by deterministic
code checks on 19 September 2026, 11 of 11 passing
(source: outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json).

**Budget gates.** Budget mode is unlimited, superseding the earlier cumulative
caps; per-run resource and timeout bounds remain, spend is gated per launch on a
technical chain and a preflight, and paid launches are lead-only authority
(source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §9.2).

**Honest partial reporting.** Lanes that produced no score are reported as such;
each of the fourteen holds an explicit terminal state in the ledger of record
(source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

### 1.4 What one actor across many suites buys

Fixing the actor at a single pinned revision is the campaign's main
methodological lever. No lane received fine-tuning, prompt tuning, or
evaluator-specific adaptation, and sampling conditions were preserved rather
than tuned per suite; differences between lanes therefore bear on what the tasks
demand rather than on per-suite optimisation. The same discipline separates
three quantities a single table would merge — what was prepared, what was
executed, and what was scored — which makes a partial lane interpretable rather
than merely missing.

### 1.5 The fourteen lanes

Each lane was designed around one original benchmark contract. The table names
that contract, the replacement scope where one was used, and the lane's terminal
state.

| Lane | Original contract suite | Replacement scope | Terminal state |
|---|---|---|---|
| E1 | SWE-bench Pro | SWE-bench Multilingual | REBUILD_READY_LAUNCH_PENDING |
| E2 | FrontierSWE | CORE-Bench | AMENDMENT_ACCEPTED_LAUNCH_PENDING |
| E3 | SDAB (private bundle) | — | CLOSED_EXTERNAL |
| E4 | BankerToolBench | — | CLOSED_PARTIAL |
| E5 | APEX-Agents | Tau3 | REBUILD_READY_LAUNCH_PENDING |
| E6 | WebBench | WebArena | PENDING_QUOTA |
| E7 | BinaryAudit | — | CLOSED_EXTERNAL |
| E8 | LifeSciBench | LAB-Bench (public split) | COMPLETE (public) |
| E9 | MLE-bench | MLDevBench | PENDING_QUOTA |
| E10 | AgentHarm | AgentDojo (benign utility) | COMPLETE (benign) |
| E11 | VerilogEval | two native framings (no replacement) | COMPLETE |
| E12 | AppBench | — | CLOSED_EXTERNAL |
| E13 | OpenReward Games | BALROG | AMENDMENT_ACCEPTED_LAUNCH_PENDING |
| E14 | FrontierMath | Omni-MATH | COMPLETE_TERMINAL_NOTE |

E8, E10 and E14 each hold two states: the replacement scope shown above, and a
`CLOSED_EXTERNAL` original contract whose private material was not released to
this campaign (source:
outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

### 1.6 What this report does not claim

- No improvement over any baseline, and none over the base model the adapter was
  trained from; no such comparison was run.
- No pooling of original-contract and replacement-scope numbers, and no
  cross-suite average or aggregate.
- No backend numerical parity between the merged BF16 serving path used for some
  lanes and the original Tinker sampler used for others.
- No replacement benchmark's score presented as the score of the benchmark it
  replaces.
- No global pretraining cleanliness, semantic disjointness, or image cleanliness
  beyond the audited inventory; the decontamination audit's scope is limited and
  is reported as limited.
- No figure for a lane that has none. Where a lane has no score, this report
  says so plainly and names the blocking gate.
