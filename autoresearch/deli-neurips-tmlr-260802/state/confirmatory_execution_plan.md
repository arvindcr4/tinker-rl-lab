# Confirmatory Matrix Execution Plan

Source tag: `w4-plan` | Date: 2026-08-02 | Status: PLAN ONLY — no runs launched, no study files modified.

Authority chain read: `preregistration.json` (+ A001/A002/A003), `results_contract.json` (sha `bef7dd99…` matches prereg binding), `claim_ledger.json` (sha `0ab19d64…` matches), `execution_authorization.json`, all four provider launchers, `remote_preflight.py`, `verify_preflight_matrix.py`, all preflight receipts under `zvf-program/next-submission/results/`.

---

## 0. Matrix size correction: 92 runs, not 64

The tasking framed the matrix as 2 tasks x 2 arms x **16** seeds = 64 runs. That is the **pre-A003** plan and is superseded. Amendment A003 (`protocol_amendment_003_confirmatory_hardening.json`, prospective, 0 confirmatory rows completed) explicitly replaces the 16-seed plan: *"normal_approximation_seed_count_before_inflation (13) and planning_seed_count_after_inflation (16) are replaced by the exact-t values"* — exact noncentral-t at the Holm worst-case alpha 0.0125 gives 19 seeds, inflated x1.2 to **23**. The frozen results contract publication gate reads: *"Every one of the 92 task-arm-seed rows planned under the A003-amended 23-seed plan, or every row added by the blinded variance reassessment up to the 24-seed cap, must be complete and hash-valid before the main table can be generated."*

**This plan therefore schedules 92 runs (23 paired seeds x 4 cells), with a contingency of 96 (seed 349 reserve) if the blinded reassessment raises n to the cap.** Planning to 64 would guarantee an INCONCLUSIVE verdict under the frozen gate.

Seeds (frozen): 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347. Reserve (24th, only if reassessment demands it): 349.

Preflight seed reuse is preregistered as harmless (A003 `preflight_seed_reuse_policy`): confirmatory units are fresh full runs; the preflight burns of 211/223/227 do **not** exclude those seeds from the confirmatory matrix.

## 1. Per-run duration estimates

### Derivation

A preflight (hash-bound `remote_preflight.py`) is exactly one full-size confirmatory optimizer step (generation_batch_size=16, num_generations=8 → 2 rollout groups/step, completions ≤1024 tok) plus a greedy held-out eval of n=8 at the hard-bound `eval_batch_size=2`. A confirmatory run per preregistration = **30 such steps + full held-out eval** (gsm8k n=1000, math500 n=500). `wall_clock_seconds` in receipts covers dataset/tokenizer/model load + training + eval (excludes session allocation, pip, upload).

Observed Colab A100 receipts (`results/preflight/results/*.json`):

| cell (seed) | wall s | eval s | setup+1 step s |
|---|---|---|---|
| gsm8k grpo_g8 (211) | 349 | 112 | 237 |
| gsm8k grpo_g8 (223) | 420 | 107 | 313 |
| gsm8k contrast (211) | 211 | 107 | 104 |
| gsm8k contrast (227) | 239 | 109 | 130 |
| math500 grpo_g8 (211) | 711 | 264 | 446 |
| math500 grpo_g8 (223) | 850 | 265 | 585 |
| math500 contrast (211) | 428 | 266 | 162 |

Setup F ≈ 90 s (inferred from the near-empty contrast steps). Per-step training: gsm8k grpo ≈ 147–223 s (mid 185); math500 grpo ≈ 356–495 s (mid 426). Contrast steps were homogeneous-early-stop in preflight (4 rollouts, ~25–72 s); in confirmatory training mixed groups expand to G=8, so contrast per-step time lies between that floor and the grpo ceiling — **planned at the grpo ceiling** (conservative; expected ~70–85% of it).

Eval extrapolation at the observed 13.4–14.0 s/prompt (gsm8k) and 33.1 s/prompt (math500): gsm8k 1000 → ≈ 13,600 s (3.8 h); math500 500 → ≈ 16,550 s (4.6 h). **Eval dominates gsm8k runs (~65%) and is ~55% of math500 runs.**

### Estimates (Colab A100, single-shot, + 0.5 h/run session-allocation + env + model download + artifact upload overhead)

| cell | training | eval | total/run | range |
|---|---|---|---|---|
| gsm8k / grpo_g8 | ~1.55 h | ~3.8 h | **~5.9 h** | 5.3–6.4 h |
| gsm8k / contrast | ≤1.55 h | ~3.8 h | **~5.6 h** | 4.9–6.4 h |
| math500 / grpo_g8 | ~3.55 h | ~4.6 h | **~8.8 h** | 8.0–9.5 h |
| math500 / contrast | ≤3.55 h | ~4.6 h | **~8.3 h** | 7.3–9.5 h |

GPU-hour totals (23 seeds): gsm8k 46 runs ≈ 265 GPU-h; math500 46 runs ≈ 395 GPU-h; **total ≈ 640–680 GPU-h** (contrast savings could pull this toward ~600).

**Biggest lever (flagged, not assumed):** the confirmatory runner does not exist yet (§7). If its frozen spec pins a larger uniform eval batch (e.g. 8) instead of the preflight's 2, eval time drops ~3x and the matrix shrinks to roughly 380–420 GPU-h. This is an evaluator-adjacent choice ("evaluator" is a preregistered fixed component) — it must be decided once, hash-bound before run 1, identical across all 92 runs, and disclosed; bf16 batched greedy decoding is not guaranteed bit-identical across batch sizes, so whichever value is frozen defines the evaluator. Baseline numbers above assume batch 2 (matches the preflighted evaluator).

## 2. Provider split, concurrency, wall-clock, cost

### Conformance finding — multi-provider parallelism is NON-CONFORMANT as implemented

The stack fingerprint (`run_preflight.py` `run_unit`, ~line 911) is `fingerprint({runtime_packages, accelerator, provider: "colab", colab_cli_version, trainer, sampler_sha256, adapter_sha256, decoder})`. It **binds provider identity, Colab CLI version, and accelerator**. Consequences, all verifiable in the repo:

- The frozen gate fingerprint `f739ee5a…` is Colab/A100 (confirmed in `results/preflight/preflight_gate.json` and the W1 log).
- The HF Jobs receipt carries a *different* fingerprint (`a12a2d66…`) for an identical scientific stack.
- `verify_preflight_matrix.py` (~line 308) requires `len(stack_fingerprints) == 1` across receipts; `results_contract.json` requires `stack_fingerprint` on every seed row; claim scope pins "one pinned open canonical GRPO stack"; the gate requires `require_shared_scientific_stack_fingerprint`.

Therefore a matrix split across Colab + HF + Kaggle + GCP would produce rows with ≥2 distinct stack fingerprints and fail hash-validation of the frozen gate/contract logic. "Provider is provenance, not treatment" is the *intent*, but the *implementation* makes provider part of the stack identity. Using a second provider would require a prospective amendment (A004-style: provider-independent fingerprint definition, hash-bound before any confirmatory row, 0 rows completed — same pattern as A003). **This plan does not assume that amendment.**

Provider-specific disqualifiers (independent of the fingerprint issue):

| provider | status | disqualifier for confirmatory |
|---|---|---|
| Colab (`run_preflight.py`) | working; all 7 valid receipts | none — primary |
| GCP (`run_gcp_preflight.py`) | working (1 receipt) | hash-bound `DEFAULT_MAX_RUN_DURATION="90m"` Spot bound + $3 cost cap cannot fit a 5.6–8.8 h single-shot run; exact-resume chaining does not exist in the bound stack. Also ~2.4x slower observed (eval 366 s vs 107 s Colab, a2-highgpu-1g A100-40GB image) |
| HF Jobs (`run_hf_jobs_preflight.py`) | failed | HTTP 402 Payment Required (credit balance empty); a100-large flavor otherwise plausible |
| Kaggle (`run_kaggle_preflight.py`) | failed | runtime allocated **P100** despite A100 request (no bf16 → `remote_preflight` fails closed), `kaggle-secret-lookup` ConnectionError, and ~30 GPU-h/week quota << 640 GPU-h need |

### Split

- **Colab A100: 92/92 runs (100%).** Sole conformant provider.
- HF Jobs a100-large: standby recovery capacity **only if** (a) an A004-style fingerprint amendment is authored and hash-bound before any confirmatory row, and (b) credits are funded. Not in the baseline.
- GCP 90-min Spot: preflight/diagnostic and receipt-bucket infrastructure only. Never a confirmatory host under the current bound.
- Kaggle: excluded outright.

### Concurrency

The protocol's `require_provider_session_absence` is a per-run cleanup verification (the run's own session must be provably terminated at receipt time — see `session_absence_verified` in `run_preflight.py`), **not** a global one-session-at-a-time rule; the prereg imposes no cross-run scheduling constraint. Colab's own product limit on concurrent A100 sessions (typically 1–3 depending on plan/availability) governs.

- Baseline **K=2** concurrent sessions (one per arm of the active seed-pair). Opportunistic K=3 when Colab grants it (start the next pair's first arm). Degrade to K=1 if allocation fails; the plan remains valid, only slower.
- One confirmatory unit per session; session named per unit fingerprint (launcher convention); commit receipt before that slot launches its next unit.

### Wall-clock and cost

| concurrency | wall-clock (92 runs, incl. ~8% retry/idle) | calendar |
|---|---|---|
| K=1 | ~710 h | ~30 d |
| **K=2 (baseline)** | **~360 h** | **~15 d** |
| K=3 | ~245 h | ~10.5 d |

Cost at Colab pay-as-you-go (A100 ≈ 11.77 compute-units/h, $9.99/100 units → ≈ $1.18/GPU-h): 640–680 GPU-h ≈ **$750–800; plan $700–950** including allocation retries and idle session tails. Reference ceiling if priced at the hash-bound GCP Spot rate ($1.928/h): ~$1,240. Contingency +4 runs (n=24): +~29 GPU-h ≈ +$35. Eval-batch lever (§1), if frozen into the runner: totals drop to ~$450–520 and ~9 days at K=2.

## 3. Seed schedule (paired semantics honored)

Pairing semantics per prereg: the analysis pairs **by seed across arms within a task cell** (`independent_unit: "training seed within a frozen task-model-stack cell"`; estimands are paired differences/log-ratios per seed). The prereg does **not** impose temporal co-scheduling of pair members — comparability is delivered by the frozen stack, pinned data revisions, and shared fingerprint, not by calendar adjacency. To keep provenance maximally comparable anyway, this plan adopts (as a convention, explicitly not a protocol requirement): **both arms of (task, seed) run concurrently in the two session slots, same provider, same GPU class, launched from the same source commit.**

Order of execution (ascending seeds, task-alternating so all 4 cells reach 8 completed pairs as early as possible):

```
for s in [211, 223, 227, 229, 233, 239, 241, 251]:      # Phase A: first 8 seeds
    slotA: gsm8k/grpo_g8(s)    ∥  slotB: gsm8k/contrast(s)        (~5.9 h)
    slotA: math500/grpo_g8(s)  ∥  slotB: math500/contrast(s)      (~8.8 h)
→ CHECKPOINT (≈ day 5 at K=2): all 4 cells have 8 completed pairs
→ run the preregistered blinded variance reassessment (pooled, arm-label-hidden;
   exact noncentral paired-t at alpha 0.0125; increase-only; cap 24;
   power implementation: zvf-program/audit/aggregate_audit.py, sha c323c4ed…)
→ record power receipt (required before final analysis); final n ∈ {23, 24} or STOP_UNDERPOWERED

for s in [257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347]:  # Phase B
    same pattern                                                   (15 seeds, ~10 d at K=2)

if reassessment fixed n = 24:                                      # Phase C (conditional)
    s = 349: all 4 cells, same pattern                             (+~15 h)
```

Rules: seeds are consumed in frozen ascending order; **no seed substitution ever** (the W1 preflight seed-escalation pattern applied only to non-evidence preflights and is prohibited here); a failed-infrastructure unit is resumed/re-attempted at the *same* seed before the schedule advances past it; if one arm of a pair fails, its partner's completed run stands (ITT) and only the failed unit is re-attempted.

## 4. Receipt-commit cadence and recovery playbook

### Cadence

Precedent: one local commit per recorded run (repo history: `2aa59c5c` single-run diagnostic, `c4a2e73b` receipt recovery, `cbdf260e` hash binding). Per unit, immediately after the local receipt validates:

1. `git add` the unit's result JSON + request JSON + log under `results/…`; nothing else.
2. One commit: `Record confirmatory run <task>/<arm>/s<seed> receipt` (or `…failed_infrastructure receipt`).
3. **Never push.** Working tree must be clean before the next launch (the launcher's `source_commit(require_clean=True)` enforces this fail-closed).

Additional single commits at: gate/status regenerations, the Phase-A power receipt, each recovery event, and any STOP declaration. Maximum receipt-commit backlog: 1 run (see STOP-7).

### Recovery playbook

Design invariant carried over from the preflight stack: the **request artifact is written before launch** and is the recovery root; receipts are reconstructed from logs or remote artifacts (HF repo + W&B), never re-run to "replace" a possibly-completed unit.

**Colab (primary).** If the local launcher dies, the CLI disconnects, or a session outcome is unknown:
1. Do NOT relaunch the unit.
2. Locate the request artifact: `results/<provider-dir>/requests/<task>__<arm>__s<seed>__<fp12>.json`.
3. Run `python zvf-program/next-submission/run_preflight.py --recover-request <request.json>` (confirmatory runner must expose the identical flag). This verifies/enforces session absence (stops a still-live session), rebuilds the result from the local log or the remote HF/W&B artifacts, re-verifies the remote, and writes the receipt atomically. Precedent: commit `c4a2e73b` "Recover stranded preflight receipts from request artifacts".
4. Commit the recovered receipt (one commit). If recovery yields `failed`, record it as such — then and only then re-attempt the same unit fresh.

**HF Jobs (only if amended in).** `provider_job_id`/`url` live in the request artifact; poll job state via API; artifacts recover from the private HF repo + W&B run. An `allocation-rejected`/402 consumed no GPU: record the failure receipt, do not count as an attempt against the unit.

**GCP (infrastructure only).** Receipts land in GCS bucket `arvindcr-tinker-rl-preflight-358208640342` (`gcp_receipt_bucket.json`); recovery = pull from bucket and validate locally. Note for any future amendment: the 90-min Spot bound means a confirmatory attempt is *guaranteed* to be preempted mid-run — without exact-resume chaining GCP can only ever produce `failed_infrastructure` rows.

**Kaggle.** Excluded. If a stranded historical artifact surfaces, the kernel output (kernel_id in the request) is the recovery source; it can never become a confirmatory receipt (P100/no-bf16 fingerprint mismatch → quarantined).

Generic: request artifacts are append-only; incompatible prior results are archived, never overwritten (`archive_incompatible_result`); a recovered receipt counts only after sha/fingerprint validation.

## 5. Failure and missingness handling (preregistered policy)

Frozen policy (`analysis.missingness` + `results_contract.allowed_statuses`): *intention-to-treat; exact resume for infrastructure interruption; no failed run is silently dropped or relabeled as scientific failure.*

- Every launched unit gets a ledger row that terminates in one of: `complete`, `failed_infrastructure`, `invalid`, `quarantined`. Nothing is deleted.
- **Infrastructure interruption** (preemption, OOM from host, disconnect, provider kill): exact resume of the same unit where the runner supports it; otherwise record `failed_infrastructure` (receipt + commit) and re-attempt the **same task/arm/seed** fresh, with the failed receipt retained as provenance. Re-attempts do not rotate seeds.
- **Scientific outcomes are never failures**: a contrast run with an extreme homogeneous fraction, zero updated groups, or poor accuracy is `complete` and enters the analysis as-is (ITT). Relabeling it is prohibited.
- **Hash/fingerprint mismatch** on any receipt → `quarantined`, never analyzed, unit re-attempted.
- The joint verdict rule makes completeness binding: *"any missing or invalid cell yields INCONCLUSIVE"* — the schedule always prioritizes completing a damaged unit over advancing to new seeds.
- The negative-result rule is preregistered: boundary-crossing interval, failed power receipt, or incomplete cell → **INCONCLUSIVE**, not equivalence, not failure. No scheduling decision may be conditioned on unblinded arm effects (`reassessment_may_inspect_arm_effect: false`).

## 6. STOP conditions (stop launching; log; escalate to owner)

1. **Gate not passed.** `preflight_gate.json` says `confirmatory_execution_gate: "blocked"` (3 mixed-update seams missing). Zero confirmatory launches until the gate regenerates green.
2. **No hash-bound confirmatory runner.** All four launchers are preflight-only (`max_steps=1`, `heldout_n=8` hard-coded). Launching "confirmatory" runs through them is impossible; a confirmatory runner must be written, hash-bound into the protocol bindings (prospective amendment, 0 rows completed), and must emit the full required telemetry (`policy_ratio_q05_q50_q95`, `clip_fraction_by_advantage_sign`, both KL fields, `parser_disagreement`, `two_sample_false_homogeneity` — not produced by the current preflight telemetry).
3. **Provider systematic failure**: ≥3 consecutive `failed_infrastructure` on distinct units, or >50% session-allocation failures over 24 h → stop launching, escalate. Never self-migrate to another provider (fingerprint, §2).
4. **Stack fingerprint drift** on any receipt (pin resolution change, CLI version bump, non-A100 allocation) → immediate stop; quarantine; escalate. No "close enough" hardware.
5. **STOP_UNDERPOWERED**: blinded reassessment requires n > 24 → preregistered terminal verdict; stop the matrix.
6. **Cost guard**: cumulative spend > 150% of the phase estimate → pause, escalate with receipts.
7. **Governance guard**: dirty working tree at launch time, or receipt-commit backlog > 1 run → pause launches until reconciled.
8. **Verification outage**: HF Hub or W&B unavailable such that receipts cannot be verified → pause; unverifiable runs are quarantined, not trusted.
9. **Pin failure**: model revision, dataset revision, or package pin no longer resolves → stop; any workaround is a stack change requiring an amendment.

## 7. Preconditions before run 1 (current blockers)

| # | blocker | owner/step |
|---|---|---|
| 1 | Preflight gate `blocked` — mixed-update seams missing for gsm8k/contrast, math500/contrast, math500/grpo_g8 (W1 in progress; seed plan s227/s223/s223) | W1, then regenerate gate |
| 2 | Hash-bound confirmatory runner does not exist (30-step, full-heldout, full telemetry, `--recover-request`, exact-resume) — requires prospective amendment binding before any row | study owner |
| 3 | Eval-batch freeze decision (batch 2 vs larger; §1 lever) must be inside blocker-2's amendment | study owner |
| 4 | Tasking's 64-run/16-seed framing is stale; contract gate requires 92 rows (23 seeds, cap 96) — acknowledge before scheduling | orchestrator |
| 5 | Multi-provider parallelism non-conformant (fingerprint binds provider/CLI/GPU); Colab-only baseline stands unless an A004-style amendment lands first | study owner (optional) |
| 6 | HF Jobs credits exhausted (HTTP 402) — moot under Colab-only baseline | only if 5 pursued |

Preflight receipts remain non-evidence throughout; nothing in this plan promotes them. No pushes, no submissions, no edits to `claim_ledger.json` or `results_contract.json`.
