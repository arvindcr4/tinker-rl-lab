# Preflight seam blocker — status after A003

Tag: `w10-seam`. Written 2026-08-02T10:00Z. Read-only analysis; no protocol file was modified.

**Verdict: the blocker still binds. A003 is orthogonal to it. An amendment is required, and the
unused id is A004.**

---

## 1. What A003 actually did

`zvf-program/next-submission/protocol_amendment_003_confirmatory_hardening.json`
(`amendment_id: A003_confirmatory_hardening`, bound into `preregistration.json` at commit
`5d3ec90d`, sha `25047e5a806651700b9112fb9f77f6b549955e54a4e1b98bad986b99d36e72f8`).

Verified by reading the amendment and by `git show 5d3ec90d -- zvf-program/next-submission/preregistration.json`.

Changed:

| Area | Change |
|---|---|
| `treatment.objective` / `.optimizer` / `.precision_and_memory` / `.adapter` | Newly pinned in the protocol (epsilon 0.2, lr 1e-6, adamw_torch_fused, LoRA r=16 …). Previously these lived only inside the preflight script. |
| `treatment.fixed_components` | += "objective hyperparameters", "adapter configuration" |
| `paired_seed_plan` | `planning_seed_count` 16 → 23; seeds 307–347 appended; derivation rule + `seed_24_reserve: 349` added |
| `power_plan` | normal approximation → exact noncentral paired-t; `planning_alpha_worst_case: 0.0125`; seed count 13/16 → 19/23; joint-power disclosure |
| `probe_size_justification` | New prose justifying G=2 (no matrix change) |
| `preflight_seed_reuse_policy` | **New**: permits non-evidence preflights on preregistered confirmatory seeds |
| `telemetry_contract_completion` | 8 fields added to the results contract |
| `protocol_self_binding` | `execution_authorization.json` now records `protocol_canonical_sha256` |
| `bindings` | `results_contract_sha256`, `execution_authorization_sha256` refreshed; A003 + power-implementation bindings added |

Not changed — checked field by field against the diff:

- `preflight_execution_gate` — **byte-identical**. `require_live_mixed_reward_optimizer_update_per_cell`
  is still `true`, `failure_action` is still `block_confirmatory_execution`.
- The preflight window. `max_steps=1`, `heldout_n=8`, and the 2-group rollout batch are set in
  `remote_preflight.py` (lines 414–418: `per_device_train_batch_size=2`,
  `gradient_accumulation_steps=8`, `generation_batch_size=16`, `num_generations=8`, `max_steps=1`)
  and its hash binding `remote_preflight_sha256` was untouched.
- The seam-verification rule in `verify_preflight_matrix.py` — its binding
  `preflight_matrix_verifier_sha256` was untouched.
- `matrix`, `tasks`, `treatment.decoder`, arms, margins, estimands.
- Seed **count** rose 16 → 23, but that is the confirmatory seed plan. Preflights are not drawn
  from a "seed count"; they are per-attempt. More available seeds does not raise the per-attempt
  seam probability, it only means more seeds could in principle be burned.

**A003 also made the fix space narrower, not wider.** `preflight_seed_reuse_policy` states verbatim:

> "A preflight is at most one optimizer step with heldout_n=8, is labeled preflight-not-evidence,
> and can never populate the confirmatory ledger."

That sentence now pins `max_steps` and `heldout_n` inside a committed amendment. It is silent on
`rollout_groups`, which is precisely the lever the seam needs. A004 is drafted to exploit that
silence rather than to reopen the clause.

**Why A003 did not invalidate the existing receipts.** All 7 receipts carry
`protocol_sha256 = 9a25e44b0a70…` (the pre-A003 protocol, which is also A003's declared
`prior_protocol_sha256`). `validate_receipt` only requires `run_config.protocol_sha256 ==
receipt.protocol_sha256` — internal consistency — and never compares either against the on-disk
protocol. It does compare `run_config.decoder` against `protocol.treatment.decoder` and the three
source hashes against `protocol.bindings`, all of which A003 left alone. Re-running
`verify_preflight_matrix.py` at HEAD reproduces the committed gate exactly:
`blocked`, `receipt_count: 7`, same three missing seams.

---

## 2. The seam rule, exactly as currently defined

`zvf-program/next-submission/verify_preflight_matrix.py`:

```python
# validate_receipt(), per receipt:
"mixed_update_observed": audit["mixed_fraction"] > 0 and audit["updated_groups"] > 0,

# evaluate_matrix(), per (task, arm) cell:
mixed_update = any(record["mixed_update_observed"] for record in cell_records)
if not mixed_update:
    missing_scientific_seams.append(f"{task}/{arm}:mixed_reward_optimizer_update")
...
ready = not missing_scientific_seams
"confirmatory_execution_gate": "pass" if ready else "blocked"
```

`_validate_audit` additionally enforces `mixed_fraction * rollout_groups == updated_groups`, so
`mixed_fraction > 0` and `updated_groups > 0` are equivalent given `rollout_groups > 0`. The seam is
therefore: **at least one rollout group in at least one receipt for that cell was classified
`mixed`.** It is a disjunction over receipts, so the fix can be "more groups per receipt" or "more
receipts"; it is *not* aggregated across cells, so the observed `gsm8k/grpo_g8` seam does nothing
for the other three cells.

What "mixed" means per arm (`contrast_sampler.py:99-168`):

- `grpo_g8`: all eight completions are generated; `group_class` is computed on all 8;
  `update_applied = (group_class == "mixed")`. So **P(seam per group) = 1 − p⁸ − (1−p)⁸**.
- `contrast_early_stop_g2_to_g8`: the class is decided by the G=2 probe. A homogeneous probe never
  expands (`update_applied = False`); a mixed probe expands by 6 and is *necessarily* mixed on all 8
  because the two probe rewards already differ. So **P(seam per group) = 2p(1−p)**, capped at 0.5.

---

## 3. Per-cell observation probability under the current window

### 3.1 Data

The receipts record only the group *class*, never per-completion rewards (the training reward trace
is not serialised; only `heldout_trace` is, and that is greedy-decoded eval, a different
distribution). The 7 receipts therefore give 14 censored group observations:

| receipt | arm | n per group | group classes |
|---|---|---|---|
| gsm8k contrast s211 | G=2 probe | 2 | all_correct, all_correct |
| gsm8k contrast s227 | G=2 probe | 2 | all_correct, all_correct |
| gsm8k grpo_g8 s211 | G=8 | 8 | all_correct, all_correct |
| gsm8k grpo_g8 s223 | G=8 | 8 | all_correct, **mixed** |
| math500 contrast s211 | G=2 probe | 2 | all_correct, all_wrong |
| math500 grpo_g8 s211 | G=8 | 8 | all_correct, all_wrong |
| math500 grpo_g8 s223 | G=8 | 8 | all_correct, all_correct |

`shuffle_dataset=True` with `data_seed=seed` (`remote_preflight.py:442-444`), so every seed draws a
different prompt pair; the 14 groups are 14 near-independent prompt draws.

### 3.2 Model

Prompt *i* has an unknown per-completion success probability *pᵢ*; completions within a group are
iid Bernoulli(*pᵢ*) under the frozen decoder (T=0.7, top_p=0.8, top_k=20, `enable_thinking=false`).
*pᵢ* ~ Beta(a,b) across the training split. Censored likelihood per group:

```
P(all correct | n) = B(a+n, b) / B(a, b)
P(all wrong   | n) = B(a, b+n) / B(a, b)
P(mixed       | n) = 1 − the two above
```

Fitted per task, pooling both arms (a prompt's difficulty does not depend on which arm samples it).

**The MLE is degenerate and is reported only as a diagnostic.** On a (μ, κ) grid spanning
κ ∈ [4.5e−5, 3.3e6] the gsm8k MLE runs to the upper κ edge (μ=0.9725, κ→∞: every prompt has the
same p=0.9725) and the math500 MLE runs to the lower κ edge (κ→0: every prompt is deterministically
right or wrong, giving P(mixed) ≈ 2e−5). Both are zero-count artefacts of 6–8 observations. The
headline numbers below are the posterior mean of P(mixed) under a standard weakly informative
hierarchical prior (μ ~ U(0,1), p(κ) ∝ (1+κ)^−3/2), integrated on the same grid.

Fitted per-group seam probabilities *q*:

| task | q at n=2 (probe) | q at n=8 | MLE lower diagnostic (n=2 / n=8) |
|---|---|---|---|
| gsm8k | **0.09393** | **0.25963** | 0.0535 / 0.2000 |
| math500 | **0.05783** | **0.14629** | ~0 / ~0 |

Cross-check, model-free: Jeffreys posterior mean (k+½)/(n+1) on the raw per-cell mixed counts gives
q = 0.100 (gsm8k contrast, 0/4), 0.167 (math500 contrast, 0/2), 0.100 (math500 grpo_g8, 0/4) — the
same order of magnitude.

### 3.3 Arithmetic, per blocked cell, per attempt (2 rollout groups per attempt)

`P(attempt clears) = 1 − (1 − q)²`

**gsm8k / contrast_early_stop_g2_to_g8** (q = 0.09393)
```
1 − q         = 0.90607
(1 − q)²      = 0.82096
P(attempt)    = 1 − 0.82096 = 0.17904
attempts for 95%: ceil( ln 0.05 / ln 0.82096 ) = ceil( −2.99573 / −0.19725 ) = ceil(15.19) = 16
E[attempts]   = 1 / 0.17904 = 5.59
```

**math500 / contrast_early_stop_g2_to_g8** (q = 0.05783)
```
1 − q         = 0.94217
(1 − q)²      = 0.88768
P(attempt)    = 1 − 0.88768 = 0.11232
attempts for 95%: ceil( −2.99573 / −0.11914 ) = ceil(25.15) = 26
E[attempts]   = 8.90
```

**math500 / grpo_g8** (q = 0.14629)
```
1 − q         = 0.85371
(1 − q)²      = 0.72882
P(attempt)    = 1 − 0.72882 = 0.27118
attempts for 95%: ceil( −2.99573 / −0.31626 ) = ceil(9.47) = 10
E[attempts]   = 3.69
```

**All three cells in one round of three attempts:**
```
0.17904 × 0.11232 × 0.27118 = 0.005453   →  0.55%
```

**Retrospective consistency check.** The five attempts spent on the three blocked cells were
gsm8k/contrast ×2, math500/contrast ×1, math500/grpo_g8 ×2. Probability all five miss:
```
0.82096² × 0.88768 × 0.72882² = 0.67398 × 0.88768 × 0.53118 = 0.3178
```
A 32% event. The five failures are the expected behaviour of a 2-group window, not a defect —
which is exactly why more seeds is the wrong lever.

**Cost of clearing the gate by burning seeds:** 16 + 26 + 10 = **52 further A100 preflight
allocations** for 95% per-cell confidence, against 23 preregistered seeds per cell and a documented
history of interrupted Colab sessions.

### 3.4 The correction to the withdrawn spec

`state/amendment_A003_spec.md` (withdrawn) assumed p ≈ 0.9 and proposed `rollout_groups = 8`,
projecting ≈0.80 per attempt. The receipt-fitted difficulty is higher than that (gsm8k posterior
mean p ≈ 0.97). At M = 8 groups the actual figures are 0.5458 / 0.3791 / 0.7178 per cell and
**0.1485 jointly**. Eight groups is not enough. Full ladder:

| groups per attempt M | gsm8k/contrast | math500/contrast | math500/grpo_g8 | joint |
|---|---|---|---|---|
| 2 (today) | 0.1790 | 0.1123 | 0.2712 | 0.0055 |
| 8 (withdrawn spec) | 0.5458 | 0.3791 | 0.7178 | 0.1485 |
| 16 | 0.7937 | 0.6145 | 0.9204 | 0.4488 |
| 24 | 0.9063 | 0.7606 | 0.9775 | 0.6738 |
| 32 | 0.9574 | 0.8514 | 0.9937 | 0.8099 |
| **48** | **0.9912** | **0.9427** | **0.9995** | **0.9339** |
| 60 (= one confirmatory unit) | 0.9973 | 0.9720 | 0.9999 | 0.9693 |

---

## 4. Draft amendment A004

`A004` is unused: `preregistration.json.amendments` has exactly three entries;
no `protocol_amendment_004*.json` and no `zvf-program/next-submission/PROPOSED_AMENDMENT_A004*.md`
exists at the time of writing. If a sibling agent lands an A004 first, this spec re-ids to **A005**
with no content change.

### 4.1 Spec

**File:** `zvf-program/next-submission/protocol_amendment_004_seam_verification_window.json`
**`amendment_id`:** `A004_seam_verification_window`
**`status`:** `prospective_before_confirmatory_execution`
**`schema_version`:** `aiml-next-protocol-amendment-v1` (same shape as A001/A002/A003)

**Scope — one sentence:** this amendment changes only how many rollout groups a *seam-verification
preflight* may draw before it stops. It changes nothing that a confirmatory unit does.

**`change`:**

1. **New preflight class.** `preflight_class ∈ {"matrix_infrastructure", "seam_verification"}`,
   recorded in `run_config` and in the receipt. Everything already committed defaults to
   `matrix_infrastructure` and is unaffected.
2. **Window for `preflight_class == "seam_verification"` only:**
   - `rollout_groups_cap`: **48** for `contrast_early_stop_g2_to_g8`, **24** for `grpo_g8`.
     (Contrast needs more groups because a G=2 probe is mixed with probability 2p(1−p); the
     baseline classes on all 8 samples and clears far sooner.)
   - **Stop-on-first-seam:** the run stops drawing new groups as soon as one group has been
     classified `mixed` *and* the optimizer update for the accumulating batch has been applied,
     or when `rollout_groups_cap` is reached — whichever comes first.
   - `heldout_n`: **8, unchanged.**
   - `num_generations`: **8, unchanged.** `initial_group_size` 2 / `expansion_group_size` 6 /
     expand rule / homogeneous rule: **unchanged.**
   - Decoder, model revision, reward parser, `max_completion_length=1024`, LoRA config, objective,
     learning rate: **unchanged.**
   - Peak generation batch stays at **16 completions per `generate()` call** — the value already
     exercised on A100-40GB — so no new memory failure mode is introduced.
   - Optimizer-step budget: **≤ 24**, strictly below the confirmatory 30.
3. **Implementation freedom, hash-bound either way.** Preferred implementation keeps
   `max_steps=1` and accumulates the extra groups inside the single optimizer step (which keeps
   A003's "at most one optimizer step" literally satisfied). If the pinned TRL 1.8.0 will not
   accumulate across generation chunks within one step, the permitted fallback is up to 24
   optimizer steps of the existing 2-groups-per-step geometry. The receipt records
   `optimizer_steps` and `rollout_groups` so the auditor can see which path ran, and A004 narrows
   A003's `preflight_seed_reuse_policy` sentence to: *"a seam-verification preflight is at most 24
   optimizer steps over at most 48 rollout groups with heldout_n=8"* — a strict subset of one
   confirmatory unit.
4. **Seeds.** Seam-verification preflights reuse the seeds already burned for that cell
   (211/223/227) under A003's existing reuse policy. No additional preregistered seed is consumed.
5. **Telemetry quarantine (new prohibition, tightening).** Because the stopping rule terminates on
   a `mixed` group, `mixed_fraction`, `all_correct_fraction`, `all_wrong_fraction`,
   `charged_generated_tokens` and `two_sample_false_homogeneity` from a `seam_verification`
   receipt are stopping-rule-biased and are barred from every descriptive, planning, blinded-
   reassessment and manuscript use — including as a prior for the G=2 false-homogeneity rate. The
   gate reads exactly one bit from them: `mixed_fraction > 0`.
6. **No relaxation of the gate.** `require_live_mixed_reward_optimizer_update_per_cell` stays
   `true`, per cell, live on GPU. A004 explicitly rejects the cheaper alternatives of (a) accepting
   the observed `gsm8k/grpo_g8` seam as covering the other cells, and (b) substituting a unit test
   for a live observation. Both would weaken a preregistered gate; widening the observation window
   does not.

**`invariants`:** `tasks_unchanged`, `arms_unchanged`, `reward_semantics_unchanged`,
`decoder_unchanged`, `completion_cap_unchanged`, `heldout_rows_unchanged`,
`primary_estimands_unchanged`, `margins_unchanged`, `seed_cap_unchanged`,
`confirmatory_training_steps_unchanged`, `confirmatory_batch_geometry_unchanged` — all `true`.

**`timing`:** `confirmatory_rows_completed_before_amendment: 0`,
`confirmatory_outcomes_inspected: false`, `preflight_is_scientific_evidence: false`.

**`bindings`:** `prior_protocol_sha256` = sha of `preregistration.json` immediately before A004;
`superseded_remote_preflight_sha256: "dafa185f35b76f7db5ef3adcfdb71eddc81532039b2d333988ede14c5a5e375c"`
(see §4.3).

**Projected effect** (from §3.3, to be stated in the amendment as a planning figure, not a result):
per-attempt clearance 0.9912 / 0.9427 / 0.9995; joint 0.9339; expected groups actually generated
before the stop fires 10.6 / 16.3 / 6.7, i.e. roughly 27 / 39 / 53 completions versus 4 / 4 / 16
today — a 3–7× rollout increase per attempt against a 52 → ~3 reduction in A100 allocations.

### 4.2 Why this cannot bias the results

1. **Constitutionally non-evidence, and A003 already re-affirmed it.** Every receipt is stamped
   `evidence_class: preflight-not-evidence` / `evidence_tier: preflight_not_scientific_evidence`,
   and `validate_receipt` *fails closed* if that stamp is ever promoted. A003's own
   `evidence_boundary` and `preflight_seed_reuse_policy` state that no preflight outcome, reward
   trace or checkpoint enters any analysis and that confirmatory units are fresh independent runs
   from the frozen initial checkpoint. A004 changes the size of a throwaway, never the ledger.
2. **The gate reads one infrastructural bit.** `mixed_fraction > 0` asks whether the code path
   "heterogeneous group → non-degenerate advantages → optimizer update" executes on live hardware.
   That is a property of the stack, not of the data. Drawing more prompts changes what the gate can
   *see*; it cannot change what a confirmatory run *does*.
3. **The widened window is strictly smaller than the thing it gates.** A confirmatory unit is 30
   optimizer steps × 2 groups/step = **60 rollout groups**. A004 caps the seam preflight at 48
   groups and 24 steps. A preflight that is smaller than one unit of the study cannot be a
   backdoor version of the study. (At the fitted q the confirmatory math500 contrast run will hit a
   mixed group with probability 1 − 0.94217⁶⁰ = 0.972 on its own — the gate is checking something
   the real runs will exercise anyway.)
4. **No leakage channel exists at heldout_n = 8.** The only outcome-shaped number a preflight
   produces is held-out accuracy on 8 rows. Its resolution is 1/8 = 0.125, twelve and a half times
   the 0.01 non-inferiority margin, and its standard error at p=0.6 is 0.17. No adaptive design
   decision can be motivated by it, and A004 does not touch `heldout_n`. The blinded variance
   reassessment is explicitly restricted to confirmatory cells and is increase-only.
5. **Prospective by construction.** Zero of the 4 cells × 23 seeds × 2 endpoints of confirmatory
   evidence exist. The amendment is written before any outcome is inspected, changes no estimand,
   margin, task, arm, seed set, analysis or multiplicity rule, and is hash-bound into the protocol
   in the same commit.
6. **The direction of the change is disclosed.** Widening the window makes the gate more likely to
   pass, so it is a reduction in the gate's incidental strictness. That strictness was accidental —
   it came from a window too small to sample the event, not from a considered threshold — and its
   effect is a deadlock in which the seam the gate wants tested is the one thing that never gets
   tested. Increasing the number of live observations strengthens verification. The amendment
   should say this in those words rather than claim the change is neutral.
7. **The one real hazard is named and quarantined.** A stop-on-first-mixed rule biases the
   preflight's group-class fractions upward. §4.1(5) bars every one of those fields from all
   downstream use; the confirmatory `two_sample_false_homogeneity` telemetry that the paper does
   report comes from full 30-step runs with no stopping rule.

### 4.3 Implementation surface (for whoever drafts and binds A004)

Ordered, and one item is a trap:

1. **Receipt-compatibility trap — handle first.** `verify_preflight_matrix.SOURCE_BINDINGS` checks
   each receipt's `manifest.source_files` against the *live* protocol bindings:
   `require(source_files.get(filename) == bindings.get(binding_key), "... differs from the frozen
   source binding")`. All 7 existing receipts carry
   `remote_preflight.py = dafa185f35b76f7db5ef3adcfdb71eddc81532039b2d333988ede14c5a5e375c`.
   Editing `remote_preflight.py` and bumping `bindings.remote_preflight_sha256` makes every existing
   receipt *raise*, so the gate stops reporting "blocked on 3 seams" and instead errors out —
   destroying the already-observed `gsm8k/grpo_g8` seam and all four cells' infrastructure
   verification. Fix: A004 declares `superseded_remote_preflight_sha256`, `preregistration.bindings`
   carries it, and `SOURCE_BINDINGS` becomes a per-file *set* of accepted hashes. Justification to
   record: the pre-A004 window is a strict subset of the post-A004 window, so a receipt produced
   under it is conservative evidence for the same seam.
2. `remote_preflight.py` — window parameterisation, stop-on-first-seam callback,
   `preflight_class` / `rollout_groups_cap` / `optimizer_steps` in `run_config`. New hash into
   `bindings.remote_preflight_sha256`.
3. `verify_preflight_matrix.py` — hash-set source binding (item 1); optionally assert that a
   receipt declaring `preflight_class: "seam_verification"` respects the caps. New hash into
   `bindings.preflight_matrix_verifier_sha256`.
4. `verify_design.py:212-217` — `len(amendments) == 3` → `== 4`, and add the A004 identity /
   prospectivity / hash block alongside the A001–A003 blocks.
5. `preregistration.json` — append the A004 record to `amendments[]`; add
   `protocol_amendment_004_path` + `_sha256` and `superseded_remote_preflight_sha256` to `bindings`;
   refresh the four launcher hashes if their plumbing changes.
6. Launchers that pass the window through: `run_preflight.py`, `run_hf_jobs_preflight.py`,
   `run_kaggle_preflight.py`, `run_gcp_preflight.py` — each has a hash binding in
   `preregistration.json` that must move in the same commit.
7. Tests — `tests/test_next_submission_design.py:47-55` enumerates the amendment ids;
   `tests/test_next_submission_preflight.py:59,63,78,79` fixes `heldout_n: 8`, `rollout_groups: 2`,
   `max_steps: 1`. **Extend, do not weaken**: keep the existing `matrix_infrastructure` assertions
   exactly as they are and add `seam_verification` cases beside them.
8. `execution_authorization.json` — A003 introduced `protocol_canonical_sha256` self-binding, so
   the authorization receipt must be recomputed after `preregistration.json` changes, and
   `bindings.execution_authorization_sha256` / `authorization.receipt_sha256` updated with the
   canonicalisation rule A003 defines (those two values blanked before hashing).
9. Governance: single local commit, repo style, e.g. *"Widen the seam-verification preflight window
   before confirmatory runs (A004)"*. **No push, no submission.**

---

## 5. Non-evidential observation worth carrying forward

The fitted per-group mixed rates (0.06–0.15) imply that under the intervention the great majority of
groups will be homogeneous and skipped, i.e. large token savings and a strong test of the censoring
mechanism. This is exactly the regime A003's `probe_size_justification` anticipated. It is a
preflight-derived number and is **not evidence**: it must not enter the manuscript, the power
reassessment, or any prior. It is recorded here only so that a large observed cost saving in the
confirmatory results is not mistaken for a surprise.

---

## 6. Resolution — A004 implemented 2026-08-02

The amendment specified in §4 exists and is bound. `A004` was still unused, so no re-id was needed.

| §4.3 item | State |
|---|---|
| 1. Receipt-compat trap | Done. `SOURCE_BINDINGS` maps each file to a tuple of accepted binding keys; `remote_preflight.py` accepts `remote_preflight_sha256` **or** `superseded_remote_preflight_sha256`. Verified: with the superseded key removed, all 7 receipts raise `remote_preflight.py differs from every frozen source binding`. |
| 2. `remote_preflight.py` | Done. `--preflight-class`, per-arm `rollout_groups_cap`, `optimizer_steps`, and a `StopOnFirstAppliedUpdate` callback keyed on `updated_groups > 0` at `on_step_end`. |
| 3. `verify_preflight_matrix.py` | Done. Hash-set binding, plus per-class window assertions and `heldout_n == 8` for every receipt. |
| 4. `verify_design.py` | Done. Ledger 3 → 4 with the A004 identity / prospectivity / hash / window / gate-unchanged block and `EXPECTED_SEAM_ROLLOUT_GROUP_CAP`. |
| 5. `preregistration.json` | Done. A004 appended; `protocol_amendment_004_path/_sha256` and `superseded_remote_preflight_sha256` added; `remote_preflight`, `preflight_matrix_verifier` and `preflight_launcher` hashes refreshed. |
| 6. Other launchers | **Not touched, by design.** `run_hf_jobs_preflight.py` / `run_kaggle_preflight.py` / `run_gcp_preflight.py` gained no window plumbing, so they default to `matrix_infrastructure` and their bindings stay valid. Only `run_preflight.py` (Colab, the frozen-stack path the seam runs use) was changed and rebound. |
| 7. Tests | Done. Every `matrix_infrastructure` assertion kept verbatim; 9 seam cases added. 63 pass. |
| 8. `execution_authorization.json` | Done. Recomputed under A003's canonicalisation; `protocol_canonical_sha256` and both authorization hashes are a verified fixed point. |
| 9. Commit | **Not committed.** Awaiting the user. |

**Implementation path taken: the permitted fallback, not the preferred path.** TRL 1.8.0's
within-step accumulation across generation chunks could not be validated without a GPU, and the
preferred path would have required re-tuning `generation_batch_size` and re-proving peak memory. The
fallback keeps the generation geometry byte-identical to the window already exercised on A100-40GB.
At 2 groups per optimizer step the caps resolve to **24 optimizer steps (contrast)** and **12
(grpo_g8)** — both inside the 24-step budget and below the 60 rollout groups of one confirmatory
unit. A004 records `path_taken: permitted_fallback` and narrows A003's policy sentence accordingly.

**Correction to §4.1's projected effect.** That line quoted `0.9912 / 0.9427 / 0.9995; joint 0.9339`,
but `0.9995` is the M=48 figure and `grpo_g8` is capped at **24** groups by §4.1's own table. The
correct projections are:

| cell | per-attempt today | per-attempt under A004 | E[groups] before stop |
|---|---|---|---|
| gsm8k/contrast | 0.1790 | 0.9912 | 11.1 |
| math500/contrast | 0.1123 | 0.9427 | 16.8 |
| math500/grpo_g8 | 0.2712 | **0.9775** | 7.2 |
| **joint** | 0.0055 | **0.9134** | — |

The amendment carries the corrected figures and records the correction explicitly.

**Two collision hazards §4.3 did not name, both fixed:**

1. `evaluate_matrix` deduped on `(task, arm, seed)`. Because A004 reuses the burned seeds 211/223/227,
   a seam receipt would have collided with the matrix receipt for the same cell and raised
   `duplicate preflight identity`. The identity key now includes `preflight_class`.
2. `result_paths` keyed the receipt file on `{task}__{arm}__s{seed}`, so a seam run would have
   archived and replaced a committed matrix receipt. Seam receipts now land at
   `{task}__{arm}__s{seed}__seam_verification.json`.

**Gate state is unchanged and must stay that way until a seam run happens:** `blocked`,
`receipt_count: 7`, the same three missing seams. A004 changes what the gate can observe, never what
it requires.

**Unrelated blocker for whoever commits.** `run_gcp_preflight.py` has an uncommitted `--framework`
addition in the working tree, so `gcp_preflight_launcher_sha256` is drifted and
`verify_design.py` fails on it. That change is not A004's and was deliberately **not** rebound —
an amendment commit must not silently bless unrelated edits. With that one file set aside,
`verify_design.py` passes in full and all 31 design tests pass. Bind or revert it separately.
