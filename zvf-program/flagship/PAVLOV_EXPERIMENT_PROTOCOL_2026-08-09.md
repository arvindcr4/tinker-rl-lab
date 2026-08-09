# Professor-ready protocol: the $18 Tinker Pavlov portfolio

Date: 2026-08-09
Protocol status: **active component campaign; no portfolio result is complete**
Paid-launch gate: **AUTHORIZED for Tinker within the recorded cap; exact-suite launches remain fail-closed on missing data/runtime/verifier receipts**
Provider: Tinker only
Primary model candidate: `Qwen/Qwen3.6-35B-A3B`

This protocol answers a portfolio question, not an xLAM question:

> Can one base candidate, frozen after the required preflight receipts, improve a
> domain-balanced portfolio of stateful and artifact-producing agent tasks,
> without a material regression in
> any listed domain, while staying inside the authorized Tinker budget?

xLAM strict tool use is the first inexpensive systems-and-signal smoke. It is a
hyperparameter and receipt gate; it is **not** the objective, does not satisfy a
portfolio suite, and cannot support a company-readiness claim.

## 1. Evidence boundary: what is known and what is planned

The following is the complete recorded evidence available when this protocol is
written; the table states which parts are admissible for which claims:

| Item | Evidence | Status and use |
| --- | --- | --- |
| Observed base xLAM evaluation | 7 perfect calls out of 100; mean strict reward `0.070`; estimated cost `$0.040038555` | Observed seed-809 slice only. It is **not yet frozen or portfolio-wide evidence**: model/dataset revisions, split-manifest, task-ID, container/runtime, and verifier receipts must be present before it can serve as the frozen reference. The two-sided 95% Wilson interval for 7/100 is approximately 3.4%–13.7% conditional on these observed rows. |
| Four-step training attempt | [`rejected_untracked_smoke.json`](../../autoresearch/orchestrator-260809-0922/rejected_untracked_smoke.json) | **Inadmissible provenance only.** It began before online W&B initialization was verified, stopped after four completed steps, and produced no trained/final checkpoint. Its reward/loss trace is never pooled, ranked, or used as an outcome. |
| Meeting decision record | [`PROFESSOR_MEETING_2026-08-09.md`](./PROFESSOR_MEETING_2026-08-09.md) | Context and operating invariant; not a replacement for a run receipt. |
| Local base receipt | [`base_eval_100.json`](../../autoresearch/orchestrator-260809-0922/base_eval_100.json) | Source for the observed 7/100, mean `0.070`, token counts, prompt/target hashes, and `$0.040038555` estimate; not a frozen reference until the Phase-0 receipt bundle is complete. |
| Budget authorization | [`pavlov_tinker_budget.json`](./pavlov_tinker_budget.json) | Prospective ceiling: `$18.00` hard maximum, `$16.50` operational cap, `$1.50` billing-lag reserve. |
| Contract/budget metadata | [`pavlovs_domain_contract.json`](./pavlovs_domain_contract.json) and [`PAVLOVS_LIST_TASK_CONTRACT.md`](./PAVLOVS_LIST_TASK_CONTRACT.md) | The contract is now `authorized`, with `paid_jobs_may_launch: true`, an `$18.00` hard maximum, a `$16.50` operational cap, and a `$1.50` reserve. Authorization does not waive suite-specific source, license, split, runtime, verifier, W&B, Tinker, or HF receipt gates. |
| Tracked xLAM 2e-5 arm | W&B run `dgn034bt`; Tinker run `a2888b9a-89a9-5899-9660-b190ca6dde90:train:0`; initial, step-5, step-10, and final public HF checkpoint receipts | **Completed component-only training receipt.** Ten training steps, reward trace `[0, 0, .25, 0, 0, .125, 0, .5, 0, 0]`; this is not held-out or portfolio evidence. |
| Tracked xLAM 1e-5 and 4e-5 arms | W&B runs `gcwywjcr` and `95527xa1` | Both reached ten training steps, but mandatory HF lifecycle publication was incomplete: 1e-5 failed at step-10 export and 4e-5 failed at the separate final export. They remain failed receipt states until repaired; their training rewards are not promoted. |

One component-only arm has a complete training/checkpoint receipt. No exact
T1-T12/E1-E14 score, held-out portfolio result, or improvement claim exists yet.
The protocol below must not be read as a portfolio result.

## 2. Inputs to freeze before paid work

### 2.1 Model, data, and verifier

The base model is **not yet frozen for claim purposes**. Before any new paid work,
record an immutable base-model revision, tokenizer revision, container/environment
digest, and adapter revision in the run manifest. The candidate, seed, and xLAM
smoke settings are:

| Field | Required/prospective value |
| --- | --- |
| Base candidate | `Qwen/Qwen3.6-35B-A3B` |
| xLAM data | `Salesforce/xlam-function-calling-60k`, deterministic seed `809`; 3,000 train and 500 proposed evaluation-pool examples as in the meeting brief |
| xLAM selection slice `S` | Re-create and seal a 100-row manifest from a pinned dataset revision before training. The prompt/target hashes in `base_eval_100.json` identify the observed slice but, without model/dataset revision and split/task receipts, do not establish frozen status. |
| xLAM final holdout `H_xlam` | A disjoint 100-row manifest from the same pinned evaluation pool, sealed and hash-recorded before training; no score or prompt may be inspected until the adapter is locked and holdout receipts are complete |
| LoRA and optimizer | LoRA rank `32`; batch/group `2/4`; all optimizer, truncation, and sampler settings are identical across arms |
| Prompt/response limits | 1,200 / 128 tokens |
| xLAM evaluation sampler | The current evaluator settings to pin and hash: temperature `0.1`, top-p `0.95`, one sample per row, maximum 128 response tokens; the service/sampling seed is fixed when supported and otherwise recorded in the receipt |
| Strict verifier | The repository's `StrictToolCallReward`, whose implementation must be hashed before paid work; a perfect call is a verifier score exactly `1.0`. No parser, target, split, or verifier change is permitted mid-campaign. |

The primary portfolio unit is a **seed within model × environment-family ×
stack**, as required by the task contract. Rows are never pooled across suites in
a way that lets the larger code family dominate the result.

### 2.2 The portfolio objective

The xLAM smoke is outside the contract registry. After it passes, the selected
learning rate is used for a single, predeclared portfolio curriculum containing all 12
contract training suites. The curriculum must satisfy the contract's composition
constraints in every paid portfolio batch: at least six domain families, at most
5% math, at least 60% stateful episodes, and at least 50% episodes ending in a
native artifact or externally visible state change. The schedule is inverse-
frequency weighted by domain, not by company count.

Training and primary evaluation tasks, repositories, environment seeds, and hidden
tests are disjoint. Before portfolio launch, each suite must have receipts for its
dataset revision, license, task-ID hashes, split-manifest hash, container/runtime
digest, model revision, and adapter revision. The word **holdout** is reserved for a
suite/manifest only after its split, task-ID, license, runtime, and decontamination
receipts are complete. If a receipt is missing, the portfolio phase is blocked
rather than silently shortened.

## 3. Required suite coverage (contract registry)

The machine-readable source of truth is [`pavlovs_domain_contract.json`](./pavlovs_domain_contract.json).
The following tables are a checklist, not evidence that any suite has run or that
its split is already a held-out split.

### 3.1 All 12 training suites

| Suite ID | Domain tags in the contract | Stateful / artifact contract |
| --- | --- | --- |
| `openreward_train` | multi-domain, tool use, browser, science, ML, games, long horizon | yes / yes |
| `swe_gym_train` | code, long horizon | yes / yes |
| `browsergym_train` | browser, computer use, enterprise, tool use, long horizon | yes / yes |
| `bfcl_train` | tool use, code | no / no |
| `scienceworld_train` | science, long horizon, tool use | yes / yes |
| `unix_ctf_train` | security, ML, code, tool use | yes / yes |
| `agentdojo_train` | alignment, security, tool use, enterprise | yes / yes |
| `rtlcoder_train` | chip design, code | no / yes |
| `crafter_train` | games, long horizon | yes / yes |
| `visual_app_train` | design, computer use, code | yes / yes |
| `api_bank_rlvr_train` | finance, enterprise, tool use, long horizon | yes / yes |
| `openr1_math_train` | math | no / no |

`math_control_train` (GSM8K) is not one of these 12 training suites. GSM8K is a
calibration-only control and is never promoted into primary training, selection, or
evidence.

### 3.2 All 14 primary evaluation suites (holdout status pending receipts)

| Suite ID | Domain tags in the contract |
| --- | --- |
| `swe_bench_pro_eval` | code, long horizon |
| `frontier_swe_eval` | code, ML, long horizon |
| `sdab_eval` | code, ML, long horizon, enterprise |
| `banker_toolbench_eval` | finance, enterprise, tool use, long horizon |
| `apex_agents_eval` | multi-domain, finance, enterprise, long horizon, tool use |
| `webbench_eval` | browser, computer use, enterprise |
| `binaryaudit_eval` | security, code, long horizon |
| `lifescibench_eval` | science, long horizon, tool use |
| `mle_bench_eval` | ML, code, long horizon |
| `agentharm_eval` | alignment, security, tool use |
| `verilog_eval` | chip design, code |
| `appbench_eval` | design, computer use, code |
| `openreward_games_eval` | games, long horizon, tool use |
| `frontiermath_eval` | math |

`math500_eval` is secondary only; `gsm8k_calibration` is calibration-only. Neither
can replace `frontiermath_eval` or any other primary suite.

### 3.3 Domain-slice map

Every one of the contract's 16 domain tags must have a non-empty training path and
a non-empty primary-evaluation path. Overlapping tags are intentional; aggregation
is domain-balanced rather than company- or row-count-weighted.

The exact machine-readable domain keys are `alignment`, `browser`, `chip_design`,
`code`, `computer_use`, `design`, `enterprise`, `finance`, `games`, `long_horizon`,
`math`, `ml`, `multi_domain`, `science`, `security`, and `tool_use`.

Coverage is also checked per company, not only by this global union: for every
company entry in the 53-company snapshot, each listed domain in `companies[].domains`
(`required_domains` in the protocol's shorthand) must occur
in the union of the 12 training-suite tags **and independently** in the union of
the 14 primary-evaluation-suite tags. The preflight emits this company-by-domain
matrix and fails if any required domain is missing; a non-empty global intersection
is insufficient.

| Domain | Training path(s) | Primary evaluation path(s) (holdout status pending receipts) |
| --- | --- | --- |
| alignment | `agentdojo_train` | `agentharm_eval` |
| browser | `openreward_train`, `browsergym_train` | `webbench_eval` |
| chip design | `rtlcoder_train` | `verilog_eval` |
| code | `swe_gym_train`, `bfcl_train`, `unix_ctf_train`, `rtlcoder_train`, `visual_app_train` | `swe_bench_pro_eval`, `frontier_swe_eval`, `sdab_eval`, `binaryaudit_eval`, `mle_bench_eval`, `verilog_eval`, `appbench_eval` |
| computer use | `browsergym_train`, `visual_app_train` | `webbench_eval`, `appbench_eval` |
| design | `visual_app_train` | `appbench_eval` |
| enterprise | `browsergym_train`, `agentdojo_train`, `api_bank_rlvr_train` | `sdab_eval`, `banker_toolbench_eval`, `apex_agents_eval`, `webbench_eval` |
| finance | `api_bank_rlvr_train` | `banker_toolbench_eval`, `apex_agents_eval` |
| games | `openreward_train`, `crafter_train` | `openreward_games_eval` |
| long horizon | `openreward_train`, `swe_gym_train`, `browsergym_train`, `scienceworld_train`, `crafter_train`, `api_bank_rlvr_train` | `swe_bench_pro_eval`, `frontier_swe_eval`, `sdab_eval`, `banker_toolbench_eval`, `apex_agents_eval`, `binaryaudit_eval`, `lifescibench_eval`, `mle_bench_eval`, `openreward_games_eval` |
| math | `openr1_math_train` | `frontiermath_eval` |
| ML | `openreward_train`, `unix_ctf_train` | `frontier_swe_eval`, `sdab_eval`, `mle_bench_eval` |
| multi-domain | `openreward_train` | `apex_agents_eval` |
| science | `openreward_train`, `scienceworld_train` | `lifescibench_eval` |
| security | `unix_ctf_train`, `agentdojo_train` | `binaryaudit_eval`, `agentharm_eval` |
| tool use | `openreward_train`, `browsergym_train`, `bfcl_train`, `scienceworld_train`, `unix_ctf_train`, `agentdojo_train`, `api_bank_rlvr_train` | `banker_toolbench_eval`, `apex_agents_eval`, `lifescibench_eval`, `agentharm_eval`, `openreward_games_eval` |

The required secondary slices are also reported for every primary suite: horizon,
reward type, verifier type, artifact-versus-stateful task, and seen-versus-unseen
environment family. A portfolio aggregate cannot erase a missing slice.

## 4. Phased execution and successive halving

All phase decisions are made from immutable receipts. A phase may be prospective or
blocked; neither state is a scientific result.

### Phase 0 — freeze and tracking preflight (no paid work)

1. Reconcile the contract metadata before using the budget authorization: the JSON
   currently declares `status: draft-awaiting-budget-cap`, while its budget gate
   permits Tinker. Until a receipt resolves this contradiction, **no paid phase may
   launch**; a validator `PASS` alone is not an authorization.
2. Pin all model and dataset revisions, licenses, split/task hashes, evaluator/
   verifier version, container/runtime digest, seed, and budget ledger. The current
   7/100 receipt may be reused as `K_base` only if these hashes prove equivalence;
   otherwise produce a new base reference on the sealed slice.
3. Seal `S` and `H_xlam`; seal the 14-suite primary evaluation manifests before
   training. The final manifests are write-once, not mounted in training, and are
   not called held out until their split/decontamination receipts are complete.
4. Start an online W&B run and verify that the Tinker run ID is recorded before
   the first paid step. Every step records loss, reward, configuration, seed,
   split identity, and cumulative cost.
5. Publish every initial, periodic, and final sampler checkpoint to Hugging Face.
   Each checkpoint must have a unique repository/revision URL and receipt. Set
   visibility to public or private according to available quota and data/license
   sensitivity; public artifacts must contain no prompts, secrets, or restricted
   data. A W&B initialization failure, failed or missing export/URL receipt, or
   projected-cap failure stops the run fail-closed.

### Phase 1 — tracked xLAM smoke (systems-and-signal gate)

Run the fixed 10-step smoke at learning rate `2e-5`, with the settings in §2.1.
Evaluate on `S` only; do not read `H_xlam` or any contract primary-evaluation task.

The smoke passes only if all 10 steps and required checkpoints are tracked, no loss
or reward is NaN, at least one logged training completion has a positive verifier
reward, and the 100-row selection score is strictly greater than the Phase-0 frozen
base reference `K_base`. The observed 7/100 is the planning comparator only; it is
not a frozen reference until the revision/split/task receipts above match. Otherwise
stop before arm selection. This is a budget-preserving operational decision, not
evidence that the broader portfolio cannot improve.

### Phase 2 — three short learning-rate arms with successive halving

The only arm difference is learning rate:

* `arm_1e-5`: `1e-5`
* `arm_2e-5`: `2e-5`
* `arm_4e-5`: `4e-5`

Use the same seed, data order, verifier, batch/group, and checkpoint schedule. The
resource schedule is fixed before launch:

1. **Round 1:** all three arms train to 50 total steps; evaluate `S`.
2. **Round 2:** retain the top two eligible arms and continue each to 100 total
   steps; evaluate `S` again.
3. **Winner extension:** retain one arm and continue it to 200 total steps. Only
   this arm may enter the 12-suite portfolio curriculum.

At every round, rank by the following predeclared key (all comparisons are on the
same 100 rows):

1. larger perfect-call count `K = Σ_i 1[score_i = 1.0]`;
2. larger mean strict reward `R̄ = (1/100)Σ_i score_i`, compared without rounding;
3. lower conservative cost estimate from the fixed token-price ledger;
4. lower learning rate (`1e-5` before `2e-5` before `4e-5`);
5. lexical arm ID, as a final deterministic tie-break.

No arm advances from Round 2 to the winner extension unless an eligible arm beats
the Phase-0 frozen base on sealed `S`. Selection-slice intervals are descriptive
only: selecting on their data makes them unsuitable for a confirmatory improvement
claim.

### Phase 3 — winner extension on the complete training portfolio

Continue only the selected arm. The portfolio extension must contain all 12
training suites in §3.1 and satisfy the contract's batch composition constraints.
The exact task IDs, per-suite counts, curriculum order, and number of portfolio
steps are frozen in the launch manifest; the manifest is rejected if its cost
projection cannot fit the Phase 3 cap. No learning-rate, data-mixture, or verifier
retuning is allowed after arm selection.

If the cap permits only an xLAM continuation but not a contract-complete 12-suite
tranche, stop and mark the portfolio phase **BLOCKED**. A partial subset is not a
portfolio result and may not be relabeled as one.

### Phase 4 — untouched final evaluation manifests

After the adapter, evaluator, manifests, and holdout-designation receipts are
locked, evaluate the Phase-0 frozen base reference and the selected adapter side by
side on:

* `H_xlam` (100 disjoint xLAM rows), as a diagnostic tool-use comparison; and
* the predeclared primary-evaluation tasks for **all 14** suites in §3.2, designated
  held out only after the split, task-ID, license, runtime, and decontamination
  receipts are complete, with target 100 episodes per suite where the suite supplies
  that many after decontamination.

If any primary suite cannot meet its declared minimum, report that suite and the
portfolio as incomplete; do not impute, substitute `MATH-500`, or claim all-domain
improvement. Final evaluation scores are never used to choose an arm, alter a prompt,
or rerun a failed suite with a more favorable split.

## 5. Metrics and uncertainty

### 5.1 xLAM selection and final diagnostic

For each 100-row xLAM slice, define `I_i = 1` exactly when the hashed Phase-0
`StrictToolCallReward` returns `1.0`; otherwise `I_i = 0`.

* Primary selection metric: `p̂ = K/100`, the perfect-call rate.
* Tie-break metric: mean strict reward `R̄`; report its paired row-level difference.
* Cost metric: conservative prompt-plus-sample estimate from the fixed pricing
  formula; it is a tie-break and budget guard, never a quality claim.

Report two-sided 95% Wilson score intervals for every 100-row proportion. For the
final `H_xlam` comparison, report the paired difference
`Δ = (b - c)/100`, where `b` is base-fail/adapter-success and `c` is
base-success/adapter-fail, together with an exact two-sided McNemar test and a
95% Newcombe hybrid score interval for the paired risk difference. Report a paired
bootstrap 95% interval (10,000 resamples, seed recorded) for the mean strict-reward
difference. These intervals quantify uncertainty over the sealed rows; they do not
estimate between-seed variance or generalization to every tool-use task. The current
observed 7/100 slice is not treated as sealed unless the Phase-0 receipts match.

### 5.2 Portfolio metrics

For each primary suite `s`, report the contract metrics separately: task success,
rubric score, state/artifact integrity, safety-violation rate, tokens per success,
tool calls per success, and wall time per success. Normalize within environment
family before comparing suites.

The single preregistered portfolio quality statistic is the equal-domain macro
average of normalized task success:

`S_d = mean of suite scores whose contract tags include domain d`
`S_portfolio = mean of S_d over all 16 domains`.

Each suite has equal weight within a domain and each domain has equal weight in the
macro average. The reward vector and efficiency metrics remain visible; no weighted
aggregate may compensate for a safety, artifact, or domain failure.

For binary suite outcomes, report two-sided Wilson 95% intervals (and an exact
Clopper–Pearson sensitivity interval when a suite has fewer than 100 rows). For
normalized means and `S_portfolio`, use a paired, stratified bootstrap over rows within suite, with
the resampling seed and number of resamples recorded. The final global comparison
is one one-sided 95% interval for `Δ_portfolio = S_portfolio(adapter) -
S_portfolio(base)` on the untouched primary evaluation manifests after holdout
receipts are complete; arm-selection and smoke scores do not enter that test.

## 6. Portfolio gates and no-regression rules

All gates are conjunctive. Passing an average while failing a domain is a failure,
not a trade-off.

1. **Coverage gate:** every one of the 12 training and 14 primary evaluation suite
   IDs has a manifest, receipt, and non-empty required slices; all 16 domains are
   represented. The six-domain, math, stateful, and artifact composition limits
   are satisfied in every paid portfolio batch.
2. **Integrity gate:** all contamination, revision, license, task-hash,
   environment, W&B, and checkpoint receipts are complete. The rejected four-step
   run and any other untracked run are excluded.
3. **Global improvement gate:** the one-sided 95% interval for `Δ_portfolio` has
   a lower bound greater than zero. This is the only global improvement claim.
4. **Domain no-regression guard:** for every domain `d`, report `Δ_d` and its
   interval. A claim is permitted only when the point estimate is no worse than
   5 percentage points (`Δ_d ≥ -0.05`) and the lower 95% bound is above `-0.05`.
   This five-point tolerance is a predeclared operational margin, not an observed
   result; a suite-level regression is never hidden by the macro average. These
   intervals are safety guards, not 16 separate positive-discovery claims.
5. **Safety guard:** no domain may show a safety-violation increase above two
   percentage points, and the one-sided 95% upper bound for that increase must be
   at most `0.02`. Any material safety regression stops further spending and
   voids an improvement claim.

If a suite or domain is missing, underpowered, contaminated, or outside the cost
cap, the correct label is **incomplete/blocked**, not negative and not positive.

## 7. Budget allocation and fail-closed accounting

The hard user authorization is `$18.00`; the operational cap is `$16.50`; `$1.50`
is retained as a billing-lag reserve and is not a planned spend target. The phase
envelopes below are prospective maximums, not billing receipts:

| Phase | Maximum new Tinker usage |
| --- | ---: |
| Existing observed base receipt (not yet a frozen claim) | `$0.040038555` estimated, already recorded |
| Tracked 10-step xLAM smoke | `$0.50` |
| Halving Round 1: three 50-step arms | `$2.50` |
| Halving Round 2: two 100-step survivors | `$2.50` |
| Winner extension to 200 xLAM steps | `$3.00` |
| Contract-complete 12-suite portfolio tranche | `$5.00` |
| Final `H_xlam` plus 14-suite primary evaluations | `$2.50` |
| **Prospective new phases** | **`$16.00`** |
| **Base estimate plus prospective phases** | **`$16.040038555`** |

The remaining operational headroom is approximately `$0.459961445` before the
`$16.50` cap; the `$1.50` reserve remains untouched. Before each phase, recompute
the worst-case token projection with the pinned price ledger and reconcile live
provider usage, including any charge later attributed to the rejected run. If
observed, pending, or projected usage would cross `$16.50`, stop before launching
the next paid phase. `$18.00` is an absolute ceiling: no reserve draw, billing lag,
or partial result authorizes an overage.

The portfolio tranche is attempted only if its complete 12-suite manifest and the
14-suite primary-evaluation projection (holdout status pending receipts) fit these
envelopes. A cost cap cannot be met by
dropping domains or suites; that condition produces a blocked protocol receipt.

## 8. Stopping and reporting rules

Stop immediately on any of the following: failed W&B initialization, failed or
missing checkpoint export or URL receipt, split/verifier drift,
contamination, NaN/Inf loss, unsafe verifier behavior, an exhausted cap, or a
checkpoint that cannot be reconciled to its Tinker run ID. Resume only from a
verified checkpoint; never count the rejected four-step run as a completed arm.

At handoff, report four separate evidence classes:

1. completed receipts (currently only the observed seed-809 7/100 base evaluation);
2. operational/provenance receipts (currently the rejected four-step record);
3. prospective protocol gates and budget projections; and
4. admissible final portfolio results, only if all 12/14 suites and all gates pass.

The strongest permissible claim from an xLAM-only success is that the tracked
pipeline produced a tool-use signal on the sealed xLAM task family. The current
7/100 observation alone cannot support that claim. A full portfolio claim requires
the complete 12-suite training and 14-suite primary evaluation holdout receipts, the
global interval, every domain guard, and the safety guard. Even then, the result is
benchmark evidence for the model revision, seed, tasks, and environment recorded in
the receipts—not evidence that any of the 53 companies' private production bars are
met.
