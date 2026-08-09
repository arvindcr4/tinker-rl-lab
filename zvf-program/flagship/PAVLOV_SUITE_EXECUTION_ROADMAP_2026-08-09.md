# Pavlov suite execution roadmap — 2026-08-09

Status: **repository audit plus post-audit integration update**. This file
records what the live checkout can actually execute, what is only a fail-closed
adapter/preflight scaffold, and what is only contract metadata. The integrated
xLAM component runner has made tracked Tinker/W&B/HF calls; that component is
outside the exact 26-suite registry and does not make any E-suite runnable.

## Bottom line

The exact 26-suite inventory is not runnable in this checkout today:

| Readiness class | Count | Meaning |
|---|---:|---|
| **Runnable now** | 0 | Exact suite data, verifier, adapter, and required runtime are all present. |
| **Fail-closed scaffold** | 18 | A validator, split/receipt adapter, or preflight runner exists, but exact data, runtime, verifier, access, or immutable receipts are still missing. |
| **Metadata-only** | 8 | The exact suite is present in the contract without an integrated suite-specific scaffold. |
| **Absent** | 0 at the contract layer | Every requested ID is recorded in `pavlovs_domain_contract.json`. |

The original audit found only **T3 `browsergym_train`** and **T4
`bfcl_train`** bridges. The consolidation subsequently added fail-closed
scaffolds for T6, T7, T11 and E1-E8/E10-E14. These additions validate pins,
splits, budget and receipt boundaries; they do not supply private datasets,
native managed environments, or production verifiers. T3 remains the required
first post-xLAM priority because it is
stateful and artifact/side-effect producing. Its existing MiniWoB smoke is an
integration receipt with zero reward and zero trainable datums, not learning
evidence. T4 has a BFCLv4-style reward scaffold and a synthetic simulator, but
its real-data path explicitly returns a placeholder failure.

Eight exact suites remain metadata-only, and none of the 26 has the complete
exact data/runtime/native-verifier bundle needed for a runnable result. Existing
GSM8K, MATH, HumanEval, synthetic tool-use, WebArena, and historical SWE-agent
files are **not** substitutes for the named Pavlov IDs. The contract remains
`BLOCKED` for a main-track campaign until model revisions, licenses, dataset
revisions, and disjoint task-ID hashes are frozen.

## Evidence vocabulary and current blockers

- **P — protocol:** `zvf-program/flagship/PAVLOVS_LIST_TASK_CONTRACT.md`,
  `zvf-program/flagship/pavlovs_domain_contract.json`, and the non-launching
  manifest preview. P is prospective design, not an experiment.
- **X — observed xLAM slice:**
  `autoresearch/orchestrator-260809-0922/base_eval_100.json`; 100 strict
  single-turn function-call examples, mean reward `0.070`, 7 perfect calls.
  xLAM is not T4 and not any E-suite.
- **R — rejected smoke:**
  `autoresearch/orchestrator-260809-0922/rejected_untracked_smoke.json`; four
  steps, no trained checkpoint, no held-out result. It is provenance only.
- **M — portfolio evidence:** immutable result receipts for the exact contract
  suites. No M receipt exists for any T1–T12 or E1–E14 suite.

The primary `.venv` now contains the Tinker/W&B stack used by the tracked xLAM
component runs. Benchmark-specific BrowserGym, managed browser/enterprise
worlds, private task sets, containers, and native verifier assets remain suite
gates. The generic Tinker loop now initializes W&B online before Tinker and
requires initial/periodic/final HF sampler exports; publication failure marks
the run failed instead of silently accepting it.

## Exact suite matrix

The per-row gap descriptions below preserve the original discovery audit. Since
that audit, fail-closed validators and preflight runners were integrated for 18
suites. A row's older `metadata-only` wording should therefore be read as an
upstream data/runtime/verifier gap, not proof that no local validator file now
exists. An exact suite is not runnable merely because a validator or related
benchmark exists.

### Training suites T1–T12

| ID | Exact contract suite | Class | Live source path(s), command, and gap |
|---|---|---|---|
| T1 | `openreward_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json` (`rg -n '"openreward_train"' ...`). No OpenReward task files, loader, state/artifact verifier, or pinned train split. No command. |
| T2 | `swe_gym_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. `platform_modal/scripts/berkeley/sweagent_passk_aci.py` is a historical Pass@K/ACI analysis, not a SWE-Gym loader or trainer. No exact repository/issue split or patch verifier. No command. |
| T3 | `browsergym_train` | **adapter-only** | Adapter: `platform_tinker/atropos/tinker_atropos/environments/browsergym_tinker.py`; configs: `platform_tinker/atropos/configs/browsergym_miniwob_qwen_8b_smoke.yaml`, `browsergym_webarena_qwen_8b_smoke.yaml`; setup: `platform_hybrid/experiments/setup_browsergym_miniwob.sh`; ReAct bridge: `platform_hybrid/experiments/webarena/react_eval.py`. Future command (blocked; do not run): `cd platform_tinker/atropos && ./run_experiment_generic.sh browsergym_tinker configs/browsergym_miniwob_qwen_8b_smoke.yaml`. BrowserGym packages, Atropos/Tinker runtime, W&B, model server, and MiniWoB assets are absent. Native `env.step` reward exists, but no Pavlov split/license/task-hash receipt wrapper. |
| T4 | `bfcl_train` | **adapter-only** | Scaffold: `platform_hybrid/experiments/bfclv4_tool_use.py` with `reward_sparse`, `reward_dense`, and `SimulatedBFCLv4`; local synthetic preflight only: `PYTHONDONTWRITEBYTECODE=1 python3 platform_hybrid/experiments/bfclv4_tool_use.py --dry-run --seeds 1 --steps 1 --out /dev/null`. The real `--dataset` branch prints “Real evaluation requires model inference pipeline” and exits 1. No BFCL dataset snapshot or native category verifier is present. `tool_use_tinker.py` uses Glaive, not BFCL. |
| T5 | `scienceworld_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No ScienceWorld environment checkout, task split, trajectory runner, or native state verifier. No command. |
| T6 | `unix_ctf_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No Unix-CTF procedural environment, shell sandbox receipt, or hidden-test verifier. No command. |
| T7 | `agentdojo_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No AgentDojo task/world loader, tool-state verifier, or safety evaluator. No command. |
| T8 | `rtlcoder_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No RTLCoder dataset snapshot, HDL compiler/simulator adapter, or artifact verifier. No command. |
| T9 | `crafter_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No Crafter environment, procedural seed manifest, or achievement/state verifier. No command. |
| T10 | `visual_app_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No AppBench/Visual-App prompt set, GUI executor, or artifact/side-effect verifier. BrowserGym is not AppBench. No command. |
| T11 | `api_bank_rlvr_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. `platform_tinker/atropos/tinker_atropos/environments/tool_use_tinker.py` loads `glaiveai/glaive-function-calling-v2`, not API-Bank RLVR, and has no API-Bank revision or state verifier. No command. |
| T12 | `openr1_math_train` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. `platform_tinker/atropos/tinker_atropos/environments/math_tinker.py` and `math_curriculum_tinker.py` load Hendrycks MATH; GSM8K is calibration-only. No OpenR1-Math-220k revision or decontamination manifest. No command. |

### Primary held-out suites E1–E14

| ID | Exact contract suite | Class | Live source path(s), command, and gap |
|---|---|---|---|
| E1 | `swe_bench_pro_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. The local SWE-agent Pass@K script is retrospective analysis of existing tables, not the Scale SWE-bench Pro evaluator. No patch execution, public split receipt, or task hash. No command. |
| E2 | `frontier_swe_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No `frontier-swe` evaluator, repository fixture runner, or patch/artifact verifier. No command. |
| E3 | `sdab_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No SDAB task/world adapter, enterprise state reset, or held-out verifier. No command. |
| E4 | `banker_toolbench_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. The Glaive tool-use environment and BFCL scaffold are different datasets and cannot stand in for held-out BankerToolBench. No finance API sandbox or task-ID manifest. No command. |
| E5 | `apex_agents_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No APEX worlds, long-horizon state reset, or native evaluator. No command. |
| E6 | `webbench_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. BrowserGym/WebArena code is a non-equivalent browser bridge; it does not implement Halluminate WebBench or its held-out verifier. No command. |
| E7 | `binaryaudit_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No BinaryAudit binaries, execution sandbox, security verdict parser, or hidden-test receipt. No command. |
| E8 | `lifescibench_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No Life-Sci-Bench task package, tool/state evaluator, or held-out split. No command. |
| E9 | `mle_bench_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No competition bundle, containerized ML task runner, or artifact verifier. No command. |
| E10 | `agentharm_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No Inspect/AgentHarm evaluator, policy grader, or held-out safety receipt. No command. |
| E11 | `verilog_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No NVLabs Verilog-Eval prompts, HDL compile/simulate command, or artifact verifier. RTLCoder metadata does not supply E11. No command. |
| E12 | `appbench_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No AppBench held-out prompt set, GUI executor, or artifact/side-effect verifier. BrowserGym is not AppBench. No command. |
| E13 | `openreward_games_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. No OpenReward held-out game environments, procedural seed separation, or game-state verifier. No command. |
| E14 | `frontiermath_eval` | metadata-only | Registry only at `zvf-program/flagship/pavlovs_domain_contract.json`. FrontierMath is private held-out evaluation; local GSM8K/MATH-500 files cannot substitute. No command. |

### Exact inventory conclusion

No row is `runnable now`. The adapter-only rows are useful for implementation
work and zero-cost syntax checks, not for Pavlov claims. The 24 metadata-only
rows require a new adapter/data/verifier package and a pinned receipt contract
before they can move to adapter-only or runnable. No row is promoted because a
related benchmark exists.

## Zero-cost preflight (run before any paid or network action)

These commands are read-only or print-only. They do not install dependencies,
clone benchmark repositories, call Tinker, start model servers, or publish
anything.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 zvf-program/flagship/pavlovs_domain_contract.py --json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program \
  python3 -m unittest \
    flagship.test_pavlovs_domain_contract \
    flagship.test_build_pavlovs_campaign_manifest \
    flagship.test_eval_pavlov_xlam -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program python3 - <<'PY'
import ast, json
from pathlib import Path

root = Path('.')
contract = json.loads((root / 'zvf-program/flagship/pavlovs_domain_contract.json').read_text())
expected = [
    'openreward_train', 'swe_gym_train', 'browsergym_train', 'bfcl_train',
    'scienceworld_train', 'unix_ctf_train', 'agentdojo_train', 'rtlcoder_train',
    'crafter_train', 'visual_app_train', 'api_bank_rlvr_train', 'openr1_math_train',
    'swe_bench_pro_eval', 'frontier_swe_eval', 'sdab_eval',
    'banker_toolbench_eval', 'apex_agents_eval', 'webbench_eval',
    'binaryaudit_eval', 'lifescibench_eval', 'mle_bench_eval', 'agentharm_eval',
    'verilog_eval', 'appbench_eval', 'openreward_games_eval', 'frontiermath_eval',
]
assert all(suite_id in contract['suite_registry'] for suite_id in expected)
for path in [
    'platform_tinker/atropos/tinker_atropos/environments/browsergym_tinker.py',
    'platform_hybrid/experiments/bfclv4_tool_use.py',
    'platform_hybrid/experiments/webarena/react_eval.py',
    'zvf-program/flagship/eval_pavlov_xlam.py',
]:
    ast.parse((root / path).read_text(), filename=path)
print('PASS exact_ids=26 adapter_syntax=ok')
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program python3 - <<'PY'
from flagship.build_pavlovs_campaign_manifest import build_manifest, load_contract
manifest = build_manifest(load_contract())
assert manifest['status'] == 'BLOCKED'
assert manifest['launches_any_job'] is False
print('PASS manifest=BLOCKED launches_any_job=False')
PY

git diff --check
git status --short --branch
```

The live import probe used during this audit reported missing
`browsergym`, `tinker`, `atropos`, `wandb`, `transformers`, and `trl`, while
`gymnasium`, `playwright`, `datasets`, and `openai` imported successfully. Do
not treat an import-only pass for `playwright` or `gymnasium` as a BrowserGym
pass.

Do **not** run these during this audit: `setup_browsergym_miniwob.sh` (it
clones/fetches and resets an external repository), the Atropos launcher, any
`TINKER_API_KEY` command, any W&B/HF publish command, or any paid model
sampling/training.

## Receipt, W&B, and HF requirements

### C0 receipt bundle

Before a result can be called M, attach the contract-required tuple from
`zvf-program/flagship/pavlovs_domain_contract.json`:

1. immutable dataset/benchmark revision and license sign-off;
2. disjoint train/evaluation split manifest and task-ID hashes;
3. container/environment digest;
4. model revision and adapter revision;
5. seed, stack, run ID, budget, and verifier identity;
6. per-step loss/reward/telemetry and final metrics;
7. held-out result rows with domain, horizon, reward/verifier, artifact-vs-state,
   and seen-vs-unseen-family slices; and
8. durable W&B/HF or equivalent artifact links.

C0 is necessary but is not itself an M result. The current X receipt has
prompt/target/response hashes and token cost, but not the full C0 tuple. R
cannot satisfy C0.

### W&B fields

The required run namespace is the existing Pavlov project
`tinker-rl-lab-pavlov`. Every paid run must initialize W&B **online before the
first Tinker call** and log at least:

- config: `campaign`, `suite_id`, `suite_role`, `model_id`, `model_revision`,
  `adapter_revision`, `dataset_revision`, `split_manifest_hash`,
  `container_digest`, `seed`, `horizon`, `group_size`, `batch_size`, `lr`,
  `max_prompt_tokens`, `max_response_tokens`, `reward_type`, `verifier_type`,
  `stateful`, `artifact_or_side_effect`, `git_sha`, `budget_cap_usd`;
- generic training: `train/loss`, `train/reward`, `train/step`, zero-loss and
  zero-reward counts, sampler path, Tinker run ID, periodic checkpoint path;
- generic held-out: `test/reward` plus per-suite/domain/horizon rows;
- BrowserGym: `train/browser_success_rate`,
  `train/browser_reward_mean`, `train/browser_action_count_mean`, and the
  corresponding `eval/*` keys, with `env_id`, actions, terminal state, error,
  and artifact/state hashes in samples; and
- BFCL-style diagnostics: sparse/dense reward, `n_correct`, `n_total`, and ZVF
  only when the exact BFCL data/verifier has been pinned. Synthetic simulator
  output must be labeled preflight, never a suite result.

The current generic loop's intended keys are visible at
`platform_tinker/tinkerrl/grpo.py:410-412` and `:467`; its W&B initializer is
currently empty at `:228-230`, so the tracking gate is not satisfied.

### HF checkpoint expectations

For every paid training seed, export and retain:

- the initial/base sampler path, every configured periodic sampler, and the
  final sampler path;
- the matching Tinker `run_id`/`tinker://` path and local checkpoint JSON;
- an HF repository/commit containing the sampler adapter, tokenizer/model
  revision metadata, C0 receipt, and a manifest linking each checkpoint to
  the W&B run; and
- explicit `adapter_revision` and `model_revision` fields. The existing helper
  `platform_tinker/tinkerrl/grpo.py:_publish_checkpoint` only runs when
  `HF_PUSH=1`; its default is private when `HF_PUSH_PRIVATE` is unset.

The existing `rejected_untracked_smoke.json` step-0 HF adapter is untrained and
must remain provenance-only. It is not an acceptable periodic or final
checkpoint for a result claim.

## Tinker cost envelope

The local budget record is
`zvf-program/flagship/pavlov_tinker_budget.json`:

- prefill: `$0.54 / 10^6` tokens;
- sampled output: `$1.335 / 10^6` tokens;
- training: `$1.177 / 10^6` tokens;
- authorized budget model: `Qwen/Qwen3.6-35B-A3B`;
- nominal operational cap: `$16.50`, safety reserve `$1.50`;
- current conservative remaining estimate after the recorded X and rejected
  smoke allowance: `$16.359961445`.

The checked-in BrowserGym smoke configs target `Qwen/Qwen3-8B`; they are
integration scaffolds, not an authorized Pavlov model configuration. A future
T3 run must pin the authorized model and its revision explicitly.

Use the conservative estimate

```text
cost = (0.54 * prefill_tokens
        + 1.335 * sampled_tokens
        + 1.177 * trained_tokens) / 1_000_000
```

and charge every sequence at its configured maximum until an actual receipt
proves lower usage. The existing xLAM evaluator's 100-example ceiling is
`maximum_eval_cost(100, 1200, 128) = $0.081888`; the recorded X receipt is
`$0.040038555`.

The following are planning envelopes, not observed charges. They assume a
1,024-token prompt, 128-token action/response, and an eight-action stateful
episode; a 2x allowance covers retries/serialization overhead:

| Future step | Token-model estimate | Conservative envelope | Evidence status |
|---|---:|---:|---|
| xLAM 100-example control | `$0.040` observed / `$0.0819` ceiling | `$0.10` | X only; already observed. |
| T3 MiniWoB 10-step integration smoke, 10 updates, B=2, G=2 | `$0.28` | `≤$0.60` | Adapter/integration only until C0 and non-degenerate rewards. |
| T3 short stateful/artifact pilot, 25 updates, B=4, G=4 | `$2.80` | `≤$5.60` | First post-xLAM multi-domain pilot; not E6 and not main-track portfolio evidence. |
| T3 held-out episode check, 100 episodes × 8 actions | `$0.58` | `≤$1.16` | Local BrowserGym check only; cannot be labeled `webbench_eval`. |
| T4 one-turn exact-data evaluation, 500 examples | `$0.41` ceiling | `≤$0.82` | Only after BFCL data/verifier is real and pinned; current scaffold is synthetic. |
| Safety reserve | — | `$1.50` | Must remain unspent. |

The illustrative BrowserGym-first pilot plus one BFCL evaluation stays below
the current `$16.359961445` working estimate, but only after the missing
adapters/data/verifiers and tracking gate are implemented. It does **not** buy
the 24 missing suites or justify a main-track claim. Do not reallocate the
reserved `$1.50` to a third training arm.

## Shortest evidence sequence within the cap

This is the shortest sequence to a **first multi-domain stateful pilot**, not
to a compliant all-26 main-track result:

1. **Zero-cost gate:** run the validator, 14 local contract/manifest/xLAM
   tests, exact-ID/source AST preflight, and import probe. Confirm no paid
   launch and no external mutation.
2. **Tracking gate:** repair and locally test online W&B initialization and
   periodic/final HF export. Freeze C0 fields before any Tinker call. The
   current `_start_wandb` defect means this step is mandatory.
3. **Keep X separate:** retain the existing 100-example xLAM base receipt as
   the strict function-call baseline. A corrected re-run is optional and costs
   at most `$0.0819`; it is not T4.
4. **First post-xLAM priority — T3:** install/pin BrowserGym + MiniWoB in an
   authorized future environment, run the one-step MiniWoB adapter smoke, and
   reject it if rewards remain all zero or no trainable datums are emitted.
   Then run one short stateful/artifact T3 pilot within the `≤$5.60` envelope,
   logging native `env_id`, actions, terminal state, artifact/side-effect
   hashes, C0, W&B, and HF checkpoints.
5. **Held-out boundary:** do not call the local MiniWoB check E6. Build or
   obtain an exact `webbench_eval` adapter and held-out verifier before using
   any browser result as a primary evaluation.
6. **Cheapest second family:** after exact BFCL data and native verification
   are pinned, use the T4 scaffold only as an exact-data adapter and spend at
   most the `$0.82` one-turn envelope. Do not use Glaive or the synthetic
   simulator as BFCL evidence.
7. **Stop/triage:** if the tracking, C0, T3, or exact E6 gates fail, stop paid
   work. The other 22 suites remain metadata-only; do not spend the cap on
   unrelated GSM8K/MATH/HumanEval/WebArena substitutes.

The shortest sequence for a **main-track Pavlov claim** is therefore not
available from the current checkout: all 12 training suites and all 14 primary
held-outs need exact adapters, data revisions, native verifiers, split hashes,
and C0 result receipts. The T3 pilot is a prioritized multi-domain signal, not
permission to call the portfolio complete.

## Audit commands and non-claims

No paid call, network mutation, commit, or PR was performed for this roadmap.
The roadmap does not claim that any company is production-ready, that xLAM is
post-training evidence, that the rejected smoke trained a model, or that a
related local benchmark substitutes for an exact T/E suite.
