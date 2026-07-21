# Execution notes

## Active objective

Produce one submission-ready flagship paper and artifact that turns the frozen
E1 audit into a general, testable result about variance starvation and, if the
evidence supports it, a ZVF-aware compute controller.

## Current gate

Stage 5 is blocked before science by fresh Colab A100 allocation failure. The
non-scientific A100 smoke remains accepted, but a clean supervisor relaunch
exhausted all six corpus jobs at the `colab new --gpu A100` step before any
remote runtime install, W&B run, or Hugging Face write. All three balanced
jobs and all three filtered jobs failed three guarded attempts each with the
same Colab CLI exception:
`TooManyAssignmentsError: ... accelerator=A100: Precondition Failed`. The
preflight receipt still stands; the corpus and 24 scientific unit layers remain
locked until A100 assignment succeeds again.

## Evidence checked

- `zvf-program/audit/results/audit.json` is `COMPLETE` with eight paired seeds.
- Final verdicts are DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO `INCONCLUSIVE`.
- The canonical R08 source and generated inventory copy contained stale
  pre-final deltas and a contribution item describing three missing units.
- `inventory_papers.py` correctly regenerates the inventory copy from the
  canonical source; the defect therefore belonged at the canonical source.
- `verify_colab_e1_campaign.py` reports 40/40 local and remote units with zero
  errors after the paper fix.
- The rebuilt R08 PDF is eight pages and renders the final aggregate values.
- The frozen flagship protocol has SHA-256
  `68237294a2cce2a0cddb5a1a413a5f954434aca1a9d1a44d806f0145c2a5171e`.
- The canonical CPU reference currently passes 15/15 focused tests, including
  fully masked completions, asymmetric DAPO clipping, flattened gradients, and
  fail-closed validation of unknown arms/policies.
- The preregistration schema/protocol suite passes 6/6 tests.
- An isolated CPU import confirms the flagship stack resolves to TRL 1.2.0,
  Transformers 5.5.4, and the external `trl/trainer/grpo_trainer.py` wheel
  rather than repository-local code.
- Exact source inspection shows native TRL uses a `1e-4` reward-standardization
  epsilon and arm-specific loss reductions (`grpo`, `dr_grpo`, and `dapo` do
  not share one denominator). These semantics must be surfaced by the adapter
  verdicts rather than normalized away.
- The exact external `verl==0.3.0.post1` distribution resolves on an isolated
  Python 3.11 CPU environment with `torch==2.4.0` and
  `transformers==4.45.2`; `verl.__file__` points into that environment rather
  than the repository-local `verl/` wrapper.
- Native verl exposes `compute_grpo_outcome_advantage` (default epsilon
  `1e-6`) and `compute_policy_loss`, whose PPO loss is one masked token mean.
  This gives a directly callable secondary-stack kernel for S1 and also makes
  reduction differences testable instead of inferred from a launcher.
- The native TRL differential now reports explicit verdicts. Current targeted
  fixtures are `MATERIAL_DIFFERENCE`; native AERO remains unsupported.
- The external verl differential reports native GRPO as
  `MATERIAL_DIFFERENCE` and DAPO/GSPO/DrGRPO/AERO as `NOT_TESTED`. A reproduced
  tensor-scalar group index raises `KeyError`; frozen Python string group IDs
  exercise the working native path.
- Intended TRL and verl integration paths now pass the canonical differential
  for GRPO, DAPO, GSPO, DrGRPO, and AERO, including selected-completion masks.
  The matrix now includes all-wrong, all-correct, graded, reward-translated,
  lower/upper clipping, unequal-length, zero-mask, and AERO-posterior cases.
  Unresolved noisy, missing, or delayed rewards fail closed to `recheck` before
  objective evaluation. TRL/reference focused suite: 29/29. External
  verl/reference focused suite: 28/28. Ruff passes the complete S1 directory.
- The fail-closed implementation freeze is `S1_PASS` with zero errors, 14
  intended cases per stack, 36 controller cases, identical fixture digest
  `c35916cf7db0b6c7ff6d0e35925a165b304fc78ff3d63845b9a853ca8af8ae9b`,
  and explicit native verdict vectors. It also binds adapter, reference,
  fixture, test, amendment, receipt, and combiner source hashes.
- The first theory witness now constructs two regimes with the same all-failure
  primary observation but opposite optimal actions: retry under a clean-hard
  verifier and recheck under a silently broken verifier. The executable default
  has positive outcome-only minimax regret `0.2589033831`.
- An adversarial pass rejected the initial zero-regret probe claim because it
  failed to charge the calibration probe in the clean regime. The corrected
  perfect-probe bound is the explicit probe cost `0.03`; with 10% probe error
  the bound is `0.093`. Theory tests pass 5/5 and Ruff is clean.
- The witness now matches silent runtime telemetry as well as primary rewards
  and exposes strict action-reversal margins. The reversal survives all 81
  combinations in a bounded neighborhood of the default costs/probability;
  theory tests pass 6/6 after this robustness check.
- The adversarial novelty audit retains the corrected action-reversal result as
  supporting theory only. False-negative verifier diagnosis and dynamic
  secondary verification are already occupied by adjacent work, so the
  flagship claim remains cross-stack semantic conformance and its causal
  training consequences.
- `pilot_preregistration.json` freezes a 24-unit screening matrix: four semantic
  conditions by two regimes by seeds 11/23/37, with disjoint five-seed
  confirmation and A100-only execution. The design is `ready_to_run` with
  `authorization.gpu=true`, explicitly limited to the staged smoke/corpus/unit
  dependency graph; confirmation remains unauthorized.
- The isolated pilot control plane expands all 24 units deterministically,
  atomically emits a content-addressed manifest and one plan per unit, binds
  the protocol/S1/theory/planner/replay sources, and gives local, W&B,
  Hugging Face, and Colab identities a namespace disjoint from frozen E1.
- Every generated scientific plan remains `dry_run_only`, has
  `allocation.allowed=false`, and contains no allocation command; the local
  launcher owns staged allocation and enforces the DAG. The numeric execution
  contract is complete with no readiness blocker. Pilot protocol SHA-256 is
  `5a0bbd25e2bdf2a6e8948ea649afd2825197cf7dafda1885ac2e33e9925a00d7`.
- The replay contract makes causal pairing executable without condition-specific
  generation: one content-addressed corpus per regime/seed is replayed in the
  same order by all four conditions. The balanced control keeps all eight rows
  with equal active optimization lengths; the positive regime deterministically
  selects six of eight rows with maximal population length CV and fails closed
  below 0.35. Rejected generations remain charged. Replay source SHA-256 is
  `21867edf878c4b379fa9895b9e378c47be0b5174abb316ac54ff06e5aefc4e2a`.
- Dataset commits, full-split hashes, and the PCG64 seed-specific 100-row orders
  are frozen for GSM8K and MATH-500. The exact resolvable runtime is Python
  3.11/3.12, TRL 1.2.0, Transformers 5.5.4, Torch 2.7.1, Datasets 4.8.4,
  PEFT 0.19.1, Hugging Face Hub 1.11.0, Accelerate 1.13.0, W&B 0.21.0, and
  NumPy 2.2.6.
- The training objective is pinned to DAPO: canonical selected-row advantages
  and per-completion reduction versus TRL's all-row `sample_std+1e-4` and
  global-active-token reduction. Exact differentials match both the canonical
  S1 reference and pinned TRL 1.2.0 loss kernel.
- One content-addressed replay corpus per regime/seed is conservatively charged
  to all four conditions. The fixed ceiling is 819,200 generated tokens for
  exactly 100 groups. Balanced groups optimize eight equal active lengths;
  filtered groups generate 16 candidates, choose the lexicographically first
  maximum-CV six-row subset, add two inactive rows, and require CV at least
  0.35.
- The remote executor now implements shared-corpus construction, model-facing
  replay updates, intended/native/ablation gradient receipts on every step,
  phase-separated FLOP profiling at steps 1/20/40/60/80/100, held-out evidence
  at 0/20/40/60/80/100, exact adapter/optimizer/scheduler/RNG/cursor resume, and
  W&B/Hugging Face final records.
- Every checkpoint carries and hashes all evaluation JSONL accumulated so far;
  resume restores those files along with model and RNG state. The independent
  verifier re-downloads the private corpus and unit commits, verifies all five
  checkpoint trees, recomputes all 128-row scores and hashes, checks local
  source fingerprints, and requires finished W&B runs.
- The preregistered screening evaluator uses seed-level units only. It enforces
  the 20/100 positive-control mechanism rule, 95/100 negative-control and
  epsilon equivalence, reduction-only 80% attribution, sign-consistent paired
  AUC effect, balanced equivalence, final non-inferiority, and matched corpus
  charges. The five-seed power gate uses 100,000 PCG64 draws and frozen
  Student-t critical values at Holm alpha 0.0125; a failed gate stops expansion.
- The launch DAG contains one non-scientific A100 smoke, six dependent corpus
  jobs, and 24 dependent units, with a tested three-session supervisor. Remote
  identities include both protocol and source-bundle hashes. The authorized
  launch manifest has 31 jobs and SHA-256
  `a53a5ee14ca5d4ba6f3e7a36895428f7b8a7dc75788e5acb750eb12274ca0b69`;
  only `preflight__a100_stack_smoke` has no dependency.
- Screening manifest SHA-256 is
  `0c83f53ee771a0967b7c812145f5d64b2c0366b9c6dd91b4e68fc00da21b2341`;
  its source-binding fingerprint is
  `f869ac52a56106d4b1d1d2429ac82514ba75eb7d1f0d302cedfa450712c641ae`.
- The exact resolved stack passes 88/88 pilot tests; the pilot preregistration
  suite passes 11/11; Ruff is clean. The smoke acceptance path now parses one
  exact receipt and fail-closes on package, Python, A100, FLOP, token, gradient,
  fingerprint, or finiteness mismatch. A read-only Colab check reports no
  active sessions and no pilot supervisor/launcher process is running.
- Guarded smoke attempt 2 is preserved at
  `zvf-program/flagship/pilot/launch/failures/preflight-attempt-2-mixed-numpy.log`
  with SHA-256
  `3f960552a0585a13435a52a77ceb3069167aef2d12bd00221a31609488f42739`.
  It proves exact package versions and A100 allocation before the mixed-NumPy
  traceback. Attempt 3 adds an explicit NumPy reinstall but repeats the same
  traceback; its log SHA-256 is
  `ca469a2e9169a2d67c522d622bf3df8b344817757d6914f46cc5870bb3f7f180`.
  Neither produced an acceptance receipt or scientific tracking artifact.
- Attempts 4-10 each isolated and fixed one defect, with logs preserved under
  `zvf-program/flagship/pilot/launch/failures/`: Colab base-image torchvision
  ABI breakage (`886dbb2f...`, fixed by uninstalling the torch-ABI-coupled
  family and failing closed if any remains importable), base-image torchao
  0.10.0 vs PEFT's >=0.16 requirement (`0f592aa1...`, same purge), a transient
  10s `restart-kernel` HTTP read timeout (`00bbb4a3...`, fixed with bounded
  retries limited to idempotent control-plane subcommands), Transformers v5
  `apply_chat_template` returning `BatchEncoding` (`fb447c2e...`, fixed by
  unwrapping `input_ids`), Transformers v5 removing the `generator` generate
  kwarg (`9f6dcc21...`, fixed with scoped `fork_rng` + `torch.manual_seed`,
  preserving the preregistered seed formula), missing
  `CUBLAS_WORKSPACE_CONFIG` under deterministic algorithms
  (`ba314410...`, fixed by setting `:4096:8` in the bootstrap before any
  remote work), and a torch 2.7.1 CUDA SDPA kernel rejecting the padded 4D
  causal mask under deterministic mode (`bcd40669...`, fixed by pinning the
  deterministic math attention backend for the training forward).
- Smoke attempt 11 passed on NVIDIA A100-SXM4-40GB. The verified result is
  `zvf-program/flagship/pilot/launch/results/preflight__a100_stack_smoke.json`
  (SHA-256 `2124a17fda7e13ff78110d5aaddd74cf2d7ef8fc9ab927da51426217ee74c91e`);
  the acceptance log SHA-256 is
  `1e49afebda61d784fddd9cba0a1725038574d08e68f3809294397dd91c52db83`.
  Receipt evidence: protocol SHA-256
  `5a0bbd25e2bdf2a6e8948ea649afd2825197cf7dafda1885ac2e33e9925a00d7`, source
  archive `7ca4f79b89b7df38108c6a54a868fca7dc97ae8a290781035d5f0cc0994b6345`,
  source bindings `616eef4030d55b2fe817317e0342229bccfd227b91459af2f272f0cd4790a019`,
  group fingerprint
  `106d5c11df5480608a4b381aab630b60c1f0142d37ddc3fe880f978603214508`, 8 active
  rows, 4096 active tokens, 3768 charged generated tokens, one `intended_full`
  update, gradient cosine 1.0022 (float rounding), relative L2 0.0094, and
  phase FLOPs 16.81e12 policy forward / 17.52e12 per backward. All Colab
  sessions were released; no W&B run or HF artifact was written, as required
  for the non-scientific preflight.
- Regenerated after the smoke fixes: screening manifest SHA-256
  `fbccbdca1611d2331d7713ef1dc660921a03aaf941ac89ca0510c69d364764c9`; launch
  manifest (31 jobs, `ready_to_run`) SHA-256
  `c42cf6bafe50a0db639e7fe4e68b88ed71ace40a517910196e83c7f36c9bf835`.
- After the deterministic-attention corpus fixes, the local gate still passes:
  pilot tests are 88/88 and Ruff is clean. A fresh manifest reset re-seeded
  only the accepted preflight against supervisor manifest fingerprint
  `ffd319336206294ad759d85631ac2b4985bdb819f34263df51b29f889d20a199`.
- The fresh corpus relaunch failed before any scientific artifact. The six
  preserved corpus logs now show the same `colab new --gpu A100` rejection:
  balanced `s11` SHA-256 `b377c8c1bc9957500505cfcafc123a8b934bab1100a2950c94a42710b82bb39e`,
  `s23` `47244948044ffaa197fcff8cf203e10bee4b1dfdf5a9fd35efbc40d211ddc099`,
  `s37` `d832e1d9bd9be1c5066798855bdb10dfeb4097dd1ac1a1fa2de3f5f6d24b3e4c`,
  filtered `s11` `d33cd275f10859e91b2edf1787ae73dfb29b0c7303ecc6011f6c767e1c14c1c3`,
  `s23` `8dcded0eff525679ec584c7ff7cc2cbd7a73fe4a8c8f362c7167c1c01d3b884c`,
  `s37` `cbfd0207ce9cdb86e1c09b696fe295625e61d46cac99216e14f67de787291919`.
  Each failed with `TooManyAssignmentsError: Failed to issue request POST
  https://colab.research.google.com/tun/m/assign?...&variant=GPU&accelerator=A100:
  Precondition Failed`; each subsequent `colab stop` reported the session name
  was not found, confirming no VM was actually allocated.
- Final supervisor state after that relaunch: preflight is `accepted`; all six
  corpus jobs are `failed_infrastructure` after `attempts=3`; all downstream
  unit jobs remain `pending`. `colab status` reports no active sessions. The
  stale unnamed rows from `colab sessions` are therefore treated as a CLI
  listing artifact rather than live allocatable sessions.

## Open items

- [x] Generate the LaTeX result macros from the frozen aggregate.
- [x] Rebuild R08 and regenerate the paper inventory.
- [x] Run audit, campaign, paper, lint, and compile checks.
- [x] Freeze the preregistered mechanism/controller protocol and compute gate.
- [x] Implement and test the canonical objective/controller reference.
- [x] Add immutable fixture digest, per-row/group mapping, completion masks,
  action ontology, source hashes, and runtime provenance receipts.
- [x] Implement exact native and intended TRL 1.2.0 and external verl
  0.3.0.post1 CPU adapters with source/version provenance.
- [x] Expand S1 fixtures across prespecified reward, clipping, normalization,
  missing/delayed/noisy observation, controller-action, and invariant cases.
- [x] Emit and verify the combined native/intended stack receipt.
- [x] Run the two-stack differential gate and write the implementation-freeze manifest.
- [x] Construct the observationally matched regimes and prove or falsify the
  identifiability/action-reversal theorem under realistic restrictions.
- [x] Freeze the 24-unit conformance pilot and generate a fail-closed dry-run
  matrix with isolated identities and immutable source bindings.
- [x] Implement deterministic balanced and filtered replay contracts with
  charged rejected-generation accounting.
- [x] Freeze the remaining numeric execution contract and clear every readiness
  blocker except the deliberate status/authorization transition.
- [x] Implement and test the remote replay trainer, gradient receipts,
  token/FLOP ledger, exact checkpoint resume, and local/W&B/HF verifier.
- [x] Freeze and test the screening go/kill and confirmatory power evaluators.
- [x] Transition the protocol to `ready_to_run` with authorization limited to
  the staged smoke/corpus/unit dependency graph.
- [x] Run and independently accept the non-scientific A100 smoke; stop before
  corpus generation if it fails.
- [ ] On smoke success, run and independently accept six shared corpora and the
  24-unit three-seed screening matrix.

## Material decisions

- Result numbers will be generated by `aggregate_audit.py`; R08 will not carry
  a second hand-maintained numeric copy.
- Incomplete aggregates remove the generated LaTeX surface, causing the paper
  build to fail closed rather than silently rendering stale verdicts.
- New compute will not be allocated until the mechanism/controller pilot and
  expansion criterion are preregistered.
- Screening uses seeds 11/23/37; confirmatory analysis uses disjoint seeds
  53/71/89/107/131.
- TRL 1.2.0 is the primary stack and verl 0.3.0.post1 the secondary stack;
  Qwen3-1.7B screens before any Qwen3-8B confirmatory expansion.
- Conformance verdicts must distinguish `PASS`, `NUMERICAL_VARIATION`,
  `MATERIAL_DIFFERENCE`, and `NOT_TESTED`; an unavailable or wrong stack may
  never be reported as agreement.
- The historical E1 runtime lock (TRL 1.8.0) and the flagship conformance lock
  (TRL 1.2.0 / Transformers 5.5.4) are separate provenance surfaces. Neither
  may be rewritten to make their versions appear identical.
- Native-framework and intended-integration verdicts are separate surfaces.
  S1 passes only when the intended paths pass on both stacks; native material
  differences and unsupported objectives remain attached to the final receipt.

## Breakthrough-chase audit — 18 canonical artifacts (2026-07-21)

- Added `BREAKTHROUGH_CHASE_18_ARTIFACTS.md`, covering P01--P08, R01--R08,
  U01, and N01 under a single novelty standard that does not count venue
  derivatives as independent inventions or evidence.
- The strongest executed contribution is **evidence-carrying treatment
  survival**: the stack contract, machine-readable registry, two-stack
  differential harness, immutable campaign records, and independent verifier.
  This is a defensible experimental-systems/reproducibility contribution, not
  a new policy-gradient optimizer.
- The highest-upside algorithmic contribution is **root-level signal survival
  plus cause-aware routing**: PAM/GSR/EGM/root-ZUF instrumentation followed by
  TRIAGE-RL actions for solved, failed, critic-lagged, transport-starved, and
  invalid roots. Its instrumentation is implemented, but its matched-budget
  PPO/SAO controller result remains unexecuted.
- Reclassified the legacy reward-only `ZVFController` as the naive symmetric
  baseline. It must not stand in for full TRIAGE-RL because high ZVF aliases
  all-correct and all-wrong groups and therefore cannot select one symmetric
  action safely.
- Recommended consolidation: one flagship scientific paper around signal
  survival and TRIAGE-RL, backed by the treatment-verification architecture;
  one artifact/methodology companion around R08/S1; derivative venue variants
  become generated views rather than novelty claims.

## Frozen-E1 reconciliation and flagship gate audit (2026-07-21)

- Re-read the final aggregate, canonical R08 source, `PROGRAM_AUDIT.md`,
  `OBLIGATIONS.md`, and `PAPERS_README.md` before inspecting adjacent code.
- Found one stale R08 prose datum: repaired GRPO seed 11 is 325/500 (`0.650`),
  while the source/PDF still printed the pre-repair `0.646`. The generated
  aggregate and paired DAPO difference already used `0.650`.
- Corrected that value, advanced the locked-directory date to 2026-07-20, fixed
  the conclusion's duplicated “together they” construction, and rebuilt the
  eight-page PDF successfully with TeX Live. The rendered PDF now prints
  `0.650` and the unchanged frozen verdicts: DAPO `DISAPPEARS`; GSPO,
  Dr.GRPO, and AERO `INCONCLUSIVE`.
- Audited the flagship screening DAG. The non-scientific A100 smoke is
  `accepted`; all six corpus jobs are `failed_infrastructure` after three
  guarded launcher attempts; all 24 scientific units remain `pending`; and
  `launch/acceptance/` contains no corpus receipt.
- The latest post-reload corpus wave failed six times before VM creation with
  `TooManyAssignmentsError: ... variant=GPU&accelerator=A100: Precondition
  Failed`. Both `colab sessions` and `colab status` report no active sessions,
  and no local supervisor/launcher/remote-training process remains. No further
  allocation was attempted because the protocol requires stopping after the
  repeated infrastructure condition and forbids accelerator substitution.
- A first local pilot-suite invocation correctly failed the environment guards
  because the shell default was Python 3.14 / NumPy 2.4.3 without the frozen
  libraries. Re-running under Python 3.12 with the exact preregistered package
  pins passed all 88 pilot tests. The six preregistration tests, eleven pilot-
  preregistration tests, seven aggregate tests, deterministic aggregate
  regeneration, and R08 LaTeX build also pass; regenerated aggregate JSON and
  TeX macros are byte-identical to the frozen checked-in outputs.

## Flagship corpus capacity restoration (2026-07-21)

- After the user confirmed renewed Colab access, reset only
  `corpus__balanced_equal_length__s11`; the other five infrastructure-terminal
  corpus jobs and all 24 scientific units remained untouched.
- The guarded attempt obtained `NVIDIA A100-SXM4-40GB`, installed and verified
  the exact Python 3.12 / TRL 1.2.0 / Transformers 5.5.4 / Torch 2.7.1 stack,
  and started W&B run `ujryg527` under the frozen corpus identity.
- The first profiled group initially showed fresh W&B heartbeats but 0% sampled
  GPU utilization. It completed without intervention at 4,096 charged tokens;
  subsequent unprofiled groups advanced normally. This was profiler overhead,
  not a session loss, and no decoding/runtime field was changed.
- The same diagnostic pattern recurred at the frozen group-20 profiler boundary:
  the W&B heartbeat stayed current, the remote kernel remained compute-bound at
  100% CPU with 13.3 GiB of A100 memory resident, and buffered group 19 then
  group 20 committed without intervention. Group 20 closed at 80,081 cumulative
  charged tokens with balanced-regime selected-length CV exactly `0`.
- A read-only Hub inspection confirmed the expected private dataset repository
  `arvindcr4/tinker-rl-lab-flagship-pilot-corpus-balanced_equal_length-s11-5a0bbd25-e22eb646`
  at skeleton commit `4def08def53bdbba144cc043ec307dfe0bedfddf` with only
  `.gitattributes` and zero payload storage. That skeleton is not acceptance
  evidence; the immutable corpus payload and manifest are uploaded only after
  all 100 groups finish.
- The frozen group-40 FLOP profiler boundary also passed without intervention.
  W&B buffered group 39 while the kernel remained compute-bound at 100% CPU
  with 13,318 MiB of A100 memory resident, then committed group 40 at 160,423
  cumulative charged tokens with selected-length CV `0`. The run remained in
  state `running` with a current heartbeat after the profile closed.
- The same exact-source run subsequently crossed the halfway point at group 50:
  199,323 cumulative charged tokens, selected-length CV `0`, W&B state
  `running`, and no downstream scientific unit unlocked before corpus
  acceptance.
- During the frozen group-60 profile, two read-only `colab console` attachment
  attempts returned `Session 'fpcorp-bala-s11-e22e' appears to be lost
  (404/401). Cleaning up.` This was isolated to the raw-console service:
  `colab sessions` continued to list the A100, the long `colab exec` launcher
  remained attached, and W&B heartbeats continued to advance. No restart was
  performed and the event was not counted as a scientific attempt.
- Buffered group 59 then committed and the group-60 profiler closed cleanly;
  the run advanced to group 61 at 240,711 cumulative charged tokens with
  selected-length CV `0` and W&B still `running`.
- The post-profile segment reached group 70 at 277,237 cumulative charged
  tokens with selected-length CV `0`; the A100 session, W&B heartbeat, and
  single-corpus supervisor topology remained healthy.
- Attempt 1 later lost its Colab VM while entering the frozen group-80 profile.
  Its immutable W&B run `ujryg527` is `crashed`; the last complete history row
  is group 78 at 309,364 cumulative charged tokens with selected-length CV `0`.
  The raw console returned `Session 'fpcorp-bala-s11-e22e' appears to be lost
  (404/401). Cleaning up.`, the Colab file service no longer exposed `/content`,
  and explicit release returned `ColabRequestError: ... /unassign/...: Not
  Found`. The HF repository remained the zero-payload skeleton, so no corpus
  checkpoint or acceptance evidence survived. This is infrastructure loss, not
  a scientific attempt.
- The orphaned local `colab exec` child was terminated after remote loss was
  independently established. The supervisor recorded `launcher exited 1` and
  started guarded attempt 2 from the unchanged frozen source and identity on
  A100 endpoint `gpu-a100-s-kkb-usc1b2-qur61wu1fkvb`. Because the protocol has
  no partial-corpus checkpoint, the replay corpus must be regenerated in full.
  New W&B run `lwjtk9dk` is `running` with a fresh heartbeat; crashed run
  `ujryg527` remains preserved separately.
- Guarded attempt 2 cleared its first frozen profiler: group 1 committed at
  4,096 charged tokens with selected-length CV `0` while W&B `lwjtk9dk`
  remained `running`. This establishes that the replacement A100 is executing
  the corpus, not merely holding an allocation.
- Attempt 2 subsequently reached group 10 at 40,246 cumulative charged tokens
  with selected-length CV `0` and W&B still `running`.
- The persistent supervisor remains live. Scientific units may unlock only
  after this corpus finishes, uploads a private immutable HF dataset, and passes
  the independent corpus verifier; no other corpus block is eligible in this
  guarded single-slot restoration.
- Attempt 2 cleared the frozen group-20 profiler without incident and advanced
  to group 44 at 176,270 cumulative charged tokens with selected-length CV `0`;
  W&B `lwjtk9dk` is `running` with a current heartbeat (2026-07-21T08:36Z), the
  launcher PID 9922 and keep-alive remain attached to A100 endpoint
  `gpu-a100-s-kkb-usc1b2-qur61wu1fkvb`, and `colab sessions` lists exactly one
  A100 session `fpcorp-bala-s11-e22e`. Next frozen profiler boundary is group
  60; no intervention during profiler windows. The five sibling corpus blocks
  remain `failed_infrastructure` from the earlier entitlement outage and are
  not eligible until this seed's corpus is accepted; that state is recorded,
  not a scientific attempt.
- Attempt 2 cleared the frozen group-60 profiler and advanced to group 62 at
  244,764 cumulative charged tokens with selected-length CV `0`; W&B
  `lwjtk9dk` remains `running` with a current heartbeat (2026-07-21T09:00Z)
  and the launcher/session topology is unchanged. Next frozen profiler
  boundary is group 80, the point at which attempt 1 lost its VM.
- Attempt 2 matched attempt 1's final row exactly (group 78 at 309,364
  cumulative charged tokens, confirming deterministic replay), then cleared
  the frozen group-80 profiler — the boundary where attempt 1 lost its VM —
  and advanced to group 98 at 389,914 cumulative charged tokens with
  selected-length CV `0`; W&B `lwjtk9dk` remains `running` with a current
  heartbeat (2026-07-21T09:46Z). The run is now in the final group-100
  profiler window; the HF dataset remains the zero-payload skeleton until all
  100 groups complete, as designed.
- Attempt 2 then lost its Colab VM during the group-100 profiler window: its
  last immutable W&B row in `lwjtk9dk` is group 99 at 393,714 cumulative
  charged tokens with selected-length CV `0` (2026-07-21T09:37:01Z), after
  which the launcher exited 1 (~09:49Z) with the same VM-loss signature as
  attempt 1. The HF dataset remained the zero-payload skeleton, so no corpus
  checkpoint or acceptance evidence survived; this is infrastructure loss, not
  a scientific attempt. The supervisor recorded `launcher exited 1` and
  started guarded attempt 3 — the final guarded attempt under the
  three-attempt policy — from the unchanged frozen source and identity on new
  A100 endpoint `gpu-a100-s-kkb-ass1c2-26pkmr7nkmght` (launcher PID 16155).
  New W&B run `hge0xhav` is `running` (initialized 2026-07-21T09:52:36Z) with
  a current heartbeat; the corpus must again be regenerated in full. If
  attempt 3 fails with the same infrastructure condition, the campaign stops
  and reports exact evidence plus the smallest required decision.
- Attempt 3 cleared its first frozen profiler: group 1 committed at 4,096
  cumulative charged tokens with selected-length CV `0` (2026-07-21T10:04:27Z),
  identical to attempts 1 and 2, further confirming deterministic replay, and
  W&B `hge0xhav` remains `running` with a current heartbeat (10:05Z). This
  establishes the replacement A100 is executing the corpus, not merely holding
  an allocation.
- Attempt 3 subsequently reached group 10 at 40,246 cumulative charged tokens
  with selected-length CV `0` (2026-07-21T10:10:35Z) — identical to attempt
  2's group-10 count — with W&B `hge0xhav` `running` and a current heartbeat
  (10:11Z); launcher PID 16155 and the A100 session remain healthy.
- Attempt 3 cleared the group-20 frozen profiler — the next boundary after
  the group-1 and group-10 milestones — with group 20 committed at 80,081
  cumulative charged tokens and selected-length CV `0` (2026-07-21T10:29:14Z),
  and advanced to group 23 at 92,369 cumulative charged tokens with W&B
  `hge0xhav` `running` and a current heartbeat (10:32Z). Launcher PID 16155
  and A100 session `fpcorp-bala-s11-e22e` remain healthy; the HF dataset
  remains the zero-payload skeleton until all 100 groups complete, as
  designed.
