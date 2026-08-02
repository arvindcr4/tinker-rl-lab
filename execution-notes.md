# Execution notes

> **2026-08-02 correction:** Any entry below that quotes DAPO as
> `DISAPPEARS` is superseded. Exact paired-t MDE80 is `0.01012`, above the
> `0.01` equivalence margin, and the executable Benjamini-Hochberg step rejects
> no difference. DAPO, GSPO, Dr.GRPO, and AERO are all `INCONCLUSIVE`. Current
> sources of truth are `zvf-program/audit/results/audit.json` and
> `zvf-program/audit/STATISTICAL_REANALYSIS.md`.

## Active objective

Produce one submission-ready flagship paper and artifact that turns the frozen
E1 audit into a general, testable result about variance starvation and, if the
evidence supports it, a ZVF-aware compute controller.

## Current gate

Stage 5 is fail-closed on a newly proven joint-zero-gradient protocol
contradiction. The first accepted r3 corpus contains 62/100 reward-degenerate
groups (59 all-wrong and 3 all-correct). Both first scientific units completed
the identical step-0 evaluation, then stopped before optimizer step 1 because
r3 requires every intended/native/selected gradient norm to be positive even
though equal rewards correctly produce zero advantages and zero gradients.
The preregistered gate simultaneously requires 100 receipts and at least 95
balanced-equivalence steps, so the current positive-norm/cosine-only receipt
contract cannot complete or score the frozen corpus. All Colab sessions are
released, the crash-looping launch agent is unloaded, and no replacement unit
may run until an explicit joint-zero representation/scoring amendment is
authorized. Balanced seed 23 has a separately verified group-20 prefix for
recovery if the amendment permits corpus reuse.

## Evidence checked

- `zvf-program/audit/results/audit.json` is `COMPLETE` with eight paired seeds.
- Final verdicts are DAPO, GSPO, Dr.GRPO, and AERO all `INCONCLUSIVE`
  (superseded 2026-08-02; the earlier DAPO `DISAPPEARS` used a
  normal-approximation MDE and skipped the preregistered multiplicity step —
  see `zvf-program/audit/STATISTICAL_REANALYSIS.md`).
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
- Evidence pass (2026-07-21T10:45Z): local flagship suite triage. A naive
  `.venv` pytest run reports 130 passed / 26 failed; every failure is
  environment-gating, not a code defect. (a) 24 `test_verl_adapter.py`
  cases fail closed with `VerlPinError: external verl is unavailable` —
  by design they require the frozen Python 3.11 environment
  (`verl==0.3.0.post1 torch==2.4.0 transformers==4.45.2`) run from outside
  the repository per `zvf-program/flagship/s1/README.md`; this Mac's `.venv`
  is Python 3.12. (b) `test_runtime_pins_are_exact_and_complete` fails on
  numpy `2.4.4` vs pinned `2.2.6` — local env drift, not source change.
  (c) `test_native_condition_matches_pinned_trl_dapo` fails closed with
  `TRLPinError` outside the pinned `uv run --with trl==1.2.0 --with
  transformers==5.5.4` env. Re-running the documented TRL harness command
  passes all TRL cases; the only errors are the same 24 fail-closed verl
  cases. Frozen receipts `trl_receipt.json`, `verl_receipt.json`, and
  `implementation_freeze.json` (`S1_PASS`, fixture digest
  `c35916cf…8ae9b`, 14 intended cases per stack, 36 controller cases)
  remain the authoritative conformance evidence from 2026-07-20. The verl
  py3.11 harness and full pinned-env re-run are queued for the pre-Done
  completion audit; no source change indicated.
- Attempt 3 cleared the group-40 frozen profiler: group 40 committed at
  160,423 cumulative charged tokens with selected-length CV `0`
  (2026-07-21T10:54:37Z), after a normal ~13-minute buffered profiler
  window, and advanced to group 43 at 172,174 cumulative charged tokens
  (10:56:41Z) with W&B `hge0xhav` `running` and a current heartbeat
  (10:57Z). Launcher PID 16155 and A100 session `fpcorp-bala-s11-e22e`
  remain healthy. A supervisor-terminal scrollback line (`corpus log must
  contain exactly one result line; found 0`) was inspected and confirmed
  to be residue from attempt 2's exit, not a new failure — the fail-closed
  result-line check in `launcher.py` fires only after the remote script
  exits, and attempt 3's launcher is still running. Next frozen profiler
  boundary: group 60; group 80 remains the boundary where attempts 1 and
  2 lost their VMs.
- Attempt 3 cleared the group-60 frozen profiler: buffered rows flushed
  after the normal profiler window, with group 66 committed at 260,853
  cumulative charged tokens and selected-length CV `0`
  (2026-07-21T11:24:12Z) and W&B `hge0xhav` `running` with a current
  heartbeat (11:24:56Z). Launcher PID 16155 and A100 session
  `fpcorp-bala-s11-e22e` remain healthy; the HF dataset remains the
  zero-payload skeleton until all 100 groups complete, as designed. Next
  frozen profiler boundary: group 80 — the boundary where attempts 1 and
  2 lost their VMs.
- Attempt 3 cleared the group-80 frozen profiler — the boundary where
  attempts 1 and 2 both lost their VMs — and advanced to group 90 at
  357,375 cumulative charged tokens with selected-length CV `0`
  (2026-07-21T11:52:39Z), W&B `hge0xhav` `running` with a current
  heartbeat (11:53:42Z). Launcher PID 16155 and A100 session
  `fpcorp-bala-s11-e22e` remain healthy. Ten groups remain plus the
  group-100 frozen profiler window before payload upload to the private
  HF dataset and the supervisor verifier receipt.
- HARD STOP (2026-07-21T12:15Z): attempt 3 — the final guarded attempt
  for `corpus__balanced_equal_length__s11` — lost its Colab VM during
  the group-100 frozen profiler window. W&B `hge0xhav` is `crashed`
  with last row group 99 at 393,714 cumulative charged tokens
  (2026-07-21T11:58:51Z), heartbeat frozen at 12:12:46Z; the A100
  session is gone (`colab sessions`: none), launcher PID 16155 is dead,
  and the supervisor recorded `failed_infrastructure` with
  `attempts=3`, `last_error="launcher exited 1"`. This is the third
  infrastructure loss for this corpus block and exhausts the
  three-attempt policy — no relaunch without an explicit user decision.
  Loss pattern across all three attempts: attempt 1 (`ujryg527`,
  created 05:34:46Z) died at the group-80 profiler boundary with last
  row group 78 at 309,364 tokens; attempt 2 (`lwjtk9dk`, created
  07:30:23Z) died during the group-100 profiler with last row group 99
  at 393,714 tokens; attempt 3 (`hge0xhav`, created 09:52:41Z) died at
  the identical point — group 99, 393,714 tokens — confirming
  deterministic replay and implicating the long group-100 profiler
  burst (~12–14 min of uninterrupted A100 compute beginning ~2h20m
  into the session) as the common loss window. Every run died at
  roughly 2h20m–2h25m of wall-clock session time, consistent with
  per-session runtime reclamation rather than credit exhaustion
  (credits were reloaded and tokens flowed until sudden death). The
  corpus carries no partial checkpoint by frozen design, so each loss
  forces full regeneration (~2h30m per attempt). All six corpus blocks
  are now `failed_infrastructure` at their attempt caps; the 24
  `fpilot__*` units remain gated. Awaiting user direction; see the
  report in-thread for the decision options.
- Evidence pass (2026-07-21T12:45Z), non-compute audit while the corpus
  decision is pending. (a) Strict frozen-E1 campaign verification
  re-run against current local + W&B + HF state: `COMPLETE`, 40/40
  locally validated and remotely verified units, zero errors. (b)
  Flagship suite invocation resolved: the checked-in tree needs
  `python -m pytest --import-mode=importlib -o
  consider_namespace_packages=true` from `zvf-program/flagship`
  (theory tests use relative imports in a namespace-package directory;
  `pilot/test_stack_differential.py` imports `s1` top-level). With that
  invocation: 139 passed / 26 failed / 11 skipped / 98 subtests passed
  — the identical failure set as the prior triage (24 fail-closed verl
  py3.11 cases, 1 numpy pin drift, 1 fail-closed TRL pin), all
  environment-gating, zero source regressions. (c) Aggregate
  `results/audit.json` remains `COMPLETE`; R08 canonical source and PDF
  present; failure documentation committed as `98f3c994`. The corpus
  compute blocker is unchanged and still awaits the user's decision.
- Evidence pass (2026-07-21T13:05Z), status/obligations doc sync. (a)
  `OBLIGATIONS.md` was stale: its 2026-07-21 narrative ended at attempt 1
  (`ujryg527`) and the E8 row still described that run as live. Appended
  the attempt-2 (`lwjtk9dk`) and attempt-3 (`hge0xhav`) VM-loss outcomes
  and refreshed the E8 row: all six corpora `failed_infrastructure` at
  their three-attempt caps, all 24 scientific units fail-closed, awaiting
  the user's decision. (b) `COLAB_EXECUTION_STATUS.md` had no flagship
  entry; added an E8 bullet recording the three corpus attempts, the
  identical group-99/393,714-token loss boundary for attempts 2 and 3,
  the ~2h20m reclamation pattern, and the pending decision. (c) Verified
  `zvf-program/flagship/preregistration.sha256` (`OK`) and re-read
  `s1/results/implementation_freeze.json`: `S1_PASS`, fixture digest
  `c35916cf…8ae9b`, 14 intended cases per stack, 36 controller cases;
  native verdicts TRL 4/5 `MATERIAL_DIFFERENCE` (1 `NOT_TESTED`), verl
  1/5 `MATERIAL_DIFFERENCE` (4 `NOT_TESTED`) — the central S1
  cross-stack conformance finding, already frozen. (d) `PROGRAM_AUDIT.md`
  (dated 2026-07-14) predates S1 completion; its R08 "objective-
  differential tests" gate is now closed by the S1 receipts, noted here
  for the pre-Done audit refresh rather than editing the dated audit
  snapshot. The corpus compute blocker is unchanged.

## Flagship corpus-resume amendment (2026-07-22)

- The user explicitly selected the intermediate-persistence route after the
  three version-1 corpus VMs were reclaimed. Added protocol version 2 and
  amendment `A1-corpus-intermediate-persistence`, binding the previous
  protocol SHA-256 `5a0bbd25...a00d7` and Git commit `296f0342...a84c6c`.
  The prior W&B runs `ujryg527`, `lwjtk9dk`, and `hge0xhav` remain immutable
  infrastructure-only records and are never pooled with version-2 units.
- Corpus generation now uploads one atomic private-Hub `resume/` prefix after
  groups 20, 40, 60, and 80. Each prefix contains every accepted group file,
  exact source hashes, cumulative token/profiler ledgers, W&B attempt ledger,
  and a content-addressed manifest. A retry restores only the greatest prefix
  whose protocol, model/dataset revisions, train order, runtime pins,
  accelerator, source rows, group/artifact hashes, token ledger, profiler
  coverage, and manifest fingerprint all match; otherwise it fails closed.
- Group generation remains deterministic by absolute group index, and the
  model, data, decoding, profiler points, 100-group horizon, token ceiling,
  treatments, estimands, and A100-only rule are unchanged. The new scheduler
  preserves version-1 control state, uses separate
  `plans-v2-corpus-resume-r1/` and `launch-v2-corpus-resume-r1/` directories, runs
  at most one corpus session at a time, and enforces three version-2 VM
  attempts per job.
- The first version-2 launch exposed a pre-scientific source-archive defect:
  `runtime_install.py` was bound by the runtime source manifest but missing
  from the uploaded archive. It created only a private-Hub skeleton and never
  initialized W&B or generated a group. Implementation revision 2
  (`A1-R1-complete-source-bundle-and-preserve-attempt-logs`) binds the missing
  file, archives each attempt log/result before retry, and allows automatic
  retry only for a recognized provider-infrastructure signature. All other
  nonzero exits fail validation and stop the DAG. The original version-2
  launch directory remains preserved as provenance.
- Verification: the focused amendment/protocol/launcher/verifier suite passes
  55/55 under repository Python 3.12. The complete pinned
  pilot/preregistration suite passes 106/106 with Python
  3.12.13, TRL 1.2.0, Transformers 5.5.4, Torch 2.7.1, Datasets 4.8.4,
  Hugging Face Hub 1.11.0, W&B 0.21.0, and NumPy 2.2.6. Ruff check and format
  verification are clean. Final protocol SHA-256 is
  `5cedb119f9810a0522d91216e746613c9bb1baec18d3d329ed4e80f0eadf019e`;
  generated screening-manifest SHA-256 is
  `f06285dc7ad370b72f30c9be720f082f15d4398271019e55d01ae5742749a0df`;
  generated launch-manifest SHA-256 is
  `3d50dc6968fd76ffde6ad5d0bde9b31a594723cf15cea861790c9ed9274643ce`;
  launch fingerprint is
  `6adce0c03b0e58b32c3a5cf373284a322aff0b1bb03ed929b2d907c10896e032`;
  source-binding SHA-256 is
  `6ba055857cb1690056a30c8eb6a8a097e3fe5c66bd9623654b0ec467fab21c68`.
- The revision-2 source-bound A100 smoke is independently accepted. The first
  revision-2 corpus job, balanced-equal-length seed 11, completed all 100
  deterministic groups on an A100 as W&B run `b8eoqd09`. Its first
  immutable prefix was committed after group 20 at exact private-Hub revision
  `46030fba999dccbabc40567ab8f605589aa6a50a`, fingerprint
  `b1ef87db8d0adb5ca99540ef9eeaada1d9c3d99cb7e43b006f4c352010b419c7`.
  A separate local invocation of the remote-checkpoint verifier downloaded and
  re-hashed the source manifest plus all 20 group files, reconciled 80,081
  charged generated tokens and 8,155 profiled tokens, and accepted the exact
  protocol/runtime/source/order bindings. The same independent procedure then
  accepted the replacement group-40 prefix at exact private-Hub revision
  `55091520f883bec456fe3f3334edf68dbc770013`, fingerprint
  `7ab66ac6e11557780060eba72ce874ec70dddb0c4c7f3565a380b7ee79457ff8`,
  with all 40 group files, 160,423 charged tokens, 12,251 profiled tokens, one
  attempt, and zero resumes. The group-60 replacement was independently
  accepted at exact private-Hub revision
  `4776e185ee8a91e924672179062380fb9423bddb`, fingerprint
  `49cbcebe4504b7cd494a20159b48f8a8da560c36f7c342ace12c5c0b22abca00`,
  with all 60 group files, 236,615 charged tokens, 15,644 profiled tokens, one
  attempt, and zero resumes. The final resumable group-80 prefix was then
  independently accepted at exact private-Hub revision
  `2faf00b02c5c81fcdcd2c4ed9e97e5fa8b721101`, fingerprint
  `500b462efb02e361fa3bbf0e8a3d09202dbfda8adb64c2493db150df22d9dda6`,
  with all 80 group files, 317,482 charged tokens, 19,740 profiled tokens, one
  attempt, and zero resumes. The final verifier independently downloaded and
  re-hashed the 100-group corpus, source manifest, token/FLOP ledger, and the
  referenced group-80 commit. It accepted corpus fingerprint
  `8b24a0520a97f0d5101c2662a1e3e369e8342c1759c9963a0ccb909b01525589`
  at exact private-Hub commit `91ec135ce5ffd562d991e535a16cae28c6552389`,
  with 396,672 charged generated tokens and zero resumes. The foreground local
  launcher had exited while the Colab execution survived; the stale PID and
  exact no-duplicate remote adoption are preserved in
  `launch-v2-corpus-resume-r1/recovery/corpus__balanced_equal_length__s11__attempt-1.json`.
  The completed A100 session was released, and a launchd-owned supervisor
  (label `ai.openai.codex.flagship-pilot-v2`) unlocked balanced seed 23 plus
  the first two seed-11 scientific units.
- The first two r1 scientific units (`intended_full` W&B `275344ae` and
  `native_trl` W&B `92a856a4`) shared the same step-0 accuracy `0.15625` and
  then failed before optimizer step 1. Preserved Colab tracebacks localized
  `value cannot be converted to type int64_t without overflow` to Qwen3 SDPA
  during gradient-checkpoint recomputation. Correction
  `A1-R2-hold-deterministic-sdpa-through-checkpoint-backward` keeps
  `SDPBackend.MATH` active through every `torch.autograd.grad`; both failed
  runs, their empty private-Hub skeletons, and the previously accepted r1
  corpus remain excluded from the corrected campaign.
- The resulting r2 non-scientific A100 smoke completed backward but emitted
  impossible cosines `1.00221848487854` and `1.0022610425949097`. The old
  verifier checked finiteness but not `[-1,1]`, so r2 is preserved as
  `superseded_no_further_launches`; its automatically queued balanced-s11
  corpus was stopped during runtime installation before W&B, private-Hub, or
  replay-group initialization. Implementation revision 4 correction
  `A1-R3-bound-cosine-diagnostics-and-verifiers` now computes receipt-only
  diagnostics in float64, tolerates/clamps only `1e-12` roundoff, and rejects
  invalid cosine, relative-L2, or gradient-norm fields in both preflight and
  full-record verification.
- The exact pinned r3 gate passes 109/109 (97 pilot + 12 preregistration) and
  the focused revision-4 gate passes 55/55; Ruff check and format verification
  are clean across all 15 changed Python files. Protocol SHA-256 is
  `04d20f712f652f80754fa4c8c0a3f48d4d2f1c5d716b3981746322c938b21970`,
  screening fingerprint `9d4af2a016552a0abaef4f39e4aa9e006f0afab731517eab16a562b2da346adb`,
  launch fingerprint `f01ad8e3991365fcf36160386b32dfdc69c034d1697773f8868b9dd5682d7de3`,
  source-binding SHA-256
  `10e481a8a3d77a336fc150c53e90fb1df9baed26aa5c43c58fa548e9105aba83`,
  and deterministic source archive SHA-256
  `f04aff3fb8ef87be2bc885263750c2cc0b6be6bd71fcc8b02ab5be8f116fac31`.
  LaunchAgent `ai.openai.codex.flagship-pilot-v2-r3` is active with isolated
  r3 state. Preflight attempt 1 is independently accepted on an
  `NVIDIA A100-SXM4-40GB`: gradient cosine `0.999957795529626`,
  selected-vs-intended cosine `0.9999999999997982`, nonnegative relative L2,
  positive norms/FLOPs, exact runtime pins, and the exact source hashes above.
  Its smoke session was deleted; only corpus
  `balanced_equal_length` seed 11 attempt 1 is now running, while the other 29
  downstream jobs remain dependency-gated at zero attempts.
- The running r3 balanced-equal-length seed-11 corpus published its first
  immutable prefix at group 20. A separate local verifier downloaded and
  re-hashed `source_manifest.json` plus all 20 group artifacts and accepted
  exact private-Hub commit `7c6d13ee7b22ef1a9ca83f2a550a43fbcff8a7e9`,
  fingerprint `a054b9c6f1ce9a69424677f201c46c242c805bb22674e8744fedb381e3fe556b`,
  80,081 charged generated tokens, profiler steps `[1,20]`, 8,155 profiled
  tokens, exact A100/runtime/source/order bindings, one W&B attempt
  (`3jpcepfy`), and zero resumes. The same A100 session and launchd-owned
  controller remain live; this resumable prefix is infrastructure evidence,
  not an accepted corpus or scientific observation.
- The same r3 run published and independently passed its group-40 replacement
  prefix at exact private-Hub commit
  `b23d1da97dc5dadd3da6d133ba3ffb048d055af0`, fingerprint
  `5c1a6cf763737d63efa116e1bac67a5061e06f34dbd360ae6e1fefd7b42dda3b`.
  The verifier downloaded and re-hashed all 40 group artifacts plus the source
  manifest, reconciling 160,423 charged tokens, profiler steps `[1,20,40]`,
  12,251 profiled tokens, exact A100/runtime/source bindings, W&B run
  `3jpcepfy`, one attempt, and zero resumes. W&B exposes the identical
  checkpoint commit/fingerprint and the launchd-owned run remains live toward
  group 60. This still does not count as a complete corpus or scientific unit.
- The group-60 replacement is now independently verified at exact private-Hub
  commit `a0c83171731c497ce13ae1dcc14b48b045c72956`, fingerprint
  `dd7caf181a7463196d86d404ea21ff2fe5b88e8878f388757b70b8a268ff5790`.
  The verifier downloaded and re-hashed the complete 60-group prefix (61
  artifact files) and all 14 source-manifest entries, reconciling 236,615
  charged tokens, profiler steps `[1,20,40,60]`, 15,644 profiled generated
  tokens, 65,619,873,117,824 generation FLOPs, exact A100/runtime/source/order
  bindings, one W&B attempt (`3jpcepfy`), and zero resumes. W&B exposes the
  identical checkpoint commit/fingerprint and the same attempt has continued
  through group 61 toward group 80. This remains restartable infrastructure,
  not an accepted corpus or scientific unit.
- The final resumable group-80 replacement independently passed at exact
  private-Hub commit `ba2a67680eee15e956f406fd9caebc83326967cf`,
  fingerprint
  `c50c78dda0978525d7bf32247087850436e844b43825234d572dc5a2ed3e4b12`.
  Independent download and re-hash covered the complete 80-group prefix (81
  artifact files) and all 14 source-manifest entries, reconciling 317,482
  charged tokens, profiler steps `[1,20,40,60,80]`, 19,740 profiled generated
  tokens, 82,337,615,882,312 generation FLOPs, exact A100/runtime/source/order
  bindings, one W&B attempt (`3jpcepfy`), and zero resumes. W&B exposes the
  identical checkpoint commit/fingerprint and token ledger. The same attempt
  continues toward the 100-group final record; this prefix is not an accepted
  corpus or scientific unit.
- Balanced-equal-length seed 11 is now the first independently accepted r3
  corpus. The full verifier downloaded all 185 remote files and re-hashed the
  100 group artifacts plus manifest and all 14 source entries at exact
  private-Hub commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`, corpus
  fingerprint
  `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.
  It reconciled 396,672 charged tokens, profiler steps
  `[1,20,40,60,80,100]`, 22,698 profiled generated tokens,
  98,454,319,002,760 profiled generation FLOPs, exact group-80 checkpoint
  commit/fingerprint, exact A100/runtime/source/order bindings, one finished
  W&B run (`3jpcepfy`), one attempt, and zero resumes. The supervisor emitted
  `acceptance/corpus__balanced_equal_length__s11.json` and released the Colab
  session. The eligible count is now one r3 corpus and zero scientific units.
  The next authorized wave contains exactly one corpus builder (balanced seed
  23) plus the intended/native balanced-seed-11 units; all three are on A100s.
- That first unit wave exposed a deterministic protocol contradiction before
  optimizer step 1. W&B runs `22107a6b` (intended) and `07c23895` (native)
  both finished `failed` after the same step-0 evaluation (`accuracy=0.15625`,
  64,038 generated tokens). The native traceback is preserved and terminates
  at `TrainingContractError: intended gradient norm is non-positive or
  non-finite`. Exact replay group 1 has rewards `[0,0,0,0,0,0,0,0]`; all
  three losses therefore have zero advantages and zero gradients. A complete
  corpus audit found 59 all-zero, 3 all-one, and 38 mixed-reward groups. Thus
  the r3 rule requiring positive norms at all 100 steps conflicts with genuine
  variance starvation and with the 95/100 balanced-equivalence gate. Neither
  failed run produced an optimizer step, checkpoint, final record, or eligible
  scientific observation.
- The supervisor correctly marked native as `failed_validation` and exited,
  but crash-only launchd then retried the persistent validation failure. The
  launch agent was unloaded; orphaned intended/native sessions were released
  after their W&B failures, and no replacement units launched. The detached
  balanced-seed-23 corpus was allowed to reach its first atomic recovery point,
  then stopped. Its group-20 prefix independently verifies at private-Hub
  commit `b1d897a968470898848ddb85ba24a334c3d59237`, fingerprint
  `67d51945e773e9e6aa50a88f8d72a182230c2452bd0285caf00be554b1aa1764`,
  with 80,988 charged tokens, profiler steps `[1,20]`, 7,797 profiled tokens,
  one W&B attempt (`ge121gt6`), and zero resumes. No Colab session or local
  controller remains. The stop propagated after W&B had logged through group
  22 (86,052 charged tokens), but no later Hub commit exists and the orphaned
  W&B run remains stale `running`; only the exact group-20 commit is recoverable
  evidence. Continuing requires an explicit receipt/scoring and
  corpus-source-reuse amendment; no such amendment has been applied.
- The user authorized `A1-R4` with corpus reuse on 2026-07-22. Implementation
  revision 5 now emits explicit `nonzero`, `joint_zero`, and named one-sided
  zero relations for both gradient comparisons. Zero-vector cosine and
  relative-L2 fields are `null`; joint-zero scores as equivalence/zero effect,
  one-sided-zero scores as maximal divergence, and all nonzero thresholds are
  unchanged. A selected zero gradient skips `optimizer.step()` entirely while
  the frozen scheduler advances once, so AdamW state and parameters are a true
  no-op. No replay group is dropped, reordered, regenerated, or filtered.
- Corpus provenance is split from unit-training provenance in
  `pilot/provenance/r3-corpus-bindings.json`. The exact r3 corpus runtime and
  full control-plane archives hash to `8d8b201d3e8e914cc6c7d35f569e389b460c53e02075932c2ab4ee417a700ede`
  and `f04aff3fb8ef87be2bc885263750c2cc0b6be6bd71fcc8b02ab5be8f116fac31`.
  Live verification reaccepted balanced seed 11 at final commit
  `2735a27d5f18bbdaaae76494a2047b39a4318e22` / fingerprint
  `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`
  and balanced seed 23 at its exact group-20 commit
  `b1d897a968470898848ddb85ba24a334c3d59237` / fingerprint
  `67d51945e773e9e6aa50a88f8d72a182230c2452bd0285caf00be554b1aa1764`.
  Incomplete corpus jobs use the frozen r3 archive; all 24 units use the new
  revision-6 source bundle.
- The final local A1-R4 gate passes 103/103 pilot tests plus 12/12
  preregistration tests (115/115 total), with the focused correction suite
  passing 69/69 and Ruff check/format clean across all 22 changed Python files.
  The revision-6 protocol SHA-256 is
  `1b001a920a042ee2a41f232175066483b4b28e5e37db2e7e9ebf48d0a561007a`.
  Final r4 plan and launch hashes are recorded in the generated manifests
  under `plans-v2-corpus-resume-r4-1/` and
  `launch-v2-corpus-resume-r4-1/`.
- The first revision-5 smoke attempt installed the exact remote pins but failed
  closed during environment validation because the unit source bundle omitted
  the frozen archive files whose existence the protocol itself verifies. It
  stopped before model loading, W&B, Hugging Face, replay generation, or any
  update. Correction `A1-R4.1-package-frozen-archives-for-remote-validation`
  adds both already-hashed archives to the revision-6 upload and runtime source
  manifest, preserves the failed r4 launch directory, and requires a new r4-1
  source/session identity and fresh smoke.
- The fresh revision-6 A100 smoke is independently accepted. It resolved the
  exact package pins on `NVIDIA A100-SXM4-40GB`, validated the embedded frozen
  archives remotely, and emitted a nonzero receipt with intended/native cosine
  `0.999957795529626`, relative L2 `0.009205099545490102`,
  selected/intended cosine `0.9999999999997982`, positive norms, positive FLOPs
  in all required phases, and `optimizer_update=applied`. The smoke session was
  terminated. The frozen balanced-s11 corpus was then reaccepted at its exact
  pinned final commit/fingerprint; the supervisor released balanced-s23 corpus
  resume plus the new intended/native balanced-s11 units under revision-6
  source identity `c4bc5205...`.
- Live post-smoke reconciliation: balanced seed 23 is W&B `ncpafe25`, restored
  exactly from group 20 with `resume_count=1`, checkpoint commit
  `b1d897a968470898848ddb85ba24a334c3d59237`, and checkpoint fingerprint
  `67d51945e773e9e6aa50a88f8d72a182230c2452bd0285caf00be554b1aa1764`.
  It has advanced through group 23 and 90,148 cumulative charged tokens. Two
  pre-allocation `TooManyAssignmentsError` exits consumed local attempts 1--2;
  its live A100 is the authorized third and final local attempt, with no
  duplicate remote corpus run. Intended W&B `a0a67b52` is live on local
  attempt 1; native W&B `87ba3535` is live on local attempt 2 after one
  pre-allocation capacity rejection. Both consume the exact pinned seed-11
  corpus and are still in the frozen step-0 evaluation before the first
  scientific receipt.
- Revision-6 then exposed a purely numeric diagnostic failure, not a scientific
  divergence. Intended W&B `a0a67b52` emitted a valid step-1 `joint_zero`
  optimizer no-op and a valid nonzero intended/native comparison at step 2,
  but its byte-identical selected/intended vectors were reduced separately to
  cosine `1.000000000002599` and rejected as outside `[-1,1]`. Native W&B
  `87ba3535` independently corroborated joint-zero no-ops at steps 1 and 4--6
  and valid nonzero receipts at steps 2--3 before its superseded source session
  was stopped. Neither revision-6 unit is eligible under the corrected source
  identity.
- Authorized correction `A1-R4.2-exact-identical-gradient-diagnostics` advances
  unit training to implementation revision 7. Exact tensor equality now emits
  cosine `1.0` and relative L2 `0.0` directly; all zero-vector and genuinely
  nonzero rules remain frozen. The countable local gate passes 104/104 pilot
  tests plus 12/12 preregistration tests (116/116), the focused gate passes
  70/70, all 22 changed Python files pass Ruff check/format, and independently
  regenerated plans are byte-identical. Protocol SHA-256 is
  `87d929d0a3af789d3ba3ee10a1f4c3e83572ecec7cc4efa28ca032008f88fbc4`;
  unit source binding is
  `005d3f8242b992cf70af2944c2b3f63351f5d3e00e95cdc5caeb40d1261b0918`.
  Generated r4-2 plan/launch fingerprints are
  `c6e5829410a85af9574c6b43c0a75bb5a78c5c2bd7c132637a72437ed0e3c37a`
  and `25ef91234d58643c2d1eaea23832e0b676cb99e66ce774551b1f0ae1de9cee0d`.
- The fresh revision-7 A100 smoke is independently accepted and its session is
  released. It reproduces intended/native cosine `0.999957795529626`, relative
  L2 `0.009205099545490102`, positive required phase FLOPs/norms, and an
  applied optimizer update; its exact-equality selected/intended diagnostic is
  now cosine `1.0`, relative L2 `0.0`. Balanced seed 11 was independently
  reaccepted at final commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`,
  fingerprint `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.
- The surviving balanced-seed-23 W&B run `ncpafe25` reached and independently
  passed its group-40 private-Hub checkpoint at commit
  `b45dc64a59a8cd7fb068d0f2182c507c34db8aec`, fingerprint
  `1d7e72efb8df8e22beb15a9756d8255aa6b44f4f4a9f4af3d53b547143138c37`,
  with 158,590 charged tokens. It remains the only corpus builder and has since
  advanced through group 58 / 230,855 tokens without a duplicate. Fresh r4-2
  intended/native seed-11 sessions `fpilot-inte-bala-s11-87d9005d` and
  `fpilot-nati-bala-s11-87d9005d` are live on A100s. Their first detached local
  launch exited before allocation because LaunchAgents lacked the Colab CLI
  path. Attempt 2 allocated both A100s, but the host rebooted at 18:52:18; the
  automatic `RunAtLoad` recovery collided with the surviving session names,
  received capacity-only 412 errors, and stopped those attempt-2 sessions
  before W&B/HF or scientific state. The agents are now non-RunAtLoad, and the
  single remaining authorized allocation attempt 3 is live for both units.
  Another host restart cannot auto-relaunch or stop them. The hard three-A100
  and one-corpus ceilings are exact.
  Confirmatory execution remains forbidden until a screening GO verdict.
