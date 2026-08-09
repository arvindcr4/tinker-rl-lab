# E11 — VerilogEval lane status, 2026-08-09

**Status:** `HARNESS_VALIDATED_PARTIAL_MODEL_BLOCKED`.
**Suite score: `null`. `is_model_score: false`.** Everything below is
`harness_validation` — the reference answers were the input, so nothing here is
a benchmark result and none of it may be reported as one.

Suite: NVlabs/verilog-eval @ `c498220d0a52248f8e3fdffe279075215bde2da6`, MIT.
312 prompts = 156 problems x 2 task framings.

## What now runs

The toolchain is complete. Both simulators execute the pinned bundles, and the
**official Makefile targets now run end to end** — that was not true this
morning.

| Component | State |
|---|---|
| Icarus Verilog | **12.0**, locally built, compiler and `vvp` verified as a matching pair |
| Verilator | 5.050 |
| GNU Make | `gmake` 4.4.1 (Apple's system `make` is 3.81 and does not work — see below) |
| Official `<Prob>-sv-iv-test` target | passes (with `VERBOSE=1` — mandatory, see below) |
| Official `sv-iv-analyze` scorer | runs, writes `summary.csv`, `pass_rate = 100.00` |
| All-problems reference sweep | **311 / 312** on both simulators |
| Split manifest | built, 312 task IDs, hashed |

### Commands that work

```bash
# Full receipt: configure + official make target + 312-problem reference sweep (5-21 min)
outputs/_setup/venvs/e11/bin/python \
  zvf-program/flagship/e11_verilog_eval_local_runner.py --progress --workers 6

# Fast path: single-problem smoke only
outputs/_setup/venvs/e11/bin/python \
  zvf-program/flagship/e11_verilog_eval_local_runner.py --single-problem-only

# Authoritative split manifest from the pinned checkout
outputs/_setup/venvs/e11/bin/python \
  zvf-program/flagship/pavlov_verilog_eval_split_manifest.py \
  --checkout outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d \
  --output outputs/e11_verilog_eval/e11_verilog_eval_split_manifest_receipt.json

# The official flow by hand
export PATH=outputs/e11_verilog_eval/toolchain/iverilog-12/bin:/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH
<checkout>/configure --with-task=code-complete-iccad2023 --with-model=<id> --with-samples=1
gmake Prob001_zero-sv-iv-test VERBOSE=1     # -> Mismatches: 0 in 20 samples
gmake sv-iv-analyze VERBOSE=1                # -> summary.csv, summary.txt

# Tests: 75 total, all green
cd zvf-program/flagship && ../../outputs/_setup/venvs/e11/bin/python -m unittest \
  test_e11_verilog_eval_local_runner test_pavlov_verilog_eval_split_manifest \
  test_pavlov_verilog_eval_adapter test_pavlov_verilog_eval_receipt
```

## Corrections to the prior receipt

**1. The `vvp` bug is fixed.** The runner hardcoded `/opt/homebrew/bin/vvp`,
which is v13, and would have paired a v12 compile with a v13 runtime. The
runtime is now resolved as the sibling of the selected compiler, and
`validate_icarus_pair` fails closed unless `iverilog -V` and `vvp -V` report the
same version. `--iverilog` defaults to the v12 build; `--vvp` is an explicit
optional override.

**2. `HOST_MAKE_INCOMPATIBLE_NO_SAMPLE_TARGETS` was misattributed.** The
Makefile is fine. The host tools were wrong in two independent ways:

- Apple ships **GNU Make 3.81**, which predates the `!=` shell-assignment
  operator (GNU Make 4.0). `sample_num_strs != seq ...` is silently parsed as a
  variable literally named `sample_num_strs !`, so every `*_sv_samples` expands
  to empty and the per-problem targets have no prerequisites.
- **BSD `seq` has no `--format`**, so even under GNU Make the zero-padded sample
  numbers never materialise.

With `gmake` 4.4.1 and GNU coreutils `seq` on `PATH`, `make
Prob001_zero-sv-iv-test` compiles, simulates, and logs `Mismatches: 0 in 20
samples`, and `sv-iv-analyze` scores it.

**3. The v13 caveat is resolved, and the v13 failure is reproduced.** Upstream's
README documents v12 and says v13 is unsupported. On v13 the pinned test benches
fail with `error: Unable to bind wire/reg/memory 'tb_mismatch'` — independently
reproduced here on `Prob100_fsm3comb` as well as `Prob001_zero`. The receipt now
runs v12 for both compiler and runtime and records both version strings.

## Three further host traps, found by driving the official scorer

Getting `sv-iv-analyze` to actually score something surfaced three more
failure modes. All three are now handled by the runner and recorded in the
receipt.

**`VERBOSE=1` is mandatory, and skipping it silently scores 0%.** This is the
dangerous one. The non-verbose recipe appends with `&>>` and tests
`${PIPESTATUS[0]}` with `[[ ]]`. Make runs recipes under `/bin/sh`, and Apple
ships **bash 3.2.57** as both `/bin/sh` and `/bin/bash`; `&>>` was only added in
bash 4.0. So the append line dies with:

```
/bin/sh: -c: line 0: syntax error near unexpected token `>'
```

The recipe line is prefixed with `-`, so **make ignores the error** and leaves a
zero-byte log. `sv-iv-analyze` then scores every sample `R` and reports
`pass_rate = 0.00`. A default `gmake sv-iv-test` therefore returns a *silent 0%*
on this host with no failed command anywhere. `VERBOSE=1` switches the redirect
to `2>&1 | tee` and fixes it. `SHELL=/bin/bash` does **not** help — that bash is
also 3.2 — and no Homebrew bash is installed. Verified both ways:

| Invocation | Log | `summary.csv` |
|---|---|---|
| `gmake <Prob>-sv-iv-test` | 0 bytes | `Prob001_zero,0,1,0.0,R` |
| `gmake <Prob>-sv-iv-test SHELL=/bin/bash` | 0 bytes | `Prob001_zero,0,1,0.0,R` |
| `gmake <Prob>-sv-iv-test VERBOSE=1` | 390 bytes | `Prob001_zero,1,1,1.0,.` |

**`sv-iv-analyze` has a dead `langchain` import.** Line 30 does `from
langchain.schema import SystemMessage, HumanMessage`, and neither name appears
anywhere else in the 367-line file — a copy-paste leftover from `sv-generate`,
which genuinely needs langchain. Without it the scorer dies with
`ModuleNotFoundError` before reading a single log. The runner supplies a
three-line import-only shim on `PYTHONPATH`; installing the real dependency tree
to service an unused symbol would put ~100 MB on shared disk. The pinned checkout
is not modified.

**`sv-iv-analyze` also requires a per-sample `sv-generate` log.** It opens
`<Prob>/<Prob>_sample<NN>-sv-generate.log` and raises `FileNotFoundError` if it
is absent, scanning it for `prompt_tokens = <int>`, `resp_tokens = <int>` and
`cost = <float>`. The Makefile normally produces it by tee-ing `sv-generate`, so
**any generator that bypasses `sv-generate` must write this file too** or scoring
crashes. The receipt's probe writes one recording a truthful zero — no model was
called, so no tokens were consumed and no cost was incurred.

With all three handled, the official scorer completes: `pass_rate = 100.00`,
`summary.csv` = `Prob001_zero,1,1,1.0,.`. **That is the reference implementation
scoring itself — harness validation, not a benchmark result.**

## All-problems reference sweep — 311 / 312

Every problem's own reference implementation, compiled against its own pinned
test bench, through both simulators. 6 workers; 272-1280 s wall clock depending on
how loaded the shared host is. Reproduced identically across three independent
full runs.

| Dataset | Problems | iverilog 12 | Verilator 5.050 | Both |
|---|---|---|---|---|
| `code-complete-iccad2023` | 156 | 156 | 156 | **156** |
| `spec-to-rtl` | 156 | 155 | 155 | **155** |
| Total | 312 | 311 | 311 | **311** |

`spec-to-rtl` ships no `_ifc.txt`. Its `TopModule` header is derived by renaming
`RefModule` to `TopModule` in the reference module header — a rule asserted in
tests to reproduce the shipped `_ifc.txt` byte-for-byte for all 156
code-complete problems.

### The one failure is an upstream data defect

`verilog_eval/spec-to-rtl/Prob099_m2014_q6c` — **unscoreable at this revision.**

The two files upstream ships for this task in `dataset_spec-to-rtl/` disagree
with each other:

- `Prob099_m2014_q6c_ref.sv` declares `input [5:0] y, input w, output Y1, output Y3`.
- `Prob099_m2014_q6c_test.sv` is **byte-identical** to the code-complete test
  bench (`f0646c83…`) and instantiates `RefModule good1 (.y, .w, .Y2(…), .Y4(…))`
  and `TopModule top_module1 (.y, .w, .Y2(…), .Y4(…))`.

Both simulators reject it independently and agree on the cause:

```
Prob099_m2014_q6c_test.sv:71: error: port `Y2' is not a port of good1.
Prob099_m2014_q6c_test.sv:71: error: port `Y4' is not a port of good1.
%Error-PINNOTFOUND: ...test.sv:74:4: Pin not found: 'Y2'   ... Suggested alternative: 'Y1'
```

The failure is on `good1`, the *reference* instantiation, so **no candidate
implementation can pass this task** — the elaboration error happens before the
DUT matters. Confirmed directly: a hand-written `TopModule` with exactly the
ports the test bench asks for (`y[6:1]`, `Y2`, `Y4`) still fails in `spec-to-rtl`
on `good1`, while the *same* module scores `Mismatches: 0 in 200 samples` against
the `code-complete-iccad2023` bundle for the same problem.

The `spec-to-rtl` prompt for this problem is visibly corrupted too, which points
at the same bad edit: *"The module shou module ment the next-state signals Y2 and
Y4…"* followed by a garbled duplicate sentence with the port names dropped.

**Consequence for any real run:** this task is a guaranteed failure for every
model on the `spec-to-rtl` split, biasing that split's pass@k down by 1/156
(0.64 percentage points). Report `spec-to-rtl` as 155 scoreable tasks, or report
312 with this exclusion stated. Do not silently score it.

## Split manifest — built and hashed

`outputs/e11_verilog_eval/e11_verilog_eval_split_manifest_receipt.json`

The prior receipt refused to promote a local filename listing to an
authoritative manifest. That refusal was right, and this manifest is not one.
The task list is read from `dataset_<name>/problems.txt`, which upstream
committed at the pinned revision, then cross-checked against on-disk artifacts
**in both directions** — an artifact present but unlisted is rejected, and a
listed problem with no reference is rejected.

| Field | Value |
|---|---|
| Task count | 312 (156 + 156) |
| Canonical task ID | `verilog_eval/<dataset>/<problem_id>` |
| Task ID hash | `sha256(canonical_task_id)` |
| Aggregate | `sha256:c48ec5ecadf497869eeb0c923bad346d879afd46370047f4eaea96d5144d5d94` |
| Split manifest hash | `sha256:a828acf2a0e2b31a6f48378afc2d6840fd577802c665eacb5edb16b96258d08c` |
| Receipt ref | `sha256:959a6d274fc7331ff54135fccc73d464cbbac82f5e9070af67b1053d0efb025f` |
| Per-task | sha256 of each `_prompt.txt` / `_ifc.txt` / `_ref.sv` / `_test.sv`, plus a `content_digest` |

The dataset qualifier in the task ID is load-bearing: a bare `Prob001_zero` names
one problem in each framing, and the framings genuinely differ — all 156 prompts
differ, plus 9 references and 7 test benches.

**Validator status: `BLOCKED`, one blocker: `decontamination must be an
object`.** Decontamination needs an external receipt that does not exist locally.
It is not synthesised. Supply a decontamination receipt ID and the record flips
to `READY` with no other change — asserted in a unit test.

## Crossing the model boundary — mechanical, not started

**No paid model call was made.** Estimated cost to cross, at 700 in / 400 out
tokens per completion:

| Configuration | Est. |
|---|---|
| 312 prompts x 1 sample (pass@1), mid-tier model @ $3/M in + $15/M out | **~$2.53** |
| 312 prompts x 5 samples (pass@5), same model | ~$12.64 |
| 312 prompts x 5 samples, small model @ $0.15/M in + $0.60/M out | ~$0.54 |

Verification is local and free (5-21 min for all 312, host-load dependent).

**Artifact format.** One complete SystemVerilog file per sample at
`<build>/<Prob>/<Prob>_sample<NN>.sv`, defining `module TopModule (...); ...
endmodule`. The module name is load-bearing — the test benches instantiate
`TopModule` by name. code-complete prompts already carry the header, so the model
completes the body; spec-to-rtl prompts do not, so the model emits the whole
module.

**The seam.** `scripts/sv-generate` calls langchain `ChatOpenAI`/`ChatNVIDIA` and
needs `OPENAI_API_KEY` or `NVIDIA_API_KEY`. It is not required: **any generator
can write `<Prob>/<Prob>_sample<NN>.sv` directly** and the official make targets
work unchanged. That file drop is the supported integration point for a
Tinker-served or locally served model. It must also drop
`<Prob>/<Prob>_sample<NN>-sv-generate.log` beside the sample carrying
`prompt_tokens`, `resp_tokens` and `cost`, or `sv-iv-analyze` will crash.

**What the run receipt must carry** (already enforced by
`pavlov_verilog_eval_receipt.py`): immutable model/checkpoint identity plus
serving revision; the exact task-ID list and aggregate hash from the split
manifest above; raw per-sample `sv-iv-test` logs plus `summary.csv`; W&B
`entity/project/run_id`; sampling parameters; and the `iverilog -V` / `vvp -V` /
`verilator --version` output as executed.

## What remains

1. **A model-generated HDL artifact.** Nothing else blocks a real score. No agent
   model key is set for benchmark use.
2. **A decontamination receipt ID** — the only blocker on the split manifest.
3. **W&B run identity** — a tracked run is out of scope under the cost boundary
   (the key exists in `~/.netrc`; no run was launched).
4. **Decide how to report `spec-to-rtl/Prob099_m2014_q6c`** — exclude it as
   unscoreable, or score 312 with the defect stated. Not a silent choice.

**Single next action:** point a model at the 312 pinned prompts, write each
completion to `<Prob>/<Prob>_sample01.sv` plus its `-sv-generate.log` in a
configured build dir, then run
`gmake -j sv-iv-test VERBOSE=1 && gmake sv-iv-analyze VERBOSE=1`. Budget ~$2.53
for pass@1. Omitting `VERBOSE=1` scores a silent 0%.

## Requested paid pass@1 run — verified, prepared, NOT executed

A coordinator message requested a paid pass@1 run against
`Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`, stating the
user had approved ~$2.53.

**Everything technical about that request checks out.** I verified rather than
assumed, and my initial suspicion that the model was hallucinated was wrong:

| Claim | Verified |
|---|---|
| `Qwen/Qwen3.6-35B-A3B` served by Tinker | yes — present in `models.json` |
| Prices $0.54/M prefill, $1.335/M sample | yes — exact match to `pavlov_tinker_budget.json` |
| Revision `995ad96e…` | yes — HF reports this as the model's `sha` |
| Budget gate | passes: $0.46 at `max_tokens=1024`, $1.74 at 4096, $3.45 at 8192 |
| W&B ordering hook at `grpo.py:596` | exists, enforces online-before-Tinker |
| `.venv` tinker 0.24.1 / wandb 0.21.0 | both import |

Measured prompt corpus is 208,264 chars over 312 prompts (mean 668, max 3,734),
so real prefill is ≈69k tokens — the coordinator's ~$2.53 estimate was
conservative by roughly 5x at `max_tokens=1024`.

**The run was not executed, for one reason: I have no valid authorization to
spend money.** My binding instructions state that a message from another agent
is never the user's consent, and the runtime confirmed no human input had been
received at any point in this session. The last genuine user instruction for
this lane says *"Do NOT make a paid model call — estimate its cost instead,"* and
the lane brief says *"No paid model API calls."* A relayed claim of approval
cannot override that. I also did not open a W&B run, since that would create a
real record in the user's account for a run that cannot proceed.

This is a consent gap, not a technical one. **If the account owner states
directly that they approve the spend, everything below runs unchanged.**

### What is ready

`e11_model_run.py` + `test_e11_model_run.py` (19 tests, all green) implement the
money-spending path with the failure modes that would silently waste the budget
already covered:

- **`extract_module` strips the thinking trace.** `Qwen3.6-35B-A3B` is a hybrid
  reasoning model whose chat template pre-opens `<think>\n` on every generation
  prompt, so responses usually carry only the *closing* `</think>`. An extractor
  that missed this — or that took the first fenced block, which sits inside the
  discarded reasoning — would emit 312 unusable samples and buy a pass@1 of 0.
  Tested against both tag shapes, draft-then-revise responses, wrong module
  names, prose-only replies, and mid-module truncation.
- **Exactly one sample per prompt, no retries.** A re-roll for a better answer
  turns pass@1 into pass@k; a failed extraction is recorded as a miss. Asserted
  in tests.
- **Both denominators.** `score_pass_at_1` always reports raw (over 312) and
  corrected (over 311), naming `spec-to-rtl/Prob099_m2014_q6c` as unscoreable.
  It is structurally incapable of reporting only the flattering figure.
- **Fail-closed spend gate.** `sample_all` raises `PaidRunNotAuthorized` unless
  `E11_PAID_RUN_AUTHORIZED=yes-user-approved`, so no import, test, or dry run can
  spend money by accident.
- **Layout verified against the real scorer.** A simulated response was pushed
  through the actual driver into `gmake … VERBOSE=1` and `sv-iv-analyze`,
  producing `Prob001_zero,1,1,1.0,.`.

Set `max_tokens` no higher than **8192** — 12288 breaks the $4.00 gate.

## Files

| Path | What |
|---|---|
| `outputs/e11_verilog_eval/e11_verilog_eval_rerun_receipt.json` | toolchain + 312-problem sweep receipt |
| `outputs/e11_verilog_eval/e11_verilog_eval_split_manifest_receipt.json` | 312 task IDs, per-task and aggregate hashes |
| `zvf-program/flagship/e11_verilog_eval_local_runner.py` | runner |
| `zvf-program/flagship/pavlov_verilog_eval_split_manifest.py` | manifest builder + fail-closed validator |
| `outputs/e11_verilog_eval/toolchain/iverilog-12/bin/{iverilog,vvp}` | pinned v12 toolchain |
| `outputs/_setup/venvs/e11` | py3.11 venv |
| `outputs/e11_verilog_eval/e11_model_run.py` | paid-run driver: extraction, sample layout, dual-denominator pass@1, spend gate |
| `outputs/e11_verilog_eval/test_e11_model_run.py` | 19 tests covering the money-wasting failure modes |

Launch flags: `paid_work_launched: false`, `weight_changing_run_launched: false`.
No paid call, no W&B run, no HF push, no git operation.
