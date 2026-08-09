# E2 — FrontierSWE (`frontier_swe_eval`) — lane status 2026-08-09

**Status: BLOCKED. `score: null`.** No model score was produced and none can be
produced on this host. Two blockers from the prior receipt survive, and this pass
found a third, harder one.

What this pass did land: the official checkout is now **durable and
revision-verified**, and the native verifier was driven **end to end without a
model** — `test.sh` and `compute_reward.py` both execute and emit a well-formed
`reward.json`. That run is labelled `harness_validation` with
`is_model_score: false`; its `0.0` is a property of this host and is **not**
promoted to the suite score.

Receipt: `e2_frontier_swe_lane_receipt_2026-08-09.json` (regenerate with
`python3 outputs/e2_frontier_swe/build_receipt.py`).

## 1. Durable checkout — DONE

The ephemeral `/private/tmp/frontier-swe-sparse-3xvxdf` sparse checkout has been
replaced by a full, durable clone.

```bash
git clone https://github.com/Proximal-Labs/frontier-swe.git \
  outputs/e2_frontier_swe/frontier-swe
git -C outputs/e2_frontier_swe/frontier-swe switch --detach \
  422b9bb95deb8efe436becb0ed3c44be23611e10
```

| Field | Value |
|---|---|
| Path | `outputs/e2_frontier_swe/frontier-swe` |
| `HEAD` | `422b9bb95deb8efe436becb0ed3c44be23611e10` — **matches the pin** |
| Commit date / subject | `2026-08-07T13:48:30+05:30` — `update: fix grok cli list if else bug (#84)` |
| Tree object | `cf29dbbec20fe3f95de4c902d354742c6933eb02` |
| Working tree | clean (`git status --porcelain` empty) |
| Size | 1.4 GB (513 MB `.git`) — full history, all blobs, no sparse filter |
| Tasks | **17**, matching the suite fact |

Unlike the old checkout this is a complete clone: no `--filter=blob:none`, no
sparse cone. It survives a reboot and needs no network to re-materialise files.

## 2. Blocker 1 — no root LICENSE. **Confirmed factually.**

Verified three independent ways at the pinned revision:

```bash
git ls-tree --name-only HEAD          # .gitignore README.md SCORING.md docker
                                      # harbor_ext pyproject.toml scripts tasks uv.lock
git ls-tree -r --name-only HEAD | grep -Ei '(^|/)(LICENSE|LICENCE|COPYING|NOTICE|EULA|TERMS)'
```

- **No root `LICENSE`, `LICENSE.md`, `COPYING`, `COPYING.md`, or `NOTICE`.**
- The only licence-ish files in the whole tree at this revision are *vendored
  third-party* fixtures inside task payloads — `tasks/dart-style-haskell/solution/dart-sdk/LICENSE`,
  `tasks/git-to-zig/environment/git-src/COPYING`, and similar. None of them
  licenses FrontierSWE itself.
- `README.md`, `SCORING.md`, and `pyproject.toml` contain **no** licence
  declaration.
- GitHub's own API agrees: `GET /repos/Proximal-Labs/frontier-swe` returns
  `"license": null`.

The repository is public but **all-rights-reserved by default**. This is a real
policy gate, not a missing-file accident, and the runner enforces it:
`frontier_swe_eval.py:_license_receipt` looks for exactly those four filenames and
emits `benchmark_license_missing` when none is present.

## 3. Blocker 2 — no candidate workspace. Still open, and now fully specified.

See section 5 for the contract a real run must satisfy.

## 4. Blocker 3 (NEW) — the task image cannot execute on this host.

This is the finding that changes the picture. The published image is
**`linux/amd64` only**; the Colima VM is **aarch64** with **plain QEMU**
user-mode emulation and **no Rosetta** binfmt handler.

The task renders video through headless Google Chrome (Puppeteer). Under QEMU,
**Chrome core-dumps inside QEMU itself**:

```
[...]/chrome --headless --no-sandbox --single-process --dump-dom about:blank
  -> rc=134, elapsed 95835 ms
Assertion failed: p_rcu_reader->depth != 0 (/qemu/include/qemu/rcu.h: rcu_read_unlock: 102)
timeout: the monitored command dumped core
```

A single-scene baseline render consequently fails inside Puppeteer's default 30 s
wait (`DEFAULT_TIMEOUT = 30000` in
`node_modules/puppeteer-core/lib/cjs/puppeteer/common/TimeoutSettings.js`, which
Revideo never overrides):

```json
[{"scene":"hidden_text_animation","time_ms":38229,"success":false,
  "error":"Timed out after waiting 30000ms"}]
```

This is **not** fixable within the lane:

- That timeout lives in the **frozen baseline** at
  `/baseline/revideo/node_modules/`, which the image ships `chmod -R a-w`. Even a
  perfect agent patch to `/app/revideo` cannot make the baseline half of the ABBA
  comparison render, and without baseline timings `compute_reward.py` hard-fails
  on `baseline_results_missing` regardless of what the candidate does.
- No arm64 image exists, and the Dockerfile cannot produce one — it installs
  Chrome from `deb [arch=amd64] http://dl.google.com/linux/chrome/deb/` and
  builds `FROM ghcr.io/.../first-party-cli-base-ubuntu22.04` (amd64).
- Enabling Rosetta would mean changing Colima's VM config, which
  `outputs/_setup/LANE_BRIEF.md` forbids, and Rosetta is not a reliable fix for
  Chrome's threading under emulation either.

**FrontierSWE `revideo-perf-opt` needs a genuine x86-64 Linux host.**

## 5. Candidate workspace contract (objective 3)

There is **no patch file** in this suite. The candidate is a *directory* that is
bind-mounted over the agent workspace.

### Layout

The submission directory is a complete copy of the image's `/app/revideo` tree —
the Revideo **v0.4.2** monorepo — with the agent's edits applied in place. The
frozen reference is Revideo **v0.4.4** at `/baseline/revideo` (read-only), so the
task is "make v0.4.2 as fast as v0.4.4". `job.yaml`'s `artifacts:` list is the
authoritative enumeration of what a rollout must export:
`package.json`, `package-lock.json`, `pnpm-lock.yaml`, `pnpm-workspace.yaml`,
`tsconfig.json`, `tsconfig.options.json`, `packages/{internal,telemetry,core,2d,ffmpeg,vite-plugin,ui,renderer}`,
`packages/benchmark/{benchmark.mjs,package.json,src,output/benchmark_results.json}`,
plus `/logs/agent` and `/logs/verifier`.

### How it is mounted (`frontier_swe_eval.py:_native_verify`)

```bash
docker run --rm --network none --cpus 8 --memory 32768m \
  --entrypoint /bin/bash \
  -v <repo>/tasks/revideo-perf-opt/tests:/tests:ro \
  -v <submission_dir>:/app/revideo:rw \
  -v <output_dir>:/logs/verifier:rw \
  ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt:v4 \
  /tests/test.sh
```

`DEFAULT_CANDIDATE_MOUNTS = {"revideo-perf-opt": "/app/revideo"}`; `--cpus` and
`--memory` come from `task.toml` (`cpus = 8`, `memory_mb = 32768`).

### What the mount must satisfy

1. **Source scan.** No file under `packages/**` (excluding `node_modules`) may
   mention verifier internals — `/tests/hidden-scenes`, `hidden-scenes.tar.gz`,
   `/tests/compute_reward`, `/tests/test.sh`, `/baseline/revideo`, `reward.json`,
   `reward.txt`, `/logs/verifier`, `/tmp/hidden-scenes`, `.oracle_solution`,
   `HARBOR_ORACLE_MODE`. One hit ⇒ immediate `reward 0.0`, `hard_fail: true`.
2. **Build.** `npm run build -w packages/<pkg>` runs for `telemetry`, `core`,
   `2d` (`build-lib` when that script exists), `ffmpeg`, `vite-plugin`,
   `renderer`. The gate is **not** the exit code — test.sh appends `|| true` — it
   is whether each package's `package.json.main` (default `lib/index.js`) exists
   afterwards. Otherwise `reward 0.0`. See section 6.1 for why that distinction
   matters.
3. **Benchmark package.** `packages/benchmark/` must exist with `src/` and a
   `package.json`. The verifier *overwrites* `packages/benchmark/benchmark.mjs`
   with the baseline's copy and injects hidden scenes into
   `packages/benchmark/src/hidden_scenes_only/`, so agent changes to those are
   discarded.
4. **Root `node_modules` must be present** — `prep_build.py` symlinks `hls.js`,
   `mp4-wasm`, `mp4-muxer`, `mp4box`, and `comlink` from it into each package.
5. **Writable.** `prep_build.py` rewrites every `packages/**/tsconfig*.json`
   (forcing `strict:false`, `skipLibCheck:true`) and the build writes `lib/`.
6. **Do not touch `/baseline/revideo`.** It is not part of the submission.

### How `compute_reward.py` is invoked

Normal path, from `tests/test.sh` step 7:

```bash
python3 /tests/compute_reward.py \
  --baseline-results   /logs/verifier/baseline_output/benchmark_results.json \
  --candidate-results  /logs/verifier/candidate_output/benchmark_results.json \
  --correctness-results /logs/verifier/correctness_results.json \
  --output-dir /logs/verifier \
  --total-time-ms <int> [--oracle]
```

Hard-fail path (source scan or build failure):

```bash
python3 /tests/compute_reward.py --fail "<reason>" --output-dir /logs/verifier
```

### What it returns

Exit code is **0 on both paths** — the score lives in the files, not the status.
It writes `/logs/verifier/reward.json` and `/logs/verifier/reward.txt`.

`reward.json` keys: `reward`, `score` (identical to `reward`),
`geometric_mean_speedup`, `num_hidden_scenes`, `num_speedups_computed`,
`hard_fail_reasons[]`, `correctness_ok`, `is_oracle`, `total_time_ms`,
`per_scene[]`, `correctness_details[]`, `subscores[]`, `reason`.
`reward.txt` is the bare scalar to 6 dp.

**Semantics.** `reward` = geometric mean of `baseline_ms / candidate_ms` across
the 8 scenes whose name starts with `hidden_`, capped at 100.0 — and forced to
`0.0` if any hard-fail reason fires (missing baseline/candidate results, any SSIM
correctness failure, any missing or failed hidden scene). Timings come from an
**ABBA** schedule (baseline, candidate, candidate, baseline) with each pair
averaged to cancel machine drift; correctness is `ffmpeg` SSIM ≥ **0.95** against
the baseline MP4, plus a ±2 % duration check. `reward ≈ 1.0` means the candidate
matched v0.4.4.

**Leaderboard mapping** (`scripts/score_from_reward.py`, the stated source of
truth): `revideo-perf-opt` is in the `performance` category, so the gated score is
`correctness * 0.5`, or `0.5 + 0.5 * speedup` once correctness reaches 1.0. The
raw `reward.json["score"]` is deliberately more conservative than the leaderboard
number.

## 6. Harness validation — `harness_validation`, `is_model_score: false`

Ran the task's **unmodified** native verifier inside the pinned image against the
**unmodified** `/app/revideo` (a freshly created container, no agent, no patch).
Verifier files were byte-verified against the checkout before the run:

| File | sha256 |
|---|---|
| `test.sh` | `e77b3e48fec45685c15ceaaef79f1d26cbe35c509a12f3bdccdb5d362905ae90` |
| `compute_reward.py` | `b8ceb06a8d9aa8b73463cdc6e9910b106f706d6f47c7074f3fbceebc8dd54f00` |
| `prep_build.py` | `f09a7bfb32b529ef4a5c4c16c8e7a1185014eddc8ee4732dbcc2e70685ccb595` |
| `hidden-scenes.tar.gz` | `4cb4a332ad02782f07d374304f764cedfa2993f2ee0ae0ea5db82415af9991df` |

### 6.1 The build gate accepts a type-error build (host-independent finding)

Step 2 rebuilds the candidate. On the **unmodified** v0.4.2 tree,
`npm run build-lib -w packages/2d` **fails**:

```
src/lib/code/diff.ts(127,37): error TS2339: Property 'at' does not exist on type '[Subsequence]'.
src/lib/curves/getPathProfile.ts(103,44): error TS2550: Property 'at' does not exist on type 'Segment[]'.
  ... 9 more TS2339/TS2550
npm error Lifecycle script `build-lib` failed with error: code 2
npm error command sh -c tspc -p src/lib/tsconfig.build.json
```

The verifier nevertheless printed **`PASS: candidate build`**. Two things combine:

- `test.sh` appends `|| true` to every `npm run build`, so the exit code is
  discarded outright.
- The only real gate is `[ -f packages/<pkg>/<main> ]`, and `tspc` has no
  `noEmitOnError`, so it emits `lib/index.js` anyway. The file's mtime
  (`07:40:28`, container started `07:29:24`) confirms it was written **by this
  run**, not inherited from the image.

So the artifact is fresh but type-check-failed, and the suite benchmarks it
happily. These are `TS2550 ... Try changing the 'lib' compiler option to 'es2022'`
errors triggered by `prep_build.py` rewriting every tsconfig — it drops `types`
and repoints `typeRoots`. TypeScript type-checking is deterministic and
architecture-independent, so **this reproduces on an x86-64 host too**; it is not
an artifact of emulation.

Practical consequence for a real submission: a candidate whose `packages/2d`
changes fail to type-check will still be scored, on partially-emitted output.
Do not treat `PASS: candidate build` as evidence that the candidate compiles.

### 6.2 End-to-end execution result

The verifier ran to completion — **`EXIT=0`, 63.4 minutes wall clock**
(`total_time_ms: 3804241`) — and produced a well-formed `reward.json`.

| Step | Outcome |
|---|---|
| 1. Source-code scan | **PASS** — 1 576 files scanned, no verifier-internal references |
| 2. Rebuild candidate | **PASS** (see 6.1 — `@revideo/2d` type-check failed but emitted) |
| 3. Hidden test scenes | **8 copied** — the tarball's macOS `._` AppleDouble twins are skipped, because the bash glob `*.tsx` does not match dotfiles |
| 4. ABBA rendering | All 5 phases finished inside the 600 s cap — warmup, A1 349.4 s, B1 336.9 s, B2 404.3 s, A2 353.1 s — but **0/8 scenes succeeded in every phase** |
| 5. Merge | ran; both merged result sets empty |
| 6. SSIM correctness | ran; `Correctness: 0/0 scenes passed` |
| 7. `compute_reward.py` | wrote `reward.json` + `reward.txt` |

Every render failure was the same Chrome/QEMU gate — 32 of 32 scene attempts:

```
hidden_dense_grid: FAILED (41.3s) — Timed out after waiting 30000ms
hidden_video_long: FAILED (30.4s) — Timed out after 30000 ms while waiting for
                                    the WS endpoint URL to appear in stdout!
```

The second message is the smoking gun: Chrome never printed its DevTools
WebSocket endpoint because it had already core-dumped.

Reward produced:

```json
{"reward": 0.0, "score": 0.0, "geometric_mean_speedup": 0.0,
 "num_hidden_scenes": 0, "num_speedups_computed": 0,
 "hard_fail_reasons": ["baseline_results_missing", "candidate_results_missing"],
 "correctness_ok": true, "is_oracle": false, "total_time_ms": 3804241}
```

Fed through the official leaderboard script:

```bash
python3 outputs/e2_frontier_swe/frontier-swe/scripts/score_from_reward.py \
  --task revideo-perf-opt outputs/e2_frontier_swe/harness_validation/reward.json
# task: revideo-perf-opt (performance) | correctness 0.0 | speedup None | score 0.0
```

**What this establishes.** The container starts, the source scan runs,
`prep_build.py` + the six package builds run, hidden-scene injection runs, the
ABBA orchestration runs, the merge and SSIM steps run, `compute_reward.py` emits
a schema-correct `reward.json` and `reward.txt`, and `score_from_reward.py`
consumes it. **The plumbing is proven end to end.**

**What this does not establish.** Nothing about performance. The `0.0` is a
statement about this host, not about any candidate. It is recorded as
`harness_validation` with `is_model_score: false` and is **not** promoted to the
suite score, which stays `null`.

One thing to watch on a real host: with zero rendered scenes the verifier reports
`correctness_ok: true` and a `correctness` subscore of `1.0` ("PASS: 0/0
correct") — a vacuous pass. The top-level reward is still `0.0` because the
missing-results hard fails fire first, and `score_from_reward.py` also returns
`0.0`, so it is not exploitable. But `correctness_ok` alone must never be read as
evidence that anything rendered.

Artifacts: `harness_validation/reward.json`, `harness_validation/reward.txt`,
`harness_validation/{baseline,candidate}_output/benchmark_results.json`,
`harness_validation/correctness_results.json`,
`harness_validation/abba_{b1,b2,c1,c2}.log`, and the full 333-line transcript at
`logs/harness_validation_verifier.log`.

## 7. Image provenance — prior receipt's digest explained

The prior receipt recorded `sha256:291ff258…` as `native_container_digest`. That
is the **image config** digest, not the manifest digest. Both are now verified
against the registry:

| Kind | Digest |
|---|---|
| Manifest (pull-by-digest identifier) | `sha256:675d298493278f891a50e41ed31ffdb71590d4583dfc1987385a48d872f25103` |
| Image config | `sha256:291ff2584113385f988a5594c3d7979ac72a071fa77a1a3752c6766110cdd73b` |
| Platform | `linux/amd64`, 34 layers, 14.8 GB on disk |

No contradiction — just an imprecise label. The runner's `_container_digest`
reads `config.digest`, which is why preflight reports `291ff258…`.

## 8. Tests and CLI — re-run, green

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program python3 -m unittest -v \
  flagship.test_frontier_swe_eval flagship.test_pavlov_frontier_swe_eval_adapter
# Ran 25 tests — OK

PYTHONPATH=zvf-program python3 -m flagship.frontier_swe_eval --help              # exit 0
PYTHONPATH=zvf-program python3 -m flagship.pavlov_frontier_swe_eval_adapter --help  # exit 0

PYTHONPATH=zvf-program python3 -m flagship.frontier_swe_eval --mode preflight --smallest \
  --benchmark-repo outputs/e2_frontier_swe/frontier-swe \
  --out outputs/e2_frontier_swe/e2_frontier_swe_preflight_durable_20260809.json
# exit 2 -> status BLOCKED, score null, blocker: benchmark_license_missing
```

Logs: `logs/focused_tests_rerun.txt`, `logs/cli_help_runner_rerun.txt`,
`logs/cli_help_adapter_rerun.txt`, `logs/preflight_durable.txt`.

System Python 3.14.6 is sufficient — both modules and both test files are
pure-stdlib, so the prior temporary venv is no longer needed.

## 9. Cost and side effects

No Tinker calls, no W&B runs, no model API calls, no HF traffic, no pushes, no
commits. `$0.00` spent. The 14.8 GB image was reused, never re-pulled and never
deleted — verified present after cleanup. The shared docker mutex was **not**
taken: nothing over 2 GB was pulled or built.

Host disk dipped from 67 GiB to 23 GiB free mid-session while another lane held
`outputs/_setup/docker.lock`, then recovered to **47 GiB** — it never approached
the 15 GiB stop threshold. This lane's own footprint is the 1.4 GB clone plus a
container that has been removed. The verifier container ran ~63 min at
`--cpus 8 --memory 14g`; on a shared 8-CPU VM that is the one thing this lane did
that other lanes would have felt.

## 10. What remains, and the single next action

| # | Blocker | What clears it |
|---|---|---|
| 1 | `benchmark_license_missing` | A licence receipt or written maintainer authorization binding `Proximal-Labs/frontier-swe@422b9bb9`. Not obtainable locally. |
| 2 | `submission_missing` | A candidate `/app/revideo` tree per section 5 — needs an agent rollout with a paid model key (job.yaml budgets 72 000 s per attempt). |
| 3 | `arch_mismatch_amd64_image_on_arm64_host` | An x86-64 Linux runner. Not obtainable locally. |

**Single next action:** request written licence clearance for
`Proximal-Labs/frontier-swe@422b9bb95deb8efe436becb0ed3c44be23611e10`. It is the
cheapest gate, it blocks preflight before anything else runs, and unlike blocker 3
it does not require new hardware.
