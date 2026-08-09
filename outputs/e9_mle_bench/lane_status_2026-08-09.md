# E9 — `mle_bench_eval` (OpenAI MLE-bench, 75 Kaggle competitions)

Date: 2026-08-09. Status: **PARTIAL** — the runner, the split/verifier binding,
the real competition split and the grading harness (host and container) all
execute. **The harness reproduces upstream's recorded reference score exactly.**
Remaining blockers are the canonical agent image, a model submission artifact,
and a contamination receipt. Suite score is `null` and no model was run.

Receipt: `outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json`

## What now runs

**Venv rebuilt 2026-08-09** after it was deleted in a disk emergency:
`uv venv --python 3.11 outputs/_setup/venvs/e9` then
`uv pip install -e outputs/e9_mle_bench/mle-bench-source`. Cold cache, ~2 min,
1597 MB. `mlebench --help` works; Python 3.11.15, sklearn 1.9.0, pandas 3.0.5,
diskcache 5.6.3 — same stack as the first pass, and the 41 pre-existing tests
plus the new ones still pass against it.

```bash
V=outputs/_setup/venvs/e9/bin

# 1. survey all 75 competitions by recorded download size
$V/python zvf-program/flagship/mle_bench_eval.py survey --top 10

# 2. prove Kaggle rules are accepted (download endpoint; exit 0 == accepted)
$V/python zvf-program/flagship/mle_bench_eval.py check-rules

# 3. prepare the real split (1.8 MB, ~4 s, verifies two checksum manifests)
cd outputs/e9_mle_bench/mle-bench-source && $V/mlebench prepare \
    -c spooky-author-identification --data-dir ../data && cd -

# 3b. drive the official grader against the REAL split
$V/python zvf-program/flagship/mle_bench_eval.py harness-validate \
    --out outputs/e9_mle_bench/evidence/harness_validation.json

# 4. fail-closed receipt
$V/python zvf-program/flagship/mle_bench_eval.py receipt \
    --harness-json outputs/e9_mle_bench/evidence/harness_validation.json \
    --rule-probe-json outputs/e9_mle_bench/evidence/kaggle_rules_check.json \
    --observed-at 2026-08-09 --out outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json

# 5. unit tests — 53 pass
PYTHONPATH=zvf-program $V/python -m unittest -q flagship.test_mle_bench_eval

# 6. the same grading, inside the container that was built this session
docker run --rm --platform linux/amd64 \
  -v "$PWD/outputs/e9_mle_bench/data:/private/data:ro" \
  --entrypoint /bin/bash mlebench-env:verifier-noheavy -lc \
  '/opt/conda/bin/conda run -n mleb mlebench grade-sample \
     /private/data/spooky-author-identification/prepared/public/sample_submission.csv \
     spooky-author-identification --data-dir /private/data'
```

New files (mine): `zvf-program/flagship/mle_bench_eval.py`,
`zvf-program/flagship/test_mle_bench_eval.py`.

## 1. Smallest competition — measured, not guessed

The upstream repository ships its own size table at
`experiments/competition_categories.csv`
(sha256 `5b6967e944e6b105f54f943cd13042158f49fec5206c0ff536d16d42cf39b634`,
75 rows, one per competition in `experiments/splits/split75.txt`). Ranking every
row by `dataset_size_GB`:

| Rank | Competition | GB | MB | Complexity |
|---|---|---|---|---|
| 1 | **spooky-author-identification** | **0.00190** | **1.95** | Low |
| 2 | detecting-insults-in-social-commentary | 0.00200 | 2.05 | Low |
| 3 | us-patent-phrase-to-phrase-matching | 0.00214 | 2.19 | Medium |
| 4 | random-acts-of-pizza | 0.00300 | 3.07 | Low |
| 5 | tweet-sentiment-extraction | 0.00300 | 3.07 | Medium |

Total across all 75: **3283.69 GB**, matching the README's "3.3TB for the full
set". Independent cross-check via the Kaggle file-listing API for the pick
(`evidence/kaggle_files_spooky_author_identification.csv`):
`sample_submission.zip` 29 KB + `test.zip` 538 KB + `train.zip` 1 MB — the same
order of magnitude at the API's rounded precision.

Full ranking: `outputs/e9_mle_bench/evidence/competition_size_survey.json`.

## 2. `mlebench prepare` — SUCCEEDED against the real split

Rules for `spooky-author-identification` were accepted by a human on 2026-08-09
and verified against the **download** endpoint (`accepted: true`,
`download_endpoint_ok: true`, exit 0 — `evidence/kaggle_rules_check.json`).

```
$ mlebench prepare -c spooky-author-identification --data-dir outputs/e9_mle_bench/data
Downloading spooky-author-identification.zip  1.81M [00:01, 1.18MB/s]
Checksum for `spooky-author-identification.zip` matches the expected checksum.
Preparing the dataset using `prepare` from `.../spooky-author-identification/prepare.py`
Data for competition `spooky-author-identification` prepared successfully.
Checksums for files in `.../spooky-author-identification` match the expected checksums.
```

EXIT=0 in 4 s. **Both checksum gates matched**: the downloaded zip against the
pinned `checksums.yaml`, and then every prepared public/private file against the
same manifest. The local split is therefore bit-identical to the official one —
this is provenance, not just "it ran".

- private answers `prepared/private/test.csv` sha256 `2cf7dc57…`
- sample submission `prepared/public/sample_submission.csv` sha256 `5a9d7015…`

### The false-positive check — keep this, it cost a round trip

An earlier "verified, rules accepted" report was wrong, and the failure mode is
easy to repeat. **`kaggle competitions files` is not an acceptance check.** That
metadata endpoint returns file names and byte sizes for competitions whose rules
have never been accepted; only the *download* endpoint is gated. During the
blocked period the two endpoints disagreed, same account, seconds apart
(`evidence/kaggle_endpoint_contrast.log`):

```
## kaggle competitions files    -> sample_submission.zip,29KB / test.zip,538KB / train.zip,1MB
## kaggle competitions download -> 403 Forbidden - You must accept this competition's rules
```

That listing was byte-identical to one captured before any acceptance attempt,
so it could not have evidenced a state change.

**Use this instead** — it probes the download endpoint and exits 0 only when
bytes are actually served:

```bash
outputs/_setup/venvs/e9/bin/python zvf-program/flagship/mle_bench_eval.py check-rules
```

### The other 74 competitions — a structural finding

Kaggle rule acceptance is **per competition** and non-automatable. Exactly one of
75 is accepted. A full MLE-bench run needs **74 further manual acceptances** by a
signed-in human, each on its own competition page. That is a reproducibility
property of the benchmark itself, not a defect of this lane, and it is recorded
in the receipt under `competition_binding.other_competitions_rule_state`.

The ten smallest competitions were probed during the blocked period; the other
nine remain un-accepted (`evidence/kaggle_rule_acceptance_probe.json`).

## 3. Container — verifier-only image BUILT; canonical `mlebench-env` not built

**Built and working:**

```bash
mkdir outputs/_setup/docker.lock          # mutex taken before the build
cd outputs/e9_mle_bench/mle-bench-source
docker build --platform linux/amd64 --build-arg INSTALL_HEAVY_DEPENDENCIES=false \
  -t mlebench-env:verifier-noheavy -f environment/Dockerfile .
```

- `sha256:59b8e1c643c5b0959f4bdc6a06bb083cf01e3be9dd1d4bf6a744871251a2cc70`
- **2.68 GB**, 18 layers, amd64/linux (qemu emulation on this arm64 host)
- Build time **~811 s of foreground build across two resumed passes**; the first
  pass reached stage 10/18 and the cached resume finished the remaining 8 in
  271 s. (Three earlier detached attempts were killed by the environment's
  background reaper at 72 s and 153 s of build time and produced nothing.)
- Container smoke test: the official CLI runs inside it and the grading-server
  deps import (`evidence/container_smoke.log`).
- **The containerised verifier reproduces the host grading result byte for
  byte** — same score 0.0, same thresholds 0.16506 / 0.26996 / 0.29381 /
  0.418785 (`evidence/container_grade_sample.log`).

**This is not `mlebench-env`.** `INSTALL_HEAVY_DEPENDENCIES=false` skips the
92-line requirements file, `tensorflow[and-cuda]==2.17` and `torch==2.2.0`, so it
contains the grading server and the `mlebench` package but no agent ML stack: it
can grade a submission, it cannot host an agent. The receipt records it under
`environment.verifier_only_variant` with `satisfies_container_gate: false`, and
`container_image_digest_present` stays **failed**.

The canonical build was not attempted, on measured grounds. Wheel sizes from
PyPI, before unpacking and before the other 87 requirements: frameworks
**1.28 GB** (torch 720 MB, tensorflow 573 MB) plus CUDA runtime deps **2.74 GB**
(cudnn 745 MB, cublas 554 MB, …) = **4.0 GB of wheels**, which unpack to roughly
2–2.5×. A 15–25 GB image, built under qemu emulation, on a shared sparse disk
that never shrinks.

### Disk cost — I overshot my estimate, please read

Host free space went **35 GiB → 18 GiB** during this build (the concurrent E1
pull accounts for part of it). My own footprint inside the VM is the 2.68 GB
image plus **9.9 GB of buildx cache** — emulated amd64 layers are expensive. I
estimated 2–3 GB; the true cost was ~12.6 GB. Per the coordinator's note this is
not recoverable on the host.

I did **not** prune: `docker builder prune` is shared and would delete other
lanes' cache. `docker system df` currently reports 9.919 GB of reclaimable build
cache — freeing it would not shrink the host file but would let other lanes
reuse that VM space without growing it further. That is the coordinator's call,
not mine.

## 4. Harness validation — PASS on the REAL split, and it is not a model score

`label: harness_validation`, `is_model_score: false`, `suite_score: null`,
`data_provenance: official_prepared_competition_data`. No model ran. Grading a
gold answer proves the grader; it is not an MLE-bench score.

Everything in the path is upstream: the `multi-class-log-loss` grader, the
medal thresholds, `CompetitionReport`, and the official `mlebench grade-sample`
CLI. The leaderboard is the real Kaggle leaderboard shipped in the repo (1242
teams, sha256 `1087afc6…`). The answers are now the **real prepared split**, not
a fixture.

| Case | Score | Medal | Valid |
|---|---|---|---|
| gold answers (== `private/test.csv`) | **0.0** | gold | yes |
| official `sample_submission.csv` | **1.08468** | none | yes |
| negative control, rows not summing to 1 | `null` | none | **rejected** |

Thresholds: gold 0.16506 / silver 0.26996 / bronze 0.29381 / median 0.418785,
lower-is-better. All seven checks pass.

### Upstream reproduction — exact

Upstream's `tests/constants.py` records `spooky-author-identification: 1.08468`
as the score its own sample submission achieves on the real split. This lane
observed:

```json
{"upstream_expected_score": 1.08468, "observed_score": 1.08468,
 "absolute_delta": 0.0, "relative_delta": 0.0, "matches_upstream": true}
```

**Delta 0.0.** Scores are rounded to 5 decimals by `grade_helpers`, so exact
equality at that precision is the correct bar and it is met. This is a
substantially stronger result than the earlier fixture run, which produced
1.0931 against synthetic labels — I called that a ~1% sanity signal rather than a
claim, and that caution was warranted: the fixture number was never the reference
value, and the real number lands on it exactly.

Reproducing 1.08468 means the download, the upstream preparer's 90/10 split with
`random_state=0`, the log-loss implementation and the leaderboard ranking all
agree with the reference implementation end to end.

The same 1.08468 was then reproduced **inside the built container** with the real
data mounted at `/private/data`, so the verifier is not host-dependent.

Evidence: `evidence/harness_validation.json`,
`evidence/mlebench_grade_sample_real.log`,
`evidence/container_grade_sample.log`.

The negative control is derived from the real sample submission and written to
`outputs/e9_mle_bench/harness_controls/`, deliberately **outside** the prepared
directory so the official data keeps matching its recorded checksums. The old
synthetic fixture in `harness_fixture_data/` is retained only as the
blocked-path artefact; it is no longer what the harness grades.

**Prerequisite discovered earlier:** the repo's 322 Git-LFS files, including
every `leaderboard.csv`, were unresolved pointers. Grading is impossible without
them (`get_leaderboard` would parse a 3-line pointer file). Fixed with
`git lfs install --local && git lfs pull` (39 MB). The runner's
`verifier_identity` now detects an unresolved pointer and fails the verifier gate
rather than grading against garbage.

## 5. Runner and receipt

`zvf-program/flagship/mle_bench_eval.py` — four subcommands (`survey`,
`harness-validate`, `check-rules`, `receipt`), 53 unit tests. The receipt binds:

- **Immutable revision** — `openai/mle-bench@507f92e1138bb6e40dac5c6ee7a6758e6424bf97`
- **Task-ID hashes** — sorted, newline-joined, terminal newline; eval split
  `440ecb5c6a13bc54e0d671e19eb532ee33c8305cca862e4d5b2af49f3d44f85b`
- **Split manifest** — per-file sha256 + count + task-ID hash for split75 and
  low/medium/high (22/38/15). Per-*sample* hashes are explicitly `null`: an
  MLE-bench task ID is a competition ID, and per-row IDs exist only after
  `prepare`.
- **Container digest** — `docker image inspect mlebench-env`, currently absent
- **Verifier identity** — grader name, `grade_fn` dotted path, sha256 of
  `grade.py`, `prepare.py`, `checksums.yaml`, the leaderboard, and
  `grading_server.py`

Fail-closed: `status` is `READY` only if all eight gates pass; a missing gate key
counts as a failure. Current state:

| Gate | |
|---|---|
| upstream_revision_pinned | pass |
| split_manifest_resolved | pass |
| verifier_identity_resolved | pass |
| dataset_license_accepted | **pass** — rules accepted 2026-08-09, verified on the download endpoint |
| competition_data_prepared | **pass** — real split, both checksum gates matched |
| container_image_digest_present | **fail** — verifier-only variant built, canonical `mlebench-env` not |
| model_submission_artifact_present | **fail** |
| contamination_disjointness_receipt | **fail** |

→ `status: BLOCKED`, `score: null`, `is_model_score: false`. Two gates flipped
this pass; three remain.

## 6. Licence position (honest)

- **Repository code: MIT.** `LICENSE` at the pinned revision, sha256
  `8a44e3d5…`.
- **The 75 competition datasets are NOT covered by it.** The licence explicitly
  excludes external datasets downloaded while using the package. Each
  competition carries its own Kaggle competition rules, accepted per competition
  by a signed-in human. An agent cannot accept them, and no blanket acceptance
  exists.
- The `leaderboard.csv` files *are* redistributed in the repository under MIT and
  are what the medal thresholds come from — so the verifier side is licence-clean
  even while the data side is not.
- Recorded in the receipt as `license_position` with
  `acceptance_is_automatable: false`.

## Single next action

Three gates remain, and none is a download:

1. **`container_image_digest_present`** — needs the canonical `mlebench-env`
   (15–25 GB, hours under qemu emulation). Deliberately not built; the
   verifier-only variant does not and should not satisfy this gate.
2. **`model_submission_artifact_present`** — needs an agent run inside that
   container with a model API key. Outside this lane's cost boundary.
3. **`contamination_disjointness_receipt`** — needs a training-corpus task-ID
   manifest to diff against the eval-split hash
   `440ecb5c6a13bc54e0d671e19eb532ee33c8305cca862e4d5b2af49f3d44f85b`. This is a
   paperwork gate, not a compute one, and it is the cheapest of the three.

Scaling beyond this one competition is gated on **74 further human rule
acceptances**, one per competition page — see section 2.
