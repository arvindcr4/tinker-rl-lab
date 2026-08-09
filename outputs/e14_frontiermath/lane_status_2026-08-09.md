# E14 — `frontiermath_eval` lane status

**Date:** 2026-08-09 · **Status: BLOCKED (terminal) · score: `null`**

## The finding that decides this lane

**FrontierMath is never released, and it never will be.** Epoch AI's own page
states they cannot share Tiers 1–4 questions and answers without OpenAI's
written permission. There is no download, no gated Hugging Face repo, no
data-use agreement, and no request form that yields the problems. **There is
also no official FrontierMath dataset on Hugging Face or GitHub** — every repo
claiming to be one is a solver or a reconstruction and cannot produce a
comparable score.

A FrontierMath number is obtained **only by Epoch AI running the evaluation
themselves** on a model they can reach. So this lane can never be local. That is
a property of the benchmark, not a gap in our setup.

`FrontierMath` now names **two products** with different regimes — conflating
them produces a wrong plan:

| | Tiers 1–4 (classic) | Open Problems (July 2026) |
|---|---|---|
| Owner | OpenAI-commissioned | **Epoch alone** (Schmidt Sciences funded) |
| Size | **338** post-v2 (295 T1–3 + 43 T4) | not established |
| Buy in? | No | **Yes — verifiers purchasable, uniform/non-exclusive** |
| Contact | `math_evals@epoch.ai` | `math@epoch.ai` |

Post-v2 sizes are authoritative: the correction pass revised 123 Tier 1–3 and 12
Tier 4 problems after errors were found in 42% of problems. **The older 350
(300+50) figure is superseded.** Epoch has *not* restated the holdout counts (53
withheld solutions, 20 withheld Tier 4 problems) against v2 — record that as
**not established**.

## What now runs

Everything here is **harness/parser validation** on public sample transcripts.
`is_model_score: false`. None of it is a benchmark result.

```bash
# 51 unit tests — schema, hashing, and the score-refusal guards
PYTHONPATH=zvf-program python3 -m unittest flagship.test_e14_frontiermath_public_samples -v

# Characterize + immutably hash the 150 public transcripts
PYTHONPATH=zvf-program python3 -m flagship.e14_frontiermath_public_samples \
  --samples-dir outputs/e14_frontiermath/public_samples/sample_question_transcripts \
  --archive outputs/e14_frontiermath/frontiermath_public_samples.zip \
  --manifest-out outputs/e14_frontiermath/public_sample_manifest.json \
  --recorded-at 2026-08-09 --summary-only
```

Both exit 0. 51/51 pass. E10's adapter tests (43) still pass — no interference.

## What the transcripts actually are

Measured from the files, not the paper. Full detail in
`PUBLIC_SAMPLE_CHARACTERIZATION.md`.

- **Exactly 6 models × 5 problems × 5 runs = 150.** No gaps, no extras.
- **Schema is two fields.** Every line is `{"role", "content"}`, both strings.
  Roles strictly alternate, start `user`, end `assistant`. No IDs, no metadata.
- **No ground truth and no grader verdict anywhere.** All 275 harness turns
  classify exhaustively into three shapes — 265 code-execution results, 6
  `Final answer failed: …stderr`, 4 `Final answer failed: timed out` — and
  **zero** unclassified. Those failures are *execution* errors, not grades. The
  model is never told whether it was right. **The corpus is structurally
  ungradable.**
- **An answer is a pickled Python object.** The model works in a 20-second
  code-execution loop and submits a block containing the literal comment
  `# This is the final answer` that `pickle.dump`s to `final_answer.p`. The
  grader unpickles that file and compares exactly — approximate is worth
  nothing. 149/150 transcripts submitted; `o1-mini_TIK2_run-4.jsonl` never did.
- **Scale check: this archive is the ORIGINAL 5 public problems from the 2024
  paper** — 5/338 ≈ **1.5%** of the private set, and outdated relative to the
  **12** problems Epoch publishes today.
- **Licence for the archive: not established.** Site-wide CC-BY carves out
  benchmark questions and answers as their creators' property. Do not
  redistribute. *(This corrects the flat CC-BY claim in the earlier
  `blocked_receipt.json`.)*

## Guard rails built

`zvf-program/flagship/e14_frontiermath_public_samples.py` is fail-closed by
construction:

- `compute_frontiermath_score()` **always raises `ScoreProhibited`** — there is
  no code path to a score.
- `assert_receipt_emits_no_score()` rejects any receipt whose status is not
  `BLOCKED`, whose score is not `null`, that carries `measured_metrics`, that
  sets `related_benchmark_substitution`, or that drops the sample label. It runs
  inside the receipt builder, so a bad receipt cannot be returned.
- A test asserts the manifest contains no `accuracy` / `pass_rate` / `correct` /
  `reward` / `pass@1` key.
- Every artifact carries `REPRESENTATIVE_PUBLIC_SAMPLES_NOT_THE_BENCHMARK` plus
  the full disclaimer, and `public_samples/READ_THIS_FIRST_NOT_THE_BENCHMARK.md`
  warns anyone browsing the directory.

Immutability: archive `sha256 7bdf3231…b0ff8`; 150-file corpus digest
`sha256 9d618deb…4345` (order-independent, per-file hashes in the manifest).

## What remains

Nothing local. The blocker is external and terminal.

## The single next action

Open `ACCESS_REQUEST_2026-08-09.md`, pick Tiers 1–4 (§3) or Open Problems (§4),
fill the four bracketed fields, and **send the email yourself** to
`math_evals@epoch.ai` or `math@epoch.ai`. It is written as a hosted-evaluation /
submission request, **not** a data request — asking for the dataset would be
asking for something Epoch is contractually unable to give.

This lane created no accounts, submitted no forms, and contacted no one.
No paid calls: Tinker 0, model API 0, W&B 0, HF pushes 0, $0.00.

## Files

| Path | What |
|---|---|
| `receipt_2026-08-09.json` | BLOCKED receipt, `score: null`, 6 blockers, full access path |
| `ACCESS_REQUEST_2026-08-09.md` | Two ready-to-send engagement emails + contacts |
| `PUBLIC_SAMPLE_CHARACTERIZATION.md` | What the 150 transcripts contain, measured |
| `public_sample_manifest.json` | Per-file hashes, grid, harness taxonomy, gradability |
| `public_samples/READ_THIS_FIRST_NOT_THE_BENCHMARK.md` | In-directory warning |
| `zvf-program/flagship/e14_frontiermath_public_samples.py` | Parser, hasher, fail-closed guards |
| `zvf-program/flagship/test_e14_frontiermath_public_samples.py` | 51 tests |

`blocked_receipt.json` is the earlier metadata-only receipt; it records a stale
checkout path and an unqualified CC-BY claim. `receipt_2026-08-09.json`
supersedes it.
