# E14 — What Epoch AI's public FrontierMath sample transcripts actually contain

**Date:** 2026-08-09 · **Suite:** `frontiermath_eval`

> **THESE ARE REPRESENTATIVE PUBLIC SAMPLES, NOT THE BENCHMARK.**
> Everything below is measured from `sample_question_transcripts.zip`, which
> Epoch AI publishes as an illustration. The FrontierMath benchmark itself is a
> private held-out problem set that is **never** distributed to third parties.
> No number on this page is a FrontierMath score, an estimate of one, or a
> bound on one. GSM8K, MATH-500, AIME, and MathArena are **not** substitutes.
>
> **Scale check:** this archive covers the **original 5 public problems from
> the 2024 paper**. Epoch currently publishes **12** sample problems (10 from
> Tiers 1–3, 2 from Tier 4), and the private set is **338 problems** post-v2
> (295 Tiers 1–3 + 43 Tier 4). So this corpus is 5/338 ≈ **1.5%** of the
> benchmark, and an outdated slice at that.
>
> **Licence: not established** for this archive specifically. Epoch's site is
> broadly CC-BY but carves out benchmark questions and answers as the property
> of their respective creators. Do not redistribute.

All facts here were computed from the files, not read off the paper.

| Artifact | Value |
|---|---|
| Source | `https://epoch.ai/files/sample_question_transcripts.zip` |
| Archive SHA-256 | `7bdf3231086cc7de000ea57380c36a64abdf2644f1111b044d1bab0b383b0ff8` (750,459 bytes) |
| Extracted to | `outputs/e14_frontiermath/public_samples/sample_question_transcripts/` |
| Corpus SHA-256 (150 files) | `9d618deb664cfce7de7721f1ce4c3199f9a90860c11efbc4cc6292924dd04345` |
| Per-file hashes + full facts | `outputs/e14_frontiermath/public_sample_manifest.json` |

## 1. The grid is exactly 6 models × 5 problems × 5 runs = 150

Every cell has exactly 5 runs; there are no gaps and no extras.

- **Models (25 transcripts each):** `claude-3-5-sonnet-20241022`,
  `gemini-1.5-pro-002`, `gpt-4o-2024-08-06`, `grok-beta`, `o1-mini`, `o1-preview`.
- **Problem tokens (30 transcripts each):** `ALL3`, `CWA2`, `CWD31`, `RAP1`, `TIK2`.
- **Run indices:** 1–5.

Filenames encode the triple: `<model>_<PROBLEM>_run-<n>.jsonl`. The problem
tokens are opaque IDs; they are not resolvable to Epoch's internal numbering
from the public data.

## 2. Transcript schema — two fields, nothing else

Each `.jsonl` line is a JSON object with **exactly** the keys `role` and
`content`, both strings. Verified across all 850 messages in all 150 files:

- roles are only `user` and `assistant`;
- roles **strictly alternate** in every transcript;
- every transcript **starts** with `user` (the task prompt) and **ends** with
  `assistant`;
- 425 user turns, 425 assistant turns; 2–10 messages per transcript (median 6).

There is no `id`, no `problem_id`, no `answer`, no `label`, no `score`, no
`reward`, and no metadata envelope of any kind.

## 3. There is no ground truth and no grader verdict — anywhere

This is the load-bearing finding and it is a positive check, not an assumption.
A verdict could only reach the transcript by one of two routes, and both are
closed:

1. **A structured field.** Impossible — the schema admits only `role` and
   `content`. The validator fails loudly if any other key appears.
2. **Harness text.** The non-first `user` turns are written by Epoch's harness.
   All 275 of them classify exhaustively into three shapes, none of which
   reveals correctness:

| Harness turn shape | Count |
|---|---|
| `Results from executing code block N:` … stdout/stderr/timeout | 265 |
| `Final answer failed: error in stderr: Traceback …` | 6 |
| `Final answer failed: timed out` | 4 |
| unclassified | **0** |

`Final answer failed` is an **execution** failure, not a grading verdict — the
submitted script crashed or exceeded the 20-second limit. The harness never
tells the model whether its answer was right. Every harness turn ends with the
same nudge to keep working.

Consequence: **the public corpus is structurally ungradable.** Even if it were
gradable, five published sample problems would not be the benchmark.

## 4. What a FrontierMath answer looks like

The task prompt is byte-identical across all 150 transcripts (single preamble
hash `7f376fef…`), so the answer contract is unambiguous:

- The model works in a **code-execution loop**: it emits a ```` ``` ````-fenced
  Python block, the harness runs it and returns stdout/stderr plus a
  timeout flag. **20-second wall clock** per execution. Each block must be
  self-contained.
- To submit, the model emits a final block that **must contain the literal
  comment `# This is the final answer`** — the automated grader greps for
  exactly this string to locate the submission.
- That block must `pickle.dump` the answer to **`final_answer.p`** in the
  script's directory. The grader unpickles that file and takes the object as
  the answer.
- The prompt states a **required return type** and demands exact equality:
  approximate answers score nothing, and a `sympy` object is not
  interchangeable with a float. Of the five sample problems, all five declare
  an integer type (four `Python integer`, one `Integer`).

So a "FrontierMath answer" is **a pickled Python object produced by executing
model-authored code**, not a string match. Grading is exact-value comparison
against a reference the public data does not contain.

Coverage of the submission contract in the corpus:

- 149 / 150 transcripts contain at least one final-answer block
  (`o1-mini_TIK2_run-4.jsonl` is the sole exception — it never submitted).
- 165 final-answer blocks total; 11 transcripts emitted more than one
  (up to 5), typically after a `Final answer failed` execution error.
- 509 code blocks total across all assistant turns.

## 5. Transcript-structure statistics (descriptive, NOT performance)

Turn counts measure how much the loop was exercised. They say nothing about
correctness, because correctness is not observable here.

| Model | median msgs | max msgs | code blocks | final-answer blocks |
|---|---|---|---|---|
| claude-3-5-sonnet-20241022 | 8 | 10 | 143 | 29 |
| gemini-1.5-pro-002 | 2 | 10 | 55 | 31 |
| gpt-4o-2024-08-06 | 6 | 10 | 78 | 25 |
| grok-beta | 10 | 10 | 121 | 29 |
| o1-mini | 6 | 8 | 74 | 24 |
| o1-preview | 2 | 4 | 38 | 27 |

The 10-message ceiling looks like a harness turn cap rather than a model
property.

## 6. The five published sample problems

These are the five sample problems released with the 2024 paper — **not** the
current 12 published samples, and not a random draw from the private set. Each
statement is byte-identical across its 30 transcripts (one canonical LaTeX
statement per token), so these are faithfully reproduced prompts.

| Token | Statement SHA-256 (12) | Subject | Asked for |
|---|---|---|---|
| `ALL3` | `062d0961dd55` | Analytic number theory — densities of primes by multiplicative order of 2 vs 3 | `floor(1e7 · d_∞)` |
| `CWA2` | `9a34f51d19cc` | Arithmetic geometry — points on `x³y+y³z+z³x=0` over `F_{5^18}` up to scaling | integer count |
| `CWD31` | `0ed9c1fa9f31` | p-adic analysis — smallest prime `p ≡ 4 mod 7` extending a linear recurrence continuously to `Z_p` | integer |
| `RAP1` | `b816bc50d0b7` | Representation theory — orbits of `GL(1000,C)` on commuting/anticommuting involution 4-tuples | integer |
| `TIK2` | `eb3dd63ed6fb` | Algebraic geometry — degree-19 odd monic polynomial with reducible `{p(x)=p(y)}`, compute `p(19)` | integer |

## 7. Incidental provenance evidence

Tracebacks in `Final answer failed` turns leak Epoch's own harness paths:

```
/Users/js/epoch/epochmath/experiment.py
/Users/js/epoch/epochmath/.venv/lib/python3.12/site-packages/{sympy,numpy}/...
```

This confirms the transcripts are genuine Epoch runs, names the harness module
(`epochmath`, entry point `experiment.py`), and shows the execution sandbox is
a Python 3.12 venv with at least `sympy` and `numpy` available. It does not
expose the grader, the reference answers, or the problem set.

## 8. Reproduce

```bash
PYTHONPATH=zvf-program python3 -m unittest flagship.test_e14_frontiermath_public_samples -v

PYTHONPATH=zvf-program python3 -m flagship.e14_frontiermath_public_samples \
  --samples-dir outputs/e14_frontiermath/public_samples/sample_question_transcripts \
  --archive outputs/e14_frontiermath/frontiermath_public_samples.zip \
  --manifest-out outputs/e14_frontiermath/public_sample_manifest.json \
  --recorded-at 2026-08-09 --summary-only
```

The module raises `ScoreProhibited` from `compute_frontiermath_score()` on every
call, and `assert_receipt_emits_no_score()` rejects any receipt whose status is
not `BLOCKED` or whose score is not `null`.
