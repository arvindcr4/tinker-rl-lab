# Public analysis toolkit

This is a **changed, function-only derivative** of retained scientific sources.
It is not a byte-identical frozen source release. The package is Python 3.10+
standard-library code; no installation, model, network, or account access is
needed. Source-fingerprint checks are independent of Python-version-specific AST
metadata. The accompanying tests use fabricated text, identities and records. Two additional
tests check arithmetic for published aggregate counts (11/64, 3/64 and 87/4,096).
No synthetic test or calculation is new empirical evidence.

## Run from the repository root

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_public_analysis.py -v
python -m research.public_analysis parse '#### 1/2' --policy 2
python -m research.public_analysis counts 0 20
```

Use `parse -` to read supplied text from standard input. Only the two documented
terminal markers are removed by this CLI. The parser functions themselves do
not perform that preprocessing. `counts` calculates the original question-level
binomial working-model interval; its output does not authenticate the input.

## Included scientific contracts

- `parser_v1`, `parser_v2`, `parser_v3`: unchanged strict numeric grammars,
  normalization, box extraction and complete parse-result objects. V3 remains a
  prospective candidate, not a deployed or independently validated policy.
  V1's narrower ambiguity behavior is preserved intentionally; versions are not
  silently substituted for one another.
- `rescue_statistics`: original Wilson intervals, homogeneous-group summaries,
  and beta-binomial rescue predictions. `statistics`: the later numerically
  stable intervals, fixed 8-initial/24-fresh prediction, quality rates and full
  pure `analyze_validated` arithmetic. Historical numerical quirks are preserved.
- `selection`: complete-screen selection and checking, uncapped strict success,
  quality gates and selected-cohort intervals. The fixed scientific design is
  4,096 ordered screen questions, eight draws each, ranks 640 through 4,735,
  selecting at most the first 64 eligible questions. These are design constants,
  not an instruction to launch a study or evidence that a study occurred.
- `reference_grader`: unchanged permissive heuristic comparator. It is not
  semantic ground truth and must not replace the registered strict parser.
- `parser_review`: duplicate-key/nonfinite-JSON rejection, blind changed-output
  packets, review validation, literal wrapper labels and unchanged decision
  arithmetic. All examples of request, prompt, case and reviewer identities in
  the tests are explicitly synthetic. These generic input field names contain
  no account-bound identifiers. Review packets can contain caller-supplied text;
  this package only handles them locally and does not publish or upload them.

`statistics.analyze_validated` and `parser_review.decide` are lower-level pure
functions. As in their original source, they rely on validated inputs: the
former assumes complete ordered 32-draw rows with binary rewards; the latter
requires the outputs of packet and review validation. Calling either directly
on arbitrary records does not establish completeness, independence, provenance,
or a legitimate pass. A parser review's decision string is a conditional
calculation, not evidence of prospective execution. Synthetic tests exercise
these contracts and explicitly distinguish complete object preservation from
matching status or value alone.

## Exactness and provenance

`PROVENANCE.json` lists the SHA-256 of each retained original **code file** under
neutral source labels, and an exact source-text digest of each copied definition. Selected
functions and constants were copied verbatim; module documentation, imports,
version labels, CLI and tests are newly assembled public derivatives. The
original whole-file hash is never claimed as the hash of a derivative.

The public test checks those copied-definition source-text digests and scientific behavior against
synthetic fixtures. The preparation check additionally compared them directly
with the retained source. Hashes cover the exact UTF-8 source span for each
definition (including its internal whitespace and comments); AST metadata only
locates that span and is not serialized into the digest. A regression check
emulates the absence of Python 3.12-specific empty type-parameter fields to
verify compatibility with Python 3.11. The source hashes are commitments only: the omitted
original files are not supplied here, so readers cannot independently establish
whole-file identity or pre-execution freezing from this package. No data hash,
private manifest, provider inventory, private URL or operational resource ID is
included. Source hashes do not prove when code existed or that it was executed.

## Reproducibility limits

This package reproduces parsing and the included mathematical/selection/review
contracts on supplied inputs. It does **not** reproduce the full empirical study.
Raw completions, complete cohorts, provenance/continuation receipts, execution
wrappers, budget/account machinery and archive-dependent tests are omitted.
The paired bootstrap's original pinned NumPy generator/index matrix and raw
question-level input are not included; no alternative bootstrap is substituted.
The aggregate-count tests reproduce stored fractions and Wilson intervals only.
They do not regenerate those event counts from raw completions.

Selection remains conditional on parser-defined, uncapped initial failures.
Intervals describe a question-level binomial working model; arbitrary dependence
or deterministic cohort selection need not have population coverage. Caps and
parse failures must not be discarded from registered denominators. A parser
change is not a semantic-correctness, training, causal, or general-capability
result. Post-outcome clean subsets and extra prefixes are descriptive only.

The repository's Apache-2.0 license applies. Existing attribution is unchanged;
this directory prominently marks its modified/derivative status.
