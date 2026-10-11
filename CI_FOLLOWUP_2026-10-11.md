# CI execution follow-up — 11 October 2026

The [first post-integration workflow](https://github.com/arvindcr4/tinker-rl-lab/actions/runs/38127725451) at `8948b413303fde9477666f93d7c41cf4cd6d0a90` executed all 15 jobs: nine succeeded and six failed. The old account-billing block did not stop this run. All three public-release Python lanes, package, lint, secret scan, environment and documentation checks passed.

Four software/CI setup problems were exposed and corrected:

- The TRL script template used conversion fields spanning a newline after `!r`, which Python 3.11 cannot parse. Precompute those values and optional template strings. Generated DeepSpeed paths use `repr` to preserve quotes safely.
- The core/dev environment deliberately omits TRL, but mypy treated its guarded optional version import as required. Add the same narrowly scoped missing-import setting used for other optional frameworks; runtime ImportError handling is unchanged.
- Number-audit checks verify historical Git tags. Core and quality jobs now fetch complete history/tags instead of a shallow tip. Claims and expected hashes are unchanged.
- Recomputed float serialization differs in the last bits across Python 3.11/3.12. Compare parsed float fields at relative tolerance 1e-12 / absolute tolerance 1e-15, while keeping structure, row order, counts, labels and hashes exact. Regression cases reject material changes and nonfinite values. Original source-byte/hash and historical-file preservation assertions remain unchanged; no stored result was regenerated.

The strict provenance job also failed because `platform_hybrid/registry/provenance/campaign.provenance.json` lacks a confidence interval or standard deviation. This is an actual evidence gap, not an infrastructure failure. The strict gate is unchanged and must remain failed until supported uncertainty evidence is supplied. Do not replace it with an invented value or reinterpret a warning as a pass.

The October 11 thesis export remains frozen with its original validation receipt. This code follow-up does not grant independent-thesis approval, resolve source availability, or authenticate training execution.

Validation of this follow-up: **1,270 passed, one expected non-CPU-device skip, 84.53% coverage** in the complete CPU/native-LaTeX suite; 21 PCD tests also pass under Python 3.11, and the TRL template module compiles under 3.11. Ruff lint/format, 20-file mypy and the nine-module wheel check pass. The follow-up changes no stored scientific result or frozen release file.


A second actual workflow, [38128256334](https://github.com/arvindcr4/tinker-rl-lab/actions/runs/38128256334), passed 13 of 15 jobs, including all four Python test shards and native document builds. Quality reached 84.53% coverage with 1,269 passing tests, but its remaining native audit test lacked the Tectonic executable. The quality job now installs the exact same checksum-pinned Tectonic already used successfully by docs-build; no test is skipped and no threshold is reduced. Local verification of both provenance records also identifies the sweep's one-seed underpowered result and missing uncertainty, in addition to the campaign's missing uncertainty. These strict evidence failures remain unchanged.

The native audit deliberately invokes Tectonic with `--only-cached`. Quality therefore also performs the same three-document scratch compilation as docs-build to populate the cache before running offline tests; an installed executable alone is insufficient. Source PDFs remain untouched.
