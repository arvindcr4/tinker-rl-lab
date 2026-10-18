# Platform low-hanging editorial pass

Branch: `review-low-hanging-2026-10-18`. No commit or push performed.

## Outcome

- All 505 integer database finding IDs for the eight assigned papers have a
  disposition: 8 fixed, 16 already-fixed, 481 wontfix-scope.
- Compound findings with only a clerical subpart addressed remain
  `wontfix-scope`, with the remaining concern recorded explicitly. No scientific
  or statistical finding is claimed resolved by copyediting.
- No empirical number, analysis, citation, bibliography, shared section, TikZ
  block, release PDF, or externally owned manuscript was changed by this pass.
- Numeric ledgers are header-only: no empirical numeric claims were touched.
  Deferred CRITICAL/MAJOR claims were not audited or altered.

## Source changes

| File | Findings | Change |
| --- | --- | --- |
| `platform_hybrid/paper/sections/p7_abstract.tex` | 447 | Join `matched-budget` across the source newline that rendered as a space. |
| `platform_hybrid/paper/sections/p7_intro.tex` | 450 | Join `static-$G{=}16$` across the source newline that rendered as a space. |
| `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex` | 626, 628, 632, 634, 640, 641 | Define ZVF on first use and distinguish it from ZUF; join actor-learner; state binary-gate idempotence; use maximized-surrogate terminology; label dispatcher invariant; make multiplication and tolerance explicit. |
| Same P12 file | 633, 636, 637 (partial only) | Define existing reward/advantage symbols and token broadcast; make prose-equation batch indices and squared-norm terminology explicit; repeat critic-symbol definitions in the floating caption. Root semantics, diagram indices, and table layout remain unclosed. |

P1/P2/P3/P4/P5/P6 have no manuscript changes: inspected easy candidates were
already corrected, no longer included, or protected/out of scope. In particular,
P4's historical caption remains after `\endinput`; it is not in the current
three-page PDF. The P7 historical replay blocks are inside `\iffalse`, so the
inactive `buton` typo and historical roadmap sections were not edited.

## Builds and inclusion checks

All eight current roots compiled successfully with `tectonic -X compile
--only-cached --keep-logs --keep-intermediates --outdir <isolated-directory>`.
The two touched roots were compiled again after editing, both successfully.
All baseline and final logs have zero undefined references/citations.

| Root | Baseline pages | Final pages | Result |
| --- | ---: | ---: | --- |
| `paper_P1_scaling.tex` | 51 | unchanged | success |
| `paper_P2_zvf.tex` | 27 | unchanged | success |
| `paper_P3_group_size.tex` | 22 | unchanged | success |
| `paper_P4_length_bias.tex` | 3 | unchanged | success |
| `paper_P5_minreport.tex` | 17 | unchanged | success |
| `paper_P6_registry.tex` | 15 | unchanged | success |
| `paper_P7_zvf_controller.tex` | 22 | 22 | success |
| `unified_signal_starvation/paper_P12_signal_starvation.tex` | 13 | 14 | success |

Builds, logs, extracted PDF text, and a rendered P12 factorization-page check
are under `.amp/in/editorial-builds/{before,after}/` in this worktree only.
The existing git-local `/.amp/in/` exclude rule was checked before scratch
creation. Tectonic used a worktree-local copy of the existing cache and
`TECTONIC_CACHE_DIR`; no network calls or release-directory build outputs.
P12's added definitions change pagination; its rendered equations remain inside
the text block. Existing package/UTF-8 and underfull-box warnings remain; this
is not a claim of warning-free typesetting.

## Remaining requests

`low_fruit_platform_shared_requests.md` contains exact unapplied replacements
for P12 bibliography capitalization (629–630), figure naming/internal caption
codes (627), and diagram summation indices (remaining part of 636).
Figure arrow layout (638), root semantics (633), table wrapping (637), and
scientific/statistical issues remain deferred, not silently resolved.

Other sessions modified unrelated files in this same worktree during the pass;
those changes were neither reverted nor included in this report's source-change
claims. No experiments, external review calls, delegation, commits, or pushes.
