# Exact-zero versus tolerance-based ZVF: stored-group recomputation

## Result and claim boundary

On the existing 600 GSM8K prompt-groups (three sampling seeds, 200 groups each,
G = 8), independent reward jitter U(0, 10^-4), using Python's local
`random.Random(0)` generator, changes the historical exact-zero population-variance
fraction from **95/600 (0.1583333) to 0/600**. The declared canonical diagnostic,
**sample variance (ddof = 1) <= 10^-6**, stays at **95/600 before and after**.
None of its 600 group classifications changes.

This probe establishes brittleness of the exact-zero implementation. It does not
falsify the declared tolerance-based ZVF. It does not measure training, gradients,
controller performance, or learning improvement from PCD. Both indicators still
alias all-correct and all-wrong groups on the original binary rewards: there are
76 all-correct, 19 all-wrong, and 505 mixed groups.

PCD retains its original definition, mean population variance (ddof = 0):

- Before: 0.15380208333333334
- After: 0.1538023817497187
- Change: +2.984163853525512e-7

PCD is unchanged when rounded to six decimals (0.153802), but is not literally
invariant. The variance threshold is distinct from the jitter amplitude.

## Threshold sensitivity

The same single jitter realization is reused at every threshold. No retraining,
resampling of the stored groups, clipping, or reward renormalization is performed.
The additive probe can therefore produce rewards slightly above one, as in the
original analysis; it is not a change to the reward parser's normalization policy.

| Sample-variance threshold epsilon | Before count | After count | Changed groups |
| --- | ---: | ---: | ---: |
| 0 | 95 | 0 | 95 |
| 10^-12 | 95 | 0 | 95 |
| 10^-10 | 95 | 0 | 95 |
| 10^-9 | 95 | 71 | 24 |
| 10^-8 | 95 | 95 | 0 |
| 10^-6 | 95 | 95 | 0 |
| 10^-4 | 95 | 95 | 0 |
| 10^-2 | 95 | 95 | 0 |

Every denominator is 600. The unchanged values over 10^-8 through 10^-2 apply to
these stored GSM8K groups only; this audit does not substantiate a cross-framework
sensitivity bound on unbundled reward traces.

## Reproduce and verify

From the repository root, using Python 3 and the standard library only:

```sh
python3 platform_modal/scripts/pcd_vs_zvf.py
python3 -m unittest discover -s tests -p test_pcd_vs_zvf.py -v
```

To leave the committed outputs untouched:

```sh
python3 platform_modal/scripts/pcd_vs_zvf.py --output-dir /tmp/pcd-zvf-check
```

The seeded sequence preserves the historical sorted source order, seeds 123, 42,
456, and original `per_problem` order within each file. Per-seed exact-zero and
canonical tolerance counts before jitter are 38/200, 26/200, and 31/200 in that
order. Exact-zero counts become zero; tolerance counts remain unchanged.

The JSON manifest records raw source SHA-256 values, seed, ordering, jitter
transform, metric definitions, all counts and fractions, and the claim boundary.
The per-group ledger provides source filename and source-local index for every
classification. The focused tests check the sample/population distinction, the
inclusive threshold boundary, seeded reproducibility, invalid inputs, all 600
groups, source integrity, and byte-for-byte regeneration of current outputs.

## Artifacts and historical compatibility

- `pcd_vs_zvf_tolerance_analysis.json`: definitions, provenance, per-seed counts,
  pooled metrics, and threshold sensitivity
- `pcd_vs_zvf_epsilon_sensitivity.tsv`: threshold-sensitivity table
- `pcd_vs_zvf_jitter_groups.tsv`: 600-row before/after variance and classification ledger
- `pcd_vs_zvf_recomputed_summary.tsv`: newly recomputed metrics with explicit
  exact-zero and tolerance names; historical unqualified ZVF key aliases continue
  to mean exact-zero variance
- `pcd_vs_zvf_shape.tsv`: unchanged original binary-reward shape table

`pcd_vs_zvf_summary.tsv` is retained **byte-for-byte as the historical artifact**.
Its unqualified jitter/ZVF interpretation is superseded by this explicit two-metric
analysis. The old before/after ZVF fields describe exact-zero population variance,
not the sample-variance tolerance used in the declared pipeline. The callable
`zvf_ind` retains that historical exact-zero definition for compatibility.

The original script also recomputes different cross-run correlations from the
current `zvf_summary.tsv` than those in the historical summary. The new JSON
records the current input's SHA-256 and recomputation for provenance only:
Spearman(ZVF, outcome) = 0.5641563618733547, Spearman(mean reward, outcome) =
0.9191945244559403, Spearman(ZVF, collapse) = 0.14388899906616576, and
Spearman(mean reward, collapse) = -0.1695305960027918 (80 rows). The historical
summary retains 0.5638, 0.9527, 0.1439, and -0.3291. This tolerance audit does not
revise or validate the broader cross-run predictive claims; statements using
those historical values must identify them as historical reported results.
