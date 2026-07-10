# 95 — P5 yield-aware MIN-REPORT v2.1 axis (iter 80, JOB B / SYNTH)

## Falsifiable headlines

- **H1 — Item 13 (zvf_yield_residual) adds 4.645 bits to the v2.0 MIN-REPORT fingerprint**, raising total from 18.27 to 22.92 bits (+25% on v2.0; +101% on v1). Item 13 has n_unique = 58 distinct values on the 98-cell corpus (at 10⁻⁴ rounding) — far more discriminating than any v1 item and any v2 stack axis. Bootstrap 95% CI on the per-cell H_bits contribution is [+3.66, +4.65].
- **H2 — Item 13 strengthens fingerprint-vs-outcome coupling** in a paired bootstrap (B=200, 2000 cell-pairs). v2.0 (v1+stack axes) gives ρ=0.3835; v2.1 (v2.0+Item13) gives ρ=0.4384. **Δρ = +0.0559, 95% CI [+0.050, +0.062]**, paired, **excludes zero**. This is the **first MIN-REPORT item** that independently strengthens the Hamming-vs-|Δzvf| coupling on the live 98-cell corpus.
- **H3 — Per-cell badge uplift deterministic = +20 points** (95% CI degenerate). All 98 cells populated by Item 13 (formula fully determined by G, mean_reward, zvf columns of cells.tsv).
- **H4 — Item 13 alone vs |Δzvf| gives ρ=0.366** — already **86%** of the v1 fingerprint's ρ=0.427, despite being a single continuous axis.

## Cross-paper coupling

- (i) **P6 iter-66 row 77, iter-70 row 82, iter-74 row 87** — Item 13 is the per-cell instantiation of the registry's `measured_yield_residual` δ_div block. It makes the per-method axis reproducible across any corpus with the (G, p, z) columns.
- (ii) **P7 iter-72 row 85, iter-75 row 88, iter-79 row 93** — together with the iter-75 exact hypergeometric CP_exact formula, Item 13 makes the structural anti-herding signal end-to-end machine-readable from cells.tsv to joint-controller per-step savings.
- (iii) **P8 iter-76 row 89, iter-80 row 94** — the same anti-herding bonus that makes Item 13 signal-bearing on the LLM-RL axis also explains why score-stream gradient (P8 row 94) is more LLM-efficient than absolute-band: rows where consecutive scores plateau are exactly where cheap-vs-expensive disagreement dominates — a per-row analog of the per-cell Item 13.

## Operational recommendation

Adopt **Item 13 (zvf_yield_residual) into a v2.1 MIN-REPORT schema**, displacing one v1 placebo item (we recommend deprecating `decontamination_notes` or `sampler_backend_precision` — both have H_bits concentrated in single values on the live 98-cell corpus). The iter-65 row 76 placebo finding (4/7 v1 items carry zero stack-discriminative bits) is sharpened: at least one yield-residual axis (Item 13) is empirically signal-bearing across every stack-pair configuration.

## Reproducibility

- Script: `scripts/p5p8/p5_delta_div_minreport.py` (290 lines, stdlib only)
- Outputs:
  - `experiments/results/p5p8/p5_delta_div_minreport.tsv` (13 rows)
  - `experiments/results/p5p8/p5_delta_div_minreport_boot.tsv` (4 rows)
  - `experiments/results/p5p8/p5_delta_div_minreport_summary.json`
- Seed: 20260705 (paired bootstrap B=200 with 2000 cell-pairs)
- **paper_P5_minreport.pdf rebuilds to 39 pages** (was 38), 0 errors / 0 undefined citations beyond pre-existing bibtex warnings (`henderson2018deep volume+number`, `schulman2017proximal empty booktitle` — both inherited from prior ledger rows)
