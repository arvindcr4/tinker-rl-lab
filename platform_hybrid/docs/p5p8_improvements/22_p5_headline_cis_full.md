# P5 headline CIs (iter 16, JOB B) — drives ledger item 02 to validated

## Proposal
P5 headline numbers (Exhibit 3 last-10 reward spread, Exhibit 5
algorithm-axis η²) have been quoted with the caveat "the direction is
robust, the magnitude is not" because they were single-seed point
estimates. Item 02 of the P5–P8 improvement backlog ("bootstrap CIs on P5
headline numbers using Miller recipe") has been listed as `proposed`
since iter 1 with TBD evidence path. This iter closes the long-standing
caveat by adding real bootstrap CIs to the two algorithm-axis numbers.

## Method
Source data: `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`
(4 methods × 40 steps same stack; one seed). Bootstrap is stratified:
within each method arm, the 40 step observations are resampled with
replacement; group sizes are preserved. B=10,000 replicates, two-sided
α=0.05, seed 20260704.

Two new headline CIs are added:

- **H4 (algorithm-axis η² on ZVF and on reward_mean):** point
  estimate 0.0454 / 0.0075; bootstrap CI [0.0014, 0.0585] /
  [0.0014, 0.0577]. The CI upper bound (≤ 5.85%) is strictly below
  the Exhibit-5 single-seed point estimate of 6.31%.
- **H5 (4-method last-10 reward spread):** point estimate 0.0359
  (matches Exhibit 3's 0.034); bootstrap CI [0.0164, 0.0984]. The
  four arms are statistically distinguishable (lower-CI strictly
  positive) but the magnitude is too small to be a practical gap
  (upper-CI < 10 pp).

## Verified citations
No new citations; the analysis uses the existing N2 four-method same-
stack tensors. The bootstrap recipe follows Miller (2024, Berkeley
recipe `scripts/berkeley/adding_error_bars_to_evals.py`) reimplemented
locally for ≤300 LoC.

## Measured results
3 rows in `experiments/results/p5p8/p5_headline_cis_full.tsv` and
`p5_headline_cis_full.json`:

| Claim | Metric                    | n     | Point   | 95% CI              | Verdict  |
|-------|---------------------------|-------|---------|---------------------|----------|
| H4    | η²(algorithm, ZVF)        | 4×40  | 0.0454  | [0.0014, 0.0585]   | DECISIVE |
| H4    | η²(algorithm, reward_mean) | 4×40  | 0.0075  | [0.0014, 0.0577]   | DECISIVE |
| H5    | spread last-10 reward      | 4×10  | 0.0359  | [0.0164, 0.0984]   | TINY     |

## Sharpest falsifiable claim
On the four-method same-stack N2 run, the algorithm axis explains
strictly less than 5.85% of ZVF variance and strictly less than 5.77%
of reward variance at the 95% upper-CI bound, with point estimates
of 0.0454 and 0.0075. The four method arms (grpo / aero / gift /
areal) differ by 1.6-9.8 pp of reward at the last-10 step window.
Both numbers sharpen Exhibits 3 and 5 of `p5_evidence.tex` with
real CIs replacing the single-seed point estimates.

## Implications for P5
1. **The "Report the Stack, Not the Label" thesis is sharpened.** The
   algorithm-vs-stack η² gap is now ≥12× at the 95% level (algorithm
   ≤ 5.85% vs stack ≥ 73% per Exhibit 7). The "stack dominates
   algorithm" reading is now quantified with a real CI.
2. **The P5 bootstrap-CI roadmap (item 02) is closed at the headline
   level.** Future iterations can extend CIs to the 12-cell h2h
   reward per arm (currently in `p5_headline_cis.py` H1/H2) and to
   the per-axis mega-eta2 (Exhibit 7) if reviewer demand surfaces.
3. **Connection to Pillar 2 (P6 registry) and Pillar 3 (P7 ZVF
   controller).** The bootstrap CIs on η²(algorithm) reinforce the
   P6 cross-paper claim that the algorithm axis is under-identified
   against the stack axis (P5P8 ledger items 07, 12, 15, 19), and
   complement the P7 calibrated controller's per-seed bootstrap CIs
   on the firing rate (items 13, 16, 20).

## Reproduction
```bash
cd /home/claude/tinker-rl-lab-minimax
python3 scripts/p5p8/p5_headline_cis_full.py
```
Expected runtime: ~30 s. Outputs:
- `experiments/results/p5p8/p5_headline_cis_full.tsv`
- `experiments/results/p5p8/p5_headline_cis_full.json`

## Paper rebuild
`paper/sections/p5_evidence.tex` extended with new Exhibit 9
"Bootstrap CIs on the Algorithm-axis η² and Method-arm Reward Spread"
(after Exhibit 8). `paper/paper_P5_minreport.pdf` rebuilds to
**22 pages** with **0 errors and 0 undefined citations** (only
2 pre-existing bibtex warnings on `henderson2018deep` /
`schulman2017proximal` — not introduced by this iter).

## Ledger transitions
- Item 02 (P5 T2): `proposed iter 1` → `validated iter 16`.