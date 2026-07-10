# ZVF Predictive Validation

This is a pseudo-prospective validation over existing completed logs. The predictor only uses the first five logged optimization steps; outcomes are computed from the last ten logged steps.

## Protocol

- Raw candidate records found: 52
- Deduplicated independent runs: 22
- Minimum usable run length: 10 steps
- Early window: first 5 steps
- Late outcome window: last 10 steps
- Collapse label: late reward mean <= 0.05
- Useful label: late reward mean >= 0.25
- Fixed early-failure rule: early ZVF >= 0.8 and early reward <= 0.05

## Primary Results

- Collapsed runs: 2 / 22
- Useful runs: 17 / 22
- Early ZVF vs late reward Spearman: -0.158 [-0.634, 0.430]
- Early GU vs late reward Spearman: 0.158 [-0.407, 0.614]
- Collapse AUC using early ZVF: 1.000 [1.000, 1.000]
- Collapse AUC using early reward only: 1.000 [1.000, 1.000]
- Collapse AUC using ZVF-minus-reward composite: 1.000 [1.000, 1.000]
- Fixed rule precision/recall/F1: 1.000 / 1.000 / 1.000
- Fixed rule confusion: TP=2, FP=0, TN=20, FN=0
- Leave-one-run-out R^2, early reward only: 0.885
- Leave-one-run-out R^2, early reward + early ZVF: 0.888

## Task Breakdown

| Task | Runs | Collapsed | Useful | Mean late reward |
|---|---:|---:|---:|---:|
| gsm8k | 8 | 0 | 6 | 0.609 |
| tool_use | 2 | 2 | 0 | 0.000 |
| unknown | 12 | 0 | 11 | 0.329 |

## Interpretation

The validation supports the narrow triage claim if the fixed first-five-step ZVF rule separates collapsed runs with high recall and few false positives. It does not by itself prove a universal causal law: the data are still retrospective, small, and drawn from the available completed experiments rather than from a newly randomized prospective campaign.

Figure outputs: `paper/figures/v2/zvf_predictive_validation.png` and `paper/figures/v2/zvf_predictive_validation.pdf`.
