> VALIDITY BOUNDARY (review 43): First-five predictors and last-ten outcomes require at least 15 observations and disjoint windows; shorter runs are ineligible. Two collapse cases and reward-only equal AUC do not establish incremental predictive validity. Recover task labels, use independent held-out runs, nonlinear reward/task controls and report incremental performance over reward-only baselines. Existing exploratory tables are not a validated predictive model.

# ZVF Predictive Validation

This is an archived exploratory analysis over existing logs; it is not an admitted predictive validation. The predictor only uses the first five logged optimization steps; outcomes are computed from the last ten logged steps.

## Protocol

- Raw candidate records found: 52
- Historical deduplicated run count: 22 (eligibility not re-established)
- Minimum eligible run length: 15 steps, with disjoint first-five and last-ten windows
- Historical analysis admitted 10-step runs. All counts, intervals and metrics below are quarantined pending a fresh eligible-cohort calculation; they must not be cited as results under the corrected protocol.
- Early window: first 5 steps
- Late outcome window: last 10 steps
- Collapse label: late reward mean <= 0.05
- Useful label: late reward mean >= 0.25
- Fixed early-failure rule: early ZVF >= 0.8 and early reward <= 0.05

## Quarantined historical results (not recomputed)

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

## Quarantined historical task breakdown

| Task | Runs | Collapsed | Useful | Mean late reward |
|---|---:|---:|---:|---:|
| gsm8k | 8 | 0 | 6 | 0.609 |
| tool_use | 2 | 2 | 0 | 0.000 |
| unknown | 12 | 0 | 11 | 0.329 |

## Interpretation

No triage or incremental-prediction claim is established by these historical numbers. Recompute the cohort after excluding overlapping windows before evaluating the fixed first-five-step rule. It does not by itself prove a universal causal law: the data are still retrospective, small, and drawn from the available completed experiments rather than from a newly randomized prospective campaign.

Figure outputs: `paper/figures/v2/zvf_predictive_validation.png` and `paper/figures/v2/zvf_predictive_validation.pdf`.
