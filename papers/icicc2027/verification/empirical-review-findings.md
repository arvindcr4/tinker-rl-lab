# Empirical evidence audit for the new critique

Read-only audit, 18 October 2026. No training, sampling, native grading or stored-result edit was performed. This report distinguishes directly retained structured fields from historical narrative and unavailable evidence.

## Original identities and replacement reasons

- `zvf-program/flagship/pavlovs_domain_contract.json`: all original contracts; E6 is `webbench_eval` (Halluminate WebBench), E13 is `openreward_games_eval` (held-out environment families absent from training)
- `zvf-program/flagship/pavlovs_open_portfolio_overrides.json`: explicitly maps WebBench to WebArena and OpenReward to BALROG as a new non-comparable portfolio. Proposed-only MLAgentBench/CyberGym/VisualAgentBench mappings are not claimed as completed campaign scopes
- `zvf-program/flagship/webbench_eval.py` lines 4–6 and 584–616, and `pavlov_webbench_eval_adapter.py` lines 73–111: public task CSV and license exist; official browser environment, task credentials/reset contract and native ground-truth verifier were missing. Do not describe all WebBench data as private
- `zvf-program/flagship/PAVLOV_SUITE_EXECUTION_ROADMAP_2026-08-09.md` row E13: missing held-out environments, seed separation and game-state verifier. A public OpenReward platform is not the missing held-out family manifest

## E1 architecture and timeout correction

- Direct `outputs/e1_swe_bench_pro/lane_receipt_2026-08-09.json`: `evidence_class=harness_validation`, `harness_validation.is_model_score=false`; one selected gold-patch task under emulation; timeout 1,200 vs 120 seconds belongs here
- `reports/public_revision_2026-10-11/thesis/ch_back_campaign_detail.md` D.5 explicitly states the full 731 scoring run used the 731 unmodified official scripts on native linux/amd64 Modal workers, harness `ca10a60a5fcae51e6948ffe1485d4153d421e6c5`
- Full scored-run receipt/logs remain absent here. Exact CPU model, allocation, container-engine version, runtime totals and timeout/assertion/compile breakdown are unresolved
- 713 native evaluations are not 713 failures: two resolved. Narrative patch audit reports 593 corrupt and 120 parsing patches, with both successes in the latter. Parse validity does not prove every remaining patch applied; these categories do not establish timeout counts

## E8/E11 extraction evidence

- Campaign detail D.4: E8 diagnostic 1,259 parse-none and 1,086 length finishes; only latter independently retallied in earlier bounded recovery. Exported grade rows lack parsed-answer indicators; intersection and nontruncated-only accuracy unavailable
- E8 arithmetic: 1,967 − 1,259 = 708; 450/708 = 0.635593220338983. Conditional sensitivity is not a verified regraded subset
- E11 aggregate 150 extraction failures; 312−150=162 extracted; 129/162=0.7962962962962963. Of 183 failures, 33 have extracted candidates without pass
- Per-design extraction membership, response text and per-sample finish reasons unavailable. Cannot apportion truncation, unclosed reasoning, formatting or other causes. The extractor could accept code from unterminated reasoning, further limiting conditional interpretation
- Reported 312,622 tokens/312 = 1,001.9936 average is not a cap-hit count
- Retained `platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.json` actually exists, correcting the prior source map's statement that it was absent: 156 design pairs, 54 both-pass, 81 both-fail, 13 code-only, 8 spec-only; 129 total. Stored 200,000-replicate percentile interval [0.3429487179, 0.4871794872]. No bootstrap rerun in this revision
- Historical 50-item protocol comparison 23/50 vs 33/50 is retained narrative, with 11/1 discordances and p≈0.006. Same-adapter protocol changed engine/thinking/temperature/budget together. It is not the paired trained/base comparison (33/50 vs33/50) and not a full 312 controlled sampler replay

## Paired outcomes and inference

- `outputs/e1_e14_small_scale_2026-09-26/E*/paired.json` supplies all fourteen lanes and seventeen primary outcome contrasts; full table now in manuscript
- E2 also has `secondary_log_net_paired`: 21 jointly passed tasks, log-runtime difference −0.1677067, bootstrap 95% interval [−0.3195513,−0.00460239]. This nominal interval excludes zero; selected both-pass population and wall-clock noise preclude treating it as reliable causal efficiency evidence. The manuscript confines its inconclusive statement to the 17 primary outcome contrasts
- E12's 151 rubric observations cluster into six apps; exact two-sided equal-app pass-rate sign-flip p = .25 is preferable to item McNemar p≈.3018. Grader is the base model
- E13 pilot has 26 episodes/five environments, not the later 255/six. Equal-environment means 25.50623885918% and 23.63101604278%; unrounded difference 1.8752228164 pp
- `inference-configuration-inventory.json/.csv`: all historical scored/attempted and pilot scope rows, evidence levels, absent fields. Null means not recovered, not default. Inference table does not fabricate top-p/context settings

## Publication-review arithmetic clarification

E12 uses equal-app pass-rate differences, not raw passed-rubric counts. In app order t1–t6 the differences are −1/24, +1/33, 0, −4/25, 0 and −1/24. The observed mean is −703/19800. Exact enumeration of all 64 sign assignments with inclusive tails gives two-sided p = 16/64 = .25; the lower one-sided value is 8/64 = .125. Sign-flipping raw count differences instead gives two-sided p = .50, a different weighting/statistic. The source-backed arithmetic is in e12-signflip-arithmetic.json and checked by verify_manuscript.py. Neither establishes a trained/base difference.

E1-R’s retained outputs/finish_pending_2026-09-27/E1/result.json declares an all-attempt failure policy. The 133 ungraded final items are 120 patch-apply/evaluation errors and 13 empty patches. Thus 1/190 is the operational primary score; 1/57 = .0175 is conditional on native grading, with 57/190 = 30% grading coverage.
