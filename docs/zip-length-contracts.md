# R15 zip length contracts

Reviewed against `eb6dd299`: all 98 inventoried calls in 35 files. Original line numbers below remain stable audit identifiers even after formatting. 93 calls require equal lengths; five explicitly permit a bounded prefix. No historical results are changed.

The optional NLL mask now branches before pairing: no mask means zero auxiliary examples; a present mask must match every response. Adjacent windows deliberately have lengths N and N-1 and visit only complete neighboring pairs. The comparability auditor records unequal cohort lengths as incompatible before comparing the common prefix.

## `platform_hybrid/paper/figures/generate_figures.py`

- Base line 163: `zip(labels, handles)` → `strict=True`. Plot metadata/legend entries represent the same series and must align one-to-one.
- Base line 227: `zip(peaks, last10s, partials, algos)` → `strict=True`. Plot metadata/legend entries represent the same series and must align one-to-one.

## `platform_local/trl_integrations/scientific.py`

- Base line 46: `zip(completions, answers)` → `strict=True`. Every completion has exactly one gold answer; existing length validation is retained.
- Base line 58: `zip(prompt_ids, rewards)` → `strict=True`. Every reward has exactly one verified prompt ID; grouping cannot discard an outcome.

## `platform_tinker/tinkerrl/critic.py`

- Base line 109: `zip(saved_groups, expected_groups)` → `strict=True`. Saved optimizer parameter groups must match the expected groups; checkpoint validation precedes pairing.
- Base line 228: `zip(targets, predicted)` → `strict=True`. Residual variance pairs every target with its prediction; explicit checks also cover constant/empty targets.
- Base line 259: `zip(advantages, values[:t])` → `strict=True`. Each of T advantages pairs with its state value; the T+1 bootstrap value is deliberately excluded by slicing.

## `platform_tinker/tinkerrl/grpo.py`

- Base line 873: `zip(advantages, token_lengths)` → `strict=True`. Each advantage has one response token length for truncation masking.
- Base line 934: `zip(rewards, truncated)` → `strict=True`. Each reward needs a truncation flag, including when truncation masking is disabled.
- Base line 1000: `zip(logprobs_list, response_lens)` → `strict=True`. Each response-logprob tensor has one response length; slicing must not drop responses.
- Base line 1085: `zip(nll_mask, accumulators)` → `strict=True`. Every response accumulator has one auxiliary-loss inclusion flag.
- Base line 1095: `zip(() if nll_mask is None else nll_mask, resp_logprobs)` → `strict=True`. No mask bypasses pairing entirely; a present mask supplies one flag per response.
- Base line 1306: `zip(sequences, resp_ids_list)` → `strict=True`. Sampler sequences and extracted response token IDs are two views of the same samples.
- Base line 2933: `zip(pre_meta["truncated"], rewards)` → `strict=True`. Pretraining critic targets and response truncation flags refer to identical scored samples.
- Base line 3041: `zip(advs, truncated)` → `strict=True`. Each normalized advantage is masked by its own response truncation flag.
- Base line 3042: `zip( resp_ids_list, advs, nll_flags, sampler_lps )` → `strict=True`. Token IDs, advantages, auxiliary flags, and sampler logprobs are one record per response.
- Base line 3060: `zip(pending, pred)` → `strict=True`. Each pending prompt group receives exactly one critic prediction before reward centering.
- Base line 3065: `zip(pending, pred)` → `strict=True`. Repeat each prompt prediction across that same group of rewards for explained variance.
- Base line 3079: `zip(flat_advs, flat_truncated)` → `strict=True`. Flattened advantages and flattened truncation flags retain identical sample order and count.
- Base line 3081: `zip( pending, pending_extra )` → `strict=True`. Pending prompt groups have exactly one corresponding metadata group.
- Base line 3085: `zip(cuts, rewards)` → `strict=True`. Critic reward targets have one truncation flag each.
- Base line 3090: `zip(resp_ids_list, nll_flags, lps)` → `strict=True`. Response token IDs, auxiliary flags, and old logprobs retain identical sample counts.
- Base line 3142: `zip(all_old, all_resp_lens)` → `strict=True`. Every stored old-logprob sequence has one response length for suffix slicing.
- Base line 3884: `zip(e1["repo"], e1["base_commit"])` → `strict=True`. Dataset repository and base-commit columns have one entry per row.

## `tests/reviews/test_hybrid_framework_review.py`

- Base line 128: `zip(adv, lp)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/reviews/test_review09_resume.py`

- Base line 37: `zip(left, right)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/reviews/test_review23_output_closure.py`

- Base line 23: `zip([4, 32], accs)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/reviews/test_review27_diagnostic_followup.py`

- Base line 351: `zip(["grpo", "ppo"], values)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_analysis_validation.py`

- Base line 90: `zip(records, [np.int64(100), np.int64(500)])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_cpu_causal_lm_smoke.py`

- Base line 61: `zip(rows, data)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 125: `zip(model.parameters(), reference.parameters())` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 133: `zip(before_parameters, model.parameters())` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 135: `zip(model.parameters(), reference.parameters())` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 142: `zip(optimizer.state.values(), reference_optimizer.state.values())` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_cpu_critic_resume_smoke.py`

- Base line 31: `zip(actual, expected)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_cpu_optimizer_smoke.py`

- Base line 78: `zip(self.rows(data), data)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 242: `zip(client.gradients, expected_gradients)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 248: `zip(client.probabilities, client.probabilities[1:])` → `strict=False`. Intentional adjacent pairs: the shifted sequence has one fewer element; omit the unpaired final element.

## `tests/test_critic.py`

- Base line 26: `zip(advs, [1.5, 0.5, 0.5])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 28: `zip(rets, [2.0, 1.0, 1.0])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 39: `zip(advs, [1.0, 0.0, 0.5])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 50: `zip(advs, [1.25, 1.0])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 52: `zip(rets, [1.25, 1.0])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_dataset_revision_fidelity.py`

- Base line 22: `zip(loader.call_args_list, ["train", "test"])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_experiments_next_quality.py`

- Base line 109: `zip((42, 43, 44), (0.2, 0.3, 0.4))` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_grpo_loss.py`

- Base line 65: `zip(advs, expected)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 241: `zip(per_group, batch_global)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_loss_numerical_stability.py`

- Base line 18: `zip(oracle_new, old)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 22: `zip(ratios, cases)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 31: `zip(new, oracle_new)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_pcd_vs_zvf.py`

- Base line 39: `zip(actual, expected)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_portfolio_comparability_audit.py`

- Base line 34: `zip(suites, scores)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_portfolio_eval_integrity.py`

- Base line 165: `zip(receipt["records"], harness.calls)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_pro_review_01_analysis.py`

- Base line 595: `zip(attempts, process.call_args_list)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_pro_review_02_modal_analysis.py`

- Base line 605: `zip(table, data["results"], data["key_findings"])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_pro_review_29.py`

- Base line 39: `zip(z, a)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_pro_review_29_mains.py`

- Base line 41: `zip(zvfs, accuracies)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_properties_advantages.py`

- Base line 89: `zip(base, moved)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 103: `zip(base, moved)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 122: `zip(g, n)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 132: `zip(adv, exclude)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 142: `zip(adv, masked, lens)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 158: `zip(adv, ret, vs)` → `zip(adv, ret, vs[:-1], strict=True)`. The T state values pair with T advantages/returns; explicitly exclude the T+1 bootstrap entry instead of silently truncating it.
- Base line 182: `zip(prompt, resp)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 195: `zip(adv, prompt, lps)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 197: `zip(prompt, lps)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 200: `zip(prompt, resp, adv, lps)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 226: `zip(prompt, lps)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_properties_stats.py`

- Base line 160: `zip(trained, base)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 161: `zip(trained, base)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_public_analysis.py`

- Base line 603: `zip(intervals[0], interval)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_registry_provenance_review.py`

- Base line 83: `zip(seeds, values)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 91: `zip(seeds, values)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_samestack_protocol.py`

- Base line 59: `zip(grpo["policy"], g2["policy"])` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tests/test_tinker_sdk_loss_integration.py`

- Base line 104: `zip(prompts, responses)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 106: `zip(prompts, responses)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.
- Base line 135: `zip(data, client.backward_data, expected)` → `strict=True`. Fixture/expected/observed entries align one-to-one; unequal lengths must not hide missing test coverage.

## `tools/audit_portfolio_comparability.py`

- Base line 268: `zip(base_rows, trained_rows)` → `strict=False`. Unequal cohorts are already marked incompatible; compare only their shared prefix to collect additional diagnostics.

## `tools/audit_thesis_numbers.py`

- Base line 183: `zip(signs, diffs)` → `strict=True`. Each exact sign-flip permutation has one sign for every paired difference.
- Base line 239: `zip(arms, holm(ps))` → `strict=True`. Holm-adjusted p-values retain one value per named comparison arm.
- Base line 446: `zip(t, b)` → `strict=True`. McNemar consumes matched trained/base binary outcomes from the same rows.
- Base line 2568: `zip(run[algo, s]["post_correct"], run[algo, s]["pre_correct"])` → `strict=True`. Per-seed pre/post correctness vectors must represent the same evaluated items.
- Base line 2574: `zip(*forms)` → `strict=True`. Transposing baseline correctness vectors requires identical item counts in all forms.
- Base line 2797: `zip(want, got)` → `strict=True`. Recorded/computed numeric tokens must have identical counts, already checked before comparison.

## `tools/check_public_results.py`

- Base line 381: `zip(cases, seeds)` → `strict=True`. Every selected case needs one identity-matched seed record.
- Base line 394: `zip(values, values[1:])` → `strict=False`. Intentional adjacent pairs: the shifted sequence has one fewer element; omit the unpaired final element.
- Base line 409: `zip(seeds, screen_indices)` → `strict=True`. Every seed record must have its original selected screen index for stride verification.

## `tools/check_thesis_evidence.py`

- Base line 110: `zip(trained, base)` → `strict=True`. McNemar discordance b requires all paired trained/base outcomes.
- Base line 111: `zip(trained, base)` → `strict=True`. McNemar discordance c uses the same full matched outcome population.
- Base line 121: `zip(trained, base)` → `strict=True`. Paired mean differences require one trained and one base observation per item.
- Base line 156: `zip(short["runs"], full["runs"])` → `strict=True`. Summary/raw run records correspond one-to-one in the validated 15-run cohort.
- Base line 211: `zip(contrast["diffs"], diffs)` → `strict=True`. Every recorded seed contrast must match its independently recomputed difference.
- Base line 459: `zip(trained, base)` → `strict=True`. E13 mean item differences use all paired episode outcomes.
- Base line 465: `zip(trained, base)` → `strict=True`. Identical-progression counts compare every paired episode.
- Base line 481: `zip(trained, base)` → `strict=True`. E4 paired mean differences cannot silently shorten either arm.
- Base line 490: `zip(trained, base)` → `strict=True`. Reported raw paired deltas use the complete trained/base cohort.

## `utils/stats.py`

- Base line 317: `zip(steps, steps[1:])` → `strict=False`. Intentional adjacent pairs: the shifted sequence has one fewer element; omit the unpaired final element.
- Base line 528: `zip(steps, steps[1:])` → `strict=False`. Intentional adjacent pairs: the shifted sequence has one fewer element; omit the unpaired final element.

## Offline validation

- Ruff 0.17.1 B905 check passes across the full curated `RUFF_PATHS` scope.
- Broad F821/F811 name-safety and repository-policy checks pass.
- Changed Python files pass all configured lint/format checks, except the existing
  R23 formatting/style issues being corrected independently by integration.
- New synthetic contract tests: 27 passed. Focused scientific/loss/CPU suites:
  368 passed, 1 skipped. Remaining modified-file suites plus the GAE property
  suite: 551 passed. The property suite overlaps the two runs.
- Full `make lint` on the isolated base still reports the six pre-existing R19/R23
  pytest-style findings; no new ignores were introduced. Integration must combine
  its independent fixes and verify the complete gate before accepting exemption
  removal. No provider/training jobs or historical-result regeneration occurred.
