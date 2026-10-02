# 8. The E1–E14 Held-Out Evaluation Campaign

This chapter reports the evaluation of a single Tinker-trained actor against fourteen benchmark suites that were not part of its training data, recorded lane by lane under sealed reservations, immutable receipts and explicit terminal states. Lanes that did not finish and lanes that could not start are reported too, because the methodological claim of this work is not that the actor performs well but that every figure attached to it can be traced to a surviving, hash-pinned artifact, and that the boundary between what was measured and what was not is drawn explicitly. Lane-by-lane detail that does not change any figure is collected in Appendix D.

Two reporting rules govern everything below, and they are worth stating before any number appears. First, original-contract results and replacement-scope results are never pooled and never averaged across suites; a replacement scope is a different suite, a different interface and often a different grader, and combining such figures would produce a number with no referent. Second, no figure in this chapter is quoted to a precision beyond what its source receipt records, and no comparison against a baseline, a sibling suite, or the untrained base model is asserted for the campaign lanes of Sections 8.2–8.6. Those lanes have no matched base-model arm. The one paired baseline comparison in this chapter is the small-scale trained-versus-base rerun of Section 8.7 (Table 8.B): 14 lanes, 5 to 151 identical items each, one vLLM engine for both arms. It finds no significant difference in any lane, and this chapter therefore claims no improvement of any kind over the base model. Where a lane's record itself contains an unresolved discrepancy, the discrepancy is stated.

The consolidated ledger of record for the campaign is `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md`, which supersedes the working tables of 2026-09-05 and 2026-09-12; the per-lane terminal-state ledger is `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`; and the arithmetic of the ledger was re-verified by deterministic code checks recorded in `outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`, which reports eleven of eleven checks passing with zero failures.

## 8.1 The evaluated actor and the one-actor design

A single actor was evaluated across the whole campaign. The base model is `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, and the trained parameter set is the LoRA adapter `arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6` at commit `64444133c55d88c3f1bf0df8a2f5d7ac646125c8`, served in bfloat16 under the identifier `pavlov-public-portfolio-bf16` (source: `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md`; the same identity pair is recorded independently in the E10 native receipt and in the E11 full receipt, which also preserves the training sampler path `tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/sampler_weights/seed809_final`: sources `outputs/public_portfolio_2026-09-05/agentdojo_native_receipt.json` and `outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json`). The Tinker training run behind the adapter has since been purged, so the actor can no longer be sampled on Tinker; the adapter itself is preserved on the Hugging Face branch `checkpoint-seed809-stepfinal-9f777c4018b6` at the recorded commit, and a check on 26 September 2026 confirmed the 2.25 GB weight file is present there, so the actor can be re-served off-Tinker (sources: `outputs/TINKER_SAMPLER_STATUS_2026-09-20.json`, `outputs/e1_e14_small_scale_2026-09-26/ADAPTER_AVAILABILITY_CHECK_2026-09-26.json`).

The one-actor design is deliberate. No lane received a separate checkpoint, per-suite prompt engineering beyond its native harness, or tuning against its own evaluation set: the actor's only training suites were API-Bank-RLVR and SWE-Gym (§8.1.1), although the E10 receipt records its decontamination inventory as absent (`TRAINING_INVENTORY_ABSENT`). Differences between suites therefore reflect the interaction of one fixed policy with fourteen different interfaces, observation spaces, tool conventions and graders, and they are not evidence about what a specialised actor could achieve. The campaign also cannot support any statement about seed variance. Inference conditions were frozen per lane and recorded in that lane's own receipt (§8.1.2).

### 8.1.1 How the actor was trained

The adapter was produced by a single Tinker training run (W&B run `bsv8vx04`), whose configuration survives in the local W&B run directory, the local run receipt and the Hugging Face checkpoint branches; where these files live and how their hashes are verified is recorded in Appendix D. Table 8.1 gives every field they record.

**Table 8.1 — Training record of the E1–E14 actor.** From the surviving run artefacts.

| Field | Value | Evidence |
|---|---|---|
| Launch | `python -m platform_tinker.tinkerrl.grpo_cli --preset pavlov_portfolio`, no overrides | `wandb-metadata.json` |
| Base model | `Qwen/Qwen3.6-35B-A3B` at revision `995ad96e…` | `config.yaml`; run receipt |
| Loss | REINFORCE with a group-mean baseline: $A_i = (r_i - \mu_g)/(\sigma_g + 10^{-8})$ with population $\sigma$; loss $-A_i \sum_t \log \pi(y_{i,t})$, averaged over completions. No importance ratio, no clipping, no KL term, no reference model | `platform_tinker/tinkerrl/grpo.py` (`normalize_rewards`, `make_grpo_loss_fn`) |
| Sampler refresh | Only at checkpoints: steps 1–20 sampled from the step-0 weights and steps 21–40 from the step-20 weights, with no importance correction | `grpo.py` |
| Reward | `PavlovNonXLAMReward`: API-Bank rows scored by `StrictToolCallReward`, SWE-Gym rows by `PatchReward` (1.0 for an exact diff, otherwise partial credit capped at 0.99) | `platform_tinker/tinkerrl/grpo_cli.py`, `grpo.py` (current code; see Appendix D) |
| Training data | Suites `api_bank_rlvr_train` and `swe_gym_train` (API-Bank-RLVR at `bf67c426…`, SWE-Gym at `bb94ed9e…`); dataset revision `bd1f0db6…`; training pool of 512 rows | run receipt; `config.yaml`; E11 decontamination receipt |
| Declared evaluation suites | 14 primary evaluation suites (the original E1–E14 contracts, including `verilog_eval`); `heldout_suite_ids` empty; `evaluate_heldout` false | run receipt |
| Group size, batch, steps | G = 4; 2 prompts per step; 40 steps; 80 prompt draws and 320 completions in total | `config.yaml`; run receipt |
| Optimiser | lr 1 × 10⁻⁵; Adam β₁ 0.9, β₂ 0.95, ε 10⁻⁸ | `config.yaml` |
| LoRA | rank 32, alpha 32, all linear layers | `config.yaml`; adapter config |
| Decoding during training | prompt 1,536 and response 384 tokens; temperature 0.7, top-p 0.95; seed 809 | `config.yaml` |
| Reward trace | mean 0.025 over the first 5 steps and 0.2425 over the last 10; reward zero on 22 of 40 steps; loss zero on 35 of 40 steps | run receipt; `output.log` |
| Date and cost envelope | started 2026-08-09 08:37:44Z; runtime 1,934 s; \$16.50 authorised, \$18 maximum | `wandb-metadata.json`; run receipt |
| Checkpoints | step 0, step 20, step 40 and final, each a separate Hugging Face repository whose weights sit on a `checkpoint-*` branch (`main` holds only `.gitattributes`); the evaluated adapter is branch `checkpoint-seed809-stepfinal-9f777c4018b6` at `64444133…`, saved immediately after step 40 with no optimiser step in between | run receipt; `outputs/e1_e14_small_scale_2026-09-26/ADAPTER_AVAILABILITY_CHECK_2026-09-26.json` |
| Tinker run | `cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0`; all sampler paths since purged | run receipt; `outputs/TINKER_SAMPLER_STATUS_2026-09-20.json` |

: Training record of the E1–E14 actor, from the surviving run artefacts.

Four items cannot be recovered --- the exact training code, which 80 training prompts were drawn, the Tinker sampler weights, and any held-out training-time evaluation --- and the run departs from the planned Pavlov protocol; both are recorded in Appendix D so that no reader fills them in.

What this means for the chapter is direct. The actor received 40 steps of sparse-reward, critic-free policy-gradient training, with zero loss on 35 of them. The fourteen suite results are properties of this checkpoint, not evidence about what GRPO-style training contributes. The campaign is reported as an exercise in evaluation governance. The only trained-versus-base evidence is the small-sample paired comparison of Table 8.B.

### 8.1.2 Serving conditions per lane

The E4 zero (§8.5) was traced to a serving bridge that sent no stop sequences. The same bridge, `zvf-program/flagship/modal_tinker_openai_bridge.py`, served several other lanes, so Table 8.2 records for every lane how the actor was served and whether that defect could apply. Its `/v1/responses` route hard-codes `stop=None` and thinking on. Its `/v1/chat/completions` route sends stop sequences only if the client supplies them, and the agent clients used here (opencode, mini-swe-agent and the APEX client) do not. The deployed bridge version (sha256 `f59c08b9…`) matches neither committed version of the file.

**Table 8.2 — Serving conditions of the original campaign runs, by lane.**

| Lane | Serving path | Stop sequences | Template / thinking | Max tokens | Temp. | Parse failure / truncation | Could the stop defect apply? |
|---|---|---|---|---|---|---|---|
| E1 SWE-bench Pro | Mixed: 476 Tinker sampler, 255 Modal vLLM (merged PEFT) | none; relies on EOS | chat template, thinking off | 8,192 | 0.2 | 14 generation failures, 4 artefacts lost; 40/476 Tinker and 22/255 vLLM responses at the cap | Not through the bridge. The Tinker path also sent no stops, but no response contains role-token lines and cap rates match across backends |
| E2 FrontierSWE | bridge (Harbor/opencode) | not recorded; client sets none | not recorded | not recorded | not recorded | not recorded | Yes |
| E3 SDAB | no model output | — | — | — | — | — | n/a |
| E4 BankerToolBench | bridge (opencode); base rerun via `/v1/responses` | none | template default (thinking on) | not recorded | bridge default 0.2 | pass3 archive missing | Yes; confirmed for the base rerun |
| E5 APEX-Agents | bridge (actor and judge) | not recorded | not recorded | not recorded | not recorded | 3 unscored failures, 1 interrupted | Yes |
| E6 WebArena | no model output | — | — | — | — | — | n/a |
| E7 BinaryAudit | bridge (mini-swe-agent) | not recorded | not recorded | 16,384 (output limit) | not recorded | attempt ended on an HTTP 402 bridge-budget error | Yes, but the attempt ended on a budget error |
| E8 LAB-Bench | Modal vLLM, merged BF16 | none; vLLM EOS | thinking off | 1,024 | 0 | 1,259/1,967 unparsed (64.0%); 1,086/1,967 length-truncated (55.2%) | No |
| E9 MLE-bench | streaming arm via bridge `/chat/completions`; merged arm via vLLM | none in either arm | thinking off | 4,096 | 0.2 | not recorded | Yes for the streaming arm (single-turn); no for the merged arm |
| E10 AgentDojo | Modal vLLM, merged BF16 | none; vLLM EOS | thinking on | 4,096 | 0 | 1 length finish in 418 recorded response bodies (407 main run, 11 continuation) | No |
| E11 VerilogEval | Tinker sampler, called directly | none | template default (thinking on) | 1,024 | 0.2 | 150/312 extraction failures (48.1%); truncation not recorded (mean response ≈ 1,002 of 1,024 tokens) | Not through the bridge; unknown, because per-item outputs are not kept locally |
| E12 AppBench | no model output | — | — | — | — | — | n/a |
| E13 BALROG | Modal vLLM, merged BF16 | none; vLLM EOS | thinking on | 8,192 | 1.0 | 78 length finishes against 189 stop finishes | No |
| E14 Omni-MATH | Modal vLLM, merged BF16 | none; vLLM EOS | thinking off | 2,048 | 0 | 2,982/4,428 responses at the 2,048-token cap (67.3%); 2 unjudged | No |

: Serving conditions of the original campaign runs, by lane.

(sources: per-lane receipts under `outputs/modal_e1_e14/2026-08-16/`, `outputs/public_portfolio_2026-09-05/`, `outputs/e2_frontier_swe/`, `outputs/e5_apex_agents/`, `outputs/e7_binaryaudit/`, `outputs/e9_mle_bench/`, and `outputs/PES_Phase2_Review_2026-09-12/finish/e13_control/`; bridge bindings in `outputs/modal_e1_e14/2026-08-16/non_e11_readiness/`)

Two consequences follow. First, the lanes served through the bridge (E2, E4, E5, E7 and the E9 streaming arm) are exposed to the same defect that produced the E4 zero, and none of their receipts records the stop, thinking or token settings, so their figures may measure the harness as much as the model. Second, E8, E11 and E14 are dominated by format and length limits, so their headline figures are end-to-end harness scores; §8.2 gives each with its parseable-only sensitivity figure. A note on the scope of the stop-sequence mechanism is in Appendix D.

## 8.2 Suites with strictly complete scopes

Three suites reached strictly complete scopes, and the ledger's own integrity check asserts exactly this set: E8, E10 and E11 (source: `outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`, check `strict_completed_suites_are_E8_E10_E11`). E8 and E10 are replacement scopes. E11 is an original-contract full-suite result that also meets the strict-completeness check, because the suite evaluated is VerilogEval itself; the ledger lists it both among its replacement scopes and among its original-contract results, but it is one run and one result. A fourth, E14, closed terminal-complete under its native protocol with a recorded note. The six replacement scopes completed on 27 September postdate that check and are reported separately in §8.4.

**E8 --- LAB-Bench, public split.** All 1,967 of 1,967 expected items were evaluated across eight categories. The receipt records 450 correct and 1,517 incorrect responses, a published overall accuracy of 450/1967 = 0.2288 (Wilson 95% CI [0.211, 0.248], unit = question, n = 1,967); per-category figures are in Appendix D. For 1,259 of the 1,967 responses the native pipeline extracted no answer (`native_answer_parse_none`), and those are counted as incorrect, so the figure is an end-to-end harness score (source: `outputs/public_portfolio_2026-09-05/labbench_final_diagnostics.json`). As a sensitivity figure, not the headline, accuracy conditional on a parsed answer is 450/708 = 0.636 (Wilson 95% CI [0.600, 0.670]). The headline therefore mostly measures the 64% of responses that yielded no answer, with length truncation (1,086 finishes) the dominant failure, and not domain knowledge.

**E10 --- AgentDojo, benign utility scope.** Over the frozen AgentDojo v1.2.2 benchmark at commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, the terminal native receipt records status `COMPLETE_NATIVE_BENIGN_UTILITY`, all 97 episodes completed, 88 utility passes and a score of 88/97 = 0.9072 (Wilson 95% CI [0.833, 0.950], unit = episode, n = 97). Its claim boundary is stated verbatim in the receipt as "Native default benign task utility only;no prompt-injection security or held-out claim", and its decontamination status is `TRAINING_INVENTORY_ABSENT` (source: `outputs/public_portfolio_2026-09-05/agentdojo_native_receipt.json`). Completion and utility are distinct: the lane is complete in coverage and 0.9072 in benign utility, never "97/97 correct", and it is not a security result of any kind. An earlier attempt stopped at 94 of 97 episodes because of a collector fault, not a model failure; the fault and its minimal repair are recorded in Appendix D.

**E11 --- VerilogEval, two native framings.** All 312 pinned problems were evaluated --- 156 code-completion (`code-complete-iccad2023`) and 156 specification-to-RTL --- under the upstream NVlabs/verilog-eval harness. The result is 129 passes of 312, or 0.4135 (Wilson 95% CI [0.360, 0.469], unit = problem), decomposed into 67/156 = 0.4295 ([0.354, 0.508]) on code-completion and 62/156 = 0.3974 ([0.324, 0.476]) on specification-to-RTL. The receipt records 150 extraction failures, which appear as FAIL verdicts in the denominators rather than being dropped (source: `outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json`). 129/312 is therefore an end-to-end harness score: 150 of the 183 failures are format failures. As a sensitivity figure, not the headline, the pass rate among the 162 responses from which code was extracted is 129/162 = 0.796 (Wilson 95% CI [0.728, 0.851]). One problem, `verilog_eval/spec-to-rtl/Prob099_m2014_q6c`, fails against its own test bench under both simulators; excluding it yields 129/311 = 0.4148, but no immutable justification for the exclusion was captured, so the raw 129/312 is canonical and 129/311 is a noncanonical sensitivity only (sources: `outputs/e11_verilog_eval/e11_verilog_eval_rerun_receipt.json`, `outputs/e11_verilog_eval/lane_status_2026-08-09.md`). Harness revisions, the verdict rule and the amendment are in Appendix D.

**E14 --- Omni-MATH replacement scope, terminal-complete.** The scope accepted 4,426 of 4,428 rows and reproduced the official scorer's output of 0.5131, which is 2,271/4,426: the scorer omits the two unjudged rows from its denominator. Under the campaign evidence rule those rows count as failures, so the reported accuracy is 2,271/4,428 = 51.29 per cent, a difference of 0.02 percentage points. For both rows the Omni-Judge output is truncated before any verdict, so the official scorer's omission is correct behaviour rather than a parser defect, and no re-judge was performed (source: `outputs/PES_Phase2_Review_2026-09-12/finish/e14_terminal_note_2026-09-19.json`).

## 8.3 Original-contract results

Seven lanes have an original-contract figure, reported separately from any replacement scope with its coverage caveats; two are full-suite results. Lane detail is in Appendix D.

- **E1 --- SWE-bench Pro.** 2 of 731 tasks resolved, a pass@1 of 2/731 = 0.00274, or 0.274 per cent (Wilson 95% CI [0.08%, 0.99%], unit = task), over the complete 731-row test split of `ScaleAI/SWE-bench_Pro`. The fourteen generation failures and four lost artifacts stay in the denominator (sources: `outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/receipt.json`, `outputs/e1_e14_results_2026-09-05/E1_E14_Results.md`).
- **E2 --- FrontierSWE.** 1 of 17 tasks evaluated, replay-normalised score 0.8628; partial, because revision-bound authorisation for the remaining sixteen was never obtained (source: `outputs/e1_e14_results_2026-09-05/E1_E14_Results.md`).
- **E4 --- BankerToolBench.** 1 of 100 tasks evaluated, recovery metric 0.3115 --- a verifier recovery metric on a single task, not the simple fraction 37/128 and not a suite score; the pass3/pass16 archive is missing (source: `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`).
- **E5 --- APEX-Agents.** 7 of 480 tasks natively scored. The prefix mean of 0.050505 treats unscored attempts as zero and is not a native 480-task score; the seven scored tasks average 0.079365 (source: `outputs/e1_e14_results_2026-09-05/E1_E14_Results.md`).
- **E7 --- BinaryAudit.** One of 46 attempts recorded, verifier reward 0.0 after an agent error, no grade (source: `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md`).
- **E9 --- MLE-bench.** 40 of 75 competitions carry native grades, a coverage of 53.33 per cent, and the suite score is null; a separate merged-vLLM arm holds one valid grade (H&M, 0.02132) and is not combined (source: `outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json`).
- **E11 --- VerilogEval** is reported in §8.2; it is the second original-contract full-suite result, alongside E1, and not an additional one.

No lane here has a matched base-model run, so none licenses a claim that the actor outperforms its untrained counterpart, a predecessor model, or a published number.

## 8.4 Replacement-scope lanes completed on 27 September 2026

At the 19 September ledger of record, six replacement scopes were started but incomplete: E1 (SWE-bench Multilingual, 35 of 300 graded), E2 (CORE-Bench, 45 capsule setups, no grades), E5 (Tau3, 20 of 97 cleaned) and E13 (BALROG, 13 of 255 episodes) were launch-pending, and E6 (WebArena, 0 of 812) and E9 (MLDevBench, 0 of 34) were `PENDING_QUOTA` because AWS granted one vCPU per region (sources: `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`, `outputs/UNBLOCK_CARRYOUT_2026-09-21.md`). All six were run to completion on 27 September.

The override was self-authorised. `outputs/finish_pending_2026-09-27/AUTHORIZATION.json` records its issuer as "lead (repository owner arvindcr4), via Claude Code session", its directive verbatim as "finish all pending lanes", recorded at 2026-09-27T00:41Z, with per-lane spending caps, and states that it supersedes the lead-issued bound-authorisation receipt, the reservation seals and the GCP owner-IAM binding. The candidate, as project lead, therefore issued the directive that set aside gates he had himself put in place; no independent party approved it. The evaluation-governance claim of this thesis is scoped accordingly. It does not mean that fail-closed gates were never overridden. It means that every override and every deviation from the reference protocol is recorded in the lane's own result file, named per lane (Table 8.D), and never pooled: no replacement-scope figure is combined with an original-contract one. All six lanes were served the same actor, `pavlov-public-portfolio-bf16`, through vLLM with the non-thinking chat template and not through the Tinker bridge of §8.1.2, although the result files do not record stop settings.

**Table 8.C — Replacement-scope results, 27 September 2026.** One run per item, native grader. Errors, timeouts and ungraded items count as failures in the denominator. Graded counts items with a native verdict. Intervals are Wilson 95% for rates; E13's uses BALROG's standard error, and its score is the mean of six per-environment means.

| Lane | Benchmark (scope) | Attempted | Graded | Metric | Result | 95% CI | Spend (USD) |
|----|---------------------|-------|------|-----------|-----------|-----------|-----|
| E1 | SWE-bench Multilingual (190 tasks not attempted before) | 190 | 57 | resolved rate | 1/190 = 0.005 | [0.001, 0.029] | 1.87 |
| E2 | CORE-Bench hard (all 45 capsules) | 45 | 45 | task accuracy | 27/45 = 0.600 | [0.455, 0.730] | 8.83 |
| E5 | Tau3 banking (all 97 tasks) | 97 | 72 | pass^1 | 8/97 = 0.082 | [0.042, 0.154] | 18.42 |
| E6 | WebArena (all 812 tasks) | 812 | 704 | success rate, lower bound | 90/812 = 0.111 | [0.091, 0.134] | 13.46 |
| E9 | ML-Dev-Bench (all 34 tasks) | 34 | 34 | success rate | 10/34 = 0.294 | [0.168, 0.462] | 1.67 |
| E13 | BALROG (all 255 episodes, 6 environments) | 255 | 255 | progression | 26.1% | [22.6, 29.6] | 0.00 |

(sources: `outputs/finish_pending_2026-09-27/<lane>/result.json` for each lane. Spend excludes the shared actor endpoint, which was billed to its own \$150 cap and is not broken down by lane. E5's figure includes its dedicated actor GPU (\$15.21). The E2, E6 and E9 figures are list-price estimates and E1's is a Modal sandbox estimate)

**Table 8.D — Deviations from the fail-closed protocol, 27 September 2026.** Condensed from the `deviations` field of each lane's result file; the full text is in that file.

| Lane | Deviation from reference protocol/scaffold | Recorded in |
|--|------------------------------------------|--------|
| E1 | Actor at 32,768-token context (65,536 on 12 Sept), per-task `max_tokens` cap; source-context collector re-implemented; thin runner `run_remaining.py` bypassing paperwork gates, native eval in Modal sandboxes; batches r06/r07 harvested from orphaned sandboxes (exec stdout/returncode lost); two driver relaunches reusing, never resampling, completed generations; context-overflow short-circuit | `E1/result.json` |
| E2 | Per-task GCP VMs instead of native Azure shapes; thin `bash`/`query_image`/`finish` loop instead of the CORE-Agent scaffold (native 8,100 s budget, 150-turn cap, 26k-token window); `query_image` answered by the same actor; infrastructure aborts rerun from scratch, none because of score | `E2/result.json` |
| E5 | Thin runner `code/run_tau3.sh` replaces the fail-closed `e5_successor27` controller; dedicated 65k-context actor copy; OpenAI roles through a spend-capped relay | `E5/result.json` |
| E6 | GCP VM instead of the AWS AMI; thin runner `code/e6_driver.py` mirroring native `run.py`; `max_tokens` 384, few-shot system turns mapped to user/assistant; retired native judge swapped to `openai/gpt-4.1`, which had no credit (108 tasks ungraded, counted as failures); 1,500 s per-task timeout; last 103 shopping-admin tasks sharded over three containers of the official image | `E6/result.json` |
| E9 | GCP instead of AWS; backported one-line OpenHands context-window fix; loopback relay forcing non-thinking and restoring dropped tool-call text; `max_input_tokens` 28,000; one run per task | `E9/result.json` |
| E13 | Non-thinking template (original run had thinking on); NetHack capped at 2,000 steps (never bound); macOS arm64 instead of the Linux x86 image, dynamics not bit-verified; killed or hung episodes rerun from scratch with new seeds | `E13/result.json` |

(sources: `outputs/finish_pending_2026-09-27/<lane>/result.json`, field `deviations`; `outputs/finish_pending_2026-09-27/AUTHORIZATION.json`)

The six figures are on six different benchmarks with six different metrics, and they are not averaged, pooled with the same lane's original-contract result, or compared with a base model. Five lanes (E2, E5, E6, E9, E13) now have a native score over their full replacement scope. E6's is a lower bound: 108 tasks needing an LLM judge could not be graded, and the true figure lies between 0.111 and 0.244. E1 has two separately reported scores, 1/190 and 4/110 (the earlier waves, served at a different context), that together cover all 300 tasks but are not combined. E5's pass^1 over the 72 tasks without an infrastructure error, 8/72 = 0.111, is a sensitivity figure only, and E13's 26.1 per cent is not compared with the 26-episode small-scale E13 rows of Tables 8.A and 8.B. The lanes spent \$44.25 in total against caps of \$270, not counting the shared actor endpoint. Per-lane narratives are in Appendix D.

## 8.5 The E4 rerun as a diagnostic

A fresh full BankerToolBench rerun, authorised separately after the lane closed `CLOSED_PARTIAL` on 19 September, ran the base model with a zero-initialised LoRA snapshot through the Modal bridge `/v1/responses` path. It completed 100 of 100 trials with one `NonZeroAgentExitCodeError` and a mean reward of exactly 0.0 (source: `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`, which cites the receipt at `outputs/e4_banker_toolbench/official_repo_ff6db552/jobs/btb-banking-tasks-tinker-bridge/result.json`). Agents stopped after 2 to 14 steps without producing deliverables, and twenty-seven or more trajectories ended in `user user assistant ...` role-token degeneration. The bridge's `/v1/responses` path passed no stop sequences, so generation ran past the end-of-turn token into invented turns; with a stop sequence set, the same base model scored 0.41 on one of six tasks in the small-scale rerun of §8.7. The 0.0 is therefore attributed to the serving bridge, not to the model's tool use or finance reasoning (source: `outputs/e1_e14_small_scale_2026-09-26/E4/BRIDGE_STOP_TOKEN_FINDING.md`). It is not set against the original single-task outcome, and no full-suite BankerToolBench score exists for this actor family. Spend was \$15.07 against a \$55.91 cap. The full audit is in Appendix D.

## 8.6 Externally blocked lanes and terminal states

Six lanes have no local or paid path and were formally closed as externally blocked under the finish-all directive of 19 September 2026, whose verbatim text and interpretation are recorded in `outputs/PES_Phase2_Review_2026-09-12/finish/ROOT_DIRECTIVE_FINISH_ALL_2026-09-19.json` at 13:50Z. The six are E3 (SDAB private bundle), E7 (BinaryAudit private payload), E8-original (the LifeSciBench package), E10-original (AgentHarm private tasks), E12 (AppBench deployment) and E14-original (the FrontierMath hosted evaluation). Closure records were written for each, with an index, under `outputs/PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/`; each records "no provider response" and specifies the conditions under which the lane could be reopened (sources: `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md`, `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`). A seventh original contract, E13's OpenReward held-out games, is also blocked on external access and is listed with the six in the ledger, but it has no closure record because the lane continues through its BALROG replacement scope (§8.4). Access requests, the deterministic ledger checks and the role of the model-based triage harness are recorded in Appendix D.

<!-- small-scale:begin -->
## 8.7 Small-scale reruns of all fourteen lanes

This section adds a separate, smaller set of runs, made on 26 September 2026, to give every one of the fourteen lanes a measured number under one fixed protocol, including the lanes that never produced a score. None of these figures is pooled with, averaged against, or substituted for a result reported earlier in this chapter.

The protocol fixes one actor family, a fixed item-selection seed (20260926), temperature 0 and a non-thinking chat template, and scores each lane with its native grader where one could be run. Lanes whose original benchmark is externally blocked were run on a public substitute, and each substitute's gap from the original suite is recorded in its lane file. Items that errored or timed out are counted as failures in the denominator (source: `outputs/e1_e14_small_scale_2026-09-26/PROTOCOL.md`).

Two arms were run. The first is the base model, `Qwen/Qwen3.6-35B-A3B` with no adapter, sampled on Tinker (Table 8.A). The second is a paired comparison of the trained actor against the same base model, both served by one vLLM configuration on Modal, on identical items (Table 8.B). The pairing exists because the adapter's effect on individual token probabilities was measured to be about the same size as the numerical difference between the two serving engines on the same base weights, so a trained-on-vLLM against base-on-Tinker comparison would confound the adapter with the engine, and none is made (source: `outputs/e1_e14_small_scale_2026-09-26/TRAINED_ACTOR_ENDPOINT.md`).

**Table 8.A — Base model on Tinker, small-scale subsets.**

| Lane | Benchmark run | Scope | Metric | Result | 95% CI |
|---|---|---|---|---|---|
| E1 | SWE-bench Pro public test split | original-public-subset | resolved rate | 0/10 = 0.000 | [0.00, 0.28] |
| E2 | EffiBench | substitute | pass@1 | 23/30 = 0.767 | [0.59, 0.88] |
| E3 | Terminal-Bench v2.0 | substitute | task success rate | 4/8 = 0.500 | [0.22, 0.78] |
| E4 | BankerToolBench (official Harbor tasks) | original-public-subset | mean verifier reward | 0.068 (n = 6; one nonzero task, 0.409) | — (not estimable) |
| E5 | tau2-bench | substitute | pass^1 | 3/10 = 0.300 | [0.11, 0.60] |
| E6 | MiniWoB++ | substitute | episode success rate | 15/36 = 0.417 | [0.27, 0.58] |
| E7 | BinaryAudit upstream cbd86c7 | original-public-subset | task success rate | 0/6 = 0.000 | [0.00, 0.39] |
| E8 | LAB-Bench public MCQ | substitute | accuracy | 26/80 = 0.325 | [0.23, 0.43] |
| E9 | MLE-bench lite/low-complexity split | original-public-subset | any-medal rate | 1/5 = 0.200 | [0.04, 0.62] |
| E10 | AgentHarm test_public | original-public-subset | harmful refusal rate | 20/30 = 0.667 | [0.49, 0.81] |
| E11 | VerilogEval | original-public-subset | pass@1 | 35/50 = 0.700 | [0.56, 0.81] |
| E12 | AppBench public tasks | original-public-subset | rubric-item pass rate | 122/151 = 0.808 | [0.74, 0.86] |
| E13 | BALROG | substitute | mean progression (%) | 23.1% | — |
| E14 | Omni-MATH test | substitute | accuracy | 38/100 = 0.380 | [0.29, 0.48] |

**Table 8.B — Trained actor versus base model on the same vLLM engine, identical items.**

| Lane | Items | Trained | Base | Trained − base | Test |
|---|---|---|---|---|---|
| E1 | 10 | 0.000 | 0.000 | 0.000 | McNemar p = 1 (b/c 0/0) |
| E2 | 30 | 0.700 | 0.800 | -0.100 | McNemar p = 0.25 (b/c 0/3) |
| E3 | 8 | 0.625 | 0.375 | 0.250 | McNemar p = 0.5 (b/c 2/0) |
| E4 | 6 | 0.066 | 0.093 | -0.027 | one nonzero pair (−0.160); no interval |
| E5 | 10 | 0.500 | 0.200 | 0.300 | McNemar p = 0.38 (b/c 4/1) |
| E6 | 36 | 0.444 | 0.472 | -0.028 | McNemar p = 1 (b/c 1/2) |
| E7 | 6 | 0.167 | 0.000 | 0.167 | McNemar p = 1 (b/c 1/0) |
| E8 | 80 | 0.312 | 0.312 | 0.000 | McNemar p = 1 (b/c 6/6) |
| E9 | 5 | 0.000 | 0.000 | 0.000 | McNemar p = 1 (b/c 0/0) |
| E9 (above median) | 5 | 0.000 | 0.200 | -0.200 | McNemar p = 1 (b/c 0/1) |
| E10 (refusal rate) | 30 harmful, 30 benign | 0.700 | 0.667 | 0.033 | McNemar p = 1 (b/c 3/2) |
| E10 (harm score) | 30 harmful, 30 benign | 0.247 | 0.236 | 0.011 | 95% CI [-0.053, 0.078] |
| E10 (benign score) | 30 harmful, 30 benign | 0.780 | 0.794 | -0.015 | 95% CI [-0.050, 0.012] |
| E11 | 50 | 0.660 | 0.660 | 0.000 | McNemar p = 1 (b/c 2/2) |
| E12 | 151 | 0.748 | 0.781 | -0.033 | McNemar p = 0.3 (b/c 5/10); app-level sign-flip p = 0.25 (n = 6) |
| E13 | 26 | 25.510 | 23.630 | 1.875 | 95% CI [-2.73, 7.09] |
| E14 | 100 | 0.430 | 0.440 | -0.010 | McNemar p = 1 (b/c 5/6) |

These are small samples. Most lanes score between five and eighty items, so a single item moves a lane's rate by several percentage points and the confidence intervals are wide. In Table 8.B, b and c count items solved only by the trained actor and only by the base model respectively, and the CI is a paired bootstrap interval on the difference. Four lane-specific caveats matter. For E12 the 151 rubric items come from only six generated applications, so its McNemar p-value overstates the evidence; the exact sign-flip test over the six per-app differences gives p = 0.25, and the grader is an LLM judge that is the base model itself, so the E12 rates are judge-dependent. For E4, only `btb-19b3361c` scores above zero in either arm (trained 0.397, base 0.557; 0.409 in Table 8.A), so no interval is valid. For E13, the 1.875 difference is the mean of per-environment differences computed from unrounded values (recomputed 1.876), not 25.510 − 23.630. For E9, the one competition that earned a medal in the Tinker run (nomad2018) has a 46.8k-token prompt that exceeds the 32,768-token vLLM limit and fails identically in both vLLM arms, so the vLLM medal rates are not comparable with Table 8.A. The paired differences are not combined across lanes, and none of the seventeen paired tests excludes zero. At these sizes that is an inconclusive result, not evidence of no effect: with 5 to 10 items, an exact McNemar test needs at least six discordant items, all in one direction, to reach p < 0.05. No equivalence test was run. Fuller E12 detail and the principal per-lane caveats from each result file are in Appendix D.
<!-- small-scale:end -->

## 8.8 Summary of the campaign at the point of writing

By the 19 September ledger the campaign had established two full-suite results and one complete replacement scope with a score: E1 at 2/731 = 0.274 per cent on SWE-bench Pro, E11 at 129/312 = 41.35 per cent pass@1 (Wilson 95% CI [36.0, 46.9]) across its two framings, and E8 at 450/1967 = 0.2288 ([0.211, 0.248]) on the LAB-Bench public split, alongside a complete benign-utility evaluation of E10 at 88 of 97 episodes (0.907, [0.833, 0.950]), a terminal-complete E14 replacement scope at 2,271/4,428 = 51.29 per cent accuracy, and five original-contract partials (E2, E4, E5, E7, E9), none of which is a suite score. On 27 September six further replacement scopes were completed under a self-issued override whose deviations are recorded per lane (Tables 8.C and 8.D). Seven original contracts remain externally blocked, and no lane now waits on a launch or a cloud quota. The campaign did not establish a multi-seed estimate, a held-out security result, a BankerToolBench suite score, a graded result for the 108 WebArena tasks that need an LLM judge, or a baseline comparison for the campaign lanes themselves; the only trained-versus-base comparison, Table 8.B, is inconclusive in every lane. The evidence chain and reproduction path for each number are given in Appendix B; the next chapter draws the cross-study synthesis and conclusions.
