# Appendix C. Notation, Symbols and Abbreviations

## C.1 Scope and convention

This appendix records the symbols and abbreviations used across this report and the papers behind it, as the sources actually use them rather than as a house style would prefer. Where a symbol carries more than one meaning across documents, both are listed and the collision flagged, because a reader carrying one definition into another chapter will misread a figure.

## C.2 Symbols

| Symbol | Meaning | As used in the sources |
|---|---|---|
| $G$ | Group size: completions sampled per prompt | Headline runs $G=8$; sweeps $G \in \{2,4,8,16\}$ |
| $K$ | Rollouts per group in the ZVF formalisation; **separately**, successes within a group | Elsewhere $K = \sum_i R_i \sim \mathrm{Binomial}(G,p)$ |
| $B_t$, $\mathcal{P}_t$ | Batch at optimizer step $t$; prompts at step $t$ | Competing denominators of ZVF |
| $p$, $\hat{p}$ | Latent success probability; sample rate $K/G$ | $\mathrm{Bernoulli}(p)$ model; $\hat{p}$ centres the advantage |
| $r_i$, $R_i$ | Scalar reward of completion $i$; binary reward | Verifiable reward; derivations appendix |
| $h_G(p)$ | $p^G + (1-p)^G$ | Bernoulli null for ZVF |
| $\mathrm{ZVF}_t$, $\mathrm{GU}_t$ | Zero-Variance Fraction: share of identical-reward groups; Gradient Utilization $1-\mathrm{ZVF}_t$ | Principal diagnostic and its complement |
| $A_i$ | Advantage of completion $i$ | Unnormalised $R_i-\hat{p}$; normalised $(r_i-\mu_g)/(\sigma_g+\varepsilon)$; clipped Dr.GRPO |
| $\mu_g$, $\sigma_g$ | Group mean and standard deviation of reward | $\mu_g = \frac{1}{G}\sum_j r_j$; $\sigma_g$ in population form ($1/G$) throughout, as every GRPO-style runner computes it (§4.2) |
| $\pi_\theta$, $\pi_{\theta_{\mathrm{old}}}$, $\pi_{\mathrm{ref}}$ | Trained, sampling and frozen reference policy; per-token importance ratio $\rho_{i,t}$ | Ratio against the sampling policy; KL penalty against the reference |
| $\mathcal{L}(\theta)$ | Policy loss | Original GRPO: clipped surrogate plus KL penalty. In this project no runner has a KL term; the Tinker runners are REINFORCE with a group-mean baseline (Table 4.2) |
| $\beta$ | KL coefficient; **separately**, a regression slope | TRL and veRL 0.04; OpenRLHF 0.02; Tinker managed; also a slope in M1/M2 |
| $\varepsilon$ | Three distinct meanings — see §C.3 | Clip range 0.2; ZVF tolerance $10^{-6}$; Adam term $10^{-8}$ |
| $\gamma$, $\lambda$, $T$ | Discount factor; GAE parameter; sampling temperature | $0.99$; $0.95$; canonical $0.8$ |
| $\mathrm{SI}$, $\mathrm{PTD}$ | Stability index; peak-to-tail drift | Reported in place of direct KL |
| PCD | Pairwise-contrast density | $\mathbb{E}[\hat{p}(1-\hat{p})] = \frac{G-1}{G}\,p(1-p)$ |
| pass@1 | Pass rate under the suite's native scorer | E11 $129/312 = 41.35\,\%$; E1 $2/731 = 0.274\,\%$ |
| $r$, $\rho$ | Correlation coefficients, Pearson and rank | $r = -0.769$, $p = 0.0008$; $\rho \approx 0.65$ |

: Notation: symbols used in this report, their meaning, and any divergence in how the underlying sources use them.

## C.3 Four collisions worth flagging

The symbol $\varepsilon$ carries three unrelated meanings. In the hyperparameter mapping it is the PPO/GRPO clip range, default 0.2, swept over $\{0.05, 0.1, 0.2, 0.3, 0.5\}$. In the ZVF definition it is a numerical tolerance, $10^{-6}$, deciding whether a group's sample variance counts as degenerate. In the optimiser configuration it is the Adam epsilon, $10^{-8}$. It also appears additively in the advantage denominator, $\sigma_g + \varepsilon$.

$K$ is used two ways: in the formal treatment of ZVF it denotes the rollouts in a group, a synonym for the group size elsewhere called $G$, whereas in the derivations appendix $G$ is the group size and $K$ is the number of successes — a design parameter in the one case, an outcome in the other. The letter $r$ is similarly the reward, the LoRA rank (swept over $\{4,8,16,32,64\}$) and the Pearson correlation.

Finally, ZVF is expanded two ways in the sources. The formal definition and the abstract both read "Zero-Variance Fraction", the expansion used here; one figure caption reads "Zero-Value-Filtered". The caption variant is noted only so that a reader meeting it is not misled.

## C.4 Abbreviations

| Abbreviation | Expansion or identity |
|---|---|
| GRPO | Group Relative Policy Optimization — critic-free, group-relative |
| PPO | Proximal Policy Optimization |
| DPO | Direct Preference Optimization |
| RLHF | Reinforcement learning from human feedback |
| IS Loss | Importance-sampling loss, the distillation objective |
| SFT | Supervised fine-tuning |
| LoRA | Low-rank adaptation of large language models |
| KL | Kullback–Leibler divergence; never spelled out in the sources |
| CoT | Chain-of-thought; spelled out in prose, written "GSM8K CoT" for the variant |
| ZVF, GU | Zero-Variance Fraction; Gradient Utilization |
| PCD, SI, PTD | Pairwise-contrast density; stability index; peak-to-tail drift |
| Dr.GRPO | GRPO with the per-response length normalisation and the division by the group standard deviation removed |
| GAE | Generalised advantage estimation; $\lambda = 0.95$ |
| MoE | Mixture-of-experts |
| BF16 | bfloat16, the serving precision for the LAB-Bench, AgentDojo and Omni-MATH lanes |
| IQM | Interquartile mean, reported via `rliable` |
| CI | Confidence interval; 95 %, 10,000 bootstrap resamples |
| W&B | Weights & Biases, the experiment-tracking service |
| MLE-bench | The suite forming lane E9's original contract; MLE never appears standalone |
| MATH-500 | The mathematics evaluation suite |
| GSM8K | The grade-school arithmetic suite; 7,473 training and 1,319 test rows |
| HumanEval | The code-generation suite used for the code arm |
| ArenaHard, NoRobots, OpenThoughts3 | Dataset names used as-is, each at a pinned revision |
| MiniWoB | Browser micro-task suite used for the proxy smoke test |
| SWE-bench (Pro, Multilingual) | The repository-repair suite forming lane E1 |
| LAB-Bench, AgentDojo, VerilogEval, Omni-MATH, BALROG, WebArena, APEX-Agents, CORE-Bench, Tau3, MLDevBench, SDAB, BankerToolBench, BinaryAudit, LifeSciBench, AgentHarm, WebBench, AppBench, OpenReward Games, FrontierSWE, FrontierMath | Suite names used as names, with no expansion in any source; see §C.5 |
| Tinker | The remote training and sampling API used to train the adapter |
| TRL | HuggingFace's LLM post-training library, in both framework rosters |
| SkyRL, verl (veRL), OpenRLHF, Atropos | Launchers in the cross-launcher roster; only Tinker and TRL produced completed runs, the veRL and OpenRLHF entries being dry-run placeholders |

: Abbreviations and their expansions, with names that no source expands recorded as identities rather than glossed.

## C.5 The E1–E14 lane identifiers, scopes and statuses

Each lane's short code names one original benchmark contract and is not a rank or a score. Several lanes hold two states at once — an original contract and a replacement scope — so the code alone does not say which scope a figure comes from. Table C.3 is the single record of lane scope, headline figure, terminal state and receipt; the chapters point here rather than restating it. "orig" is the original contract and "repl" a declared replacement scope. Terminal states are from `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md`; states for the E1, E2, E5, E6, E9 and E13 replacement scopes are as of 2026-09-21 and those lanes are being rerun. Receipt paths are relative to `outputs/`.

| Lane | Original contract | Replacement scope | Headline figure (scope) | Terminal state | Receipt |
|---|---|---|---|---|---|
| E1 | SWE-bench Pro | SWE-bench Multilingual | 2/731 = 0.274% pass@1 (orig, full suite); 35/300 graded (repl) | orig: complete, no ledger label; repl: `REBUILD_READY_LAUNCH_PENDING` | `modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/receipt.json` |
| E2 | FrontierSWE | CORE-Bench | 1/17 tasks, replay 0.8628 (orig, partial); 0/45 graded (repl) | `AMENDMENT_ACCEPTED_LAUNCH_PENDING` | `e2_frontier_swe/e2_terminal_attempt_receipt_2026-08-22.json` |
| E3 | SDAB (private bundle) | — | none | `CLOSED_EXTERNAL` | `PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/E3_sdab.json` |
| E4 | BankerToolBench | — | 1/100 tasks, recovery metric 0.3115 (orig, partial) | `CLOSED_PARTIAL` | `modal_e1_e14/2026-08-16/e4_recovery_pass16_receipt.json` |
| E5 | APEX-Agents | Tau3 | 7/480 natively scored (orig, partial); 20/97 cleaned (repl) | `REBUILD_READY_LAUNCH_PENDING` | `e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json` |
| E6 | WebBench | WebArena | 0/812 (repl) | `PENDING_QUOTA` | `PES_Phase2_Review_2026-09-12/finish/e6_continuation/status_2026-09-19.json` |
| E7 | BinaryAudit (private payload) | — | 1/46 attempted, verifier reward 0.0, no grade (orig) | `CLOSED_EXTERNAL` | `e7_binaryaudit/2026-08-22_e7_paid_attempt_receipt.json` |
| E8 | LifeSciBench (private package) | LAB-Bench public split | 450/1,967 = 0.2288 (repl, complete; end-to-end harness score) | repl: `COMPLETE`; orig: `CLOSED_EXTERNAL` | `public_portfolio_2026-09-05/labbench_native_receipt.json` |
| E9 | MLE-bench | MLDevBench | 40/75 competitions graded, suite score null (orig, partial); 0/34 graded (repl) | `PENDING_QUOTA` | `e9_mle_bench/modal_streaming/`; `PES_Phase2_Review_2026-09-12/finish/e9_completion/status_2026-09-19.json` |
| E10 | AgentHarm (private tasks) | AgentDojo benign utility | 88/97 = 0.9072 utility, 97/97 completed (repl, complete) | repl: `COMPLETE`; orig: `CLOSED_EXTERNAL` | `public_portfolio_2026-09-05/agentdojo_native_receipt.json` |
| E11 | VerilogEval | none (two native framings) | 129/312 = 41.35% pass@1 (orig, complete; end-to-end harness score) | `COMPLETE` | `modal_e1_e14/2026-08-16/e11_full_receipt.json` |
| E12 | AppBench (deployment) | — | none | `CLOSED_EXTERNAL` | `PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/E12_appbench.json` |
| E13 | OpenReward Games (held-out games) | BALROG | 13/255 episodes (repl, partial) | repl: `AMENDMENT_ACCEPTED_LAUNCH_PENDING`; orig: externally blocked, no closure record | `PES_Phase2_Review_2026-09-12/finish/e13_control/E13-native-20260912-01-completed13-native-score.json` |
| E14 | FrontierMath (hosted) | Omni-MATH | 2,271 correct of 4,428 dispositions (repl, terminal-complete; §9.2) | repl: `COMPLETE_TERMINAL_NOTE`; orig: `CLOSED_EXTERNAL` | `public_portfolio_2026-09-05/native_finish_v5/e14_official_collector/official_result01/native_score.json` |

: The E1–E14 lanes: original contract, replacement scope, headline figure, terminal state and receipt.

Counted from this table: three scopes are strictly complete (E8 and E10 replacement scopes, E11 original contract), E14 is terminal-complete, and six original contracts are `CLOSED_EXTERNAL` with written closure records (E3, E7, E8-original, E10-original, E12, E14-original). E13's original contract is also blocked on external access but carries no closure record, because the lane continues through its replacement scope. Original-contract full-coverage scores exist for E1 and E11.
