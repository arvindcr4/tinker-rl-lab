## Small-scale reruns of all fourteen lanes

The lane results above were produced by the trained actor at the time of the campaign and are left exactly as recorded. This section adds a separate, smaller set of runs, made on 26 September 2026, whose purpose is narrower: to give every one of the fourteen lanes a measured number under one fixed protocol, including the lanes that never produced a score. None of the figures below is pooled with, averaged against, or substituted for an original-contract or replacement-scope result reported earlier in this chapter.

The protocol fixes one actor family, a fixed item-selection seed (20260926), temperature 0 and a non-thinking chat template, and scores each lane with its native grader where one could be run. Lanes whose original benchmark is externally blocked were run on a public substitute, and each substitute's gap from the original suite is recorded in its lane file. Items that errored or timed out are counted as failures in the denominator. Each lane's items, raw outputs and grader logs are kept so that every figure can be recomputed (source: `outputs/e1_e14_small_scale_2026-09-26/PROTOCOL.md`).

Two arms were run. The first is the base model, `Qwen/Qwen3.6-35B-A3B` with no adapter, sampled on Tinker (Table 9.A). The second is a paired comparison of the trained actor against the same base model, both served by one vLLM configuration on Modal, on identical items (Table 9.B). The pairing exists because the adapter's effect on individual token probabilities was measured to be about the same size as the numerical difference between the two serving engines on the same base weights. A comparison of trained-on-vLLM against base-on-Tinker would therefore confound the adapter with the engine, and no such cross-engine comparison is made here (source: `outputs/e1_e14_small_scale_2026-09-26/TRAINED_ACTOR_ENDPOINT.md`).

**Table 9.A — Base model on Tinker, small-scale subsets.**

| Lane | Benchmark run | Scope | Metric | Result | 95% CI |
|---|---|---|---|---|---|
| E1 | SWE-bench Pro public test split | original-public-subset | resolved rate | 0/10 = 0.000 | [0.00, 0.28] |
| E2 | EffiBench | substitute | pass@1 | 23/30 = 0.767 | [0.59, 0.88] |
| E3 | Terminal-Bench v2.0 | substitute | task success rate | 4/8 = 0.500 | [0.22, 0.78] |
| E4 | BankerToolBench official Harbor tasks… | original-public-subset | mean native verifier rewa… | 0.068 (n = 6) | ≈ [0.01, 0.47] |
| E5 | tau2-bench | substitute | pass^1 | 3/10 = 0.300 | [0.11, 0.60] |
| E6 | MiniWoB++ | substitute | episode success rate | 15/36 = 0.417 | [0.27, 0.58] |
| E7 | BinaryAudit upstream cbd86c7 | original-public-subset | task success rate | 0/6 = 0.000 | [0.00, 0.39] |
| E8 | LAB-Bench public MCQ | substitute | accuracy | 26/80 = 0.325 | [0.23, 0.43] |
| E9 | MLE-bench lite/low-complexity split | original-public-subset | any-medal rate | 1/5 = 0.200 | [0.04, 0.62] |
| E10 | AgentHarm test_public | original-public-subset | harmful refusal rate | 20/30 = 0.667 | [0.49, 0.81] |
| E11 | VerilogEval | original-public-subset | pass@1 | 35/50 = 0.700 | [0.56, 0.81] |
| E12 | AppBench public tasks | original-public-subset | rubric-item pass rate | 122/151 = 0.808 | [0.74, 0.86] |
| E13 | BALROG | substitute | BALROG native progression… | 23.1% | — |
| E14 | Omni-MATH test | substitute | accuracy | 38/100 = 0.380 | [0.29, 0.48] |

**Table 9.B — Trained actor versus base model on the same vLLM engine, identical items.**

| Lane | Items | Trained | Base | Trained − base | Test |
|---|---|---|---|---|---|
| E1 | 10 | 0.000 | 0.000 | 0.000 | McNemar p = 1 (b/c 0/0) |
| E2 | 30 | 0.700 | 0.800 | -0.100 | McNemar p = 0.25 (b/c 0/3) |
| E3 | 8 | 0.625 | 0.375 | 0.250 | McNemar p = 0.5 (b/c 2/0) |
| E4 | 6 | 0.066 | 0.093 | -0.027 | 95% CI [-0.080, 0.000] |
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
| E12 | 151 | 0.748 | 0.781 | -0.033 | McNemar p = 0.3 (b/c 5/10) |
| E13 | 26 | 25.510 | 23.630 | 1.875 | 95% CI [-2.73, 7.09] |
| E14 | 100 | 0.430 | 0.440 | -0.010 | McNemar p = 1 (b/c 5/6) |

These are small samples, and the tables should be read accordingly. Most lanes score between five and eighty items, so a single item moves a lane's rate by several percentage points and the confidence intervals are wide. In Table 9.B, b and c count items solved only by the trained actor and only by the base model respectively, and the CI is a paired bootstrap interval on the difference. For E12 the 151 rubric items come from only six generated applications, so they are not independent and its McNemar p-value overstates the evidence. For E9, the one competition that earned a medal in the Tinker run (nomad2018) has a 46.8k-token prompt that exceeds the vLLM serving limit of 32,768 tokens, so it fails identically in both vLLM arms; the vLLM medal rates are therefore not comparable with Table 9.A. The paired differences are reported with their own test and are not combined across lanes; no lane-level difference is interpreted as evidence that the adapter improves or degrades performance unless its test excludes zero, and none is extrapolated to the full suite. The principal caveat for each lane, as recorded in its result file, is listed below; the full list is in each lane's `result.json`.

- **E1.** New base-model arm; not comparable to or poolable with the original 2/731 (lost seed809 adapter, temperature 0.2).
- **E2.** Substitute benchmark; never pool with or present as the FrontierSWE lane score (original: 1/17 tasks, replay 0.8628, lost adapter).
- **E3.** New base-model arm; never pool with or present as the original SDAB/seed809 lane score.
- **E4.** Small n=6 of 100; wide interval. Wilson95 computed on summed fractional reward (approximation for a [0,1] score).
- **E5.** n=10, not the requested 20: the E4+E5 6M Tinker-token cap was exhausted (E4 1.56M, abandoned APEX attempt 2.33M, tau2 2.05M).
- **E6.** Tasks drawn by random.Random(20260926).sample from all 128 sorted miniwob env ids; episode seeds 0-2.
- **E7.** New base-model arm; never pool with or present as the original lane score.
- **E8.** New base-model arm; not comparable to or poolable with the lost-adapter 1967/1967 run (22.88%).
- **E9.** Competition pool: 6 low-split competitions with small non-image-heavy data (detecting-insults, jigsaw-toxic, leaf-classification, nomad2018, random-acts-of-pizza, spooky); 5 drawn by random.Random(20260926).sample. n=5 so the CI is very wide.
- **E10.** New base-model arm on AgentHarm itself; the lost-adapter lane figure (AgentDojo benign 97/97, 90.72%) is a different benchmark and not comparable.
- **E11.** New base-model arm; not comparable to the retained adapter receipt 129/312 = 41.35% (that run used thinking-default template, temperature 0.2, all 312).
- **E12.** Public tasks, not held-out. Label mandatory. New base-model arm; never pool with the original lane.
- **E13.** New base-model arm; not comparable to or pooled with the original seed809-LoRA E13 plan (21-episode BabyAI admission set).
- **E14.** New base-model arm; the prior 2271/4428 = 51.31% figure is not comparable (different actor, full split).
