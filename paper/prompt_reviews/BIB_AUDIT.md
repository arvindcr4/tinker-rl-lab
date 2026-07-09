# Bibliography and Consistency Audit

## paper_P1_scaling.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | `arXiv2507.18014` |

## paper_P2_zvf.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## paper_P3_group_size.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## paper_P4_length_bias.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## paper_P5_minreport.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## paper_P7_zvf_controller.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## paper_P8_fraud.tex
### Findings
| Issue Type | Details |
|---|---|
| Unresolved Key | *None* |

## Cross-Paper Consistency
### Contradictions in Claims
- **ZVF Scoping**: `paper_P2_zvf.tex` claims ZVF is a *descriptive* diagnostic, while `paper_P1_scaling.tex` repeatedly uses it as a control or gating mechanism (e.g., `paper/sections/scaling_laws.tex:171` 'motivates the variance-based ZVF gate', `paper/sections/scaling_laws.tex:1300` 'use the variance-based ZVF gate as the primary scaling signal').
- **70+ Runs & 7 Libraries**: All papers consistently report '70+ runs across seven libraries' based on the checked files.

## Suspicious Citations
The following entries could not be easily verified or appeared highly suspicious, requiring manual verification:
- [tinker-rl-lab-iter25]: {TinkerRL-Bench Team} - {TinkerRL-Bench}: Iteration-25 Scaling-Law Identifiability Analysis - **VERIFY-MANUALLY**
- [gptoss]: {OpenAI} - gpt-oss-120b and gpt-oss-20b - **VERIFY-MANUALLY**
- [frontier2026]: {TinkerRL-Bench Team} - Frontier-Model Cross-Examination of GRPO Post-Training - **VERIFY-MANUALLY**
- [gift2025]: Wang, Zhichao and others - {GIFT}: Group-Relative Implicit Fine-Tuning Integrates {GRPO} with {DPO} and {UNA} - **VERIFY-MANUALLY**
- [mcgrpo2025]: Kim, Youngeun and others - {MC-GRPO}: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning - **VERIFY-MANUALLY**
- [nan2025ngrpo]: Nan, Gongrui and others - {NGRPO}: Negative-enhanced Group Relative Policy Optimization - **VERIFY-MANUALLY**
- [liu2026gdpo]: Liu, Shih-Yang and Dong, Xin and Lu, Ximing and Diao, Shizhe and Belcak, Peter and Liu, Mingjie and Chen, Min-Hung and Yin, Hongxu and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Choi, Yejin and Kautz, Jan and Molchanov, Pavlo - {GDPO}: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization - **VERIFY-MANUALLY**
- [yu2025dapo]: Yu, Qiying and Zhang, Zheng and Zhu, Ruofei and Yuan, Yufeng and Zuo, Xiaochen and Yue, Yu and Dai, Weinan and Fan, Tiantian and Liu, Gaohong and Liu, Lingjun and others - {DAPO}: An Open-Source {LLM} Reinforcement Learning System at Scale - **VERIFY-MANUALLY**
- [zhang2026greso]: Zheng, Haizhong and Zhou, Yang and Bartoldson, Brian R. and Kailkhura, Bhavya and Lai, Fan and Zhao, Jiawei and Chen, Beidi - Act Only When It Pays: Efficient Reinforcement Learning for {LLM} Reasoning via Selective Rollouts - **VERIFY-MANUALLY**

## Prioritized Fix List (Top 10)
1. Fix unresolved citation `arXiv2507.18014` in `paper_P1_scaling.tex`
2. Merge duplicate entries for 'scaling laws for neural language models' under keys `kaplan2020scaling` and `kaplan2020scalinglaws`
3. Fix malformed arXiv ID in `bytedance2025vapo`
4. Fix malformed arXiv ID in `zhang2026greso`
5. Fix malformed arXiv ID in `miller2024erroreval`
6. Fix malformed arXiv ID in `hu2024openrlhf`
7. Fix malformed arXiv ID in `colas2019hitchhiker`
8. Fix malformed arXiv ID in `riddell2024contamination`
9. Fix malformed arXiv ID in `hochlehnert2025sober`
10. Fix malformed arXiv ID in `rafailov2024rlhfscaling`

## Counts
- **Keys checked**: 82
- **Unresolved**: 1
- **Malformed**: 30
- **Duplicates**: 1
- **Flagged-for-manual-verification**: 9