# P9_dnb — shared-file change requests (references.bib)

All metadata below was checked against the arXiv API (export.arxiv.org, 2026-10-18) by
`tools/review_ledger/P9_dnb_arxiv_check.py`; results quoted per entry.
Keys are unchanged (so no citing file has to change); only fields are corrected.
Entries already correct in the current `references.bib` (no request): `ichihara2025mogrpo` (2509.22047),
`liu2025rlsubnetworks` (Mukherjee et al., 2505.11711), `lu2025grpolead` (Zhang & Zuo).

## 1. `hu2026openvlthinker` (finding 513, CRITICAL)
arXiv:2603.04567 = "Necessary conditions for the Markovian Mpemba effect" (Avitan et al.). The G^2RPO paper is
arXiv:2604.08539 "OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks",
Wenbo Hu, Xin Chen, Yan Gao-Tian, Yihe Deng, Nanyun Peng, Kai-Wei Chang (2026).
```
old: author = {Hu, Shuhao and Wang, Chenyu and others},
     title  = {Open{VL}Thinker: Multimodal Reasoning with {G$^{2}$RPO}},
     doi = {10.48550/arXiv.2603.04567}, note = {arXiv:2603.04567}
new: author = {Hu, Wenbo and Chen, Xin and Gao-Tian, Yan and Deng, Yihe and Peng, Nanyun and Chang, Kai-Wei},
     title  = {{OpenVLThinkerV2}: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks},
     doi = {10.48550/arXiv.2604.08539}, note = {arXiv:2604.08539}
```

## 2. `yang2025sspo` (finding 517, CRITICAL)
arXiv:2511.09983 = circle-packing geometry paper. SSPO is arXiv:2511.04256 "SSPO: Subsentence-level Policy
Optimization", Kun Yang, Zikang Chen, Yanmeng Wang, Zhigen Li, Ning Cheng, Shaojun Wang, Jing Xiao (2025).
```
new: author = {Yang, Kun and Chen, Zikang and Wang, Yanmeng and Li, Zhigen and Cheng, Ning and Wang, Shaojun and Xiao, Jing},
     title  = {{SSPO}: Subsentence-level Policy Optimization},
     doi = {10.48550/arXiv.2511.04256}, note = {arXiv:2511.04256}
```
(P9 text in related_work_v2 changed "sentence granularity" -> "subsentence granularity" to match.)

## 3. `zhang2025rewardstep` (finding 518, CRITICAL)
arXiv:2502.02508 = "Satori" (Shen et al.). The cited title is arXiv:2310.10080 "Let's reward step by step:
Step-Level reward model as the Navigators for Reasoning", Qianli Ma, Haotian Zhou, Tingkai Liu, Jianbo Yuan,
Pengfei Liu, Yang You, Hongxia Yang (2023).
```
new: author = {Ma, Qianli and Zhou, Haotian and Liu, Tingkai and Yuan, Jianbo and Liu, Pengfei and You, Yang and Yang, Hongxia},
     title  = {Let's Reward Step by Step: Step-Level Reward Model as the Navigators for Reasoning},
     year = {2023}, doi = {10.48550/arXiv.2310.10080}, note = {arXiv:2310.10080}
```

## 4. `lu2025lcpo` (finding 530)
arXiv:2503.04697 = "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning",
Pranjal Aggarwal, Sean Welleck.
```
new: author = {Aggarwal, Pranjal and Welleck, Sean},
     title  = {{L1}: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning}
```

## 5. `pu2025ttrl` (finding 531)
arXiv:2504.16084 authors: Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, ... 
```
new: author = {Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and others}
```

## 6. `singh2025artist` (finding 532)
arXiv:2505.01441 = "Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning",
Joykirat Singh, Raghav Magazine, Yash Pandya, Akshay Nambi.
```
new: author = {Singh, Joykirat and Magazine, Raghav and Pandya, Yash and Nambi, Akshay},
     title  = {Agentic Reasoning and Tool Integration for {LLMs} via Reinforcement Learning}
```

## 7. `subramanian2025slm` (finding 533)
arXiv:2410.20011 authors: Chien Van Nguyen, Xuan Shen, Ryan Aponte, Yu Xia, Samyadeep Basu, ... ; posted 2024-10-25.
```
new: author = {Nguyen, Chien Van and Shen, Xuan and Aponte, Ryan and Xia, Yu and Basu, Samyadeep and others}, year = {2024}
```
(P9 in-text "Subramanian et al." changed to "Nguyen et al." in related_work_v2.tex.)

## 8. `yuan2025versaprm` (finding 534)
arXiv:2502.06737 = "VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data", Thomas Zeng,
Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, ...
```
new: author = {Zeng, Thomas and Zhang, Shuibai and Wu, Shutong and Classen, Christian and Chae, Daewon and others},
     title  = {{VersaPRM}: Multi-Domain Process Reward Model via Synthetic Reasoning Data}
```

## 9. `zhang2025verifybench` (finding 535)
arXiv:2505.15801 = "VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models",
Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, ... (venue ICLR 2026 per reviewer; not independently
verified here — if unverifiable, use journal = {arXiv preprint}).
```
new: author = {Yan, Yuchen and Jiang, Jin and Ren, Zhenbang and Li, Yijun and Cai, Xudong and others},
     title  = {{VerifyBench}: Benchmarking Reference-based Reward Systems for Large Language Models},
     doi = {10.48550/arXiv.2505.15801}, note = {arXiv:2505.15801}
```

## 10. Internal notes leaked into rendered bibliography (findings 536, 537, 563)
- `thinkingmachines2024tinker`: note -> `{Technical blog post; not a peer-reviewed venue}` (drop "this worktree ... (P7 iter-134 row 152)").
- `vonwerra2022trl`: note -> `{Open-source library; not a peer-reviewed venue}` (drop "cited only to identify the codebase used by P7 iter-119 row 138 and P5 iter-125 row 149").
- `krakovna2020specification`: note -> keep only the URL (drop "not a peer-reviewed venue --- retained for traceability of the specification-gaming term").
- `trl2020` duplicates `vonwerra2022trl` (same URL/authors). P9 no longer cites `trl2020`; suggest retiring it portfolio-wide.

## 11. `xia2024reasoneval` (finding 562)
arXiv:2404.05692 title = "Evaluating Mathematical Reasoning Beyond Accuracy" (Xia, Li, Liu, Wu, Liu).
```
new: title = {Evaluating Mathematical Reasoning Beyond Accuracy}, note = {arXiv:2404.05692; ReasonEval}
```

## 12. `song2024prmbench` (finding 571)
arXiv:2501.03124 posted 2025-01-06; title "PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models".
```
new: year = {2025}, title = {{PRMBench}: A Fine-grained and Challenging Benchmark for Process-Level Reward Models}
```

## 13. `wang2025ragen` (finding 572)
arXiv:2504.20073 title = "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning".

## 14. `yamerenko2025generating` (finding 573)
No change requested. Crossref metadata for DOI 10.1109/LCSYS.2025.3573943
(`tools/review_ledger/P9_dnb_crossref_check.py`) lists the first author as "Grigory Yamerenko", matching the bib;
finding marked refuted.

## 15. `nie2026gradstarvation` (finding 570)
arXiv:2605.07689 lists: Wenhua Nie, Jianan Wu, Junlin Liu, Ziwei Li, Zheng Lin, Zhang Zijian, Yilong Fan, Haoran Zheng, Jyh-Shing Roger Jang.
```
new: author = {Nie, Wenhua and Wu, Jianan and Liu, Junlin and Li, Ziwei and Lin, Zheng and Zijian, Zhang and Fan, Yilong and Zheng, Haoran and Jang, Jyh-Shing Roger}
```

## 16. `zheng2026dara` (finding 565)
arXiv:2610.00574 eighth author is "Jingcheng Liang" (bib has "Liang, Jing").

## 17. `li2026oxrl` (finding 569)
Title already capitalised in bib ("? A Controlled ..."); plainnat lower-cases it. Protect with braces:
`title = {Do Post-Training Algorithms Actually Differ? {A} Controlled Study ...}`.

## 18. FYI for other papers sharing P9's former inputs (no edit made to these files)
P9 now inputs P9-only forks (`neurips_2026_variants/sections/*_dnb.tex`, `ethics_statement_dnb.tex`).
The originals still carry content found unsourced during the P9 audit and are still input by P8
(`paper_P8_workshop.tex`) and, for some, P2 / `paper.tex`:
- `sections/stratified_diagnostics.tex`: both tables unsourced (inputs `experiments/analysis/v5/*.csv` never existed; N=15 r=-0.769 reproduces only with `zero_reward_pct`, not ZVF).
- `../sections/group_size_reconcile.tex`: `tab:groupsize-tokennorm` is the hand-entered `FALLBACK_ROWS`; `GU_SCALE=55000` was tuned to reproduce it.
- `sections/run_manifest.tex`: Qwen2.5-0.5B Ckpt checkmark and Wave-6 held-out checkmark unsupported; Wave-6 is a temperature/rank/batch sweep, not group size; classic-RL "math" is a two-integer addition gym env with an MLP policy (accuracy 0.2-1.4%).
- `ethics_statement_zvf.tex`: compute (`tab:compute_spend`) and carbon (`tab:carbon2`) tables have no billing/GPU-hour source artifact; the "0.4% epsilon sweep" in `../sections/zvf_pipeline_spec.tex` has no artifact.
- `appendix_full_results_dnb.tex`-style rows in other papers: frontier/group-size values 56.2/36.4 etc. are 6-sample log recoveries of an aborted batch, and "Qwen3-8B (G=8) 81.2/34.4" has no run (see `tools/review_ledger/P9_dnb_table6.py`).
