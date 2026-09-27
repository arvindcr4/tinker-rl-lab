# Citation fixes (2026-09-27)

Covers the issues in `citation_audit.md`. Files touched: `thesis/references.bib`, `thesis/ch02_literature.md`, `thesis/ch06_results_core.md` (cite keys only) and `thesis/ch07_results_infra.md` (cite keys only). `thesis_master.tex` was not edited, the PDF was not rebuilt, and nothing was committed.

The metadata sources were the arXiv API (abstracts and author lists, `export.arxiv.org`), the Semantic Scholar batch API (venues and DBLP keys), Crossref (the DOIs for Nature and ACL), and the COLM-2025 camera-ready PDF on OpenReview (via Exa search). Midway through the session arXiv began rate-limiting this IP, so I generated the new entries' authors from the arXiv author lists I had already retrieved in the same session. Semantic Scholar supplied authors only for 13 small-author-count entries, and those match arXiv.

Validation: 88 bib entries, 0 duplicate keys, 0 cite keys missing from the bib, 0 unused entries, and balanced braces in every entry. The count covers the chapters listed in `build_thesis.py`. `pandoc --natbib` renders the multi-key cites as `\citep{a, b}`. BibTeX is not installed, so I have not run a compile check.

## A. Bibliographic metadata (references.bib)

| # | Issue | Old | New | Source verified |
|---|---|---|---|---|
| A1 | DeepSeekMath authors truncated | 9 authors, with Bi and H. Zhang missing | 11 authors in arXiv order: Shao, Wang, Zhu, Xu, Song, **Bi, H. Zhang**, M. Zhang, Li, Wu, Guo. Now `@misc` + DOI | arXiv 2402.03300v3 author list |
| A2 | DeepSeek-R1 cited as the arXiv version only, but the thesis quotes Nature numbers | `@article` arXiv preprint, v1 author order | `@article` *Nature* 645(8081):633–638, 2025, doi 10.1038/s41586-025-09422-z. Nature title and author order (Guo, Yang, H. Zhang, Song, Wang, Zhu, Xu, R. Zhang, Ma, Bi, others). The arXiv preprint is kept in `note` | Crossref 10.1038/s41586-025-09422-z; arXiv journal_ref |
| A3 | Gao 2023 typed as a journal | `@article`, journal=ICML | `@inproceedings` ICML 2023, PMLR 202, pp. 10835–10866. The note gives arXiv 2210.10760 (2022) | S2 (DBLP conf/icml/GaoSH23, pages) |
| A4 | Christiano, Stiennon, Ouyang, DPO, QLoRA, LoRA typed as journals | `@article` with a venue in `journal` | `@inproceedings`, booktitle NeurIPS (vols 30/33/35/36/36) and ICLR 2022 for LoRA. QLoRA title corrected to arXiv's "…Quantized LLMs" | S2 venue fields; arXiv titles |
| A5 | `hu2021lora` key/year mismatch | year 2022 with a 2021 key, no explanation | year 2022 (ICLR) kept and `note = arXiv:2106.09685 (2021)` added. The key is unchanged so no cites break | DBLP conf/iclr/HuSWALWWC22; arXiv 2021 |
| A6 | Agarwal: NeurIPS booktitle but arXiv DOI | `doi = 10.48550/arXiv.2108.13264` | DOI removed. The arXiv ID stays in `note` | S2 (conf/nips/AgarwalSCCB21, pp. 29304–29320) |
| A7 | Lightman cited as a preprint | arXiv preprint; author "Yuri Burda" | ICLR 2024; author "Yura Burda" (as arXiv lists him) | S2 DBLP conf/iclr/LightmanKBEBLLS24 |
| A8 | Liu (Dr. GRPO) cited as a preprint | arXiv | COLM 2025 | OpenReview PDF footer "Published as a conference paper at COLM 2025" |
| A9 | Ahmadian (RLOO) cited as a preprint | arXiv | ACL 2024 (Vol. 1 Long), pp. 12248–12267, doi 10.18653/v1/2024.acl-long.662 | Crossref |
| A10 | Jordan cited as a preprint | arXiv | ICML 2024 (position track) | arXiv comment; S2 venue |
| A11 | Rafailov 2024 cited as a preprint | arXiv | NeurIPS 2024 (vol 37) | arXiv comment; S2 venue |
| A12 | Mukherjee cited as a preprint (not flagged by the audit) | arXiv | NeurIPS 2025 (vol 38) | S2 DBLP conf/nips/MukherjeeYHP25 |

## B. Claims fixed (ch02_literature.md unless noted)

| # | Audit ref | Old | New | Source verified |
|---|---|---|---|---|
| B1 | #1 Christiano | "established the three-stage pattern of supervised fine-tuning, reward-model training …, and reinforcement learning" | Christiano et al. established RM training on pairwise preferences followed by RL, "on Atari and MuJoCo tasks starting from a randomly initialised policy". The SFT stage is attributed to later work [@stiennon2020summarize; @ouyang2022instructgpt] | Audit full-text check (App. A, "untrained (randomly initialized) policy") |
| B2 | #2 Ziegler | "introducing the per-token KL penalty" | "following Jaques et al. [@jaques2017sequencetutor; @jaques2019wayoffpolicy], adopting the per-token KL penalty" | Ziegler §2 per the audit; both Jaques papers verified on S2 (DBLP conf/icml/JaquesGBHTE17; arXiv 1907.00456) |
| B3 | #3 Stiennon "first" | "produced the first empirical evidence of reward over-optimisation" | "produced early empirical evidence …" | Audit (overclaim) |
| B4 | #5 DeepSeekMath ε / memory | "G = 64, β = 0.04, and ε = 0.2 … cuts memory requirements by approximately 25%" | "G = 64 and β = 0.04 … the paper motivates this as reducing PPO's memory usage but does not quantify the saving" | Audit (not in paper); DeepSeekMath abstract "optimizing the memory usage of PPO" |
| B5 | #6 R1 group size | "16 sampled outputs per question, a group size of 512" | "16 sampled outputs per question (the group size), a training batch of 512 outputs (32 questions × 16 samples)" | Audit (§2.2 p.3, v2/Nature) |
| B6 | #6 R1 71.0 vs 77.9 | "rose from 15.6% to 71.0% … the revised *Nature* version reports 77.9%" | "rose from 15.6% to 71.0% over training in the original preprint — the *Nature* version cited here reports 77.9%" | Audit; bib now points to Nature (A2) |
| B7 | #7 Wu et al. | "Wu et al. prove that GRPO is 'secretly DPO' … retains 98.1%" (no cite) | "Wu et al. [@wu2025ittakestwo] argue that … its efficacy stems from an implicit contrastive objective, which makes it structurally related to DPO … retains 97.6% … at 12.5% of the rollouts (arXiv v3; the earlier v2 reported 98.1%)". This agrees with the Ch. 6 use of 97.6% | arXiv 2510.00977 v3 abstract (97.6%, "we propose a different view … structurally related"); v2 abstract (98.1%) |
| B8 | #7 Wu in Ch. 6 | ch06 "the Wu et al. 97.6% reference bar" (no cite) | same text + [@wu2025ittakestwo] | as B7. Fig 6.3d text left to the figures agent |
| B9 | #8 Dr. GRPO | "motivated explicitly as a fix for reward hacking through length" | "motivated as a fix for an optimisation bias in GRPO that artificially inflates response length, especially of incorrect responses" | Liu et al. abstract (arXiv/OpenReview) |
| B10 | Dr. SAC | "Production-side variants include VAPO and Dr. SAC, the latter arguing that … REINFORCE with a value baseline matches GRPO" | "Production-side variants include VAPO [@yue2025vapo], a value-based PPO framework for long chain-of-thought reasoning". **Dr. SAC claim removed**: no such paper exists (the only match is DR-SAC, 2506.12622, distributionally robust SAC, which is unrelated) | arXiv search; VAPO 2504.05118 abstract |
| B11 | #11 Hilton | "for RL in games and robotics" | "for single-agent RL across Procgen games, Dota 2 and a toy MNIST-based task" | Hilton abstract (MNIST); audit full-text (§1, §3: Procgen, Dota 2) |
| B12 | #13 Rafailov 2024 | "provided scaling laws for DPO and PPO that show a strong dependence on reward-model quality" | "extended over-optimisation scaling laws to direct alignment algorithms (DPO, IPO and SLiC), which have no separate proxy reward model" | arXiv 2406.02900 abstract; audit §3.2 |
| B13 | #15 Tan | "treat group size as a budget knob" | + "in an appendix analysis showing that the optimal rollout size shifts with the training budget" | Audit (App. B.2 p.14) |
| B14 | #16 LoRA 128× | "achieving a 128× reduction … which is on the order of 0.1% of the model's parameters" | "a d × d projection at d = 4096 with rank r = 16 trains 2dr rather than d² parameters, a 128× reduction (this chapter's arithmetic; the paper itself reports up to 10,000× fewer trainable parameters for GPT-3 175B)". The unsourced "0.1%" was dropped: 1/128 ≈ 0.8% per matrix | LoRA abstract ("10,000 times") |
| B15 | #17 QLoRA | "NF4 with double quantisation at 0.37 bits per parameter" | "NF4 and adding double quantisation, which saves about 0.37 bits per parameter — approximately 3 GB for a 65B model" | Audit (QLoRA §1 p.2, §3 p.5) |
| B16 | #18 Table 2.1 | "Liu et al. on sparse RL subnetworks" | "Mukherjee et al. [@mukherjee2025rlsubnetworks] on sparse RL subnetworks" | arXiv 2505.11711 |
| B17 | #19 Lightman | "outperform outcome reward models on GSM8K and MATH" | "… on MATH" | Audit (§5; GSM8K only mentioned) |
| B18 | #20 Jordan | "argued that most published RL comparisons lack statistical power" | "argued that rigorous benchmarking in RL is computationally prohibitive at the scale it would require, and called for an additional experimentation paradigm to complement it" | Audit (abstract p.1) |
| B19 | SSPO granularity (audit §2) | "operates at sentence granularity … improving by 3.56 points on mathematics benchmarks" (§2.3); "SSPO's sentence-granularity objective" (§2.6) | "subsentence granularity … raising GRPO's five-benchmark mathematics average from 43.01 to 46.72 on Qwen2.5-Math-1.5B"; "subsentence-granularity objective". The 3.56 was the v1 figure (46.57 − 43.01) | arXiv 2511.04256 v1 and v2 abstracts |
| B20 | "AERO / RL-ZVP" treated as one method | "AERO / RL-ZVP … extracts signal from zero-variance groups through entropy-guided advantage shaping"; §2.6 "in AERO / RL-ZVP"; Table 2.1 "AERO/RL-ZVP" | Now two distinct papers: RL-ZVP [@le2025rlzvp] = entropy-guided shaping of zero-variance prompts; AERO [@zhang2026aero] = adaptive rollout allocation and pruning to avoid zero-advantage dead zones. §2.6 says "in RL-ZVP". The table lists "AERO, RL-ZVP" | arXiv 2509.21880, 2602.14338 abstracts |
| B21 | G²RPO-A mislabelled (found in this pass) | Table 2.1 P7: "heuristic adaptive-G (G²RPO-A)"; §2.3 "G²RPO" | "adaptive training-signal control (G²RPO-A adaptive guidance [@guo2025g2rpoa])"; §2.3 "G²RPO-A [@guo2025g2rpoa]". G²RPO-A adapts guidance strength, not group size | arXiv 2508.13023 abstract |
| B22 | LIMR / TTRL descriptions (found in this pass) | "GRPO-Lead, LCPO, and LIMR tune reasoning depth through length-aware rewards … TTRL performs test-time RL against self-generated verifiers" | "GRPO-Lead and LCPO tune reasoning length through length-aware rewards, LIMR selects a small high-impact subset of RL training data … TTRL performs test-time RL on unlabelled data using majority-vote rewards" | arXiv 2502.11886, 2504.16084 abstracts |
| B23 | ReasonEval description (found in this pass) | "ReasonEval introduces reasoning-specific reward models" | "ReasonEval scores reasoning steps for validity and redundancy with trained LLM evaluators" | arXiv 2404.05692 abstract |
| B24 | "Miller et al." (single author) | "(Miller et al.; Zhang et al.)" | "(Miller [@miller2024errorbars]; Zhang et al.)". Also cited at ch06 "Miller-style [@miller2024errorbars] error-bars audit" | arXiv 2411.00640 (sole author Evan Miller) |

## C. Named works given a bib entry and a cite (57 new entries)

Every entry below was checked against arXiv: the ID resolves, and the title and first author match the named work. I also checked the ch02 claim against the abstract; where the claim did not match, it was reworded (section B).

- **Load-bearing:** Wu et al. `wu2025ittakestwo` (2510.00977; ch02, ch06). GSPO `zheng2025gspo` (2507.18071; ch02, and ch07 twice: the 12-cell head-to-head and the P6 variant deltas).
- **RLHF/PPO:** `schulman2017ppo` (1707.06347), `schulman2016gae` (1506.02438, ICLR 2016), `zheng2023secretsrlhf` (2307.04964), `jaques2017sequencetutor` (1611.02796, ICML 2017), `jaques2019wayoffpolicy` (1907.00456).
- **DPO family:** `azar2023ipo` (2310.12036), `ethayarajh2024kto` (2402.01306), `hong2024orpo` (2403.07691), `meng2024simpo` (2405.14734), `zhao2023slichf` (2305.10425).
- **GRPO variants:** `yue2025vapo` (2504.05118), `zheng2025greso` (2506.02177), `lin2025cppo` (2503.22342), `nan2025ngrpo` (2509.18851), `zhang2025scafgrpo` (2510.19807, ICLR 2026), `ichihara2025mogrpo` (2509.22047), `yang2025sspo` (2511.04256), `le2025rlzvp` (2509.21880, ICLR 2026), `zhang2026aero` (2602.14338), `zhang2025gvpo` (2504.19599, NeurIPS 2025), `wang2025lambdagrpo` (2510.06870), `kim2026mcgrpo` (2601.22582), `guo2025g2rpoa` (2508.13023), `zhang2026conspo` (2605.12969), `he2026avspo` (2605.21125, ICML 2026), `mahrooghi2026goldilocks` (2602.14868).
- **Scaling/eval:** `kaplan2020scaling` (2001.08361), `hoffmann2022chinchilla` (2203.15556), `snell2024testtime` (2408.03314), `liang2022helm` (2211.09110, TMLR 2023), `biderman2024lmeval` (2405.14782, the LM Evaluation Harness paper), `miller2024errorbars` (2411.00640), `hochlehnert2025sober` (2504.07086, COLM 2025).
- **Benchmarks:** `cobbe2021gsm8k` (2110.14168), `vendrow2025platinum` (2502.03461, the GSM8K-Platinum paper), `hendrycks2021math` (2103.03874, NeurIPS 2021 D&B).
- **Process supervision and search:** `wang2024mathshepherd` (2312.08935), `song2025prmbench` (2501.03124), `luo2024omegaprm` (2406.06592), `xia2025reasoneval` (2404.05692), `yuan2024implicitprm` (2412.01981), `zeng2025versaprm` (2502.06737), `yao2023tot` (2305.10601, NeurIPS 2023), `wang2023selfconsistency` (2203.11171, ICLR 2023). PRM800K now cites `lightman2023lets`.
- **Reasoning RL:** `zhang2025grpolead` (2504.09696), `aggarwal2025l1` (2503.04697; this is the LCPO paper, COLM 2025), `li2025limr` (2502.11886), `jin2025searchr1` (2503.09516), `zuo2025ttrl` (2504.16084).
- **Systems:** `hu2024openrlhf` (2405.11143), `sheng2025hybridflow` (2409.19256, veRL), `yao2023deepspeedchat` (2308.01320), `kwon2023vllm` (2309.06180, SOSP 2023), `zheng2024sglang` (2312.07104), `dao2022flashattention` (2205.14135). LoRA is also cited as a kernel.
- ch07: the 12-cell head-to-head now also cites the GRPO, DAPO and Dr. GRPO source papers, which were already in the bib.

Typing convention for new entries: an accepted venue that is stated on arXiv or found in S2/DBLP gets `@inproceedings`/`@article`. Everything else is `@misc` with `howpublished = arXiv preprint` and an arXiv DOI.

## D. Unresolved (left as is or hedged, no reference invented)

1. **VerifyBench.** Two different arXiv papers carry this name (2505.15801, Yan et al.; 2507.09884, Li et al.), and the related-work source does not say which one it means. The name is left uncited.
2. **"Zhang et al." (evaluation noise, §2.4).** No identifiable paper; left uncited.
3. **"Query recycling" (§2.3).** No identifiable paper; left uncited.
4. **Software with no paper:** TRL, NeMo-RL, trlX, Axolotl, Tinker, Papers with Code (including the "discontinued July 2025" claim). None has a bib entry. trlX has an EMNLP 2023 paper, but I did not verify it this session.
5. **GSM8K "fine-tuned GPT-3 verifier reached 35%" and MATH "original results reached 6.9%".** These figures come from the capstone survey. The dataset sizes match Cobbe and Hendrycks, and the cites sit on the dataset descriptions, but I did not check the two accuracy figures against the full texts.
6. **Other §2 numbers not checked against full text:** InstructGPT "TruthfulQA roughly doubled"; the R1 cons@64 86.7% and MATH-500 95.9% (possibly v1 figures); DPO "40–60% savings"; GVPO "re-weights by within-group variance" (the abstract describes variance-of-implicit-reward gradient weights, so this is loose but not contradicted); the Zheng et al. detail "reward hacking, length explosion, pattern collapse" (the abstract supports only "policy constraints … key factor").
7. **Venue for Tulu 3, Hilton, Nimmaturi, Tan, DAPO, DeepSeekMath.** S2 shows no peer-reviewed venue, so all remain arXiv.
8. **Out of scope for this agent:** the `[?]` label at `figures/fig_attribution.tex:92` and the Wu "97.6% claim holds at 0/4 budgets" text in `figures/fig_group_size.tex` belong to the figures agent.
