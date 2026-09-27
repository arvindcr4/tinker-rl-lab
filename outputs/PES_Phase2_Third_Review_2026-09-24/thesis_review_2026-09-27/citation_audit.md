# Citation Audit: Thesis_Report_ArvindCR.pdf (2026-09-27)

Scope: `references.bib` (31 entries), `thesis_master.tex` (the compiled source), `ch*.md`, and the PDF as built on 27 Sep 06:25 (byte-identical to `build/thesis_master.pdf`).
Method: I parsed cite keys from `thesis_master.tex` and cross-checked them against the bib, the `.blg`/`.log` and the PDF text. I checked metadata against the arXiv API (all 25 arXiv-backed entries) and DOIs by inspection. For claims, I downloaded the full-text PDFs from arXiv and grepped them for the supporting passage. Page numbers for cited papers are PDF pages of the arXiv version. "Thesis p." is the physical PDF page.
Tools: the bundled `validate_citations.py` script is not installed, so these checks were done by hand. No alphaXiv MCP tools are available (ToolSearch found none). OpenAlex rate-limited (HTTP 429), so the uncited-work existence checks used arXiv search.

## 1. Mechanical checks

| Check | Result |
|---|---|
| Cite keys used in tex | 31 distinct keys, each cited exactly once |
| Keys missing from bib | **0** |
| Unused bib entries | **0** |
| Duplicate keys / duplicate works | **0 / 0** |
| BibTeX/LaTeX warnings | 0 "undefined" warnings in `thesis_master.log`; `.blg` is clean |
| `[?]` in PDF | 1 hit, thesis p.64. It is **not a citation**: it is a literal label `unstated [?]` in `figures/fig_attribution.tex:92`. An examiner will read it as a broken cite. Replace it with "(unstated)" or "?". |

## 2. Bibliographic metadata (all 30 external entries are real papers; none fabricated)

`group6capstone2026survey` is an internal PES tech report and cannot be verified externally. It is acceptable as-is.

| Key | Issue | Severity |
|---|---|---|
| `shao2024deepseekmath` | Author list truncated with no "and others": it gives 9 of 11 authors. **Xiao Bi and Haowei Zhang are missing** (arXiv order: Shao, Wang, Zhu, Xu, Song, **Bi**, **H. Zhang**, M. Zhang, Li, Wu, Guo). | Medium (wrong metadata) |
| `guo2025deepseekr1` | The entry cites the arXiv version only. The thesis quotes numbers that exist only in the Nature/v2 version: AIME 77.9%, and the training config of lr 3e-6, KL 0.001, 10,400 steps. Add the Nature 645:633–638 (2025) venue. | Medium |
| `gao2023scaling` | `@article` with `journal={ICML}`. It should be `@inproceedings` in ICML 2023 (PMLR 202). The arXiv version is 2022 (2210.10760). | Low |
| `christiano2017deeprlhf`, `stiennon2020summarize`, `ouyang2022instructgpt`, `rafailov2023dpo`, `dettmers2023qlora`, `hu2021lora` | Conference papers typed as `@article` with a venue string in `journal`, so IEEEtran renders them as journals. | Low (style) |
| `hu2021lora` | The key says 2021 and `year` says 2022 (ICLR). The key's year does not match the entry. | Cosmetic |
| `agarwal2021deep` | `booktitle` is NeurIPS, but the `doi` is the arXiv DOI. | Cosmetic |
| `lightman2023lets`, `liu2025understanding`, `ahmadian2024backtobasics`, `jordan2024benchmarking`, `rafailov2024rlhfscaling` | Cited as arXiv preprints, but published versions exist: ICLR 2024 / COLM 2025 / ACL 2024 / ICML 2024 (position track) / NeurIPS 2024. | Low |
| All others | Title, authors, year and IDs match arXiv/DOI. | OK |

### Named works with no bib entry (citation completeness)

Chapter 2 names about 30 works in prose with no `\cite`. Two of them are load-bearing:

- **Wu et al., "It Takes Two: Your GRPO Is Secretly DPO" (arXiv 2510.00977).** Chapter 2 cites it as prior theory (p.35). Chapter 6 uses its **97.6% retention ratio as a quantitative reference bar** (pp.110–111, in the figure and the text), still without a citation.
- **GSPO (arXiv 2507.18071).** It is used in Chapter 7 variant-delta records and the 12-cell head-to-head.

Verified to exist on arXiv and could be added: Zheng et al. "Secrets of RLHF Part I: PPO" 2307.04964; VAPO 2504.05118; CPPO 2503.22342; NGRPO 2509.18851; Scaf-GRPO; MO-GRPO 2509.22047; SSPO 2511.04256 (it is **Subsentence**-level, not "sentence granularity" as ch02 says); RL-ZVP "No Prompt Left Behind" 2509.21880; GVPO 2504.19599; MC-GRPO 2601.22582; Goldilocks RL 2602.14868. Also named but uncited: Kaplan; Hoffmann; Snell; HELM; LM Evaluation Harness; GSM8K (Cobbe); MATH (Hendrycks); Math-Shepherd; and others.

Not found by title search: GRESO, G²RPO-A, λ-GRPO, AERO. **"Dr. SAC"** is also not found: the only close match is DR-SAC (distributionally robust soft actor-critic, 2506.12622). That paper is unrelated to the claim that "REINFORCE with a value baseline matches GRPO on RLHF tasks". Treat the Dr. SAC name as probably wrong, and remove it or source it.

## 3. Claim spot-check (load-bearing literature claims)

| # | Thesis claim (thesis p.) | Cited paper | Verdict | Supporting passage |
|---|---|---|---|---|
| 1 | Christiano et al. "established the three-stage pattern of SFT, reward-model training, and RL" (p.31) | christiano2017deeprlhf | **Not supported** | There is no SFT stage: the paper covers Atari/MuJoCo with an "untrained (randomly initialized) policy" (App. A, p.14). The 3-stage SFT→RM→RL pipeline comes from Ziegler/Stiennon/Ouyang. |
| 2 | Ziegler et al. collected ~60,000 comparisons and "introduc[ed] the per-token KL penalty" (p.31) | ziegler2019finetuning | **Partial** | The 60k figure is supported (§2, p.2: "60,000 human samples" for summarization). The KL penalty is explicitly "Following Jaques et al. (2017; 2019)" (§2, p.2), so "introducing" is wrong. |
| 3 | Stiennon et al.: 6.7B, 64,832 comparisons, CNN/DM transfer, first evidence of RM over-optimisation (p.32) | stiennon2020summarize | Supported ("first" is an overclaim) | §3, p.3 ("64,832 summary comparisons"); §4.3/Fig. 5, p.7 ("What happens as we optimize the reward model?") |
| 4 | InstructGPT: 40 contractors; 13k/33k/31k prompts; 6B RM because 175B was unstable; K∈{4..9}; 85%; hallucination 41→21%; toxicity −25%; agreement ~73% (p.32) | ouyang2022instructgpt | Supported | p.3 (85±3%, 21% vs 41%, 25% fewer toxic outputs); §3.3–3.5, p.8 (datasets, "unstable and thus was less suitable", K=4..9); agreement 72.6±1.5% (§3.4) |
| 5 | GRPO/DeepSeekMath: G=64, β=0.04, **ε=0.2**, "cuts memory by **~25%**", MATH 46.8→51.7, SC@64 60.9, GSM8K 82.9→88.2, CMATH 84.6→88.8, 10% replay (p.32) | shao2024deepseekmath | **Partial** | G=64, β=0.04 and the 10% replay are supported (§4.2, p.15). The results are supported (p.2 and Table 5). The paper does not state ε=0.2, and it gives **no 25% memory figure**: it says only "optimizing the memory usage of PPO" (abstract). |
| 6 | DeepSeek-R1: "16 sampled outputs per question, **a group size of 512**, lr 3e-6, KL 0.001, ~10,400 steps"; AIME 15.6→71.0 (77.9 in Nature); 7B/16B bases failed (p.34) | guo2025deepseekr1 | **Wrong detail** | 512 is the **training batch size** (32 questions × 16 outputs); the group size is 16 (§2.2, p.3, v2). The other numbers are supported: p.4 (15.6→77.9) and App. G.1, p.62 (7B dense / 16B MoE "consistently failed"). The 71.0 figure is from v1. |
| 7 | Wu et al. "prove GRPO is secretly DPO"; 2-GRPO retains **98.1%** of 16-GRPO at 12.5% of rollouts (p.35) | *(no bib entry)* | **Unsupported / uncited; internal inconsistency** | The current v3 abstract (p.1) says **97.6%**. The 98.1% figure is from v2. Chapter 6 (pp.110–111) uses 97.6%. The paper "propose[s] a view" that GRPO is structurally related to DPO, so "prove" overstates it. |
| 8 | Dr. GRPO removes 1/L and std normalisation; "motivated explicitly as a fix for **reward hacking** through length" (p.36) | liu2025understanding | **Partial** | The two removed terms are supported (§3.1 "GRPO Leads to Biased Optimization", p.5). The paper frames this as an *optimization bias* that inflates length, especially of incorrect responses. The phrase "reward hacking" does not appear. |
| 9 | DAPO: four techniques; 50 AIME with a 32B model; ε_low=0.2, ε_high=0.28, KL removed (p.36; ch07) | yu2025dapo | Supported | Abstract/§1, p.1–2 (4 techniques, 50 pts on Qwen2.5-32B); §4, p.8 ("εlow to 0.2 and εhigh to 0.28") |
| 10 | Tülu 3 formalised RLVR (p.38) | lambert2024tulu3 | Supported | Abstract, p.1 ("a novel method we call RLVR"); §6 |
| 11 | Hilton et al. derived compute-efficiency frontiers "for RL in games and **robotics**" (p.41) | hilton2023rlscaling | **Partial** | The environments are Procgen, 1v1 Dota 2 and a toy MNIST task (§1, p.1; §3). There is **no robotics**. |
| 12 | Gao et al. characterised the RM over-optimisation curve (p.41) | gao2023scaling | Supported | Abstract, p.1 (gold-standard RM vs proxy RM) |
| 13 | Rafailov et al. "provided scaling laws for **DPO and PPO** that show a strong dependence on **reward-model quality**" (p.41) | rafailov2024rlhfscaling | **Not supported** | The paper studies over-optimisation in *direct alignment algorithms* (DPO/IPO/SLiC), which "do not use a separate proxy reward model" (abstract, p.1; §3.2 scaling fits). It gives no PPO scaling law and no RM-quality dependence. |
| 14 | Nimmaturi et al.: three phases (slow start, rapid improvement, plateau); benefit exhausted by ~80% of one epoch (p.41) | nimmaturi2025predictive | Supported | Abstract, p.1; §1, p.1 ("training beyond roughly 80% of a single epoch yields negligible reward gains") |
| 15 | Tan et al. treat group size as a budget knob (pp.41, 45) | tan2025scalingrl | Supported (appendix only) | App. B.2, p.14: "The optimal rollout size G is not fixed but shifts with the training budget." The main text and abstract do not make this point. |
| 16 | LoRA gives a "128× reduction in trainable parameters at d=4096, r=16" (p.41) | hu2021lora | **Not in source** | This is the thesis's own arithmetic (4096/(2·16)). The paper states "10,000 times" for GPT-3 175B (abstract, p.1). Label it as derived. |
| 17 | QLoRA "quantis[es] the frozen base to NF4 with double quantisation **at 0.37 bits per parameter**, ≈3 GB for 65B"; 65B fits on one 48 GB GPU (p.41) | dettmers2023qlora | **Misstated** | Double quantization **saves** ≈0.37 bits/param (0.5→0.127) (§1, p.2; §3, p.5). It is not the storage rate. The 48 GB figure is supported (abstract). |
| 18 | Mukherjee et al.: RL updates only 5–30% of parameters across 7 algorithms / 10 models (p.42). Table 2.1 (p.45) attributes this to "**Liu et al.** on sparse RL subnetworks". | mukherjee2025rlsubnetworks | Supported (text); **wrong author in Table 2.1** | Abstract, p.1 |
| 19 | Lightman: step-level PRMs outperform ORMs "on **GSM8K** and MATH" (p.42) | lightman2023lets | **Partial** | The evaluation is on MATH (plus OOD STEM, §5). GSM8K is only mentioned in comparison (p.12). MATH-500 originating here is supported. |
| 20 | Jordan et al. argued "most published RL comparisons lack statistical power" (p.40) | jordan2024benchmarking | Partial (paraphrase stretch) | The abstract (p.1) argues that rigorous benchmarking is computationally prohibitive and calls for an additional experimentation paradigm. |

Not full-text-checked (the claims are generic or match the abstract, and none drives a design choice): Henderson, Colas, Agarwal (rliable IQM and performance profiles), Dodge, Pineau, Model Cards / Datasheets / Data Statements / Data Cards, DPO, RLOO. All are consistent with their abstracts. Zheng et al. "Secrets of RLHF" (the "policy constraints" claim) is consistent with its abstract but has no citation.

## 4. Counts

- Unresolved cite keys: **0**. There is one false-positive `[?]`, a figure label at p.64.
- Fabricated entries: **0**. Wrong or incomplete metadata: **2 substantive** (DeepSeekMath authors; DeepSeek-R1 version/venue) and about 12 minor type/venue issues.
- Claims checked: 20. **Not supported / wrong: 4** (#1 Christiano SFT; #6 R1 "group size 512"; #13 Rafailov DPO/PPO/RM-quality; #7 Wu 98.1%, which is uncited). **Partial or misstated: 9** (#2, #5, #8, #11, #16, #17, #18 author, #19, #20). **Supported: 7**.
- Uncited named works: about 30. Two are load-bearing (Wu et al. 2510.00977, GSPO 2507.18071).

## 5. Priority fixes

1. Add a bib entry for Wu et al. (2510.00977) and use one figure in both chapters. The current version says 97.6%; Chapter 2 says 98.1%. Change "prove" to "argue/show".
2. Rewrite the Rafailov 2024 sentence: over-optimisation scaling laws for *direct alignment algorithms* without a reward model.
3. Christiano 2017: drop "SFT" and attribute the three-stage pipeline to Ziegler/Stiennon/Ouyang. Ziegler: change "introducing" the KL penalty to "adopting" it (following Jaques et al.).
4. DeepSeek-R1: "group size of 512" should read "batch size 512 (32 prompts × 16 samples)". Add the Nature venue. Drop "ε = 0.2" and "~25% memory" from the DeepSeekMath sentence, or source them elsewhere.
5. Table 2.1: change "Liu et al." to "Mukherjee et al. [30]". Also fix: QLoRA (DQ saves 0.37 bits/param); Hilton (no robotics); Lightman (MATH, not GSM8K); Dr. GRPO ("optimization bias", not "reward hacking"); the missing DeepSeekMath authors; the "Dr. SAC" name; and the figure `[?]` label.
