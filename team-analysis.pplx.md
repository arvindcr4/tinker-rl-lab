# Team Contribution Analysis & Experiment Allocation Strategy

## Executive Summary

This analysis is based on a complete reading of the Sem 3_Project WhatsApp group chat (2,187 messages, Nov 2025 – Apr 2026), cross-referenced with GitHub commits, HuggingFace model cards, W&B logs, and shared artifacts. The goal: map each member's authentic contributions, identify their strengths, and allocate the world-class experiment suite so each person has a flagship result that showcases their best work.

---

## Member-by-Member Deep Dive

### 1. Arvind C R (PES2PGE24DS140) — Project Lead & Chief Architect

**Role in Group Dynamics:**
- Created the intellectual direction from Day 1 — proposed the RL/agent/tool-use framing when others were still debating RAG vs. drones
- Compiled 200+ papers into 19 RL categories, shared the taxonomy with the group (Jan 8)
- Set up all infrastructure: GitHub repo (`pes-llm-research/tinker-rl-lab`), Overleaf, W&B integration, Tinker experiments
- Consistently pushed the team ("if you are being stingy for 50$ remember you paid 6 lakhs for this course" — Apr 8)
- Submitted every deliverable: literature survey (Jan 18), final report (Apr 4), conference paper compilation
- First to run Tinker experiments, demonstrated fine-tuning workflow to the team (Jan 12)

**Technical Contributions:**
- 32+ Tinker GRPO runs across GSM8K, tool-use, multi-seed experiments
- Discovered the "10x Structural Ceiling" phenomenon (reward improvement bounds)
- Tool-call LoRA fine-tuning on Qwen-0.5B (initial proof-of-concept)
- Built the entire `tinker-rl-lab` codebase and experiment pipeline
- Cross-task generalization experiments (GSM8K → tool_use)

**HF Model:** `arvindcr4/tool-call-lora-qwen0.5b`
**W&B:** Primary contributor to `tinker-rl-lab-world-class` project

**Best-Fit Experiments for Paper:**
- **Scaling law analysis** (8B → 32B → 235B) — his signature contribution
- **Cross-task transfer** (GSM8K → tool_use) — demonstrates generalization
- **Frontier model evaluation** (DeepSeek-V3.1, Qwen3-235B, Nemotron-120B)
- **World-class experiment orchestration** — the 20-experiment parallel suite itself

---

### 2. Sandhya Jeyaraj (PES2PGE24DS144) — Algorithm Specialist & RL Methods Lead

**Role in Group Dynamics:**
- Focused, pragmatic contributor — consistently delivered on time
- Read 9-10 papers by Jan 4, focused on DPO variants (Step-DPO, Full-Step-DPO), tree search for agents, reward modeling
- Provided critical feedback on presentation flow ("I really didn't like the way we presenting" — Dec 13)
- First to attempt fine-tuning with Tinker after Arvind's demo
- Reported accuracy issues with 1B models (Feb 28), pivoted to larger models

**Technical Contributions:**
- 3-phase tool-call scaling pipeline: 0% → 92% JSON validity
- Multi-turn tool chaining experiments
- Algorithm comparison work (Step-DPO, GRPO reward modeling)
- Shared experiment reports and PPTs consistently

**HF Model:** `Balasandhya/llm-multiturn-tool-call-grpo-QloRA-Qwen2.5-3B`
**Artifacts:** `Exp_report.pdf`, `exp3_multiturn.pptx`, `Capstone_paper_work.numbers`

**Best-Fit Experiments for Paper:**
- **Multi-turn tool chaining** — her flagship result (0% → 92%)
- **Algorithm comparison** (GRPO vs. DPO on tool-call tasks) — matches her literature focus
- **MoE vs. Dense comparison** — natural extension of her scaling interest

---

### 3. Madhu Kumara L (PES2PGE24DS176) — Code Generation & RAG Integration Lead

**Role in Group Dynamics:**
- Created the WhatsApp group (Nov 4), consistently organized meetings
- Strong on RAG knowledge — proposed Agentic RAG with human-in-the-loop feedback from the start
- Eager learner but faced technical challenges (Tinker payment issues, Overleaf compilation errors)
- Pivoted to Modal and Vast.ai when Tinker didn't work for him
- Delivered Qwen3-8B code reasoning model with 86% HumanEval accuracy (Apr 5)

**Technical Contributions:**
- SWE code generation via GRPO (HumanEval: 32% → 40% with 300 steps, claimed 86% with extended training)
- Fine-tuned Qwen3-8B for code reasoning format
- Read papers on multi-agent RL, model-based RL, RAG integration

**HF Model:** `Madhu2133/qwen3-8b-code-grpo-v10`
**Artifacts:** `Qwen3-8B Code Reasoning Model — Training Report.pdf`, `Fine-Tuning Qwen 3.8B_Madhu.pptx`

**Best-Fit Experiments for Paper:**
- **HumanEval GRPO evaluation** — directly extends his code-generation work
- **Code reasoning format analysis** — unique contribution area
- **RAG-augmented reward shaping** — bridges his RAG interest with RL

---

### 4. Mohammad Rafi (PES2PGE24DS157) — Logical Reasoning & Mathematical Optimization Lead

**Role in Group Dynamics:**
- Organized, process-oriented — proposed scheduling meetings, assigning tasks, voting for TL
- Sometimes delayed by work commitments ("I am in ofc... will be available on Thursday")
- Came through with a strong result in the final stretch: Qwen3-4B achieving 41.6% → 100% on logical reasoning
- Read papers on DeepSeekMath, CRITIC, Conservative Q-Learning, scaling laws for math reasoning
- Wrote his own research paper (`mohammad_capstone_research_paper.pdf`)

**Technical Contributions:**
- Multi-stage GRPO optimization pipeline on Qwen3-4B-Instruct
- Logical reasoning improvement: 41.6% → 100% (impressive for a 4B model)
- Chose 4B over 30B to demonstrate RL can match larger model capabilities
- Research methodology documentation (`06_research_methodology_09_02_26.tex`)

**HF Model:** `MohammadRafiML` (hub page)
**Artifacts:** `Initial_draft_on_RL_1.pdf`, `mohammad_capstone_research_paper.pdf`, `presentation_md.pptx`

**Best-Fit Experiments for Paper:**
- **Small-model amplification** (4B achieving 100% logical reasoning) — his headline result
- **Mathematical reasoning scaling** — extends his DeepSeekMath literature
- **Multi-stage optimization ablation** — showcase the pipeline stages

---

### 5. Dhruva N Murthy (PES2PGE23DS169) — MLOps Pipeline & Evaluation Framework Lead

**Role in Group Dynamics:**
- Senior batch (2023 admission vs. 2024 for others) — brings engineering maturity
- Set up Google Meet series, Zoom meetings, GitHub collaboration
- Read papers on agentic RL surveys, tool augmented language models
- Honest about progress ("I have only been able to complete till baseline results" — Mar 26)
- Built a complete 3-stage MLOps pipeline: data → training → evaluation
- Created 59 W&B runs for baseline comparisons and negative controls

**Technical Contributions:**
- 3-stage MLOps pipeline with SFT → GRPO → evaluation stages
- Baseline evaluation framework for Qwen3-8B tool-use
- 59 W&B runs tracking training metrics
- Attempted Qwen2.5-0.5B and 1.5B models (didn't improve beyond baseline — important negative result)
- Final SFT + GRPO runs on Qwen3-8B via Tinker

**HF Model:** `dhruvanmurthy/llm_efficiency`
**W&B:** `capstone-proj/qwen3-8b-tool-use`, `capstone-proj/qwen3-8b-multi-tool-use`
**Repos:** `dhruvanmurthy/capstone-trials`, `dhruvanmurthy/Qwen3-8B-FineTuning`

**Best-Fit Experiments for Paper:**
- **Negative control analysis** — models that DON'T improve are scientifically valuable
- **Baseline evaluation harness** — establishes the comparison framework
- **MLOps pipeline reproducibility** — demonstrates the end-to-end system
- **PPO vs. GRPO comparison** — extends his baseline work with a new algorithm

---

### 6. Arumugam K (PES2PGE24DS137) — Independent Validation & Domain-Specific Applications Lead

**Role in Group Dynamics:**
- Working at a defense site — often had restricted phone/laptop access
- Consistently communicated constraints clearly ("I'm currently out of station at a defence site")
- Independent implementer — built his own pipeline from scratch
- Explored DPO+LoRA for aerospace Q&A — unique domain application

**Technical Contributions:**
- Independent validation of tool-call pipeline (separate from Arvind's)
- DPO+LoRA aerospace Q&A system (small dataset, 5 examples, 1 epoch)
- Browser-based agent experiments
- Results were below baseline (Arvind's feedback: "5 examples one epoch and model scores below baseline, its weak")

**HF Model:** None published
**Repos:** `ArumugamKrishnan/Capstone-Project-Qwen-0.5B-Agent-`, `ArumugamKrishnan/Agentic-RLHF-for-Aerospace-using-Browser-DPO`

**Best-Fit Experiments for Paper:**
- **Independent replication study** — validates main pipeline on separate implementation
- **Domain transfer analysis** — aerospace as out-of-domain test case
- **Data efficiency study** — how few examples are needed for meaningful RL signal
- **DPO vs. GRPO comparison** — natural extension using his DPO expertise

---

## Experiment Allocation Matrix

| Experiment | Primary Owner | Why This Person | Running On | Status |
|---|---|---|---|---|
| **Scaling Law (8B→32B→235B)** | Arvind | His core thesis, led all scaling work | Tinker | Running |
| **Frontier Models (DeepSeek-V3.1, Nemotron-120B, GPT-OSS)** | Arvind | Designed the model selection, API expertise | Tinker | Running |
| **Cross-Task Transfer (GSM8K→tool_use)** | Arvind + Sandhya | Arvind's framework, Sandhya's tool-call data | Tinker | Running |
| **MoE vs. Dense (Qwen3-30B variants)** | Sandhya | Extends her scaling/algorithm comparison work | Tinker | Running |
| **Multi-Turn Tool Chaining** | Sandhya | Her flagship 0%→92% result | Prior work | Complete |
| **HumanEval GRPO Evaluation** | Madhu | Directly extends his code-generation pipeline | Modal H100 | Running |
| **Code Reasoning Format Analysis** | Madhu | His Qwen3-8B code model | Prior work | Complete |
| **Small-Model Amplification (4B→100%)** | Mohammad | His headline Qwen3-4B result | Prior work | Complete |
| **PPO Baseline Comparison** | Dhruva | Extends his baseline framework with new algo | Modal H100 | Running |
| **Negative Controls & Baselines** | Dhruva | His 59 W&B runs establish the comparison | Prior work | Complete |
| **KL Divergence Tracking** | Dhruva + Arvind | MLOps monitoring aligns with his pipeline work | Modal H100 | Relaunching |
| **Held-Out Evaluation (Qwen3-32B, Qwen3.5-27B)** | Dhruva | Evaluation harness is his strength | Modal H100 | Running |
| **Independent Replication** | Arumugam | Validates pipeline independently | Prior work | Complete |
| **DPO Domain Transfer (Aerospace)** | Arumugam | His unique domain application | Prior work | Complete |
| **New Architecture (GPT-OSS-20B, Kimi-K2)** | Arvind | Frontier model evaluation | Tinker | Running |

---

## How Each Member's Work Maps to Paper Sections

| Paper Section | Team Members | Content |
|---|---|---|
| §3 Methodology | Arvind (pipeline), Dhruva (MLOps) | TinkerRL framework, evaluation harness |
| §4.1 Scaling Experiments | Arvind | 8B→32B→235B GRPO scaling curves |
| §4.2 Algorithm Comparison | Sandhya, Dhruva | GRPO vs. DPO vs. PPO, MoE vs. Dense |
| §4.3 Cross-Task Transfer | Arvind, Sandhya | GSM8K→tool_use generalization |
| §4.4 Domain-Specific | Madhu (code), Mohammad (logic), Arumugam (aerospace) | Task-specific GRPO results |
| §4.5 World-Class Suite | All 6 | 20-experiment parallel run results |
| §5 Analysis | Arvind (scaling laws), Sandhya (algorithm), Dhruva (baselines) | Quantitative analysis |
| §6 Negative Results | Dhruva, Arumugam | What doesn't work and why |

---

## Recommendations for Strengthening Each Member's Contribution

### Arvind — Already strong. Focus on:
- Generating publication-quality figures from the 14 Tinker experiments currently running
- Writing the scaling law analysis with proper power-law fitting

### Sandhya — Strong results, needs visibility:
- Add her 3-phase tool-call results as a dedicated subsection in the paper
- Create a figure showing the 0%→92% JSON validity improvement curve

### Madhu — Good results, needs rigor:
- The 86% HumanEval claim needs proper benchmarking against published baselines
- Run the Modal HumanEval evaluation to get standardized pass@1/pass@10 metrics

### Mohammad — Impressive result, needs context:
- 41.6%→100% on logical reasoning is remarkable — but needs standard benchmark comparison
- Should run on ARC, HellaSwag, or MMLU to contextualize the improvement

### Dhruva — Undervalued contributions:
- His negative results (0.5B and 1.5B failing to improve) are scientifically important
- The 59 W&B runs constitute a serious ablation study — should be highlighted
- PPO baseline from Modal will be his key new contribution

### Arumugam — Needs strengthening:
- 5-example training is too weak — increase to 50+ examples
- The aerospace domain is unique and interesting — needs proper evaluation
- Independent replication angle is valuable for the paper's credibility

---

## Author Order Recommendation (for Conference Paper)

1. **Arvind C R** — First author (project lead, 80%+ of infrastructure and experiments)
2. **Sandhya Jeyaraj** — Second (strongest independent results after Arvind)
3. **Madhu Kumara L** — Third (code-generation results, consistent contributor)
4. **Mohammad Rafi** — Fourth (impressive single result, late but impactful)
5. **Dhruva N Murthy** — Fifth (MLOps and evaluation infrastructure)
6. **Arumugam K** — Sixth (independent validation, domain transfer)

This order reflects both quantity and quality of contributions as evidenced by the WhatsApp record, GitHub commits, HF models, and W&B logs.
