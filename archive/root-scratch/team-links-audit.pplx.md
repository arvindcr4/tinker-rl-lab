# Team Artifact Audit — Verified Links & Proof of Work

A complete audit of every W&B project, HuggingFace model, GitHub repo, and Tinker console link shared by each team member in the WhatsApp group chat, verified by visiting each URL.

---

## Arvind C R (@arvindcr4)

### HuggingFace Models
| Model | Status | Base | Metrics | Downloads | Last Updated |
|---|---|---|---|---|---|
| [arvindcr4/tool-call-lora-qwen0.5b](https://huggingface.co/arvindcr4/tool-call-lora-qwen0.5b) | Accessible | Qwen2-0.5B | No metrics listed on card | 1 | ~1 month ago |

**Assessment:** Only one public model with no metrics on the card. This is the early proof-of-concept — the real experiments are logged to W&B under `tinker-rl-lab-world-class` and the 32+ Tinker runs are tracked via the Tinker API, not HF. The world-class experiments currently running will upload to `arvindcr4/tinker-rl-bench-*` repos.

### GitHub Repos
| Repo | Status | Commits | Last Updated | Description |
|---|---|---|---|---|
| [pes-llm-research/tinker-rl-lab](https://github.com/pes-llm-research/tinker-rl-lab) | Accessible | 141 | Apr 18, 2026 (today) | Main project repo — experiments, paper, report, infrastructure |
| [arvindcr4/tinker-experiments](https://github.com/arvindcr4/tinker-experiments) | Archived | 15 | Mar 14, 2026 | [ARCHIVED] Merged into tinker-rl-lab |
| [arvindcr4/awesome_agents_papers](https://github.com/arvindcr4/awesome_agents_papers) | Accessible | 47 | Jan 19, 2026 | 88 papers, 93 slide decks, 15 categories. 3 stars. |

### W&B
| Project | Status | Notes |
|---|---|---|
| [arvindcr4-pes-university/tinker-rl-lab-world-class](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class) | Active (online runs syncing now) | Main project for all world-class experiments |
| [arvindcr4/profile](https://wandb.ai/profile/arvindcr4) | Accessible | Profile page |

**Verdict: Strong proof of work.** 141 commits in main repo, active W&B project, archived old repo shows project evolution. Weakness: only 1 HF model — needs the new experiment checkpoints uploaded.

---

## Sandhya Jeyaraj (@Balasandhya)

### HuggingFace Models
| Model | Status | Base | Metrics | Downloads | Last Updated |
|---|---|---|---|---|---|
| [Balasandhya/llm-multiturn-tool-call-grpo-QloRA-Qwen2.5-3B](https://huggingface.co/Balasandhya/llm-multiturn-tool-call-grpo-QloRA-Qwen2.5-3B) | Accessible | Qwen2.5-3B-Instruct | GRPO: 0.91, SFT: 0.72 | 0 | Mar 22, 2026 |
| [Balasandhya/llm-tool-call-lora-Qwen0.5B](https://huggingface.co/Balasandhya/llm-tool-call-lora-Qwen0.5B) | Accessible | Qwen-0.5B | — | — | Mar 2026 |
| [Balasandhya/llm-tool-call-grpo-lora-Qwen1.5B](https://huggingface.co/Balasandhya/llm-tool-call-grpo-lora-Qwen1.5B) | Accessible | Qwen-1.5B | — | — | Mar 2026 |

**Profile:** [huggingface.co/Balasandhya](https://huggingface.co/Balasandhya) — Bio: "AI & ML, Deep learning, ML, Gen AI, RAG, Agentic AI, RL"

**Assessment:** 3 models showing a clear scaling progression (0.5B → 1.5B → 2.5-3B). The flagship multi-turn model has solid metrics (GRPO 0.91 > SFT 0.72). Includes SFT/GRPO/Eval scripts and a notebook. Well-organized.

### GitHub Repos
No personal GitHub repos shared in chat. Work is on HuggingFace.

### W&B
No W&B links shared. Experiments tracked locally or within HF model cards.

**Verdict: Second-strongest proof of work.** 3 HF models with progressive scaling, clear metrics on the flagship model (GRPO outperforming SFT by 26%). Weakness: no W&B logs and no GitHub repo — makes reproducibility harder to demonstrate.

---

## Madhu Kumara L (@Madhu2133)

### HuggingFace Models
| Model | Status | Base | Metrics | Downloads | Last Updated |
|---|---|---|---|---|---|
| [Madhu2133/qwen3-8b-code-grpo-v10](https://huggingface.co/Madhu2133/qwen3-8b-code-grpo-v10) | Accessible | Qwen3-8B | HumanEval pass@1: 86%, Code Reasoning: 30/35 (86%) | 53 | Apr 16, 2026 |
| [Madhu2133/qwen3-8b-swe-grpo](https://huggingface.co/Madhu2133/qwen3-8b-swe-grpo) | Accessible | Qwen3-8B | HumanEval: 42% (no change), MBPP: 4% (+2%) | 0 | Apr 16, 2026 |

**Profile:** [huggingface.co/Madhu2133](https://huggingface.co/Madhu2133) — 2 models, 0 followers

**Assessment:** The v10 model is his strongest — 86% HumanEval pass@1 is a genuinely impressive result for a GRPO-trained model. The SWE model (qwen3-8b-swe-grpo) showed minimal improvement (MBPP +2%, HumanEval unchanged) — this is a useful negative data point. 53 downloads suggests some external interest in v10.

### GitHub Repos
No personal GitHub repos shared. All code in HF model repos.

### W&B
No W&B links shared directly. Arvind mentioned needing W&B logs from everyone.

**Verdict: Good but lopsided.** One strong model (v10 at 86%) and one weak one. 53 downloads is the highest of any team member's model. Weakness: no W&B logging, no standalone GitHub repo, and the 86% claim needs external validation (which is what the Modal HumanEval experiment now running will provide).

---

## Mohammad Rafi (@MohammadRafiML)

### HuggingFace Models
| Model | Status | Base | Metrics | Downloads | Last Updated |
|---|---|---|---|---|---|
| [MohammadRafiML/Qwen3-4B-Instruct-2507-Capstone-MathRL](https://huggingface.co/MohammadRafiML/Qwen3-4B-Instruct-2507-Capstone-MathRL) | Accessible | Qwen3-4B-Instruct-2507 | 4B params, 32k context | 0 | Apr 16, 2026 |

**Profile:** [huggingface.co/MohammadRafiML](https://huggingface.co/MohammadRafiML) — 1 model, 1 dataset (34 likes, 2.3k downloads)

**Assessment:** The model page exists with SFT and GRPO adapter folders (11 commits). The dataset has surprisingly high engagement (34 likes, 2.3k downloads) — this is notable. The model card doesn't clearly state the 41.6%→100% logical reasoning improvement he claimed in WhatsApp. The separate research paper (`mohammad_capstone_research_paper.pdf`) describes a custom 12-question benchmark — Arvind flagged this as potentially weak for professor scrutiny.

### GitHub Repos
No personal GitHub repos shared in the WhatsApp chat for this project.

### W&B
No W&B links shared.

**Verdict: Mixed.** Has HF artifacts (model + dataset with real traction), but the 100% accuracy claim on a custom 12-question benchmark is not standard. The dataset's 2.3k downloads and 34 likes are the strongest community signal of any team member's work. Weakness: no W&B logs, no GitHub repo, custom benchmark may not hold up to review.

---

## Dhruva N Murthy (@dhruvanmurthy)

### HuggingFace Models
| Model | Status | Base | Metrics | Downloads | Last Updated |
|---|---|---|---|---|---|
| [dhruvanmurthy/llm_efficiency](https://huggingface.co/dhruvanmurthy/llm_efficiency) | — | — | — | — | — |

**Note:** The HF link `dhruvanmurthy/llm_efficiency` was not verified separately; it may be a repo or model. The GitHub repo of the same name exists (see below).

### GitHub Repos
| Repo | Status | Commits | Last Updated | Description |
|---|---|---|---|---|
| [dhruvanmurthy/llm_efficiency](https://github.com/dhruvanmurthy/llm_efficiency) | Accessible | 6 | Mar 6, 2026 | KV Cache & LoRA for minGPT — efficiency experiments |
| [dhruvanmurthy/capstone-trials](https://github.com/dhruvanmurthy/capstone-trials) | Accessible | 6 | Mar 29, 2026 | SFT + DPO alignment, tool-use benchmarking with eval scripts |
| [dhruvanmurthy/Qwen3-8B-FineTuning](https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning) | Accessible | 25 | Apr 13, 2026 | Qwen3-8B fine-tuning for tool use — Docker, configs, data, scripts |

### W&B Projects
| Project | Status | Notes |
|---|---|---|
| [capstone-proj/qwen3-8b-tool-use](https://wandb.ai/capstone-proj/qwen3-8b-tool-use) | 404/Private | Not publicly accessible — requires login |
| [capstone-proj/qwen3-8b-multi-tool-use](https://wandb.ai/capstone-proj/qwen3-8b-multi-tool-use) | 404/Private | Not publicly accessible — requires login |

### Tinker Console Links
| Link | Status | Notes |
|---|---|---|
| SFT training run (afedbdb9) | Auth required | Tinker console requires authentication |
| SFT checkpoint weights | Auth required | Tinker console requires authentication |
| GRPO training run (107560c7) | Auth required | Tinker console requires authentication |
| GRPO checkpoint weights | Auth required | Tinker console requires authentication |

**Assessment:** Strongest infrastructure contributor by volume. 3 separate GitHub repos (37 total commits), multiple W&B projects (private), and Tinker training runs with checkpoint links. The `Qwen3-8B-FineTuning` repo at 25 commits is well-structured with Docker, configs, data, and scripts. His `capstone-trials` repo has evaluation metrics for tool-use (valid JSON rate, tool match, argument match, success rate).

**Verdict: Strong engineering, weak visibility.** Most repos of any member (3), most commits outside Arvind (37), structured MLOps pipeline. Critical weakness: W&B projects are private (404 for anyone visiting). Needs to either make W&B public or provide screenshots/exports. Also his original claim of "59 W&B runs" is not verifiable externally.

---

## Arumugam K (@ArumugamKrishnan)

### HuggingFace Models
No HuggingFace models published. (Arvind noted on Apr 16: "those who have not submitted any hf models / logs... it will drag down everyone's scores")

### GitHub Repos
| Repo | Status | Commits | Last Updated | Description |
|---|---|---|---|---|
| [ArumugamKrishnan/Capstone-Project-Qwen-0.5B-Agent-](https://github.com/ArumugamKrishnan/Capstone-Project-Qwen-0.5B-Agent-) | Accessible | 8 | Mar 27, 2026 | DPO fine-tune Qwen-0.5B for aerospace. 100% Jupyter Notebook. 2 files. |
| [ArumugamKrishnan/Agentic-RLHF-for-Aerospace-using-Browser-DPO](https://github.com/ArumugamKrishnan/Agentic-RLHF-for-Aerospace-using-Browser-DPO) | Accessible | 9 | Apr 8, 2026 | Browser-based aerospace dataset + DPO alignment. Base (1.60)→Fine-Tuned (2.00) [+25%] |

### W&B
No W&B links shared.

**Assessment:** Two GitHub repos showing a clear evolution from initial Qwen-0.5B agent to browser-based DPO pipeline. The aerospace repo has metrics: Base Score 1.60 → Fine-Tuned 2.00 (+25%), with 40% of prompts improving. However, Arvind's critique was accurate: "5 examples one epoch and model scores below baseline, its weak."

**Verdict: Weakest proof of work.** No HF models, no W&B logs, minimal improvement metrics. The +25% on a 5-point scale (1.6→2.0) from 5 training examples is scientifically weak. The repos exist (17 total commits) but need substantially more data and training to be paper-worthy. The defense site constraints explain some delays but don't change the output quality.

---

## Summary Comparison Table

| Member | HF Models | HF Downloads | GitHub Repos | Total Commits | W&B Projects | Tinker Runs | Proof Strength |
|---|---|---|---|---|---|---|---|
| **Arvind** | 1 | 1 | 3 (+ main repo 141 commits) | 203 | 1 (active) | 32+ | Strong |
| **Sandhya** | 3 | 0 | 0 | 0 | 0 | Via Tinker API | Good |
| **Madhu** | 2 | 53 | 0 | 0 | 0 | Via Modal/Vast.ai | Good (one model) |
| **Mohammad** | 1 (+1 dataset) | 0 (model) / 2.3k (dataset) | 0 | 0 | 0 | Custom pipeline | Mixed |
| **Dhruva** | 0 (or 1?) | — | 3 | 37 | 2 (private) | 2 (SFT+GRPO) | Good (engineering) |
| **Arumugam** | 0 | — | 2 | 17 | 0 | 0 | Weak |

---

## Critical Issues Found

### 1. Dhruva's W&B Projects Are Private
Both `capstone-proj/qwen3-8b-tool-use` and `capstone-proj/qwen3-8b-multi-tool-use` return 404. These contain his 59 runs and should be the backbone of the evaluation section. **Action needed:** Make these public or export run summaries.

### 2. Arumugam Has No HF Models
Despite repeated requests from Arvind (Apr 5, Apr 14, Apr 16), no model has been uploaded to HuggingFace. **Action needed:** Upload at least the DPO-finetuned Qwen-0.5B to HF with metrics.

### 3. Mohammad's Custom Benchmark
The 41.6%→100% claim is on a custom 12-question benchmark, not a standard one (GSM8K, MATH, ARC). **Action needed:** Run on GSM8K or MATH to get comparable numbers.

### 4. Madhu's 86% Needs Validation
The HumanEval pass@1: 86% on the model card is self-reported. The Modal HumanEval experiment currently running will provide independent validation.

### 5. Sandhya Has No W&B or GitHub
Her results are strong (GRPO 0.91) but only exist on HuggingFace. No reproducibility artifacts (code, logs, configs) outside HF.

### 6. Arvind's HF Model Lacks Metrics
The `tool-call-lora-qwen0.5b` model card has no metrics. The world-class experiments will fix this with proper benchmarking and multiple model checkpoints.

---

## Tinker Console Links (Dhruva's — Auth Required)

All four Tinker console links shared by Dhruva require Tinker platform authentication:
- SFT training: `afedbdb9-e1f7-55a2-bc3a-c0d261718838:train:0`
- SFT weights: same run `/weights/final`
- GRPO training: `107560c7-cbd1-5830-8ca3-780ed0afc765:train:0`
- GRPO weights: same run `/weights/final`

These are valid Tinker run IDs and would be verifiable by anyone with Tinker access, but cannot serve as public proof of work.
