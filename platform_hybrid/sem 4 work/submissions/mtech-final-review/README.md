# M.Tech Final Thesis Review Deck

Reviewer-facing presentation for the **Tinker RL Lab** project — a multi-framework
benchmark and study of GRPO-style RL post-training of LLMs.

- **Student:** Arvind C R (Arvind Chitra Rajasekaran), SRN `<SRN>` (placeholder)
- **Program:** M.Tech Data Science & AI, PES University
- **Guide:** Ramesh Prakash Guledgudd
- **Deck:** `Arvind_MTech_Thesis_Review.pptx` (21 slides, 16:9)

## Files

| File | Purpose |
|---|---|
| `Arvind_MTech_Thesis_Review.pptx` | The presentation (generated artifact) |
| `build_review_pptx.py` | Reproducible generator (python-pptx 1.0.2) |
| `README.md` | This file |

## Rebuild

```bash
cd "sem 4 work/submissions/mtech-final-review"
python build_review_pptx.py --check     # writes the .pptx and round-trip loads it
```

The script resolves the repo root relative to its own location and reads figure
PNGs from `paper/figures/` and `paper/figures/v2/`. If a figure PNG is missing it
draws a text panel instead of a broken image, and prints the missing path.

## Design

Clean academic: white background, PES-blue titles (`#1F4E79`), Calibri, an accent
rule under each title, slide numbers, and captioned figures. Navy table headers
with alternating row fills. No clip-art.

## Slide map

1. Title · 2. Agenda · 3. Problem & motivation · 4. Objectives ·
5. Literature context (RLHF→PPO→DPO→GRPO) · 6. System architecture (6 frameworks) ·
7. Four de-confound pillars · 8. Same-stack PPO vs GRPO · 9. ZVF diagnostic ·
10. Group size & trainability · 11. Length bias & held-out generalization ·
12. Scaling behavior · 13. P1–P8 contribution map · 14. LLM vs XGBoost fraud (P8) ·
15. Reproducibility & audit apparatus · 16. Publications & submissions ·
17. Demo pointer · 18. Semester 3 vs 4 ownership · 19. Limitations & future work ·
20. Conclusions · 21. Thank you / Q&A.

## Embedded figures (6)

All are real PNGs from the paper figure sets:

- `paper/figures/v2/ppo_vs_grpo.png` (slide 8)
- `paper/figures/zvf_by_library.png` (slide 9)
- `paper/figures/v2/group_size_ablation.png` (slide 10)
- `paper/figures/length_vs_reward.png` (slide 11)
- `paper/figures/scaling_law_fit.png` (slide 12)
- `paper/figures/p8_cost_per_caught.png` (slide 14)

## Claim → source map

Every number on a slide is pulled from a repository file:

| Claim | Value | Source file |
|---|---|---|
| Headline GRPO Qwen3-8B GSM8K | peak 62.5%, last-10 34.4% | `REPRODUCE.md`, `experiments/experiment_summary.md` |
| Held-out GSM8K (5 seeds × 200) | GRPO 83.3% (SD 2.2%) vs base 82.0%; t=1.32, p=0.26 | `reports/final/grpo_agentic_llm_paper.tex`, `result_ledger.md` |
| Same-stack PPO vs GRPO | Welch p=0.7605; paired Δ=−0.002, p=0.374 | `paper/sections/_shared_methods.tex`, `frontier_synthesis_scaling.tex` |
| PPO vs GRPO by model | Qwen3-8B GRPO +11.9pp; Llama-3.1-8B PPO +13.1pp | `experiments/experiment_summary.md` |
| ZVF by library | vanilla GRPO 0.481; AERO 0.220; N=80 | `paper/sections/zvf.tex` |
| Group-size ablation | G=2..16 peak/last-10; SNR ~52% of √G; retention ≈1.00 | `REPRODUCE.md §4.1`, `paper/sections/p3_abstract.tex` |
| Scaling | flat slope over ~2.4 OOM; 2/12 anchors match 3-phase; Nemotron collapse 0.55 | `paper/sections/p1_abstract.tex` |
| TRL vs Tinker same task | 73.4% vs 99.9%, p=0.0014 | `LIMITATIONS_AND_IMPACT.md §8` |
| Tool-use JSON validity | 0% → 92% (SFT+GRPO) | `demo_recording/README.md`, `reports/final/grpo_agentic_llm_paper.tex` |
| P8 XGBoost | AUC 0.7955, F1 0.356, prec 0.723, recall 0.236 | `xgboost_results.json`, `paper/sections/p8_abstract.tex` |
| P8 LLM scorer | Qwen3.5-4B acc 0.792, AUC 0.483 (chance); ~85× cheaper triage | `paper/sections/p8_abstract.tex` |
| Total runs | 44 curated; 70+ across roster | `experiments/experiment_summary.md`, `paper/sections/p1_abstract.tex` |
| Audit suite | 13-audit `run_all_audits.py`; 17 `*_audit.py` total | `run_all_audits.py` |
| Semester boundary | tag `capstone-final-2026-04-25` (`21a99ef7`); workshop `b0ac85bf` 2026-06-21 | `PROJECT_HISTORY.md`, `sem 4 work/submissions/neurips-workshop/README.md` |
| Workshop title | "RL-Finetuning Bench: An Exploratory Workshop Artifact…" | `sem 4 work/submissions/neurips-workshop/README.md` |
| P1–P8 titles | full titles | `sem 4 work/README.md` |
| Reproduction budget | ~446 GPU-h full; ~5 GPU-h headline | `REPRODUCE.md §6` |

## Notes / unverified

- **`<SRN>`** is a placeholder — no student ID was found in the repo.
- The **date "July 2026"** on the title slide is derived from the current review
  window (workshop PDF compiled 2026-07-10); it is not a fixed submission date in
  the repo.
- The task referenced a `thesis_demo` notebook. No file named `thesis_demo.ipynb`
  exists; the demo artifacts that do exist are the HuggingFace Space, the
  `submission/demo/demo.sh` runbook (`reports/esa_phase1/CODE_WALKTHROUGH.md`),
  and `demo_recording/demo.mp4`. Slide 17 points to those instead.
