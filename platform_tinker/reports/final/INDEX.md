# reports/final/ — INDEX

**Purpose:** Final capstone deliverables for the GRPO agentic-LLM project (Group 6): the integrated final report, standalone conference-paper drafts, reviewer-response material, held-out GSM8K evaluation code + result JSONs, and build/audit scripts. See `README.md` for status + A-grade completion path.

**Key files:**
- `capstone_final_report.md` / `.docx` — full capstone report (honest-about-limitations narrative)
- `group6_final_report.tex` / `.docx` / `group6_extracted.txt` — canonical integrated report (paper findings across chapters); `build_group6_final_report.sh` rebuilds it; `group6_final_report_with_appended_paper.tex` = legacy wrapper
- `grpo_agentic_llm_paper.tex` / `.md` / `_anonymous.tex` + `references.bib` + `nips_style.sty` — standalone conference-paper drafts
- `supplementary_appendix.tex`, `group6_experiment_coverage_addendum.tex` — supplements
- `evaluate_gsm8k_test.py` — held-out GSM8K eval (Tinker API or local HF); `run_*batches*.sh`, `run_heldout_parallel.sh`, `retry_even_batches.sh` — batch drivers
- `gsm8k_*results*.json`, `gsm8k_heldout_seed{042,137,256,512,999}.json`, `gsm8k_base_control_200.json` — eval outputs (5-seed held-out + base control)
- `PAPER_IMPROVEMENT_PLAN.md`, `CONSOLIDATED_REVIEW_IMPROVEMENTS.md`, `SUBMISSION_CHECKLIST.md`, `SUBMISSION_README.md` — planning/review docs
- Reviewer-response dumps: `edison_ablation_report.md` + `GRPO_Ablation_..._figure_{1-9}.png` (Edison research report)
- `fig{1,2,3}_*.png` + `generate_figures.py` — capacity-threshold / diagnostics / synthetic-vs-real figures
- `prepare_blind_review_package.py`, `update_paper_with_results.py` — packaging/update helpers

**Subfolders:**
- `addendum/` — 8 numbered reviewer-response addenda notes (see its INDEX.md)
- `_staging/` — pre-8-agent draft snapshot (see its INDEX.md)

**Find it fast:**
- to run held-out eval → `evaluate_gsm8k_test.py` (see README A-grade path)
- to rebuild the final report → `build_group6_final_report.sh`
- to see reviewer-fix roadmap → `PAPER_IMPROVEMENT_PLAN.md` + `CONSOLIDATED_REVIEW_IMPROVEMENTS.md`
