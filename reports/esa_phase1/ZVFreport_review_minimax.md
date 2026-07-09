## Adversarial Review — ZVF Phase-1 Report

### A. TOP 5 WEAKNESSES

1. **Implementation chapter is a façade.** §5 lists module names and one-line responsibilities but contains **zero code, zero listings, zero file paths, zero LOC counts**. The PES rubric requires ~80% code implementation; a Phase-1 evaluator will read §5.1–5.4 (≈1.5 pages) and conclude no implementation was submitted. Fix: append full module listings (tinkerrl/grpo.py, p2_collapse_analysis.py, p8_detector.py) and a Git repo link.

2. **The headline contribution is "nothing works."** Six of eight studies report null/negative/single-seed results (§6.1–6.6). PES Phase-1 expects demonstrable intermediate results. A reviewer will ask: if every lever you tested loses, what is the candidate actually contributing? Fix: reframe P5/P8 as the *deployed* contribution with measurable, multi-seed wins, and demote failed algorithmic bets to a single "negative findings" subsection.

3. **Held-out n=8–20 makes every gain statistically meaningless.** §6.8 admits this, yet Table 6.3 still runs a t-test on n=3 seeds (§6.4), and §6.5 reports layer-overlap at n=2 seeds. A NeurIPS reviewer would reject on this. Fix: hold out ≥100 examples per arm or report strictly descriptive statistics; drop the "t≈0.6" language which is misleading.

4. **Internal contradictions across sections.** Abstract claims P1 is "positive at 1.5B"; Figure 4.2 colours P1 amber ("measured/null"). P8 is cited as AUROC 0.84 (abstract), 0.63 (Fig. 4.2), and 0.838±0.010 (§6.6). "Qwen3.5-4B" appears in §3.6/§6.4 but no Qwen3.5 exists — likely Qwen2.5 or Qwen3. Fix: a single canonical numbers table, corrected model name.

5. **"Two frontier models as adversarial auditors" is circular.** §6.8 claims LLMs validated the experiments — but the experiments *train* LLMs, and the auditors are LLMs. This is not independent verification. Fix: replace with code-based recomputation by a non-LLM script (or human spot-check) and a SHA-pinned artifact per Appendix B.

### B. FACTUAL / TECHNICAL / OVERCLAIMS

- "**ZVF ≈ 0.72–0.77** is the **single largest source of wasted compute** in GRPO" (§1.1) — overclaim; no comparison vs. PPO/DPO cost.
- "**AUROC 0.84 vs 0.43**" (§6.6) cited inconsistently (0.838 in body, 0.63 in Fig. 4.2).
- Bibliography: refs [18], [22], [23] are **dated 2026** with implausible arXiv IDs (2602.x, 2601.x, 2605.x) and forward-dated venues ("NeurIPS 2025", "ICML 2026") — verify these exist.
- "**t ≈ 0.6**" (§6.4) on n=3 is not meaningful.
- Eq. (4.1) uses population std (1/G) instead of sample std (1/(G−1)) — a real GRPO uses sample std; silently inconsistent with DeepSeek-Math.

### C. RUBRIC / FORMAT GAPS

- **Candidate's Contribution** is buried in §1.6, not a stand-alone section as rubric requires.
- **Page count** appears ≈30 body pages (front matter + 7 chapters + 3 appendices), short of the **≥40 pages excl. front matter**.
- No **plagiarism report** referenced.
- Future Work (§7.2) is 4 bullets — too thin for a Phase-2 plan.
- No **List of Abbreviations**, no **List of Symbols** beyond Appendix C (rubric-typical).

### D. MISSING CONTENT

- No system architecture diagram for telemetry pipeline (§4.2).
- No discussion of **safety / ethical / compute-impact** of RLVR post-training.
- No **comparison with reported DeepSeek-R1 / TÜLU-3 numbers**.
- No **GitHub repo / Docker / dependency lockfile** in Appendix B.
- §1.6 doesn't state candidate's *specific* contribution (what code, what analysis).

### E. VERDICT

**Major revision.** Scientifically honest but implementation-thin, internally inconsistent, and below PES Phase-1 page/code thresholds.
