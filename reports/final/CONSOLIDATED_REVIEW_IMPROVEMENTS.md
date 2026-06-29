# Consolidated Review: 60 Questions Across 3 Sources
**Date:** 2026-04-04
**Sources:** ChatGPT Pro Extended (held-out analysis), ChatGPT Instant (batches 1,3,5,7,9), Claude Sonnet 4.6 (Q1-50 Area Chair review)

---

## IMPROVEMENTS ALREADY APPLIED TO PAPER

1. **Title**: Changed to "When Does GRPO Work? Capacity Thresholds and Training Dynamics for Agentic LLM Fine-Tuning Across Three Domains"
2. **Abstract**: Foregrounded 3 novel findings with p-values; added base-model caveat for held-out results; code result labeled "not significant"
3. **Agentic definition**: Added operational definition in Introduction
4. **Held-out reframing**: Explained training reward vs test accuracy incomparability; added base-model control caveat
5. **HumanEval**: Labeled "randomly sampled"; added Fisher's exact p=0.53
6. **Capacity threshold**: Reframed from "novel" to "hypothesis" with confound acknowledgment
7. **SFT+GRPO**: Reframed from "novel" to "observed" with warm-starting acknowledgment
8. **Cross-seed stability**: Reframed from "novel" to "methodological"
9. **MoE volatility**: Added Levene's test p-value in conclusion
10. **Hyperparameter selection**: Disclosed as manual tuning
11. **Limitations**: Strengthened with base-model control, group-size ablation gaps
12. **ReST/STaR citations**: Added to Related Work (P3)
13. **MoE literature**: Added Switch Transformer, GLaM, Mixtral context (P4)
14. **ToolLLM/Gorilla citations**: Added to Related Work tool-use paragraph (M4)
15. **Anonymous paper**: Fully synced with main paper content
16. **MoE mechanism**: Proposed optimization interference hypothesis (policy gradient vs load-balancing loss)
17. **Unified narrative**: Added "GRPO succeeds when model generates within-group reward variance" framing to conclusion
18. **3B negative result**: Formalized with binomial CI and p-value vs random baseline
19. **Synthetic-real vs train-test**: Explicitly distinguished as different phenomena
20. **Claim-to-run table**: Added to supplementary (Table 8)
21. **Base-model control**: Qwen3-8B without LoRA = 82.0%, GRPO delta +1.3pp not significant (p=0.26). Updated abstract, held-out section, limitations, cross-domain table in both papers + capstone report.
22. **Group size ablation**: G=4→23.8%, G=8→24.4%, G=16→36.2%, G=32→54.7% on 8B. New subsection + table in main paper, anonymous paper, capstone report, supplementary appendix.
23. **4B multi-seed replication**: Seeds 42/137/256/512, mean 84.7% (SD=12.0%). Updated capacity threshold, cross-domain table, conclusion in all three documents.
24. **3B G=32 control**: Llama-3.2-3B with G=32 yields 5.0% (vs 2.3% at G=4), zero-loss 18% (vs 56%). Confirms capacity threshold is not exploration artifact. Updated capacity section, cross-domain table, conclusion, limitations in both papers + supplementary.
25. **Configuration table (P5)**: Claim-to-configuration table added to supplementary. Main papers reference inline.
26. **Repository link (Q47)**: Code at github.com/arvindcr4/grpo-agentic-anonymous. Anonymous proxy URL in anonymous paper.
27. **Reward ablation + baselines**: Acknowledged in limitations as future work.

---

## TOP PRIORITY IMPROVEMENTS (Not Yet Applied)

### P1: Base Model Control (All reviewers flagged) ✅ DONE
**Result:** Base Qwen3-8B without LoRA = 164/200 = 82.0% (CI [76.5%, 87.5%])
**Impact:** GRPO adds only +1.3pp (83.3% vs 82.0%), t=1.32, p=0.26 (not significant). This fundamentally changes the narrative — held-out accuracy is attributable to base model capability, not GRPO.
**Updated:** Abstract, held-out section, limitations, cross-domain table in both main and anonymous papers + capstone report.

### P2: Group Size Ablation (Q31, Q38 — "most critical missing ablation") ✅ DONE
**Result:** 8B: G=4→23.8%, G=8→24.4%, G=16→36.2%, G=32→54.7%. 3B: G=32→5.0% (vs G=4→2.3%), zero-loss drops 56%→18% but accuracy stays near random. **Capacity threshold confirmed as real, not exploration artifact.**
**Updated:** Main paper (Group Size Ablation subsection + Table + 3B G=32 in capacity section), anonymous paper (same), capstone report, supplementary (run registry + zero-loss table + practitioner guide).

### P3: Add ReST/STaR Citations (Q13 — "significant omission") ✅ DONE
**Action:** Added ReST (Gulcehre 2023), STaR (Zelikman 2022) to Related Work "Preference-based and iterative self-training" paragraph

### P4: Add MoE Literature (Q15 — "cannot assess if 2.43x is surprising") ✅ DONE
**Action:** Added Switch Transformer, GLaM, Mixtral citations in new "MoE training instability" paragraph in Related Work

### P5: Configuration Table (Q20 — "essential for reproducibility") ✅ DONE
**Result:** Added claim-to-configuration table (9 claims × 6 hyperparameters) in Supplementary Appendix. Main paper references it inline. Combined with claim-to-run mapping (M1), all claims are now fully traceable.

---

## MEDIUM PRIORITY IMPROVEMENTS

### M1: Claim-to-Run Mapping (Q45) ✅ DONE
Added Table 8 in supplementary mapping all 11 claims to supporting runs. 5 claims marked as single-run.

### M2: Figure Caption Self-Containedness (Q18) ✅ DONE
All 3 figure captions rewritten with axis labels, metric definitions, model names, and key numerical values.

### M3: Separate Synthetic-Real Gap from Train-Test Discrepancy (Q19) ✅ DONE
Added explicit sentence in Section 5.2 distinguishing data distribution effect from metric discrepancy.

### M4: Add ToolBench/APIBench/BFCL Context (Q11, Q43) ✅ DONE
Added ToolLLM and Gorilla citations to Related Work "Tool-use evaluation" paragraph.

### M5: Negative Result Formalization (Q50) ✅ DONE
Added binomial CI [0.3%, 4.4%], p=0.68 vs random baseline in main paper + practitioner guideline in supplementary.

---

## KEY REVIEWER REBUTTAL POINTS

### "No PPO/REINFORCE baselines" (Q41)
**Rebuttal:** Paper frames as training-dynamics characterization study, not comparative efficiency claim. $50/person budget precludes PPO value network. REINFORCE++ comparison in progress via Colab notebook.

### "This is just a systems paper" (Q10 from original)
**Rebuttal:** We report quantified phenomena not derivable from existing papers: (1) zero-gradient group saturation rates explaining the 3B-4B boundary, (2) Levene's test-confirmed MoE variance amplification (p=7e-6), (3) formal held-out replication across 5 seeds where most GRPO papers report single runs.

### "50 steps isn't meaningful RL" (Q42)
**Rebuttal:** 50 steps x batch 2 x G=8 = 800 unique samples. The 100-step run achieves identical performance (27.5% last-10), suggesting saturation. Budget-constrained; extended run included.

---

## QUESTIONS THAT EXPOSED REAL WEAKNESSES

| # | Question | Severity | Status |
|---|----------|----------|--------|
| Q7 | Capacity threshold from 2 data points | Critical | ✅ Now 4-seed 4B replication + group ablation; reframed as hypothesis |
| Q31 | No group size ablation | Critical | ✅ DONE — G=4/8/16/32 on 8B (Table in paper) |
| Q44 | RLHF confound in Qwen3 | High | ✅ Base-model control: 82.0% without LoRA, delta not significant |
| Q35 | Reward function not ablated | High | Acknowledged in limitations as future work |
| Q33 | SFT+GRPO compute matching | High | Partially reframed |
| Q9 | Code result not significant | Medium | ✅ Added Fisher p=0.53 |
| Q47 | No repository link | High | ✅ Anonymous GitHub repo created |
