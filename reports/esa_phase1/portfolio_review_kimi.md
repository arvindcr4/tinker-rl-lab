**A. GO / NO-GO per paper**

- **P1 [RISKY]** — Frontier-scale claims (“∼685B parameters,” “Nemotron-120B”) are implausible for an M.Tech benchmark; author list is inflated for a “solo” deliverable.  
- **P2 [SAFE]** — Descriptive, modest scope; explicit limits (ZVF “not a standalone predictor”) are appropriate.  
- **P3 [FIX-FIRST]** — Contains an impossible >100% accuracy and admits a key G=32 cell was “reconstructed,” not measured.  
- **P4 [SAFE]** — Negative result is carefully scoped; limitations are explicit.  
- **P5 [FIX-FIRST]** — The 17× swing numbers contradict P6; “solo” submission with identical equal-contribution co-authors on all eight papers is suspect.  
- **P6 [FIX-FIRST]** — Same swing story as P5 but with inconsistent values; registry lists suspicious model name “Qwen3.5-4B.”  
- **P7 [RISKY]** — Promotes ZVF to a causal controller, directly against P2’s refusal to over-read the diagnostic; interventional results are “single-task, small-n.”  
- **P8 [RISKY]** — Completely off-topic (credit-card fraud vs. GRPO); uses “Qwen3.5-4B” and a 500-row positive-enriched AUC; breaks portfolio coherence.

**B. INTEGRITY RED FLAGS**

- P3: *“≈ 100.3% … accuracy retention”* — accuracy cannot exceed 100%.  
- P3: *“G=4 versus G=32 token-budget reanalysis (illustrative, reconstructed from ablation logs—no G=32 cell was measured)”* — fabricated/derived data presented as a result.  
- P5: *“final training reward from 5.0% to 84.4%”* vs. P6: *“from 85.6% to 5.0%”* — same exhibit, inconsistent numbers.  
- P6 registry: *“tinker_grpo_qwen3.5-4b_gsm8k.json”*; P8: *“Qwen3.5-4B SFT”* — “Qwen3.5” is a suspicious/nonexistent model name.  
- P1: *“Nemotron-120B”* — verify existence; parameter ceiling also disagrees with the rest of the portfolio (*P1 “∼685B”* vs. P2–P6 *“∼671B”*).  
- P8: *“experiments/results/quick_20260704”* — dated three days before today, undermining reproducibility claims.

**C. CROSS-PAPER CONSISTENCY**

- Parameter ceiling: P1 says ∼685B; P2–P6 say ∼671B.  
- G effect: P7 T2 claims *“larger G lowers ZVF”*; P3 claims *“non-monotone … no universal optimum”* and frames G as preference-density dial — tension.  
- P2 deliberately refuses to promote ZVF causally; P7 builds a theory and controller on it — a direct escalation of an explicitly rejected claim.  
- P5/P6 17× swing numbers disagree.  
- P8 has no ZVF/GRPO connection and sits outside the program narrative.  
- Figure 1 and TinkerRL intro boilerplate are duplicated across papers.

**D. TOP 5 FIXES**

1. **P3: retract or rerun** — remove the impossible 100.3% and the reconstructed G=32 “result”; replace with actual measured cells.  
2. **Reconcile P5/P6 numbers** — decide the true backend-swing values and cite the same underlying run.  
3. **Verify model names** — replace “Qwen3.5” and “Nemotron-120B” with verified, real checkpoints or cite release documentation.  
4. **Resolve P8** — remove it from the ZVF portfolio or reframe it as a short, clearly labeled side-probe; fix the positive-enriched AUC interpretation.  
5. **Add multi-seed replication and artifact hashes** — most claims are single-seed; release exact configs and random seeds to support reproducibility.

**E. PORTFOLIO VERDICT**

**do-not-submit-yet.** The portfolio contains fabricated data, impossible numbers, inconsistent cross-paper statistics, suspicious model names, and an off-topic eighth paper that together would trigger serious review and integrity scrutiny.
