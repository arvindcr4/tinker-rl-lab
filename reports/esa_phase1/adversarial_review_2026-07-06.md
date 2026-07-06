# Adversarial Review — Unified GRPO Benchmark flagship (2026-07-06)

**Method:** flattened flagship `paper/main.tex` + 116 inlined sections (~318k tokens) submitted to two frontier models via the CloakBrowser chat sessions, each prompted as a hostile NeurIPS/ICLR area chair + M.Tech thesis examiner told to *refute*, not praise.

- **GPT‑5.5 Pro** (Pro Extended) → 18 ranked objections. Raw: `review_gpt55pro_raw.md`.
- **Gemini 3 Pro** (Extended; Deep Think errored on the 318k-token file) → 6 fatal objections. Raw: `review_gemini_raw.md`.

Both models ran **independently** and **converge** on the same core problems → high confidence these are real, not model artifacts.

---

## CONSENSUS FATAL FINDINGS (both models)

1. **P5–P8 are absent from the flagship manuscript.** `main.tex` ends its pillars after P4; no MIN-REPORT schema validation, no GRPO-Registry methodology, no ZVF-controller architecture, no fraud/anomaly section. → *Partly an artifact of what was sent* (P5–P8 live in separate `paper_P5..P8.tex` wrappers, not `\input` into `main.tex`), **but** it means the flagship overclaims to "subsume 8 pillars." Fix: either fold real P5–P8 sections in, or scope the flagship's claims to P1–P4.
2. **"Framework Gap" is confounded.** Tinker run uses Qwen3‑8B‑**Base**, TRL/verl/OpenRLHF use Qwen3‑8B‑**Instruct**; and verl/OpenRLHF cells are **dry-run placeholders**, not measured. The "17× gap" is uninterpretable.
3. **n=1 single-seed → zero statistical power**, yet scaling-law fits and Benjamini‑Hochberg p-values were computed on them (Welch t-tests with undefined variance). The paper's own addendum concedes this.
4. **Cross-library baseline is LLM-vs-MLP.** TRL/Tinker (autoregressive LLMs) vs SB3/CleanRL/Tianshou (small MLPs on an arithmetic MDP). Cohen's d=14.59 is an architecture-mismatch artifact, not a library comparison.
5. **Generalization failure, self-reported.** Training-reward vs held-out accuracy Spearman ρ=**−0.02**; post-GRPO vs pre-RL held-out 83.3% vs 82.0%, **p=0.26** (non-significant). The benchmark optimizes a metric decoupled from held-out reasoning.
6. **ZVF is (near) trivial.** Mechanically coupled to reward sparsity, group size, and accuracy — a property of binomial variance under binary rewards; not shown to add predictive power beyond reward/entropy/difficulty. Prior work (AERO, NGRPO, Dr.GRPO) already targets zero-advantage groups.

## ADDITIONAL (GPT‑5.5 Pro)

7. **Pseudoreplication** — treats training *steps* as independent samples; the paper elsewhere reports lag‑1 autocorr ≈0.9. Unit of replication must be the seed/run.
8. **Scaling "law" unidentifiable** — λ hits the optimizer bound in 4/5 runs; 70/30 holdout gives zero improvement over a constant-mean predictor.
9. **Novelty overstated** — GRPO (DeepSeekMath/R1), Dr.GRPO length-bias fix, AERO/NGRPO zero-advantage all predate this.
10. **Count inconsistencies** — "70+ runs / 7 libraries" vs 32 models / 42 / 44 experiments / N=15 ZVF / N=80 synthetic — "p-hacking-by-denominator." Needs one master run registry.
11. **Base-vs-instruct claim already retracted** by the paper's own audit (reporting bug, missing sources).
12. **Variance-mitigation table (AERO/CPPO/NGRPO/Scaf-GRPO) is synthetic projection** written like measured results.
13. **Provenance/LLM-use disclosures contradict each other** across sections (one says LLMs not used; others say frontier-model reasoning was distilled in).
14. **Not submission-clean** — author/GitHub/PES identifiers still visible despite "anonymize pending" comments.

## The two "defense-sinker" questions (near-identical across models)

- *GPT:* After applying your own Tier‑A rule (≥5 seeds, ≥100 steps, matched open-stack impl, traceable logs, held-out eval), which single central GRPO-scaling claim remains statistically supported — other than the admitted cross-paradigm TRL-LLM vs classic-RL-MLP mismatch?
- *Gemini:* Base-vs-Instruct framework comparison, LLM-vs-MLP baselines, n=1 scaling, p=0.26 generalization, and P5–P8 missing — what is the scientific contribution beyond a rigorous measurement of your own confounding variables?

---

## Implication for the Phase‑1 ESA (11/12 July — ~5 days)

The honest, defensible ESA framing is **not** "we built a leaderboard that shows GRPO wins." It is: *a rigorous measurement/diagnostic study showing how much end-to-end stack choice dominates GRPO outcomes, an honest ZVF/GU logging diagnostic, and candid negative results (RL gains are within noise on held-out; training reward doesn't predict held-out).* That reframing turns every finding above from an attack into a pre-empted talking point on the "Suggestions from Review‑3" and "Project Progress" slides.

**Do NOT** present P5–P8 as validated in the flagship, single-seed frontier runs as ranked results, the framework gap as a clean comparison, or the LLM-vs-MLP row as a library comparison — the panel (two Great Learning examiners) will sink it with the questions above.

---

# ADDENDUM — P5–P8 re-review (both models read the actual pillar papers)

Bundle: `paper_P5..P8.tex` flattened (~358k tokens). Raw: `review_P5P8_gemini_raw.md`, `review_P5P8_gpt55pro_raw.md`.

| Pillar | Gemini | GPT‑5.5 Pro | Why |
|---|---|---|---|
| **P5 MIN-REPORT** | THIN | THIN | Reporting spec, not a released/validated standard. Own audit: 12 structured subfields score **0/98** ("satisfiable as a shallow key set"). Novelty pressure vs Model Cards, Datasheets, W&B/MLflow. |
| **P6 GRPO-Registry** | VAPORWARE | THIN | JSON catalog that mostly passes its own schema. **Entry count contradicts itself: 12 / 15 / 20 / 31 / 35** across sections. Badge scores *reportability*, not correctness; 9 red-flag >50%-null leaves; DAPO/GSPO/PPO deltas "CLAIM-ONLY". |
| **P7 ZVF-controller** | VAPORWARE | THIN | Adaptive-G only **ties** Dr.GRPO on held-out (+0.575) using more rollouts; deltas within noise. "Closed-loop" evidence is a forward-sim on frozen step-0 tensors. Counterfactual saves contradict: 0 / 466.75 / 0 / 159-of-160. |
| **P8 Fraud/anomaly** | VAPORWARE | VAPORWARE | **Not RL-run anomaly detection at all** — it's synthetic credit-card fraud (sklearn make_classification), XGBoost vs LLM. Headline AUC self-contradicts (0.7955 vs 0.9988). |

**Verdict:** with P5–P8 fairly read, none clears a "SOUND" bar. They are engineering artifacts / prototypes / a mis-scoped side-probe — not validated research pillars. This confirms the **honest diagnostic + engineering-artifacts framing** for the ESA. P8 in particular must be either re-scoped to real RL-run telemetry or dropped from the "8 validated pillars" claim before any defense.
