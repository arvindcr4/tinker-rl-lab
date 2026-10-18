# Platform editorial requests (not applied)

No bibliography, shared section, or TikZ source was edited. These are exact
old → new requests for the coordinator, not resolved findings.

## P12, findings 629–630: protect bibliography capitalization

File: `platform_hybrid/paper/unified_signal_starvation/references.bib`.
In the following `title` fields, replace exactly the old text with the new text.
The current compiled bibliography lowercases these proper names/acronyms.

| Key | Old | New |
| --- | --- | --- |
| ahmadian2024back | `Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs` | `Back to Basics: Revisiting {REINFORCE} Style Optimization for Learning from Human Feedback in {LLMs}` |
| che2026single | `Learning with a Single Rollout via Monte Carlo Pass@k Critic` | `Learning with a Single Rollout via {Monte Carlo} Pass@k Critic` |
| deepseekai2025r1 | `DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning` | `{DeepSeek-R1}: Incentivizing Reasoning Capability in {LLMs} via Reinforcement Learning` |
| espeholt2018impala | `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures` | `{IMPALA}: Scalable Distributed {Deep-RL} with Importance Weighted Actor-Learner Architectures` |
| gong2026basis | `BASIS: Batchwise Advantage Estimation from Single-Rollout Information Sharing for LLM Reasoning` | `{BASIS}: Batchwise Advantage Estimation from Single-Rollout Information Sharing for {LLM} Reasoning` |
| shao2024deepseekmath | `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models` | `{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models` |
| song2026staleness | `Staleness-Learning Rate Scaling Laws for Asynchronous RLHF` | `Staleness-Learning Rate Scaling Laws for Asynchronous {RLHF}` |
| glm2026glm52 | `GLM-5.2: Built for Long-Horizon Tasks` | `{GLM-5.2}: Built for Long-Horizon Tasks` |
| zong2026cero | `Cross-Epoch Adaptive Rollout Optimization for RL Post-Training` | `Cross-Epoch Adaptive Rollout Optimization for {RL} Post-Training` |

## P12, finding 627: diagram naming and internal caption code

File: `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex`.
Deferred together because the node text is inside TikZ, which this pass may not edit.

- Old: `\textbf{TRIAGE-RL Action Router}`
  → New: `\textbf{TriageRL Action Router}`.
- Old: `Signal Starvation Diagnostic and TRIAGE-RL Action Router Pipeline (N01).`
  → New: `Signal Starvation Diagnostic and TriageRL Action Router Pipeline.`
- Old: `Token-Level Credit Factorization and Zero-Update Certificate (N01).`
  → New: `Token-Level Credit Factorization and Zero-Update Certificate.`

## P12, finding 636: remaining diagram summation indices

Same file, Figure 2 TikZ only. Prose equations now explicitly index the batch and
call GUN a squared norm; these diagram changes remain unapplied:

- Old: `\frac{1}{|\mathcal{B}|}\sum (\rho_t A_t)^2`
  → New: `\frac{1}{|\mathcal{B}|}\sum_{t\in\mathcal{B}} (\rho_t A_t)^2`.
- Old: `\frac{\sum m_t(\rho_t A_t)^2}{\sum (\rho_t A_t)^2}`
  → New: `\frac{\sum_{t\in\mathcal{B}} m_t(\rho_t A_t)^2}{\sum_{t\in\mathcal{B}} (\rho_t A_t)^2}`.

Finding 638 (overlapping routing arrows) requires visual/layout work, not a
verified exact clerical replacement, and remains out of scope.
