# Adversarial review (codex exec / gpt-5.5) — findings + fixes applied

Run 2026-07-03, read-only sandbox, hostile-auditor prompt over the repo (verified claims against experiments/results/).

## Fixes applied (all four papers rebuilt: 0 errors, 0 undefined citations)
- **[Critical] P1 phase table** (scaling_law_iter65.tex): table listed 5 classes summing to 5 and caption assigned Nemotron->collapse / Llama,Kimi,27B->drift, contradicting the TSV + the section's own self-check. Rebuilt from scaling_law_iter65_phase_pieces.tsv: valley 6, three-phase 2, constant 2, anomalous 1, monotonic-rising 1 (sum 12); caption now states no collapse/drift class and Nemotron=valley.
- **[Critical] P4 'base checkpoint'** (p4_intro.tex): the p=0.256 held-out result is the Qwen3-8B-**Instruct** checkpoint's pre-RL control; Base has no matched held-out measurement. Reworded to 'Qwen3-8B-Instruct ... same checkpoint's pre-RL held-out accuracy (a matched Base control is not available)'.
- **[Critical] liu2026gdpo citation**: entry (arXiv:2602.01987, 'Group Decoupled Preference Optimization', Yixin Liu/Hang Zhao) was fabricated. Replaced with the real GDPO: arXiv:2601.05242, 'Group reward-Decoupled Normalization Policy Optimization', Shih-Yang Liu et al. (NVIDIA).
- **[Major] '5/5 lambda pinned' -> 4/5** (scaling_law_iter29.tex, p1_abstract, p1_results_intro): iter25_identifiability.tsv shows Nemotron lambda=0.99, at_bound=False. Corrected to 4/5 with Nemotron as the exception (reinforces the distinct-phase point). (scaling_laws.tex already said 4/5.)
- **[Major] Overclaims** (group_size.tex): 3x 'decisively' softened to benchmark-scoped language; the CI-based statistical statement retained.
- **[Major] Undefined refs** (_shared_methods.tex): neutralized ef{sec:framework_gap}, ef{app:stat-rigor-addendum} (x2), ef{app:compute} to plain descriptive text.
- **[Major] Model roster** (_shared_methods.tex): '32+ models' -> 'targets a roster of 32+ configurations, not all completed'; Qwen3.5-397B-A17B relabeled a partial run whose held-out eval did not complete.
- **[Minor] Frontier-synthesis headings**: 5 assertive headings ('The ... Theorem', 'is the true scaling axis', 'phase-gated law') reworded as 'A proposed ...'.
- **[Minor] nimmaturi2025scalinglaws**: fabricated duplicate (arXiv:2511.00213) of the real nimmaturi2025predictive (arXiv:2507.18014). Citations redirected; fake entry removed.

## codex verified as SUPPORTED-in-data (unchanged)
ZVF/outcome Spearman rho=0.2687; Nemotron ZVF 0.55 vs 0.0/0.067; G-retention 100.34%; SNR 2.16/4.13=52%; 2/12 three-phase. The quantitative backbone checks out; the fixes were stale/overstated *framing* and 2 bad citations.
