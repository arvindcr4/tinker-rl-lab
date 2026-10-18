# P1_identifiability_note: requested edits (deferred to the concurrent Amp session that owns the file)

I did not edit `platform_hybrid/paper/paper_P1_identifiability_note.tex`. Every number below was recomputed on the released per-step traces by `tools/review_ledger/P1_scaling_rawtrace_refit.py`, with outputs in `platform_hybrid/experiments/results/review_2026-10-18/P1_scaling/`. These edits are optional; the current note is already conservative.

## 589 / 597: Leave-Nemotron and MoE–dense (replace the two "Withdrawn" rows of Table 1)

OLD (row "MoE--dense comparison"):
`Withdrawn: historical split misclassified Nemotron. Neither the original nor the corrected small-panel $p$-value identifies an architecture effect.`

NEW:
`Historical 6-vs-6 split misclassified Nemotron. Recomputed on trace-mean reward, one point per anchor, exact permutation: 7 MoE vs.\ 5 dense gap $+0.188$, one-sided $p = 141/792 = 0.178$; excluding Nemotron (6 vs.\ 5) $+0.278$, one-sided $p = 0.048$ (two-sided $0.093$), but that exclusion is post hoc and mixes 3--5-step console-transcribed probes with full runs; on the seven full runs the gap is $+0.020$ ($p = 0.57$). No architecture effect is claimed.`

OLD (row "Leave-Nemotron analysis"):
`Withdrawn: no trusted sensitivity receipt is established here; earlier claims of robustness or reversal are not retained.`

NEW:
`Recomputed (\texttt{cross\_scale\_regressions.tsv}): OLS slope of trace-mean reward on $\log_{10} N_{\text{tot}}$ is $+0.118 \pm 0.118$ per decade on the 11 non-Nemotron anchors (permutation $p = 0.36$) and $+0.073 \pm 0.102$ on the 6 non-Nemotron full runs ($p = 0.62$). Excluding the collapse does not reveal a detectable trend; the intervals remain too wide to support flatness.`

## 588 / 590: Panel specification (add after the panel list in Section 2)

`Seven anchors are full runs (20--30 logged steps); the five short probes (Qwen3-32B, Qwen3.5-27B, Qwen3-30B-MoE, Qwen3-30B-MoE-Inst, Qwen3-235B-MoE) are interrupted runs whose 3--5 rewards were transcribed from console output (\texttt{run\_id = partial-from-console}). Per W\&B run configs, 4B--32B runs use 8 completions $\times$ 2 prompts per step, LoRA rank 32, learning rate $3\times10^{-5}$; frontier runs use $4 \times 2$, rank 16, $10^{-5}$. Step-1 reward $R(1) \ge 0.875$ on Qwen3.5-4B, Llama-3.1-8B-Instruct, DeepSeek-V3.1 and Kimi-K2, i.e.\ these checkpoints start near the GSM8K ceiling, which by itself prevents identifying a saturation rate.`

Provenance flag for the note: the DeepSeek-V3.1 and Nemotron-120B trace JSONs carry `run_id`s that W&B attributes to gpt-oss-20B runs (56b99b24…, 657a920a…). The traces themselves differ, so the run_id field is mislabelled.

## 598: References

The note has no bibliography. Suggested minimal set, all already in `references.bib` (keys as used in P1): `shao2024deepseekmath` (GRPO), `cobbe2021training` (GSM8K), `hoffmann2022chinchilla`, `burnham2002model` (AICc), `nimmaturi2025predictive`, `tan2025scalingrl` (Tan et al., arXiv:2509.25300). Add `\bibliographystyle{plainnat}\bibliography{references}` and cite them where the note mentions GRPO, GSM8K, AICc and "related public scaling-law work". All six keys were checked to exist in `platform_hybrid/paper/references.bib`.
