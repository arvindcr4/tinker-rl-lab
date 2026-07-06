Act as a senior ML research scientist doing a novelty/landscape survey. Survey the 2025–2026 literature (NeurIPS 2025, ICLR 2026, ICML 2025, COLM 2025, and recent arXiv) on RL post-training of LLMs with GRPO / RLVR (RL with verifiable rewards). I am writing 8 papers and need each to be genuinely ORIGINAL — not a reinvention.

For EACH of the 8 subtopics below, report: (a) the 4–6 most relevant recent papers (title, authors, venue, month/year, one-line contribution); (b) what is now established/known; (c) the specific OPEN GAP or under-explored angle a new paper could originally own; (d) what would make a new contribution NON-novel (already covered) so I can avoid it. Prioritise 2025–2026 work; be concrete with citations.

Subtopics:
1. Scaling laws / training dynamics of GRPO reward curves vs model size; whether parametric saturation laws are identifiable and predictive out-of-sample.
2. Zero-variance / zero-advantage / homogeneous groups in GRPO under binary verifiable rewards — diagnostics and mitigations beyond AERO and NGRPO.
3. Group size (samples-per-prompt, G) in GRPO and its compute-normalised effect on learning.
4. Length bias in GRPO and corrections (Dr.GRPO, DAPO, length-normalised advantages).
5. Reporting standards / documentation for RL post-training runs beyond Model Cards, Datasheets, ML reproducibility checklists — is there a GRPO-specific reporting standard?
6. Machine-readable registries / catalogs of RL training stacks / experiment metadata beyond W&B/MLflow — schema-level cataloguing of RL post-training configurations.
7. Closed-loop controllers that adapt GRPO rollout allocation / group size / sampling based on live variance or zero-advantage signals.
8. Anomaly / integrity / fraud detection for RL training runs themselves (reward hacking, metric spoofing, backend swaps, telemetry manipulation) — is there a labelled benchmark?

End with a ranked list of which of the 8 has the LARGEST open originality opportunity versus the most crowded / hardest-to-be-novel, with a one-sentence justification each.
