# P1-B: Live-probe validation of the pre-GRPO audit table

Date: 2026-04-22. Tinker base-policy sampling with no training step.
30 prompts per config (25 for MATH-500), rank-4 LoRA heads instantiated at
init (sampler uses the zero-LoRA initial weights).

## Measured vs audit-table estimates

| Row | Task | Model | G | Est. $p_x$ | Meas. $p_x$ | Est. ZVF | Meas. ZVF | Audit verdict | Meas. verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | GSM8K | Llama-3.1-8B (base) | 32 | 0.05 | **0.121** | 0.95 | **0.133** | Dead: cold-start | Alive: reasonable signal |
| 2 | GSM8K | Llama-3.1-8B-Instruct | 32 | 0.45 | **0.758** | 0.15 | **0.267** | Alive: prior works | Alive, partially saturating |
| 3 | GSM8K | Qwen3-8B | 16 | 0.35 | **0.060** | 0.10 | **0.733** | Alive: healthy gradient | Near-dead: p_x 5.8× lower than est. |
| 4 | MATH-500 | Qwen3-8B | 32 | 0.15 | **0.001** | 0.20 | **0.960** | Alive-but-hard | Dead: sampler never hits |
| 5 | GSM8K (xref) | Qwen3.5-4B | 16 | n/a | 0.194 | n/a | 0.500 | n/a | Mid-range |
| 6 | Arithmetic | Llama-3.2-1B | 16 | 0.95 | **0.015** | 0.60 | **0.800** | Saturating: already easy | Near-dead: base can't do 3-digit add |

## Headline findings

1. **The audit table's $p_x$ estimates are heuristic, not measurements.** Four
   of four directly comparable rows have measured $p_x$ that differs from the
   estimate by at least 2×. Two rows flip the verdict (Llama-8B Base goes from
   "dead" to "alive"; Qwen3-8B G=16 goes from "alive" to "near-dead").
2. **MATH-500 Qwen3-8B base is completely dead at $p_x=0.001$, ZVF=0.960.**
   This is consistent with the observed "MATH training reward near 0 without
   warm-up" in the main campaign, and resolves the audit table's optimistic
   "alive but hard" verdict.
3. **Qwen3-8B GSM8K base with G=16 is near-dead ($p_x=0.06$, usable=0.23,
   ZVF=0.73).** The audit table over-estimated the base hit rate; the G=32
   version of the paper's structural-ceiling campaign may partially rescue this
   via a larger $G$, consistent with $1-(1-0.06)^{32}\approx 0.86$ usable at
   G=32. The paper should clarify that the Qwen3-8B campaign runs are only
   alive because $G\in\{16, 32\}$ was chosen large enough to rescue the low
   base hit rate, not because the base hit rate was high.

## Implication for the thesis

The audit table is still useful as a qualitative framework — predicted usable
and observed ZVF track each other directionally — but the specific $p_x$
numbers in the original table should be flagged as estimates rather than
measurements. A revised table with measured columns is proposed in
Chapter~5; the data live in `experiments/results/p1b_*.{json,csv}`.
