# P1-A: completion-only mask diagnostic

Model: `Qwen/Qwen3.5-4B`, rank 4, seed 20260422.
Two sequences: short prompt + long prompt, same response `\boxed{12}`.

## Sequence shapes

| seq | prompt_len | resp_len | total_len |
|---|---:|---:|---:|
| short | 11 | 15 | 26 |
| long  | 42 | 12 | 54 |

## Loss comparison (same data, three mask formulations)

| formulation | mean_loss |
|---|---:|
| full_sequence_sum (current runner) | 94.3470687866211 |
| completion_only_sum (canonical) | 16.432968139648438 |
| completion_only_mean (length-norm) | 1.2180172204971313 |

## Prompt-gradient attribution

| seq | full loss | completion loss | prompt contribution | prompt/full |
|---|---:|---:|---:|---:|
| 0 | 47.2817 | 18.1676 | +29.1141 | +0.616 |
| 1 | 141.4124 | 14.6983 | +126.7141 | +0.896 |

## Verdict: **LEAK**

Average |prompt fraction of full-sequence loss|: 0.756. Threshold for CLEAN: < 0.05.

Current runner uses full_sequence_sum. If the prompt fraction is
non-trivial (>5%), the recorded GRPO loss is contaminated by
gradient on prompt tokens, which canonical GRPO does not do.

