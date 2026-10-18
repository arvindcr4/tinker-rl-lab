> INTERPRETATION CORRECTION (review 43): Length/reward association does not identify parser bias. Failures at the 512-token cap may reflect truncation. Report truncation, parser failures and completed-but-incorrect responses separately before making mechanism claims.

# P2: Length-bias diagnostic (post-hoc from held-out GSM8K)

Source: `experiments/results/heldout_gsm8k.json` — 10 Qwen3-8B checkpoints evaluated on
500 held-out GSM8K problems each (5000 rollouts total, not the same
slice used for the original held-out-accuracy claim).

## Aggregate

| outcome | n | mean length | std | median | Q25 | Q75 |
|---|---:|---:|---:|---:|---:|---:|
| rewarded (r=1) | 4604 | 237.7 | 103.7 | 219 | 162 | 296 |
| failed (r=0)   | 396 | 408.5 | 131.7 | 512 | 266 | 512 |

**Δ (mean_pos − mean_neg) = -170.8 tokens**, 95% paired-bootstrap CI [-183.7, -157.5] over 5000 rollouts.

## Reading

Rewarded responses are 170.8 tokens shorter on average in this historical summary (reported CI [-183.7, -157.5]). This is a length/reward association; the parser mechanism remains unresolved.

The failed-response median at the 512-token cap is compatible with truncation as well as parsing or completion errors. Panels A and B describe length distributions, Panel C describes reward rate by length bin, and Panel D stratifies by checkpoint. None identifies a verbosity penalty. Separate truncation, parser failure, and completed-but-incorrect measurements are required before assigning a mechanism.

Figure: `paper/figures/v2/p2_length_bias.png`.
