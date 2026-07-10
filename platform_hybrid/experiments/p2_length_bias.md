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

The reward parser *favours shorter completions* -- rewarded responses are
170.8 tokens shorter on average (CI [-183.7, -157.5]).

Panel A of the figure shows the bimodal pattern: failed completions often
run long (the model writes reasoning that never boxes an answer), while
rewarded completions have a narrower length distribution. Panel B confirms
the median difference. Panel C plots P(reward=1) against length bin: the
curve rises then falls, consistent with a verbosity tax above some
length at which the model stops producing valid boxed answers. Panel D
shows the effect persists across all 10 checkpoints.

Figure: `paper/figures/v2/p2_length_bias.png`.
