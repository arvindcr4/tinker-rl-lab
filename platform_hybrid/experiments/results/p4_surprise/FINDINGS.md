# #10 P4 length-bias / surprise-weighted loss (Tinker, Qwen3.5-4B, 2026-07-06)

sum (standard GRPO) vs mean (Dr.GRPO length-norm) vs surprise-weighted loss × 2 seeds. W&B group `p4-surprise`.

| loss | mean held-out gain | mean completion-len delta |
|---|---|---|
| sum (standard, "length-biased") | +0.125 | -16 |
| mean (length-normalized) | +0.000 | -8 |
| surprise-weighted | +0.042 | -2 |

## Finding — the length-bias "fixes" do not help here (underpowered)
On this GSM8K/4B setup, standard **sum** was (if anything) best on held-out and completions did **not** inflate under it (len went down, -16) — so there was little length bias to correct. The proposed fixes (mean-norm, surprise-weighting) did **not** beat sum. n=2 seeds, held-out=12, so all differences are within noise → treat as a null, not "sum wins". Consistent with the portfolio: no proposed GRPO tweak robustly beats baseline at this scale.
