# Hard-curriculum: is the curriculum null a ceiling effect? (Tinker, Qwen3.5-4B, 2026-07-06)

Pre-filtered to hard-learnable prompts (base pass-rate 0<p<1, i.e. all mixed-variance; pool=40 from 300 probed), then baseline vs curriculum × 3 seeds. W&B group `hard-curriculum`.

| arm | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| baseline | +0.083 | +0.083 | -0.083 | **+0.028** |
| curriculum | +0.083 | -0.167 | +0.083 | **+0.000** |

## Finding — NOT a ceiling effect; the null is robust
Even on hard-learnable prompts (headroom + all groups mixed), curriculum does NOT beat baseline (+0.000 vs +0.028; curriculum slightly worse, within noise). Curriculum drives zero-variance frac to 0.0 (all steps gradient-bearing) vs baseline 0.1-0.2, yet gains no held-out advantage. So the earlier curriculum null was NOT just because GSM8K is too easy — difficulty-filtering genuinely doesn't help, even in the regime designed to favor it. Note: GSM8K/4B is bimodal (first attempt found only ~13% of prompts in the learnable band), which is itself why there's little for curriculum to exploit.
