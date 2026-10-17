# Same-stack GRPO vs PPO: v1 recompute and v2 analysis

Source of the numbers in thesis §6.5.5 (Tables 6.1 and 6.2), Table 1.1 and §9.3 RQ2.

- `v1_recompute.py` → `v1_recompute.json`: v1 (200-token cap, cold critic, 200 test items, 5 seeds) from
  `platform_hybrid/experiments/results/samestack_gsm8k_cot_full.json`. Reproduces +0.050, paired-t p = 0.016,
  sign-flip p = 0.0625, Holm sign-flip p = 0.19. PPO post 0.195 < base 0.200.
- `v2_analysis.py` → `v2_analysis.json`: v2 (512-token cap, warm-started critic, 1,319 test items) from the
  per-run JSON and stored greedy completions in `platform_hybrid/experiments/results/samestack_gsm8k_v2s/`.
  Re-grades every stored completion against `gsm8k_test_gold.json` (GSM8K test split) with an independent
  boxed-answer parser; 0 disagreements with the stored scores.

Run from the repository root:

```sh
D=reports/project_finish_2026-10-11/validation/samestack_v2
python3 -I $D/v1_recompute.py platform_hybrid/experiments/results > $D/v1_recompute.json
python3 -I $D/v2_analysis.py $D platform_hybrid/experiments/results/samestack_gsm8k_v2s $D/gsm8k_test_gold.json > $D/v2_analysis.json
```

Status at this revision (16 October 2026): GRPO G8 and G2 have 10 seeds, PPO 7 of 10; the exposure-matched
`ppo_8x8` arm is pending. Holm in `v2_analysis.json` is across the three computed contrasts, not the
pre-specified family (which includes G8 − ppo_8x8). When the remaining runs land, pull them into
`samestack_gsm8k_v2s/`, add a `g8-ppo_8x8` entry to the contrast dictionaries in `v2_analysis.py` (lines 97 and 110), rerun the second command, and update Table 6.2, the v2 row of Table 1.1, §9.3 RQ2
(marked `SAMESTACK-V2 NUMBERS` in the chapter sources), slide 5, and the defense documents.

Scripts originate from the 16 October 2026 publication-readiness audit.
