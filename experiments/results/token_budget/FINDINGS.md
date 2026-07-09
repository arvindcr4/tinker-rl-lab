# #9 Token-budget curriculum — the "better lever" test (Tinker, Qwen3.5-4B, 2026-07-06)

baseline vs curriculum at a MATCHED 30k rollout-token budget × 3 seeds. W&B group `token-budget`.

| arm | mean held-out gain | optim steps (s0/s1/s2) | groups skipped |
|---|---|---|---|
| baseline (train on everything) | **+0.028** | 12/11/10 | 0 |
| curriculum (mixed-variance only) | **+0.028** | 2/8/5 (all gradient-bearing) | 20/11/14 |

## Finding — the token-budget framing does NOT rescue curriculum
At **equal token cost**, curriculum ties baseline **exactly** (+0.028 vs +0.028). Curriculum spends much of its budget on sampling that it discards (skips 11–20 collapsed groups), doing far fewer optim steps — but every step is gradient-bearing. Net held-out effect: identical to just training on everything. So concentrating compute on gradient-bearing steps buys **nothing** over the naive baseline, even cost-matched. Combined with the campaign (curriculum loses at 5× cost) and P3 (no group-size sweet spot), the "filter/reallocate by difficulty" direction is a **dead end** on this setup. The real lever, if any, is elsewhere.
