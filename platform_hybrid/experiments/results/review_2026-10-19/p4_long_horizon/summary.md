# P4 long-horizon GSM8K-CoT (200 steps, cap 1024, Qwen2.5-1.5B-Instruct)

Windows: first/last 20 steps. Held-out: 200 fixed GSM8K items, greedy.

| algo | seed | len cor first->last | len inc first->last | gap first->last | rho(step,len_inc) | at-cap last | pre | post | +/- | steps w/o inc |
|---|---:|---|---|---|---:|---:|---:|---:|---|---:|
| dr_grpo | 123 | 261->247 | 361->319 | +100->+72 | -0.38 | 0.0000 | 0.695 | 0.700 | +24/-23 | 0 |
| dr_grpo | 42 | 252->261 | 358->333 | +106->+72 | -0.07 | 0.0000 | 0.695 | 0.685 | +21/-23 | 0 |
| dr_grpo | 456 | 261->267 | 358->325 | +96->+58 | -0.22 | 0.0000 | 0.695 | 0.685 | +21/-23 | 1 |
| grpo | 123 | 259->244 | 363->330 | +105->+86 | -0.38 | 0.0008 | 0.695 | 0.715 | +26/-22 | 0 |
| grpo | 42 | 253->228 | 362->297 | +109->+69 | -0.42 | 0.0000 | 0.695 | 0.705 | +24/-22 | 0 |
| grpo | 456 | 260->220 | 357->274 | +97->+54 | -0.47 | 0.0000 | 0.695 | 0.685 | +23/-25 | 2 |

dr_grpo: post mean 0.6900, transitions +66/-69 over 3 seeds

grpo: post mean 0.7017, transitions +73/-69 over 3 seeds
