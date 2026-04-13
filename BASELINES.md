# TinkerRL Baselines: Floor and Ceiling Reference

Establishing performance bounds is critical for meaningful benchmark evaluation
(Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021).

## Arithmetic Task (Addition, max_num=99)

| Baseline | Accuracy | Notes |
|---|---|---|
| Random (uniform) | 0.50% | 1/(2×99+1) = 1/199 ≈ 0.50% |
| Random (range-aware) | ~1.0% | Sampling from [2, 198] |
| Constant prediction | ~1.0% | Always predicting the mean (100) |
| Nearest-neighbor | ~5% | Memorize and match closest seen pair |
| Human (college-level) | ~99.5% | Near-perfect on simple addition |
| Oracle | 100.0% | Direct computation of num1 + num2 |

### Floor Definition
**Floor = Random uniform baseline (0.50%)**. Any trained agent must exceed this to demonstrate learning.

### Ceiling Definition  
**Ceiling = Oracle (100.0%)**. This is the maximum achievable accuracy.

### Human Baseline Protocol
- Task: 200 two-digit addition problems (e.g., "47 + 83 = ?")
- Conditions: No calculator, 5-second time limit per problem
- Participants: 10 college-level adults
- Expected accuracy: 99.0–99.8% (errors from carelessness, not inability)
- Source: Estimated from cognitive arithmetic literature (Ashcraft, 1992)

## GSM8K Task (Grade School Math)

| Baseline | Accuracy | Notes |
|---|---|---|
| Random | ~0% | Vanishingly small probability of correct free-form answer |
| Few-shot GPT-2 (124M) | ~2% | Small LM baseline |
| Few-shot GPT-3 (175B) | ~35% | Large LM without fine-tuning |
| Chain-of-thought GPT-4 | ~92% | Strong LM ceiling reference |
| Human (college-level) | ~95% | With time to work through problems |

### Floor: Random / GPT-2 few-shot (~0–2%)
### Ceiling: Human performance (~95%)

## Distillation Task (Response Shortening)

| Baseline | Accuracy | Notes |
|---|---|---|
| No distillation | 0% compression | Original response length |
| Truncation | Variable | Naive first-N-tokens, quality degrades |
| Extractive summary | ~40% compression | Select key sentences |
| Human editor | ~50% compression | Professional editing |

## Statistical Methodology

All results should be reported with:
- **5 independent seeds** (42, 123, 456, 789, 1024)
- **Mean ± standard error** across seeds
- **95% confidence intervals** via bootstrap (n=2000)
- **Interquartile mean (IQM)** as recommended by Agarwal et al. (2021)

### Recommended Statistical Tests
- Mann-Whitney U test for pairwise comparisons
- Bootstrap confidence intervals for aggregate metrics
- Performance profiles for algorithm comparison

## How to Run Baselines

```bash
# Random baseline (built into each experiment)
python experiments/implementations/sb3_ppo_math.py --seed 42

# Run across multiple seeds
bash scripts/run_seeds.sh experiments/implementations/sb3_ppo_math.py
```

## References

- Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., & Bellemare, M.G. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning that Matters. AAAI.
- Ashcraft, M.H. (1992). Cognitive arithmetic: A review of data and theory. Cognition.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. arXiv.
