# Flagship policy simulation

This directory contains a deliberately synthetic, reproducible decision
simulation for the frozen flagship protocol. It is **not** a training run and
does not change the preregistration.

Run from the repository root:

```bash
python3 zvf-program/flagship/research/simulation/run_simulation.py \
  --replicates 120 --sensitivity-replicates 48
```

The script uses only Python's standard library and NumPy. It reads the frozen
E1 summary from `zvf-program/colab-experiments/results/e1_grad_signal.json`,
copies the small input subset to `e1_frozen_inputs.json`, and writes all other
generated files in this directory:

- `simulation_results.json`: replicate-level Monte Carlo summaries;
- `policy_regime_summary.csv`: main table, including simulated 95% intervals;
- `sensitivity_summary.csv`: rank stability under explicitly synthetic
  perturbations;
- `run_manifest.json`: seed, configuration, source digest, and command.

The learning dynamics are a transparent stress test: a latent skill determines
binary task success by difficulty; a verifier can flip rewards; observed
within-group contrast produces learning only when it agrees with latent
contrast. Every policy receives the same generated-token ceiling per
scenario. Thus these are conditional comparisons of allocation rules, not
estimates of the protocol's H1--H4 effects.
