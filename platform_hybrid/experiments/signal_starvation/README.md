# PPO/SAO signal-starvation instrumentation

This directory makes the diagnostic portion of the signal-starvation paper
executable without implying that its confirmatory experiments have run.

- `metrics.py` implements PPO's sign-dependent gate, SAO's strict two-sided
  DIS gate, and PAM/GSR/EGM/root-ZUF aggregation.
- `analyze_trace.py` validates token-level gates and produces root-macro JSON.
- `preregistration.json` freezes H1-H4, arms, budgets, and endpoints.
- `test_metrics.py` checks boundary semantics, exact-zero behavior, and
  invariance to a lossless chunk repartition.

Run the deterministic checks:

```bash
python3 -m unittest platform_hybrid/experiments/signal_starvation/test_metrics.py
```

Analyze a token JSONL trace:

```bash
python3 platform_hybrid/experiments/signal_starvation/analyze_trace.py trace.jsonl \
  --algorithm ppo --epsilon 0.2 --output trace.metrics.json
```

Required token fields are `root_trajectory_id`, `ratio`, and `advantage`.
Optional `chunk_id` is ignored during root aggregation; `is_action_token=false`
excludes observation tokens. If a trace contains `gate`, the analyzer rejects
any value inconsistent with the selected algorithm's formula.

No SAO or GLM-5.2 training outcome is included. Those require the frozen
matched-budget experiment and suitable accelerator access.
