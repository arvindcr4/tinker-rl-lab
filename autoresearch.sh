#!/usr/bin/env bash
set -euo pipefail
python reports/final/evaluate_gsm8k_test.py \
  --use_tinker \
  --model_name Qwen/Qwen3-8B \
  --limit 40 \
  --output reports/final/gsm8k_base_smoke_results.json
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/final/gsm8k_base_smoke_results.json')
results = json.loads(p.read_text())
summary = results.get('summary', {})
print(f"METRIC heldout_examples_scored={summary.get('attempted', 0)}")
print(f"METRIC heldout_accuracy_pct={100.0 * summary.get('accuracy', 0.0):.2f}")
PY
