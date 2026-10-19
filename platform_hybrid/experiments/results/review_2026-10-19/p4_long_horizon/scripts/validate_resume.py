"""Real-GPU interrupted/resumed smoke against the saved uninterrupted trace."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/content/p4-longhorizon')
validation = root / 'resume-validation'
validation.mkdir(exist_ok=True)
for steps, name in ((1, 'interrupted'), (2, 'resumed')):
    output = validation / name / 'result.json'
    env = {**os.environ, 'WANDB_MODE': 'disabled', 'P4_CHECKPOINT_EVERY': '1'}
    if name == 'resumed':
        env['P4_RESUME_FROM'] = str(validation / 'interrupted/per_run/grpo_s42_checkpoint.pt')
    subprocess.run([
        sys.executable, '-u', str(root / 'run_one.py'), str(root / 'driver.py'),
        '1024', str(output), str(steps), 'grpo', '42', '8',
    ], env=env, check=True)

reference = json.loads((root / 'reference-smoke.json').read_text())['runs'][0]
resumed = json.loads((validation / 'resumed/result.json').read_text())['runs'][0]
expected = {k: v for k, v in reference.items() if k != 'elapsed_seconds'}
observed = {k: v for k, v in resumed.items() if k != 'elapsed_seconds'}
different = [k for k in expected if observed.get(k) != expected[k]]
assert expected == observed, f'Resumed run differs from uninterrupted run: {different}'
report = dict(exact_results_match=True, steps=2, checkpoint_after_step=1,
              n_eval=8, max_new=1024, algorithm='grpo', seed=42)
(validation / 'validation.json').write_text(json.dumps(report, indent=2))
print('RESUME VALIDATION', json.dumps(report), flush=True)
