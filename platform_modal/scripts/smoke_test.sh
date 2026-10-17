#!/usr/bin/env bash
# review27-v2: offline, production-code numerical smoke by default.
# No dependency installs or dataset downloads. An ambient key never starts a job.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
LIVE=0
case "${1:-}" in
  '') ;;
  --live) LIVE=1 ;;
  *) echo "Usage: $0 [--live]" >&2; exit 2 ;;
esac
PY="${PYTHON:-python3}"
"$PY" - <<'PY'
from platform_tinker.tinkerrl.grpo import _numeric_equal, normalize_rewards, is_degenerate_group
assert _numeric_equal('42','42')
assert not _numeric_equal('99','42')
assert not _numeric_equal('NaN','NaN')
assert normalize_rewards([1.,1.,1.]) == [0.,0.,0.]
assert is_degenerate_group([0.,0.])
assert abs(sum(normalize_rewards([0.,1.,0.,1.]))) < 1e-10
print('Offline production reward/advantage checks passed')
PY
if [[ "$LIVE" == 1 ]]; then
  [[ -n "${TINKER_API_KEY:-}" ]] || { echo 'Live smoke requires TINKER_API_KEY' >&2; exit 2; }
  # Explicit live request only; normal launcher authorization checks still apply.
  timeout 480 "$PY" platform_tinker/grpo_gsm8k_base.py \
    --model "Qwen/Qwen3.5-4B" --seed 42 --rank 4 --steps 3 --group 2 --batch 1 \
    --tag "smoke_$(date +%s)"
fi
