#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
"$ROOT/offline/run.sh"
open "$ROOT/offline/output/dashboard.html"

echo
echo "Offline verification passed. The dashboard is open in your browser."
