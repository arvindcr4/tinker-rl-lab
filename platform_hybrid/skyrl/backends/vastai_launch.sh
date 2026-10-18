#!/usr/bin/env bash
set -euo pipefail
exec python -m platform_hybrid.skyrl.backends.vastai_runner "$@"
