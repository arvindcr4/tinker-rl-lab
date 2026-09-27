#!/bin/sh
# --swebench-bin for zvf-program/e1_wave10/driver.py: same argv, native CLI runs in a Modal dockerd sandbox.
exec /Users/arvind/.local/share/uv/tools/modal/bin/python "$(dirname "$0")/remote_swebench.py" "$@"
