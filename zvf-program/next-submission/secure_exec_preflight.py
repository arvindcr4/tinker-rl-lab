#!/usr/bin/env python3
"""Execute the uploaded preflight without exposing credentials in argv/logs."""

import json
import os
from pathlib import Path
import runpy
import sys


SECRET_PATH = Path("/content/.next-preflight-secrets.json")
REQUEST_PATH = Path("/content/next-preflight-request.json")
SCRIPT_PATH = Path("/content/remote_preflight.py")


def read_and_remove(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    path.unlink()
    return payload


secrets = read_and_remove(SECRET_PATH)
request = read_and_remove(REQUEST_PATH)
for key in ("HF_TOKEN", "WANDB_API_KEY"):
    value = secrets.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"missing required secret: {key}")
    os.environ[key] = value

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_NOTEBOOK_NAME"] = "remote_preflight.py"
sys.path.insert(0, "/content")
sys.argv = [str(SCRIPT_PATH), *request["script_args"]]
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
