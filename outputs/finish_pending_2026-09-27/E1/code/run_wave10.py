"""Thin launcher for zvf-program/e1_wave10/driver.py --execute (unchanged driver code).

Two adaptations, both recorded in wave10/adaptations.json:
  1. The shared actor serves max_model_len=32768 (the 2026-09-12 sessions served 65536). vLLM rejects
     prompt+max_tokens > max_model_len, so each request's max_tokens is capped at
     min(8192, 32768 - prompt_tokens) (prompt counted by the server's /tokenize). All other params
     (temperature 0, seed 809, top_p 0.95, enable_thinking false, prompt bytes) are the driver's.
  2. --swebench-bin points at code/swebench_remote.sh (same native CLI, in a Modal dockerd sandbox).
Credentials come from the environment (never argv / never written).
"""
import copy
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
E1 = REPO / "outputs/finish_pending_2026-09-27/E1"
sys.path.insert(0, str(REPO / "zvf-program/e1_wave10"))
sys.path.insert(0, str(REPO / "zvf-program/flagship"))
import driver as D  # noqa: E402
import public_swe_multilingual_native as psml  # noqa: E402

BASE = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/")
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
MAX_MODEL_LEN = 32768
ADAPT = E1 / "wave10/adaptations.json"
caps = {}


def prompt_tokens(request: dict) -> int:
    body = {"model": request["model"], "messages": request["messages"],
            "chat_template_kwargs": request["chat_template_kwargs"], "add_generation_prompt": True}
    r = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["count"]


_orig = D.generate_one


def generate_one(iid, source, identity, args, attempts_root, log):
    actor = {k: source["actor_task"][k] for k in psml.ACTOR_FIELDS}
    req = psml.build_actor_request(actor, source["files"], identity["served_model_id"],
                                   max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)
    n = prompt_tokens(req)
    a = copy.copy(args)
    a.max_tokens = max(1, min(args.max_tokens, MAX_MODEL_LEN - n))
    caps[iid] = {"prompt_tokens": n, "max_tokens": a.max_tokens}
    ADAPT.write_text(json.dumps({"max_model_len": MAX_MODEL_LEN, "per_task": caps}, indent=2))
    return _orig(iid, source, identity, a, attempts_root, log)


D.generate_one = generate_one

if __name__ == "__main__":
    wandb = json.load(open(E1 / "wandb_run.json"))["id"]
    sys.argv = ["driver.py", "--execute",
                "--endpoint", BASE + "/v1/chat/completions", "--api-key", KEY,
                "--attempts-root", str(E1 / "wave10/attempts"),
                "--deployment", str(E1 / "actor_deployment_0927.json"),
                "--output-dir", str(E1 / "wave10/run"),
                "--run-id", "e1multilingual0927wave10",
                "--wandb-run-id", wandb,
                "--request-timeout", "900",
                "--swebench-bin", str(E1 / "code/swebench_remote.sh")]
    sys.exit(D.main())
