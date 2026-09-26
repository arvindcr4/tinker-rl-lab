#!/usr/bin/env python3
"""E13 small-scale BALROG run: native Evaluator.run_episode + NaiveAgent + native vllm (OpenAI) client
pointed at the local Tinker shim. Items fixed by seed 20260926. One process per env (sequential inside).

Usage (cwd must contain tw_games/):
  python run_e13.py --plan            # write raw/plan.json (item selection)
  python run_e13.py --env babyai      # run that env's selected episodes
  python run_e13.py --summarize       # native collect_and_summarize_results over raw/
"""
import argparse, json, os, random, sys, time, traceback
from pathlib import Path

from omegaconf import OmegaConf

SRC = Path(os.environ["BALROG_SRC"])
OUT = Path(os.environ["E13_OUT"])
SEED = 20260926
PLAN = OUT / "plan.json"
# per-env: (tasks to sample, episodes per task, step cap or None=native)
SPEC = {"babyai": (5, 2, None), "textworld": (3, 2, None), "babaisai": (4, 1, None),
        "minihack": (4, 1, None), "crafter": (1, 2, 150)}


def config(env):
    cfg = OmegaConf.load(SRC / "balrog/config/config.yaml")
    cfg.client.client_name = "vllm"
    cfg.client.model_id = "Qwen/Qwen3.6-35B-A3B"
    cfg.client.base_url = os.environ.get("SHIM_URL", "http://127.0.0.1:8773/v1")
    cfg.client.generate_kwargs.temperature = 0.0
    cfg.client.generate_kwargs.max_tokens = int(os.environ.get("E13_MAX_TOKENS", "128"))
    cfg.eval.num_workers = 1
    cfg.eval.max_steps_per_episode = SPEC[env][2]
    cfg.envs.names = env
    cfg.envs.textworld_kwargs.textworld_games_path = os.path.abspath("tw_games")  # native assets, absolute path
    return cfg


def plan():
    cfg = OmegaConf.load(SRC / "balrog/config/config.yaml")
    rng = random.Random(SEED)
    items = []
    for env, (k, n_ep, cap) in SPEC.items():
        tasks = list(cfg.tasks[f"{env}_tasks"])
        chosen = tasks if k >= len(tasks) else [tasks[i] for i in sorted(rng.sample(range(len(tasks)), k))]
        for t in chosen:
            for ep in range(n_ep):
                items.append({"env": env, "task": t, "episode_idx": ep, "env_seed": SEED + ep,
                              "step_cap": cap, "id": f"{env}/{t}/run_{ep:02d}"})
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps({"seed": SEED, "spec": SPEC, "items": items}, indent=1))
    print(len(items), "items")


def run(env):
    from balrog.agents import AgentFactory
    from balrog.evaluator import Evaluator
    items = [i for i in json.loads(PLAN.read_text())["items"] if i["env"] == env]
    only = os.environ.get("E13_ONLY")  # optional: run a single item id (smoke)
    cfg = config(env)
    for it in items:
        if only and it["id"] != only:
            continue
        jf = OUT / env / it["task"] / f"{it['task']}_run_{it['episode_idx']:02d}.json"
        if jf.exists():
            continue
        cfg.envs.env_kwargs.seed = it["env_seed"]
        ev = Evaluator(env, cfg, original_cwd=os.getcwd(), output_dir=str(OUT))
        t0 = time.time()
        try:
            log = ev.run_episode(it["task"], AgentFactory(cfg).create_agent(), process_num="p0",
                                 episode_idx=it["episode_idx"])
            print(json.dumps({"id": it["id"], "progression": log.get("progression"), "steps": log.get("num_steps"),
                              "dt": round(time.time() - t0)}), flush=True)
        except Exception as e:  # counted as failure (progression 0) in the result
            err = OUT / "errors" / (it["id"].replace("/", "__") + ".txt")
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(traceback.format_exc())
            print(json.dumps({"id": it["id"], "error": repr(e)[:200]}), flush=True)


def summarize():
    from balrog.utils import collect_and_summarize_results
    s = collect_and_summarize_results(str(OUT))
    (OUT / "native_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--env"); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(SRC))
    if a.plan: plan()
    if a.env: run(a.env)
    if a.summarize: summarize()
