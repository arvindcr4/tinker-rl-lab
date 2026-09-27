#!/usr/bin/env python3
"""E13 BALROG replacement scope: all 255 native episodes (58 task configs, 6 envs incl. NetHack).

Native pieces used unchanged: balrog@b7afe79 config.yaml (episode counts, env kwargs, step caps, temperature 1.0,
max_tokens 8192, max_retries 5, timeout 60), Evaluator.run_episode, NaiveAgent via AgentFactory, the native `vllm`
OpenAI client and collect_and_summarize_results. Seeds: native (env_kwargs.seed=null -> get_unique_seed, logged
in each episode JSON). The client points at code/actor_proxy.py, which forwards to the shared trained actor.

  python run_full.py --plan                # raw/plan.json (255 items, native enumeration order)
  python run_full.py --run --workers 6     # run all items lacking a native JSON (resumable)
  python run_full.py --summarize           # raw/native_summary.json via native collect_and_summarize_results
Env: BALROG_SRC, E13_OUT (=raw dir), PROXY_URL (http://127.0.0.1:8793), TW_GAMES (abs path to tw_games).
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
from pathlib import Path

SRC = Path(os.environ["BALROG_SRC"])
OUT = Path(os.environ["E13_OUT"])
PLAN = OUT / "plan.json"
MODEL = "pavlov-public-portfolio-bf16"
# longest-first scheduling order (does not change the item set)
ORDER = ["crafter", "babaisai", "minihack", "textworld", "babyai", "nle"]  # NLE last (lead decision 01:40 UTC)
# Deviation: NetHack capped at NLE_STEP_CAP agent steps via the native eval.max_steps_per_episode knob; the native
# evaluator then records progression-at-truncation. Only applied to nle; all other envs keep native caps.
NLE_STEP_CAP = int(os.environ.get("NLE_STEP_CAP", "2000"))


def native_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(SRC / "balrog/config/config.yaml")


def config(env):
    cfg = native_cfg()
    cfg.client.client_name = "vllm"
    cfg.client.model_id = MODEL
    cfg.client.base_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8793") + f"/{env}/v1"
    cfg.envs.names = env
    if env == "nle":
        cfg.eval.max_steps_per_episode = NLE_STEP_CAP
    cfg.envs.textworld_kwargs.textworld_games_path = os.environ["TW_GAMES"]
    return cfg


def jpath(it):
    return OUT / it["env"] / it["task"] / f"{it['task']}_run_{it['episode_idx']:02d}.json"


def plan():
    cfg = native_cfg()
    items = []
    for env in cfg.envs.names.split("-"):
        for task in cfg.tasks[f"{env}_tasks"]:
            for ep in range(cfg.eval.num_episodes[env]):
                items.append({"env": env, "task": task, "episode_idx": ep, "id": f"{env}/{task}/episode-{ep:02d}"})
    assert len(items) == 255, len(items)
    OUT.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        PLAN.write_text(json.dumps({"n": len(items), "items": items}, indent=1))
    print(len(items), "items")


def run_one(it):
    sys.path.insert(0, str(SRC))
    os.chdir(os.environ["WORKDIR"])
    from balrog.agents import AgentFactory
    from balrog.evaluator import Evaluator
    cfg = config(it["env"])
    t0 = time.time()
    rec = {"id": it["id"], "start": t0, "pid": os.getpid()}
    try:
        ev = Evaluator(it["env"], cfg, original_cwd=os.getcwd(), output_dir=str(OUT))
        log = ev.run_episode(it["task"], AgentFactory(cfg).create_agent(), process_num=f"w{os.getpid()}",
                             episode_idx=it["episode_idx"])
        rec.update(status="ok", progression=log.get("progression"), steps=log.get("num_steps"), seed=log.get("seed"))
    except Exception as e:
        err = OUT / "errors" / (it["id"].replace("/", "__") + f".{int(t0)}.txt")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        rec.update(status="error", error=repr(e)[:300])
    rec["dt"] = round(time.time() - t0, 1)
    with open(OUT / "attempts.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def run(workers):
    items = json.loads(PLAN.read_text())["items"]
    todo = [i for i in items if not jpath(i).exists()]
    todo.sort(key=lambda i: ORDER.index(i["env"]))
    print(f"{len(todo)} to run", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(run_one, todo, chunksize=1):
            pass


def summarize():
    sys.path.insert(0, str(SRC))
    from balrog.utils import collect_and_summarize_results
    s = collect_and_summarize_results(str(OUT))
    (OUT / "native_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.plan: plan()
    if a.run: run(a.workers)
    if a.summarize: summarize()
