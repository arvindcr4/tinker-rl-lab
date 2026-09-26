"""E6 substitute: MiniWoB++ (miniwob==1.1.0) text-DOM agent driven by the base actor via local Tinker shim.
Selection: random.Random(20260926).sample(sorted(miniwob env ids), N_TASKS); episodes use seeds 0..EPS-1.
Task timers disabled in code/html (EPISODE_MAX_TIME=1e6, BrowserGym convention).
Success = final env reward > 0 (MiniWoB native reward; >0 means task solved)."""
import json, os, random, re, sys, time, traceback, urllib.request
import gymnasium, miniwob
from miniwob.action import ActionTypes

gymnasium.register_envs(miniwob)
SEED, N_TASKS, EPS, MAX_STEPS, MAX_TOKENS = 20260926, 12, 3, 10, 256
BASE_URL = "file://" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "html", "miniwob") + "/"
SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:18769/chat")
OUT = sys.argv[1]

SYSTEM = """You are a web agent operating a small web page (160x210 px task area). Each turn you see the task instruction and the page's DOM elements
(ref, tag, text/value, id/class, flags, and bounding box left,top,width,height in px). Reply with exactly ONE action on the first line, no other text:
CLICK(ref)                    - click element by ref
TYPE(ref, "text")             - focus element ref and type text
CLICK_XY(x, y)                - click page coordinates
KEY("key")                    - press key, e.g. "Enter", "Backspace", "C-a", "C-c", "C-v"
The episode ends automatically when the task is complete."""


def dom_text(obs):
    lines = []
    for e in obs["dom_elements"]:
        l, t, w, h = (float(e[k][0]) for k in ("left", "top", "width", "height"))
        txt = (e["text"] or "")[:60].replace("\n", " ")
        extra = []
        if e["value"]: extra.append(f'value="{str(e["value"])[:40]}"')
        if e["id"]: extra.append(f'id={e["id"]}')
        if e["classes"]: extra.append(f'class={e["classes"][:30]}')
        f = e["flags"]
        if f[0]: extra.append("focused")
        if f[1]: extra.append("tampered")
        lines.append(f'[{e["ref"]}] <{e["tag"]}> "{txt}" {" ".join(extra)} box=({l:.0f},{t:.0f},{w:.0f},{h:.0f})')
    return "\n".join(lines[:150])


def ask(messages, tag):
    req = urllib.request.Request(SHIM, data=json.dumps({"messages": messages, "max_tokens": MAX_TOKENS, "tag": tag}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def parse(env, text):
    s = text.strip().splitlines()[0] if text.strip() else ""
    u = env.unwrapped
    if m := re.match(r'\s*TYPE\(\s*(\d+)\s*,\s*"(.*)"\s*\)', s):
        return u.create_action(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=int(m[1]), text=m[2]), s
    if m := re.match(r"\s*CLICK_XY\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", s):
        import numpy as np
        return u.create_action(ActionTypes.CLICK_COORDS, coords=np.array([float(m[1]), float(m[2])], dtype="float32")), s
    if m := re.match(r"\s*CLICK\(\s*(\d+)\s*\)", s):
        return u.create_action(ActionTypes.CLICK_ELEMENT, ref=int(m[1])), s
    if m := re.match(r'\s*KEY\(\s*"(.+)"\s*\)', s):
        keys = list(u.action_space_config.allowed_keys)
        k = m[1]
        if k in keys:
            return u.create_action(ActionTypes.PRESS_KEY, key=keys.index(k)), s
    return None, s


def main():
    ids = sorted(k for k in gymnasium.registry if k.startswith("miniwob/"))
    tasks = random.Random(SEED).sample(ids, N_TASKS)
    json.dump({"seed": SEED, "canonical_count": len(ids), "tasks": tasks, "episodes_per_task": EPS}, open(f"{OUT}/selection.json", "w"), indent=1)
    rows = []
    for task in tasks:
        env = gymnasium.make(task, base_url=BASE_URL)
        for ep in range(EPS):
            rec = {"task": task, "episode_seed": ep, "steps": [], "reward": 0.0, "success": False, "error": None}
            try:
                obs, _ = env.reset(seed=ep)
                rec["utterance"] = obs["utterance"]
                history = []
                for step in range(MAX_STEPS):
                    user = f'Task: {obs["utterance"]}\nPrevious actions: {history or "none"}\nDOM:\n{dom_text(obs)}\nAction:'
                    r = ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], f"{task}#{ep}#{step}")
                    if "error" in r:
                        raise RuntimeError(r["error"])
                    action, line = parse(env, r["text"])
                    history.append(line)
                    st = {"model": r["text"][:300], "parsed": action is not None}
                    if action is None:
                        rec["steps"].append(st)
                        continue
                    obs, reward, term, trunc, info = env.step(action)
                    st["reward"] = float(reward)
                    rec["steps"].append(st)
                    if term or trunc:
                        rec["reward"] = float(reward)
                        break
            except Exception as e:
                rec["error"] = repr(e)[:500]
                traceback.print_exc()
            rec["success"] = rec["reward"] > 0
            rows.append(rec)
            print(task, ep, rec["reward"], rec["success"], rec["error"], flush=True)
            with open(f"{OUT}/episodes.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        env.close()


if __name__ == "__main__":
    main()
