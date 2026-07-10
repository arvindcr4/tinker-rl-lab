"""Cross-tab the zvf-audit runs (model x group-size ZVF) and summarize the other
big W&B projects (world-class, scaling) by model. Direct GraphQL; key from env.

Usage:  WANDB_API_KEY=... python3 platform_hybrid/experiments/tinker-runs/wandb_projects_explore.py
"""
import os, json, base64, urllib.request, collections, statistics

KEY = os.environ["WANDB_API_KEY"]
ENT = "arvindcr4-pes-university"
URL = "https://api.wandb.ai/graphql"
AUTH = base64.b64encode(f"api:{KEY}".encode()).decode()
Q = ("query($e:String!,$p:String!,$c:String){ project(name:$p,entityName:$e){ "
     "runs(first:250,after:$c){ edges{ node{ name state config summaryMetrics } } "
     "pageInfo{ hasNextPage endCursor } } } }")
unwrap = lambda v: v["value"] if isinstance(v, dict) and "value" in v else v
def pick(d, *names):
    for n in names:
        if n in d and d[n] not in (None, ""): return d[n]
    return None
def num(x):
    try: return float(x)
    except Exception: return None
def short(m): return str(m).split("/")[-1].replace("-Instruct-2507", "-Inst").replace("NVIDIA-", "")

def fetch(proj):
    runs, cursor = [], None
    while True:
        body = json.dumps({"query": Q, "variables": {"e": ENT, "p": proj, "c": cursor}}).encode()
        req = urllib.request.Request(URL, body, {"Authorization": "Basic " + AUTH, "Content-Type": "application/json"})
        conn = json.load(urllib.request.urlopen(req, timeout=90))["data"]["project"]["runs"]
        for e in conn["edges"]:
            n = e["node"]
            runs.append((n["state"], {k: unwrap(v) for k, v in json.loads(n.get("config") or "{}").items()},
                         json.loads(n.get("summaryMetrics") or "{}"), n.get("name", "")))
        if conn["pageInfo"]["hasNextPage"]: cursor = conn["pageInfo"]["endCursor"]
        else: break
    return runs

# ---- 1. zvf-audit: model x group-size ZVF cross-tab ----
audit = fetch("zvf-audit")
cell = collections.defaultdict(list); models = set(); gs = set()
for s, c, m, nm in audit:
    mdl, G, z = pick(c, "model_short", "model"), pick(c, "group", "group_size"), num(pick(m, "mean_zvf", "zvf"))
    if mdl and G is not None and z is not None:
        cell[(short(mdl), str(int(float(G))))].append(z); models.add(short(mdl)); gs.add(str(int(float(G))))
gcols = sorted(gs, key=int)
print("=== zvf-audit: mean ZVF by model x group-size ===")
print(f"{'model':26s} " + " ".join(f"G={g:>3}" for g in gcols))
for mdl in sorted(models, key=lambda x: -statistics.mean([v for (mm, gg), vs in cell.items() if mm == x for v in vs])):
    row = []
    for g in gcols:
        vs = cell.get((mdl, g))
        row.append(f"{statistics.mean(vs):.2f}" if vs else "  · ")
    print(f"{mdl:26s} " + " ".join(f"{c:>5}" for c in row))

# ---- 2. other big projects: by-model reward/ZVF ----
for proj in ["tinker-rl-lab-world-class", "tinker-rl-scaling"]:
    rs = fetch(proj)
    states = dict(collections.Counter(s for s, c, m, nm in rs))
    sk = collections.Counter()
    for s, c, m, nm in rs: sk.update(m.keys())
    print(f"\n=== {proj}: {len(rs)} runs · states {states} ===")
    print("  summary keys:", [k for k, _ in sk.most_common(14)])
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, c, m, nm in rs:
        mdl = pick(c, "model_short", "model", "base_model") or "?"
        for mk, names in {"reward": ["final_reward", "reward", "eval_reward"], "zvf": ["mean_zvf", "zvf"]}.items():
            v = num(pick(m, *names))
            if v is not None: by[short(mdl)][mk].append(v)
    for mdl in sorted(by):
        r = {k: round(statistics.mean(v), 3) for k, v in by[mdl].items()}
        n = max((len(v) for v in by[mdl].values()), default=0)
        print(f"  {mdl:34s} n={n:>3}  reward={r.get('reward','-')}  ZVF={r.get('zvf','-')}")
