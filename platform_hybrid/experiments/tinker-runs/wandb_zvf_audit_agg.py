"""Aggregate the W&B `zvf-audit` runs via direct GraphQL (the wandb python client
is broken in this env). Reads WANDB_API_KEY from env. Paginates all runs, parses
difficulty from run names, and breaks ZVF / GU / reward / frac_solved down by
model / group-size / difficulty. Writes results/zvf_audit_summary.{json,md}.

Usage:  WANDB_API_KEY=... python3 platform_hybrid/experiments/tinker-runs/wandb_zvf_audit_agg.py
"""
import os, re, json, base64, urllib.request, collections, statistics, pathlib

KEY = os.environ["WANDB_API_KEY"]
ENT, PROJ = "arvindcr4-pes-university", "zvf-audit"
URL = "https://api.wandb.ai/graphql"
AUTH = base64.b64encode(f"api:{KEY}".encode()).decode()
OUT = pathlib.Path("platform_hybrid/experiments/tinker-runs/results")
Q = ("query($e:String!,$p:String!,$c:String){ project(name:$p,entityName:$e){ "
     "runs(first:250,after:$c){ edges{ node{ name state config summaryMetrics } } "
     "pageInfo{ hasNextPage endCursor } } } }")

def gql(cursor):
    body = json.dumps({"query": Q, "variables": {"e": ENT, "p": PROJ, "c": cursor}}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": "Basic " + AUTH,
                                             "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))

unwrap = lambda v: v["value"] if isinstance(v, dict) and "value" in v else v
def pick(d, *names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None
def num(x):
    try: return float(x)
    except Exception: return None
def difficulty_of(name):
    m = re.search(r"_(easy|medium|hard)_", name or "")
    return m.group(1) if m else None

runs, cursor = [], None
while True:
    conn = gql(cursor)["data"]["project"]["runs"]
    for e in conn["edges"]:
        n = e["node"]
        cfg = {k: unwrap(v) for k, v in json.loads(n.get("config") or "{}").items()}
        summ = json.loads(n.get("summaryMetrics") or "{}")
        runs.append((n["state"], cfg, summ, n.get("name", "")))
    if conn["pageInfo"]["hasNextPage"]:
        cursor = conn["pageInfo"]["endCursor"]
    else:
        break

states = dict(collections.Counter(s for s, c, m, nm in runs))
METRICS = {"zvf": ["mean_zvf", "zvf"], "gu": ["gu", "mean_gu"],
           "reward": ["final_reward", "reward"], "frac_solved": ["frac_solved"]}
DIMS = {
    "model": lambda c, m, nm: pick(c, "model_short", "model", "base_model"),
    "group_size": lambda c, m, nm: pick(c, "group", "group_size", "G"),
    "difficulty": lambda c, m, nm: difficulty_of(nm),
}

def aggregate(dim_fn):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, c, m, nm in runs:
        dv = dim_fn(c, m, nm)
        if dv is None:
            continue
        for mk, names in METRICS.items():
            v = num(pick(m, *names))
            if v is not None:
                g[str(dv)][mk].append(v)
    return g

summary = {"project": PROJ, "entity": ENT, "total_runs": len(runs), "states": states, "breakdowns": {}}
md = [f"# zvf-audit summary\n\n{len(runs)} runs · states: {states}\n"]
for label, fn in DIMS.items():
    g = aggregate(fn)
    summary["breakdowns"][label] = {}
    md.append(f"\n## by {label}\n\n| {label} | n | ZVF | GU | reward | frac_solved |\n|---|---|---|---|---|---|")
    print(f"\n=== by {label} ===")
    for k in sorted(g, key=lambda x: -statistics.mean(g[x]["zvf"]) if g[x].get("zvf") else 0):
        row = {mk: round(statistics.mean(v), 3) for mk, v in g[k].items()}
        row["n"] = max((len(v) for v in g[k].values()), default=0)
        summary["breakdowns"][label][k] = row
        print(f"  {k:30s} n={row['n']:>3}  ZVF={row.get('zvf','-')}  GU={row.get('gu','-')}  "
              f"reward={row.get('reward','-')}  solved={row.get('frac_solved','-')}")
        md.append(f"| {k} | {row['n']} | {row.get('zvf','-')} | {row.get('gu','-')} | "
                  f"{row.get('reward','-')} | {row.get('frac_solved','-')} |")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "zvf_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "zvf_audit_summary.md").write_text("\n".join(md) + "\n")
print("\nwrote results/zvf_audit_summary.{json,md}  ·", len(runs), "runs")
