"""Pull per-step histories of the 368 W&B `zvf-audit` runs (input to zvf_confound.py).
Usage: python pull_zvf_audit_histories.py experiments/results/publication_fixes/zvf_confound/zvf_audit_histories.json
Reads the W&B API key from ~/.netrc (api.wandb.ai)."""
import netrc,base64,json,urllib.request,sys,pathlib
from concurrent.futures import ThreadPoolExecutor
REG=pathlib.Path(__file__).resolve().parents[1]/"experiments/results/tinker_wandb_registry/wandb_runs.jsonl"
OUT=sys.argv[1]
k=netrc.netrc().authenticators('api.wandb.ai')[2]
AUTH=base64.b64encode(f"api:{k}".encode()).decode()
runs=[json.loads(l) for l in open(REG)]
runs=[r for r in runs if r['project']=='zvf-audit']
spec=json.dumps({"keys":["step","zvf","reward","gu","n_groups","n_datums","frac_solved"],"samples":1000})
def get(r):
    Q='query($s:[JSONString!]!,$n:String!){ project(name:"zvf-audit",entityName:"arvindcr4-pes-university"){ run(name:$n){ name sampledHistory(specs:$s) } } }'
    req=urllib.request.Request("https://api.wandb.ai/graphql",json.dumps({"query":Q,"variables":{"s":[spec],"n":r['run_id']}}).encode(),{"Authorization":"Basic "+AUTH,"Content-Type":"application/json"})
    for _ in range(3):
        try:
            d=json.load(urllib.request.urlopen(req,timeout=60))
            h=d['data']['project']['run']['sampledHistory'][0]
            return {"run_id":r['run_id'],"name":r['display_name'],"model":r['model'],"state":r['state'],"config":{k:v for k,v in r['config'].items() if not k.startswith('_')},"summary":{k:v for k,v in r['summary'].items() if not k.startswith('_')},"history":h}
        except Exception as e: err=e
    return {"run_id":r['run_id'],"error":str(err)}
with ThreadPoolExecutor(12) as ex: res=list(ex.map(get,runs))
json.dump(res,open(OUT,'w'))
print(len(res), sum('error' in x for x in res))
