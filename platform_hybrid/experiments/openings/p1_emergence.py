"""P1 mechanism — WHEN does the dominant layer band emerge? (why step-1 prediction fails)

The scaled P1 result: step-1 top-k does NOT predict the final top-k (overlap 0.11). This measures
the emergence curve: overlap(step_k top-k, FINAL top-k) as a function of k. If it starts low and
rises over training, the band emerges gradually -> the mechanistic reason step-1 prediction fails.
Colab GPU; writes /content/p1_emergence_result.json.
"""
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U","torchao>=0.16","transformers","peft","datasets"], check=False)
import os, re, json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROBLEMS, N_STEPS, G, SEEDS = 24, 12, 4, [0, 1]
SYS = "Solve the problem. End with 'The answer is <number>'."
def gold(a):
    t=a.split("####")[-1].strip().replace(",","").replace("$",""); m=re.search(r"-?\d+\.?\d*",t); return m.group(0) if m else t
def reward(txt,g):
    n=re.findall(r"-?\d+\.?\d*",txt.replace(",","")); return 1.0 if n and abs(float(n[-1])-float(g))<1e-4 else 0.0
def layer_of(nm):
    m=re.search(r"\.layers\.(\d+)\.",nm); return int(m.group(1)) if m else None

def load_problems(tok):
    ds=load_dataset("openai/gsm8k","main",split="train").select(range(N_PROBLEMS)); out=[]
    for ex in ds:
        p=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":ex["question"]}],tokenize=False,add_generation_prompt=True)
        out.append({"prompt":p,"gold":gold(ex["answer"])})
    return out

def run_seed(seed, tok, problems):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16).to("cuda")
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
    n_layers=model.config.num_hidden_layers
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    import random as _r; _r.seed(seed)
    per_step=[]  # per-step per-layer grad norm
    for step in range(N_STEPS):
        model.eval(); opt.zero_grad(); used=0
        for pr in _r.sample(problems, min(4,len(problems))):
            ids=tok(pr["prompt"],return_tensors="pt").to("cuda"); plen=ids.input_ids.shape[1]
            with torch.no_grad():
                gen=model.generate(**ids,do_sample=True,temperature=0.9,top_p=0.95,num_return_sequences=G,max_new_tokens=160,pad_token_id=tok.eos_token_id)
            rews=[reward(tok.decode(gen[j][plen:],skip_special_tokens=True),pr["gold"]) for j in range(G)]
            mr=sum(rews)/G; sd=(sum((x-mr)**2 for x in rews)/G)**0.5+1e-8
            if mr in (0.0,1.0): continue
            model.train()
            for j in range(G):
                adv=(rews[j]-mr)/sd; seq=gen[j:j+1]
                logits=model(seq).logits[:,plen-1:-1,:]; tgt=seq[:,plen:]
                lp=logits.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(logits,dim=-1)
                (-adv*lp.sum()/(4*G)).backward(); used+=1
        gn=[0.0]*n_layers
        for nm,p in model.named_parameters():
            if p.grad is not None and "lora" in nm.lower():
                L=layer_of(nm)
                if L is not None and L<n_layers: gn[L]+=p.grad.detach().float().norm().item()**2
        per_step.append([x**0.5 for x in gn]); opt.step()
    del model; torch.cuda.empty_cache()
    M=np.array(per_step)  # [steps, layers]
    k=max(3,n_layers//4)
    final_top=set(np.argsort(-M.mean(0))[:k].tolist())
    # emergence: overlap(step_k top-k, FINAL top-k) for each k
    emergence=[len(set(np.argsort(-M[s])[:k].tolist()) & final_top)/k for s in range(len(M))]
    return dict(seed=seed, n_layers=n_layers, k=k, emergence_curve=[round(x,3) for x in emergence],
                step1_overlap=round(emergence[0],3), final_overlap=round(emergence[-1],3),
                step_reaches_half=next((s+1 for s,v in enumerate(emergence) if v>=0.5), None))

tok=AutoTokenizer.from_pretrained(MODEL); problems=load_problems(tok)
res=[run_seed(s,tok,problems) for s in SEEDS]
for r in res: print(f"[seed {r['seed']}] emergence={r['emergence_curve']} half@step={r['step_reaches_half']}", flush=True)
import statistics as st
mean_curve=[round(st.mean([r["emergence_curve"][i] for r in res]),3) for i in range(min(len(r["emergence_curve"]) for r in res))]
summary=dict(model=MODEL, per_seed=res, mean_emergence_curve=mean_curve,
             mean_step1=round(st.mean([r["step1_overlap"] for r in res]),3))
open("/content/p1_emergence_result.json","w").write(json.dumps(summary,indent=2))
print("RESULT: "+json.dumps(summary))
