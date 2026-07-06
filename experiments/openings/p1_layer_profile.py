# P1 white-box experiment (Colab L4) — per-layer adaptation profile under GRPO-style updates.
# Tests: do GRPO gains concentrate in a few layers, and is the concentration PREDICTABLE from step 1?
# (the premise of "causal predictive layer-freezing" that distinguishes P1 from SALF).
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U","torchao>=0.16","transformers>=4.55","peft"], check=False)
import os, re, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
G, STEPS, N_PROMPTS = 4, 5, 4
print("loading", MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.train()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# tiny GSM8K-style set (hardcoded to avoid dataset download flakiness)
PROB = [("What is 12*11?","132"),("What is 15+27?","42"),("A pen costs 3, buy 7. Total?","21"),
        ("Half of 46?","23"),("What is 9*8?","72"),("100 minus 37?","63")]
SYS = "Solve. Put the final number in \\boxed{}."
def reward(text, ans):
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    for b in m:
        try:
            if abs(float(b.strip())-float(ans))<0.01: return 1.0
        except: pass
    nums = re.findall(r"-?\d+\.?\d*", text)
    return 1.0 if nums and abs(float(nums[-1])-float(ans))<0.01 else 0.0

n_layers = model.config.num_hidden_layers
def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

profile = []   # per step: {layer: grad_norm}
import random; random.seed(0)
for step in range(STEPS):
    batch = random.sample(PROB, N_PROMPTS)
    opt.zero_grad()
    total_adv_logp = 0.0; used = 0
    for q, ans in batch:
        msgs = [{"role":"system","content":SYS},{"role":"user","content":q}]
        _enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        pids = (_enc["input_ids"] if isinstance(_enc, dict) or hasattr(_enc,"keys") else _enc).to("cuda")
        with torch.no_grad():
            gen = model.generate(pids, max_new_tokens=80, do_sample=True, temperature=0.9,
                                 num_return_sequences=G, pad_token_id=tok.eos_token_id)
        rews, seqs = [], []
        for g in range(G):
            out = gen[g][pids.shape[1]:]
            txt = tok.decode(out, skip_special_tokens=True)
            rews.append(reward(txt, ans)); seqs.append((pids, gen[g:g+1]))
        mr = sum(rews)/len(rews); sd = (sum((r-mr)**2 for r in rews)/len(rews))**0.5 + 1e-8
        if mr in (0.0,1.0):  # collapsed group -> no signal (P2 phenomenon), skip
            continue
        for (pids_i, full), r in zip(seqs, rews):
            adv = (r-mr)/sd
            logits = model(full).logits[:, pids_i.shape[1]-1:-1, :]
            tgt = full[:, pids_i.shape[1]:]
            logp = torch.log_softmax(logits.float(),-1).gather(-1, tgt.unsqueeze(-1)).sum()
            (-adv*logp/ (N_PROMPTS*G)).backward()
            used += 1
    # record per-layer LoRA grad norm
    gn = {}
    for name, p in model.named_parameters():
        if p.grad is not None and "lora" in name.lower():
            L = layer_of(name)
            if L>=0: gn[L] = gn.get(L,0.0) + p.grad.detach().float().norm().item()**2
    gn = {L: v**0.5 for L,v in gn.items()}
    profile.append(gn)
    opt.step()
    top = sorted(gn, key=gn.get, reverse=True)[:5]
    print(f"step {step+1}/{STEPS} used={used} top5_layers={top}", flush=True)

# analysis: is the top-k set stable from step 1? (predictability)
import numpy as np
M = np.zeros((len(profile), n_layers))
for s,gn in enumerate(profile):
    for L,v in gn.items(): M[s,L]=v
meanL = M.mean(0)
rank = np.argsort(-meanL)
topk = set(rank[:max(3,n_layers//4)].tolist())
step1_topk = set(np.argsort(-M[0])[:len(topk)].tolist()) if len(profile)>0 else set()
overlap = len(topk & step1_topk)/max(len(topk),1)
res = dict(model=MODEL, n_layers=n_layers, steps=len(profile),
           mean_gradnorm_by_layer=[round(x,4) for x in meanL.tolist()],
           top_layers_overall=rank[:8].tolist(),
           concentration_top25pct_share=round(float(meanL[rank[:max(1,n_layers//4)]].sum()/ (meanL.sum()+1e-9)),3),
           step1_predicts_final_topk_overlap=round(overlap,3))
open("/content/p1_layer_profile_result.json","w").write(json.dumps(res, indent=2))
print("RESULT:", json.dumps(res))
