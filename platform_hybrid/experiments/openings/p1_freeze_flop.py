"""P1 follow-up — the ACTUAL layer-freeze test (does freezing cold layers keep accuracy?).

The scaled P1 showed step-1 predictability collapses, but concentration holds: a hot mid-late
band (+ layer 0) dominates adaptation. This tests the paper's real claim: apply LoRA ONLY to the
hot band, FREEZE the cold layers, and compare held-out accuracy + trainable-param count to full-LoRA.
If frozen ~= full on held-out with far fewer trainable params, the freeze lever works (read off the
EMERGED band, not step-1). Two arms x seeds. Colab GPU; writes /content/p1_freeze_result.json.
"""
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U","torchao>=0.16","transformers","peft","datasets"], check=False)
import os, re, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROBLEMS, N_STEPS, G, SEEDS = 24, 10, 4, [0, 1]
SYS = "Solve the problem. End with 'The answer is <number>'."

def gold(ans):
    t = ans.split("####")[-1].strip().replace(",","").replace("$","")
    m = re.search(r"-?\d+\.?\d*", t); return m.group(0) if m else t
def reward(txt, g):
    nums = re.findall(r"-?\d+\.?\d*", txt.replace(",",""))
    return 1.0 if nums and abs(float(nums[-1])-float(g))<1e-4 else 0.0
def layer_of(name):
    m = re.search(r"\.layers\.(\d+)\.", name); return int(m.group(1)) if m else None

def load_problems(tok):
    ds = load_dataset("openai/gsm8k","main",split="train").select(range(N_PROBLEMS))
    out=[]
    for ex in ds:
        p = tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":ex["question"]}],
                                    tokenize=False, add_generation_prompt=True)
        out.append({"prompt":p,"gold":gold(ex["answer"])})
    return out

def run_arm(arm, hot_layers, n_layers, tok, problems, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    if arm == "frozen":
        # LoRA only on the hot band (layer 0 + mid-late); cold layers stay frozen (no adapter)
        pat = [f"model.layers.{L}." for L in hot_layers]
        lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                          layers_to_transform=hot_layers, task_type="CAUSAL_LM")
    else:
        lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    def heldout():
        model.eval(); c=0
        for pr in problems[:8]:
            ids = tok(pr["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
            c += reward(tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True), pr["gold"])
        return c/8
    import random as _r; _r.seed(seed)
    h0 = heldout()
    for step in range(N_STEPS):
        model.eval(); opt.zero_grad()
        for pr in _r.sample(problems[8:], min(4, len(problems)-8)):
            ids = tok(pr["prompt"], return_tensors="pt").to("cuda"); plen = ids.input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(**ids, do_sample=True, temperature=0.9, top_p=0.95,
                                     num_return_sequences=G, max_new_tokens=160, pad_token_id=tok.eos_token_id)
            rews = [reward(tok.decode(gen[j][plen:], skip_special_tokens=True), pr["gold"]) for j in range(G)]
            mr = sum(rews)/G; sd = (sum((x-mr)**2 for x in rews)/G)**0.5 + 1e-8
            if mr in (0.0,1.0): continue
            model.train()
            for j in range(G):
                adv = (rews[j]-mr)/sd
                seq = gen[j:j+1]
                logits = model(seq).logits[:, plen-1:-1, :]
                tgt = seq[:, plen:]
                # memory-efficient token log-prob (logsumexp, no full-vocab float32 softmax)
                tok_lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
                (-adv * tok_lp.sum() / (4*G)).backward()
        opt.step()
    h1 = heldout()
    del model; torch.cuda.empty_cache()
    return dict(arm=arm, seed=seed, trainable_params=trainable, heldout_before=h0, heldout_after=h1, heldout_gain=h1-h0)

tok = AutoTokenizer.from_pretrained(MODEL)
problems = load_problems(tok)
_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
n_layers = _m.config.num_hidden_layers; del _m
# hot band = layer 0 + top ~55% (mid-late), from the scaled-P1 emerged profile
hot = sorted(set([0] + list(range(int(n_layers*0.45), n_layers))))
print(f"n_layers={n_layers} hot_band={hot} (frozen arm trains only these)", flush=True)
results=[]
for seed in SEEDS:
    for arm in ["full", "frozen"]:
        r = run_arm(arm, hot, n_layers, tok, problems, seed)
        results.append(r); print(f"[{arm} s{seed}] trainable={r['trainable_params']} gain={r['heldout_gain']:+.3f}", flush=True)
import statistics as st
summary = {"model": MODEL, "n_layers": n_layers, "hot_band": hot, "per_run": results}
for arm in ["full","frozen"]:
    g=[r["heldout_gain"] for r in results if r["arm"]==arm]; tp=[r["trainable_params"] for r in results if r["arm"]==arm]
    summary[arm] = {"mean_gain": st.mean(g), "trainable_params": tp[0]}
summary["param_ratio_frozen_over_full"] = summary["frozen"]["trainable_params"]/summary["full"]["trainable_params"]
open("/content/p1_freeze_result.json","w").write(json.dumps(summary, indent=2))
print("RESULT: " + json.dumps(summary))
