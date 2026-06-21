"""E3: reproducibility audit in ONE controlled open trainer + live adaptive-G.

Colab-only: the original head-to-head ran on CLOSED Tinker (loss kernel
unauditable, not swappable). Here every loss arm is re-implemented in the same
open loop with identical sampler/precision/KL handling, so we can see which
algorithmic gains survive the stack being held fixed (MIN-REPORT Pillar 4), and
whether a zvf-triage-style adaptive-G actually reduces ZVF live (Pillar 3).

Arms (all share old-policy logprob caching + 2 inner epochs so ratio/clip engage):
  grpo            : adv=(r-mean)/(std+eps), symmetric clip 0.2
  drgrpo          : adv=(r-mean)  [NO /std], symmetric clip 0.2
  dapo            : adv=(r-mean)/(std+eps), asymmetric clip [0.2, 0.28],
                    dynamic sampling (resample zero-variance groups)
  grpo_adaptiveG  : grpo + zvf-triage controller raises G when recent ZVF high

Run:  colab run --gpu T4 --timeout 1200 e3_open_audit.py
"""
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
BATCH, G0, GMAX, MAX_NEW, LR, STEPS, INNER = 3, 4, 10, 40, 2e-6, 10, 2
HELDOUT_N = 20
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():
    a, b = random.randint(11, 60), random.randint(11, 60)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

def seq_logprob(model, pids, gen_row, grad):
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(ids).logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return lp[:, pids.shape[0] - 1:].sum()

def gen_group(model, prompt, gold, g):
    model.eval()
    enc = tok([prompt] * g, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(t) == gold else 0.0 for t in tok.batch_decode(gens, skip_special_tokens=True)]
    pids = enc.input_ids[0]
    old = [seq_logprob(model, pids, gens[i], grad=False) for i in range(g)]
    return pids, gens, rewards, old

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval(); c = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=PAD)
        if parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)) == gold:
            c += 1
    return c / len(evalset)

def advantages(rewards, arm):
    m = sum(rewards) / len(rewards)
    v = statistics.pvariance(rewards); s = v ** 0.5
    if v == 0.0:
        return None
    if arm == "drgrpo":
        return [r - m for r in rewards]                 # no /std
    return [(r - m) / (s + 1e-6) for r in rewards]      # grpo / dapo

def clipped_loss(new_lp, old_lp, adv, arm):
    ratio = torch.exp(new_lp - old_lp.detach())
    lo, hi = (0.2, 0.28) if arm == "dapo" else (0.2, 0.2)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - lo, 1 + hi) * adv
    return -torch.min(unclipped, clipped)

def run(arm, seed):
    random.seed(seed); torch.manual_seed(seed)
    evalset = [problem() for _ in range(HELDOUT_N)]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    pre = heldout_acc(model, evalset)
    g, recent_zvf, traj, rollouts = G0, [], [], 0
    for step in range(1, STEPS + 1):
        if arm == "grpo_adaptiveG" and len(recent_zvf) >= 2 and statistics.mean(recent_zvf[-2:]) > 0.4:
            g = min(GMAX, g + 2)                         # zvf-triage: escalate G under collapse
        groups, zv = [], 0
        for _ in range(BATCH):
            q, gold = problem()
            tries = 0
            while True:
                pids, gens, rewards, old = gen_group(model, prompt_of(q), gold, g)
                rollouts += g
                adv = advantages(rewards, arm)
                if adv is not None or arm != "dapo" or tries >= 2:
                    break
                tries += 1                               # DAPO dynamic sampling: resample dead group
            if adv is None:
                zv += 1; continue
            groups.append((pids, gens, adv, old))
        for _ in range(INNER):
            opt.zero_grad(set_to_none=True)
            losses = []
            for pids, gens, adv, old in groups:
                for i, a in enumerate(adv):
                    if a == 0:
                        continue
                    new = seq_logprob(model, pids, gens[i], grad=True)
                    if new is not None and old[i] is not None:
                        losses.append(clipped_loss(new, old[i], a, arm))
            if losses:
                torch.stack(losses).sum().backward(); opt.step()
        zvf = zv / BATCH
        recent_zvf.append(zvf)
        # rough mean reward proxy: recompute correctness count from last groups not stored; track via zvf-comp
        traj.append({"step": step, "zvf": round(zvf, 3), "G": g})
        print(f"[e3:{arm[:8]:8s} s{seed}] step={step:2d} G={g} ZVF={zvf:.2f}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"arm": arm, "seed": seed, "heldout_pre": round(pre, 3), "heldout_post": round(post, 3),
           "heldout_delta": round(post - pre, 3), "mean_zvf": round(statistics.mean(t["zvf"] for t in traj), 3),
           "final_G": g, "total_rollouts": rollouts}
    del model, opt; torch.cuda.empty_cache()
    return out

results = []
for arm in ["grpo", "drgrpo", "dapo", "grpo_adaptiveG"]:
    for seed in SEEDS:
        results.append(run(arm, seed))

agg = {}
for arm in ["grpo", "drgrpo", "dapo", "grpo_adaptiveG"]:
    rs = [r for r in results if r["arm"] == arm]
    agg[arm] = {"mean_heldout_delta": round(statistics.mean(r["heldout_delta"] for r in rs), 3),
                "mean_zvf": round(statistics.mean(r["mean_zvf"] for r in rs), 3),
                "mean_rollouts": round(statistics.mean(r["total_rollouts"] for r in rs), 1)}
print("E3_RESULT " + json.dumps({"experiment": "E3_open_audit", "model": MODEL,
                                 "seeds": SEEDS, "by_arm": agg}), flush=True)
