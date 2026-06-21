"""E6 (Pillar 3, operationalization): does the live ZVF-triage controller beat a
fixed-G baseline at MATCHED total-rollout budget?

Codex-review fixes vs the first draft:
  * FIXED PROMPT POOL with STABLE integer IDs so the per-prompt drop logic can
    actually fire across steps (the previous design had no persistent IDs).
  * MATCH ON TOTAL ROLLOUTS (primary); optimizer steps reported as secondary.
  * THREE arms to separate the controller's two effects:
       fixed_G          : G constant, no triage (control)
       adaptiveG        : adaptive group size only (no drop)
       adaptiveG_drop   : adaptive group size + per-prompt drop + auto-stop

The controller logic is vendored verbatim from zvf_triage.controller.ZVFController
(adaptive_fn, drop_k, stop_k, smoothed-ZVF group sizing) because `colab run` ships
ONE file to a fresh VM. Semantics match the repo package.

Colab-only because: live within-run mutation of group size, per-prompt dropping,
and gradient-based updates under a swappable loss are impossible on closed,
LoRA-only, fixed-loop Tinker.

Run:  colab run --gpu T4 --timeout 1500 e6_live_triage.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
POOL_N, BATCH_P, MAX_NEW, LR = 48, 4, 32, 2e-6
G0, GMIN, GMAX = 4, 2, 12
BUDGET = 600           # total rollouts per arm (matched compute, primary axis)
HELDOUT_N = 20
WINDOW, DROP_K, STOP_K, ZVF_MAX, EPS_LO = 5, 3, 4, 0.85, 0.05
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

# ---- vendored controller (mirrors zvf_triage.controller.ZVFController) ----
def adaptive_fn(z):
    z = min(max(float(z), 0.0), 1.0)
    return 0.5 + (z / 0.4) * 0.5 if z < 0.4 else 1.0 + ((z - 0.4) / 0.6) * 1.0

class Controller:
    def __init__(self, adaptive_G, drop):
        self.adaptive_G, self.drop = adaptive_G, drop
        self.hist, self.streak = [], {}
        self.global_streak, self.dropped, self.stopped = 0, set(), False
    def rolling(self):
        w = self.hist[-WINDOW:]
        return sum(w) / len(w) if w else 0.0
    def group_size(self):
        if not self.adaptive_G:
            return G0
        return int(round(min(max(G0 * adaptive_fn(self.rolling()), GMIN), GMAX)))
    def step(self, per_prompt_zerovar, batch_zvf, mean_reward):
        """per_prompt_zerovar: {gid: bool}. Returns (newly_dropped, auto_stop)."""
        self.hist.append(batch_zvf)
        newly = []
        if self.drop:
            for gid, zv in per_prompt_zerovar.items():
                if gid in self.dropped:
                    continue
                if zv:
                    self.streak[gid] = self.streak.get(gid, 0) + 1
                    if self.streak[gid] >= DROP_K:
                        self.dropped.add(gid); newly.append(gid)
                else:
                    self.streak[gid] = 0
        if batch_zvf >= 1.0 - 1e-12 and mean_reward < EPS_LO:
            self.global_streak += 1
        else:
            self.global_streak = 0
        auto_stop = self.drop and self.global_streak >= STOP_K and mean_reward < EPS_LO
        if auto_stop:
            self.stopped = True
        return newly, auto_stop

# ---- task / harness ----
def make_pool(rng):
    pool = []
    for i in range(POOL_N):
        if i % 2 == 0:                       # solvable-ish mediums
            a, b = rng.randint(20, 70), rng.randint(20, 70)
        else:                                # hard: base model ~always wrong -> drop candidates
            a, b = rng.randint(200, 900), rng.randint(200, 900)
        pool.append({"id": i, "q": f"{a} + {b}", "gold": a + b})
    return pool

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

def gen_group(model, prompt, gold, g):
    model.eval()
    enc = tok([prompt] * g, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(t) == gold else 0.0
               for t in tok.batch_decode(gens, skip_special_tokens=True)]
    return enc.input_ids[0], gens, rewards

def seq_logprob(model, pids, gen_row):
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, pids.shape[0] - 1:].sum()

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval(); c = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=PAD)
        if parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)) == gold:
            c += 1
    return c / len(evalset)

def run(arm, seed):
    rng = random.Random(seed); torch.manual_seed(seed)
    pool = make_pool(rng)
    evalset = [(p["q"], p["gold"]) for p in pool[::2][:HELDOUT_N]]   # held-out = solvable mediums
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ctrl = Controller(adaptive_G=(arm != "fixed_G"), drop=(arm == "adaptiveG_drop"))
    pre = heldout_acc(model, evalset)

    rollouts, step, zvf_hist, zvf_suppress = 0, 0, [], None
    while rollouts < BUDGET and not ctrl.stopped:
        step += 1
        g = ctrl.group_size()
        avail = [p for p in pool if p["id"] not in ctrl.dropped]
        if not avail:
            break
        batch = rng.sample(avail, min(BATCH_P, len(avail)))
        groups, per_prompt_zv, all_r = [], {}, []
        for p in batch:
            pids, gens, rewards = gen_group(model, prompt_of(p["q"]), p["gold"], g)
            rollouts += g
            all_r += rewards
            zerovar = (min(rewards) == max(rewards))
            per_prompt_zv[p["id"]] = zerovar
            if not zerovar:
                m = sum(rewards) / g; s = statistics.pvariance(rewards) ** 0.5
                adv = [(r - m) / (s + 1e-6) for r in rewards]
                groups.append((pids, gens, adv))
        # GRPO update on live groups (per-term backward = bounded activation memory)
        opt.zero_grad(set_to_none=True)
        n_terms = 0
        for pids, gens, adv in groups:
            for i, a in enumerate(adv):
                if a:
                    lp = seq_logprob(model, pids, gens[i])
                    if lp is not None:
                        (-a * lp).backward(); n_terms += 1
        if n_terms:
            opt.step()
        batch_zvf = 1 - len(groups) / len(batch)
        mean_r = sum(all_r) / len(all_r)
        zvf_hist.append(batch_zvf)
        ctrl.step(per_prompt_zv, batch_zvf, mean_r)
        roll = ctrl.rolling()
        if zvf_suppress is None and roll < 0.2:
            zvf_suppress = step
        print(f"[e6:{arm:14s} s{seed}] step={step:2d} G={g} ZVF={batch_zvf:.2f} "
              f"roll={roll:.2f} r={mean_r:.2f} roll_used={rollouts} dropped={len(ctrl.dropped)}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"arm": arm, "seed": seed, "heldout_pre": round(pre, 3), "heldout_post": round(post, 3),
           "heldout_delta": round(post - pre, 3), "mean_zvf": round(statistics.mean(zvf_hist), 3) if zvf_hist else None,
           "total_rollouts": rollouts, "opt_steps": step, "dropped": len(ctrl.dropped),
           "zvf_suppress_step": zvf_suppress, "auto_stopped": ctrl.stopped}
    del model, opt; torch.cuda.empty_cache()
    return out

results = []
for arm in ["fixed_G", "adaptiveG", "adaptiveG_drop"]:
    for seed in SEEDS:
        results.append(run(arm, seed))

agg = {}
for arm in ["fixed_G", "adaptiveG", "adaptiveG_drop"]:
    rs = [r for r in results if r["arm"] == arm]
    agg[arm] = {"mean_heldout_delta": round(statistics.mean(r["heldout_delta"] for r in rs), 3),
                "mean_zvf": round(statistics.mean(r["mean_zvf"] for r in rs if r["mean_zvf"] is not None), 3),
                "mean_rollouts": round(statistics.mean(r["total_rollouts"] for r in rs), 1),
                "mean_opt_steps": round(statistics.mean(r["opt_steps"] for r in rs), 1),
                "mean_dropped": round(statistics.mean(r["dropped"] for r in rs), 1)}
print("E6_RESULT " + json.dumps({"experiment": "E6_live_triage", "model": MODEL, "seeds": SEEDS,
                                "budget_rollouts": BUDGET, "by_arm": agg, "runs": results}), flush=True)
