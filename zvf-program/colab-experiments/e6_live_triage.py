"""E6 (Pillar 3, operationalization): does the live ZVF-triage controller beat a
fixed-G baseline at matched total-rollout budget?

The pool is built to be triage-relevant (a too-easy regime lets fixed-G win
outright and makes triage look neutral):
  * ~75% persistent dead-hard prompts (3-digit multiplication, p~0: zero variance,
    low reward) that only waste budget -> drop should remove them.
  * ~25% borderline learnable prompts (2-digit multiplication) where adaptive-G
    fishing for contrast can help.
  * Held-out = disjoint borderline set. Lower budget so saved rollouts matter.

The controller follows the package API: `step(rewards, group_ids) -> decision`
computes ZVF, per-prompt drop streaks, global auto-stop, and the smoothed-ZVF
adaptive group size internally (mirrors zvf_triage.controller.ZVFController;
vendored because colab run ships one file).

Three arms separate the two effects:
  fixed_G | adaptiveG (group size only) | adaptiveG_drop (size + drop + auto-stop)

Run:  colab run --gpu T4 --timeout 1800 e6_live_triage.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, re, random, statistics
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1]
POOL_N, FRAC_DEAD, BATCH_P, MAX_NEW, LR = 48, 0.75, 4, 128, 2e-6
G0, GMIN, GMAX = 4, 2, 12
BUDGET = 420           # total rollouts per arm (matched; lower so savings matter)
HELDOUT_N = 24
WINDOW, DROP_K, STOP_K, ZVF_MAX, EPS_LO = 5, 3, 4, 0.85, 0.05
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

FEWSHOT = [
    {"role": "user", "content": "Compute 3 + 4. Reason briefly, then put the final integer after '####'."},
    {"role": "assistant", "content": "3 + 4 = 7.\n#### 7"},
]

# ---- faithful vendored controller (zvf_triage.controller.ZVFController) ----
def adaptive_fn(z):
    z = min(max(float(z), 0.0), 1.0)
    return 0.5 + (z / 0.4) * 0.5 if z < 0.4 else 1.0 + ((z - 0.4) / 0.6) * 1.0

def _group_zerovar(rewards, gids):
    groups = {}
    for r, g in zip(rewards, gids):
        groups.setdefault(g, []).append(r)
    return {g: (min(v) == max(v)) for g, v in groups.items()}

class Decision:
    __slots__ = ("zvf", "mean_reward", "group_size", "dropped_prompts", "auto_stop", "regime")
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

class ZVFController:
    """Mirrors the package: step(rewards, group_ids) -> Decision, internal state."""
    def __init__(self, adaptive_G):
        self.adaptive_G = adaptive_G
        self.hist, self.streak, self.dropped = [], {}, set()
        self.global_streak, self.stopped = 0, False
    def _rolling(self):
        w = self.hist[-WINDOW:]
        return sum(w) / len(w) if w else 0.0
    def _group_size(self):
        if not self.adaptive_G:
            return G0
        return int(round(min(max(G0 * adaptive_fn(self._rolling()), GMIN), GMAX)))
    def step(self, rewards, group_ids):
        keep = [(r, g) for r, g in zip(rewards, group_ids) if g not in self.dropped]
        if not keep:
            self.hist.append(1.0)
            return Decision(zvf=1.0, mean_reward=0.0, group_size=self._group_size(),
                            dropped_prompts=[], auto_stop=False, regime="all_dropped")
        rs, gs = zip(*keep)
        pp = _group_zerovar(rs, gs)
        batch_zvf = sum(pp.values()) / len(pp)
        mean_reward = sum(rs) / len(rs)
        self.hist.append(batch_zvf)
        newly = []
        for gid, zv in pp.items():
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
        auto_stop = self.global_streak >= STOP_K and mean_reward < EPS_LO
        if auto_stop:
            self.stopped = True
        regime = ("cold_start_collapse" if batch_zvf > ZVF_MAX and mean_reward < EPS_LO
                  else "saturation" if batch_zvf > ZVF_MAX and mean_reward > 1 - EPS_LO
                  else "exploitable_contrast")
        return Decision(zvf=batch_zvf, mean_reward=mean_reward, group_size=self._group_size(),
                        dropped_prompts=newly, auto_stop=auto_stop, regime=regime)

# ---- task / harness ----
def make_pool(rng):
    # DEAD = 3-digit multiplication (0.5B ~never correct -> persistent zero-variance,
    # low reward -> drop candidates). BORDERLINE = 2-digit multiplication: 0.5B gets it
    # ~0.3 with headroom to LEARN (3-digit addition is aced natively -> no headroom).
    pool = []
    n_dead = int(POOL_N * FRAC_DEAD)
    for i in range(POOL_N):
        if i < n_dead:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            pool.append({"id": i, "q": f"{a} * {b}", "gold": a * b, "kind": "dead"})
        else:
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            pool.append({"id": i, "q": f"{a} * {b}", "gold": a * b, "kind": "borderline"})
    rng.shuffle(pool)
    return pool

def heldout_set(rng):                        # DISJOINT borderline 2-digit multiplications
    return [(f"{a} * {b}", a * b) for a, b in
            [(rng.randint(11, 99), rng.randint(11, 99)) for _ in range(HELDOUT_N)]]

def prompt_of(q):
    msgs = FEWSHOT + [{"role": "user",
                       "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text):
    if "####" not in text:            # no marker -> not parseable (don't grab question digits)
        return None
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
    evalset = heldout_set(rng)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ctrl = ZVFController(adaptive_G=(arm != "fixed_G"))
    act_drop = (arm == "adaptiveG_drop")
    pre = heldout_acc(model, evalset)

    rollouts, step, zvf_hist, g = 0, 0, [], G0
    while rollouts < BUDGET and not (act_drop and ctrl.stopped):
        step += 1
        avail = [p for p in pool if not (act_drop and p["id"] in ctrl.dropped)]
        if not avail:
            break
        batch = rng.sample(avail, min(BATCH_P, len(avail)))
        flat_r, flat_g, groups = [], [], []
        for p in batch:
            pids, gens, rewards = gen_group(model, prompt_of(p["q"]), p["gold"], g)
            rollouts += g
            flat_r += rewards; flat_g += [p["id"]] * g
            if min(rewards) != max(rewards):
                m = sum(rewards) / g; s = statistics.pvariance(rewards) ** 0.5
                groups.append((pids, gens, [(r - m) / (s + 1e-6) for r in rewards]))
        # GRPO update (per-term backward = bounded memory)
        opt.zero_grad(set_to_none=True); n_terms = 0
        for pids, gens, adv in groups:
            for i, a in enumerate(adv):
                if a:
                    lp = seq_logprob(model, pids, gens[i])
                    if lp is not None:
                        (-a * lp).backward(); n_terms += 1
        if n_terms:
            opt.step()
        dec = ctrl.step(flat_r, flat_g)        # FAITHFUL controller call
        g = dec.group_size                      # adapt G for next step
        zvf_hist.append(dec.zvf)
        print(f"[e6:{arm:14s} s{seed}] step={step:2d} G={g} ZVF={dec.zvf:.2f} "
              f"roll={ctrl._rolling():.2f} r={dec.mean_reward:.2f} used={rollouts} "
              f"dropped={len(ctrl.dropped)} regime={dec.regime}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"arm": arm, "seed": seed, "heldout_pre": round(pre, 3), "heldout_post": round(post, 3),
           "heldout_delta": round(post - pre, 3), "mean_zvf": round(statistics.mean(zvf_hist), 3) if zvf_hist else None,
           "total_rollouts": rollouts, "opt_steps": step, "dropped": len(ctrl.dropped),
           "auto_stopped": ctrl.stopped}
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
                                "budget_rollouts": BUDGET, "frac_dead": FRAC_DEAD,
                                "by_arm": agg, "runs": results}), flush=True)
