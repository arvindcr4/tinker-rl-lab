"""P2/P3 opening #9 — the token-budget-optimal test (the 'better lever').

The campaign showed naive curriculum eliminates gradient waste but costs ~5-6x the
sampling and does NOT beat baseline. That was at UNEQUAL cost. The honest question:
at a MATCHED rollout-token budget, does difficulty-targeting (train only on mixed-
variance groups) beat training-on-everything? I.e. are few gradient-bearing steps
worth more than many mostly-collapsed steps, per token?

Each arm samples until it has spent TOKEN_BUDGET rollout tokens:
  - baseline:   train on every group (incl. collapsed -> zero-grad steps)
  - curriculum: skip collapsed groups; only spend optim steps on gradient-bearing data
Both consume the SAME tokens. Compare held-out gain. Multi-seed. Process-parallel.
"""
import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

def run_one(cfg):
    import torch, tinker, tinker.types as T, wandb
    from transformers import AutoTokenizer
    from datasets import load_dataset
    name, mode, seed, budget, G, heldout_n, model, rank, lr, batch = cfg
    random.seed(seed); torch.manual_seed(seed)
    SYS = "You are a math assistant. Solve step by step, then give the final numerical answer inside \\boxed{}."
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ex = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m: continue
        ex.append((f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n",
                   m.group(1).replace(",", "").strip()))
    random.shuffle(ex); HO = ex[:heldout_n]; POOL = ex[heldout_n:]
    def reward(resp, ans):
        r = resp.strip()
        for b in re.findall(r"\\boxed\{([^}]+)\}", r):
            bc = b.strip().replace(",", "").replace(" ", "")
            try:
                if abs(float(bc)-float(ans)) < 0.01: return 1.0
            except:
                if bc == ans: return 1.0
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", r)
        if nums:
            try:
                if abs(float(nums[-1].replace(",",""))-float(ans)) < 0.01: return 1.0
            except: pass
        return 0.0
    def mkloss(advs):
        def _fn(data, lp):
            l = torch.stack([(-advs[i]*lp[i].sum()) for i in range(len(lp))]).mean()
            return l, {"loss": l.item()}
        return _fn
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model, rank=rank)
    sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_0").result().path)
    def held():
        c = 0
        for pt, ans in HO:
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
            c += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
        return c/len(HO)
    run = wandb.init(project="rlvr-openings", name=name, group="token-budget",
                     config=dict(mode=mode, seed=seed, budget=budget, G=G, model=model),
                     tags=["token-budget", mode, f"seed{seed}"], reinit=True)
    ho0 = held()
    tokens = 0; optim_steps = 0; grad_steps = 0; skipped = 0; step_i = 0; rewards = []
    while tokens < budget:
        data, advs, br = [], [], []
        for _ in range(batch):
            pt, ans = random.choice(POOL)
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            tokens += sum(len(r.tokens) for r in resp.sequences)
            rr = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rr)/len(rr)
            if mode == "curriculum" and mr in (0.0, 1.0):
                skipped += 1; continue
            sd = (sum((x-mr)**2 for x in rr)/len(rr))**0.5 + 1e-8; br.extend(rr)
            for rs, a in zip(resp.sequences, [(x-mr)/sd for x in rr]):
                fid = pid + list(rs.tokens); tid = fid[1:]+[0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs.append(a)
        if not data:
            continue
        res = tc.forward_backward_custom(data=data, loss_fn=mkloss(advs)).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        optim_steps += 1
        lv = res.metrics.get("loss", 0.0)
        if abs(lv) > 1e-6: grad_steps += 1
        avg = sum(br)/len(br) if br else 0.0; rewards.append(avg)
        wandb.log({"reward": avg, "loss": lv, "tokens": tokens}, step=step_i); step_i += 1
        sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_{step_i}").result().path)
    ho1 = held()
    out = dict(name=name, mode=mode, seed=seed, budget=budget, tokens_spent=tokens,
               optim_steps=optim_steps, grad_bearing_steps=grad_steps, groups_skipped=skipped,
               heldout_before=ho0, heldout_after=ho1, heldout_gain=ho1-ho0)
    for k, v in out.items(): run.summary[k] = v
    wandb.finish(); print("DONE", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--budget", type=int, default=40000)   # matched rollout-token budget per arm
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--heldout", type=int, default=12)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = [(f"{mode}-b{a.budget}-s{s}", mode, s, a.budget, a.G, a.heldout, a.model, a.rank, a.lr, a.batch)
            for s in seeds for mode in ["baseline", "curriculum"]]
    print(f"TOKEN-BUDGET: {len(cfgs)} runs, budget={a.budget} tok/arm, {a.workers} parallel", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_one, cfgs)
    os.makedirs("experiments/results/token_budget", exist_ok=True)
    json.dump(results, open("experiments/results/token_budget/results.json", "w"), indent=2)
    print("=== TOKEN-BUDGET COMPLETE ===", flush=True); print(json.dumps(results, indent=2))
