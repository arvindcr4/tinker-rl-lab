"""Weakness-addressing campaign — PROCESS-parallel Tinker runs (each its own W&B run).

Directly targets the top adversarial-review weaknesses:
  - single-seed / no power  -> MULTI-SEED (>=3 seeds) matched comparisons
  - training-reward only     -> held-out accuracy every run
  - no matched baseline      -> baseline vs curriculum, same model/seed/steps
  - P2/P3 openings           -> curriculum (mixed-variance) + group-size

Each config runs in its OWN process (multiprocessing) => independent wandb run
(fixes the thread-unsafe wandb bug from parallel_sweep.py). W&B group: campaign.
Usage: python campaign.py --steps 8 --seeds 0,1,2 --workers 4
"""
import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

def run_one(cfg):
    import torch, tinker, tinker.types as T, wandb
    from transformers import AutoTokenizer
    from datasets import load_dataset
    name, mode, G, seed, steps, heldout_n, model, rank, lr, batch, oversample_cap = cfg
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
            l=torch.stack([(-advs[i]*lp[i].sum()) for i in range(len(lp))]).mean(); return l, {"loss": l.item()}
        return _fn
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model, rank=rank)
    sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_s0").result().path)
    def held():
        c = 0
        for pt, ans in HO:
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
            c += reward(tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True), ans)
        return c/len(HO)
    run = wandb.init(project="rlvr-openings", name=name, group="campaign",
                     config=dict(mode=mode, G=G, seed=seed, steps=steps, model=model),
                     tags=["campaign", mode, f"seed{seed}", f"G{G}"], reinit=True)
    ho0 = held(); zero_loss = 0; sampled = 0; accepted = 0; rewards = []
    for step in range(steps):
        data, advs, br = [], [], []; slots = 0; att = 0
        while slots < batch and att < batch*oversample_cap:
            pt, ans = random.choice(POOL); att += 1; sampled += 1
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            rr = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rr)/len(rr)
            if mode == "curriculum" and mr in (0.0, 1.0):
                continue
            slots += 1; accepted += 1
            sd = (sum((x-mr)**2 for x in rr)/len(rr))**0.5 + 1e-8; br.extend(rr)
            for rs, a in zip(resp.sequences, [(x-mr)/sd for x in rr]):
                fid = pid + list(rs.tokens); tid = fid[1:]+[0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs.append(a)
        if not data: rewards.append(0.0); continue
        res = tc.forward_backward_custom(data=data, loss_fn=mkloss(advs)).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        lv = res.metrics.get("loss", 0.0)
        if abs(lv) < 1e-6: zero_loss += 1
        avg = sum(br)/len(br) if br else 0.0; rewards.append(avg)
        wandb.log({"reward": avg, "loss": lv, "collapsed": int(abs(lv)<1e-6)}, step=step)
        sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_s{step+1}").result().path)
    ho1 = held()
    out = dict(name=name, mode=mode, G=G, seed=seed, zero_loss_frac=zero_loss/steps,
               oversample=sampled/max(accepted,1), heldout_before=ho0, heldout_after=ho1, heldout_gain=ho1-ho0)
    for k, v in out.items():
        run.summary[k] = v
    wandb.finish()
    print("DONE", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--heldout", type=int, default=12)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = []
    # matched baseline vs curriculum across seeds (fixes single-seed weakness) at G=4
    for s in seeds:
        for mode in ["baseline", "curriculum"]:
            cfgs.append((f"{mode}-G4-s{s}", mode, 4, s, a.steps, a.heldout, a.model, a.rank, a.lr, a.batch, 8))
    # group-size at seed 0 (P3), baseline mode
    for G in [2, 8, 16]:
        cfgs.append((f"baseline-G{G}-s0", "baseline", G, 0, a.steps, a.heldout, a.model, a.rank, a.lr, a.batch, 8))
    print(f"CAMPAIGN: {len(cfgs)} runs, {a.workers} parallel, seeds={seeds}", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_one, cfgs)
    os.makedirs("experiments/results/campaign", exist_ok=True)
    json.dump(results, open("experiments/results/campaign/results.json", "w"), indent=2)
    print("=== CAMPAIGN COMPLETE ===", flush=True)
    print(json.dumps(results, indent=2))
