"""P4 opening #10 — length bias & the surprise-weighted loss.

Standard GRPO loss -adv * sum_t logprob_t is length-biased (longer completions get
larger gradient magnitude). Two known/proposed alternatives:
  - sum     : standard GRPO (length-biased baseline)
  - mean    : length-normalized (Dr.GRPO-style fix)     loss = -adv * mean_t logprob_t
  - surprise: weight each token by its (detached) surprise so gradient concentrates on
              decision tokens, not boilerplate.  w_t = softmax(-logprob.detach());
              loss = -adv * sum_t w_t * logprob_t
Question: do these change held-out accuracy and/or the completion-length trajectory
(the length bias)? Multi-seed, process-parallel, matched steps.
"""
import os, re, json, random, argparse, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

def run_one(cfg):
    import torch, tinker, tinker.types as T, wandb
    from transformers import AutoTokenizer
    from datasets import load_dataset
    name, loss_mode, seed, steps, G, heldout_n, model, rank, lr, batch = cfg
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
            terms = []
            for i in range(len(lp)):
                l = lp[i]
                if loss_mode == "mean":
                    t = -advs[i] * l.mean()
                elif loss_mode == "surprise":
                    w = torch.softmax(-l.detach(), dim=0)   # emphasise high-surprise tokens
                    t = -advs[i] * (w * l).sum()
                else:  # sum (standard GRPO, length-biased)
                    t = -advs[i] * l.sum()
                terms.append(t)
            loss = torch.stack(terms).mean()
            return loss, {"loss": loss.item()}
        return _fn
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model, rank=rank)
    sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_0").result().path)
    def held():
        c = 0; L = 0
        for pt, ans in HO:
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)).result()
            toks = list(resp.sequences[0].tokens); L += len(toks)
            c += reward(tok.decode(toks, skip_special_tokens=True), ans)
        return c/len(HO), L/len(HO)
    run = wandb.init(project="rlvr-openings", name=name, group="p4-surprise",
                     config=dict(loss_mode=loss_mode, seed=seed, steps=steps, G=G, model=model),
                     tags=["P4", loss_mode, f"seed{seed}"], reinit=True)
    ho0, len0 = held(); rewards = []; mean_lens = []
    for step in range(steps):
        data, advs, br, slens = [], [], [], []
        for _ in range(batch):
            pt, ans = random.choice(POOL)
            pid = tok.encode(pt, add_special_tokens=False)[:1024]
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                             sampling_params=T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)).result()
            rr = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            slens.extend(len(r.tokens) for r in resp.sequences)
            mr = sum(rr)/len(rr)
            if mr in (0.0, 1.0):   # need variance for a gradient signal
                continue
            sd = (sum((x-mr)**2 for x in rr)/len(rr))**0.5 + 1e-8; br.extend(rr)
            for rs, a in zip(resp.sequences, [(x-mr)/sd for x in rr]):
                fid = pid + list(rs.tokens); tid = fid[1:]+[0]
                data.append(T.Datum(model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}))
                advs.append(a)
        ml = sum(slens)/len(slens) if slens else 0; mean_lens.append(ml)
        if not data:
            wandb.log({"mean_len": ml}, step=step); continue
        res = tc.forward_backward_custom(data=data, loss_fn=mkloss(advs)).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        avg = sum(br)/len(br) if br else 0.0; rewards.append(avg)
        wandb.log({"reward": avg, "loss": res.metrics.get("loss", 0.0), "mean_len": ml}, step=step)
        sc = tc.create_sampling_client(model_path=tc.save_weights_for_sampler(name=f"{name}_{step+1}").result().path)
    ho1, len1 = held()
    out = dict(name=name, loss_mode=loss_mode, seed=seed, heldout_before=ho0, heldout_after=ho1,
               heldout_gain=ho1-ho0, len_before=len0, len_after=len1, len_delta=len1-len0,
               train_len_first=mean_lens[0] if mean_lens else 0, train_len_last=mean_lens[-1] if mean_lens else 0)
    for k, v in out.items(): run.summary[k] = v
    wandb.finish(); print("DONE", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--heldout", type=int, default=12)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = [(f"{lm}-s{s}", lm, s, a.steps, a.G, a.heldout, a.model, a.rank, a.lr, a.batch)
            for s in seeds for lm in ["sum", "mean", "surprise"]]
    print(f"P4-SURPRISE: {len(cfgs)} runs ({len(seeds)} seeds x 3 loss modes), {a.workers} parallel", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(run_one, cfgs)
    os.makedirs("experiments/results/p4_surprise", exist_ok=True)
    json.dump(results, open("experiments/results/p4_surprise/results.json", "w"), indent=2)
    print("=== P4-SURPRISE COMPLETE ===", flush=True); print(json.dumps(results, indent=2))
