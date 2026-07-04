#!/usr/bin/env python3
"""QP7 adaptive-G: two sequential GRPO arms on GSM8K (Tinker stack).

Arm A: fixed G=4. Arm B: adaptive-G controller (start G=4; step-ZVF>0.5 ->
escalate 4->6->8 cap 8; ZVF<0.2 -> de-escalate one level). Same 128-prompt
GSM8K train pool + seed for both arms, 16 steps each. Feeds paper P7.

Usage:
  python3 qp7_adaptive_g_20260704.py --smoke
  python3 qp7_adaptive_g_20260704.py
"""
import argparse, json, os, re, sys, time, traceback

import tinker
import tinker.types as T
from datasets import load_dataset

G_LADDER = [4, 6, 8]

def reward(response, answer):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        try:
            if abs(float(all_nums[-1].replace(",", "")) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0

def pvar(g):
    if not g:
        return 0.0
    m = sum(g) / len(g)
    return sum((x - m) ** 2 for x in g) / len(g)

def write_manifest(args, out_dir):
    manifest = {
        "experiment": "qp7-adaptive-g",
        "date": "2026-07-04",
        "model": args.model,
        "loss_form": ("Tinker built-in loss_fn='importance_sampling': token-level "
                      "IS-weighted policy gradient; advantages = r_i - group_mean(r) "
                      "(GRPO-style mean baseline, no std normalization, no clipping "
                      "param exposed client-side)"),
        "ref_policy_kl_handling": ("none: no KL penalty or reference-policy term; "
                                   "pure reward with group-mean baseline"),
        "sampler_backend_precision": "unknown/closed-stack (Tinker managed sampler)",
        "per_step_zvf_path": ("computed client-side each step: fraction of the step's "
                              "prompt groups whose G rewards have zero variance; "
                              "written to zvf column of qp7_adaptive.tsv and W&B"),
        "group_size_schedule": ("arm A: fixed G=4 all 16 steps; arm B: start G=4, "
                                "after each step ZVF>0.5 escalates one ladder level "
                                "(4->6->8, cap 8), ZVF<0.2 de-escalates one level "
                                "(floor 4); ladder [4,6,8]"),
        "heldout_split": ("none: training reward only, on a fixed 128-prompt pool "
                          "from GSM8K train (seed 42 shuffle); no heldout eval in "
                          "this quick run"),
        "decontamination_notes": ("GSM8K train via HF openai/gsm8k 'main'; no "
                                  "additional decontamination performed; base model "
                                  "pretraining overlap with GSM8K unknown/closed-stack"),
        "config": vars(args),
    }
    path = os.path.join(out_dir, "qp7_adaptive_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path

def run_arm(arm, svc, args, pool, tsv_f, wandb_run, global_state):
    """One GRPO arm. arm in {'A','B'}. Returns per-step records."""
    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    tok = tc.get_tokenizer()
    processed = []
    for i, (q, ans) in enumerate(pool):
        prompt = (f"<|im_start|>system\nYou are a math assistant. Solve the problem "
                  f"step by step, then give your final numerical answer inside "
                  f"\\boxed{{}}.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        pid = tok.encode(prompt, add_special_tokens=False)[: args.max_prompt_tokens]
        processed.append((i, pid, ans))

    g_idx = 0  # start G=4 for both arms
    cum_rollouts = 0
    records = []
    for step in range(args.steps):
        t0 = time.time()
        G = 4 if arm == "A" else G_LADDER[g_idx]
        w = tc.save_weights_for_sampler(name=f"qp7{arm}_s{step}").result()
        sc = tc.create_sampling_client(model_path=w.path)
        start = (step * args.prompts_per_step) % len(processed)
        batch = [processed[(start + i) % len(processed)]
                 for i in range(args.prompts_per_step)]
        sp = T.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp,
                              top_p=args.top_p)
        futs = [(oi, pid, ans,
                 sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                           sampling_params=sp))
                for oi, pid, ans in batch]

        all_data, group_pvars, all_rewards = [], [], []
        for oi, pid, ans, fut in futs:
            try:
                resp = fut.result()
            except Exception:
                try:
                    resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G,
                                     sampling_params=sp).result()
                except Exception:
                    traceback.print_exc()
                    continue
            rews = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
                    for r in resp.sequences]
            cum_rollouts += len(rews)
            all_rewards.extend(rews)
            group_pvars.append(pvar(rews))
            mean_r = sum(rews) / len(rews)
            for r_seq, r in zip(resp.sequences, rews):
                a = r - mean_r
                if a == 0:
                    continue
                rid = list(r_seq.tokens)
                full = pid + rid
                tgt = full[1:]
                lp = [0.0] * (len(pid) - 1) + list(r_seq.logprobs)
                adv = [0.0] * (len(pid) - 1) + [a] * len(rid)
                all_data.append(T.Datum(
                    model_input=T.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=tgt, dtype="int64", shape=[len(tgt)]),
                        "logprobs": T.TensorData(data=lp, dtype="float32", shape=[len(lp)]),
                        "advantages": T.TensorData(data=adv, dtype="float32", shape=[len(adv)]),
                    }))

        loss_val = None
        if all_data:
            try:
                res = tc.forward_backward(data=all_data, loss_fn="importance_sampling").result()
                tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9,
                                           beta2=0.95, eps=1e-8)).result()
                try:
                    loss_val = res.metrics.get("loss", None)
                except AttributeError:
                    loss_val = getattr(res.metrics, "loss", None)
            except Exception as e:
                print(f"[qp7:{arm}] step {step} optim failed: {e}", flush=True)

        n_grp = len(group_pvars)
        zvf = sum(1 for v in group_pvars if v == 0.0) / n_grp if n_grp else 1.0
        rmean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        dt = time.time() - t0

        tsv_f.write(f"{arm}\t{step}\t{G}\t{rmean:.6f}\t{zvf:.6f}\t{cum_rollouts}\t"
                    f"{loss_val}\t{dt:.1f}\n")
        tsv_f.flush(); os.fsync(tsv_f.fileno())
        if wandb_run is not None:
            import wandb
            wandb.log({f"arm{arm}/G": G, f"arm{arm}/reward_mean": rmean,
                       f"arm{arm}/zvf": zvf, f"arm{arm}/cum_rollouts": cum_rollouts,
                       f"arm{arm}/loss": loss_val, f"arm{arm}/step": step,
                       f"arm{arm}/step_seconds": dt},
                      step=global_state["gs"])
        global_state["gs"] += 1
        records.append((step, G, rmean, zvf, cum_rollouts))
        print(f"[qp7:{arm}] step {step+1}/{args.steps} G={G} reward={rmean:.3f} "
              f"zvf={zvf:.3f} cum={cum_rollouts} ({dt:.0f}s)", flush=True)

        if arm == "B":  # adaptive controller
            if zvf > 0.5 and g_idx < len(G_LADDER) - 1:
                g_idx += 1
            elif zvf < 0.2 and g_idx > 0:
                g_idx -= 1
    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--wandb-project", default="tinker-new-research")
    p.add_argument("--run-name", default="qp7-adaptive-g-20260704")
    p.add_argument("--out-dir",
                   default="/home/claude/tinker-rl-lab/experiments/results/quick_20260704")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    tsv_name = "qp7_adaptive.tsv"
    if args.smoke:
        args.steps = 1
        args.prompts_per_step = 4
        args.max_tokens = 256
        tsv_name = "qp7_adaptive_smoke.tsv"

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set")

    print("Loading GSM8K train...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)
    pool = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        pool.append((row["question"], m.group(1).replace(",", "").strip()))
        if len(pool) >= args.pool:
            break
    print(f"Pool: {len(pool)} prompts (seed {args.seed})", flush=True)

    wandb_run = None
    if not args.smoke:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                               config=vars(args), tags=["qp7", "adaptive-g", "20260704"])
        print(f"wandb: {wandb_run.entity}/{wandb_run.project}/{wandb_run.id}", flush=True)

    manifest_path = write_manifest(args, args.out_dir)
    print(f"Manifest: {manifest_path}", flush=True)

    svc = tinker.ServiceClient()
    tsv_path = os.path.join(args.out_dir, tsv_name)
    hdr = not os.path.exists(tsv_path)
    tsv_f = open(tsv_path, "a")
    if hdr:
        tsv_f.write("arm\tstep\tG\treward_mean\tzvf\tcum_rollouts\tloss\tstep_seconds\n")
        tsv_f.flush()

    gstate = {"gs": 0}
    rec_a = run_arm("A", svc, args, pool, tsv_f, wandb_run, gstate)
    rec_b = run_arm("B", svc, args, pool, tsv_f, wandb_run, gstate)
    tsv_f.close()

    if wandb_run is not None:
        import wandb
        final = {}
        if rec_a:
            final["armA/final_reward"] = rec_a[-1][2]
            final["armA/total_rollouts"] = rec_a[-1][4]
        if rec_b:
            final["armB/final_reward"] = rec_b[-1][2]
            final["armB/total_rollouts"] = rec_b[-1][4]
        wandb.log(final)
        wandb.finish()
    print("DONE", flush=True)
    sys.exit(0 if (rec_a and rec_b) else 1)

if __name__ == "__main__":
    main()
