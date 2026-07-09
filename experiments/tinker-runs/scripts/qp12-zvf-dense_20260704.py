#!/usr/bin/env python3
"""
Experiment qp12-zvf-dense: dense ZVF + per-group reward tensor logging under plain GRPO.
Feeds P1 (scaling), P2 (ZVF), PCD analyses. Adapted from n2_reward_tensor_20260704.py.

Usage:
  python qp12-zvf-dense_20260704.py --smoke
  python qp12-zvf-dense_20260704.py            # full: seeds 0,1 x 16 steps
"""

import argparse
import json
import os
import random
import re
import sys
import traceback

import tinker
import tinker.types as T
from datasets import load_dataset

EXP = "qp12-zvf-dense"
RES_DIR = "/home/claude/tinker-rl-lab/experiments/results/quick_20260704"


def reward(response, answer):
    response = response.strip()
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
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


def grpo_advantages(rews):
    n = len(rews)
    m = sum(rews) / n
    var = sum((x - m) ** 2 for x in rews) / n
    sd = var ** 0.5
    if sd == 0.0:
        return [0.0] * n
    return [(x - m) / sd for x in rews]


def write_manifest(args):
    manifest = {
        "exp": EXP,
        "date": "2026-07-04",
        "model": args.model,
        "lora_rank": args.rank,
        "loss_form": ("token-level importance_sampling surrogate (Tinker loss_fn='importance_sampling'); "
                      "per-token advantage = group-standardized reward (r - mean)/std, GRPO-style; "
                      "zero-variance groups get all-zero advantage and are dropped from the batch"),
        "ref_policy_kl_handling": ("none: no KL penalty and no reference-policy term in the objective; "
                                   "behavior logprobs come from the sampler snapshot refreshed every step"),
        "sampler_backend_precision": "unknown/closed-stack (Tinker managed sampling backend; precision not exposed)",
        "per_step_zvf_path": f"{RES_DIR}/{EXP}.tsv (column 'zvf'); per-group reward tensors in {RES_DIR}/qp12_group_tensors.jsonl",
        "group_size_schedule": f"constant G={args.group} for all steps, both seeds (no adaptation)",
        "heldout_split": ("none evaluated in-run; training prompts are the first 128 of a seed-shuffled "
                          "openai/gsm8k train split; gsm8k test split untouched"),
        "decontamination_notes": ("no additional decontamination performed beyond using the standard "
                                  "openai/gsm8k train split as released; base-model pretraining contamination "
                                  "unknown/closed-stack"),
        "config": {k: v for k, v in vars(args).items()},
    }
    path = os.path.join(RES_DIR, f"{EXP}_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {path}")


def run_seed(args, svc, examples, seed, tsv_path, jsonl_path, wandb, global_step0):
    rng = random.Random(seed)
    ex = list(examples)
    rng.shuffle(ex)
    pool = ex[:args.pool]

    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    tok = tc.get_tokenizer()

    processed = []
    for i, (prompt_text, ans) in enumerate(pool):
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > args.max_prompt_tokens:
            pid = pid[:args.max_prompt_tokens]
        processed.append((i, pid, ans))
    print(f"[seed {seed}] pool size: {len(processed)}")

    jsonl_f = open(jsonl_path, "a")
    write_header = not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0
    tsv_f = open(tsv_path, "a")
    if write_header:
        tsv_f.write("seed\tstep\treward_mean\tzvf\tfrac_all_zero\tfrac_all_one\tpcd\tmean_len\tn_groups\tloss\n")
        tsv_f.flush()

    sp = T.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)
    G = args.group

    for step in range(args.steps):
        w = tc.save_weights_for_sampler(name=f"s{seed}_step{step}").result()
        sc = tc.create_sampling_client(model_path=w.path)

        start_idx = (step * args.prompts_per_step) % len(processed)
        batch = [processed[(start_idx + i) % len(processed)] for i in range(args.prompts_per_step)]

        futs = [sc.sample(T.ModelInput.from_ints(pid), num_samples=G, sampling_params=sp)
                for _, pid, _ in batch]

        all_data = []
        groups = []  # (prompt_idx, rewards, lengths, advantages)
        for (orig_idx, pid, ans), fut in zip(batch, futs):
            try:
                resp = fut.result()
            except Exception:
                try:
                    resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=G, sampling_params=sp).result()
                except Exception:
                    traceback.print_exc()
                    continue
            rews = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            lens = [len(list(r.tokens)) for r in resp.sequences]
            advs = grpo_advantages(rews)
            groups.append((orig_idx, rews, lens, advs))

            for r_seq, a in zip(resp.sequences, advs):
                if a == 0:
                    continue
                rid = list(r_seq.tokens)
                lp = list(r_seq.logprobs)
                full = pid + rid
                target_tokens = full[1:]
                logprobs = [0.0] * (len(pid) - 1) + lp
                advantages = [0.0] * (len(pid) - 1) + [a] * len(rid)
                assert len(target_tokens) == len(logprobs) == len(advantages) == len(full) - 1
                all_data.append(T.Datum(
                    model_input=T.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
                        "logprobs": T.TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)]),
                        "advantages": T.TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
                    }))

        loss_val = None
        if all_data:
            try:
                result = tc.forward_backward(data=all_data, loss_fn="importance_sampling").result()
                tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
                try:
                    loss_val = result.metrics.get("loss", None)
                except AttributeError:
                    loss_val = getattr(result.metrics, "loss", None)
            except Exception as e:
                print(f"[seed {seed}] step {step} optim failed: {e}")

        n_groups = len(groups)
        all_rewards = [r for _, rews, _, _ in groups for r in rews]
        all_lens = [l for _, _, lens, _ in groups for l in lens]
        zvf = sum(1 for _, rews, _, _ in groups if len(set(rews)) == 1) / n_groups if n_groups else 0.0
        frac_all_zero = sum(1 for _, rews, _, _ in groups if all(r == 0.0 for r in rews)) / n_groups if n_groups else 0.0
        frac_all_one = sum(1 for _, rews, _, _ in groups if all(r == 1.0 for r in rews)) / n_groups if n_groups else 0.0
        phats = [sum(rews) / len(rews) for _, rews, _, _ in groups]
        pcd = ((G - 1) / G) * (sum(p * (1 - p) for p in phats) / n_groups) if n_groups else 0.0
        rmean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        lmean = sum(all_lens) / len(all_lens) if all_lens else 0.0

        jsonl_f.write(json.dumps({
            "exp": EXP, "seed": seed, "step": step, "group_size": G,
            "prompt_indices": [g[0] for g in groups],
            "rewards": [g[1] for g in groups],
            "lengths": [g[2] for g in groups],
            "advantages": [g[3] for g in groups],
            "zvf": zvf, "pcd": pcd, "reward_mean": rmean, "loss": loss_val,
        }) + "\n")
        jsonl_f.flush()
        os.fsync(jsonl_f.fileno())

        tsv_f.write(f"{seed}\t{step}\t{rmean}\t{zvf}\t{frac_all_zero}\t{frac_all_one}\t{pcd}\t{lmean}\t{n_groups}\t{loss_val}\n")
        tsv_f.flush()
        os.fsync(tsv_f.fileno())

        if wandb is not None:
            wandb.log({
                f"s{seed}/reward_mean": rmean, f"s{seed}/zvf": zvf, f"s{seed}/pcd": pcd,
                f"s{seed}/frac_all_zero": frac_all_zero, f"s{seed}/frac_all_one": frac_all_one,
                f"s{seed}/mean_len": lmean, f"s{seed}/loss": loss_val,
                "seed": seed, "seed_step": step,
            }, step=global_step0 + step)

        print(f"[{EXP} s{seed}] {step+1:3d}/{args.steps} | reward {rmean:.3f} | zvf {zvf:.3f} | pcd {pcd:.4f}", flush=True)

    jsonl_f.close()
    tsv_f.close()
    return global_step0 + args.steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--wandb-project", default="tinker-new-research")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 1
        args.prompts_per_step = 4
        args.group = 4
        args.pool = 16
        args.max_tokens = 128
        args.seeds = "0"
        args.no_wandb = True

    os.makedirs(RES_DIR, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    prefix = "smoke_" if args.smoke else ""
    tsv_path = os.path.join(RES_DIR, f"{prefix}{EXP}.tsv")
    jsonl_path = os.path.join(RES_DIR, f"{prefix}qp12_group_tensors.jsonl")

    if not args.smoke:
        write_manifest(args)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    SYS = ("You are a math assistant. Solve the problem step by step, then give your final "
           "numerical answer inside \\boxed{}.")
    examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        prompt = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        examples.append((prompt, ans))
    print(f"GSM8K examples parsed: {len(examples)}")

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, name=f"{EXP}-20260704",
                   config=vars(args), tags=["qp12", "zvf-dense", "20260704"])

    svc = tinker.ServiceClient(base_url=None)

    gstep = 0
    done = 0
    for seed in seeds:
        try:
            gstep = run_seed(args, svc, examples, seed, tsv_path, jsonl_path, wandb, gstep)
            done += 1
        except Exception:
            print(f"Seed {seed} failed:")
            traceback.print_exc()

    if wandb is not None:
        wandb.finish()
    sys.exit(0 if done > 0 else 1)


if __name__ == "__main__":
    main()
