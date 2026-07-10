#!/usr/bin/env python3
"""
Experiment N2 "reward-tensor-instrumented short GRPO-variant runs" on the Tinker RL API.

Usage:
  python n2_reward_tensor_20260704.py --smoke
  python n2_reward_tensor_20260704.py --model Qwen/Qwen3.5-4B --methods "grpo,aero,gift,areal" --seed 0
"""

import argparse
import json
import math
import os
import random
import re
import sys
import traceback

sys.path.insert(0, "/home/claude/tinker-rl-lab/experiments")
from variance_mitigation_integration import MethodConfig, advantage_computation, rollout_sampling

import tinker
import tinker.types as T
from datasets import load_dataset
from transformers import AutoTokenizer

def reward(response, answer):
    response = response.strip()
    # Check \boxed{answer}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    # Check last number in response
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0

def pvar(g):
    if not g: return 0.0
    m = sum(g) / len(g)
    return sum((x - m)**2 for x in g) / len(g)

def pearson_autocorr(series):
    if len(series) < 3: return float('nan')
    y1 = series[:-1]
    y2 = series[1:]
    m1 = sum(y1) / len(y1)
    m2 = sum(y2) / len(y2)
    var1 = sum((x - m1)**2 for x in y1)
    var2 = sum((x - m2)**2 for x in y2)
    if var1 == 0 or var2 == 0: return float('nan')
    cov = sum((x - m1)*(y - m2) for x, y in zip(y1, y2))
    return cov / math.sqrt(var1 * var2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default="grpo,aero,gift,areal")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--prompts-per-step", type=int, default=16)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--pool", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--wandb-project", default="tinker-new-research")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default="/home/claude/tinker-rl-lab/experiments/results/n2_reward_tensor")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 2
        args.prompts_per_step = 4
        args.group = 4
        args.pool = 32
        args.max_tokens = 128
        prefix = "smoke_"
    else:
        prefix = ""

    methods_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    
    if args.run_name is None:
        run_name = f"n2-reward-tensor_{'-'.join(methods_list)}_s{args.seed}_20260704"
        if args.smoke:
            run_name += "_smoke"
    else:
        run_name = args.run_name
        if args.smoke and not run_name.endswith("_smoke"):
            run_name += "_smoke"

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    SYS = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
    for row in ds:
        ans_match = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not ans_match: continue
        answer = ans_match.group(1).replace(",", "").strip()
        q = row["question"]
        prompt = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        examples.append((prompt, answer))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    pool_examples = examples[:args.pool]
    print(f"Pool size: {len(pool_examples)}")

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["N2", "reward-tensor", "20260704"]
        )
        print(f"WANDB_RUN_PATH={wandb.run.entity}/{wandb.run.project}/{wandb.run.id}", flush=True)
    else:
        wandb = None

    svc = tinker.ServiceClient(base_url=None)
    
    try:
        dummy_tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
        tok = dummy_tc.get_tokenizer()
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    processed_pool = []
    for i, (prompt_text, ans) in enumerate(pool_examples):
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > args.max_prompt_tokens:
            pid = pid[:args.max_prompt_tokens]
        processed_pool.append((i, pid, ans))
    
    jsonl_files_produced = []
    global_step = 0
    results_summary = []

    for method in methods_list:
        print(f"=== Starting method: {method} ===")
        try:
            cfg = MethodConfig.for_method(method)
            cfg.base_group_size = args.group
            if method == "aero":
                cfg.g_max = min(cfg.g_max, args.group)
                cfg.g_min = min(4, args.group)
            
            current_g = args.group
            running_mean = None
            rolling_zvf_window = []
            refresh_every = 4 if method == "areal" else 1

            tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
            w = tc.save_weights_for_sampler(name=f"{method}_s0").result()
            sc = tc.create_sampling_client(model_path=w.path)

            jsonl_path = os.path.join(args.out_dir, f"{prefix}{method}_s{args.seed}_tensors.jsonl")
            jsonl_files_produced.append(jsonl_path)
            jsonl_f = open(jsonl_path, "a")

            tsv_path = os.path.join(args.out_dir, f"{prefix}n2_metrics.tsv")
            write_header = not os.path.exists(tsv_path)
            tsv_f = open(tsv_path, "a")
            if write_header:
                tsv_f.write("method\tseed\tstep\tgroup_size\tzvf\tfrac_all_zero\tfrac_all_one\tpcd\tlarq\treward_mean\tmean_len\tcv_len\tlag1_autocorr\tloss\n")
            
            reward_mean_series = []

            for step in range(args.steps):
                if step > 0 and step % refresh_every == 0:
                    w = tc.save_weights_for_sampler(name=f"{method}_s{step}").result()
                    sc = tc.create_sampling_client(model_path=w.path)

                start_idx = (step * args.prompts_per_step) % len(processed_pool)
                batch_indices = [(start_idx + i) % len(processed_pool) for i in range(args.prompts_per_step)]
                batch = [processed_pool[i] for i in batch_indices]
                
                sp = T.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)
                
                futs = []
                for orig_idx, pid, ans in batch:
                    futs.append(sc.sample(T.ModelInput.from_ints(pid), num_samples=current_g, sampling_params=sp))
                
                step_responses = []
                for (orig_idx, pid, ans), fut in zip(batch, futs):
                    try:
                        resp = fut.result()
                    except Exception:
                        try:
                            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=current_g, sampling_params=sp).result()
                        except Exception:
                            resp = None
                    step_responses.append((orig_idx, pid, ans, resp))

                all_data = []
                step_tensors = {
                    "rewards": [], "lengths": [], "advantages": [], "prompt_indices": []
                }
                
                group_pvars = []
                group_phats = []
                all_rewards = []
                all_lengths = []
                frac_zero_count = 0
                frac_one_count = 0

                for orig_idx, pid, ans, resp in step_responses:
                    if resp is None:
                        continue
                    
                    rews = [reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
                    lens = [len(list(r.tokens)) for r in resp.sequences]
                    
                    advs, running_mean = advantage_computation(cfg, rews, running_mean)
                    
                    step_tensors["prompt_indices"].append(orig_idx)
                    step_tensors["rewards"].append(rews)
                    step_tensors["lengths"].append(lens)
                    step_tensors["advantages"].append(advs)
                    
                    all_rewards.extend(rews)
                    all_lengths.extend(lens)
                    
                    gv = pvar(rews)
                    group_pvars.append(gv)
                    
                    phat = sum(rews) / len(rews) if rews else 0.0
                    group_phats.append(phat)
                    
                    if all(r == 0.0 for r in rews) and len(rews) > 0: frac_zero_count += 1
                    if all(r == 1.0 for r in rews) and len(rews) > 0: frac_one_count += 1

                    for r_seq, a in zip(resp.sequences, advs):
                        if a != 0:
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
                                    "advantages": T.TensorData(data=advantages, dtype="float32", shape=[len(advantages)])
                                }
                            ))

                loss_val = None
                if all_data:
                    try:
                        result = tc.forward_backward(data=all_data, loss_fn="importance_sampling").result()
                        tc.optim_step(T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
                        try:
                            metrics = result.metrics
                            if hasattr(metrics, "get"):
                                loss_val = metrics.get("loss", None)
                                if loss_val is None:
                                    for k, v in metrics.items():
                                        if "loss" in str(k).lower():
                                            loss_val = v
                                            break
                            else:
                                loss_val = getattr(metrics, "loss", None)
                            if loss_val is not None and not isinstance(loss_val, (int, float)):
                                loss_val = float(loss_val)
                        except Exception:
                            loss_val = None
                    except Exception as e:
                        print(f"Step {step} optim failed: {e}")
                
                num_groups = len(group_pvars)
                zvf = sum(1 for v in group_pvars if v == 0.0) / num_groups if num_groups else 0.0
                frac_all_zero = frac_zero_count / num_groups if num_groups else 0.0
                frac_all_one = frac_one_count / num_groups if num_groups else 0.0
                pcd = sum(group_pvars) / num_groups if num_groups else 0.0
                larq_term1 = sum(group_phats) / num_groups if num_groups else 0.0
                larq_term2 = sum(4 * p * (1 - p) for p in group_phats) / num_groups if num_groups else 0.0
                larq = larq_term1 + 0.5 * larq_term2
                
                rmean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
                reward_mean_series.append(rmean)
                
                lmean = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
                lstd = (sum((l - lmean) ** 2 for l in all_lengths) / len(all_lengths)) ** 0.5 if all_lengths else 0.0
                cv_len = lstd / lmean if lmean > 0 else 0.0
                
                lag1_autocorr = pearson_autocorr(reward_mean_series[-10:])

                rolling_zvf_window.append(zvf)
                if len(rolling_zvf_window) > cfg.zvf_window:
                    rolling_zvf_window.pop(0)
                rolling_zvf = sum(rolling_zvf_window) / len(rolling_zvf_window)
                
                current_g_used = current_g
                current_g = rollout_sampling(cfg, current_g, rolling_zvf)
                
                jsonl_obj = {
                    "method": method,
                    "seed": args.seed,
                    "step": step,
                    "group_size": current_g_used,
                    "prompt_indices": step_tensors["prompt_indices"],
                    "rewards": step_tensors["rewards"],
                    "lengths": step_tensors["lengths"],
                    "advantages": step_tensors["advantages"],
                    "zvf": zvf,
                    "frac_all_zero": frac_all_zero,
                    "frac_all_one": frac_all_one,
                    "pcd": pcd,
                    "larq": larq,
                    "larq_term1": larq_term1,
                    "larq_term2": larq_term2,
                    "reward_mean": rmean,
                    "mean_len": lmean,
                    "cv_len": cv_len,
                    "lag1_autocorr": lag1_autocorr,
                    "loss": loss_val
                }
                jsonl_f.write(json.dumps(jsonl_obj) + "\n")
                jsonl_f.flush()
                os.fsync(jsonl_f.fileno())

                tsv_line = f"{method}\t{args.seed}\t{step}\t{current_g_used}\t{zvf}\t{frac_all_zero}\t{frac_all_one}\t{pcd}\t{larq}\t{rmean}\t{lmean}\t{cv_len}\t{lag1_autocorr}\t{loss_val}\n"
                tsv_f.write(tsv_line)
                tsv_f.flush()
                os.fsync(tsv_f.fileno())

                if wandb is not None:
                    wandb.log({
                        f"{method}/zvf": zvf,
                        f"{method}/frac_all_zero": frac_all_zero,
                        f"{method}/frac_all_one": frac_all_one,
                        f"{method}/pcd": pcd,
                        f"{method}/larq": larq,
                        f"{method}/reward_mean": rmean,
                        f"{method}/mean_len": lmean,
                        f"{method}/cv_len": cv_len,
                        f"{method}/lag1_autocorr": lag1_autocorr,
                        f"{method}/group_size": current_g_used,
                        f"{method}/loss": loss_val,
                        "method_step": step
                    }, step=global_step)
                
                print(f"[n2:{method}] {step+1:3d}/{args.steps} | reward {rmean:.3f} | zvf {zvf:.3f} | pcd {pcd:.3f}")
                global_step += 1

            jsonl_f.close()
            tsv_f.close()
            results_summary.append((method, step + 1, rmean, sum(rolling_zvf_window) / len(rolling_zvf_window) if rolling_zvf_window else 0, frac_all_zero, frac_all_one))

        except Exception as e:
            print(f"Method {method} failed with exception:")
            traceback.print_exc()
            continue

    if wandb is not None:
        art = wandb.Artifact("zvf-tensor-instrumented", type="reward-tensors")
        for jf in jsonl_files_produced:
            if os.path.exists(jf):
                art.add_file(jf)
        wandb.log_artifact(art)
        wandb.finish()

    print("\n=== Summary ===")
    print(f"{'Method':<10} | {'Steps':<6} | {'Reward':<6} | {'ZVF':<5} | {'Frac0':<6} | {'Frac1':<6}")
    for m, st, r, z, f0, f1 in results_summary:
        print(f"{m:<10} | {st:<6} | {r:<6.3f} | {z:<5.3f} | {f0:<6.3f} | {f1:<6.3f}")

    if results_summary:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
