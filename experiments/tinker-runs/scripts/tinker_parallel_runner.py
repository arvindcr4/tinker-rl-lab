"""Parallel Tinker experiment runner with W&B logging + HF Hub checkpointing.
Launches all experiments concurrently via ThreadPoolExecutor.
"""
import os, json, re, random, time, traceback, threading, shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("TINKER_API_KEY", "")
os.environ["TINKER_API_KEY"] = API_KEY
os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
os.environ["WANDB_MODE"] = "online"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi, create_repo

HF_USER = "arvindcr4"
HF_REPO_PREFIX = f"{HF_USER}/tinker-rl-bench"
WANDB_PROJECT = "tinker-rl-lab-world-class"
WANDB_ENTITY = None  # offline mode

RESULTS_DIR = "/home/user/workspace/tinker-rl-lab/experiments/tinker-runs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SYSTEM_PROMPT_MATH = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
SYSTEM_PROMPT_TOOL = """You are a helpful assistant with access to these tools:
[{"name":"calculator","description":"Performs arithmetic","parameters":{"expression":{"type":"string"}}},
{"name":"weather","description":"Gets weather","parameters":{"city":{"type":"string"}}},
{"name":"search","description":"Web search","parameters":{"query":{"type":"string"}}},
{"name":"reminder","description":"Sets reminder","parameters":{"message":{"type":"string"},"time":{"type":"string"}}},
{"name":"translate","description":"Translates text","parameters":{"text":{"type":"string"},"target_lang":{"type":"string"}}}]
Respond with JSON: {"name":"tool_name","arguments":{...}}"""

# ── Datasets ─────────────────────────────────────────────────────────────
_gsm8k_cache = None
_gsm8k_lock = threading.Lock()

def get_gsm8k():
    global _gsm8k_cache
    with _gsm8k_lock:
        if _gsm8k_cache is None:
            print("[DATA] Loading GSM8K...")
            ds = load_dataset("openai/gsm8k", "main", split="train")
            examples = []
            for row in ds:
                m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
                if not m: continue
                ans = m.group(1).replace(",","").strip()
                prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_MATH}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
                examples.append((prompt, ans, "gsm8k"))
            _gsm8k_cache = examples
            print(f"[DATA] GSM8K: {len(examples)} examples")
    return list(_gsm8k_cache)

def get_tool_use():
    tasks = [
        ("What is 234 * 567?", '{"name":"calculator","arguments":{"expression":"234 * 567"}}'),
        ("What's the weather in Tokyo?", '{"name":"weather","arguments":{"city":"Tokyo"}}'),
        ("Search for recent advances in RLHF", '{"name":"search","arguments":{"query":"recent advances in RLHF"}}'),
        ("Remind me to submit the paper at 5pm", '{"name":"reminder","arguments":{"message":"submit the paper","time":"5pm"}}'),
        ("Translate 'hello world' to French", '{"name":"translate","arguments":{"text":"hello world","target_lang":"French"}}'),
        ("Calculate sqrt(144)", '{"name":"calculator","arguments":{"expression":"sqrt(144)"}}'),
        ("What's the weather in New York?", '{"name":"weather","arguments":{"city":"New York"}}'),
        ("Search for NeurIPS 2026 deadline", '{"name":"search","arguments":{"query":"NeurIPS 2026 deadline"}}'),
        ("Remind me about the meeting at 9am", '{"name":"reminder","arguments":{"message":"meeting","time":"9am"}}'),
        ("Translate 'good morning' to Japanese", '{"name":"translate","arguments":{"text":"good morning","target_lang":"Japanese"}}'),
    ]
    return [(f"<|im_start|>system\n{SYSTEM_PROMPT_TOOL}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", a, "tool_use") for q,a in tasks]

# ── Rewards ──────────────────────────────────────────────────────────────
def reward_math(response, answer):
    response = response.strip()
    for b in re.findall(r'\\boxed\{([^}]+)\}', response):
        try:
            if abs(float(b.strip().replace(",","")) - float(answer)) < 0.01: return 1.0
        except:
            if b.strip().replace(",","") == answer: return 1.0
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if nums:
        try:
            if abs(float(nums[-1].replace(",","")) - float(answer)) < 0.01: return 1.0
        except: pass
    return 0.0

def reward_tool(response, expected):
    score = 0.0
    try:
        parsed = json.loads(response.strip())
        score += 0.3
        exp = json.loads(expected)
        if parsed.get("name") == exp.get("name"): score += 0.4
        if "arguments" in parsed and set(exp.get("arguments",{}).keys()).issubset(set(parsed.get("arguments",{}).keys())): score += 0.3
    except: pass
    return score

# ── HuggingFace upload helper ────────────────────────────────────────────
def upload_to_hf(exp_tag, model_id, results, step_log):
    """Upload experiment results and metadata to HF Hub."""
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_id = f"{HF_USER}/tinker-rl-bench-{exp_tag}"
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ["HF_TOKEN"])
        except Exception:
            pass  # repo may already exist

        # Create model card
        card = f"""---
tags:
  - grpo
  - reinforcement-learning
  - tinker
  - tinker-rl-bench
base_model: {model_id}
datasets:
  - openai/gsm8k
library_name: tinker
license: mit
---

# {exp_tag}

GRPO experiment from TinkerRL-Bench world-class experiment suite.

## Training Details
- **Base model:** {model_id}
- **Method:** GRPO (Group Relative Policy Optimization)
- **Platform:** Tinker API v0.18.1
- **Task:** {results.get('task', 'gsm8k')}
- **Seed:** {results.get('seed', 42)}
- **LoRA rank:** {results.get('rank', 32)}
- **Learning rate:** {results.get('lr', 3e-5)}
- **Group size:** {results.get('group', 8)}
- **Steps:** {results.get('steps', 30)}

## Results
- **First-5 avg reward:** {results.get('first5_avg', 0)*100:.1f}%
- **Last-10 avg reward:** {results.get('last10_avg', 0)*100:.1f}%
- **Peak reward:** {results.get('peak', 0)*100:.1f}%
- **Zero-loss steps:** {results.get('zero_loss_pct', 0):.0f}%
- **Tinker Run ID:** {results.get('run_id', 'N/A')}

## Reward Trace
```
{json.dumps(results.get('reward_trace', []), indent=2)}
```

## Citation
```bibtex
@misc{{tinker-rl-bench-2026,
  title={{TinkerRL-Bench: A Unified Benchmark for RL Post-Training}},
  author={{Arvind C R and Sandhya Jeyaraj and Madhu Kumara L and Mohammad Rafi and Dhruva N Murthy and Arumugam K}},
  year={{2026}},
  url={{https://github.com/arvindcr4/tinker-rl-lab}}
}}
```
"""
        # Upload model card and results
        tmp_dir = f"/tmp/hf_upload_{exp_tag}"
        os.makedirs(tmp_dir, exist_ok=True)
        with open(f"{tmp_dir}/README.md", "w") as f: f.write(card)
        with open(f"{tmp_dir}/results.json", "w") as f: json.dump(results, f, indent=2)
        with open(f"{tmp_dir}/step_log.json", "w") as f: json.dump(step_log, f, indent=2)

        api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, repo_type="model", token=os.environ["HF_TOKEN"])
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[HF] ✓ Uploaded {repo_id}")
        return repo_id
    except Exception as e:
        print(f"[HF] ✗ Upload failed for {exp_tag}: {e}")
        return None

# ── Single experiment runner ─────────────────────────────────────────────
def run_single(model_name, model_id, task, seed=42, rank=32, lr=3e-5, group=8, steps=30, batch=2, tag=""):
    exp = tag or f"{task}_{model_name}_s{seed}"
    rng = random.Random(seed)

    # ── W&B init ──
    wb_run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY, name=exp,
        config={"model": model_id, "model_short": model_name, "task": task,
                "seed": seed, "rank": rank, "lr": lr, "group": group,
                "steps": steps, "batch": batch, "platform": "tinker"},
        reinit=True, tags=["tinker", task, model_name],
    )

    print(f"[{exp}] ▶ START model={model_id} task={task}")

    examples = get_gsm8k() if task == "gsm8k" else get_tool_use()
    rfn = reward_math if task == "gsm8k" else reward_tool
    rng.shuffle(examples)

    try:
        svc = tinker.ServiceClient(api_key=API_KEY)
        tc = svc.create_lora_training_client(base_model=model_id, rank=rank)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        w0 = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=w0.path)
    except Exception as e:
        print(f"[{exp}] ✗ INIT FAILED: {e}")
        wb_run.finish(exit_code=1)
        return {"experiment": exp, "model": model_id, "error": str(e), "stage": "init"}

    print(f"[{exp}] Connected: {tc.model_id}")
    wb_run.config.update({"tinker_run_id": tc.model_id}, allow_val_change=True)

    _adv = []
    def loss_fn(data, lp):
        losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
        return torch.stack(losses).mean(), {"loss": torch.stack(losses).mean().item()}

    step_rewards, step_log = [], []
    zl, zr = 0, 0
    cumulative_rewards = []

    for step in range(steps):
        bex = [examples[i % len(examples)] for i in rng.sample(range(len(examples)), min(batch, len(examples)))]
        all_data, all_advs, batch_r = [], [], []

        for prompt_text, ans, _ in bex:
            pid = tok.encode(prompt_text, add_special_tokens=False)[:1024]
            sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
            try:
                resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=group, sampling_params=sp).result()
            except Exception as e:
                print(f"[{exp}] Sample err step {step}: {e}")
                continue

            rews = [rfn(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rews)/len(rews)
            sr = (sum((r-mr)**2 for r in rews)/len(rews))**0.5 + 1e-8
            advs = [(r-mr)/sr for r in rews]
            batch_r.extend(rews)

            for r, a in zip(resp.sequences, advs):
                rid = list(r.tokens)
                fid = pid + rid
                tid = fid[1:] + [0]
                all_data.append(T.Datum(
                    model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={"target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])}
                ))
                all_advs.append(a)

        if not all_data: continue
        _adv = all_advs

        try:
            result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
            tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        except Exception as e:
            print(f"[{exp}] Train err step {step}: {e}")
            continue

        avg = sum(batch_r)/len(batch_r) if batch_r else 0
        step_rewards.append(avg)
        cumulative_rewards.append(sum(step_rewards)/len(step_rewards))
        lv = result.metrics.get("loss", 0)
        if abs(lv) < 1e-6: zl += 1
        if avg == 0: zr += 1

        # ZVF = fraction of groups where all completions got same reward
        zvf = 1.0 if len(set(batch_r)) <= 1 else 0.0
        gu = 1.0 - zvf

        step_log.append({"step": step+1, "loss": lv, "reward": avg, "zvf": zvf, "gu": gu})

        # ── W&B log ──
        wb_run.log({
            "step": step + 1,
            "train/loss": lv,
            "train/reward": avg,
            "train/cumulative_reward": cumulative_rewards[-1],
            "train/peak_reward": max(step_rewards),
            "train/zvf": zvf,
            "train/gradient_utilization": gu,
            "train/zero_loss_frac": zl / (step + 1),
            "train/zero_reward_frac": zr / (step + 1),
        }, step=step + 1)

        if (step+1) % 5 == 0 or step == 0:
            print(f"[{exp}] {step+1:3d}/{steps} loss={lv:.4f} reward={avg:.3f} zvf={zvf:.1f}")

        if (step+1) % max(steps//3, 5) == 0:
            try:
                ckpt = tc.save_weights_for_sampler(name=f"s{step+1}").result()
                sc = tc.create_sampling_client(model_path=ckpt.path)
            except: pass

    try:
        tc.save_state(name="final")
        fckpt = tc.save_weights_for_sampler(name="final").result()
        ckpt_path = fckpt.path
    except:
        ckpt_path = "save_failed"

    last10 = step_rewards[-10:] if len(step_rewards) >= 10 else step_rewards
    first5 = step_rewards[:5] if len(step_rewards) >= 5 else step_rewards

    summary = {
        "experiment": exp, "model": model_id, "model_short": model_name,
        "task": task, "seed": seed, "rank": rank, "lr": lr, "group": group, "steps": steps,
        "run_id": tc.model_id, "checkpoint": ckpt_path,
        "first5_avg": sum(first5)/len(first5) if first5 else 0,
        "last10_avg": sum(last10)/len(last10) if last10 else 0,
        "peak": max(step_rewards) if step_rewards else 0,
        "zero_loss_pct": zl/max(steps,1)*100, "zero_reward_pct": zr/max(steps,1)*100,
        "reward_trace": [round(r,4) for r in step_rewards],
        "step_log": step_log, "timestamp": datetime.now().isoformat(),
    }

    # ── W&B summary ──
    wb_run.summary.update({
        "final/first5_avg": summary["first5_avg"],
        "final/last10_avg": summary["last10_avg"],
        "final/peak": summary["peak"],
        "final/zero_loss_pct": summary["zero_loss_pct"],
        "final/zero_reward_pct": summary["zero_reward_pct"],
        "final/tinker_run_id": summary["run_id"],
        "final/checkpoint": ckpt_path,
    })

    # ── Save to disk ──
    with open(os.path.join(RESULTS_DIR, f"{exp}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Upload to HF Hub ──
    hf_repo = upload_to_hf(exp, model_id, summary, step_log)
    if hf_repo:
        summary["hf_repo"] = hf_repo
        wb_run.summary["hf_repo"] = hf_repo

    # ── W&B artifacts ──
    try:
        artifact = wandb.Artifact(name=exp.replace("/","-"), type="experiment-results", description=f"Results for {exp}")
        artifact.add_file(os.path.join(RESULTS_DIR, f"{exp}.json"))
        wb_run.log_artifact(artifact)
    except: pass

    wb_run.finish()

    print(f"[{exp}] ✓ DONE | last10={summary['last10_avg']*100:.1f}% peak={summary['peak']*100:.1f}%")
    return summary

# ── Experiment configs ───────────────────────────────────────────────────
EXPERIMENTS = [
    # === SCALING (GSM8K across model sizes) ===
    {"model_name":"qwen3-8b","model_id":"Qwen/Qwen3-8B","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"scale_gsm8k_qwen3-8b"},
    {"model_name":"qwen3-32b","model_id":"Qwen/Qwen3-32B","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"scale_gsm8k_qwen3-32b"},
    {"model_name":"qwen3.5-4b","model_id":"Qwen/Qwen3.5-4B","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"scale_gsm8k_qwen3.5-4b"},
    {"model_name":"qwen3.5-27b","model_id":"Qwen/Qwen3.5-27B","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"scale_gsm8k_qwen3.5-27b"},
    {"model_name":"llama-8b-inst","model_id":"meta-llama/Llama-3.1-8B-Instruct","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"scale_gsm8k_llama-8b-inst"},
    # === FRONTIER (largest models) ===
    {"model_name":"qwen3-235b-moe","model_id":"Qwen/Qwen3-235B-A22B-Instruct-2507","task":"gsm8k","seed":42,"rank":16,"lr":1e-5,"group":4,"steps":20,"tag":"frontier_gsm8k_qwen3-235b"},
    {"model_name":"deepseek-v3.1","model_id":"deepseek-ai/DeepSeek-V3.1","task":"gsm8k","seed":42,"rank":16,"lr":1e-5,"group":4,"steps":20,"tag":"frontier_gsm8k_deepseek-v3.1"},
    {"model_name":"nemotron-120b","model_id":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16","task":"gsm8k","seed":42,"rank":16,"lr":1e-5,"group":4,"steps":20,"tag":"frontier_gsm8k_nemotron-120b"},
    # === MOE vs DENSE ===
    {"model_name":"qwen3-30b-moe","model_id":"Qwen/Qwen3-30B-A3B","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"moe_gsm8k_qwen3-30b-moe"},
    {"model_name":"qwen3-30b-moe-inst","model_id":"Qwen/Qwen3-30B-A3B-Instruct-2507","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"moe_gsm8k_qwen3-30b-inst"},
    # === CROSS-TASK (Tool-use) ===
    {"model_name":"qwen3-32b","model_id":"Qwen/Qwen3-32B","task":"tool_use","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"cross_tool_qwen3-32b"},
    {"model_name":"llama-8b-inst","model_id":"meta-llama/Llama-3.1-8B-Instruct","task":"tool_use","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"cross_tool_llama-8b-inst"},
    # === NEW ARCHITECTURES ===
    {"model_name":"gpt-oss-20b","model_id":"openai/gpt-oss-20b","task":"gsm8k","seed":42,"rank":32,"lr":3e-5,"group":8,"steps":30,"tag":"arch_gsm8k_gpt-oss-20b"},
    {"model_name":"kimi-k2","model_id":"moonshotai/Kimi-K2-Thinking","task":"gsm8k","seed":42,"rank":16,"lr":1e-5,"group":4,"steps":20,"tag":"arch_gsm8k_kimi-k2"},
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.filter:
        exps = [e for e in exps if args.filter in e["tag"]]

    # Skip already-completed experiments
    completed_tags = set()
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            tag = fname.replace('.json', '')
            try:
                with open(os.path.join(RESULTS_DIR, fname)) as f:
                    data = json.load(f)
                if 'error' not in data and data.get('peak', -1) >= 0:
                    completed_tags.add(tag)
            except Exception:
                pass
    before = len(exps)
    exps = [e for e in exps if e['tag'] not in completed_tags]
    if before != len(exps):
        print(f"Skipping {before - len(exps)} already-completed experiments: {completed_tags}")
        print(f"Running {len(exps)} remaining experiments")

    print(f"\n{'='*70}")
    print(f"LAUNCHING {len(exps)} TINKER EXPERIMENTS (max {args.max_parallel} parallel)")
    print(f"W&B: {WANDB_PROJECT} | HF: {HF_USER}/tinker-rl-bench-*")
    print(f"{'='*70}\n")

    get_gsm8k()  # pre-load

    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {}
        for exp in exps:
            f = pool.submit(run_single, **{k:v for k,v in exp.items()}, batch=2)
            futures[f] = exp["tag"]

        for f in as_completed(futures):
            tag = futures[f]
            try:
                r = f.result()
                all_results.append(r)
                status = "error" not in r
                sym = "★" if status else "✗"
                msg = f"last10={r['last10_avg']*100:.1f}% peak={r['peak']*100:.1f}%" if status else r.get("error","unknown")[:60]
                print(f"\n{sym} [{tag}] {'COMPLETED' if status else 'FAILED'}: {msg}\n")
            except Exception as e:
                print(f"\n✗ [{tag}] EXCEPTION: {e}\n")
                all_results.append({"experiment": tag, "error": str(e)})

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined = os.path.join(RESULTS_DIR, f"tinker_parallel_{ts}.json")
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL {len(all_results)} EXPERIMENTS COMPLETE")
    print(f"Results: {combined}")
    print(f"{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  ✗ {r['experiment']}: {r['error'][:80]}")
        else:
            hf = r.get('hf_repo', 'N/A')
            print(f"  ✓ {r['experiment']}: last10={r['last10_avg']*100:.1f}% peak={r['peak']*100:.1f}% HF={hf}")
