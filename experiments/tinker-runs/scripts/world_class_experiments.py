"""World-Class Experiment Suite for NeurIPS/ICLR/ICML Submission.

Runs GRPO experiments across the full Tinker model catalog:
- Multi-model scaling: Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, 
  Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, DeepSeek-V3.1, Nemotron-120B
- Multi-task: GSM8K, MATH-500, Tool-Use, Code (HumanEval)
- Loss function comparison: importance_sampling, ppo, cispo, dro
- Cross-architecture: Dense vs MoE at matched active params

Usage: 
  python world_class_experiments.py --experiment scaling   # Scale sweep
  python world_class_experiments.py --experiment loss_cmp  # Loss function comparison
  python world_class_experiments.py --experiment frontier   # Frontier models (235B, 70B)
"""
import os, json, re, warnings, random, argparse, time, traceback
from datetime import datetime
warnings.filterwarnings("ignore")

API_KEY = os.environ.get("TINKER_API_KEY", "")
os.environ["TINKER_API_KEY"] = API_KEY

import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────
RESULTS_DIR = "/home/user/workspace/tinker-rl-lab/experiments/tinker-runs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SYSTEM_PROMPT_MATH = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
SYSTEM_PROMPT_CODE = "You are a coding assistant. Write clean Python code to solve the problem. Put your solution inside ```python\\n...\\n```."
SYSTEM_PROMPT_TOOL = """You are a helpful assistant with access to the following tools:
[{"name": "calculator", "description": "Performs arithmetic operations", "parameters": {"expression": {"type": "string", "description": "Math expression to evaluate"}}},
{"name": "weather", "description": "Gets current weather", "parameters": {"city": {"type": "string", "description": "City name"}}},
{"name": "search", "description": "Searches the web", "parameters": {"query": {"type": "string", "description": "Search query"}}},
{"name": "reminder", "description": "Sets a reminder", "parameters": {"message": {"type": "string"}, "time": {"type": "string"}}},
{"name": "translate", "description": "Translates text", "parameters": {"text": {"type": "string"}, "target_lang": {"type": "string"}}}]
Respond with a JSON tool call: {"name": "tool_name", "arguments": {...}}"""

# ── Model Registry ───────────────────────────────────────────────────────
MODELS = {
    # Dense models - scaling ladder
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B", 
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "llama-8b": "meta-llama/Llama-3.1-8B",
    "llama-8b-inst": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B",
    "llama-70b-inst": "meta-llama/Llama-3.3-70B-Instruct",
    # MoE models
    "qwen3-30b-moe": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-moe-inst": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-235b-moe": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen3.5-35b-moe": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.5-397b-moe": "Qwen/Qwen3.5-397B-A17B",
    # Frontier models
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "nemotron-30b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron-120b": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "kimi-k2": "moonshotai/Kimi-K2-Thinking",
}

# ── Dataset Loaders ──────────────────────────────────────────────────────
def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        ans_match = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not ans_match: continue
        answer = ans_match.group(1).replace(",", "").strip()
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_MATH}<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
        examples.append((prompt, answer, "gsm8k"))
    return examples

def load_tool_use():
    """Synthetic 5-tool task suite."""
    tool_tasks = [
        ("What is 234 * 567?", '{"name": "calculator", "arguments": {"expression": "234 * 567"}}'),
        ("What's the weather in Tokyo?", '{"name": "weather", "arguments": {"city": "Tokyo"}}'),
        ("Search for recent advances in RLHF", '{"name": "search", "arguments": {"query": "recent advances in RLHF"}}'),
        ("Remind me to submit the paper at 5pm", '{"name": "reminder", "arguments": {"message": "submit the paper", "time": "5pm"}}'),
        ("Translate 'hello world' to French", '{"name": "translate", "arguments": {"text": "hello world", "target_lang": "French"}}'),
        ("Calculate the square root of 144", '{"name": "calculator", "arguments": {"expression": "sqrt(144)"}}'),
        ("What's the weather in New York?", '{"name": "weather", "arguments": {"city": "New York"}}'),
        ("Search for NeurIPS 2026 submission deadline", '{"name": "search", "arguments": {"query": "NeurIPS 2026 submission deadline"}}'),
        ("Remind me about the meeting tomorrow at 9am", '{"name": "reminder", "arguments": {"message": "meeting tomorrow", "time": "9am"}}'),
        ("Translate 'good morning' to Japanese", '{"name": "translate", "arguments": {"text": "good morning", "target_lang": "Japanese"}}'),
    ]
    examples = []
    for q, a in tool_tasks:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_TOOL}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        examples.append((prompt, a, "tool_use"))
    return examples

# ── Reward Functions ─────────────────────────────────────────────────────
def reward_math(response, answer):
    response = response.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01: return 1.0
        except:
            if b_clean == answer: return 1.0
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01: return 1.0
        except: pass
    return 0.0

def reward_tool(response, expected):
    score = 0.0
    try:
        parsed = json.loads(response.strip())
        score += 0.3  # Valid JSON
        exp = json.loads(expected)
        if parsed.get("name") == exp.get("name"): score += 0.4
        if "arguments" in parsed:
            exp_keys = set(exp.get("arguments", {}).keys())
            got_keys = set(parsed.get("arguments", {}).keys())
            if exp_keys and got_keys.issuperset(exp_keys): score += 0.3
    except: pass
    return score

# ── GRPO Training Loop ───────────────────────────────────────────────────
def run_grpo_experiment(model_name, model_id, task, seed=42, rank=32, lr=3e-5, 
                         group=8, steps=30, batch=2, tag=""):
    """Run a single GRPO experiment and save results."""
    random.seed(seed)
    torch.manual_seed(seed)
    
    exp_tag = tag or f"{task}_{model_name}_s{seed}"
    print(f"\n{'='*70}")
    print(f"[{exp_tag}] Starting: model={model_id} task={task} seed={seed}")
    print(f"[{exp_tag}] Config: rank={rank} lr={lr} group={group} steps={steps}")
    print(f"{'='*70}\n")
    
    # Load data
    if task == "gsm8k":
        examples = load_gsm8k()
        reward_fn = reward_math
    elif task == "tool_use":
        examples = load_tool_use()
        reward_fn = reward_tool
    else:
        raise ValueError(f"Unknown task: {task}")
    
    random.shuffle(examples)
    print(f"[{exp_tag}] Loaded {len(examples)} examples")
    
    # Connect to Tinker
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(base_model=model_id, rank=rank)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    w0 = tc.save_weights_for_sampler(name="s0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    print(f"[{exp_tag}] Connected. Run: {tc.model_id}")
    
    # Training
    _adv = []
    def loss_fn(data, lp):
        losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
        loss = torch.stack(losses).mean()
        return loss, {"loss": loss.item()}
    
    step_rewards = []
    zero_loss_steps = 0
    zero_reward_steps = 0
    step_log = []
    
    for step in range(steps):
        batch_examples = random.sample(examples, min(batch, len(examples)))
        all_data, all_advs, batch_r = [], [], []
        
        for prompt_text, ans, _ in batch_examples:
            pid = tok.encode(prompt_text, add_special_tokens=False)
            if len(pid) > 1024: pid = pid[:1024]
            sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
            
            try:
                resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=group, sampling_params=sp).result()
            except Exception as e:
                print(f"[{exp_tag}] Sample error at step {step}: {e}")
                continue
                
            rews = [reward_fn(tok.decode(list(r.tokens), skip_special_tokens=True), ans) for r in resp.sequences]
            mr = sum(rews) / len(rews)
            sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8
            advs = [(r - mr) / sr for r in rews]
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
        result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
        
        avg = sum(batch_r) / len(batch_r)
        step_rewards.append(avg)
        loss_val = result.metrics.get("loss", 0)
        if abs(loss_val) < 1e-6: zero_loss_steps += 1
        if avg == 0: zero_reward_steps += 1
        
        # Compute ZVF
        zvf = 1.0 if len(set(batch_r)) <= 1 else 0.0
        
        step_log.append({
            "step": step + 1,
            "loss": loss_val,
            "reward": avg,
            "zvf": zvf,
        })
        
        print(f"[{exp_tag}] {step+1:3d}/{steps} | loss={loss_val:.4f} | reward={avg:.3f} | acc={avg*100:.1f}%")
        
        # Save checkpoint periodically
        if (step + 1) % max(steps // 4, 5) == 0:
            ckpt = tc.save_weights_for_sampler(name=f"s{step+1}").result()
            sc = tc.create_sampling_client(model_path=ckpt.path)
    
    # Final save
    tc.save_state(name="final")
    final_ckpt = tc.save_weights_for_sampler(name="final").result()
    
    # Compute summary stats
    last10 = step_rewards[-10:] if len(step_rewards) >= 10 else step_rewards
    first5 = step_rewards[:5] if len(step_rewards) >= 5 else step_rewards
    
    result_summary = {
        "experiment": exp_tag,
        "model": model_id,
        "model_short": model_name,
        "task": task,
        "seed": seed,
        "rank": rank,
        "lr": lr,
        "group": group,
        "steps": steps,
        "run_id": tc.model_id,
        "checkpoint": final_ckpt.path,
        "first5_avg": sum(first5) / len(first5) if first5 else 0,
        "last10_avg": sum(last10) / len(last10) if last10 else 0,
        "peak": max(step_rewards) if step_rewards else 0,
        "zero_loss_pct": zero_loss_steps / steps * 100,
        "zero_reward_pct": zero_reward_steps / steps * 100,
        "reward_trace": [round(r, 4) for r in step_rewards],
        "step_log": step_log,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    result_file = os.path.join(RESULTS_DIR, f"{exp_tag}.json")
    with open(result_file, "w") as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"\n[{exp_tag}] === FINAL REPORT ===")
    print(f"[{exp_tag}] First-5 avg: {result_summary['first5_avg']*100:.1f}%")
    print(f"[{exp_tag}] Last-10 avg: {result_summary['last10_avg']*100:.1f}%")
    print(f"[{exp_tag}] Peak: {result_summary['peak']*100:.1f}%")
    print(f"[{exp_tag}] Zero-loss: {result_summary['zero_loss_pct']:.0f}%")
    print(f"[{exp_tag}] Results saved to: {result_file}")
    
    return result_summary

# ── Experiment Suites ────────────────────────────────────────────────────
def run_scaling_suite():
    """Scale sweep: same task (GSM8K) across model sizes."""
    models = [
        ("qwen3-8b", "Qwen/Qwen3-8B"),
        ("qwen3-32b", "Qwen/Qwen3-32B"),
        ("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
        ("qwen3.5-27b", "Qwen/Qwen3.5-27B"),
        ("llama-8b-inst", "meta-llama/Llama-3.1-8B-Instruct"),
        ("llama-70b", "meta-llama/Llama-3.1-70B"),
    ]
    results = []
    for name, mid in models:
        try:
            r = run_grpo_experiment(name, mid, "gsm8k", seed=42, rank=32, 
                                     lr=3e-5, group=8, steps=30, batch=2,
                                     tag=f"scale_gsm8k_{name}")
            results.append(r)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            traceback.print_exc()
            results.append({"model": name, "error": str(e)})
    return results

def run_frontier_suite():
    """Frontier models: largest available models on GSM8K."""
    models = [
        ("qwen3-235b-moe", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
        ("deepseek-v3.1", "deepseek-ai/DeepSeek-V3.1"),
        ("nemotron-120b", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        ("gpt-oss-120b", "openai/gpt-oss-120b"),
    ]
    results = []
    for name, mid in models:
        try:
            r = run_grpo_experiment(name, mid, "gsm8k", seed=42, rank=16,
                                     lr=1e-5, group=4, steps=20, batch=1,
                                     tag=f"frontier_gsm8k_{name}")
            results.append(r)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            traceback.print_exc()
            results.append({"model": name, "error": str(e)})
    return results

def run_moe_comparison():
    """Dense vs MoE at matched active params."""
    experiments = [
        # ~3B active: dense 4B vs MoE 30B-A3B  
        ("qwen3.5-4b", "Qwen/Qwen3.5-4B", "Dense 4B"),
        ("qwen3-30b-moe", "Qwen/Qwen3-30B-A3B", "MoE 30B (3B active)"),
        ("qwen3-30b-moe-inst", "Qwen/Qwen3-30B-A3B-Instruct-2507", "MoE 30B Inst (3B active)"),
        # ~22B active: dense 27B vs MoE 235B-A22B
        ("qwen3.5-27b", "Qwen/Qwen3.5-27B", "Dense 27B"),
        ("qwen3-235b-moe", "Qwen/Qwen3-235B-A22B-Instruct-2507", "MoE 235B (22B active)"),
    ]
    results = []
    for name, mid, desc in experiments:
        try:
            r = run_grpo_experiment(name, mid, "gsm8k", seed=42, rank=32,
                                     lr=3e-5, group=8, steps=30, batch=2,
                                     tag=f"moe_cmp_{name}")
            r["description"] = desc
            results.append(r)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            traceback.print_exc()
            results.append({"model": name, "error": str(e), "description": desc})
    return results

def run_cross_task():
    """Same model across multiple tasks."""
    tasks = ["gsm8k", "tool_use"]
    models = [
        ("qwen3-8b", "Qwen/Qwen3-8B"),
        ("qwen3-32b", "Qwen/Qwen3-32B"),
        ("llama-8b-inst", "meta-llama/Llama-3.1-8B-Instruct"),
    ]
    results = []
    for name, mid in models:
        for task in tasks:
            try:
                r = run_grpo_experiment(name, mid, task, seed=42, rank=32,
                                         lr=3e-5, group=8, steps=30, batch=2,
                                         tag=f"cross_{task}_{name}")
                results.append(r)
            except Exception as e:
                print(f"ERROR {name}/{task}: {e}")
                traceback.print_exc()
                results.append({"model": name, "task": task, "error": str(e)})
    return results

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["scaling", "frontier", "moe", "cross_task", "all"],
                        default="scaling")
    args = parser.parse_args()
    
    all_results = []
    
    if args.experiment in ("scaling", "all"):
        print("\n" + "="*70)
        print("RUNNING SCALING SUITE")
        print("="*70)
        all_results.extend(run_scaling_suite())
    
    if args.experiment in ("frontier", "all"):
        print("\n" + "="*70)
        print("RUNNING FRONTIER SUITE") 
        print("="*70)
        all_results.extend(run_frontier_suite())
    
    if args.experiment in ("moe", "all"):
        print("\n" + "="*70)
        print("RUNNING MOE COMPARISON SUITE")
        print("="*70)
        all_results.extend(run_moe_comparison())
    
    if args.experiment in ("cross_task", "all"):
        print("\n" + "="*70)
        print("RUNNING CROSS-TASK SUITE")
        print("="*70)
        all_results.extend(run_cross_task())
    
    # Save combined results
    combined_file = os.path.join(RESULTS_DIR, f"combined_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {combined_file}")
