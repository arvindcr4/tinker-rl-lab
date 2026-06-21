"""GRPO on GSM8K — parameterized for multi-seed, scaling, and ablation runs.
Usage: python grpo_gsm8k_base.py --model Qwen/Qwen3-8B --seed 137 --rank 32 --steps 50
"""

import os, json, re, warnings, random, argparse

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--group", type=int, default=4)
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--tag", default="")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

MODEL = args.model
EXP = args.tag or f"gsm8k_{MODEL.split('/')[-1]}_s{args.seed}_r{args.rank}"
STEPS, GROUP, LR, RANK = args.steps, args.group, args.lr, args.rank
SAVE_EVERY = max(args.steps // 4, 10)

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."

# ── Load GSM8K ───────────────────────────────────────────────────────────
print(f"[{EXP}] Loading GSM8K...")
ds = load_dataset("openai/gsm8k", "main", split="train")
examples = []
for row in ds:
    q = row["question"]
    # Extract final numeric answer from "#### <number>"
    ans_match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
    if not ans_match:
        continue
    answer = ans_match.group(1).replace(",", "").strip()
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    examples.append((prompt, answer))

random.shuffle(examples)
print(
    f"[{EXP}] {len(examples)} GSM8K examples | model={MODEL} seed={args.seed} rank={RANK} lr={LR}"
)


# ── Reward: binary exact match on \\boxed{} or final number ─────────────
def reward(response, answer):
    response = response.strip()
    # Check \boxed{answer}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    # Check last number in response
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0  # Binary reward


# ── GRPO ─────────────────────────────────────────────────────────────────
from utils.tinker_grpo import run_grpo_training

run_grpo_training(
    exp_name=EXP,
    model_name=MODEL,
    rank=RANK,
    steps=STEPS,
    lr=LR,
    group_size=GROUP,
    batch_size=args.batch,
    save_every=SAVE_EVERY,
    examples=examples,
    reward_fn=reward,
    max_tokens=512,
    temperature=0.8,
    top_p=0.95
)
