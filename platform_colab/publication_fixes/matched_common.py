"""Shared pieces for the matched TRL-vs-Tinker run (claim C, experiment 4).

Prompt string, reward function, GSM8K loading and the per-step prompt stream are taken
verbatim from platform_hybrid/experiments/tinker-runs/campaign_v2.py. Both sides import
this file, so they see byte-identical prompts in the same order.
"""
import random
import re

MODEL = "Qwen/Qwen3-8B"
SEEDS = [42, 123, 456]
STEPS = 30
PROMPTS_PER_STEP = 2
GROUP = 8
LR = 1e-5
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
LORA = dict(r=32, lora_alpha=32, lora_dropout=0.0,  # = Tinker adapter_config.json (probed 2026-10-16)
            # Tinker default also trains the unembedding; TRL 1.16 rejects lm_head LoRA, so both sides
            # train only these 7 projections (Tinker: train_unembed=False).
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
TEMP, TOP_P, MAX_TOK = 0.8, 0.95, 512
N_EVAL = 200

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def build_prompt(question):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question + QUESTION_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0


def boxed_only_reward(text, expected):
    """modal_grpo_trl.py reward (boxed match only), for crossed scoring."""
    for b in re.findall(r'\\boxed\{([^}]+)\}', text):
        b_clean = b.strip().replace(",", "")
        try:
            if expected and abs(float(b_clean) - float(expected)) < 0.01:
                return 1.0
        except:
            if b_clean == expected:
                return 1.0
    return 0.0


def gsm8k(split):
    from datasets import load_dataset
    out = []
    for row in load_dataset("openai/gsm8k", "main", split=split):
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if m:
            out.append((row["question"], m.group(1).replace(",", "").strip()))
    return out


def prompt_stream(seed, examples):
    """campaign_v2: random.seed(seed); random.shuffle(examples); random.sample(examples, 2) per step."""
    rnd = random.Random(seed)
    ex = list(examples)
    rnd.shuffle(ex)
    return [rnd.sample(ex, min(PROMPTS_PER_STEP, len(ex))) for _ in range(STEPS)]
