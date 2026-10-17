
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import execute_fixture_tool, TurnRecord, json_object

# ============================================================
#  ADVANCED EXPERIMENT — STEP 2: TRUE Multi-Turn GRPO
#
#  🔧 FIXES vs previous version:
#   FIX 1 — generate_chain: wrap-up message is now injected
#            IMMEDIATELY after all tool responses are exhausted,
#            before the next generation — not at the last turn.
#            This ensures the model sees the nudge when it still
#            has room to respond with a final answer.
#   FIX 2 — generate_chain: if model calls the SAME tool twice
#            in a row (repeated call), skip injecting a new tool
#            result and inject the wrap-up instead — stopping
#            the loop right there.
#   (reward function, loss, dataset loading: unchanged from v2)
# ============================================================

# !pip install -q transformers datasets peft trl accelerate bitsandbytes

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

import json, os, re, random, torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import PeftModel

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTER  = "/content/qwen3b-toolbench-sft"
OUTPUT_DIR   = "/content/qwen3b-toolbench-grpo"
NUM_SAMPLES  = 50
MAX_TURNS    = 4         # +1 vs before: extra room for the answer turn
NUM_ROLLOUTS = 2
TRAIN_STEPS  = 40
LR           = 5e-6
LOG_EVERY    = 10
HUB_REPO_ID  = "qwen3b-toolbench-grpo"
WANDB_PROJECT = "qwen-grpo-multiturn"

SYSTEM_PROMPT = """You are a helpful AI assistant with access to real-world APIs and tools.

For each user request:
1. Decide which tool to call and respond ONLY with valid JSON:
   {"name": "<tool_name>", "arguments": {<key>: <value>}}

2. After receiving a tool result, either:
   - Call another tool if more information is needed
   - Give a final natural language answer if the task is complete

Never explain your reasoning — just output the tool call JSON or the final answer."""

WRAPUP_MSG = "You now have all the tool results you need. Give your final answer in plain text — do NOT call any more tools."

# ── FAKE TOOL EXECUTOR ────────────────────────────────────────
def fake_tool_executor(tool_name: str, arguments: dict) -> str:
    return json.dumps(execute_fixture_tool(tool_name, arguments))

# ── PARSING HELPERS ───────────────────────────────────────────
def extract_json(text: str):
    text = text.strip()
    try:
        start = text.index("{");  end = text.rindex("}") + 1
        return json.loads(text[start:end]), True
    except:
        return None, False

def is_tool_call(obj):
    if obj is None:
        return False
    return "name" in obj and isinstance(obj.get("arguments"), dict)

def is_final_answer(text, obj):
    if obj is not None and is_tool_call(obj):
        return False
    return len(text.strip()) > 20

# ── REWARD FUNCTION ───────────────────────────────────────────
def score_full_chain(turns: list) -> float:
    if not turns:
        return -1.0

    score = 0.0
    n     = len(turns)

    # R1: first turn must be a tool call
    first_obj, first_valid = extract_json(turns[0])
    if first_valid and is_tool_call(first_obj):
        score += 0.25
    else:
        score -= 0.5

    # R2: tool calls should be valid and have non-empty args
    tool_turns    = turns[:-1] if n > 1 else turns
    valid_calls   = 0
    nonempty_args = 0
    seen_sigs     = []

    for turn in tool_turns:
        obj, valid = extract_json(turn)
        if valid and is_tool_call(obj):
            valid_calls += 1
            args = obj.get("arguments", {})
            if args:
                score += 0.15
                nonempty_args += 1
            # penalise repeated identical call
            sig = (obj.get("name", ""), frozenset(str(v) for v in args.values()))
            if sig in seen_sigs:
                score -= 0.3
            seen_sigs.append(sig)

    if tool_turns:
        score += 0.2 * (valid_calls / len(tool_turns))

    # penalise all-empty-args
    if valid_calls > 0 and nonempty_args == 0:
        score -= 0.3

    # R3: must end with a natural language answer
    last_obj, _ = extract_json(turns[-1])
    if is_final_answer(turns[-1], last_obj):
        score += 0.3
    else:
        score -= 0.3

    # R4: clean JSON outputs
    clean_count = 0
    for turn in turns:
        obj, valid = extract_json(turn)
        if valid:
            text = turn.strip()
            s = text.index("{");  e = text.rindex("}") + 1
            if len(text[:s].strip()) + len(text[e:].strip()) < 15:
                clean_count += 1
    if n > 0:
        score += 0.1 * (clean_count / n)

    # R5: chain length
    if 2 <= n <= 3:
        score += 0.1
    elif n == 1:
        score -= 0.15
    elif n > 4:
        score -= 0.05

    return round(max(-1.0, min(1.0, score)), 3)


# ── GENERATE ONE COMPLETE CHAIN ───────────────────────────────
def generate_chain(model, tokenizer, prompt_messages: list, max_turns: int) -> list:
    """Exact-state unassisted trajectories, sampled at temperature one, no top-k/p."""
    model.eval()  # identical dropout state for generation and gradient scoring
    history, turns, records = list(prompt_messages), [], []
    for _ in range(max_turns):
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80, temperature=1.0,
                do_sample=True, top_k=0, top_p=1.0, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)
        generated = tuple(output[0, inputs["input_ids"].shape[1]:].tolist())
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        turns.append(response)
        history.append({"role": "assistant", "content": response})
        obj = json_object(response)
        observation = fake_tool_executor(obj.get("name", ""), obj.get("arguments")) if obj and is_tool_call(obj) else None
        records.append(TurnRecord(tuple(inputs["input_ids"][0].tolist()),
            tuple(inputs["attention_mask"][0].tolist()), generated,
            str(getattr(model, "behavior_policy_version", "initial")),
            (("temperature", 1.0), ("top_k", 0), ("top_p", 1.0), ("max_new_tokens", 80), ("do_sample", True), ("eos_token_id", tokenizer.eos_token_id), ("pad_token_id", tokenizer.eos_token_id)), observation))
        if observation is None:
            break
        history.append({"role": "tool", "content": observation})
    return turns, tuple(records)


# ── COMPUTE REINFORCE LOSS ────────────────────────────────────
def compute_reinforce_loss(model, tokenizer, prompt_messages, records, reward, denominator=1):
    """Score immutable sampled state/action IDs; normalization precedes backward."""
    if not records or reward == 0:
        return None
    if denominator <= 0:
        raise ValueError("Positive sampled-chain denominator required")
    model.eval()
    total_loss = 0.0
    for record in records:
        if not isinstance(record, TurnRecord):
            raise ValueError("Exact immutable TurnRecord required; decoded strings are not actions")
        if not record.generated_ids:
            continue
        full = torch.tensor([record.input_ids + record.generated_ids], device=model.device)
        attention = torch.tensor([record.attention_mask + (1,) * len(record.generated_ids)], device=model.device)
        logits = model(input_ids=full, attention_mask=attention).logits[:, :-1]
        start = len(record.input_ids) - 1
        lp = F.log_softmax(logits[:, start:], dim=-1)
        labels = full[:, 1:][:, start:]
        loss = -reward * lp.gather(2, labels.unsqueeze(-1)).sum() / denominator
        loss.backward()
        total_loss += loss.item()
    return total_loss


# ── LOAD DATASET ─────────────────────────────────────────────
print("Loading ToolBench dataset ...")
raw = load_dataset("tuandunghcmut/toolbench-v1", split="train")
print(f"  Columns: {list(raw[0].keys())}")

def _parse_conversations(raw_convs):
    if raw_convs is None:
        return []
    if isinstance(raw_convs, str):
        try: raw_convs = json.loads(raw_convs)
        except: return []
    if isinstance(raw_convs, list):
        return raw_convs
    if isinstance(raw_convs, dict):
        froms  = raw_convs.get("from")  or raw_convs.get("role")    or []
        values = raw_convs.get("value") or raw_convs.get("content") or []
        if isinstance(froms, list) and isinstance(values, list):
            return [{"from": f, "value": v} for f, v in zip(froms, values)]
    return []

def _first_user_text(convs):
    for turn in convs:
        if not isinstance(turn, dict): continue
        role = turn.get("from") or turn.get("role") or ""
        text = turn.get("value") or turn.get("content") or ""
        if role.lower() in ("human", "user") and len(text.strip()) > 10:
            return text.strip()
    return ""

def extract_prompt(row):
    try:
        user_text = _first_user_text(_parse_conversations(row.get("conversations")))
        if not user_text:
            for f in ("instruction", "input", "query", "prompt", "question"):
                c = row.get(f, "")
                if isinstance(c, str) and len(c.strip()) > 10:
                    user_text = c.strip(); break
        if not user_text: return None
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_text}]
    except:
        return None

print("Extracting prompts ...")
prompts, skipped = [], 0
for row in raw:
    if len(prompts) >= NUM_SAMPLES: break
    p = extract_prompt(row)
    if p: prompts.append(p)
    else: skipped += 1

print(f"Loaded {len(prompts)} prompts (skipped {skipped})")
if not prompts:
    raise RuntimeError("No prompts loaded — check dataset schema.")

# ── LOAD MODEL ───────────────────────────────────────────────
print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model + SFT adapter ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    device_map=None if "LOCAL_RANK" in os.environ else "auto", trust_remote_code=True,
)
base.config.use_cache = False
model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
model.train()
print("Model ready.\n")

# ── OPTIMIZER ────────────────────────────────────────────────
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=5, num_training_steps=TRAIN_STEPS
)

# ── TRAINING LOOP ─────────────────────────────────────────────
wandb.init(project=WANDB_PROJECT, name="qwen-3b-multiturn-grpo")
print("=" * 60)
print("  TRUE MULTI-TURN GRPO TRAINING")
print("=" * 60)
print(f"  Model          : Qwen2.5-3B")
print(f"  Prompts        : {len(prompts)}")
print(f"  Rollouts/step  : {NUM_ROLLOUTS}")
print(f"  Max turns/chain: {MAX_TURNS}")
print(f"  Train steps    : {TRAIN_STEPS}")
print()
print("  Key fixes active:")
print("  • Protocol v2: unassisted stopping; exact sampled token records")
print("  • Attempt/token caps; sampled-chain loss denominator before backward")
print("  • Repeated calls penalised in reward (-0.3 each)")
print("  • Non-empty args rewarded (+0.15 each)")
print("=" * 60 + "\n")

step         = 0
total_loss   = 0.0
total_reward = 0.0
reward_attempts = 0

attempts = 0
sampled_tokens = 0
max_attempts = TRAIN_STEPS * 10
max_sampled_tokens = max_attempts * NUM_ROLLOUTS * MAX_TURNS * 80
while step < TRAIN_STEPS and attempts < max_attempts and sampled_tokens < max_sampled_tokens:
    attempts += 1
    prompt_messages = random.choice(prompts)

    chains, rewards = [], []
    for _ in range(NUM_ROLLOUTS):
        turns, records = generate_chain(model, tokenizer, prompt_messages, MAX_TURNS)
        reward = score_full_chain(turns)
        chains.append(records)
        rewards.append(reward)

    sampled_tokens += sum(len(r.generated_ids) for chain in chains for r in chain)
    mean_reward      = sum(rewards) / len(rewards)
    wandb.log({"attempt": attempts, "sampled_tokens": sampled_tokens, "sampled_reward": mean_reward, "reward_zvf": float(len(set(rewards)) == 1)})
    total_reward += mean_reward
    reward_attempts += 1
    relative_rewards = [r - mean_reward for r in rewards]

    optimizer.zero_grad()
    batch_loss  = 0.0
    valid_count = 0

    for turns, rel_reward in zip(chains, relative_rewards):
        if abs(rel_reward) < 1e-6:
            continue
        loss_val = compute_reinforce_loss(
            model, tokenizer, prompt_messages, turns, rel_reward, denominator=NUM_ROLLOUTS
        )
        if loss_val is not None:
            batch_loss  += loss_val
            valid_count += 1

    if valid_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss   += batch_loss
        step += 1
        model.behavior_policy_version = f"update-{step}"

        if step % LOG_EVERY == 0:
            avg_loss   = total_loss   / LOG_EVERY
            avg_reward = total_reward / reward_attempts
            total_loss = total_reward = 0.0
            reward_attempts = 0

            chain_summary = " → ".join([
                "TOOL" if is_tool_call(extract_json(t)[0]) else "ANSWER"
                for t in [tokenizer.decode(r.generated_ids, skip_special_tokens=True) for r in chains[0]]
            ])
            print(f"Step {step:3d}/{TRAIN_STEPS} | "
                  f"loss={avg_loss:.4f} | "
                  f"avg_reward={avg_reward:.3f} | "
                  f"chain=[{chain_summary}] reward={rewards[0]:.2f}")
            
            wandb.log({
                "train/loss": avg_loss,
                "train/reward": avg_reward,
                "train/step": step
            })

wandb.log({"termination_reason": "update_budget" if step >= TRAIN_STEPS else "attempt_or_token_budget", "attempted_groups": attempts, "optimizer_updates": step})

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nPushing to HuggingFace Hub: {HUB_REPO_ID}")
model.push_to_hub(HUB_REPO_ID)
tokenizer.push_to_hub(HUB_REPO_ID)

print(f"\n✅ GRPO adapter saved to {OUTPUT_DIR} and pushed to hub")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")

# ── QUICK CHAIN TEST ──────────────────────────────────────────
print("\n── Final chain test ──")
test_prompt = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "Get the weather in Paris and search for things to do there."},
]
model.eval()
turns, _records = generate_chain(model, tokenizer, test_prompt, MAX_TURNS)
print(f"User: {test_prompt[1]['content']}")
for i, t in enumerate(turns):
    obj, valid = extract_json(t)
    label = f"Tool call {i+1}" if (valid and is_tool_call(obj)) else "Final answer"
    print(f"  [{label}]: {t[:120]}")
print(f"  Chain score: {score_full_chain(turns)}")
print("\n✅ Step 2 complete! Run Step 3 (Eval) next.")
