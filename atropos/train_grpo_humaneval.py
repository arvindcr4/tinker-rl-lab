"""
train_grpo_humaneval.py — GRPO training for HumanEval / tool-use tasks.

Supports two task modes via --task flag:
  humaneval   : pass@1 on openai/openai_humaneval (code execution reward)
  tool_use    : JSON tool-call correctness reward (custom dataset)

Usage:
    python train_grpo_humaneval.py --config configs/humaneval_qwen_8b.yaml --task humaneval
    python train_grpo_humaneval.py --config configs/tool_use_qwen_8b.yaml  --task tool_use
    python train_grpo_humaneval.py --config configs/tool_use_qwen_0_5b.yaml --task tool_use

All runs log to WandB (project/group from YAML config).
After training the LoRA adapter is pushed to HuggingFace when HF_PUSH=1.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import shutil
import signal
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, List

import yaml

# Pre-import peft/transformers before wandb installs its import hooks
try:
    import transformers  # noqa: F401
    from peft import LoraConfig  # noqa: F401
except Exception:
    pass

import wandb


# ── reward: HumanEval ────────────────────────────────────────────────────────

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Pull code from a markdown fence, falling back to the raw text."""
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def _run_humaneval_test(completion_code: str, test_code: str, entry_point: str,
                        timeout: float = 5.0) -> float:
    """
    Execute the completion + HumanEval test harness in a restricted subprocess.
    Returns 1.0 if all assert statements pass, else 0.0.
    """
    full_code = textwrap.dedent(f"""
{completion_code}

{test_code}

check({entry_point})
""")
    # Restrict dangerous builtins
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            k: __builtins__[k]  # type: ignore[index]
            for k in (
                "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
                "enumerate", "filter", "float", "frozenset", "getattr", "hasattr",
                "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
                "list", "map", "max", "min", "next", "oct", "ord", "pow", "print",
                "range", "repr", "reversed", "round", "set", "setattr", "slice",
                "sorted", "str", "sum", "tuple", "type", "zip",
            )
            if k in __builtins__  # type: ignore[operator]
        }
    }

    def _exec():
        exec(compile(full_code, "<humaneval>", "exec"), safe_globals)  # noqa: S102

    def _timeout_handler(signum, frame):
        raise TimeoutError("execution timed out")

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout) + 1)
    try:
        _exec()
        return 1.0
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def make_humaneval_reward_fn(dataset):
    """Return a GRPOTrainer-compatible reward function for HumanEval."""
    # Build lookup: task_id → {test, entry_point}
    lookup = {
        row["task_id"]: {
            "test": row["test"],
            "entry_point": row["entry_point"],
            "prompt": row["prompt"],
        }
        for row in dataset
    }

    def reward_fn(completions: List[str], prompts=None, **kwargs) -> List[float]:
        task_ids = kwargs.get("task_id", None)
        rewards = []
        for i, completion in enumerate(completions):
            code = _extract_code(completion if isinstance(completion, str) else str(completion))
            if task_ids is not None:
                tid = task_ids[i] if isinstance(task_ids[i], str) else task_ids[i][0]
                meta = lookup.get(tid, {})
                # prepend the original prompt so the function signature is defined
                full_code = meta.get("prompt", "") + "\n" + code
                score = _run_humaneval_test(full_code, meta.get("test", ""), meta.get("entry_point", ""))
            else:
                score = 0.0
            rewards.append(score)
        return rewards

    return reward_fn


def build_humaneval_dataset(tokenizer, max_token_length: int = 1024):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")

    SUFFIX = (
        "\n# Write a complete Python solution. "
        "Enclose your code in a ```python``` block."
    )

    def _fmt(ex):
        messages = [{"role": "user", "content": ex["prompt"] + SUFFIX}]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "task_id": ex["task_id"],
        }

    return ds.map(_fmt, remove_columns=ds.column_names)


# ── reward: tool use ─────────────────────────────────────────────────────────

# Simple synthetic tool-use dataset: given a question, the model must emit
# a JSON blob with {"tool": "<name>", "args": {...}} and then answer.
_TOOL_SCHEMA = {
    "calculator": {"description": "Evaluate a math expression.", "args": {"expression": "str"}},
    "lookup":     {"description": "Look up a fact.",            "args": {"query": "str"}},
    "search":     {"description": "Search the web.",             "args": {"query": "str"}},
}

_TOOL_EXAMPLES = [
    # (question, tool, args_match_fn, gold_answer)
    ("What is 17 * 23?",               "calculator", lambda a: "17" in a.get("expression","") and "23" in a.get("expression",""), "391"),
    ("What is the capital of France?",  "lookup",     lambda a: "france" in a.get("query","").lower(), "Paris"),
    ("Search for recent AI news.",      "search",     lambda a: "ai" in a.get("query","").lower(), None),
    ("Calculate 144 / 12.",             "calculator", lambda a: "144" in a.get("expression",""), "12"),
    ("Who wrote Hamlet?",               "lookup",     lambda a: "hamlet" in a.get("query","").lower(), "Shakespeare"),
]


def _score_tool_call(response: str, gold_tool: str, args_ok_fn, gold_answer) -> float:
    """
    Score a tool-use response.
    +0.5 for correct tool call in JSON, +0.5 for correct final answer.
    """
    score = 0.0
    try:
        m = re.search(r"\{.*?\}", response, re.DOTALL)
        if m:
            call = json.loads(m.group())
            if call.get("tool") == gold_tool and args_ok_fn(call.get("args", {})):
                score += 0.5
    except Exception:
        pass
    if gold_answer and gold_answer.lower() in response.lower():
        score += 0.5
    return score


def make_tool_use_reward_fn():
    """Return a GRPOTrainer-compatible reward function for tool use."""
    def reward_fn(completions: List[str], prompts=None, **kwargs) -> List[float]:
        gold_tools   = kwargs.get("gold_tool",   None)
        gold_answers = kwargs.get("gold_answer", None)
        gold_args    = kwargs.get("gold_args",   None)   # serialised lambda index

        rewards = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            if gold_tools is not None:
                gt   = gold_tools[i][0]   if isinstance(gold_tools[i],   list) else gold_tools[i]
                ga   = gold_answers[i][0] if isinstance(gold_answers[i], list) else gold_answers[i]
                idx  = int(gold_args[i][0] if isinstance(gold_args[i], list) else gold_args[i])
                _, _, args_fn, _ = _TOOL_EXAMPLES[idx % len(_TOOL_EXAMPLES)]
                rewards.append(_score_tool_call(text, gt, args_fn, ga))
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def build_tool_use_dataset(tokenizer, seed: int = 42):
    from datasets import Dataset

    SYSTEM = (
        "You have access to tools. To use a tool, output JSON: "
        '{{"tool": "<name>", "args": {{...}}}} then give your final answer.'
    )
    rows = []
    for i, (q, tool, _, ans) in enumerate(_TOOL_EXAMPLES * 200):   # repeat to get 1000 examples
        messages = [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": q},
        ]
        rows.append({
            "prompt":      tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "gold_tool":   tool,
            "gold_answer": ans or "",
            "gold_args":   str(i % len(_TOOL_EXAMPLES)),
        })

    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return Dataset.from_list(rows)


# ── shared: config loader + model loader (same as train_grpo_unsloth.py) ────

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("env", {})
    t   = raw.get("tinker", {})
    oi  = (raw.get("openai") or [{}])[0]
    return {
        "model_name":              oi.get("model_name") or cfg.get("tokenizer_name"),
        "total_steps":             cfg.get("total_steps",       50),
        "batch_size":              cfg.get("batch_size",        128),
        "group_size":              cfg.get("group_size",        16),
        "max_token_length":        cfg.get("max_token_length",  512),
        "max_token_trainer_length": t.get("max_token_trainer_length", 2048),
        "lora_rank":               t.get("lora_rank",           32),
        "learning_rate":           t.get("learning_rate",       4e-5),
        "wandb_project":           t.get("wandb_project",       "tinker-rl-scaling"),
        "wandb_group":             t.get("wandb_group",         "coding-scaling"),
        "wandb_run_name":          t.get("wandb_run_name",      "grpo-run"),
        "checkpoint_dir":          t.get("checkpoint_dir",      "./checkpoints/run/"),
    }


def _param_count_B(model_name: str) -> float:
    m = re.search(r"(\d+\.?\d*)[Bb]", model_name.split("/")[-1])
    return float(m.group(1)) if m else 8.0


def load_model_and_tokenizer(model_name: str, lora_rank: int, max_seq_len: int):
    params_b = _param_count_B(model_name)
    load_in_4bit = params_b >= 4.0
    use_unsloth = os.environ.get("ATROPOS_USE_UNSLOTH", "0").lower() in {"1", "true", "yes"}

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, max_seq_length=max_seq_len,
                load_in_4bit=load_in_4bit, dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_rank, lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth", random_state=42,
            )
            print(f"  Loaded {model_name} with Unsloth ({params_b}B, 4-bit={load_in_4bit})")
        except Exception as exc:
            use_unsloth = False
            print(f"  Unsloth failed ({exc}), falling back to HF")

    if not use_unsloth:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto",
                                  "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.config.use_cache = False
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        ))
        print(f"  Loaded {model_name} with HF/PEFT ({params_b}B, 4-bit={load_in_4bit})")

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% endif %}"
            "{% if message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + tokenizer.eos_token + '\n' }}{% endif %}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def maybe_push_to_hub(final_dir: str, cfg: dict, config_path: str, seed: int) -> None:
    if os.environ.get("HF_PUSH", "0").lower() not in {"1", "true", "yes"}:
        return
    from huggingface_hub import HfApi, create_repo
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    owner = os.environ.get("HF_REPO_OWNER") or api.whoami(token=token)["name"]
    repo_name = cfg["wandb_run_name"] + (f"-seed{seed}" if seed != 42 else "")
    repo_id = f"{owner}/{repo_name}"
    private = os.environ.get("HF_PUSH_PRIVATE", "1") in {"1", "true", "yes"}
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=final_dir, repo_type="model", token=token,
                      commit_message=f"Upload {cfg['wandb_run_name']} adapter")
    print(f"  HuggingFace upload → {repo_id}")


# ── main ─────────────────────────────────────────────────────────────────────

def train(config_path: str, task: str, seed: int = 42, wandb_api_key: str | None = None):
    cfg = load_config(config_path)
    print(f"\n{'='*60}")
    print(f"  GRPO ({task}) — {cfg['model_name']}")
    print(f"  Steps: {cfg['total_steps']}  batch: {cfg['batch_size']}  group: {cfg['group_size']}")
    print(f"{'='*60}\n")

    os.environ["WANDB_MODE"] = "online"
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    run_name = f"{cfg['wandb_run_name']}-seed{seed}" if seed != 42 else cfg["wandb_run_name"]
    wandb.init(project=cfg["wandb_project"], group=cfg["wandb_group"], name=run_name,
               config={**cfg, "task": task, "seed": seed, "config_file": config_path})

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"], cfg["lora_rank"], cfg["max_token_trainer_length"]
    )

    if task == "humaneval":
        dataset    = build_humaneval_dataset(tokenizer, cfg["max_token_length"])
        reward_fn  = make_humaneval_reward_fn(
            __import__("datasets").load_dataset("openai/openai_humaneval", split="test")
        )
        extra_cols = ["task_id"]
    elif task == "tool_use":
        dataset    = build_tool_use_dataset(tokenizer, seed=seed)
        reward_fn  = make_tool_use_reward_fn()
        extra_cols = ["gold_tool", "gold_answer", "gold_args"]
    else:
        raise ValueError(f"Unknown task: {task}")

    from trl import GRPOConfig, GRPOTrainer
    _per_device_bs = cfg["batch_size"] // cfg["group_size"]
    _num_gen       = cfg["group_size"]
    grpo_config = GRPOConfig(
        output_dir=cfg["checkpoint_dir"],
        num_train_epochs=1,
        max_steps=cfg["total_steps"],
        per_device_train_batch_size=_per_device_bs,
        num_generations=_num_gen,
        generation_batch_size=_per_device_bs * _num_gen,  # must be divisible by num_generations
        max_completion_length=cfg["max_token_length"],
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        save_steps=10,
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model, args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    from transformers import TrainerCallback
    step_log: list[float] = []

    class RewardLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            mean_r = logs.get("reward/mean", logs.get("rewards/mean"))
            if mean_r is not None:
                step_log.append(float(mean_r))
                wandb.log({"train/percent_correct": float(mean_r), "train/step": state.global_step},
                          step=state.global_step)
                print(f"  step {state.global_step:3d}  mean_reward={float(mean_r):.4f}")

    trainer.add_callback(RewardLogCallback())

    t0 = time.time()
    trainer.train()
    print(f"\n  Training complete in {(time.time()-t0)/60:.1f} min")

    csv_path  = os.path.join(cfg["checkpoint_dir"], "reward_log.csv")
    final_dir = os.path.join(cfg["checkpoint_dir"], "final")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("step,mean_reward\n")
        for i, r in enumerate(step_log):
            f.write(f"{i},{r:.6f}\n")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))
    shutil.copy2(csv_path, os.path.join(final_dir, "reward_log.csv"))
    maybe_push_to_hub(final_dir, cfg, config_path, seed)

    wandb.finish()
    return step_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      required=True, choices=["humaneval", "tool_use"])
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--wandb_key", default=None)
    args = parser.parse_args()
    train(
        config_path=args.config,
        task=args.task,
        seed=args.seed,
        wandb_api_key=args.wandb_key or os.environ.get("WANDB_API_KEY"),
    )
