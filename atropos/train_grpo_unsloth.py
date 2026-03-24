"""
train_grpo_unsloth.py — Drop-in Unsloth replacement for Atropos + Tinker RL.

Reads any existing configs/gsm8k_*.yaml config and runs GRPO training with
Unsloth's memory-efficient kernels via TRL's GRPOTrainer.

Preserves the exact reward function from gsm8k_tinker.py (math_verify +
latex2sympy2_extended) and emits the same WandB metric keys.

Usage:
    python train_grpo_unsloth.py --config configs/gsm8k_qwen_0_6b.yaml
    python train_grpo_unsloth.py --config configs/gsm8k_qwen_8b.yaml --seed 1

Tier mapping (matches VRAM budget doc):
    0.6B  → free T4  (no quantisation needed)
    1.7B  → free T4
    4B    → free T4  (4-bit quant) / Colab A100
    8B    → Colab A100 / Vast.ai A100
    14B   → A100 80 GB
    30B   → A100 80 GB
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List

import yaml

# ── Pre-import peft/transformers before wandb installs its import hooks ─────
# wandb monkey-patches importlib; importing these first avoids the lazy-load
# chain: peft→transformers→image_utils→torchvision (broken nms in 2.3.0 env)
try:
    import transformers  # noqa: F401
    from peft import LoraConfig  # noqa: F401
except Exception:
    pass  # will surface a cleaner error later when actually used

# ── WandB (required) ────────────────────────────────────────────────────────
import wandb


# ── reward helpers (verbatim logic from gsm8k_tinker.py) ────────────────────

def _setup_math_verify():
    """Import math_verify lazily so the module works without it for tests."""
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    return NormalizationConfig, LatexExtractionConfig, parse, verify


def _extract_gold(answer_raw: str) -> str:
    """GSM8K gold answers are in the form '... #### 42'."""
    return "\\boxed{" + answer_raw.split("#")[-1].strip().replace(",", "") + "}"


def _score_response(response: str, gold_boxed: str) -> float:
    """
    Return 1.0 if response contains a correct \\boxed{} answer, else 0.0.
    Mirrors GSM8kEnv.score() exactly.
    """
    NormalizationConfig, LatexExtractionConfig, parse, verify = _setup_math_verify()

    gold_parsed = parse(
        gold_boxed,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if not gold_parsed:
        return 0.0

    # Strip <think>…</think> if present (Qwen3 thinking mode)
    response_tail = response.split("</think>")[-1]

    answer_parsed = parse(
        response_tail,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return 1.0 if verify(answer_parsed, gold_parsed) else 0.0


def _completion_to_text(completion) -> str:
    """
    Normalize TRL completion payloads across versions.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion and isinstance(completion["content"], str):
            return completion["content"]
        if "text" in completion and isinstance(completion["text"], str):
            return completion["text"]
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
        if isinstance(last, str):
            return last
    return str(completion)


# Few-shot prefix — identical to gsm8k_tinker.py's convo_prefix
_QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
_FEW_SHOT_Q = "How many r's are in strawberry?" + _QUESTION_SUFFIX
_FEW_SHOT_A = (
    "Let's spell the word out and number all the letters: "
    "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
    "We have r's at positions 3, 8, and 9. \\boxed{3}"
)


def build_prompt(question: str, tokenizer, use_prefix: bool = True) -> str:
    """
    Format a GSM8K question into the chat template expected by the model.
    Matches the message structure in gsm8k_tinker.py.
    """
    messages = []
    if use_prefix:
        messages += [
            {"role": "user",      "content": _FEW_SHOT_Q},
            {"role": "assistant", "content": _FEW_SHOT_A},
        ]
    messages.append({"role": "user", "content": question + _QUESTION_SUFFIX})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for base models whose chat template references 'tokenizer'
        # or is otherwise broken in the Jinja2 sandbox (transformers 5.x)
        parts = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts) + "\nAssistant:"


# ── config loader ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("env", {})
    t   = raw.get("tinker", {})
    oi  = (raw.get("openai") or [{}])[0]

    return {
        "model_name":       oi.get("model_name") or cfg.get("tokenizer_name"),
        "total_steps":      cfg.get("total_steps",       50),
        "batch_size":       cfg.get("batch_size",        128),
        "group_size":       cfg.get("group_size",        16),
        "max_token_length": cfg.get("max_token_length",  512),
        "max_token_trainer_length": t.get("max_token_trainer_length", 2048),
        "use_prompt_prefix": cfg.get("use_prompt_prefix", True),
        "data_seed":        cfg.get("data_seed", 42),
        "lora_rank":        t.get("lora_rank",           32),
        "learning_rate":    t.get("learning_rate",       4e-5),
        "wandb_project":    t.get("wandb_project",       "tinker-rl-scaling"),
        "wandb_group":      t.get("wandb_group",         "unsloth-runs"),
        "wandb_run_name":   t.get("wandb_run_name",      "grpo-run"),
        "checkpoint_dir":   t.get("checkpoint_dir",      "./checkpoints/run/"),
    }


# ── VRAM-aware load strategy ─────────────────────────────────────────────────

def _param_count_B(model_name: str) -> float:
    """Heuristic: extract parameter count from model name."""
    m = re.search(r"(\d+\.?\d*)[Bb]", model_name.split("/")[-1])
    return float(m.group(1)) if m else 8.0


def load_model_and_tokenizer(model_name: str, lora_rank: int, max_seq_len: int):
    """
    Load model with Unsloth. Uses 4-bit quantisation for ≥4B models to fit
    smaller GPUs; full BF16 for tiny models.
    """
    params_b = _param_count_B(model_name)
    load_in_4bit = os.environ.get("ATROPOS_FORCE_4BIT", "0").lower() in {"1", "true", "yes"}
    if not load_in_4bit:
        load_in_4bit = params_b > 8.0

    use_unsloth = os.environ.get("ATROPOS_USE_UNSLOTH", "0").lower() in {"1", "true", "yes"}
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            print(f"  Loading {model_name} with Unsloth ({params_b}B, 4-bit={load_in_4bit}) ...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_len,
                load_in_4bit=load_in_4bit,
                dtype=None,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_rank,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            backend = "unsloth"
        except Exception as exc:
            print(f"  Unsloth load failed, falling back to Transformers/PEFT: {exc}")
            model = None
            tokenizer = None
            backend = "hf"
    else:
        model = None
        tokenizer = None
        backend = "hf"

    if backend == "hf":
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"  Loading {model_name} with Transformers/PEFT ({params_b}B, 4-bit={load_in_4bit}) ...")
        quantization_config = None
        compute_dtype = torch.float32 if load_in_4bit else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        # BnB 4-bit requires entire model on GPU (no CPU offload).
        # Use device_map={"":0} when quantizing to avoid auto-offloading to CPU.
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": 0} if load_in_4bit else "auto",
            "torch_dtype": compute_dtype,
        }
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = quantization_config

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.config.use_cache = False
        model.config.torch_dtype = compute_dtype

        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                output_embeddings.to(torch.float32)
        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # Ensure chat template exists (needed for base models)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\n' + message['content'] + tokenizer.eos_token + '\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ── dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(tokenizer, use_prefix: bool = True, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed)

    def _format(example):
        prompt = build_prompt(example["question"], tokenizer, use_prefix)
        return {
            "prompt":      prompt,
            "gold_boxed":  _extract_gold(example["answer"]),
        }

    return ds.map(_format, remove_columns=ds.column_names)


# ── reward function for GRPOTrainer ──────────────────────────────────────────

def make_reward_fn():
    """Return a reward function compatible with TRL GRPOTrainer."""

    def reward_fn(completions: List[str], prompts=None, **kwargs) -> List[float]:
        # GRPOTrainer passes gold answers via kwargs when the dataset has them.
        # We store gold_boxed in the dataset column and TRL surfaces it here.
        gold_list = kwargs.get("gold_boxed", None)
        rewards = []
        for i, completion in enumerate(completions):
            if gold_list is not None:
                gold = gold_list[i] if isinstance(gold_list[i], str) else gold_list[i][0]
            else:
                gold = ""   # fallback (shouldn't happen)
            rewards.append(_score_response(_completion_to_text(completion), gold))
        return rewards

    return reward_fn


# ── WandB logging helper ──────────────────────────────────────────────────────

class StepLogger:
    """Accumulates per-completion scores and logs step-level metrics."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.step_scores: list[float] = []
        self.step_log: list[float] = []   # per-step mean rewards

    def record(self, scores: List[float]):
        self.step_scores.extend(scores)

    def flush(self, step: int):
        if not self.step_scores:
            return
        mean_r = sum(self.step_scores) / len(self.step_scores)
        self.step_log.append(mean_r)
        metrics = {
            "train/percent_correct": mean_r,
            "train/step": step,
        }
        wandb.log(metrics, step=step)
        print(f"  step {step:3d}  mean_reward={mean_r:.4f}  "
              f"n={len(self.step_scores)}")
        self.step_scores = []

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("step,mean_reward\n")
            for i, r in enumerate(self.step_log):
                f.write(f"{i},{r:.6f}\n")
        print(f"  Reward log saved → {path}")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def maybe_push_to_hub(final_dir: str, cfg: dict, config_path: str, seed: int) -> None:
    """
    Upload the saved adapter/tokenizer directory to Hugging Face when enabled.
    """
    if not _bool_env("HF_PUSH", default=False):
        return

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_PUSH=1 requires HF_TOKEN to be set.")

    api = HfApi(token=token)
    owner = os.environ.get("HF_REPO_OWNER")
    if not owner:
        owner = api.whoami(token=token)["name"]

    repo_name = os.environ.get("HF_REPO_NAME")
    if not repo_name:
        repo_name = cfg["wandb_run_name"]
        if seed != 42:
            repo_name = f"{repo_name}-seed{seed}"

    repo_id = f"{owner}/{repo_name}"
    private = _bool_env("HF_PUSH_PRIVATE", default=True)
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=final_dir,
        repo_type="model",
        token=token,
        commit_message=f"Upload adapter for {cfg['model_name']} ({cfg['wandb_run_name']})",
    )
    print(f"  Hugging Face upload complete → {repo_id}")


# ── main training loop ───────────────────────────────────────────────────────

def train(config_path: str, seed: int = 42, wandb_api_key: str | None = None):
    cfg = load_config(config_path)
    print(f"\n{'='*60}")
    print(f"  GRPO (Unsloth) — {cfg['model_name']}")
    print(f"  Config: {config_path}")
    print(f"  Steps: {cfg['total_steps']}  |  batch: {cfg['batch_size']}  "
          f"|  group: {cfg['group_size']}  |  seed: {seed}")
    print(f"{'='*60}\n")

    # ── WandB init ──────────────────────────────────────────────────────────
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    run_name = f"{cfg['wandb_run_name']}-seed{seed}" if seed != 42 else cfg["wandb_run_name"]
    wandb.init(
        project=cfg["wandb_project"],
        group=cfg["wandb_group"],
        name=run_name,
        config={**cfg, "seed": seed, "config_file": config_path},
    )
    logger = StepLogger(cfg["wandb_run_name"])

    # ── model + tokenizer ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"], cfg["lora_rank"], cfg["max_token_trainer_length"]
    )

    # ── dataset ─────────────────────────────────────────────────────────────
    dataset = prepare_dataset(
        tokenizer,
        use_prefix=cfg["use_prompt_prefix"],
        seed=cfg["data_seed"],
    )

    # ── GRPOTrainer ─────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    per_device_train_batch_size = cfg["group_size"]
    gradient_accumulation_steps = max(1, cfg["batch_size"] // per_device_train_batch_size)

    # num_generations = group_size (TRL calls it num_generations)
    grpo_config = GRPOConfig(
        output_dir=cfg["checkpoint_dir"],
        num_train_epochs=1,
        max_steps=cfg["total_steps"],
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=cfg["group_size"],
        max_completion_length=cfg["max_token_length"],
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        save_steps=10,
        seed=seed,
        report_to="wandb",
        run_name=cfg["wandb_run_name"],
        # Unsloth-compatible settings
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    reward_fn = make_reward_fn()

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # Attach step-level logging via callback
    from transformers import TrainerCallback

    class RewardLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            # TRL logs reward/mean from the reward function
            mean_r = logs.get("reward/mean", logs.get("rewards/mean", None))
            if mean_r is not None:
                logger.step_log.append(float(mean_r))
                logger.flush(step)

    trainer.add_callback(RewardLogCallback())

    # ── run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min")

    # Save reward log as CSV for offline analysis
    csv_path = os.path.join(cfg["checkpoint_dir"], "reward_log.csv")
    logger.save_csv(csv_path)

    final_dir = os.path.join(cfg["checkpoint_dir"], "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))
    shutil.copy2(csv_path, os.path.join(final_dir, "reward_log.csv"))
    maybe_push_to_hub(final_dir, cfg, config_path, seed)

    wandb.finish()

    return logger.step_log


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training with Unsloth (drop-in for Atropos+Tinker)"
    )
    parser.add_argument("--config",  required=True, help="Path to YAML config")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--wandb_key", default=None, help="WandB API key (or set WANDB_API_KEY)")
    args = parser.parse_args()

    train(
        config_path=args.config,
        seed=args.seed,
        wandb_api_key=args.wandb_key or os.environ.get("WANDB_API_KEY"),
    )
