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

import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass


import argparse
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_grpo_unsloth")

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
try:
    import torch, wandb
    if not getattr(wandb, '_vram_patched', False):
        _old_log = wandb.log
        def _vram_log(data, *args, **kwargs):
            if torch.cuda.is_available():
                data['system/vram_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
                data['system/vram_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
            _old_log(data, *args, **kwargs)
        wandb.log = _vram_log
        wandb._vram_patched = True
except ImportError:
    pass


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

def _generative_score_response(prompt: str, response: str, gold_raw: str, api_url: str = "http://localhost:8001/v1") -> float:
    """
    Area 10: Generative Verifier.
    Uses the local LLM inference server to generate a reasoning trace and verify correctness.
    """
    import requests
    system_prompt = (
        "You are a strict math teacher grading a student's answer. "
        "First, write a short reasoning trace comparing the student's answer to the exact gold answer. "
        "Then, if they are mathematically equivalent, end your response with exactly: <SCORE>1</SCORE>. "
        "Otherwise, end with exactly: <SCORE>0</SCORE>."
    )
    user_prompt = f"Question: {prompt}\n\nGold Answer: {gold_raw}\n\nStudent Answer: {response}"
    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 128
            },
            timeout=10.0
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if "<SCORE>1</SCORE>" in content:
                return 1.0
        return 0.0
    except Exception:
        # Fallback to 0.0 on timeout/error
        return 0.0

def _execution_score_response(response: str, gold_raw: str) -> float:
    """
    Area 5: Execution-Based Rewards.
    Extracts a Python block, executes it, captures stdout, and compares it to the gold answer.
    """
    import io, contextlib, signal, re
    
    # Extract Python code
    match = re.search(r"```python(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0
    code = match.group(1).strip()
    
    # Secure Sandbox
    safe_globals = {
        "__builtins__": {
            k: __builtins__[k] for k in (
                "abs", "all", "any", "bool", "dict", "float", "int", "len",
                "list", "map", "max", "min", "pow", "print", "range", "round",
                "set", "str", "sum", "tuple", "zip"
            ) if k in __builtins__
        },
        "math": __import__("math")
    }
    
    output = io.StringIO()
    def _timeout_handler(signum, frame):
        raise TimeoutError()
        
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(3)
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(code, "<math_exec>", "exec"), safe_globals)
        printed = output.getvalue().strip()
        gold_answer = gold_raw.split("#")[-1].strip().replace(",", "")
        return 1.0 if printed == gold_answer else 0.0
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

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
        # TODO: 30-50 steps is insufficient to observe meaningful RL convergence. Increase total_steps.
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
        "early_stopping_patience": t.get("early_stopping_patience", 3),
        "eval_steps":       t.get("eval_steps",          10),
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
    # Env var ATROPOS_FORCE_BF16 forces bf16 regardless of model size
    if os.environ.get("ATROPOS_FORCE_BF16", "0").lower() in {"1", "true", "yes"}:
        load_in_4bit = False

    use_unsloth = os.environ.get("ATROPOS_USE_UNSLOTH", "0").lower() in {"1", "true", "yes"}
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            logger.info(f"Loading {model_name} with Unsloth ({params_b}B, 4-bit={load_in_4bit}) ...")
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
            logger.warning(f"Unsloth load failed, falling back to Transformers/PEFT: {exc}")
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

        logger.info(f"Loading {model_name} with Transformers/PEFT ({params_b}B, 4-bit={load_in_4bit}) ...")
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
        # TRL GRPOTrainer accesses model.warnings_issued; ensure it exists even
        # after PEFT wrapping (transformers 5.x sets it on PreTrainedModel but
        # some new Qwen3 models omit it)
        if not hasattr(model.base_model.model, 'warnings_issued'):
            model.base_model.model.warnings_issued = {}

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

def prepare_dataset(tokenizer, use_prefix: bool = True, seed: int = 42, split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    if split == "train":
        ds = ds.shuffle(seed=seed)

    def _format(example):
        prompt = build_prompt(example["question"], tokenizer, use_prefix)
        return {
            "prompt":      prompt,
            "gold_boxed":  _extract_gold(example["answer"]),
        }

    return ds.map(_format, remove_columns=ds.column_names)


# ── reward function for GRPOTrainer ──────────────────────────────────────────

class StatefulRewardFunction:
    """A stateful reward function compatible with TRL GRPOTrainer, tracking metrics thread-safely."""
    
    def __init__(self, group_size: int, evaluation_mode: str = "static"):
        self.group_size = group_size
        self.evaluation_mode = evaluation_mode
        self.lock = threading.Lock()
        self.metrics = {}

    def __call__(self, completions: List[str], prompts=None, **kwargs) -> List[float]:
        # GRPOTrainer passes gold answers via kwargs when the dataset has them.
        # We store gold_boxed in the dataset column and TRL surfaces it here.
        gold_list = kwargs.get("gold_boxed", None)
        rewards = []
        comp_texts = []
        for i, completion in enumerate(completions):
            if gold_list is not None:
                gold = gold_list[i] if isinstance(gold_list[i], str) else gold_list[i][0]
            else:
                gold = ""   # fallback (shouldn't happen)
            
            comp_text = _completion_to_text(completion)
            comp_texts.append(comp_text)

            # Outcome Reward Model (ORM) based on Evaluation Mode
            if self.evaluation_mode == "generative":
                # Extract prompt text
                prompt_text = ""
                if prompts is not None and len(prompts) > i:
                    if isinstance(prompts[i], str):
                        prompt_text = prompts[i]
                    elif isinstance(prompts[i], list):
                        prompt_text = prompts[i][-1].get("content", "") if isinstance(prompts[i][-1], dict) else str(prompts[i][-1])
                base_reward = _generative_score_response(prompt_text, comp_text, gold)
            elif self.evaluation_mode == "execution":
                base_reward = _execution_score_response(comp_text, gold)
            else:
                base_reward = _score_response(comp_text, gold)
            
            # Process Reward Model (PRM)
            text_lower = comp_text.lower()
            step_reward = 0.0
            steps_found = text_lower.count("step") + text_lower.count("first") + text_lower.count("then")
            if steps_found > 0:
                step_reward += min(0.5, 0.1 * steps_found)
                
            rewards.append(base_reward + step_reward)
        
        # calculate zvf, advantage variance, and ACR
        zvf_sum = 0.0
        advantage_variances = []
        n_groups = len(rewards) // self.group_size
        if n_groups > 0:
            for idx in range(n_groups):
                chunk = rewards[idx*self.group_size:(idx+1)*self.group_size]
                var = np.var(chunk)
                advantage_variances.append(var)
                mr = sum(chunk) / self.group_size
                if all(abs(r - mr) < 1e-6 for r in chunk):
                    zvf_sum += 1.0
                    
            mean_adv_var = float(np.mean(advantage_variances))
            acr = 1.0 if mean_adv_var < 1e-4 else 0.0
            zvf = zvf_sum / n_groups
        else:
            mean_adv_var = 0.0
            acr = 0.0
            zvf = 0.0
            
        # Length confounding panel
        lengths = [len(t) for t in comp_texts]
        correct_lens = [l for l, r in zip(lengths, rewards) if r > 0.5]
        incorrect_lens = [l for l, r in zip(lengths, rewards) if r <= 0.5]
        
        mean_len_correct = float(np.mean(correct_lens)) if correct_lens else 0.0
        mean_len_incorrect = float(np.mean(incorrect_lens)) if incorrect_lens else 0.0
        
        if correct_lens and incorrect_lens:
            mean_y1 = mean_len_correct
            mean_y0 = mean_len_incorrect
            n1 = len(correct_lens)
            n0 = len(incorrect_lens)
            n = n1 + n0
            std_y = np.std(lengths)
            len_reward_corr = float(((mean_y1 - mean_y0) / (std_y + 1e-8)) * np.sqrt((n1 * n0) / (n * (n - 1))))
        else:
            len_reward_corr = 0.0

        with self.lock:
            self.metrics["zvf"] = zvf
            self.metrics["diagnostics/advantage_variance"] = mean_adv_var
            self.metrics["diagnostics/advantage_collapse_rate"] = acr
            self.metrics["diagnostics/mean_len_correct"] = mean_len_correct
            self.metrics["diagnostics/mean_len_incorrect"] = mean_len_incorrect
            self.metrics["diagnostics/length_reward_corr"] = len_reward_corr
            
        return rewards

    def get_metrics_and_reset(self) -> dict:
        with self.lock:
            metrics = self.metrics.copy()
            return metrics


# ── WandB logging helper ──────────────────────────────────────────────────────

class StepTracker:
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
        logger.info(f"step {step:3d}  mean_reward={mean_r:.4f}  n={len(self.step_scores)}")
        self.step_scores = []

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("step,mean_reward\n")
            for i, r in enumerate(self.step_log):
                f.write(f"{i},{r:.6f}\n")
        logger.info(f"Reward log saved → {path}")


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
    logger.info(f"Hugging Face upload complete → {repo_id}")


# ── main training loop ───────────────────────────────────────────────────────

def train(config_path: str, seed: int = 42, wandb_api_key: str | None = None,
          total_token_budget: int | None = None, tokens_per_sample: int = 2048,
          group_size_override: int | None = None, batch_size_override: int | None = None,
          evaluation_mode_override: str | None = None):
    cfg = load_config(config_path)
    if group_size_override:
        cfg['group_size'] = group_size_override
    if batch_size_override:
        cfg['batch_size'] = batch_size_override
        
    if total_token_budget:
        total_samples = total_token_budget / tokens_per_sample
        effective_batch_size = cfg['group_size']
        cfg['total_steps'] = max(1, int(total_samples / effective_batch_size))
        logger.info(f"[P1 Ablation] Matching token budget: {total_token_budget} tokens -> total_steps={cfg['total_steps']}")
        
    logger.info("\n" + "="*60)
    logger.info(f"GRPO (Unsloth) — {cfg['model_name']}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Steps: {cfg['total_steps']}  |  batch: {cfg['batch_size']}  |  group: {cfg['group_size']}  |  seed: {seed}")
    logger.info("="*60 + "\n")

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
    step_tracker = StepTracker(cfg["wandb_run_name"])

    # ── model + tokenizer ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"], cfg["lora_rank"], cfg["max_token_trainer_length"]
    )

    # ── dataset ─────────────────────────────────────────────────────────────
    dataset = prepare_dataset(
        tokenizer,
        use_prefix=cfg["use_prompt_prefix"],
        seed=cfg["data_seed"],
        split="train"
    )
    # TODO: Improve evaluation to rigorously prove generalized reasoning uplift on held-out test sets.
    eval_dataset = prepare_dataset(
        tokenizer,
        use_prefix=cfg["use_prompt_prefix"],
        seed=cfg["data_seed"],
        split="test"
    )

    # ── GRPOTrainer ─────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    per_device_train_batch_size = cfg["group_size"]
    gradient_accumulation_steps = max(1, cfg["batch_size"] // per_device_train_batch_size)

    # TODO: Implement micro-partitioning and reference offloading to close the performance gap with Tinker API.
    # num_generations = group_size (TRL calls it num_generations)
    grpo_config = GRPOConfig(
        output_dir=cfg["checkpoint_dir"],
        num_train_epochs=1,
        max_steps=cfg["total_steps"],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=cfg["group_size"],
        max_completion_length=cfg["max_token_length"],
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        save_steps=cfg["eval_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward/mean",
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

    eval_mode = evaluation_mode_override if evaluation_mode_override else cfg.get("evaluation_mode", "static")
    reward_fn = StatefulRewardFunction(cfg["group_size"], evaluation_mode=eval_mode)

    from transformers import EarlyStoppingCallback

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    # Attach step-level logging via callback
    from transformers import TrainerCallback

    class RewardLogCallback(TrainerCallback):
        def __init__(self, step_tracker, reward_fn: StatefulRewardFunction):
            self.step_tracker = step_tracker
            self.reward_fn = reward_fn

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            # TRL logs reward/mean from the reward function
            mean_r = logs.get("reward/mean", logs.get("rewards/mean", None))
            if mean_r is not None:
                self.step_tracker.step_log.append(float(mean_r))
                self.step_tracker.flush(step)
            
            metrics_to_log = self.reward_fn.get_metrics_and_reset()
            if metrics_to_log:
                wandb.log(metrics_to_log, step=step, commit=False)

    class GoodputCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()
            self.last_step_time = self.start_time
            self.total_steps = 0
            
        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            step_time = now - self.last_step_time
            self.last_step_time = now
            self.total_steps += 1
            
            # Goodput: total steps completed / elapsed time (steps per second)
            goodput = self.total_steps / (now - self.start_time)
            
            if wandb.run is not None:
                wandb.log({
                    "infrastructure/goodput_steps_per_sec": goodput,
                    "infrastructure/step_time_sec": step_time,
                }, step=state.global_step, commit=False)

    trainer.add_callback(RewardLogCallback(step_tracker, reward_fn))
    trainer.add_callback(GoodputCallback())

    # ── run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save reward log as CSV for offline analysis
    csv_path = os.path.join(cfg["checkpoint_dir"], "reward_log.csv")
    step_tracker.save_csv(csv_path)

    final_dir = os.path.join(cfg["checkpoint_dir"], "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))
    shutil.copy2(csv_path, os.path.join(final_dir, "reward_log.csv"))
    maybe_push_to_hub(final_dir, cfg, config_path, seed)

    wandb.finish()

    return step_tracker.step_log


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training with Unsloth (drop-in for Atropos+Tinker)"
    )
    parser.add_argument("--config",  required=True, help="Path to YAML config")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--wandb_key", default=None, help="WandB API key (or set WANDB_API_KEY)")
    parser.add_argument("--total_token_budget", type=int, default=None, help="Total token budget for causal SFT warm-up ablation")
    parser.add_argument("--tokens_per_sample", type=int, default=2048, help="Avg tokens per sample")
    parser.add_argument("--group_size", type=int, default=None, help="Override config group_size for Pareto ablation")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size for Pareto ablation")
    parser.add_argument("--evaluation_mode", type=str, default="static", choices=["static", "execution", "generative"], help="Reward evaluation mode")
    args = parser.parse_args()

    cfg = TinkerAtroposConfig.from_yaml(args.config)
    cfg.env.data_seed = args.seed
    if args.group_size:
        cfg.env.group_size = args.group_size
    if args.batch_size:
        cfg.env.batch_size = args.batch_size
    
    if args.total_token_budget:
        total_samples = args.total_token_budget / args.tokens_per_sample
        # For Unsloth trainer, batch size is group_size
        effective_batch_size = cfg.env.group_size
        cfg.env.total_steps = max(1, int(total_samples / effective_batch_size))
        logger.info(f"[P1 Ablation] Matching token budget: {args.total_token_budget} tokens -> total_steps={cfg.env.total_steps}")

    train(
        config_path=args.config,
        seed=args.seed,
        wandb_api_key=args.wandb_key or os.environ.get("WANDB_API_KEY"),
        total_token_budget=args.total_token_budget,
        tokens_per_sample=args.tokens_per_sample,
        group_size_override=args.group_size,
        batch_size_override=args.batch_size,
        evaluation_mode_override=args.evaluation_mode,
    )
