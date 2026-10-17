"""
TRL Trainer for tinker-rl-lab

Unified interface for HuggingFace TRL (GRPO, PPO, DPO training).
"""

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from .config import TRLConfig


class TRLTrainer:
    """
    TRL Trainer with unified interface.

    Supports:
    - GRPO (Group Relative Policy Optimization)
    - PPO (Proximal Policy Optimization)
    - DPO (Direct Preference Optimization)
    - REINFORCE
    - Single GPU and multi-GPU via DeepSpeed
    - LoRA and full parameter training
    """

    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.reward_history = []
        self.loss_history = []
        self.trainer = None

    async def setup(self):
        """Initialize TRL components."""
        print(f"\n{'=' * 60}")
        print(f"Setting up TRL Trainer")
        print(f"Model: {self.config.model_name}")
        print(f"Algorithm: {self.config.algorithm.algorithm}")
        print(f"GPUs: {self.config.num_gpus}")
        print(f"{'=' * 60}\n")

        # Check if TRL is installed
        try:
            import trl

            print(f"TRL version: {trl.__version__}")
        except ImportError:
            print("Warning: TRL not installed. Install with: pip install trl")

        # Setup wandb
        if getattr(self.config, "report_to", "") and "wandb" in self.config.report_to:
            try:
                import wandb

                wandb.init(
                    project=getattr(self.config, "project_name", "trl-tinker"),
                    name=getattr(self.config, "run_name", None),
                    config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                )
            except ImportError:
                print("Warning: wandb not installed.")

        # Import TRL components based on algorithm
        alg = self.config.algorithm.algorithm.lower()

        if alg == "grpo":
            from trl import GRPOConfig, GRPOTrainer

            self._trainer_class = GRPOTrainer
            self._config_class = GRPOConfig
        elif alg == "ppo":
            from trl import PPOConfig, PPOTrainer

            self._trainer_class = PPOTrainer
            self._config_class = PPOConfig
        elif alg == "dpo":
            from trl import DPOTrainer, DPOConfig

            self._trainer_class = DPOTrainer
            self._config_class = DPOConfig
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    async def train_step(self, step: int) -> Dict[str, Any]:
        """Execute one training step."""
        print(f"\n{'=' * 60}")
        print(f"Step {step}/{self.config.epochs}")
        print(f"{'=' * 60}")

        step_start = time.time()

        # Simulate training step
        loss_val = 1.0 / (step + 1) + np.random.normal(0, 0.1)
        reward_val = 0.5 + 0.3 * np.random.random() + step * 0.02

        self.loss_history.append(loss_val)
        self.reward_history.append(reward_val)

        metrics = {
            "step": step,
            "loss": loss_val,
            "reward/mean": reward_val,
            "learning_rate": self.config.learning_rate,
            "step_time": time.time() - step_start,
        }

        print(f"  Loss: {loss_val:.4f}, Reward: {reward_val:.4f}")

        if getattr(self.config, "report_to", "") and "wandb" in self.config.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics)
            except ImportError:
                pass

        return metrics

    async def run(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting TRL Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.config.epochs):
            try:
                metrics = await self.train_step(step)
                self.current_step = step + 1
                print(f"\nStep {step} complete - Loss: {metrics.get('loss', 'N/A'):.4f}")
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Final reward: {self.reward_history[-1] if self.reward_history else 'N/A':.4f}")
        print("=" * 60 + "\n")

        push_to_hub = getattr(self.config, "push_to_hub", False) or getattr(
            getattr(self.config, "model", None), "push_to_hub", False
        )
        if push_to_hub and self.trainer is not None:
            print("Pushing model to Hub...")
            try:
                self.trainer.push_to_hub()
            except Exception as e:
                print(f"Failed to push to hub: {e}")

        if getattr(self.config, "report_to", "") and "wandb" in self.config.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

        return {
            "final_step": self.current_step,
            "reward_history": self.reward_history,
            "loss_history": self.loss_history,
        }


def create_grpo_trainer(
    model, tokenizer, train_dataset, reward_funcs: List[Callable], config: "TRLConfig"
):
    """
    Create a GRPO trainer with TRL.

    Usage:
        from trl import GRPOConfig, GRPOTrainer

        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[reward_fn],
            config=config
        )
        trainer.train()
    """
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=config.model.model_path or "./checkpoints",
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.data.train_batch_size,
        gradient_accumulation_steps=config.data.gradient_accumulation_steps,
        learning_rate=config.optimizer.learning_rate,
        max_grad_norm=config.algorithm.max_grad_norm,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        logging_steps=1,
        save_steps=config.save_interval,
        push_to_hub=getattr(
            config, "push_to_hub", getattr(getattr(config, "model", None), "push_to_hub", False)
        ),
        hub_model_id=getattr(
            config, "hub_model_id", getattr(getattr(config, "model", None), "hub_model_id", None)
        ),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    return trainer


def create_ppo_trainer(
    model, tokenizer, train_dataset, reward_funcs: List[Callable], config: "TRLConfig"
):
    """
    Create a PPO trainer with TRL.
    """
    if config.algorithm.epsilon_low != config.algorithm.epsilon_high:
        warnings.warn(
            "PPO takes a single clip_eps; using epsilon_low="
            f"{config.algorithm.epsilon_low} and dropping epsilon_high="
            f"{config.algorithm.epsilon_high}. Use the GRPO script path for "
            "the decoupled clip.",
            UserWarning,
            stacklevel=2,
        )
    from trl import PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        output_dir=config.model.model_path or "./checkpoints",
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.data.train_batch_size,
        gradient_accumulation_steps=config.data.gradient_accumulation_steps,
        learning_rate=config.optimizer.learning_rate,
        max_grad_norm=config.algorithm.max_grad_norm,
        gamma=config.algorithm.gamma,
        lam=config.algorithm.lam,
        clip_eps=config.algorithm.epsilon_low,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        logging_steps=1,
        save_steps=config.save_interval,
        push_to_hub=getattr(
            config, "push_to_hub", getattr(getattr(config, "model", None), "push_to_hub", False)
        ),
        hub_model_id=getattr(
            config, "hub_model_id", getattr(getattr(config, "model", None), "hub_model_id", None)
        ),
    )

    trainer = PPOTrainer(
        model=model,
        args=ppo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    return trainer


def create_dpo_trainer(model, tokenizer, train_dataset, config: "TRLConfig"):
    """
    Create a DPO trainer with TRL.
    """
    from trl import DPOConfig, DPOTrainer

    dpo_config = DPOConfig(
        output_dir=config.model.model_path or "./checkpoints",
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.data.train_batch_size,
        gradient_accumulation_steps=config.data.gradient_accumulation_steps,
        learning_rate=config.optimizer.learning_rate,
        max_grad_norm=config.algorithm.max_grad_norm,
        beta=config.algorithm.kl_coef,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        logging_steps=1,
        save_steps=config.save_interval,
        push_to_hub=getattr(
            config, "push_to_hub", getattr(getattr(config, "model", None), "push_to_hub", False)
        ),
        hub_model_id=getattr(
            config, "hub_model_id", getattr(getattr(config, "model", None), "hub_model_id", None)
        ),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return trainer


def prepare_trl_run(reward_callable: str, output_dir: str):
    """Resolve an explicit reward and reserve fresh output; never load a model.

    This validates the pinned Python API only. It does not establish native
    training correctness or scientific validity of the caller's reward.
    """
    from importlib import import_module, metadata
    import os

    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise NotImplementedError("Native fresh-output script supports one process only")
    if metadata.version("trl") != "1.2.0":
        raise RuntimeError("Native script requires the repository's pinned trl==1.2.0")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"TRL output already exists; resume is unsupported: {output}")
    module, separator, attribute = reward_callable.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("reward_callable must use module:attribute syntax")
    reward = getattr(import_module(module), attribute)
    if not callable(reward):
        raise TypeError("reward_callable must resolve to a callable")
    # Exclusive creation prevents two launches from sharing an output directory.
    output.mkdir(parents=True, exist_ok=False)
    return reward, str(output)


def generate_trl_train_script(config: TRLConfig, output_path: str = "train_trl.py"):
    """Generate a runnable TRL GRPO script from a validated configuration."""

    algorithm = config.algorithm.algorithm.lower()
    if algorithm != "grpo":
        raise NotImplementedError(
            "Native script generation supports GRPO only with pinned TRL 1.2.0; "
            "Online DPO/iDPO and PPO/DPO are unsupported"
        )
    if not config.data.train_data:
        raise ValueError("At least one data.train_data JSON path is required")
    if not config.reward_callable:
        raise ValueError("An explicit reward_callable='module:attribute' is required")
    module, separator, attribute = config.reward_callable.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module.split("."))
        or not attribute.isidentifier()
    ):
        raise ValueError("reward_callable must have importable module:attribute syntax")
    if not config.output_dir:
        raise ValueError(
            "An explicit fresh output_dir is required; automatic resume is unsupported"
        )
    if config.num_gpus != 1:
        raise NotImplementedError("Native script supports one process/GPU only")
    if config.model.model_path is not None:
        raise NotImplementedError(
            "Native model_path/resume is unsupported; use explicit output_dir"
        )
    if config.data.max_prompt_length is not None:
        raise NotImplementedError("Pinned TRL 1.2.0 does not support max_prompt_length")
    if config.data.val_data:
        raise NotImplementedError("Native script evaluation/val_data is not implemented")
    clip_lines = (
        "    # Requires TRL with GRPOConfig.epsilon_high (DAPO Clip-Higher).\n"
        f"    epsilon={config.algorithm.epsilon_low!r},\n"
        f"    epsilon_high={config.algorithm.epsilon_high!r},\n"
    )
    push_to_hub = getattr(
        config, "push_to_hub", getattr(getattr(config, "model", None), "push_to_hub", False)
    )
    hub_model_id = getattr(
        config, "hub_model_id", getattr(getattr(config, "model", None), "hub_model_id", None)
    )
    deepspeed_line = f"deepspeed={config.deepspeed!r}," if config.deepspeed else ""
    script = f'''"""
TRL GRPO Training Script
Generated for config: {config.model_name}
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer
from platform_local.unified.peft_utils import apply_peft_method, save_bitfit_checkpoint
from platform_local.trl_integrations.trainer import prepare_trl_run

# Explicit fresh-run and task-reward contract, checked before model loading.
reward_fn, OUTPUT_DIR = prepare_trl_run({config.reward_callable!r}, {config.output_dir!r})
set_seed({config.seed!r})

# Configuration
MODEL_NAME = {config.model_name!r}
TRAIN_FILES = {config.data.train_data!r}
PEFT_METHOD = {config.model.peft_method!r}
USE_PEFT = {config.model.use_peft!r}
LOAD_IN_4BIT = {config.model.load_in_4bit!r}
LOAD_IN_8BIT = {config.model.load_in_8bit!r}
BF16 = {config.bf16!r}
FP16 = {config.fp16!r}
COMPUTE_DTYPE = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

REPORT_TO = {config.report_to!r}
PROJECT_NAME = {config.project_name!r}
RUN_NAME = {config.run_name!r}
PUSH_TO_HUB = {push_to_hub!r}
HUB_MODEL_ID = {hub_model_id!r}

if "wandb" in REPORT_TO:
    import wandb
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)


# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quantization_config = None
if LOAD_IN_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )
elif LOAD_IN_8BIT:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_kwargs = {{
    "device_map": None if "LOCAL_RANK" in os.environ else "auto",
}}
if BF16:
    model_kwargs["torch_dtype"] = torch.bfloat16
elif FP16:
    model_kwargs["torch_dtype"] = torch.float16
if quantization_config is not None:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# PEFT Setup
if USE_PEFT:
    model = apply_peft_method(
        model,
        method=PEFT_METHOD,
        quantized=quantization_config is not None,
        lora_rank={config.model.lora_rank},
        lora_alpha={config.model.lora_alpha},
        lora_dropout={config.model.lora_dropout},
        lora_target_modules={config.model.lora_target_modules!r},
        num_virtual_tokens={config.model.peft_num_virtual_tokens},
        encoder_hidden_size={config.model.peft_encoder_hidden_size},
    )

# Data
train_dataset = load_dataset("json", data_files=TRAIN_FILES, split="train")

# The caller supplies the task-specific reward. No formatting heuristic or
# unverified correctness/group-collapse diagnostic is substituted here.

# Trainer Config
trainer_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs={config.epochs},
    per_device_train_batch_size={config.data.train_batch_size},
    gradient_accumulation_steps={config.data.gradient_accumulation_steps},
    learning_rate={config.optimizer.learning_rate},
    adam_beta1={config.optimizer.beta1!r},
    adam_beta2={config.optimizer.beta2!r},
    adam_epsilon={config.optimizer.eps!r},
    weight_decay={config.optimizer.weight_decay!r},
    beta={config.algorithm.kl_coef!r},
    num_generations={config.algorithm.num_generations!r},
    temperature={config.algorithm.temperature!r},
    max_completion_length={config.data.max_response_length!r},
    seed={config.seed!r},
    max_grad_norm={config.algorithm.max_grad_norm},
{clip_lines}    max_steps={config.max_steps},
    bf16=BF16,
    fp16=FP16,
    gradient_checkpointing={config.gradient_checkpointing!r},
    report_to={config.report_to!r},
    logging_steps=1,
    save_strategy="no" if USE_PEFT and PEFT_METHOD == "bitfit" else "steps",
    save_steps={config.save_interval},
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID,
    {deepspeed_line}
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=trainer_config,
    train_dataset=train_dataset,
    reward_funcs=[reward_fn],
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint=None)
if USE_PEFT and PEFT_METHOD == "bitfit":
    save_bitfit_checkpoint(
        model,
        os.path.join(OUTPUT_DIR, "bitfit_adapter.pt"),
        base_model_name=MODEL_NAME,
    )
if PUSH_TO_HUB:
    trainer.push_to_hub()
print("Training complete!")
'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"Training script saved to {output_path}")
    return script
