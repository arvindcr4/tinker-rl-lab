import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass

import asyncio
import os
import time
import numpy as np
import torch
from typing import Dict, Any, List

import tinker
from tinker.types import AdamParams, ModelInput, SamplingParams
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
import requests
from transformers import AutoTokenizer

from tinker_atropos.config import TinkerAtroposConfig
from tinker_atropos.dataset import DatasetPreprocessor
from tinker_atropos.api import create_app

class TinkerAtroposTrainer:
    """
    Trainer that handles both RL training and inference through Tinker API.
    Connects to Atropos Trajectory API to coordinate environment interaciton.
    """

    def __init__(self, config: TinkerAtroposConfig):
        self.config = config

        # Model and training config
        self.base_model = config.base_model
        self.lora_rank = config.lora_rank
        self.learning_rate = config.learning_rate
        self.atropos_api_url = config.atropos_api_url
        self.num_steps = config.num_steps
        
        # Dataset Preprocessor
        self.dataset = DatasetPreprocessor(self.atropos_api_url)

        # Tinker clients
        self.service_client = None
        self.training_client = None
        self.current_sampling_client = None
        self.tokenizer = None

        # Atropos registration
        self.trainer_id = None
        self.wandb_group = None

    async def setup(self):
        print("Setting up Tinker-Atropos Trainer...")

        # Create single ServiceClient for both training and inference
        print(f"Creating ServiceClient for {self.base_model}...")
        self.service_client = tinker.ServiceClient()

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        print(f"Loaded tokenizer for {self.base_model}")

        # Create training client - use tinker_model if different from tokenizer
        tinker_model = self.config.tinker_model
        if self.config.use_lora:
            print(f"Creating LoRA training client for {tinker_model}...")
            self.training_client = await self.service_client.create_lora_training_client_async(
                base_model=tinker_model,
                rank=self.lora_rank,
            )
        else:
            print(f"Creating full fine-tuning training client for {tinker_model}...")
            self.training_client = await self.service_client.create_training_client_async(
                base_model=tinker_model,
            )
        print("Training client created")

        # Save initial weights and create sampling client
        print("Saving initial weights...")
        initial_path = self.training_client.save_weights_for_sampler(name="step_0").result().path
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=initial_path
        )
        print(f"Initial sampling client created: {initial_path}")

        self.wandb_group = self.config.wandb_group or wandb.sdk.lib.runid.generate_id()

        print("Registering with Atropos API...")
        self.trainer_id = await self._register_trainer()
        print(f"Registered as trainer: {self.trainer_id}")

        if self.config.use_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"{self.config.wandb_run_name}-trainer-{self.config.wandb_run_suffix}",
                    group=self.wandb_group,
                    tags=["trainer"],
                )
                print(f"Wandb initialized (trainer): {wandb.run.name} in group: {self.wandb_group}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                self.config.env.use_wandb = False

    async def _register_trainer(self) -> str:
        """Register this trainer with the Atropos API server."""
        url = f"{self.atropos_api_url}/register"

        payload = {
            "wandb_project": self.config.wandb_project,
            "wandb_group": self.wandb_group,
            "batch_size": self.config.batch_size,
            "max_token_len": self.config.max_token_trainer_length,
            "starting_step": 0,
            "checkpoint_dir": self.config.checkpoint_dir,
            "save_checkpoint_interval": self.config.save_checkpoint_interval,
            "num_steps": self.num_steps,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("uuid")

    async def train_step(self, step: int) -> Dict[str, Any]:
        """Execute one training step: fetch batch, forward-backward, optimizer step."""
        print(f"\n{'='*60}")
        print(f"Step {step}/{self.num_steps}")
        print(f"{'='*60}")

        step_start = time.time()
        metrics = {"step": step}

        # Fetch batch from Atropos via DatasetPreprocessor
        print("Fetching data from Atropos...")
        data, has_distil = self.dataset.get_data()
        print(f"Got {len(data)} Datum objects")
        if has_distil:
            print("  with on-policy distillation (advantages = logp_t - logp_s)")
            if self.dataset.distil_stats:
                ds = self.dataset.distil_stats
                print(
                    f"  teacher_logp={ds.get('distil/teacher_logp_mean', 0):.4f} "
                    f"student_logp={ds.get('distil/student_logp_mean', 0):.4f} "
                    f"adv_mean={ds.get('distil/advantage_mean', 0):.4f} "
                    f"kl≈{ds.get('distil/kl_approx', 0):.4f} "
                    f"({ds.get('distil/num_tokens', 0)} tokens)"
                )

        # Forward-backward pass
        print("Running forward-backward pass...")
        fwd_bwd_result = await self.training_client.forward_backward_async(
            data, loss_fn="importance_sampling"
        )

        # Optimizer step
        print("Running optimizer step...")
        adam_params = AdamParams(learning_rate=self.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        optim_result = await self.training_client.optim_step_async(adam_params)

        # Await results
        if hasattr(fwd_bwd_result, "result_async"):
            fwd_bwd_result = await fwd_bwd_result.result_async()
        elif hasattr(fwd_bwd_result, "result"):
            fwd_bwd_result = fwd_bwd_result.result()
        optim_result = await optim_result.result_async()

        loss_val = (
            fwd_bwd_result.metrics["loss:sum"] if "loss:sum" in fwd_bwd_result.metrics else 0.0
        )

        print(f"Loss: {loss_val}")

        if has_distil:
            metrics["distil/active"] = 1

        # Calculate training logprob stats
        training_logprobs_all = []
        for datum, output in zip(data, fwd_bwd_result.loss_fn_outputs):
            training_logprobs = output["logprobs"].to_torch()
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = advantages != 0.0
            training_lp_masked = training_logprobs[mask]
            training_logprobs_all.extend(training_lp_masked.cpu().numpy().tolist())

        if training_logprobs_all:
            training_lp_array = np.array(training_logprobs_all)
            self.training_logprob_stats = {
                "logprobs/mean_training": float(np.mean(training_lp_array)),
                "logprobs/std_training": float(np.std(training_lp_array)),
                "logprobs/min_training": float(np.min(training_lp_array)),
                "logprobs/p50_training": float(np.percentile(training_lp_array, 50)),
                "train/policy_entropy": float(-np.mean(training_lp_array)),
            }

            # Calculate logprob drift
            if self.dataset.logprob_stats and "logprobs/mean" in self.dataset.logprob_stats:
                ref_mean = self.dataset.logprob_stats["logprobs/mean"]
                train_mean = float(np.mean(training_lp_array))
                self.training_logprob_stats["logprobs/diff"] = ref_mean - train_mean
        else:
            self.training_logprob_stats = {}

        # Update sampling client with new weights
        print("Saving checkpoint and updating sampling client...")
        new_path = (
            self.training_client.save_weights_for_sampler(name=f"step_{step+1}").result().path
        )
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=new_path
        )
        print(f"Sampling client updated: {new_path}")

        step_time = time.time() - step_start
        metrics["step_time"] = step_time
        metrics["learning_rate"] = self.learning_rate
        metrics["loss"] = loss_val

        if self.dataset.group_mean_rewards:
            metrics["reward/mean"] = np.mean(self.dataset.group_mean_rewards)
            print(f"Reward/mean: {metrics['reward/mean']:.4f}")

        if self.config.use_wandb:
            wandb_metrics = {
                "train/loss": loss_val,
                "train/learning_rate": self.learning_rate,
                "reward/mean": metrics.get("reward/mean", 0.0),
            }
            if hasattr(self.dataset, "zvf"):
                wandb_metrics["zvf"] = self.dataset.zvf

            if hasattr(self.dataset, "logprob_stats"):
                wandb_metrics.update(self.dataset.logprob_stats)
            if hasattr(self, "training_logprob_stats"):
                wandb_metrics.update(self.training_logprob_stats)
            if hasattr(self.dataset, "advantage_stats"):
                wandb_metrics.update(self.dataset.advantage_stats)
            if hasattr(self.dataset, "erf_stats"):
                wandb_metrics.update(self.dataset.erf_stats)

            if has_distil:
                wandb_metrics["distil/active"] = 1
            if hasattr(self.dataset, "distil_stats"):
                wandb_metrics.update(self.dataset.distil_stats)

            wandb.log(wandb_metrics, step=step + 1)

        return metrics

    async def run(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Tinker-Atropos Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.num_steps):
            try:
                metrics = await self.train_step(step)
                print(f"\nStep {step} complete - Loss: {metrics.get('loss', 'N/A')}")
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60 + "\n")

        print(
            f"Final weights are available here: tinker://{str(self.training_client.model_id)}/sampler_weights/final"
        )


trainer: TinkerAtroposTrainer | None = None
app = create_app(trainer)

def run_fastapi_server(port=8001):
    """Run FastAPI server in background thread."""
    import uvicorn
    if trainer is not None:
        app.state.trainer = trainer
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

async def main():
    # TODO: Avoid extrapolating RL training dynamics from single-seed runs. Add support for multiple seeds to address statistical vulnerability.
    global trainer

    config = TinkerAtroposConfig(
        lora_rank=int(os.getenv("LORA_RANK", "32")),
        learning_rate=float(os.getenv("LEARNING_RATE", "4e-5")),
        # TODO: 30-50 steps is an "Early-Training Snapshot" and insufficient to observe meaningful RL convergence, long-horizon reward hacking, catastrophic forgetting, or true policy collapse. Increase num_steps.
        num_steps=int(os.getenv("NUM_STEPS", "50")),
    )

    print(f"Using wandb run: {config.wandb_run_name}")

    trainer = TinkerAtroposTrainer(config)

    # Start FastAPI server in background thread for Atropos to call
    import threading

    server_thread = threading.Thread(target=run_fastapi_server, args=(8001,), daemon=True)
    server_thread.start()

    print("Waiting for FastAPI server to start...")
    await asyncio.sleep(3)

    await trainer.run()

if __name__ == "__main__":
    asyncio.run(main())
