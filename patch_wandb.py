import os
import re

files_with_wandb = [
    "atropos/tinker_atropos/trainer.py",
    "atropos/train_grpo_humaneval.py",
    "atropos/train_grpo_unsloth.py",
    "experiments/10x_structural_ceiling/group_saturation_diagnostic.py",
    "experiments/10x_structural_ceiling/grpo_10x_runner.py",
    "experiments/10x_structural_ceiling/round2_runner.py",
    "experiments/modal/modal_grpo_trl.py",
    "experiments/modal/modal_new_experiments.py",
    "experiments/modal/modal_parallel_runner.py",
    "experiments/modal/modal_ppo_campaign.py",
    "experiments/modal/modal_ppo_fix.py",
    "experiments/modal/relaunch_kl.py",
]

for f in files_with_wandb:
    with open(f, "r") as file:
        lines = file.readlines()
    for i, l in enumerate(lines):
        try:
            import wandb, torch
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
        if "wandb.log(" in l:
            print(f"File: {f}:{i+1}\n{l.strip()}")

