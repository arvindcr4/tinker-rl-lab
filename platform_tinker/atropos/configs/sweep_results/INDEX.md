# platform_tinker/atropos/configs/sweep_results/ — INDEX

**Purpose:** Auto-generated exhaustive GRPO hyperparameter sweep for GSM8K on Qwen3-0.6B (open-source Unsloth/TRL backend). 108 config YAMLs spanning the full grid plus one runner.

**Contents (data dump — not enumerated):** Files follow `config_lr<LR>_lora<RANK>_bs<BATCH>_gs<GROUP>.yaml`, the Cartesian product of learning rate ∈ {1e-05, 3e-05, 0.0001}, LoRA rank ∈ {8, 16, 32, 64}, batch size ∈ {64, 128, 256}, and group size ∈ {8, 16, 32} = 108 configs. Each is a standard training config (env/openai/tinker sections, `wandb_group: grpo-exhaustive-sweep`, `wandb_project: tinker-rl-scaling`) differing only in those four hyperparameters, with a matching `wandb_run_name`. To select one, build the filename from your desired (lr, lora, bs, gs) tuple. `run_sweep.sh` runs the entire grid sequentially via `train_grpo_unsloth.py --seed 42`.

**Find it fast:**
- to run the whole sweep → `run_sweep.sh`
- to grab one setting → `config_lr<lr>_lora<rank>_bs<bs>_gs<gs>.yaml`
