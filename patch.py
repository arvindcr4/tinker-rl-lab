import sys

# 1. ai-scientist-template/experiment.py
file1 = "ai-scientist-template/experiment.py"
with open(file1, "r") as f:
    c = f.read()
c = c.replace(
    '    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)\n    parser.add_argument("--num_seeds", type=int, default=NUM_SEEDS)\n    return parser.parse_args()',
    '    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)\n    parser.add_argument("--full_finetune", action="store_true", help="Use full fine-tuning instead of LoRA")\n    parser.add_argument("--num_seeds", type=int, default=NUM_SEEDS)\n    return parser.parse_args()'
)
c = c.replace(
    '    # LoRA configuration (passed to GRPOTrainer, not GRPOConfig)\n    peft_config = LoraConfig(\n        r=args.lora_rank,\n        lora_alpha=args.lora_rank * 2,\n        lora_dropout=0.05,\n        bias="none",\n        task_type="CAUSAL_LM",\n    )',
    '    # LoRA configuration (passed to GRPOTrainer, not GRPOConfig)\n    peft_config = None\n    if not args.full_finetune:\n        peft_config = LoraConfig(\n            r=args.lora_rank,\n            lora_alpha=args.lora_rank * 2,\n            lora_dropout=0.05,\n            bias="none",\n            task_type="CAUSAL_LM",\n        )'
)
with open(file1, "w") as f:
    f.write(c)

# 2. ai-scientist-v2-integration/ai_scientist/ideas/tinker_grpo_rl.py
file2 = "ai-scientist-v2-integration/ai_scientist/ideas/tinker_grpo_rl.py"
with open(file2, "r") as f:
    c = f.read()
c = c.replace(
    'MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Tinker catalog. Small+fast.\nLORA_RANK = 16',
    'MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Tinker catalog. Small+fast.\nUSE_LORA = True\nLORA_RANK = 16'
)
c = c.replace(
    '    svc = tinker.ServiceClient(base_url=None)\n    tc = svc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)',
    '    svc = tinker.ServiceClient(base_url=None)\n    if USE_LORA:\n        tc = svc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)\n    else:\n        tc = svc.create_training_client(base_model=MODEL)'
)
with open(file2, "w") as f:
    f.write(c)

# 3. atropos/tinker_atropos/config.py
file3 = "atropos/tinker_atropos/config.py"
with open(file3, "r") as f:
    c = f.read()
c = c.replace(
    'class TinkerConfig(BaseModel):\n    """Tinker-specific configuration for LoRA training"""\n\n    lora_rank: int = 32',
    'class TinkerConfig(BaseModel):\n    """Tinker-specific configuration for LoRA training"""\n\n    use_lora: bool = True\n    lora_rank: int = 32'
)
c = c.replace(
    '    @property\n    def lora_rank(self) -> int:\n        return self.tinker.lora_rank',
    '    @property\n    def use_lora(self) -> bool:\n        return self.tinker.use_lora\n\n    @property\n    def lora_rank(self) -> int:\n        return self.tinker.lora_rank'
)
with open(file3, "w") as f:
    f.write(c)

# 4. atropos/tinker_atropos/trainer.py
file4 = "atropos/tinker_atropos/trainer.py"
with open(file4, "r") as f:
    c = f.read()
c = c.replace(
    '        # Create LoRA training client - use tinker_model if different from tokenizer\n        tinker_model = self.config.tinker_model\n        print(f"Creating training client for {tinker_model}...")\n        self.training_client = await self.service_client.create_lora_training_client_async(\n            base_model=tinker_model,\n            rank=self.lora_rank,\n        )\n        print("Training client created")',
    '        # Create training client - use tinker_model if different from tokenizer\n        tinker_model = self.config.tinker_model\n        if self.config.use_lora:\n            print(f"Creating LoRA training client for {tinker_model}...")\n            self.training_client = await self.service_client.create_lora_training_client_async(\n                base_model=tinker_model,\n                rank=self.lora_rank,\n            )\n        else:\n            print(f"Creating full fine-tuning training client for {tinker_model}...")\n            self.training_client = await self.service_client.create_training_client_async(\n                base_model=tinker_model,\n            )\n        print("Training client created")'
)
with open(file4, "w") as f:
    f.write(c)

# 5. atropos/launch_training.py
file5 = "atropos/launch_training.py"
with open(file5, "r") as f:
    c = f.read()
c = c.replace(
    '    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")',
    '    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")\n    parser.add_argument("--full-finetune", action="store_true", help="Use full fine-tuning instead of LoRA")'
)
c = c.replace(
    '    if args.lora_rank is not None:\n        overrides["lora_rank"] = args.lora_rank',
    '    if args.lora_rank is not None:\n        overrides["lora_rank"] = args.lora_rank\n    if args.full_finetune:\n        overrides["use_lora"] = False'
)
c = c.replace(
    '        if "use_wandb" in overrides:\n            config_dict["env"]["use_wandb"] = overrides.pop("use_wandb")\n        config_dict.update(overrides)',
    '        if "use_wandb" in overrides:\n            config_dict["env"]["use_wandb"] = overrides.pop("use_wandb")\n        if "use_lora" in overrides:\n            config_dict["tinker"]["use_lora"] = overrides.pop("use_lora")\n        if "lora_rank" in overrides:\n            config_dict["tinker"]["lora_rank"] = overrides.pop("lora_rank")\n        config_dict.update(overrides)'
)
c = c.replace(
    '    print(f"LoRA Rank: {config.lora_rank}")',
    '    print(f"LoRA Enabled: {config.use_lora}")\n    if config.use_lora:\n        print(f"LoRA Rank: {config.lora_rank}")'
)
with open(file5, "w") as f:
    f.write(c)

print("Patching complete.")
