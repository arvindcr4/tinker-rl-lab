import os
import torch
from pathlib import Path

def analyze_sparsity(checkpoint_dir, epsilon=1e-4):
    """
    Computes the fraction of LoRA parameters with absolute value < epsilon.
    This evaluates the hypothesis that RL finetuning only updates a small subnetwork (5-30%).
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        print(f"Checkpoint directory {checkpoint_dir} not found.")
        return
    
    # Walk through the directory and find safetensors or bin files
    lora_files = list(path.glob("**/*.safetensors")) + list(path.glob("**/*.bin"))
    if not lora_files:
        print("No LoRA checkpoint files found in", checkpoint_dir)
        return

    total_params = 0
    near_zero_params = 0

    # Note: actual loading of model weights requires safetensors or torch.
    # This is scaffolding for Paper 3 (Scaling Laws and Saturation Dynamics).
    print(f"Found {len(lora_files)} checkpoint files. Scaffolding sparsity analysis...")
    print("Target: Compare active parameters vs arXiv 2505.11711 (5-30% active expected).")

if __name__ == '__main__':
    # Example placeholder path
    analyze_sparsity('modal/checkpoints')
