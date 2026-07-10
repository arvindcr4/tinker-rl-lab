import os
from typing import Optional, List, Dict, Any
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model
)

def get_peft_config(
    method: str = "lora",
    task_type: TaskType = TaskType.CAUSAL_LM,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_target_modules: Optional[List[str]] = None,
    num_virtual_tokens: int = 32,
    encoder_hidden_size: int = 128,
):
    """
    Returns the appropriate PEFT configuration based on the requested method.
    
    Supported methods:
    - "lora": Standard Low-Rank Adaptation
    - "prefix_tuning": Prefix Tuning (keeping KV cache)
    - "p_tuning": P-Tuning (with an encoder)
    - "prompt_tuning": Soft Prompt Tuning
    - "bitfit": Returns None (needs manual parameter freezing, handled by apply_bitfit)
    """
    method = method.lower()
    
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
    if method == "lora":
        return LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            task_type=task_type,
            bias="none",
        )
    elif method == "prefix_tuning":
        return PrefixTuningConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )
    elif method == "p_tuning":
        return PromptEncoderConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
        )
    elif method == "prompt_tuning":
        return PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )
    elif method == "bitfit":
        return None
    else:
        raise ValueError(f"Unknown PEFT method: {method}")

def apply_bitfit(model):
    """
    Freezes all parameters in the model except for the biases (BitFit).
    """
    for name, param in model.named_parameters():
        if "bias" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model
