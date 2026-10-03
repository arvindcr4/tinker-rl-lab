"""Helpers shared by the Modal GSM8K launchers in this directory.

Stdlib-only at import time so it can ride along in any Modal image via
``.add_local_python_source("_modal_common")``; torch/wandb are imported lazily.
"""

import os
import re


def patch_wandb_vram():
    """Wrap ``wandb.log`` once so every call also records peak/reserved VRAM."""
    try:
        import torch
        import wandb

        if not getattr(wandb, "_vram_patched", False):
            _old_log = wandb.log

            def _vram_log(data, *args, **kwargs):
                if torch.cuda.is_available():
                    data["system/vram_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (
                        1024**3
                    )
                    data["system/vram_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
                    torch.cuda.reset_peak_memory_stats()
                _old_log(data, *args, **kwargs)

            wandb.log = _vram_log
            wandb._vram_patched = True
    except ImportError:
        pass


def push_to_hub_private(model, tok, repo_id):
    """Push model + tokenizer to a private HF repo; skip (return False) without HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found in environment, skipping push_to_hub")
        return False
    model.push_to_hub(repo_id, token=token, private=True)
    tok.push_to_hub(repo_id, token=token, private=True)
    return True


def chatml_prompt(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def gsm8k_gold(answer):
    """Numeric gold from a GSM8K ``#### N`` answer, or None when absent."""
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def boxed_or_last_number_reward(response: str, answer: str) -> float:
    """Boxed answer (numeric or exact string), else the last number in the text."""
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


def boxed_only_reward(text, gt):
    """Boxed answer only (numeric or exact string); no last-number fallback."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text or "")
    for b in boxed:
        b_clean = b.strip().replace(",", "")
        try:
            if gt and abs(float(b_clean) - float(gt)) < 1e-2:
                return 1.0
        except Exception:
            if b_clean == gt:
                return 1.0
    return 0.0
