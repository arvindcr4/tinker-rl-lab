"""Helpers shared by the Modal GSM8K launchers in this directory.

Stdlib-only at import time so it can ride along in any Modal image via
``.add_local_python_source("_modal_common", "platform_hybrid")``; torch/wandb are imported lazily.
"""

import sys as _contract_sys
from pathlib import Path as _ContractPath
if len(_ContractPath(__file__).resolve().parents) > 3:
    _contract_sys.path.insert(0, str(_ContractPath(__file__).resolve().parents[3]))
from platform_hybrid.experiments.framework_contracts import score_gsm8k, score_batch, SCORER_ID, completion_receipt


import os
import re


def patch_wandb_vram():
    """Deprecated no-op. Use per-run orchestrator telemetry; never patch global logging."""
    return None


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


def boxed_or_last_number_reward(response, answer):
    """Compatibility alias for the shared v3 final-answer scorer."""
    return score_gsm8k(response, answer)


def boxed_only_reward(text, gt):
    """Compatibility alias for the shared v3 final-answer scorer."""
    return score_gsm8k(text, gt)
