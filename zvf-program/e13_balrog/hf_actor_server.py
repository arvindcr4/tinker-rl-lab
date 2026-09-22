"""Minimal OpenAI-compatible generation server over transformers (no custom kernels).

Rationale 2026-09-20: vLLM on this stack cannot start without nvcc (FlashInfer
JIT for sampling/GDN) and the registry image is not Modal-deployable. torch
SDPA wheels are prebuilt, so plain transformers inference works.
Bounded: bearer auth, per-request token caps, single worker.
"""

from __future__ import annotations

import os
import time
import uuid

import torch
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TOKEN = os.environ["E13_ACTOR_API_KEY"]
MAX_NEW_TOKENS_CAP = 8192

app = FastAPI()
tok = None
model = None


def ensure_loaded():
    global tok, model
    if model is not None:
        return
    sha = HfApi(token=os.environ["HF_TOKEN"]).model_info(REPO, revision=REVISION).sha
    if sha != REVISION:
        raise RuntimeError(f"actor base drift: {sha} != {REVISION}")
    local = snapshot_download(REPO, revision=REVISION, token=os.environ["HF_TOKEN"])
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        local, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False, attn_implementation="sdpa",
    )
    model.eval()


def generate_text(prompt: str, max_tokens: int) -> str:
    ensure_loaded()
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs,
                             max_new_tokens=min(max_tokens, MAX_NEW_TOKENS_CAP),
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@app.on_event("startup")
def startup():
    ensure_loaded()


def check(auth: str | None):
    if not auth or not auth.startswith("Bearer ") or auth[7:] != TOKEN:
        raise HTTPException(status_code=401, detail="bad bearer")


@app.get("/health")
def health():
    return {"loaded": model is not None}


@app.post("/v1/chat/completions")
def chat(body: dict, authorization: str | None = Header(default=None)):
    check(authorization)
    messages = body.get("messages", [])
    max_tokens = min(int(body.get("max_tokens", 512)), MAX_NEW_TOKENS_CAP)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": REPO,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
