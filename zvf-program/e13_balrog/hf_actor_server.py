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
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "n", "stream", "chat_template_kwargs"}
    if set(body) - allowed:
        raise HTTPException(400, "unsupported generation options: " + ", ".join(sorted(set(body)-allowed)))
    if body.get("model") != REPO:
        raise HTTPException(400, "requested model is not the pinned served model")
    if body.get("temperature", 0) != 0 or body.get("top_p", 1) != 1:
        raise HTTPException(400, "this actor supports greedy decoding only")
    if body.get("stream", False) is not False or body.get("n", 1) != 1:
        raise HTTPException(400, "only one non-streaming completion is supported")
    max_tokens = body.get("max_tokens", 512)
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_NEW_TOKENS_CAP:
        raise HTTPException(400, "max_tokens outside supported cap")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
        or not isinstance(m.get("content"), str) for m in messages):
        raise HTTPException(400, "messages must be nonempty text chat messages")
    template = body.get("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template)-{"enable_thinking"} or any(type(v) is not bool for v in template.values()):
        raise HTTPException(400, "unsupported chat template options")
    ensure_loaded()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion = out[0][prompt_tokens:]
    completion_tokens = len(completion)
    text = tok.decode(completion, skip_special_tokens=True)
    finish_reason = "stop" if tok.eos_token_id in completion.tolist() else "length"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": REPO,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens+completion_tokens},
    }
