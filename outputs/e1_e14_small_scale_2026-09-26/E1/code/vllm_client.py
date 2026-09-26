"""Minimal streaming /v1/completions client for the paired vLLM arm (stdlib only).

The Tinker-base harnesses sampled raw token IDs, so here the identical chat-templated prompt token IDs go to
/v1/completions (prompt = list[int]). Endpoint and model id are the only things that change between arms.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

ENDPOINTS = {
    "trained": ("https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "pavlov-public-portfolio-bf16"),
    "base": ("https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "qwen36-base-bf16"),
}
MAX_MODEL_LEN = 32768  # server flag in TRAINED_ACTOR_ENDPOINT.md


def _hdr():
    return {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["TRAINED_ACTOR_API_KEY"]}


def wait_ready(arm: str) -> tuple[str, float]:
    url, model = ENDPOINTS[arm]
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers=_hdr()), timeout=900) as r:
                ids = [m["id"] for m in json.loads(r.read())["data"]]
            assert model in ids, ids
            return model, time.time() - t0
        except AssertionError:
            raise
        except Exception as e:
            if time.time() - t0 > 1200:
                raise
            print("waiting for", arm, repr(e)[:120])
            time.sleep(15)


def complete(arm: str, prompt_ids: list[int], max_tokens: int) -> dict:
    """Greedy (temperature 0) streamed completion. Returns text + usage."""
    url, model = ENDPOINTS[arm]
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers=_hdr())
    parts, usage, finish = [], {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                parts.append(ch.get("text") or "")
                finish = ch.get("finish_reason") or finish
    return {"text": "".join(parts), "prompt_tokens": usage.get("prompt_tokens", len(prompt_ids)),
            "completion_tokens": usage.get("completion_tokens"), "finish_reason": finish}
