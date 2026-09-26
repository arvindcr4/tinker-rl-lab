"""Local OpenAI-compatible shim: base Qwen/Qwen3.6-35B-A3B on Tinker (no adapter).

Usage: python tinker_shim.py --port 8765 --ledger raw/shim_ledger.jsonl --cap-file raw/cap.json
Every request is appended to the ledger (token counts, finish reason, raw output text).
The cap file ({"cap_total_tokens": N}) is re-read per request; a request whose
prefill + max_tokens would push the ledger total past the cap gets HTTP 402 and
is never sampled, so the cap cannot be exceeded.
Fixed per protocol: enable_thinking=False, temperature=0, stop on <|im_end|>/<|endoftext|>.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "zvf-program" / "flagship"))
from tinker_openai_bridge_protocol import (  # noqa: E402
    build_responses_object,
    iter_responses_sse_events,
    normalise_openai_messages_for_qwen,
    openai_chat_stream_events,
    parse_qwen_tool_calls,
    responses_tools_to_chat_tools,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
ALIAS = "pavlov-qwen36-tinker"
CONTEXT = 65_536
PREFIX_KEEP = 8_192
MAX_COMPLETION = 16_384

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--ledger", required=True)
ap.add_argument("--cap-file", required=True)
ap.add_argument("--default-max-tokens", type=int, default=8192)
ap.add_argument("--backend", choices=["tinker", "vllm"], default="tinker")
ap.add_argument("--vllm-url", default=None)      # e.g. https://...modal.run  (key: TRAINED_ACTOR_API_KEY env)
ap.add_argument("--vllm-model", default=None)
ap.add_argument("--context", type=int, default=65_536)
ap.add_argument("--max-inflight", type=int, default=4)
args = ap.parse_args()
CONTEXT = args.context

import tinker  # noqa: E402
import tinker.types as T  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID)
STOP_IDS = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
if args.backend == "tinker":
    svc = tinker.ServiceClient()
    sampler = svc.create_sampling_client(base_model=MODEL_ID)
else:
    import httpx  # noqa: E402
    vclient = httpx.Client(base_url=args.vllm_url, timeout=1800,
                           headers={"Authorization": f"Bearer {os.environ['TRAINED_ACTOR_API_KEY']}"})
    vsem = threading.Semaphore(args.max_inflight)  # PROTOCOL: <=4 in-flight per endpoint per lane group (summed over shims)


def _sample(ids, max_tokens):
    """Greedy sample from the same chat-templated token ids on either backend -> output token ids."""
    if args.backend == "tinker":
        res = sampler.sample(T.ModelInput.from_ints(ids), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=STOP_IDS)).result()
        return list(res.sequences[0].tokens)
    with vsem:
        r = vclient.post("/v1/completions", json={
            "model": args.vllm_model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "stop_token_ids": STOP_IDS, "skip_special_tokens": False, "include_stop_str_in_output": True})
    r.raise_for_status()
    text = r.json()["choices"][0]["text"]
    return tok.encode(text, add_special_tokens=False)
lock = threading.Lock()
ledger_path = Path(args.ledger)
ledger_path.parent.mkdir(parents=True, exist_ok=True)
state = {"used": 0, "reserved": 0}
if ledger_path.exists():
    for line in ledger_path.read_text().splitlines():
        r = json.loads(line)
        state["used"] += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)


def responses_input_to_chat_messages(inp: Any) -> list[dict[str, Any]]:
    """Responses input -> chat messages. Unlike the shared helper, keeps string
    content and turns function_call items into assistant tool_calls so the model
    sees its own prior calls."""
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    if not isinstance(inp, list):
        raise ValueError("responses input must be a string or list")
    out: list[dict[str, Any]] = []
    for item in inp:
        kind = item.get("type", "message")
        if kind == "message":
            c = item.get("content", "")
            if isinstance(c, list):
                c = "\n".join(str(x.get("text", "")) for x in c
                              if isinstance(x, dict) and x.get("type") in ("input_text", "output_text", "text"))
            role = {"developer": "system"}.get(item.get("role", "user"), item.get("role", "user"))
            if role == "system" and out and out[0]["role"] == "system":
                out[0]["content"] += "\n\n" + c
            else:
                out.append({"role": role, "content": c})
        elif kind == "function_call":
            call = {"id": item.get("call_id") or item.get("id"), "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}}
            if out and out[-1]["role"] == "assistant" and "tool_calls" in out[-1]:
                out[-1]["tool_calls"].append(call)
            elif out and out[-1]["role"] == "assistant":
                out[-1]["tool_calls"] = [call]
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            o = item.get("output")
            if isinstance(o, list):
                o = "\n".join(str(x.get("text", "")) for x in o if isinstance(x, dict))
            out.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": "" if o is None else str(o)})
    if not out:
        raise ValueError("no usable messages")
    return out


def cap() -> int:
    return int(json.loads(Path(args.cap_file).read_text())["cap_total_tokens"])


def log(rec: dict[str, Any]) -> None:
    with lock:
        with ledger_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


async def complete(messages, tools, max_tokens, endpoint, tag=None):
    max_tokens = min(int(max_tokens or args.default_max_tokens), MAX_COMPLETION)
    kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    if tools:
        kw["tools"] = tools
    try:
        msgs = normalise_openai_messages_for_qwen(messages)
        def _txt(c):
            return c if isinstance(c, str) else "\n".join(
                str(x.get("text", "")) for x in (c or []) if isinstance(x, dict))
        sys_parts = [_txt(m.get("content")) for m in msgs if m.get("role") in ("system", "developer")]
        msgs = [m for m in msgs if m.get("role") not in ("system", "developer")]
        for m in msgs:
            if isinstance(m.get("content"), list):
                m["content"] = _txt(m["content"])
            if m.get("content") is None:
                m["content"] = ""
        if sys_parts:
            msgs = [{"role": "system", "content": "\n\n".join(sys_parts)}] + msgs
        text = tok.apply_chat_template(msgs, **kw)
    except Exception as exc:  # template/argument errors -> 400, not sampled
        raise HTTPException(status_code=400, detail=f"template error: {exc}") from exc
    ids = tok.encode(text, add_special_tokens=False)
    orig = len(ids)
    budget = CONTEXT - max_tokens
    if len(ids) > budget:
        ids = ids[:PREFIX_KEEP] + ids[-(budget - PREFIX_KEEP):]
    need = len(ids) + max_tokens
    with lock:
        if state["used"] + state["reserved"] + need > cap():
            refused = True
        else:
            refused = False
            state["reserved"] += need
    if refused:
        log({"ts": time.time(), "endpoint": endpoint, "refused_cap": True, "prompt_tokens": 0,
             "completion_tokens": 0, "would_need": need})
        raise HTTPException(status_code=402, detail="token cap reached")
    t0 = time.time()
    try:
        out = await asyncio.to_thread(_sample, ids, max_tokens)
    except Exception as exc:
        with lock:
            state["reserved"] -= need
            state["used"] += len(ids)  # conservative: count prefill on failure
        log({"ts": time.time(), "endpoint": endpoint, "error": repr(exc)[:300],
             "prompt_tokens": len(ids), "completion_tokens": 0})
        raise HTTPException(status_code=500, detail="sampling failed") from exc
    finish = "stop" if out and out[-1] in STOP_IDS else "length"
    out_text = tok.decode(out, skip_special_tokens=True)
    content, calls = parse_qwen_tool_calls(out_text)
    if content:
        content = content.replace("<think>", "").replace("</think>", "").strip() or None
    with lock:
        state["reserved"] -= need
        state["used"] += len(ids) + len(out)
    log({"ts": time.time(), "backend": args.backend, "tag": tag, "endpoint": endpoint, "prompt_tokens": len(ids), "orig_prompt_tokens": orig,
         "completion_tokens": len(out), "max_tokens": max_tokens, "finish": finish,
         "n_tool_calls": len(calls), "latency_s": round(time.time() - t0, 2), "output_text": out_text})
    return content, calls, len(ids), len(out)


from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

SHIM_KEY = os.environ["SHIM_KEY"]  # bearer required: shim binds 0.0.0.0 so containers can reach it


class Auth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.headers.get("authorization", "") != f"Bearer {SHIM_KEY}":
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


app = FastAPI()
app.add_middleware(Auth)


@app.get("/health")
async def health():
    return {"status": "READY", "model": MODEL_ID, "adapter": None, "used": state["used"], "cap": cap()}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": ALIAS, "object": "model"}, {"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(payload: dict[str, Any] = Body(...)):
    content, calls, p, c = await complete(
        payload["messages"], payload.get("tools"),
        payload.get("max_completion_tokens") or payload.get("max_tokens"), "chat", payload.get("model"))
    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    if payload.get("stream"):
        return StreamingResponse(iter(openai_chat_stream_events(
            completion_id=cid, created=created, model=ALIAS, content=content, tool_calls=calls,
            prompt_tokens=p, completion_tokens=c)), media_type="text/event-stream")
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"id": cid, "object": "chat.completion", "created": created, "model": ALIAS,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}


@app.post("/v1/responses")
async def responses(payload: dict[str, Any] = Body(...)):
    try:
        messages = responses_input_to_chat_messages(payload.get("input"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("instructions"):
        messages = [{"role": "system", "content": str(payload["instructions"])}] + messages
    content, calls, p, c = await complete(
        messages, responses_tools_to_chat_tools(payload.get("tools")),
        payload.get("max_output_tokens"), "responses")
    resp = build_responses_object(response_id=f"resp-{uuid.uuid4().hex}", model=ALIAS, content=content,
                                  tool_calls=calls, prompt_tokens=p, completion_tokens=c,
                                  created_at=int(time.time()))
    if payload.get("stream"):
        return StreamingResponse(iter_responses_sse_events(resp), media_type="text/event-stream")
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
