#!/usr/bin/env python3
"""Minimal local OpenAI-compatible shim over Tinker base-model sampling.

Actor: Qwen/Qwen3.6-35B-A3B base weights, no adapter, non-thinking template.
Every call is appended to $SHIM_LOG (jsonl) with prompt/completion token counts.
A hard token budget ($SHIM_BUDGET, prefill+sample) is enforced from that log:
once reached, requests fail with HTTP 400 (the agent fails; the item is scored 0).

Usage: SHIM_LOG=... SHIM_BUDGET=... PORT=8765 python tinker_shim.py
"""
import json, os, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tinker
import tinker.types as T
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
LOG = os.environ["SHIM_LOG"]
BUDGET = int(os.environ["SHIM_BUDGET"])
PORT = int(os.environ.get("PORT", "8765"))
CTX = int(os.environ.get("SHIM_CTX", "32768"))
DEFAULT_MAX = int(os.environ.get("SHIM_MAX_TOKENS", "4096"))
FORCE_TEMP = os.environ.get("SHIM_FORCE_TEMP", "0")  # protocol default temperature=0

tok = AutoTokenizer.from_pretrained(MODEL)
sampler = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
lock = threading.Lock()


_USED = None


def used():
    global _USED
    if _USED is None:
        _USED = 0
        if os.path.exists(LOG):
            with open(LOG) as f:
                for line in f:
                    r = json.loads(line)
                    _USED += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    return _USED


def add_used(n):
    global _USED
    used()
    _USED += n


def norm(messages):
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):  # OpenAI content parts -> text
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append({"role": m["role"], "content": c or ""})
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        max_tokens = min(req.get("max_completion_tokens") or req.get("max_tokens") or DEFAULT_MAX, DEFAULT_MAX)
        temp = float(FORCE_TEMP) if FORCE_TEMP != "" else float(req.get("temperature") or 0)
        text = tok.apply_chat_template(norm(req["messages"]), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        orig = len(ids)
        budget_ctx = CTX - max_tokens
        if len(ids) > budget_ctx:  # keep head + tail
            ids = ids[:2048] + ids[-(budget_ctx - 2048):]
        with lock:
            u = used()
            if u + len(ids) + max_tokens > BUDGET:
                return self._send(400, {"error": {"message": f"token budget exhausted ({u}/{BUDGET})",
                                                  "type": "budget_exhausted"}})
        t0 = time.time()
        stop = req.get("stop") or []
        stop = [stop] if isinstance(stop, str) else stop
        try:
            res = sampler.sample(T.ModelInput.from_ints(ids), num_samples=1,
                                 sampling_params=T.SamplingParams(max_tokens=max_tokens, temperature=temp,
                                                                  stop=stop or None)).result()
        except Exception as e:  # counted as failure by the agent
            with lock, open(LOG, "a") as f:
                f.write(json.dumps({"t": t0, "error": repr(e)[:300], "prompt_tokens": len(ids),
                                    "completion_tokens": 0}) + "\n")
                add_used(len(ids))
            return self._send(500, {"error": {"message": "tinker sampling failed"}})
        out = list(res.sequences[0].tokens)
        content = tok.decode(out, skip_special_tokens=True)
        if "</think>" in content:
            content = content.split("</think>", 1)[1].lstrip()
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"t": t0, "dt": round(time.time() - t0, 2), "prompt_tokens": len(ids),
                                "orig_prompt_tokens": orig, "completion_tokens": len(out),
                                "temperature": temp, "max_tokens": max_tokens}) + "\n")
            add_used(len(ids) + len(out))
        self._send(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(t0),
            "model": req.get("model", MODEL),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "length" if len(out) >= max_tokens else "stop"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                      "total_tokens": len(ids) + len(out)},
        })


if __name__ == "__main__":
    print(f"shim on :{PORT} budget={BUDGET} used={used()}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
