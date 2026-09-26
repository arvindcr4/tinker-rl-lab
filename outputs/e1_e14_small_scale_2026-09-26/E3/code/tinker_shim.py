#!/usr/bin/env python3
"""Minimal local OpenAI-compatible shim over Tinker base-model sampling.

Actor: Qwen/Qwen3.6-35B-A3B base weights, no adapter, non-thinking template.
Every call is appended to $SHIM_LOG (jsonl) with prompt/completion token counts.
A hard token budget ($SHIM_BUDGET, prefill+sample) is enforced from that log:
once reached, requests fail with HTTP 400 (the agent fails; the item is scored 0).

Usage: SHIM_LOG=... SHIM_BUDGET=... PORT=8765 python tinker_shim.py

Paired vLLM arm: SHIM_BACKEND=vllm VLLM_BASE_URL=<endpoint> VLLM_MODEL=<model id> with
TRAINED_ACTOR_API_KEY in the environment. The identical (truncated) token ids go to
/v1/completions, so only the serving engine/weights differ from the Tinker run.
"""
import json, os, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
LOG = os.environ["SHIM_LOG"]
BUDGET = int(os.environ["SHIM_BUDGET"])
PORT = int(os.environ.get("PORT", "8765"))
CTX = int(os.environ.get("SHIM_CTX", "32768"))
DEFAULT_MAX = int(os.environ.get("SHIM_MAX_TOKENS", "4096"))
FORCE_TEMP = os.environ.get("SHIM_FORCE_TEMP", "0")  # protocol default temperature=0

tok = AutoTokenizer.from_pretrained(MODEL)
BACKEND = os.environ.get("SHIM_BACKEND", "tinker")
if BACKEND == "tinker":
    import tinker
    import tinker.types as T
    sampler = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
else:
    VLLM_URL = os.environ["VLLM_BASE_URL"].rstrip("/") + "/v1/completions"
    VLLM_MODEL = os.environ["VLLM_MODEL"]


def sample(ids, max_tokens, temp, stop):
    """Return completion token ids from the configured backend."""
    if BACKEND == "tinker":
        res = sampler.sample(T.ModelInput.from_ints(ids), num_samples=1,
                             sampling_params=T.SamplingParams(max_tokens=max_tokens, temperature=temp,
                                                              stop=stop or None)).result()
        return list(res.sequences[0].tokens)
    body = {"model": VLLM_MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": temp,
            "stop": stop or None, "return_token_ids": True, "skip_special_tokens": False}
    rq = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["TRAINED_ACTOR_API_KEY"]})
    with urllib.request.urlopen(rq, timeout=900) as r:
        ch = json.load(r)["choices"][0]
    toks = ch.get("token_ids")
    return list(toks) if toks is not None else tok.encode(ch["text"], add_special_tokens=False)
lock = threading.Lock()


def used():
    if not os.path.exists(LOG):
        return 0
    n = 0
    with open(LOG) as f:
        for line in f:
            r = json.loads(line)
            n += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    return n


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
            out = sample(ids, max_tokens, temp, stop)
        except Exception as e:  # counted as failure by the agent
            with lock, open(LOG, "a") as f:
                f.write(json.dumps({"t": t0, "error": repr(e)[:300], "prompt_tokens": len(ids),
                                    "completion_tokens": 0}) + "\n")
            return self._send(500, {"error": {"message": f"{BACKEND} sampling failed"}})
        content = tok.decode(out, skip_special_tokens=True)
        if "</think>" in content:
            content = content.split("</think>", 1)[1].lstrip()
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"t": t0, "dt": round(time.time() - t0, 2), "prompt_tokens": len(ids),
                                "orig_prompt_tokens": orig, "completion_tokens": len(out),
                                "temperature": temp, "max_tokens": max_tokens,
                                "backend": BACKEND}) + "\n")
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
