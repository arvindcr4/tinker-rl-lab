#!/usr/bin/env python3
"""Pass-through OpenAI proxy: native BALROG `vllm` client -> shared trained-actor endpoint.

The native vllm client hardcodes api_key="EMPTY" and sends no chat_template_kwargs, so this proxy
(1) adds the bearer key, (2) adds chat_template_kwargs.enable_thinking=false (cost deviation, see README),
(3) caps in-flight requests at PROXY_CONC (lane rule <=4), (4) logs every call to PROXY_LOG.
Messages, model, temperature and max_tokens are forwarded unchanged; vLLM applies the chat template.
Path prefix /<tag>/v1/... is used only to tag the log line with the env (e.g. /babyai/v1/chat/completions).
If the prompt + max_tokens exceeds the 32768 context, vLLM rejects it; the proxy then retries once with
max_tokens = remaining context (logged as ctx_clamped).
"""
import json, os, re, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/")
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
LOG = os.environ["PROXY_LOG"]
PORT = int(os.environ.get("PORT", "8793"))
sem = threading.Semaphore(int(os.environ.get("PROXY_CONC", "4")))
lock = threading.Lock()
CTX = 32768


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with sem, urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        self._send(200, {"object": "list", "data": [{"id": "pavlov-public-portfolio-bf16", "object": "model"}]})

    def do_POST(self):
        tag = self.path.strip("/").split("/")[0] if not self.path.startswith("/v1") else "-"
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        req["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        rec = {"t": t0, "tag": tag, "temperature": req.get("temperature"), "max_tokens": req.get("max_tokens")}
        code, out = 200, None
        try:
            try:
                out = post(req)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                m = re.search(r"(\d+) input tokens|prompt contains (\d+)|has (\d+) input", msg)
                if e.code == 400 and "context" in msg.lower() and m:
                    n_in = int(next(g for g in m.groups() if g))
                    req["max_tokens"] = max(1, CTX - n_in - 16)
                    rec["ctx_clamped"] = req["max_tokens"]
                    out = post(req)
                else:
                    raise RuntimeError(f"HTTP {e.code}: {msg[:300]}")
            u = out.get("usage") or {}
            rec.update(dt=round(time.time() - t0, 3), prompt_tokens=u.get("prompt_tokens", 0),
                       completion_tokens=u.get("completion_tokens", 0),
                       finish=out["choices"][0].get("finish_reason"))
        except Exception as e:  # native client retries (max_retries=5); episode fails if all fail
            code = 502
            rec.update(dt=round(time.time() - t0, 3), error=repr(e)[:400], prompt_tokens=0, completion_tokens=0)
            out = {"error": {"message": "upstream failure: " + repr(e)[:200]}}
        with lock, open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._send(code, out)


if __name__ == "__main__":
    print(f"proxy :{PORT} -> {BASE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
