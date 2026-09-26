"""Paired vLLM arm shim. Same HTTP interface and same chat-templated token ids as tinker_shim.py
(Qwen3.6 chat template, enable_thinking=False, temperature=0), but sampled via raw /v1/completions on a Modal vLLM endpoint.
Usage: vllm_shim.py <port> <base_url> <model_id>   (env TRAINED_ACTOR_API_KEY, SHIM_LOG)"""
import json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import AutoTokenizer

port, BASE, MODEL_ID = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")
LOG = os.environ.get("SHIM_LOG", "shim_usage.jsonl")
lock = threading.Lock()


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        t0 = time.time()
        resp = None
        for attempt in range(3):
            try:
                r = post("/v1/completions", {"model": MODEL_ID, "prompt": ids, "max_tokens": int(req.get("max_tokens", 512)),
                                             "temperature": 0.0})
                resp = {"text": r["choices"][0]["text"], "prompt_tokens": r["usage"]["prompt_tokens"],
                        "sample_tokens": r["usage"]["completion_tokens"]}
                break
            except Exception as e:
                resp = {"error": repr(e)[:300], "prompt_tokens": len(ids), "sample_tokens": 0}
                time.sleep(10)
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"tag": req.get("tag"), "prompt_tokens": resp["prompt_tokens"], "sample_tokens": resp["sample_tokens"],
                                "error": resp.get("error"), "latency_s": round(time.time() - t0, 2)}) + "\n")
        b = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


print(f"vllm shim ready on {port} -> {BASE} ({MODEL_ID})", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
