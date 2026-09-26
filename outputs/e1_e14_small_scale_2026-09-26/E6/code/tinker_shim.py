"""Local HTTP shim: POST /chat {"messages":[...], "max_tokens":N, "tag":...} -> {"text","prompt_tokens","sample_tokens"}.
Actor: Qwen/Qwen3.6-35B-A3B base, no adapter, enable_thinking=False, temperature=0. Appends usage to $SHIM_LOG."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tinker
from tinker import types as T
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
tok = AutoTokenizer.from_pretrained(MODEL)
svc = tinker.ServiceClient()
sampler = svc.create_sampling_client(base_model=MODEL)
LOG = os.environ.get("SHIM_LOG", "shim_usage.jsonl")
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        try:
            r = sampler.sample(T.ModelInput.from_ints(ids), num_samples=1,
                               sampling_params=T.SamplingParams(max_tokens=int(req.get("max_tokens", 512)), temperature=0.0)).result()
            out = list(r.sequences[0].tokens)
            resp = {"text": tok.decode(out, skip_special_tokens=True), "prompt_tokens": len(ids), "sample_tokens": len(out)}
        except Exception as e:
            resp = {"error": repr(e), "prompt_tokens": len(ids), "sample_tokens": 0}
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"tag": req.get("tag"), "prompt_tokens": resp["prompt_tokens"],
                                "sample_tokens": resp["sample_tokens"], "error": resp.get("error")}) + "\n")
        b = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
print(f"shim ready on {port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
