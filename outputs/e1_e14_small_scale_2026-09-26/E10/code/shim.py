"""Local OpenAI-compatible /v1/chat/completions shim over Tinker (Qwen3.6-35B-A3B base, no adapter, non-thinking).
Tools rendered by the model's own chat template; <tool_call><function=..><parameter=..> output parsed back to OpenAI tool_calls.
Every request/response is appended to raw/shim_log.jsonl. Localhost only."""
import json, re, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
if os.environ.get('ARM'):  # paired vLLM arm backend (same rendered prompt via /v1/completions)
    import vc as tk
else:
    import tk

LOG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("shim_log.jsonl")
DEFAULT_MAX = 4096
_lock = threading.Lock()
CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def convert_messages(msgs):
    out = []
    for m in msgs:
        # inspect sends OpenAI "developer" role for system prompts; Qwen template only knows "system".
        m2 = {"role": "system" if m["role"] == "developer" else m["role"], "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args = {"_raw": args}
                calls.append({"type": "function", "function": {"name": c["function"]["name"], "arguments": args}})
            m2["tool_calls"] = calls
        out.append(m2)
    return out


def parse_calls(text, tools):
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools or []}
    calls = []
    for name, body in CALL_RE.findall(text):
        name = name.strip(); props = schemas.get(name, {}); args = {}
        for k, v in PARAM_RE.findall(body):
            k = k.strip()
            if props.get(k, {}).get("type") == "string":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    content = text.split("<tool_call>")[0].strip() if calls else text.strip()
    return content, calls


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "messages" not in req:
            self.send_response(404); self.end_headers(); return
        tools = req.get("tools") or None
        msgs = convert_messages(req["messages"])
        mt = req.get("max_tokens") or req.get("max_completion_tokens") or DEFAULT_MAX
        t0 = time.time()
        try:
            r = tk.sample(msgs, mt, tools=tools, temperature=req.get("temperature") or 0.0)
            content, calls = parse_calls(r["text"], tools)
            msg = {"role": "assistant", "content": content or None}
            if calls:
                msg["tool_calls"] = calls
            body = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(t0),
                    "model": req.get("model"), "choices": [{"index": 0, "message": msg,
                    "finish_reason": "tool_calls" if calls else ("length" if r["stop_reason"] == "length" else "stop")}],
                    "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                              "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}}
            code, raw = 200, r["text"]
        except Exception as e:
            code, body, raw = 500, {"error": {"message": f"{type(e).__name__}: {e}"}}, None
        with _lock, LOG.open("a") as f:
            f.write(json.dumps({"t": t0, "request": req, "raw_completion": raw, "response": body}) + "\n")
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
