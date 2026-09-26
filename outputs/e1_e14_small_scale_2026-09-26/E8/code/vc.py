"""Drop-in replacement for tk.sample() that hits the paired vLLM endpoints (trained / base) instead of Tinker.
Text-only: POST /v1/completions with the IDENTICAL chat-templated prompt string tk.build_input renders.
With images: /v1/chat/completions (enable_thinking=false), same messages, image bytes preprocessed exactly as tk did.
Streams (SSE) to avoid long-request limits. ARM env var selects endpoint: trained | base."""
import base64, io, json, os, threading, time, urllib.request
import tk

ENDPOINTS = {"trained": ("https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "pavlov-public-portfolio-bf16"),
             "base": ("https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "qwen36-base-bf16")}
ARM = os.environ["ARM"]
URL, MODEL = ENDPOINTS[ARM]
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
USAGE = {"prefill": 0, "sample": 0, "calls": 0}
_lock = threading.Lock()


def tok():
    return tk.tok()


def _post_stream(path, body, timeout=1800):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    text, usage, finish = [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = ch.get("text") if "text" in ch else (ch.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(text), usage or {}, finish


def _img_bytes(data):
    c = tk._img_chunk(data)  # same re-encode (Tinker 2 MiB limit) so both engines see identical bytes
    return c.data, c.format


def sample(messages, max_tokens, tools=None, images=None, temperature=0.0, enable_thinking=False, stop=None):
    images = list(images or [])
    for attempt in range(3):
        try:
            if images:
                it = iter(images); msgs = []
                for m in messages:
                    if isinstance(m["content"], list):
                        parts = []
                        for p in m["content"]:
                            if p.get("type") == "image":
                                d, fmt = _img_bytes(next(it))
                                parts.append({"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{base64.b64encode(d).decode()}"}})
                            else:
                                parts.append(p)
                        msgs.append({**m, "content": parts})
                    else:
                        msgs.append(m)
                body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
                        "stop": stop or ["<|im_end|>"], "chat_template_kwargs": {"enable_thinking": enable_thinking}}
                if tools:
                    body["tools"] = tools
                text, usage, finish = _post_stream("/v1/chat/completions", body)
            else:
                _, prompt = tk.build_input(messages, tools, None, enable_thinking)
                body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                        "stop": stop or ["<|im_end|>"], "skip_special_tokens": False}
                text, usage, finish = _post_stream("/v1/completions", body)
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 2:
                raise
            time.sleep(30)
    for s in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(s)[0]
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    with _lock:
        USAGE["prefill"] += pt; USAGE["sample"] += ct; USAGE["calls"] += 1
    return {"text": text, "prompt_tokens": pt, "completion_tokens": ct, "stop_reason": finish}
