"""Loopback relay: OpenHands/LiteLLM -> shared Modal vLLM actor.

Only change to requests: injects chat_template_kwargs.enable_thinking=false (campaign
non-thinking default) and forces the served model id. Logs per-request usage to a JSONL ledger.
Listens on 127.0.0.1 only.
"""
import json, os, time, uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/") + "/v1"
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
MODEL = "pavlov-public-portfolio-bf16"
LEDGER = os.environ.get("RELAY_LEDGER", "/opt/e9/runs/relay_ledger.jsonl")
app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(1200.0, connect=60.0))


def log(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def repair(j, stop):
    """Restore content dropped by the server-side tool parser, byte-exact from the token ids."""
    fixed = False
    for c in j.get("choices", []):
        m = c.get("message") or {}
        if (m.get("content") or m.get("tool_calls")) or not c.get("token_ids"):
            continue
        d = await client.post(UPSTREAM.rsplit("/v1", 1)[0] + "/detokenize",
                              json={"model": MODEL, "tokens": c["token_ids"]},
                              headers={"Authorization": f"Bearer {KEY}"})
        text = d.json()["prompt"]
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]
        if c.get("finish_reason") == "stop":  # vLLM excludes the matched stop string from text
            for s in ([stop] if isinstance(stop, str) else (stop or [])):
                if s and text.endswith(s):
                    text = text[: -len(s)]
                    break
        m["content"] = text
        m["tool_calls"] = None
        c["message"] = m
        fixed = True
    return fixed


@app.get("/v1/models")
async def models():
    r = await client.get(UPSTREAM + "/models", headers={"Authorization": f"Bearer {KEY}"})
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    body["model"] = MODEL
    body.pop("stream_options", None)
    ctk = body.get("chat_template_kwargs") or {}
    ctk["enable_thinking"] = False
    body["chat_template_kwargs"] = ctk
    # vLLM runs the qwen3_coder tool parser even when the request carries no `tools`; it then
    # swallows OpenHands' prompt-based (mock) <function=...> calls, returning empty content.
    # Ask for the raw token ids so the relay can restore the model's exact text (see repair()).
    no_tools = not body.get("tools")
    if no_tools:
        body["return_token_ids"] = True
    rid, t0 = uuid.uuid4().hex[:12], time.time()
    if os.environ.get("RELAY_DUMP_DIR"):
        with open(os.path.join(os.environ["RELAY_DUMP_DIR"], rid + ".req.json"), "w") as f:
            json.dump(body, f)
    task = req.headers.get("x-e9-task", os.environ.get("E9_TASK", ""))
    try:
        r = await client.post(UPSTREAM + "/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {KEY}"})
    except Exception as e:  # transport failure -> 502 so litellm sees an error
        log({"id": rid, "t": t0, "dt": time.time() - t0, "status": 502, "err": repr(e)[:500]})
        return JSONResponse({"error": {"message": repr(e)[:500]}}, status_code=502)
    rec = {"id": rid, "t": t0, "dt": round(time.time() - t0, 2), "status": r.status_code,
           "n_msgs": len(body.get("messages", [])), "max_tokens": body.get("max_tokens"),
           "temperature": body.get("temperature")}
    if no_tools and r.status_code == 200:
        try:
            j = r.json()
            if await repair(j, body.get("stop")):
                rec["repaired"] = True
            for c in j.get("choices", []):
                c.pop("token_ids", None)
            j.pop("prompt_token_ids", None)
            r = httpx.Response(200, json=j)
        except Exception as e:
            rec["repair_err"] = repr(e)[:300]
    try:
        j = r.json()
        rec["usage"] = j.get("usage")
        try:
            m = j["choices"][0]["message"]
            rec["finish"] = j["choices"][0].get("finish_reason")
            rec["content"] = (m.get("content") or "")[:1500]
            rec["reasoning"] = (m.get("reasoning") or m.get("reasoning_content") or "")[:1500]
            rec["tool_calls"] = json.dumps(m.get("tool_calls"))[:1500] if m.get("tool_calls") else None
        except Exception:
            pass
        rec["n_tools"] = len(body.get("tools") or [])
        if r.status_code != 200:
            rec["err"] = str(j)[:500]
    except Exception:
        rec["err"] = r.text[:500]
    log(rec)
    return Response(r.content, status_code=r.status_code, media_type="application/json")
