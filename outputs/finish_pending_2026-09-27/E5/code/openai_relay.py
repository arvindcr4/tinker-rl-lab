"""E5 (2026-09-27) token-gated OpenAI relay with a hard spend guard.

Why: the local OPENAI_API_KEY has no credits and OpenRouter is exhausted; the only funded OpenAI key
lives in Modal secret `ai-scientist-keys` and cannot be read locally. This relay forwards ONLY the two
native tau3 roles (gpt-4.1-2025-04-14 chat for user simulator / NL judge, text-embedding-3-large for
query embeddings) to api.openai.com, unchanged, and refuses once cumulative priced spend >= CAP_USD.

WARNING: anyone holding RELAY_TOKEN can spend on the funded key until the cap; the token lives only in a
chmod-600 scratch file + Modal secret `e5-relay-token-0927`. Stop the app and delete the secret after the run.
"""
import json
import os
import time

import modal

app = modal.App("e5-openai-relay-0927")
image = modal.Image.debian_slim().pip_install("fastapi[standard]", "httpx")
ledger = modal.Dict.from_name("e5-relay-ledger-0927", create_if_missing=True)

CAP_USD = 60.0
ALLOWED = {"gpt-4.1-2025-04-14", "text-embedding-3-large"}
PRICE = {  # USD per token, OpenAI list prices recorded 2026-09-12 in EXECUTION_REQUIREMENTS.md
    "gpt-4.1-2025-04-14": {"in": 2e-6, "cached": 0.5e-6, "out": 8e-6},
    "text-embedding-3-large": {"in": 0.13e-6, "cached": 0.13e-6, "out": 0.0},
}


def cost(model, usage):
    p = PRICE[model]
    pin = usage.get("prompt_tokens", 0) or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    out = usage.get("completion_tokens", 0) or 0
    return (pin - cached) * p["in"] + cached * p["cached"] + out * p["out"]


@app.function(image=image, secrets=[modal.Secret.from_name("ai-scientist-keys"),
                                    modal.Secret.from_name("e5-relay-token-0927")],
              timeout=900, min_containers=1, max_containers=1)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def relay():
    import httpx
    from fastapi import FastAPI, Request, Response

    web = FastAPI()
    TOTALS = {"spent_usd": ledger.get("spent_usd", 0.0), "n_chat_completions": ledger.get("n_chat_completions", 0),
              "n_embeddings": ledger.get("n_embeddings", 0)}
    client = httpx.AsyncClient(timeout=600)

    @web.post("/v1/{path:path}")
    async def forward(path: str, request: Request):
        if request.headers.get("authorization", "") != "Bearer " + os.environ["RELAY_TOKEN"]:
            return Response(status_code=401, content=b'{"error":"bad relay token"}')
        if path not in ("chat/completions", "embeddings"):
            return Response(status_code=404, content=b'{"error":"path not allowed"}')
        body = await request.body()
        req = json.loads(body)
        if req.get("model") not in ALLOWED or req.get("stream"):
            return Response(status_code=400, content=b'{"error":"model/stream not allowed"}')
        spent = TOTALS["spent_usd"]
        if spent >= CAP_USD:
            return Response(status_code=402, content=json.dumps({"error": f"E5 relay cap reached ({spent:.4f} USD)"}).encode())
        r = await client.post("https://api.openai.com/v1/" + path, content=body,
                              headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                       "Content-Type": "application/json"})
        c = 0.0
        if r.status_code == 200:
            usage = r.json().get("usage") or {}
            c = cost(req["model"], usage)
            key = "n_" + path.replace("/", "_")
            # single container (max_containers=1); no await between read and write of the in-process totals
            TOTALS["spent_usd"] += c
            TOTALS[key] = TOTALS.get(key, 0) + 1
            await ledger.put.aio("spent_usd", TOTALS["spent_usd"])
            await ledger.put.aio(key, TOTALS[key])
        print(json.dumps({"t": time.time(), "path": path, "model": req["model"], "status": r.status_code, "usd": c}), flush=True)
        return Response(status_code=r.status_code, content=r.content, media_type="application/json")

    @web.get("/ledger")
    async def show(request: Request):
        if request.headers.get("authorization", "") != "Bearer " + os.environ["RELAY_TOKEN"]:
            return Response(status_code=401)
        return TOTALS

    return web

