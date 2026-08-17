"""Deploy the frozen Pavlov Tinker sampler behind a secured OpenAI API bridge.

The deployed bridge is infrastructure, not a benchmark result. Its default
paid-inference budget is zero: a separate, explicit budget change is required
before any chat-completion request can reach Tinker.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from tinker_openai_bridge_protocol import (
    bearer_token,
    estimate_tinker_usd,
    normalise_openai_messages_for_qwen,
    openai_chat_stream_events,
    parse_qwen_tool_calls,
)


APP_NAME = "pavlov-tinker-openai-bridge"
MODEL_ALIAS = "pavlov-qwen36-tinker"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SAMPLER_PATH = (
    "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/"
    "sampler_weights/seed809_final"
)
HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-"
    "tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
HF_REVISION = "checkpoint-seed809-stepfinal-9f777c4018b6"
HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
DEFAULT_MAX_USD = Decimal("0.00")
MAX_COMPLETION_TOKENS = 16_384
SOURCE_DIR = Path(__file__).resolve().parent

bridge_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi[standard]==0.116.1",
        "huggingface-hub==1.27.0",
        "tinker==0.24.1",
        "transformers==5.5.4",
        "wandb==0.21.0",
    )
    .add_local_file(
        SOURCE_DIR / "tinker_openai_bridge_protocol.py",
        "/root/tinker_openai_bridge_protocol.py",
        copy=True,
    )
)
core_secret = modal.Secret.from_name("pavlov-e1-e14")
auth_secret = modal.Secret.from_name("pavlov-tinker-bridge-auth")
budget_secret = modal.Secret.from_name("pavlov-tinker-bridge-budget")
ledger = modal.Dict.from_name("pavlov-tinker-bridge-ledger", create_if_missing=True)
app = modal.App(APP_NAME, include_source=True)


@app.cls(
    image=bridge_image,
    secrets=[core_secret, auth_secret, budget_secret],
    cpu=4.0,
    memory=16_384,
    timeout=60 * 60,
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=1)
class TinkerOpenAIBridge:
    @modal.enter()
    def startup(self) -> None:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        import tinker
        import wandb

        info = HfApi(token=os.environ["HF_TOKEN"]).model_info(
            HF_REPO,
            revision=HF_REVISION,
        )
        if info.sha != HF_COMMIT:
            raise RuntimeError(f"immutable HF checkpoint drift: {info.sha}")

        self.maximum_usd = Decimal(
            os.environ.get(
                "TINKER_BRIDGE_AUTHORIZED_TOTAL_USD",
                os.environ.get("TINKER_BRIDGE_MAX_USD", "0.00"),
            )
        )
        if self.maximum_usd < DEFAULT_MAX_USD:
            raise RuntimeError("TINKER_BRIDGE_MAX_USD must be non-negative")
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.wandb_run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group="pavlov-e1-e14-modal-bridge-20260816",
            job_type="inference-bridge",
            name=f"tinker-openai-bridge-{uuid.uuid4().hex[:8]}",
            tags=["modal", "tinker", "openai-compatible", "infrastructure"],
            mode="online",
            config={
                "evidence_class": "infrastructure_not_model_score",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampler_path": SAMPLER_PATH,
                "hf_repo": HF_REPO,
                "hf_revision": HF_REVISION,
                "hf_commit": HF_COMMIT,
                "maximum_usd": str(self.maximum_usd),
            },
            reinit=True,
        )
        if self.wandb_run is None or not getattr(self.wandb_run, "id", None):
            raise RuntimeError("W&B online initialization failed before Tinker setup")
        self.sampling_client = tinker.ServiceClient(
            user_metadata={
                "campaign": "pavlov-e1-e14-modal",
                "component": "openai-compatible-bridge",
                "wandb_run_id": self.wandb_run.id,
            }
        ).create_sampling_client(model_path=SAMPLER_PATH)

    @modal.exit()
    def shutdown(self) -> None:
        if getattr(self, "wandb_run", None) is not None:
            self.wandb_run.summary.update(
                {
                    "status": "STOPPED",
                    "bridge/charged_usd": float(ledger.get("charged_usd", 0.0)),
                    "bridge/reserved_usd": float(ledger.get("reserved_usd", 0.0)),
                }
            )
            self.wandb_run.finish(exit_code=0)

    def _authorize(self, authorization: str | None) -> None:
        credential = bearer_token(authorization)
        if credential is None or not hmac.compare_digest(
            credential,
            os.environ["TINKER_BRIDGE_API_KEY"],
        ):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="invalid bearer credential")

    def _reserve(self, projected_usd: Decimal) -> None:
        with self.lock:
            charged = Decimal(str(ledger.get("charged_usd", 0.0)))
            reserved = Decimal(str(ledger.get("reserved_usd", 0.0)))
            if charged + reserved + projected_usd > self.maximum_usd:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=402,
                    detail=(
                        "Tinker bridge paid inference is not authorized within the "
                        "current persistent budget"
                    ),
                )
            ledger["reserved_usd"] = float(reserved + projected_usd)

    def _settle(self, projected_usd: Decimal, actual_usd: Decimal) -> None:
        with self.lock:
            charged = Decimal(str(ledger.get("charged_usd", 0.0)))
            reserved = Decimal(str(ledger.get("reserved_usd", 0.0)))
            ledger["reserved_usd"] = float(max(Decimal("0"), reserved - projected_usd))
            ledger["charged_usd"] = float(charged + actual_usd)
            ledger["completed_calls"] = int(ledger.get("completed_calls", 0)) + 1

    def _charge_failed_reservation(self, projected_usd: Decimal) -> None:
        with self.lock:
            charged = Decimal(str(ledger.get("charged_usd", 0.0)))
            reserved = Decimal(str(ledger.get("reserved_usd", 0.0)))
            ledger["reserved_usd"] = float(max(Decimal("0"), reserved - projected_usd))
            ledger["charged_usd"] = float(charged + projected_usd)
            ledger["failed_calls"] = int(ledger.get("failed_calls", 0)) + 1

    @modal.asgi_app()
    def web(self):
        from fastapi import Body, FastAPI, Header, HTTPException
        from pydantic import BaseModel, ConfigDict, Field
        from starlette.responses import StreamingResponse

        bridge = self
        web_app = FastAPI(title="Pavlov Tinker OpenAI Bridge", docs_url=None, redoc_url=None)

        class ChatCompletionRequest(BaseModel):
            model_config = ConfigDict(extra="allow")

            model: str = MODEL_ALIAS
            messages: list[dict[str, Any]]
            tools: list[dict[str, Any]] | None = None
            tool_choice: Any | None = None
            max_tokens: int | None = Field(default=None, ge=1)
            max_completion_tokens: int | None = Field(default=None, ge=1)
            temperature: float = Field(default=0.2, ge=0.0)
            top_p: float = Field(default=0.95, gt=0.0, le=1.0)
            stop: list[str] | str | None = None
            seed: int | None = None
            n: int = Field(default=1, ge=1)
            stream: bool = False

        @web_app.get("/health")
        async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            bridge._authorize(authorization)
            return {
                "status": "READY",
                "evidence_class": "infrastructure_not_model_score",
                "score": None,
                "model": MODEL_ALIAS,
                "model_revision": MODEL_REVISION,
                "hf_commit": HF_COMMIT,
                "wandb_run_id": bridge.wandb_run.id,
                "wandb_url": bridge.wandb_run.url,
                "budget": {
                    "maximum_usd": str(bridge.maximum_usd),
                    "charged_usd": ledger.get("charged_usd", 0.0),
                    "reserved_usd": ledger.get("reserved_usd", 0.0),
                },
            }

        @web_app.get("/v1/models")
        async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            bridge._authorize(authorization)
            return {
                "object": "list",
                "data": [{"id": MODEL_ALIAS, "object": "model", "owned_by": "pavlov"}],
            }

        @web_app.post("/v1/chat/completions")
        async def chat_completions(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> Any:
            bridge._authorize(authorization)
            request = ChatCompletionRequest.model_validate(payload)
            if request.model not in {MODEL_ALIAS, MODEL_ID}:
                raise HTTPException(status_code=404, detail="unknown model")
            if request.n != 1:
                raise HTTPException(status_code=400, detail="only n=1 pass@1 requests are supported")
            max_tokens = request.max_completion_tokens or request.max_tokens or 4096
            if max_tokens > MAX_COMPLETION_TOKENS:
                raise HTTPException(
                    status_code=400,
                    detail=f"max completion tokens exceeds {MAX_COMPLETION_TOKENS}",
                )

            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if request.tools:
                template_kwargs["tools"] = request.tools
            try:
                template_messages = normalise_openai_messages_for_qwen(request.messages)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            prompt_text = await asyncio.to_thread(
                bridge.tokenizer.apply_chat_template,
                template_messages,
                **template_kwargs,
            )
            prompt_tokens = await asyncio.to_thread(
                bridge.tokenizer.encode,
                prompt_text,
                add_special_tokens=False,
            )
            projected_usd = Decimal(str(estimate_tinker_usd(len(prompt_tokens), max_tokens)))
            bridge._reserve(projected_usd)
            started = time.monotonic()

            try:
                import tinker.types as T

                stops = [request.stop] if isinstance(request.stop, str) else request.stop or []

                def sample():
                    return bridge.sampling_client.sample(
                        T.ModelInput.from_ints(prompt_tokens),
                        num_samples=1,
                        sampling_params=T.SamplingParams(
                            max_tokens=max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            stop=stops,
                            seed=request.seed,
                        ),
                    ).result()

                result = await asyncio.to_thread(sample)
                output_tokens = list(result.sequences[0].tokens)
                output_text = await asyncio.to_thread(
                    bridge.tokenizer.decode,
                    output_tokens,
                    skip_special_tokens=True,
                )
                content, tool_calls = parse_qwen_tool_calls(output_text)
                actual_usd = Decimal(
                    str(estimate_tinker_usd(len(prompt_tokens), len(output_tokens)))
                )
                bridge._settle(projected_usd, actual_usd)
                bridge.wandb_run.log(
                    {
                        "bridge/prompt_tokens": len(prompt_tokens),
                        "bridge/completion_tokens": len(output_tokens),
                        "bridge/actual_usd": float(actual_usd),
                        "bridge/tool_call_count": len(tool_calls),
                        "bridge/latency_seconds": time.monotonic() - started,
                    }
                )
            except Exception as exc:
                bridge._charge_failed_reservation(projected_usd)
                bridge.wandb_run.log(
                    {
                        "bridge/request_failed": 1,
                        "bridge/conservative_charged_usd": float(projected_usd),
                    }
                )
                raise HTTPException(status_code=500, detail="Tinker sampling failed") from exc

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if request.stream:
                return StreamingResponse(
                    iter(
                        openai_chat_stream_events(
                            completion_id=completion_id,
                            created=created,
                            model=MODEL_ALIAS,
                            content=content,
                            tool_calls=tool_calls,
                            prompt_tokens=len(prompt_tokens),
                            completion_tokens=len(output_tokens),
                        )
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ALIAS,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(output_tokens),
                    "total_tokens": len(prompt_tokens) + len(output_tokens),
                },
            }

        return web_app
