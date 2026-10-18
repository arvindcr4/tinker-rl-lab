"""Deploy the frozen Pavlov Tinker sampler behind a secured OpenAI API bridge.

The deployed bridge is infrastructure, not a benchmark result. Its default
paid-inference budget is zero: a separate, explicit budget change is required
before any chat-completion request can reach Tinker.
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import warnings
import os
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

try:
    from .tinker_openai_bridge_protocol import (
        bearer_token,
        build_responses_object,
        estimate_tinker_usd,
        iter_responses_sse_events,
        normalise_openai_messages_for_qwen,
        openai_chat_stream_events,
        parse_qwen_tool_calls,
        responses_input_to_chat_messages,
        responses_tools_to_chat_tools,
    )
except ImportError:  # Modal deploy imports this file as a top-level module.
    from tinker_openai_bridge_protocol import (
        bearer_token,
        build_responses_object,
        estimate_tinker_usd,
        iter_responses_sse_events,
        normalise_openai_messages_for_qwen,
        openai_chat_stream_events,
        parse_qwen_tool_calls,
        responses_input_to_chat_messages,
        responses_tools_to_chat_tools,
    )


APP_NAME = "pavlov-tinker-openai-bridge"
MODEL_ALIAS = "pavlov-qwen36-tinker"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SAMPLER_PATH = "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/sampler_weights/seed809_final"
HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
HF_REVISION = "checkpoint-seed809-stepfinal-9f777c4018b6"
HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
DEFAULT_MAX_USD = Decimal("0.00")
MAX_COMPLETION_TOKENS = 16_384
MODEL_CONTEXT_TOKENS = 65_536
CONTEXT_PREFIX_TOKENS = 8_192
SOURCE_DIR = Path(__file__).resolve().parent
BASE_MODEL_ALIAS = "pavlov-qwen36-base-v2"
CAMPAIGN_ID = "pavlov-e1-e14"

try:
    from .campaign_ledger_v2 import CampaignLedger
    from .evidence_v2 import approval_valid, canonical
except ImportError:
    from campaign_ledger_v2 import CampaignLedger
    from evidence_v2 import approval_valid, canonical


def resolve_serving_identity(config: dict, *, binding: dict, trusted_issuers: dict) -> dict:
    mode = config.get("SERVING_MODE")
    sampler = config.get("SAMPLER_PATH", "").strip()
    if mode not in {"final_adapter", "base"} or not sampler:
        raise RuntimeError("explicit serving mode and immutable sampler required")
    if mode == "final_adapter" and sampler != SAMPLER_PATH:
        raise RuntimeError("final adapter sampler identity mismatch")
    expected = {
        "campaign": CAMPAIGN_ID,
        "phase": "serving",
        "suite": "bridge",
        "run": sampler,
        "checkpoint": HF_COMMIT if mode == "final_adapter" else MODEL_REVISION,
        "plan": "pavlov-serving-v2",
    }
    if not approval_valid(
        binding,
        "sampler_export_binding",
        trusted_issuers=trusted_issuers,
        expected_identity=expected,
    ):
        raise RuntimeError("trusted sampler-to-export binding required")
    identity = {
        "serving_mode": mode,
        "sampler_path": sampler,
        "tokenizer_revision": MODEL_REVISION,
        "checkpoint": expected["checkpoint"],
        "model": MODEL_ALIAS if mode == "final_adapter" else BASE_MODEL_ALIAS,
    }
    identity["run_manifest_sha256"] = hashlib.sha256(canonical(identity)).hexdigest()
    return identity


def is_supported_request_model(model: str) -> bool:
    """Accept only the pinned model identities and LiteLLM routing forms."""

    return model in {
        MODEL_ALIAS,
        BASE_MODEL_ALIAS,
        f"openai/{MODEL_ALIAS}",
        f"openai/{BASE_MODEL_ALIAS}",
    }


bridge_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi[standard]==0.116.1",
        "huggingface-hub==1.27.0",
        "tinker==0.30.0",
        "transformers==5.5.4",
        "wandb==0.21.0",
        "cryptography==50.0.0",
    )
    .add_local_file(SOURCE_DIR / "campaign_ledger_v2.py", "/root/campaign_ledger_v2.py", copy=True)
    .add_local_file(SOURCE_DIR / "evidence_v2.py", "/root/evidence_v2.py", copy=True)
    .add_local_file(
        SOURCE_DIR / "tinker_openai_bridge_protocol.py",
        "/root/tinker_openai_bridge_protocol.py",
        copy=True,
    )
)
core_secret = modal.Secret.from_name("pavlov-e1-e14")
auth_secret = modal.Secret.from_name("pavlov-tinker-bridge-auth")
budget_secret = modal.Secret.from_name("pavlov-tinker-bridge-budget")
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
        # Fail before any service interaction if provenance or durable accounting is absent.
        self.identity = resolve_serving_identity(
            dict(os.environ),
            binding=json.loads(Path(os.environ["TINKER_SERVING_BINDING_PATH"]).read_bytes()),
            trusted_issuers=json.loads(
                Path(os.environ["TINKER_TRUSTED_ISSUERS_PATH"]).read_bytes()
            ),
        )
        self.ledger = CampaignLedger(
            os.environ["TINKER_CAMPAIGN_LEDGER_PATH"], campaign=CAMPAIGN_ID
        )
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
        if not self.maximum_usd.is_finite() or self.maximum_usd < DEFAULT_MAX_USD:
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
                "sampler_path_configured": self.identity["sampler_path"],
                "serving_identity": self.identity,
                "hf_repo": HF_REPO if self.identity["serving_mode"] == "final_adapter" else None,
                "hf_revision": HF_REVISION
                if self.identity["serving_mode"] == "final_adapter"
                else None,
                "hf_commit": HF_COMMIT
                if self.identity["serving_mode"] == "final_adapter"
                else None,
                "maximum_usd": str(self.maximum_usd),
            },
            reinit=True,
        )
        if self.wandb_run is None or not getattr(self.wandb_run, "id", None):
            raise RuntimeError("W&B online initialization failed before Tinker setup")
        sampler_path = self.identity["sampler_path"]
        svc = tinker.ServiceClient(
            user_metadata={
                "campaign": "pavlov-e1-e14-modal",
                "component": "openai-compatible-bridge",
                "wandb_run_id": self.wandb_run.id,
            }
        )
        self.sampler_path_resolved = sampler_path
        self.sampling_client = svc.create_sampling_client(model_path=self.sampler_path_resolved)
        self.wandb_run.config.update(
            {
                "sampler_path_resolved": self.sampler_path_resolved,
                "serving_mode": self.identity["serving_mode"],
                "run_manifest_sha256": self.identity["run_manifest_sha256"],
            },
            allow_val_change=True,
        )

    @modal.exit()
    def shutdown(self) -> None:
        if getattr(self, "wandb_run", None) is not None:
            self.wandb_run.summary.update(
                {
                    "status": "STOPPED",
                    "bridge/charged_usd": float(self.ledger.snapshot()["charged_usd"]),
                    "bridge/reserved_usd": float(self.ledger.snapshot()["reserved_usd"]),
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

    def _reserve(self, projected_usd: Decimal, *, request_identity: str | None = None) -> str:
        from fastapi import HTTPException

        try:
            snapshot = self.ledger.snapshot()
            if (
                Decimal(snapshot["charged_usd"]) + Decimal(snapshot["reserved_usd"]) + projected_usd
                > self.maximum_usd
            ):
                raise RuntimeError("bridge authorization cap exhausted")
            return self.ledger.reserve(
                uuid.uuid4().hex,
                projected_usd,
                identity=request_identity or self.identity["run_manifest_sha256"],
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc

    def _settle(self, request_id: str, actual_usd: Decimal) -> None:
        self.ledger.settle(request_id, actual_usd)

    def _charge_failed_reservation(self, request_id: str) -> None:
        self.ledger.fail(request_id)

    def _track(self, values: dict) -> None:
        try:
            self.wandb_run.log(values)
        except Exception as exc:
            try:
                warnings.warn(
                    f"nonfinancial tracking delivery failed: {type(exc).__name__}", stacklevel=2
                )
            except Exception:
                self.tracking_delivery_failed = (
                    True  # cannot affect settlement or response delivery
                )

    async def _complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | str | None,
        seed: int | None,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Shared sampling core for chat/completions and responses.

        Identical budgeting (reserve/settle), W&B accounting, truncation,
        and tool-call parsing on both paths. Returns content, tool_calls,
        and token counts. Raises HTTPException (lazy fastapi import) on bad
        input or sampling failure.
        """
        from fastapi import HTTPException

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            template_kwargs["tools"] = tools
        try:
            template_messages = normalise_openai_messages_for_qwen(messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        prompt_text = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            template_messages,
            **template_kwargs,
        )
        prompt_tokens = await asyncio.to_thread(
            self.tokenizer.encode,
            prompt_text,
            add_special_tokens=False,
        )
        original_prompt_tokens = len(prompt_tokens)
        prompt_budget = MODEL_CONTEXT_TOKENS - max_tokens
        truncated_prompt_tokens = 0
        if len(prompt_tokens) > prompt_budget:
            raise HTTPException(
                status_code=400, detail="context limit exceeded; implicit truncation is unsupported"
            )
        projected_usd = Decimal(str(estimate_tinker_usd(len(prompt_tokens), max_tokens)))
        prompt_sha256 = hashlib.sha256(prompt_text.encode()).hexdigest()
        request_identity = hashlib.sha256(
            canonical(
                {
                    "serving": self.identity,
                    "prompt_sha256": prompt_sha256,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "seed": seed,
                    "enable_thinking": enable_thinking,
                }
            )
        ).hexdigest()
        request_id = self._reserve(projected_usd, request_identity=request_identity)
        started = time.monotonic()

        try:
            import tinker.types as T

            stops = [stop] if isinstance(stop, str) else stop or []

            def sample():
                return self.sampling_client.sample(
                    T.ModelInput.from_ints(prompt_tokens),
                    num_samples=1,
                    sampling_params=T.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stops,
                        seed=seed,
                    ),
                ).result()

            result = await asyncio.to_thread(sample)
            output_tokens = list(result.sequences[0].tokens)
            output_text = await asyncio.to_thread(
                self.tokenizer.decode,
                output_tokens,
                skip_special_tokens=True,
            )
            content, tool_calls = parse_qwen_tool_calls(output_text)
            actual_usd = Decimal(str(estimate_tinker_usd(len(prompt_tokens), len(output_tokens))))
            self._settle(request_id, actual_usd)
            self._track(
                {
                    "bridge/prompt_tokens": len(prompt_tokens),
                    "bridge/original_prompt_tokens": original_prompt_tokens,
                    "bridge/truncated_prompt_tokens": truncated_prompt_tokens,
                    "bridge/completion_tokens": len(output_tokens),
                    "bridge/estimated_usd": float(actual_usd),
                    "bridge/tool_call_count": len(tool_calls),
                    "bridge/latency_seconds": time.monotonic() - started,
                }
            )
        except Exception as exc:
            self._charge_failed_reservation(request_id)
            self._track(
                {
                    "bridge/request_failed": 1,
                    "bridge/billing_pending_reserved_usd": float(projected_usd),
                }
            )
            raise HTTPException(status_code=500, detail="Tinker sampling failed") from exc
        stop_reason = getattr(result.sequences[0], "stop_reason", None)
        if stop_reason == "length":
            finish_reason = "length"
        elif stop_reason == "stop":
            finish_reason = "tool_calls" if tool_calls else "stop"
        elif stop_reason is None and len(output_tokens) >= max_tokens:
            finish_reason = "length"
        else:
            raise HTTPException(
                status_code=502, detail="backend termination evidence missing or unsupported"
            )
        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "identity": self.identity,
            "prompt_sha256": prompt_sha256,
            "request_identity_sha256": request_identity,
            "reservation_id": request_id,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(output_tokens),
            "original_prompt_tokens": original_prompt_tokens,
            "truncated_prompt_tokens": truncated_prompt_tokens,
        }

    @modal.asgi_app()
    def web(self):
        from fastapi import Body, FastAPI, Header, HTTPException
        from pydantic import BaseModel, ConfigDict, Field, ValidationError
        from starlette.responses import StreamingResponse

        bridge = self
        web_app = FastAPI(title="Pavlov Tinker OpenAI Bridge", docs_url=None, redoc_url=None)

        class ChatCompletionRequest(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

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
            enable_thinking: bool = True

        @web_app.get("/health")
        async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            bridge._authorize(authorization)
            return {
                "status": "READY",
                "evidence_class": "infrastructure_not_model_score",
                "score": None,
                "model": bridge.identity["model"],
                "model_revision": MODEL_REVISION,
                "hf_commit": HF_COMMIT
                if bridge.identity["serving_mode"] == "final_adapter"
                else None,
                "resolved_identity": bridge.identity,
                "wandb_run_id": bridge.wandb_run.id,
                "wandb_url": bridge.wandb_run.url,
                "budget": {
                    "maximum_usd": str(bridge.maximum_usd),
                    "charged_usd": bridge.ledger.snapshot()["charged_usd"],
                    "reserved_usd": bridge.ledger.snapshot()["reserved_usd"],
                },
            }

        @web_app.get("/v1/models")
        async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            bridge._authorize(authorization)
            return {
                "object": "list",
                "data": [{"id": bridge.identity["model"], "object": "model", "owned_by": "pavlov"}],
            }

        @web_app.post("/v1/chat/completions")
        async def chat_completions(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> Any:
            bridge._authorize(authorization)
            try:
                request = ChatCompletionRequest.model_validate(payload)
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if request.tool_choice not in (None, "auto"):
                raise HTTPException(
                    status_code=400, detail="only automatic tool choice is supported"
                )
            if request.max_tokens is not None and request.max_completion_tokens is not None:
                raise HTTPException(status_code=400, detail="conflicting token limits")
            # LiteLLM retains the provider prefix in the OpenAI-compatible
            # request body even when api_base targets this private bridge.
            # Accept only the two pinned identities, with or without that
            # routing prefix; every other model still fails closed.
            if request.model not in {
                bridge.identity["model"],
                "openai/" + bridge.identity["model"],
            }:
                raise HTTPException(status_code=404, detail="unknown model")
            if request.n != 1:
                raise HTTPException(
                    status_code=400, detail="only n=1 pass@1 requests are supported"
                )
            max_tokens = request.max_completion_tokens or request.max_tokens or 4096
            if max_tokens > MAX_COMPLETION_TOKENS:
                raise HTTPException(
                    status_code=400,
                    detail=f"max completion tokens exceeds {MAX_COMPLETION_TOKENS}",
                )
            out = await bridge._complete(
                messages=request.messages,
                tools=request.tools,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                seed=request.seed,
                enable_thinking=request.enable_thinking,
            )
            content = out["content"]
            tool_calls = out["tool_calls"]
            prompt_count = out["prompt_tokens"]
            completion_count = out["completion_tokens"]

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if request.stream:
                return StreamingResponse(
                    iter(
                        openai_chat_stream_events(
                            completion_id=completion_id,
                            created=created,
                            model=bridge.identity["model"],
                            content=content,
                            tool_calls=tool_calls,
                            prompt_tokens=prompt_count,
                            completion_tokens=completion_count,
                            finish_reason=out["finish_reason"],
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
                "model": bridge.identity["model"],
                "resolved_identity": bridge.identity,
                "prompt_sha256": out.get("prompt_sha256"),
                "request_identity_sha256": out.get("request_identity_sha256"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": out["finish_reason"],
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_count,
                    "completion_tokens": completion_count,
                    "total_tokens": prompt_count + completion_count,
                },
            }

        @web_app.post("/v1/responses")
        async def create_response(
            payload: dict[str, Any] = Body(...),
            authorization: str | None = Header(default=None),
        ) -> Any:
            bridge._authorize(authorization)
            supported = {
                "model",
                "input",
                "instructions",
                "tools",
                "tool_choice",
                "max_output_tokens",
                "temperature",
                "top_p",
                "stream",
            }
            if set(payload) - supported:
                raise HTTPException(status_code=400, detail="unsupported Responses control")
            model = payload.get("model", bridge.identity["model"])
            if not isinstance(model, str) or model not in {
                bridge.identity["model"],
                "openai/" + bridge.identity["model"],
            }:
                raise HTTPException(status_code=404, detail="unknown model")
            if payload.get("background"):
                raise HTTPException(
                    status_code=400,
                    detail="background responses are not supported",
                )
            if payload.get("previous_response_id"):
                raise HTTPException(
                    status_code=400,
                    detail="previous_response_id is not supported: "
                    "this bridge keeps no server-side conversation state",
                )
            max_tokens = payload.get("max_output_tokens", 4096)
            try:
                messages = responses_input_to_chat_messages(payload.get("input"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            instructions = payload.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise HTTPException(status_code=400, detail="instructions must be text")
            if instructions:
                messages = [{"role": "system", "content": str(instructions)}] + messages
            try:
                request = ChatCompletionRequest.model_validate(
                    {
                        "model": model,
                        "messages": messages,
                        "tools": responses_tools_to_chat_tools(payload.get("tools")),
                        "tool_choice": payload.get("tool_choice"),
                        "max_tokens": max_tokens,
                        "temperature": payload.get("temperature", 0.2),
                        "top_p": payload.get("top_p", 0.95),
                        "stream": payload.get("stream", False),
                    }
                )
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if request.max_tokens > MAX_COMPLETION_TOKENS:
                raise HTTPException(status_code=400, detail="max completion tokens exceeded")
            if request.tool_choice not in (None, "auto"):
                raise HTTPException(
                    status_code=400, detail="only automatic tool choice is supported"
                )
            out = await bridge._complete(
                messages=request.messages,
                tools=request.tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=None,
                seed=None,
                enable_thinking=True,
            )
            response = build_responses_object(
                response_id=f"resp-{uuid.uuid4().hex}",
                model=bridge.identity["model"],
                content=out["content"],
                tool_calls=out["tool_calls"],
                prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"],
                created_at=int(time.time()),
            )
            if out["finish_reason"] == "length":
                response["status"] = "incomplete"
                response["incomplete_details"] = {"reason": "max_output_tokens"}
            response["resolved_identity"] = bridge.identity
            response["prompt_sha256"] = out.get("prompt_sha256")
            response["request_identity_sha256"] = out.get("request_identity_sha256")
            if request.stream:
                return StreamingResponse(
                    iter_responses_sse_events(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return response

        return web_app
