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


def is_supported_request_model(model: str) -> bool:
    """Accept only the pinned model identities and LiteLLM routing forms."""

    return model in {
        MODEL_ALIAS,
        MODEL_ID,
        f"openai/{MODEL_ALIAS}",
        f"openai/{MODEL_ID}",
    }


bridge_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi[standard]==0.116.1",
        "huggingface-hub==1.27.0",
        "tinker==0.30.0",
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
                "sampler_path_configured": SAMPLER_PATH,
                "hf_repo": HF_REPO,
                "hf_revision": HF_REVISION,
                "hf_commit": HF_COMMIT,
                "maximum_usd": str(self.maximum_usd),
            },
            reinit=True,
        )
        if self.wandb_run is None or not getattr(self.wandb_run, "id", None):
            raise RuntimeError("W&B online initialization failed before Tinker setup")
        # Sampler weights: explicit SAMPLER_PATH env wins (fine-tuned route);
        # empty/unset means BASE-MODEL serving via a zero-initialized LoRA
        # snapshot (== base model sampling; no training steps, no weight
        # reuse). The resolved path is recorded in W&B config below so the
        # served weights are always auditable. Never silently fall back: a
        # configured-but-missing path raises instead of serving base.
        sampler_path = os.environ.get("SAMPLER_PATH", "").strip()
        svc = tinker.ServiceClient(
            user_metadata={
                "campaign": "pavlov-e1-e14-modal",
                "component": "openai-compatible-bridge",
                "wandb_run_id": self.wandb_run.id,
            }
        )
        if sampler_path:
            self.sampler_path_resolved = sampler_path
        else:
            trainer = svc.create_lora_training_client(
                base_model=MODEL_ID, rank=4)
            initial = trainer.save_weights_for_sampler(name="pool0").result()
            self.sampler_path_resolved = initial.path
        self.sampling_client = svc.create_sampling_client(
            model_path=self.sampler_path_resolved)
        self.wandb_run.config.update(
            {"sampler_path_resolved": self.sampler_path_resolved,
             "serving_mode": ("fine-tuned" if sampler_path
                              else "base-zero-init")},
            allow_val_change=True,
        )

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

    async def _complete(self, *, messages: list[dict[str, Any]],
                        tools: list[dict[str, Any]] | None,
                        max_tokens: int, temperature: float, top_p: float,
                        stop: list[str] | str | None, seed: int | None,
                        enable_thinking: bool) -> dict[str, Any]:
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
            # Agent frameworks normally compact their own conversation, but
            # a failed compaction can still submit an overlong prompt. Keep
            # the system/tool prefix and the newest working context so the
            # request remains usable while failing closed on the model's
            # fixed context limit.
            prefix_tokens = min(CONTEXT_PREFIX_TOKENS, prompt_budget // 2)
            suffix_tokens = prompt_budget - prefix_tokens
            truncated_prompt_tokens = len(prompt_tokens) - prompt_budget
            prompt_tokens = prompt_tokens[:prefix_tokens] + prompt_tokens[-suffix_tokens:]
        projected_usd = Decimal(str(estimate_tinker_usd(len(prompt_tokens), max_tokens)))
        self._reserve(projected_usd)
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
            actual_usd = Decimal(
                str(estimate_tinker_usd(len(prompt_tokens), len(output_tokens)))
            )
            self._settle(projected_usd, actual_usd)
            self.wandb_run.log(
                {
                    "bridge/prompt_tokens": len(prompt_tokens),
                    "bridge/original_prompt_tokens": original_prompt_tokens,
                    "bridge/truncated_prompt_tokens": truncated_prompt_tokens,
                    "bridge/completion_tokens": len(output_tokens),
                    "bridge/actual_usd": float(actual_usd),
                    "bridge/tool_call_count": len(tool_calls),
                    "bridge/latency_seconds": time.monotonic() - started,
                }
            )
        except Exception as exc:
            self._charge_failed_reservation(projected_usd)
            self.wandb_run.log(
                {
                    "bridge/request_failed": 1,
                    "bridge/conservative_charged_usd": float(projected_usd),
                }
            )
            raise HTTPException(status_code=500, detail="Tinker sampling failed") from exc
        return {"content": content, "tool_calls": tool_calls,
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(output_tokens),
                "original_prompt_tokens": original_prompt_tokens,
                "truncated_prompt_tokens": truncated_prompt_tokens}

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
            enable_thinking: bool = True

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
            # LiteLLM retains the provider prefix in the OpenAI-compatible
            # request body even when api_base targets this private bridge.
            # Accept only the two pinned identities, with or without that
            # routing prefix; every other model still fails closed.
            if not is_supported_request_model(request.model):
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
                            model=MODEL_ALIAS,
                            content=content,
                            tool_calls=tool_calls,
                            prompt_tokens=prompt_count,
                            completion_tokens=completion_count,
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
            model = payload.get("model", MODEL_ALIAS)
            if not is_supported_request_model(model):
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
            max_tokens = (payload.get("max_output_tokens")
                          or payload.get("max_completion_tokens") or 4096)
            if max_tokens > MAX_COMPLETION_TOKENS:
                raise HTTPException(
                    status_code=400,
                    detail=f"max completion tokens exceeds {MAX_COMPLETION_TOKENS}",
                )
            try:
                messages = responses_input_to_chat_messages(payload.get("input"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            instructions = payload.get("instructions")
            if instructions:
                messages = [{"role": "system", "content": str(instructions)}
                            ] + messages
            out = await bridge._complete(
                messages=messages,
                tools=responses_tools_to_chat_tools(payload.get("tools")),
                max_tokens=max_tokens,
                temperature=payload.get("temperature", 0.2),
                top_p=payload.get("top_p", 0.95),
                stop=None,
                seed=None,
                enable_thinking=True,
            )
            response = build_responses_object(
                response_id=f"resp-{uuid.uuid4().hex}",
                model=MODEL_ALIAS,
                content=out["content"],
                tool_calls=out["tool_calls"],
                prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"],
                created_at=int(time.time()),
            )
            if payload.get("stream"):
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
