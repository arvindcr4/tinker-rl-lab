"""
Standalone tinker inference server. No training, no atropos — just chat.

Usage:
    python serve.py                                    # default model
    python serve.py --model Qwen/Qwen3-30B-A3B        # specific model
    python serve.py --weights tinker://...path...      # from saved weights
    python serve.py --port 8001                        # custom port

LIMITATION: limitations from the adversarial review:
- The Closed-Source Confound: Tinker is a closed-source "black box". We need to document or open-source Tinker's managed defaults, micro-partitioning, and reference offloading to ensure fair algorithmic comparisons.
- The ZVF Metric, Early-Training Snapshots, Failure to Prove Generalization, and Single-Seed Extrapolations limitations apply to training and are not directly applicable to this inference server.
"""

import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass


import argparse
import asyncio
import logging
import time
import uuid

import tinker
from tinker.types import ModelInput, SamplingParams
from fastapi import FastAPI, HTTPException, Request, Depends
from transformers import AutoTokenizer

from tinker_atropos.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    LogprobsRequest,
    LogprobsResponse,
    TokenLogprob,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tinker.serve")

app = FastAPI(title="Tinker Inference Server")


def get_app_state(request: Request):
    return request.app.state


@app.get("/health")
async def health(state=Depends(get_app_state)):
    is_ready = hasattr(state, "sampling_client") and state.sampling_client is not None
    return {"status": "ok", "model": getattr(state, "model_name", None), "ready": is_ready}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, state=Depends(get_app_state)):
    if not hasattr(state, "sampling_client") or state.sampling_client is None:
        logger.error("Chat completion failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        prompt_text = await asyncio.to_thread(
            state.tokenizer.apply_chat_template,
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_tokens = await asyncio.to_thread(
            state.tokenizer.encode,
            prompt_text,
            add_special_tokens=False
        )
        model_input = ModelInput.from_ints(prompt_tokens)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop if request.stop else [],
        )

        result = await state.sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=request.n,
        )

        choices = []
        for i, sequence in enumerate(result.sequences):
            output_text = await asyncio.to_thread(
                state.tokenizer.decode,
                sequence.tokens,
                skip_special_tokens=True
            )
            choices.append(
                {
                    "message": {"role": "assistant", "content": output_text},
                    "index": i,
                    "finish_reason": "stop",
                }
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            choices=choices,
            created=int(time.time()),
            model=state.model_name,
        )
    except ValueError as ve:
        logger.error(f"Value error during chat completion: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        logger.error(f"Internal error during chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, state=Depends(get_app_state)):
    if not hasattr(state, "sampling_client") or state.sampling_client is None:
        logger.error("Completion failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        all_choices = []

        for prompt in prompts:
            prompt_tokens = await asyncio.to_thread(
                state.tokenizer.encode,
                prompt,
                add_special_tokens=False
            )
            model_input = ModelInput.from_ints(prompt_tokens)

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop if request.stop else [],
            )

            result = await state.sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=request.n,
            )

            for sequence in result.sequences:
                output_text = await asyncio.to_thread(
                    state.tokenizer.decode,
                    sequence.tokens,
                    skip_special_tokens=True
                )
                all_choices.append(
                    {
                        "text": output_text,
                        "index": len(all_choices),
                        "finish_reason": "stop",
                    }
                )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            choices=all_choices,
            created=int(time.time()),
            model=state.model_name,
        )
    except ValueError as ve:
        logger.error(f"Value error during completion: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        logger.error(f"Internal error during completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/logprobs", response_model=LogprobsResponse)
async def logprobs(request: LogprobsRequest, state=Depends(get_app_state)):
    if not hasattr(state, "sampling_client") or state.sampling_client is None:
        logger.error("Logprobs failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if request.input_ids is not None:
            token_ids = request.input_ids
        elif request.text is not None:
            token_ids = await asyncio.to_thread(
                state.tokenizer.encode,
                request.text,
                add_special_tokens=False
            )
        else:
            logger.warning("Logprobs request missing input_ids and text")
            raise HTTPException(status_code=400, detail="input_ids or text required")

        if len(token_ids) == 0:
            logger.warning("Logprobs request with empty input")
            raise HTTPException(status_code=400, detail="Empty input")

        model_input = ModelInput.from_ints(token_ids)
        prompt_lps = await state.sampling_client.compute_logprobs_async(model_input)

        token_logprobs = []
        for i, token_id in enumerate(token_ids):
            lp = prompt_lps[i] if i < len(prompt_lps) and prompt_lps[i] is not None else 0.0
            if request.return_text:
                token_text = await asyncio.to_thread(state.tokenizer.decode, [token_id])
            else:
                token_text = None
            token_logprobs.append(TokenLogprob(token_id=token_id, logprob=lp, token=token_text))

        return LogprobsResponse(logprobs=token_logprobs, num_tokens=len(token_ids))
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Value error during logprobs: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        logger.error(f"Internal error during logprobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def main():
    parser = argparse.ArgumentParser(description="Tinker inference server")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help="Model name")
    parser.add_argument("--weights", default=None, help="Saved weights path (tinker://...)")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    app.state.model_name = args.model
    logger.info(f"Loading model: {args.model}")

    app.state.tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Tokenizer loaded")

    service_client = tinker.ServiceClient()

    if args.weights:
        logger.info(f"Loading weights from: {args.weights}")
        app.state.sampling_client = service_client.create_sampling_client(model_path=args.weights)
    else:
        logger.info("Using base model weights")
        app.state.sampling_client = service_client.create_sampling_client(base_model=args.model)

    logger.info("Sampling client ready")

    import uvicorn

    logger.info(f"Serving on http://0.0.0.0:{args.port}")
    logger.info("  /v1/chat/completions  - OpenAI chat")
    logger.info("  /v1/completions       - OpenAI completions")
    logger.info("  /logprobs             - per-token logprobs")
    logger.info("  /health               - health check")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

