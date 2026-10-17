import time
import asyncio
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, Request, Depends

from tinker.types import ModelInput
from tinker_atropos.types import (
    GenerateRequest,
    GenerateResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    LogprobsRequest,
    LogprobsResponse,
    TokenLogprob,
)

from tinker_atropos.http_contract import (
    sampling_params,
    finish_reason,
    probabilities,
    PROTOCOL_VERSION,
)


def get_trainer(request: Request):
    trainer = request.app.state.trainer
    if (
        trainer is None
        or getattr(trainer, "tokenizer", None) is None
        or getattr(trainer, "current_sampling_client", None) is None
    ):
        raise HTTPException(503, "Tokenizer and sampler are not ready")
    return trainer


def create_app(trainer=None) -> FastAPI:
    app = FastAPI(title="Tinker-Atropos Service")
    app.state.trainer = trainer

    app.state.scoring_limit = asyncio.Semaphore(8)

    @app.get("/health")
    async def health():
        instance = app.state.trainer
        ready = (
            instance is not None
            and getattr(instance, "tokenizer", None) is not None
            and getattr(instance, "current_sampling_client", None) is not None
        )
        return {
            "status": "ok" if ready else "starting",
            "trainer_initialized": ready,
            "ready": ready,
            "protocol_version": PROTOCOL_VERSION,
        }

    @app.get("/training_state")
    async def training_state(trainer_instance=Depends(get_trainer)):
        return {
            "optimizer_steps": trainer_instance.optimizer_steps,
            "policy_version": trainer_instance.policy_version,
            "run_status": trainer_instance.run_status,
        }

    @app.post("/logprobs", response_model=LogprobsResponse)
    async def logprobs(request: LogprobsRequest, trainer_instance=Depends(get_trainer)):
        if (request.input_ids is None) == (request.text is None):
            raise HTTPException(400, "Supply exactly one of input_ids and text")
        tokens = (
            request.input_ids
            if request.input_ids is not None
            else trainer_instance.tokenizer.encode(request.text, add_special_tokens=False)
        )
        if not tokens:
            raise HTTPException(400, "At least one token is required")
        sampler = trainer_instance.current_sampling_client
        policy_version = getattr(trainer_instance, "policy_version", None)
        policy_version = policy_version if isinstance(policy_version, str) else None
        try:
            async with app.state.scoring_limit:
                values = await sampler.compute_logprobs_async(ModelInput.from_ints(tokens))
        except Exception as exc:
            raise HTTPException(502, "Teacher scoring failed") from exc
        values = probabilities(tokens, values, allow_unavailable=True)
        return LogprobsResponse(
            policy_version=policy_version,
            num_tokens=len(tokens),
            logprobs=[
                TokenLogprob(
                    token_id=tid,
                    logprob=lp,
                    token=trainer_instance.tokenizer.decode([tid]) if request.return_text else None,
                )
                for tid, lp in zip(tokens, values)
            ],
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest, trainer_instance=Depends(get_trainer)):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

        sampler = trainer_instance.current_sampling_client
        try:
            if isinstance(request.prompt, str):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            all_choices = []
            choice_index = 0

            for prompt in prompts:
                prompt_tokens = trainer_instance.tokenizer.encode(prompt, add_special_tokens=False)
                model_input = ModelInput.from_ints(prompt_tokens)

                params = sampling_params(
                    {
                        k: getattr(request, k)
                        for k in ("max_tokens", "temperature", "stop", "seed", "top_p", "top_k")
                    }
                )

                result = await sampler.sample_async(
                    prompt=model_input,
                    sampling_params=params,
                    num_samples=request.n,
                )

                for sequence in result.sequences:
                    output_text = trainer_instance.tokenizer.decode(
                        sequence.tokens, skip_special_tokens=True
                    )
                    all_choices.append(
                        {
                            "text": output_text,
                            "index": choice_index,
                            "finish_reason": finish_reason(sequence),
                        }
                    )
                    choice_index += 1

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4()}",
                choices=all_choices,
                created=int(time.time()),
                model=trainer_instance.base_model,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")

    @app.get("/wandb_info")
    async def wandb_info(trainer_instance=Depends(get_trainer)):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")
        return {
            "group": trainer_instance.wandb_group,
            "project": trainer_instance.config.wandb_project,
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest, trainer_instance=Depends(get_trainer)
    ):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

        sampler = trainer_instance.current_sampling_client
        try:
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            prompt_text = trainer_instance.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = trainer_instance.tokenizer.encode(prompt_text, add_special_tokens=False)
            model_input = ModelInput.from_ints(prompt_tokens)

            params = sampling_params(
                {
                    k: getattr(request, k)
                    for k in ("max_tokens", "temperature", "stop", "seed", "top_p", "top_k")
                }
            )

            result = await sampler.sample_async(
                prompt=model_input,
                sampling_params=params,
                num_samples=request.n,
            )

            choices = []
            for i, sequence in enumerate(result.sequences):
                output_text = trainer_instance.tokenizer.decode(
                    sequence.tokens, skip_special_tokens=True
                )
                choices.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "index": i,
                        "finish_reason": finish_reason(sequence),
                    }
                )

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                choices=choices,
                created=int(time.time()),
                model=trainer_instance.base_model,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

    @app.post("/generate", response_model=GenerateResponse | List[GenerateResponse])
    async def generate(request: GenerateRequest, trainer_instance=Depends(get_trainer)):
        if request.stream or request.top_logprobs_num or request.logprob_start_len is not None:
            raise HTTPException(
                400, "Streaming, top logprobs and prompt logprob offsets are unsupported"
            )
        if (request.input_ids is None) == (request.text is None):
            raise HTTPException(400, "Supply exactly one of input_ids and text")
        if request.input_ids is not None:
            prompts = request.input_ids
            if not prompts or isinstance(prompts[0], int):
                prompts = [prompts]
        else:
            texts = [request.text] if isinstance(request.text, str) else request.text
            prompts = [
                trainer_instance.tokenizer.encode(text, add_special_tokens=False) for text in texts
            ]
        settings = request.sampling_params or {}
        n = settings.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise HTTPException(400, "n must be a positive integer")
        params = sampling_params(settings)
        sampler = trainer_instance.current_sampling_client
        policy = getattr(trainer_instance, "policy_version", "unreported")
        output = []
        for tokens in prompts:
            if not tokens:
                raise HTTPException(400, "At least one token is required")
            result = await sampler.sample_async(
                prompt=ModelInput.from_ints(tokens), sampling_params=params, num_samples=n
            )
            if len(result.sequences) != n:
                raise HTTPException(502, "Backend returned an unexpected sequence count")
            for sequence in result.sequences:
                pairs = []
                if request.return_logprob:
                    values = probabilities(sequence.tokens, sequence.logprobs)
                    pairs = [
                        (
                            lp,
                            tid,
                            trainer_instance.tokenizer.decode([tid])
                            if request.return_text_in_logprobs
                            else "",
                        )
                        for tid, lp in zip(sequence.tokens, values)
                    ]
                output.append(
                    GenerateResponse(
                        text=trainer_instance.tokenizer.decode(
                            sequence.tokens, skip_special_tokens=True
                        ),
                        meta_info={
                            "prompt_tokens": len(tokens),
                            "completion_tokens": len(sequence.tokens),
                            "finish_reason": finish_reason(sequence),
                            "output_token_logprobs": pairs,
                            "policy_version": policy,
                            "protocol_version": PROTOCOL_VERSION,
                        },
                    )
                )
        return output[0] if len(output) == 1 else output

    return app
