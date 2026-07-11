import time
import random
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, Request, Depends

from tinker.types import ModelInput, SamplingParams
from tinker_atropos.types import (
    GenerateRequest,
    GenerateResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)

def get_trainer(request: Request):
    return request.app.state.trainer

def create_app(trainer=None) -> FastAPI:
    app = FastAPI(title="Tinker-Atropos Service")
    app.state.trainer = trainer

    @app.get("/health")
    async def health(trainer_instance=Depends(get_trainer)):
        return {
            "status": "ok",
            "trainer_initialized": trainer_instance is not None,
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest, trainer_instance=Depends(get_trainer)):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

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

                sampling_params = SamplingParams(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop if request.stop else [],
                )

                result = await trainer_instance.current_sampling_client.sample_async(
                    prompt=model_input,
                    sampling_params=sampling_params,
                    num_samples=request.n,
                )

                for sequence in result.sequences:
                    output_text = trainer_instance.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                    all_choices.append(
                        {
                            "text": output_text,
                            "index": choice_index,
                            "finish_reason": "stop",
                        }
                    )
                    choice_index += 1

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4()}",
                choices=all_choices,
                created=int(time.time()),
                model=trainer_instance.base_model,
            )
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
    async def chat_completions(request: ChatCompletionRequest, trainer_instance=Depends(get_trainer)):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

        try:
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            prompt_text = trainer_instance.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = trainer_instance.tokenizer.encode(prompt_text, add_special_tokens=False)
            model_input = ModelInput.from_ints(prompt_tokens)

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop if request.stop else [],
            )

            result = await trainer_instance.current_sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=request.n,
            )

            choices = []
            for i, sequence in enumerate(result.sequences):
                output_text = trainer_instance.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                choices.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "index": i,
                        "finish_reason": "stop",
                    }
                )

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                choices=choices,
                created=int(time.time()),
                model=trainer_instance.base_model,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

    @app.post("/generate", response_model=GenerateResponse | List[GenerateResponse])
    async def generate(request: GenerateRequest, trainer_instance=Depends(get_trainer)):
        if trainer_instance is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

        try:
            if request.input_ids is None:
                raise HTTPException(status_code=400, detail="input_ids is required")

            prompt_tokens = request.input_ids
            sampling_params = request.sampling_params or {}
            n = sampling_params.get("n", 1)
            max_tokens = sampling_params.get("max_new_tokens", 256)
            temperature = sampling_params.get("temperature", 1.0)
            stop = sampling_params.get("stop", [])

            model_input = ModelInput.from_ints(prompt_tokens)
            tinker_sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop if isinstance(stop, list) else [stop],
            )

            result = await trainer_instance.current_sampling_client.sample_async(
                prompt=model_input,
                sampling_params=tinker_sampling_params,
                num_samples=n,
            )

            if n == 1:
                sequence = result.sequences[0]
                output_tokens = sequence.tokens
                output_logprobs = sequence.logprobs if sequence.logprobs else []
                output_text = trainer_instance.tokenizer.decode(output_tokens, skip_special_tokens=True)

                output_token_logprobs = []
                for token_id, logprob in zip(output_tokens, output_logprobs):
                    token_text = trainer_instance.tokenizer.decode([token_id])
                    output_token_logprobs.append((logprob, token_id, token_text))

                return GenerateResponse(
                    text=output_text,
                    meta_info={
                        "prompt_tokens": len(prompt_tokens),
                        "completion_tokens": len(output_tokens),
                        "finish_reason": "stop",
                        "output_token_logprobs": output_token_logprobs,
                    },
                )
            else:
                results = []
                for sequence in result.sequences:
                    output_tokens = sequence.tokens
                    output_logprobs = sequence.logprobs if sequence.logprobs else []
                    output_text = trainer_instance.tokenizer.decode(output_tokens, skip_special_tokens=True)

                    output_token_logprobs = []
                    for token_id, logprob in zip(output_tokens, output_logprobs):
                        token_text = trainer_instance.tokenizer.decode([token_id])
                        output_token_logprobs.append((logprob, token_id, token_text))

                    results.append(
                        GenerateResponse(
                            text=output_text,
                            meta_info={
                                "prompt_tokens": len(prompt_tokens),
                                "completion_tokens": len(output_tokens),
                                "finish_reason": "stop",
                                "output_token_logprobs": output_token_logprobs,
                            },
                        )
                    )
                return results

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generate failed: {str(e)}")

    return app
