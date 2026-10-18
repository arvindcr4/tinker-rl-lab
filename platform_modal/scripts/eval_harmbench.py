"""
HarmBench evaluation entrypoint, protocol v2.

Official benchmark mode is unavailable until the pinned grader is integrated.
The opt-in keyword mode measures keyword presence only, with no safety claim.
"""

import modal
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from _harmbench_protocol import (
    load_behaviors,
    keyword_diagnostic,
    require_revision,
    PROTOCOL_VERSION,
)

app = modal.App("tinkerrl-harmbench")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.46.3",
    "datasets==3.1.0",
    "accelerate==1.1.1",
    "vllm==0.6.4.post1",
)

RESULTS_DIR = "/results/harmbench"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/results": results_vol},
)
def run_harmbench_eval(
    model_name: str,
    seed: int = 42,
    model_revision: str = "",
    tokenizer_revision: str = "",
    dataset_revision: str = "",
    mode: str = "benchmark",
    image_identity: str = "",
) -> dict:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    import time

    print(f"Starting HarmBench evaluation for {model_name} (seed={seed})")

    if mode != "refusal_keyword_diagnostic":
        raise RuntimeError(
            "Official HarmBench grading requires a pinned upstream grader integration; benchmark evaluation aborted"
        )
    for revision, name in [(model_revision, "model"), (tokenizer_revision, "tokenizer")]:
        require_revision(revision, name)
    if not image_identity:
        raise ValueError("record the immutable execution image identity")
    dataset, selection = load_behaviors(load_dataset, dataset_revision)
    prompts = [
        (item.get("ContextString", "") + "\n\n" + item["Behavior"]).strip() for item in dataset
    ]

    # 2. Generate completions
    t0 = time.time()
    llm = LLM(
        model=model_name,
        revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        tensor_parallel_size=1,
        trust_remote_code=False,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=150, seed=seed)

    tokenizer = llm.get_tokenizer()

    formatted_prompts = []
    for p in prompts:
        if (
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None) is not None
        ):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        else:
            formatted_prompts.append(f"User: {p}\nAssistant:")

    outputs = llm.generate(formatted_prompts, sampling_params)
    completions = [out.outputs[0].text for out in outputs]

    generation_time = time.time() - t0

    result_summary, results_list = keyword_diagnostic(dataset, completions)
    import importlib.metadata

    result_summary.update(
        {
            "model": model_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "dataset_revision": dataset_revision,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "decoding": {"temperature": 0.0, "max_tokens": 150, "seed": seed},
            "image_identity": image_identity,
            "selection": selection,
            "package_versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "datasets", "vllm")
            },
            "total_prompts": len(prompts),
            "generation_time_s": generation_time,
            "grader_revision": None,
            "grader_status": "official_grading_unavailable",
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = f"{RESULTS_DIR}/{model_name.replace('/', '_')}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump({"summary": result_summary, "details": results_list}, f, indent=2)

    results_vol.commit()
    print(f"Keyword diagnostic complete; no HarmBench safety score. Results saved to {out_file}")

    return result_summary


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_revision: str = "",
    tokenizer_revision: str = "",
    dataset_revision: str = "",
    mode: str = "benchmark",
    image_identity: str = "",
):
    """
    Local entry point. Run with:
    modal run platform_modal/scripts/eval_harmbench.py --model-name "Qwen/Qwen2.5-0.5B-Instruct"
    """
    res = run_harmbench_eval.remote(
        model_name,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_revision=dataset_revision,
        mode=mode,
        image_identity=image_identity,
    )
    print("Final Result:", res)
