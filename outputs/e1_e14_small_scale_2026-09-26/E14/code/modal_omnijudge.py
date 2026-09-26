"""Native Omni-Judge (KbsdJames/Omni-Judge @ pinned rev) on Modal A10G via vLLM.
Mirrors Omni-Judge_eval/omni_judge_vllm.py: tokenizer.get_context, temperature 0, max_tokens 300,
stop=[eos, <|eot_id|>], response prefixed with '## Student Final Answer'."""
import json
import modal

JUDGE_REPO = "KbsdJames/Omni-Judge"
JUDGE_REVISION = "de5bdca15ff3c366b90718c4b4be555d25c655b0"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm==0.6.3.post1", "transformers==4.45.2", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
app = modal.App("e14-omnijudge-small", image=image)


@app.function(gpu="A10G", timeout=2400)
def judge(rows_json: str) -> str:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = json.loads(rows_json)
    path = snapshot_download(JUDGE_REPO, revision=JUDGE_REVISION)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    llm = LLM(model=path, trust_remote_code=True, enable_prefix_caching=True, dtype="float16",
              max_model_len=8192, gpu_memory_utilization=0.92)
    ctx = [tok.get_context(r["problem"], r["answer"], r["model_generation"]) for r in rows]
    sp = SamplingParams(n=1, stop=[tok.eos_token, "<|eot_id|>"], max_tokens=300, temperature=0)
    outs = llm.generate(ctx, sampling_params=sp)
    res = []
    for r, c, o in zip(rows, ctx, outs):
        res.append({"row_index": r["row_index"], "judge_prompt_tokens": len(o.prompt_token_ids),
                    "omni_judge": "## Student Final Answer\n" + o.outputs[0].text.strip()})
    return json.dumps({"judge_repo": JUDGE_REPO, "judge_revision": JUDGE_REVISION, "results": res})


@app.local_entrypoint()
def main(inp: str, out: str):
    rows = [json.loads(l) for l in open(inp) if l.strip()]
    open(out, "w").write(judge.remote(json.dumps(rows)))
