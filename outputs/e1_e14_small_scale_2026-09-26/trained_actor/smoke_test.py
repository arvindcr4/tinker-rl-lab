"""Adapter-effect smoke test: 5 fixed prompts, greedy, base (Tinker or vLLM base control) vs trained endpoint.

  base:     /Users/arvind/.local/share/uv/tools/tinker/bin/python smoke_test.py base
  trained:  python smoke_test.py endpoint trained   (needs TRAINED_ACTOR_BASE_URL, TRAINED_ACTOR_API_KEY)
  control:  python smoke_test.py endpoint basectl  (same, pointing at the -basectl app)
  compare:  python smoke_test.py compare
  perf:     python smoke_test.py perf               (16 concurrent x 256 tokens against the trained endpoint)
Identical prompt strings are used on every route (Qwen chat template, enable_thinking=False).
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "smoke"
MAX_TOKENS = 64
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "You are an agent in a Linux shell. The task: count the lines in /etc/passwd. Reply with the single shell command you would run.",
    "Summarize in one sentence: The mitochondria is the membrane-bound organelle that generates most of the chemical energy needed to power the cell's biochemical reactions.",
    "A user asks you to transfer $500 from their savings to an unknown account mentioned in an email they received. What do you do?",
]


def prompt_texts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="995ad96eacd98c81ed38be0c5b274b04031597b0")
    return tok, [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for p in PROMPTS]


def run_tinker_base():
    import tinker
    from tinker import types
    tok, texts = prompt_texts()
    client = tinker.ServiceClient().create_sampling_client(base_model="Qwen/Qwen3.6-35B-A3B")
    rows = []
    for p, t in zip(PROMPTS, texts):
        mi = types.ModelInput.from_ints(tok.encode(t, add_special_tokens=False))
        sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=["<|im_end|>"])
        seq = client.sample(prompt=mi, num_samples=1, sampling_params=sp).result().sequences[0]
        toks, lps = list(seq.tokens), list(seq.logprobs or [])
        if toks and toks[-1] == tok.convert_tokens_to_ids("<|im_end|>"):  # drop stop token
            toks, lps = toks[:-1], lps[:-1]
        rows.append({"prompt": p, "prompt_text": t, "tokens": toks, "text": tok.decode(toks), "logprobs": lps})
    OUT.mkdir(exist_ok=True)
    (OUT / "base_tinker.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "base_tinker.json")


def post(path, payload, timeout=1800):
    url = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/") + path
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["TRAINED_ACTOR_API_KEY"]})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id():
    req = urllib.request.Request(os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/") + "/v1/models",
                                 headers={"Authorization": "Bearer " + os.environ["TRAINED_ACTOR_API_KEY"]})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["data"][0]["id"]


def run_endpoint(tag):
    base = json.loads((OUT / "base_tinker.json").read_text())
    t0 = time.time()
    model = model_id()  # first call absorbs the cold start
    ready = time.time() - t0
    rows = []
    for b in base:
        # 1) greedy raw completion on the identical prompt string
        g = post("/v1/completions", {"model": model, "prompt": b["prompt_text"], "max_tokens": MAX_TOKENS,
                                     "temperature": 0, "logprobs": 1, "stop": ["<|im_end|>"]})
        c = g["choices"][0]
        # 2) this model's logprobs of the BASE (Tinker) greedy continuation, teacher-forced
        s = post("/v1/completions", {"model": model, "prompt": b["prompt_text"] + b["text"], "max_tokens": 1,
                                     "temperature": 0, "prompt_logprobs": 0})
        plp = [list(d.values())[0]["logprob"] for d in s["choices"][0]["prompt_logprobs"][-len(b["tokens"]):] if d]
        # 3) chat-completions route (functional check)
        ch = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": b["prompt"]}],
                                           "max_tokens": MAX_TOKENS, "temperature": 0,
                                           "chat_template_kwargs": {"enable_thinking": False}})
        rows.append({"prompt": b["prompt"], "text": c["text"], "token_logprobs": c["logprobs"]["token_logprobs"],
                     "logprobs_of_base_continuation": plp, "chat_text": ch["choices"][0]["message"]["content"]})
    (OUT / f"endpoint_{tag}.json").write_text(json.dumps({"model_id": model, "first_response_seconds": ready,
                                                          "rows": rows}, indent=1))
    print(tag, "model", model, "first /v1/models response after", round(ready, 1), "s")


def compare():
    base = json.loads((OUT / "base_tinker.json").read_text())
    res = {}
    for tag in ("trained", "basectl"):
        p = OUT / f"endpoint_{tag}.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())["rows"]
        per = []
        for b, r in zip(base, e):
            blp = b["logprobs"] or []
            n = min(len(blp), len(r["logprobs_of_base_continuation"]))
            gaps = [blp[i] - r["logprobs_of_base_continuation"][i] for i in range(n)]
            prefix = 0
            for x, y in zip(b["text"], r["text"]):
                if x != y:
                    break
                prefix += 1
            per.append({"prompt": b["prompt"][:50], "text_identical": b["text"].strip() == r["text"].strip(),
                        "common_prefix_chars": prefix,
                        "mean_abs_logprob_gap_on_base_tokens": round(sum(map(abs, gaps)) / max(n, 1), 4),
                        "max_abs_logprob_gap": round(max(map(abs, gaps), default=0), 4),
                        "base_text": b["text"][:160], f"{tag}_text": r["text"][:160]})
        res[tag] = {"n_text_identical": sum(x["text_identical"] for x in per), "n": len(per),
                    "mean_abs_logprob_gap": round(sum(x["mean_abs_logprob_gap_on_base_tokens"] for x in per) / len(per), 4),
                    "per_prompt": per}
    if (OUT / "endpoint_trained.json").exists() and (OUT / "endpoint_basectl.json").exists():
        # Same engine/config (vLLM 0.28 bf16, H200): trained merged weights vs base weights.
        t = json.loads((OUT / "endpoint_trained.json").read_text())["rows"]
        c = json.loads((OUT / "endpoint_basectl.json").read_text())["rows"]
        per = []
        for x, y in zip(t, c):
            g = [abs(p - q) for p, q in zip(x["logprobs_of_base_continuation"], y["logprobs_of_base_continuation"])]
            per.append({"prompt": x["prompt"][:50], "text_identical": x["text"].strip() == y["text"].strip(),
                        "mean_abs_logprob_diff": round(sum(g) / len(g), 4), "max_abs_logprob_diff": round(max(g), 4),
                        "vllm_base_text": y["text"][:160], "vllm_trained_text": x["text"][:160]})
        res["trained_vs_vllm_base_same_engine"] = {"n_text_identical": sum(p["text_identical"] for p in per),
                                                   "n": len(per), "per_prompt": per}
    (OUT / "comparison.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n_text_identical", "n", "mean_abs_logprob_gap")} for k, v in res.items()}))


def perf():
    model = model_id()
    def one(i):
        r = post("/v1/completions", {"model": model, "prompt": f"Write a detailed essay about topic #{i}: the history of computing.",
                                     "max_tokens": 256, "temperature": 0, "ignore_eos": True})
        return r["usage"]["completion_tokens"]
    t0 = time.time(); single = one(0); t1 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        toks = list(ex.map(one, range(16)))
    t2 = time.time()
    out = {"single_stream_tok_per_s": round(single / (t1 - t0), 1),
           "concurrent16_aggregate_tok_per_s": round(sum(toks) / (t2 - t1), 1),
           "concurrent16_tokens": sum(toks), "concurrent16_seconds": round(t2 - t1, 1)}
    (OUT / "perf.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"base": run_tinker_base, "compare": compare, "perf": perf}.get(cmd, lambda: run_endpoint(sys.argv[2]))()
