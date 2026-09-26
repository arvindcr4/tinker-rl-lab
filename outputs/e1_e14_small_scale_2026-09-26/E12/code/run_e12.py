#!/usr/bin/env python3
"""E12 small-scale: AppBench public tasks (6 tasks / 151 rubric items), one-shot code generation.

Actor: Qwen/Qwen3.6-35B-A3B base (no adapter) on Tinker, non-thinking, temperature 0.
Judge: the same base model on Tinker (self-judge), non-thinking, temperature 0, static review of
the generated source against each rubric item. No build, no deploy, no browser testing.
Outputs: raw/gen_<n>.txt, raw/judge_<n>.json, raw/calls.jsonl (token accounting), raw/scores.json.
"""
import csv, json, pathlib, re, sys, time

import tinker
import tinker.types as T
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
CSV = HERE.parents[2] / "e12_appbench/hf_dataset/AppBench vExternal.csv"
GEN_MAX, JUDGE_MAX = 16384, 6144
TOKEN_CAP = 800_000  # E12 share of the 6M lane-group Tinker cap

tok = AutoTokenizer.from_pretrained(MODEL)
sampler = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
LOG = RAW / "calls.jsonl"


def used():
    return sum(r["prompt_tokens"] + r["completion_tokens"]
               for r in map(json.loads, LOG.read_text().splitlines())) if LOG.exists() else 0


def chat(messages, max_tokens, tag):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    if used() + len(ids) + max_tokens > TOKEN_CAP:
        raise RuntimeError("E12 token cap would be exceeded")
    t0 = time.time()
    res = sampler.sample(T.ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=T.SamplingParams(max_tokens=max_tokens, temperature=0.0)).result()
    out = list(res.sequences[0].tokens)
    with LOG.open("a") as f:
        f.write(json.dumps({"tag": tag, "t": t0, "dt": round(time.time() - t0, 1), "prompt_tokens": len(ids),
                            "completion_tokens": len(out), "hit_max": len(out) >= max_tokens}) + "\n")
    return tok.decode(out, skip_special_tokens=True)


GEN_SYS = ("You are an expert full-stack engineer. You have no shell access in this session: deliver the "
           "complete implementation in one response. Output every file needed on top of a fresh Next.js "
           "(App Router, TypeScript) template, each as:\n=== FILE: <relative/path> ===\n```<lang>\n<content>\n```\n"
           "Include the Supabase SQL schema as a file. Read API keys from environment variables. "
           "Do not ask questions; do not leave TODOs or placeholders.")

JUDGE_SYS = ("You are a strict senior full-stack reviewer grading an AppBench submission by static source review. "
             "For EACH numbered rubric item decide PASS only if the submitted code clearly and concretely implements "
             "the requirement end-to-end (UI + logic + persistence/API wiring where implied), such that it would work "
             "once deployed with the needed API keys. Stubs, mock data standing in for real data, TODOs, missing "
             "files, or code that only mentions the feature are FAIL. Missing or truncated code is FAIL.\n"
             "Answer with ONLY a JSON object mapping each rubric item number (string) to "
             "{\"pass\": true|false, \"reason\": \"<=20 words\"}.")


def rubric_items(rub):
    items = re.findall(r"^\s*(\d+)\.\s+(.*\S)", rub, flags=re.M)
    return [(int(n), t) for n, t in items]


def parse_judge(txt, n_items):
    m = re.search(r"\{.*\}", txt, flags=re.S)
    verdict = {}
    if m:
        try:
            obj = json.loads(m.group(0))
            for k, v in obj.items():
                if not str(k).isdigit():
                    continue
                if isinstance(v, dict):
                    v = v.get("pass")
                if isinstance(v, str):
                    v = {"PASS": True, "FAIL": False}.get(v.strip().upper())
                if isinstance(v, bool):
                    verdict[int(k)] = v
        except json.JSONDecodeError:
            pass
    return {i: verdict.get(i, None) for i in range(1, n_items + 1)}


def main():
    RAW.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(CSV, newline="")))
    scores = []
    for r in rows:
        n = r["#"].strip()
        items = rubric_items(r["Rubric"])
        gen_p = RAW / f"gen_{n}.txt"
        if not gen_p.exists():
            gen = chat([{"role": "system", "content": GEN_SYS},
                        {"role": "user", "content": r["Prompt"].strip() + "\n\n" + r["Addition for CLI Tools"].strip()}],
                       GEN_MAX, f"gen_{n}")
            gen_p.write_text(gen)
        gen = gen_p.read_text()
        judge_p = RAW / f"judge_{n}.json"
        if not judge_p.exists():
            rub = "\n".join(f"{i}. {t}" for i, t in items)
            jtxt = chat([{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": f"# Task given to the builder\n{r['Prompt'].strip()}\n\n"
                                                     f"# Rubric items\n{rub}\n\n# Submitted code\n{gen}"}],
                        JUDGE_MAX, f"judge_{n}")
            v = parse_judge(jtxt, len(items))
            judge_p.write_text(json.dumps({"raw": jtxt, "verdicts": v, "rubric": dict(items)}, indent=1))
        v = parse_judge(json.loads(judge_p.read_text())["raw"], len(items))  # recomputed from raw text
        passed = sum(1 for x in v.values() if x is True)
        missing = sum(1 for x in v.values() if x is None)
        scores.append({"task": n, "app": r["App Name"], "n_items": len(items), "passed": passed,
                       "unparsed_as_fail": missing, "gen_chars": len(gen),
                       "n_files": gen.count("=== FILE:")})
        print(scores[-1], flush=True)
    tot = sum(s["n_items"] for s in scores); p = sum(s["passed"] for s in scores)
    json.dump({"per_task": scores, "passed": p, "total": tot, "pass_rate": p / tot,
               "tinker_tokens": used()}, open(RAW / "scores.json", "w"), indent=1)
    print(p, tot, p / tot)


if __name__ == "__main__":
    main()
