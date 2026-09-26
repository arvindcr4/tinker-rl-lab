"""E2 small-scale substitute: EffiBench (huangd1999/EffiBench @ d29e43bc), base Qwen3.6-35B-A3B.

Stages:
  select : any python    -> raw/selection.json
  sample : tinker python -> raw/generations.jsonl
  build  : any python    -> raw/programs/{canonical,completion}/<idx>.py  (upstream add_string_to_py_file logic)
  exec   : docker (python:3.11-slim, --network none) runs code/timer.py -> raw/exec_results.json
  score  : any python    -> raw/score.json

Prompt = upstream prompts/prompt.txt + markdown_description + small_test_cases (upstream
open_source_model_completion.py format). Correctness = upstream harness: completion + large `test_case`
asserts run to exit 0 with empty stderr within 5 s. Efficiency = wall-clock NET (completion / canonical
median of 3 runs), re-implemented with time.perf_counter instead of mprof.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
RAW = LANE / "raw"
CODE = LANE / "code"
SEED = 20260926
N = 30
MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_TOKENS = 2048

ListNode_text = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""
TreeNode_text = """
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
import_pkg = """
from typing import *
from bisect import *
from collections import *
from copy import *
from datetime import *
from heapq import *
from math import *
from re import *
from string import *
from random import *
from itertools import *
from functools import *
from operator import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import itertools
import functools
import operator
"""


def data():
    return json.loads((RAW / "effibench_dataset_with_difficulty_and_algorithm.json").read_text())


def select():
    d = data()
    idx = sorted(random.Random(SEED).sample(range(len(d)), N))
    (RAW / "selection.json").write_text(json.dumps({
        "seed": SEED, "method": "sorted(random.Random(seed).sample(range(1000), 30)) over upstream data/dataset_with_difficulty_and_algorithm.json order",
        "positions": idx, "problem_idx": [d[i]["problem_idx"] for i in idx]}, indent=1))


def sample():
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    d = data()
    sel = json.loads((RAW / "selection.json").read_text())["positions"]
    header = (CODE / "upstream_prompt.txt").read_text()
    tok = AutoTokenizer.from_pretrained(MODEL)
    sc = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
    futs = []
    for i in sel:
        e = d[i]
        prompt = (f"{header}\n# Task description:\n```python\n{e['markdown_description']}\n```\n"
                  f"# Test case:\n```python\n{e['small_test_cases']}\n```")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        futs.append((e["problem_idx"], len(ids), sc.sample(
            T.ModelInput.from_ints(ids), num_samples=1,
            sampling_params=T.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))))
    with (RAW / "generations.jsonl").open("w") as fh:
        for pidx, n_prompt, fu in futs:
            try:
                toks = list(fu.result().sequences[0].tokens)
                rec = {"problem_idx": pidx, "completion": tok.decode(toks, skip_special_tokens=True),
                       "prompt_tokens": n_prompt, "response_tokens": len(toks), "error": None}
            except Exception as ex:
                rec = {"problem_idx": pidx, "completion": "", "prompt_tokens": n_prompt, "response_tokens": 0,
                       "error": repr(ex)[:1000]}
            fh.write(json.dumps(rec) + "\n")
            print(pidx, rec["response_tokens"], rec["error"])


def _program(completion: str, test_case: str) -> str | None:
    # upstream add_string_to_py_file
    if "class Solution" not in completion:
        return None
    if "```python" in completion:
        completion = completion[completion.find("```python") + 9:]
        if "```" in completion:
            completion = completion[:completion.find("```")]
    return import_pkg + "\n" + TreeNode_text + "\n" + ListNode_text + "\n" + completion + "\nsolution=Solution()\n" + test_case


def build():
    d = {e["problem_idx"]: e for e in data()}
    gens = [json.loads(l) for l in (RAW / "generations.jsonl").read_text().splitlines()]
    for kind in ("canonical", "completion"):
        (RAW / "programs" / kind).mkdir(parents=True, exist_ok=True)
    manifest = {}
    for g in gens:
        e = d[g["problem_idx"]]
        (RAW / "programs/canonical" / f"{g['problem_idx']}.py").write_text(_program(e["canonical_solution"], e["test_case"]))
        p = _program(g["completion"], e["test_case"])
        manifest[g["problem_idx"]] = p is not None
        if p is not None:
            (RAW / "programs/completion" / f"{g['problem_idx']}.py").write_text(p)
    (RAW / "programs/extractable.json").write_text(json.dumps(manifest, indent=1))


def execute():
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "2g",
           "-v", f"{RAW / 'programs'}:/programs:ro", "-v", f"{CODE / 'timer.py'}:/timer.py:ro",
           "python:3.11-slim", "python", "/timer.py"]
    print(" ".join(cmd))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    (RAW / "exec_results.json").write_text(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def score():
    gens = {json.loads(l)["problem_idx"]: json.loads(l) for l in (RAW / "generations.jsonl").read_text().splitlines()}
    ex = json.loads((RAW / "exec_results.json").read_text())
    per, nets = {}, []
    for pidx in gens:
        k = str(pidx)
        can = ex["canonical"].get(k, {})
        com = ex["completion"].get(k)
        passed = bool(com and com["passed"])
        rec = {"passed": passed, "canonical_passed": bool(can.get("passed")),
               "extractable": com is not None, "gen_error": gens[pidx]["error"]}
        if passed and can.get("passed"):
            rec["net"] = com["median_s"] / can["median_s"]
            nets.append(rec["net"])
        per[k] = rec
    n = len(per)
    k = sum(r["passed"] for r in per.values())
    out = {"numerator": k, "denominator": n, "value": k / n, "wilson95": wilson(k, n),
           "net_mean_over_passed": statistics.mean(nets) if nets else None,
           "net_geomean_over_passed": math.exp(statistics.mean(math.log(x) for x in nets)) if nets else None,
           "n_faster_than_canonical": sum(x < 1 for x in nets),
           "canonical_pass": sum(r["canonical_passed"] for r in per.values()),
           "tinker_prompt_tokens": sum(g["prompt_tokens"] for g in gens.values()),
           "tinker_sample_tokens": sum(g["response_tokens"] for g in gens.values()),
           "n_errors": sum(bool(g["error"]) for g in gens.values()), "per_problem": per}
    (RAW / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "per_problem"}, indent=1))


if __name__ == "__main__":
    {"select": select, "sample": sample, "build": build, "exec": execute, "score": score}[sys.argv[1]]()
