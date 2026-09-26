#!/usr/bin/env python3
"""E12 paired vLLM arm. Generation: identical system/user messages and max_tokens=16384 as run_e12.py, sent
through the local shim (E3/code/tinker_shim.py, SHIM_BACKEND=vllm, SHIM_MAX_TOKENS=16384) -> same chat template,
same token ids, temperature 0. Judge: unchanged judge v2 (evidence-first prompt, Tinker Qwen3.6-35B-A3B base, T=0).
Usage: run_e12_paired.py <arm: vllm_trained|vllm_base> <shim port>"""
import csv, json, sys, time, urllib.request

import run_e12 as R
import judge_v2 as J

arm, port = sys.argv[1], int(sys.argv[2])
RAW = R.HERE.parent / arm / "raw"
RAW.mkdir(parents=True, exist_ok=True)
R.LOG = RAW / "judge_calls.jsonl"  # Tinker judge token ledger for this arm


def gen_via_shim(messages):
    body = {"model": "shim", "messages": messages, "max_tokens": R.GEN_MAX, "temperature": 0}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open(R.CSV, newline="")))
    per = []
    for r in rows:
        n = r["#"].strip(); items = R.rubric_items(r["Rubric"])
        gp = RAW / f"gen_{n}.txt"
        if not gp.exists():
            gp.write_text(gen_via_shim([{"role": "system", "content": R.GEN_SYS},
                                        {"role": "user", "content": r["Prompt"].strip() + "\n\n" + r["Addition for CLI Tools"].strip()}]))
        jp = RAW / f"judge_v2_{n}.json"
        if not jp.exists():
            rub = "\n".join(f"{i}. {t}" for i, t in items)
            txt = R.chat([{"role": "system", "content": J.JUDGE_SYS},
                          {"role": "user", "content": f"# Task given to the builder\n{r['Prompt'].strip()}\n\n"
                                                      f"# Rubric items\n{rub}\n\n# Submitted code\n{gp.read_text()}\n\n"
                                                      f"Now grade all {len(items)} rubric items, one line each."}],
                         R.JUDGE_MAX, f"judge_v2_{n}")
            jp.write_text(json.dumps({"raw": txt, "rubric": dict(items)}, indent=1))
        v = J.parse(json.loads(jp.read_text())["raw"], len(items))
        per.append({"task": n, "app": r["App Name"], "n_items": len(items), "verdicts": v,
                    "passed": sum(x is True for x in v.values()), "unparsed_as_fail": sum(x is None for x in v.values())})
        print(arm, n, per[-1]["passed"], len(items), flush=True)
    tot = sum(p["n_items"] for p in per); ps = sum(p["passed"] for p in per)
    json.dump({"arm": arm, "judge": "v2", "per_task": per, "passed": ps, "total": tot, "pass_rate": ps / tot,
               "judge_tinker_tokens": R.used(), "wall_s_this_invocation": round(time.time() - t0, 1)},
              open(RAW / "scores_v2.json", "w"), indent=1)
    print(arm, ps, tot, ps / tot)


if __name__ == "__main__":
    main()
