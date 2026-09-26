#!/usr/bin/env python3
"""E12 judge v2 (evidence-first). v1 (run_e12.py) ignored the requested per-item reason and returned bare
PASS/FAIL, with two tasks all-FAIL; v2 forces one line per item: `<n> | <evidence> | PASS|FAIL`.
Same judge model (Qwen3.6-35B-A3B base on Tinker, non-thinking, T=0), same generated code (raw/gen_<n>.txt)."""
import csv, json, re
import run_e12 as R

JUDGE_SYS = ("You are a strict senior full-stack reviewer grading an AppBench submission by static source review. "
             "For each numbered rubric item, find the concrete code that implements it. PASS only if the code clearly "
             "implements the requirement end-to-end (UI + logic + persistence/API wiring where implied) so it would work "
             "once deployed with the needed API keys. Stubs, mock data standing in for real data, TODOs, missing files, "
             "truncated code, or code that only mentions the feature are FAIL.\n"
             "Output exactly one line per rubric item, in order, and nothing else:\n"
             "<item number> | <evidence: file path and what it does, or what is missing; max 25 words> | PASS or FAIL")


def parse(txt, n_items):
    """Per line `<n>[.):|] ...`: collect standalone PASS/FAIL tokens (judge puts the verdict in field 1, 2 or 3, and
    uses `n |` or `n. VERDICT |` prefixes). Lines without a verdict token are ignored; first verdict line per item
    wins. Consistent tokens -> that verdict; conflicting or none -> None (scored as FAIL)."""
    v = {}
    for line in txt.splitlines():
        m = re.match(r"^\s*\**\s*(\d+)\s*[.):|]\s*(.*)$", line)
        if not m or int(m.group(1)) in v:
            continue
        toks = {t.upper() for t in re.findall(r"\b(PASS|FAIL)\b", m.group(2), flags=re.I)}
        if not toks:
            continue
        v[int(m.group(1))] = (toks == {"PASS"}) if len(toks) == 1 else None
    return {i: v.get(i) for i in range(1, n_items + 1)}


def main():
    rows = list(csv.DictReader(open(R.CSV, newline="")))
    per = []
    for r in rows:
        n = r["#"].strip(); items = R.rubric_items(r["Rubric"])
        p = R.RAW / f"judge_v2_{n}.json"
        if not p.exists():
            gen = (R.RAW / f"gen_{n}.txt").read_text()
            rub = "\n".join(f"{i}. {t}" for i, t in items)
            txt = R.chat([{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": f"# Task given to the builder\n{r['Prompt'].strip()}\n\n"
                                                      f"# Rubric items\n{rub}\n\n# Submitted code\n{gen}\n\n"
                                                      f"Now grade all {len(items)} rubric items, one line each."}],
                         R.JUDGE_MAX, f"judge_v2_{n}")
            p.write_text(json.dumps({"raw": txt, "rubric": dict(items)}, indent=1))
        v = parse(json.loads(p.read_text())["raw"], len(items))
        per.append({"task": n, "app": r["App Name"], "n_items": len(items),
                    "passed": sum(x is True for x in v.values()), "unparsed_as_fail": sum(x is None for x in v.values())})
        print(per[-1], flush=True)
    tot = sum(s["n_items"] for s in per); ps = sum(s["passed"] for s in per)
    json.dump({"judge": "v2", "per_task": per, "passed": ps, "total": tot, "pass_rate": ps / tot,
               "tinker_tokens_all_e12_calls": R.used()}, open(R.RAW / "scores_v2.json", "w"), indent=1)
    print(ps, tot, ps / tot)


if __name__ == "__main__":
    main()
