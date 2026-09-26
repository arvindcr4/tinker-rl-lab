"""Write result.json for E8/E11 (and E10/E14 when their raw files exist) purely from raw/ files."""
import json, math, sys
from pathlib import Path

BASE = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26")
ACTOR = "Qwen/Qwen3.6-35B-A3B base (no adapter), Tinker"
PY = "/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad/venv/bin/python (tinker==0.30.4, transformers, pillow)"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def write(lane, **kw):
    k, n = kw["numerator"], kw["denominator"]
    r = {"lane": lane, "actor": ACTOR, "thinking": False, "temperature": 0, **kw,
         "value": round(k / n, 4), "wilson95": wilson(k, n), "raw_dir": f"{lane}/raw/"}
    order = ["lane", "original_benchmark", "benchmark_run", "scope", "substitute_gap", "actor", "thinking", "temperature",
             "max_tokens", "n_selected", "n_scored", "n_errors", "item_ids", "metric", "value", "numerator", "denominator",
             "wilson95", "grader", "compute_route", "cost", "started_utc", "finished_utc", "commands", "raw_dir", "caveats"]
    r = {key: r[key] for key in order if key in r} | {key: v for key, v in r.items() if key not in order}
    (BASE / lane / "result.json").write_text(json.dumps(r, indent=1) + "\n")
    print(lane, k, n, r["value"], r["wilson95"])


def e8():
    s = json.loads((BASE / "E8/raw/summary.json").read_text())
    g = [json.loads(l) for l in (BASE / "E8/raw/graded.jsonl").read_text().splitlines()]
    k = sum(x["correct"] for x in g)
    write("E8", original_benchmark="LifeSciBench (private)", benchmark_run="LAB-Bench public MCQ (Future-House/LAB-Bench @998a8e0), 10 per category x 8",
          scope="substitute", substitute_gap="LifeSciBench is private; LAB-Bench public is a different, public biology MCQ suite with no private/held-out items.",
          max_tokens=4096, n_selected=len(g), n_scored=len(g), n_errors=s["errors"], item_ids=s["item_ids"],
          metric="accuracy (native LAB-Bench Evaluator.compute_metrics; unsure/unanswerable = incorrect)",
          numerator=k, denominator=len(g), grader="native",
          compute_route="Tinker sampling (text+image chunks) from local Mac; native LAB-Bench prompt/parser/scorer executed locally",
          cost={"tinker_tokens": s["tokens"]["prefill"] + s["tokens"]["sample"], "colab_units": 0.0, "modal_usd": 0.0},
          started_utc="2026-09-26T14:12:56Z", finished_utc=s["finished_utc"],
          commands=["set -a; source .env; set +a", f"cd outputs/e1_e14_small_scale_2026-09-26/E8/code && {PY.split(' ')[0]} run_e8.py"],
          native_metrics=s["native_metrics"], per_category_correct_of_10=s["per_category_correct_of_10"],
          caveats=["New base-model arm; not comparable to or poolable with the lost-adapter 1967/1967 run (22.88%).",
                   "Selection: random.Random(20260926).sample of 10 per category from the prepared manifest's sorted task order; choice order uses the original prepared manifest (native shuffle, seed 809).",
                   "Non-thinking template with native 'Think step by step' CoT prompt; 22/80 responses hit the 4096-token cap (still parsed natively).",
                   "Two FigQA images exceeded Tinker's 2 MiB asset limit (HTTP 400, no model output); they were re-sent after JPEG re-encoding (same pixels, q90/q80). First-attempt errors kept in raw/transport_errors_first_attempt/.",
                   "n=80 is small: the Wilson 95% interval is wide; per-category n=10."])


def e11():
    s = json.loads((BASE / "E11/raw/summary.json").read_text())
    items = sorted(s["per_item"])
    write("E11", original_benchmark="VerilogEval (NVlabs verilog-eval @c498220d, spec-to-rtl + code-complete-iccad2023)",
          benchmark_run="VerilogEval, 25 problems per framing (50 total)", scope="original-public-subset", substitute_gap=None,
          max_tokens=4096, n_selected=50, n_scored=s["n"], n_errors=s["errors"], item_ids=items,
          metric="pass@1 (1 sample/problem, native sv-iv-test testbench: 'Mismatches: 0 in N samples')",
          numerator=s["passes"], denominator=s["n"], grader="native",
          compute_route="Tinker sampling; local native harness (configure + gmake sv-iv-test) with repo-pinned iverilog-12 toolchain",
          cost={"tinker_tokens": s["prefill"] + s["sample"], "colab_units": 0.0, "modal_usd": 0.0},
          started_utc="2026-09-26T14:14:19Z", finished_utc=s["finished_utc"],
          commands=["set -a; source .env; set +a", f"cd outputs/e1_e14_small_scale_2026-09-26/E11/code && {PY.split(' ')[0]} run_e11.py"],
          pass_at_1_both_denominators=s["score"], extraction_failures=s["extraction_failures"],
          caveats=["New base-model arm; not comparable to the retained adapter receipt 129/312 = 41.35% (that run used thinking-default template, temperature 0.2, all 312).",
                   "This run: non-thinking, temperature 0, raw prompt file as the single user message (same as retained driver).",
                   "Selection: random.Random(20260926).sample(25) per framing over the sorted *_prompt.txt list; Prob099 (known unscoreable) was not selected.",
                   "Verdicts come from per-problem native test-bench logs (driver's direct_results); upstream sv-iv-analyze/summary.csv failed locally (missing langchain), which does not affect simulation.",
                   "First harness attempt scored 0/50 because `python` was not on PATH (make aborted before simulation); rerun reused the identical saved samples. Log kept in raw/."])


def e14():
    a = json.loads((BASE / "E14/raw/actor_summary.json").read_text())
    s = json.loads((BASE / "E14/raw/score_summary.json").read_text())
    g = [json.loads(l) for l in (BASE / "E14/raw/graded.jsonl").read_text().splitlines()]
    fin = [x for x in g if not x["truncated"]]
    write("E14", original_benchmark="FrontierMath (blocked; only public samples)",
          benchmark_run="Omni-MATH test (KbsdJames/Omni-MATH @40ba231d), 100 rows", scope="substitute",
          substitute_gap="FrontierMath is gated; Omni-MATH is public olympiad-level competition math, not unpublished research problems, and has no code-execution grading.",
          max_tokens=2048, n_selected=100, n_scored=s["n"], n_errors=a["errors"],
          item_ids=[f"row{i:05d}" for i in a["row_indices"]],
          metric="accuracy (native Omni-Judge 'Equivalence Judgement' == TRUE; unparsed judge output = incorrect)",
          numerator=s["correct"], denominator=s["n"], grader=f"llm-judge({s['judge']}, native prompt/parser, vLLM 0.6.3 on Modal A10G, temp 0, max 300)",
          compute_route="Actor: Tinker. Judge: Modal A10G (app e14-omnijudge-small, stopped on completion).",
          cost={"tinker_tokens": a["prefill"] + a["sample"], "colab_units": 0.0, "modal_usd": 0.0807},
          started_utc=a["started_utc"], finished_utc="2026-09-26T14:23:14Z",
          commands=["set -a; source .env; set +a", f"cd outputs/e1_e14_small_scale_2026-09-26/E14 && {PY.split(' ')[0]} code/run_actor.py",
                    "modal run code/modal_omnijudge.py --inp raw/actor_generations.jsonl --out raw/omnijudge_raw.json",
                    "python3 code/score.py"],
          truncated_at_max_tokens=s["truncated"], correct_among_truncated=s["correct_among_truncated"],
          correct_among_finished=f"{sum(x['correct'] for x in fin)}/{len(fin)}",
          caveats=["New base-model arm; the prior 2271/4428 = 51.31% figure is not comparable (different actor, full split).",
                   "Judge is the official open Omni-Judge (not the GPT-4o leaderboard judge); Omni-Judge does not need Tinker.",
                   "Native actor protocol (system prompt + problem, max_tokens 2048) truncates 73/100 answers in non-thinking mode; the lost-adapter actor also hit the 2048 limit on ~87% of the first 402 rows checked.",
                   "Omni-Judge marked 15 truncated answers correct from partial work; 23 of the 27 answers that finished were marked correct. The native score is reported unchanged.",
                   "Selection: random.Random(20260926).sample(range(4428), 100), sorted; items are row indices of pinned test.jsonl.",
                   "Modal cost from `modal billing report --for today` for app ap-XJkxPRXfhaFSWijRF6CNrH."])


def e10():
    import glob
    sel = json.loads((BASE / "E10/raw/selected_ids.json").read_text())
    res = {}
    for k in ("harmful", "benign"):
        d = json.loads(Path(glob.glob(str(BASE / f"E10/raw/inspect_logs/{k}/*.json"))[0]).read_text())
        v = [x["scores"]["combined_scorer"]["value"] for x in d["samples"]]
        res[k] = {"n": len(v), "refusals": int(sum(x["refusal"] for x in v)), "avg_score": round(sum(x["score"] for x in v) / len(v), 4),
                  "full_score_count": sum(1 for x in v if x["score"] == 1.0), "errors": sum(1 for x in d["samples"] if x.get("error")),
                  "log_status": d["status"]}
    shim = [json.loads(l) for l in (BASE / "E10/raw/shim_log.jsonl").read_text().splitlines()]
    toks = sum(x["response"]["usage"]["total_tokens"] for x in shim if "usage" in x["response"]) + 328  # +328: pre-run shim smoke test (failed_attempts/)
    h, b = res["harmful"], res["benign"]
    write("E10", original_benchmark="AgentHarm (private test split, blocked)",
          benchmark_run="AgentHarm test_public (ai-safety-institute/AgentHarm @e23b3fe6), 30 harmful + 30 benign, inspect_evals @b935c0e5",
          scope="original-public-subset", substitute_gap="Public split of the same distribution; the private held-out split is unobtainable.",
          max_tokens=4096, n_selected=60, n_scored=h["n"] + b["n"], n_errors=h["errors"] + b["errors"],
          item_ids={"harmful": sel["harmful"], "benign": sel["benign"]},
          metric="harmful refusal rate (native AgentHarm refusal judge); see harm_score / benign_score for the other native metrics",
          numerator=h["refusals"], denominator=h["n"],
          grader="llm-judge(Qwen/Qwen3.6-35B-A3B base via same Tinker shim, replacing default gpt-4o-2024-08-06 refusal+semantic judges); native grading functions and scorer",
          compute_route="Tinker via local OpenAI-compatible chat.completions shim (127.0.0.1, stopped); inspect_ai 0.3.269 locally",
          cost={"tinker_tokens": toks, "colab_units": 0.0, "modal_usd": 0.0},
          started_utc=(BASE / "E10/raw/started_utc.txt").read_text().strip(), finished_utc=(BASE / "E10/raw/finished_utc.txt").read_text().strip(),
          commands=["set -a; source .env; set +a", "cd outputs/e1_e14_small_scale_2026-09-26/E10/code && <venv>/bin/python shim.py 18765 ../raw/shim_log.jsonl &",
                    "OPENAI_BASE_URL=http://127.0.0.1:18765/v1 OPENAI_API_KEY=local-shim <ve10>/bin/python run_e10.py"],
          harm_score={"value": h["avg_score"], "note": "mean native AgentHarm score over 30 harmful behaviours (lower is safer)", "full_score_count": h["full_score_count"]},
          benign_score={"value": b["avg_score"], "note": "mean native score over 30 benign behaviours", "full_score_count": b["full_score_count"],
                        "benign_refusals": f'{b["refusals"]}/{b["n"]}'},
          per_split=res,
          caveats=["New base-model arm on AgentHarm itself; the lost-adapter lane figure (AgentDojo benign 97/97, 90.72%) is a different benchmark and not comparable.",
                   "Judges are the actor model itself (Qwen3.6 base), not GPT-4o: judge self-bias possible; refusal and semantic judgments are not the leaderboard setup.",
                   "Tool calls use the Qwen3.6 chat template's XML tool format, parsed back to OpenAI tool_calls by a local shim (code/shim.py); parser errors would show as no tool call.",
                   "Selection: random.Random(20260926).sample(30) per split over test_public behaviour ids (file order); default settings (no hint/detailed filter, 0 irrelevant tools, message_limit 20).",
                   "Two failed launches produced no scored samples (Responses API path; 'developer' role unsupported by template); logs in raw/failed_attempts/."])


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
