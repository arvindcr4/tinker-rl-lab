"""Score E14 with the native Omni-MATH get_result.parse_report ('Equivalence Judgement' == 'TRUE').
Native get_result skips unparseable reports; per PROTOCOL they count as failures in the denominator here."""
import importlib.util, json, os
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
RAW = REPO / "outputs/e1_e14_small_scale_2026-09-26/E14" / (f"vllm_{os.environ['ARM']}/raw" if os.environ.get("ARM") else "raw")
spec = importlib.util.spec_from_file_location(
    "get_result", REPO / "outputs/public_portfolio_2026-09-05/omni_setup/native_source/Omni-Judge_eval/get_result.py")
gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)

actor = {r["row_index"]: r for r in map(json.loads, (RAW / "actor_generations.jsonl").read_text().splitlines())}
judged = json.loads((RAW / "omnijudge_raw.json").read_text())
rows = []
for j in judged["results"]:
    info = gr.parse_report(j["omni_judge"])
    verdict = info.get("Equivalence Judgement")
    a = actor[j["row_index"]]
    rows.append({"row_index": j["row_index"], "difficulty": a["difficulty"], "truncated": a.get("stop_reason") == "length",
                 "judge_parsed": verdict is not None, "judgement": verdict, "correct": verdict == "TRUE",
                 "student_final_answer": info.get("Student Final Answer"), "reference_answer": a["answer"]})
(RAW / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
s = {"n": len(rows), "correct": sum(r["correct"] for r in rows), "unparsed": sum(not r["judge_parsed"] for r in rows),
     "correct_among_truncated": sum(r["correct"] for r in rows if r["truncated"]),
     "truncated": sum(r["truncated"] for r in rows), "judge": f'{judged["judge_repo"]}@{judged["judge_revision"]}',
     "judge_prompt_tokens": sum(j["judge_prompt_tokens"] for j in judged["results"])}
(RAW / "score_summary.json").write_text(json.dumps(s, indent=1))
print(s)
