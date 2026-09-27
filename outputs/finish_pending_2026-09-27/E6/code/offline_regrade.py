"""E6: grade judge_pending records once judge credit is available. Uses the native evaluator_router on the
recorded final answer (StringEvaluator) and final URL (URLEvaluator via PseudoPage). Only string_match/url_match
tasks can be pending (program_html never calls the LLM judge). Run from webarena root with venv + env.
Usage: python offline_regrade.py results.jsonl [...]  (rewrites each file in place, keeps a .bak)"""
import json, os, shutil, sys
sys.path.insert(0, os.getcwd())
sys.argv, files = sys.argv[:1], sys.argv[1:]
import e6_driver  # noqa: F401  (patches the judge to gpt-4.1)
from e6_driver import JudgeUnavailable, USAGE
from browser_env import create_stop_action
from evaluation_harness import evaluator_router
from evaluation_harness.helper_functions import PseudoPage

for fn in files:
    recs = [json.loads(l) for l in open(fn) if l.strip()]
    for r in recs:
        if not r.get("judge_pending"):
            continue
        cfg = f"config_files/{r['task_id']}.json"
        traj = [{}, create_stop_action(r.get("stop") or "")]
        c0 = USAGE["judge_cost"]
        try:
            score = 1.0  # EvaluatorComb semantics (product), called per-evaluator: Comb's hints demand a live CDPSession
            for ev in evaluator_router(cfg).evaluators:
                score *= ev(trajectory=traj, config_file=cfg, page=PseudoPage(None, r.get("final_url", "")))
            r["score"] = float(score)
        except JudgeUnavailable as e:
            print("still no judge credit:", e); break
        r["judge_pending"] = False
        r["regraded_offline"] = True
        r.setdefault("usage", {})["judge_cost"] = r["usage"].get("judge_cost", 0) + USAGE["judge_cost"] - c0
        print(r["task_id"], r["score"])
    shutil.copy(fn, fn + ".bak")
    with open(fn, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
