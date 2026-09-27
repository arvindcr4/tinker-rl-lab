"""E6: grade judge_pending records with the native evaluator_router on the recorded final answer
(StringEvaluator) and final URL (URLEvaluator via PseudoPage). Pending records are string_match(+url_match) only.

Two passes per record:
  1. judge-free bound: the LLM judge calls (llm_fuzzy_match / llm_ua_match) are stubbed to 1.0 (optimistic).
     EvaluatorComb/StringEvaluator scores are products of per-check scores in {0,1}, so if the optimistic
     product is 0 the true native score is 0 regardless of the judge -> resolved (resolved_without_judge).
     If no judge call was needed at all, the stubbed score IS the native score -> resolved.
  2. otherwise call the real judge (openai/gpt-4.1 via OpenRouter); on JudgeUnavailable the record stays pending.
Run from webarena root with venv + env and PYTHONPATH=<dir of e6_driver.py>.
Usage: python offline_regrade.py results.jsonl [...]  (rewrites each file in place, keeps a .bak)"""
import json, os, shutil, sys
sys.path.insert(0, os.getcwd())
sys.argv, files = sys.argv[:1], sys.argv[1:]
import e6_driver  # noqa: F401  (patches the judge to gpt-4.1)
from e6_driver import JudgeUnavailable, USAGE
from browser_env import create_stop_action
from evaluation_harness import evaluator_router
import evaluation_harness.evaluators as E
from evaluation_harness.helper_functions import PseudoPage

REAL = (E.llm_fuzzy_match, E.llm_ua_match)
CALLS = [0]


def _stub(*a, **k):
    CALLS[0] += 1
    return 1.0


def native_score(r, cfg):
    traj = [{}, create_stop_action(r.get("stop") or "")]
    score = 1.0  # EvaluatorComb semantics (product), called per-evaluator: Comb's hints demand a live CDPSession
    for ev in evaluator_router(cfg).evaluators:
        score *= ev(trajectory=traj, config_file=cfg, page=PseudoPage(None, r.get("final_url", "")))
    return float(score)


judge_ok = True
for fn in files:
    recs = [json.loads(l) for l in open(fn) if l.strip()]
    for r in recs:
        if not r.get("judge_pending"):
            continue
        cfg = f"config_files/{r['task_id']}.json"
        E.llm_fuzzy_match = E.llm_ua_match = _stub
        CALLS[0] = 0
        s_opt = native_score(r, cfg)
        E.llm_fuzzy_match, E.llm_ua_match = REAL
        if s_opt == 0.0 or CALLS[0] == 0:
            r.update(score=s_opt, judge_pending=False, regraded_offline=True, resolved_without_judge=True)
            print(r["task_id"], s_opt, "resolved_without_judge")
            continue
        if not judge_ok:
            continue
        c0 = USAGE["judge_cost"]
        try:
            r["score"] = native_score(r, cfg)
        except JudgeUnavailable as e:
            print("still no judge credit:", str(e)[:120]); judge_ok = False
            continue
        r.update(judge_pending=False, regraded_offline=True)
        r.setdefault("usage", {})["judge_cost"] = r["usage"].get("judge_cost", 0) + USAGE["judge_cost"] - c0
        print(r["task_id"], r["score"], "judged")
    shutil.copy(fn, fn + ".bak")
    with open(fn, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
