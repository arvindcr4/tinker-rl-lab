#!/usr/bin/env python3
"""P6 submission §5.2 (matched TRL-vs-Tinker rerun): truncation check from existing artifacts.

Reads platform_hybrid/experiments/results/publication_fixes/matched_trl_tinker/{eval_heldout.json,matched_trl_tinker_result.json}
Writes platform_hybrid/experiments/results/review_2026-10-18/P6_submission/matched_truncation.json
"""
import json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "platform_hybrid/experiments/results/publication_fixes/matched_trl_tinker"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P6_submission"
ev = json.loads((SRC / "eval_heldout.json").read_text())
res = json.loads((SRC / "matched_trl_tinker_result.json").read_text(), parse_constant=lambda c: None)
rows = {}
for k, v in ev.items():
    if k.startswith("_"): continue
    toks = v["n_tokens"]
    rows[k] = dict(acc_regex=v["acc_campaign_reward"], acc_boxed=v["acc_boxed_only"], trunc_frac=v["trunc_frac"],
                   mean_tokens=statistics.fmean(toks), share_ge_512=sum(t >= 512 for t in toks) / len(toks),
                   gain_vs_base=v["acc_campaign_reward"] - ev["base"]["acc_campaign_reward"])
summary = dict(decoding=ev["_env"]["decoding"], items=ev["_env"]["items"], per_adapter=rows,
               all_trained_below_base=all(r["gain_vs_base"] < 0 for k, r in rows.items() if k != "base"),
               train_trunc_by_seed=dict(trl=res["train"]["trl"]["trunc_by_seed"], tinker=res["train"]["tinker"]["trunc_by_seed"]),
               last10_gap=res["train"]["last10_gap_tinker_minus_trl"], heldout_gap=res["heldout"]["gap_tinker_minus_trl"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "matched_truncation.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
