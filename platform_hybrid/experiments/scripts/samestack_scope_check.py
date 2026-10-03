"""Completion-length scope check for the same-stack GSM8K-CoT rerun.

Recomputes, from the per-step logs in samestack_gsm8k_cot_full.json, the mean
training completion length per arm (all steps and last ten) against the
200-token cap. Thesis §6.5.5 cites the output.

Usage (repo root):
  python platform_hybrid/experiments/scripts/samestack_scope_check.py
"""
import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
SRC = RES / "samestack_gsm8k_cot_full.json"
OUT = RES / "samestack_scope_check.json"


def main():
    d = json.loads(SRC.read_text())
    cap = d["config"]["max_new"]
    arms = {}
    for arm in sorted({r["arm"] for r in d["runs"]}):
        rs = [r for r in d["runs"] if r["arm"] == arm]
        all_len = [st.mean(s["mean_comp_len"] for s in r["step_log"]) for r in rs]
        last10 = [st.mean(s["mean_comp_len"] for s in r["step_log"][-10:]) for r in rs]
        arms[arm] = {"n_seeds": len(rs),
                     "mean_train_comp_len": round(st.mean(all_len), 2),
                     "mean_train_comp_len_last10": round(st.mean(last10), 2),
                     "max_seed_comp_len_last10": round(max(last10), 2),
                     "fraction_of_cap": round(st.mean(all_len) / cap, 3)}
    out = {"source": str(SRC.relative_to(RES.parents[2])), "max_new": cap, "arms": arms,
           "note": "Mean generated tokens per completion (mask sum). Per-completion "
                   "truncation was not logged in this run; see samestack_gsm8k_cot_v2 for it."}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
