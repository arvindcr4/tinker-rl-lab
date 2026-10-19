"""LaTeX (booktabs) version of table.py: per (task+cap, algo) row, seeds aggregated as mean \\pm SD.

Numbers reuse table.py definitions exactly: rollout-pooled correct/incorrect mean length over the
first 5 and last 5 steps, Spearman trend of per-step len_correct_mean / len_incorrect_mean vs step,
and mean frac_at_cap over the last 5 steps.

Usage: python latex_table.py raw/arith_cap10.json [raw/cot_cap200.json ...] [--out table.tex]
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

TASK_NAMES = {
    "Qwen2.5-0.5B": "Arithmetic",
    "Qwen2.5-1.5B-Instruct": "GSM8K-CoT",
}


def task_name(model):
    for key, name in TASK_NAMES.items():
        if key in model:
            return name
    return model.split("/")[-1]


def pooled(steps, key):
    L = np.concatenate([np.array(s["lengths"])[np.array(s["rewards"]) == key] for s in steps])
    return float(L.mean()) if L.size else float("nan")


def trend(steps, key):
    xs, ys = [], []
    for s in steps:
        m = s["len_correct_mean" if key == 1 else "len_incorrect_mean"]
        if m is not None:
            xs.append(s["step"])
            ys.append(m)
    return float(spearmanr(xs, ys).statistic) if len(xs) > 2 else float("nan")


def run_metrics(r):
    sl = r["step_log"]
    f5, l5 = sl[:5], sl[-5:]
    c0, c1 = pooled(f5, 1), pooled(l5, 1)
    i0, i1 = pooled(f5, 0), pooled(l5, 0)
    return {
        "c0": c0,
        "c1": c1,
        "i0": i0,
        "i1": i1,
        "g0": i0 - c0,
        "g1": i1 - c1,
        "rho_inc": trend(sl, 0),
        "cap1": float(np.mean([s["frac_at_cap"] for s in l5])),
        "acc": r.get("heldout_post_acc", r.get("heldout_acc")),
        "pre": r.get("heldout_pre_acc"),
    }


def fmt(ms, metric, kind):
    v = np.array([m[metric] for m in ms], dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    if kind == "len":
        s, f = "\\pm", lambda x: f"{x:.1f}"
    elif kind == "rho":
        s, f = "\\pm", lambda x: f"{x:.2f}"
    elif kind == "gap":
        s, f = "\\pm", lambda x: f"{abs(x):+.1f}"
    elif kind == "frac":
        s, f = "\\pm", lambda x: f"{x:.2f}"
    else:
        s, f = "\\pm", lambda x: f"{x:.3f}"
    mean = f(v.mean())
    sd = f(v.std(ddof=1)) if v.size > 1 else ""
    if sd and mean.startswith("+") and sd.startswith("+"):
        sd = sd[1:]
    return f"{mean}${s}${sd}" if sd else mean


def pair(ms, metric_lo, metric_hi, kind):
    return f"{fmt(ms, metric_lo, kind)} $\\to$ {fmt(ms, metric_hi, kind)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()

    groups = {}
    for path in args.paths:
        d = json.load(open(path))
        for r in d["runs"]:
            key = (task_name(d["model"]), d["max_new"], r["algo"])
            groups.setdefault(key, []).append(run_metrics(r))

    lines = [
        "\\small",
        "\\begin{tabular}{lllrcccccc}",
        "\\toprule",
        "Task & cap & Algo & $n$ & len correct & len incorrect & gap (inc$-$cor) & $\\rho$(step, len$_{inc}$) & at-cap & held-out acc \\\\",
        " & & & & first5\\(\\to\\)last5 & first5\\(\\to\\)last5 & first5\\(\\to\\)last5 & & last5 & post \\\\",
        "\\midrule",
    ]
    for (task, cap, algo), ms in sorted(groups.items()):
        has_pre = all(m["pre"] is not None for m in ms)
        acc = (
            f"{fmt(ms, 'pre', 'acc')} $\\to$ {fmt(ms, 'acc', 'acc')}"
            if has_pre
            else fmt(ms, "acc", "acc")
        )
        lines.append(
            " & ".join(
                [
                    task,
                    str(cap),
                    algo.replace("_", "\\_"),
                    str(len(ms)),
                    pair(ms, "c0", "c1", "len"),
                    pair(ms, "i0", "i1", "len"),
                    pair(ms, "g0", "g1", "gap"),
                    fmt(ms, "rho_inc", "rho"),
                    fmt(ms, "cap1", "frac"),
                    acc,
                ]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
