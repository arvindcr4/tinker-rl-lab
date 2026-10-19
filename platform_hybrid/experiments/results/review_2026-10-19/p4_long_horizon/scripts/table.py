"""Summarize the 200-step, 1024-token-cap GSM8K-CoT runs from raw/*.json.

Writes summary.md and length_long_horizon_table.tex next to raw/.
Windows are the first and last 20 steps; lengths are pooled per run over all
rollouts in the window, then mean +/- SD over seeds.
"""

import hashlib
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
W = 20


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    x, y = list(x), list(y)
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def window(steps):
    cor, inc, cap = [], [], []
    for s in steps:
        for length, reward in zip(s["lengths"], s["rewards"]):
            (cor if reward > 0 else inc).append(length)
        cap.append(s["frac_at_cap"])
    return st.mean(cor), st.mean(inc), st.mean(cap)


def run_stats(path):
    run = json.loads(path.read_text())["runs"][0]
    log = run["step_log"]
    assert [s["step"] for s in log] == list(range(200)), path
    pre, post = run["pre_correct"], run["post_correct"]
    assert len(pre) == len(post) == run["n_eval"] == 200, path
    c0, i0, _ = window(log[:W])
    c1, i1, cap1 = window(log[-W:])
    return {
        "algo": run["algo"],
        "seed": run["seed"],
        "cor": (c0, c1),
        "inc": (i0, i1),
        "gap": (i0 - c0, i1 - c1),
        # steps with no incorrect rollout have len_incorrect_mean=None and are skipped
        "rho_inc": spearman(
            *zip(*[(s["step"], s["len_incorrect_mean"]) for s in log if s["n_incorrect"]])
        ),
        "steps_no_inc": sum(not s["n_incorrect"] for s in log),
        "cap_last": cap1,
        "pre": sum(pre) / 200,
        "post": sum(post) / 200,
        "improved": sum(a == 0 and b == 1 for a, b in zip(pre, post)),
        "regressed": sum(a == 1 and b == 0 for a, b in zip(pre, post)),
        "pre_items": tuple(pre),
    }


def ms(values, fmt):
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{format(st.mean(values), fmt)}$\\pm${format(sd, fmt)}"


def main():
    validation = json.loads((RAW / "validation.json").read_text())
    expected = {(r["algo"], r["seed"]): r["sha256"] for r in validation["runs"]}
    rows = []
    for path in sorted(RAW.glob("*_s*.json")):
        s = run_stats(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected[(s["algo"], s["seed"])], path
        rows.append(s)
    assert len(rows) == 6 and len({r["pre_items"] for r in rows}) == 1

    md = [
        "# P4 long-horizon GSM8K-CoT (200 steps, cap 1024, Qwen2.5-1.5B-Instruct)",
        "",
        f"Windows: first/last {W} steps. Held-out: 200 fixed GSM8K items, greedy.",
        "",
        "| algo | seed | len cor first->last | len inc first->last | gap first->last | rho(step,len_inc) | at-cap last | pre | post | +/- | steps w/o inc |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['algo']} | {r['seed']} | {r['cor'][0]:.0f}->{r['cor'][1]:.0f} | "
            f"{r['inc'][0]:.0f}->{r['inc'][1]:.0f} | {r['gap'][0]:+.0f}->{r['gap'][1]:+.0f} | "
            f"{r['rho_inc']:+.2f} | {r['cap_last']:.4f} | {r['pre']:.3f} | {r['post']:.3f} | "
            f"+{r['improved']}/-{r['regressed']} | {r['steps_no_inc']} |"
        )
    tex = []
    for algo in ("dr_grpo", "grpo"):
        g = [r for r in rows if r["algo"] == algo]
        name = algo.replace("_", "\\_")
        tex.append(
            f"{name} & {len(g)} & "
            + " & ".join(
                f"{ms([r[k][0] for r in g], '.0f')} $\\to$ {ms([r[k][1] for r in g], '.0f')}"
                for k in ("cor", "inc", "gap")
            )
            + f" & {ms([r['rho_inc'] for r in g], '.2f')}"
            + f" & {ms([r['cap_last'] for r in g], '.3f')}"
            + f" & {g[0]['pre']:.3f} $\\to$ {ms([r['post'] for r in g], '.3f')} \\\\"
        )
        md.append(
            f"\n{algo}: post mean {st.mean(r['post'] for r in g):.4f}, "
            f"transitions +{sum(r['improved'] for r in g)}/-{sum(r['regressed'] for r in g)} over 3 seeds"
        )
    (ROOT / "summary.md").write_text("\n".join(md) + "\n")
    (ROOT / "length_long_horizon_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print("\n".join(tex))


if __name__ == "__main__":
    main()
