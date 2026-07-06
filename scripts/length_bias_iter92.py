#!/usr/bin/env python3
"""Iter 92 -- Pillar 4 (Length Bias / Dr.GRPO): Transfer-entropy decomposition.

Iter 84 settled linear frequency-domain (Hurst, coherence, Granger F) and
iter 88 settled non-linear conditional-quantile coupling.  Iter 92 attacks
the *causal* direction of the L<->R coupling, which linear / quantile
methods cannot resolve.

AIM.  Decompose the L_t <-> R_t coupling into two directed, time-asymmetric
information flows:

    TE_{L->R} = I( R_{t+1} ; L_t  | R_t )    [bits]
    TE_{R->L} = I( L_{t+1} ; R_t  | L_t )    [bits]

where I(. ; . | .) is conditional mutual information.  TE_{L->R} is the
information that *length* carries about the *next-step reward* that is
not already in the *current* reward -- i.e. the predictive "length
bias".  TE_{R->L} is the dual: the information the current reward
carries about the *next-step length* that is not already in current
length -- the "length adaptation" feedback that Dr.GRPO is designed to
sever.

The sharpest Dr.GRPO prediction is therefore on the ASYMMETRY

    Delta_TE = TE_{L->R} - TE_{R->L}

Dr.GRPO should INCREASE Delta_TE by depressing TE_{R->L} (because the
reward's effect on subsequent length is removed), while keeping or
increasing TE_{L->R} (because the *causal* length-bias signal, if
anything, becomes more visible once the spurious R->L feedback is
removed).  Equivalently, the L->R : R->L ratio R_TE = TE_{L->R} / TE_{R->L}
should GROW under Dr.GRPO.

We additionally decompose each TE flow into the bin-resolved "witness
matrix" -- the (l_t, r_t) cell of the transition probability table that
contributes the most conditional log-likelihood per bit -- to identify
WHICH joint state is the dominant channel for the directed flow.

NULL MODEL.  Short step logs (n in {30, 40}) make asymptotic MI
asymptotics unreliable.  We construct a *time-shift* surrogate null per
run by circularly shifting L_t by a random offset delta drawn from
{4, 5, ..., n-4} -- this preserves the marginal distribution of L AND
the autocorrelation of L AND any pure (L_t)-self structure of R, while
destroying the (L_t -> R_{t+1}) temporal coupling.  We use 200 such
surrogates per run to compute a per-run pseudo-p; the *seed-level*
test is the share of seeds with p_surr < 0.05.

Inputs : drgrpo_vs_grpo.json          (arithmetic_easy, n=40)
         drgrpo_gsm8k_cot_full.json   (gsm8k_cot, n=30)
Outputs: experiments/results/length_bias_iter92_{perrun,paired,
         witness,summary}.tsv + meta.json
Stdlib + numpy + scipy.stats.
"""
import json, os, math
from collections import defaultdict
import numpy as np
from scipy import stats as scs

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
B_BOOT = 2000
N_SURR = 200
RNG_SEED = 92
N_BINS = 3                # ternary {low, mid, high} binning of L and R


# ============================================================ binned TE ====
def _discretise(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile-binned version of x -- bins are {0, 1, ..., n_bins-1}."""
    x = np.asarray(x, float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs[1:-1])
    edges = np.unique(edges)
    if len(edges) < n_bins - 1:
        # not enough unique values for a clean split -- use rank bins
        ranks = scs.rankdata(x, method="average")
        return np.minimum((ranks - 1) * n_bins // len(x), n_bins - 1).astype(int)
    return np.digitize(x, edges).astype(int)


def _joint(*xs: np.ndarray) -> np.ndarray:
    """Flatten multiple integer arrays into a single 1-D index via base-N
    encoding.  This is the canonical way to address a joint table cell."""
    out = np.zeros(len(xs[0]), dtype=np.int64)
    for x in xs:
        out = out * (int(x.max()) + 1) + x
    return out


def _te_binned(target_next: np.ndarray, src: np.ndarray, cond: np.ndarray) -> float:
    """Binned transfer entropy TE_{src -> target_next} given cond at lag 1.

    target_next[t] = target[t+1] for t = 0..T-2
    src[t]         = source[t]   for t = 0..T-2
    cond[t]        = cond[t]     for t = 0..T-2

    TE = sum_{a,b,c} p(a,b,c) * log2( p(a|b,c) / p(a|c) )

    The denominator uses only the target's own past (c) so the metric is
    the *conditional* mutual information and not contaminated by
    target-self predictability.
    """
    T = len(target_next)
    assert len(src) == T and len(cond) == T
    if T < 8:
        return float("nan")

    jt = _joint(target_next, src, cond)             # p(a,b,c)
    jtc = _joint(target_next, cond)                  # p(a,c)
    jc = cond                                        # p(c)
    n = float(T)

    cnt_full = defaultdict(int)
    cnt_ac = defaultdict(int)
    cnt_c = defaultdict(int)
    for k in range(T):
        cnt_full[(target_next[k], src[k], cond[k])] += 1
        cnt_ac[(target_next[k], cond[k])] += 1
        cnt_c[cond[k]] += 1

    te = 0.0
    for key, n_abc in cnt_full.items():
        a, b, c = key
        p_abc = n_abc / n
        p_ac = cnt_ac[(a, c)] / n
        p_c = cnt_c[c] / n
        # p(a|b,c) = p_abc / p(b,c)  and  p(a|c) = p_ac / p_c
        p_bc = (p_abc * p_ac) ** 0.0  # placeholder; we will compute directly
        p_bc = (sum(v for (a2, b2, c2), v in cnt_full.items()
                    if b2 == b and c2 == c) / n)
        if p_abc > 0 and p_ac > 0 and p_bc > 0 and p_c > 0:
            # p(a|b,c) = p_abc / p_bc
            p_a_given_bc = p_abc / p_bc
            p_a_given_c = p_ac / p_c
            if p_a_given_bc > 0 and p_a_given_c > 0:
                te += p_abc * (math.log2(p_a_given_bc) - math.log2(p_a_given_c))
    return float(te)


def te_pair(R: np.ndarray, L: np.ndarray) -> dict:
    """Compute {TE_LR, TE_RL, Delta_TE, R_TE} on discretised R and L.

    R, L are 1-D numpy arrays of per-step values; we drop the last
    index to form the (t -> t+1) transition.
    """
    T = min(len(R), len(L))
    if T < 8:
        return dict(te_lr=float("nan"), te_rl=float("nan"),
                    delta=float("nan"), ratio=float("nan"))
    Rd = _discretise(R[:T])
    Ld = _discretise(L[:T])
    R_next = Rd[1:]
    L_next = Ld[1:]
    R_cur = Rd[:-1]
    L_cur = Ld[:-1]
    te_lr = _te_binned(R_next, L_cur, R_cur)         # I(R_{t+1}; L_t | R_t)
    te_rl = _te_binned(L_next, R_cur, L_cur)         # I(L_{t+1}; R_t | L_t)
    delta = te_lr - te_rl
    ratio = (te_lr / te_rl) if (te_rl is not None and te_rl > 1e-9) else float("nan")
    return dict(te_lr=te_lr, te_rl=te_rl, delta=delta, ratio=ratio)


# ====================================================== surrogate null ====
def te_surrogate_p(R: np.ndarray, L: np.ndarray, n_surr: int = N_SURR) -> dict:
    """Per-run null test: circular-shift L by random lag, recompute TE.

    Returns {te_lr_obs, te_rl_obs, p_lr, p_rl, p_delta} where the p is
    the fraction of 200 surrogates with TE >= observed.  The
    null-hypothesis test is therefore one-sided ('TE_LR is positive
    beyond chance').
    """
    obs = te_pair(R, L)
    T = min(len(R), len(L))
    if T < 8:
        return dict(te_lr_obs=obs["te_lr"], te_rl_obs=obs["te_rl"],
                    p_lr=float("nan"), p_rl=float("nan"),
                    p_delta=float("nan"))
    rng = np.random.default_rng(RNG_SEED)
    surr_lr = np.empty(n_surr, float)
    surr_rl = np.empty(n_surr, float)
    surr_d = np.empty(n_surr, float)
    # surrogate shifts we draw from -- avoid trivial 0, 1, 2 shifts
    shifts = np.arange(4, max(5, T - 4))
    for i in range(n_surr):
        s = int(rng.choice(shifts))
        L_sh = np.roll(L, s)
        d = te_pair(R, L_sh)
        surr_lr[i] = d["te_lr"]
        surr_rl[i] = d["te_rl"]
        surr_d[i] = d["delta"]
    p_lr = float((surr_lr >= obs["te_lr"]).sum() + 1) / (n_surr + 1)
    p_rl = float((surr_rl >= obs["te_rl"]).sum() + 1) / (n_surr + 1)
    p_d = float((surr_d >= obs["delta"]).sum() + 1) / (n_surr + 1)
    return dict(te_lr_obs=obs["te_lr"], te_rl_obs=obs["te_rl"],
                p_lr=p_lr, p_rl=p_rl, p_delta=p_d)


def te_reversal_asymmetry(R: np.ndarray, L: np.ndarray) -> dict:
    """Time-reversal asymmetry test (Schreiber 2000).

    A truly *directional* coupling (L_t -> R_{t+1}) should DECREASE when
    the time axis is reversed -- the (L_t -> R_{t+1}) predictive link
    becomes (L_t -> R_{t-1}), which has no temporal precedence and
    should look no different from surrogate.  A *symmetric* (instantaneous)
    correlation is unchanged by reversal.

    A_te = (TE_obs - TE_reversed) / (TE_obs + TE_reversed)
    A_te > 0 means the L->R coupling is directional; A_te < 0 means the
    reversed direction is stronger (a reverse arrow).
    """
    T = min(len(R), len(L))
    if T < 8:
        return dict(a_lr=float("nan"), a_rl=float("nan"))
    obs = te_pair(R, L)
    rev = te_pair(R[::-1], L[::-1])
    s_lr = obs["te_lr"] + rev["te_lr"]
    s_rl = obs["te_rl"] + rev["te_rl"]
    a_lr = (obs["te_lr"] - rev["te_lr"]) / s_lr if s_lr > 1e-9 else float("nan")
    a_rl = (obs["te_rl"] - rev["te_rl"]) / s_rl if s_rl > 1e-9 else float("nan")
    return dict(a_lr=a_lr, a_rl=a_rl)


# ============================================================ witness ====
def witness_winner(R: np.ndarray, L: np.ndarray) -> dict:
    """Return the (l_t, r_t) joint cell that contributes the most to
    TE_{L->R} per step (the 'dominant channel' of the L->R flow).

    contribution(a, b, c) = p(a, b, c) * log2( p(a|b,c) / p(a|c) )
    We find the (b, c) pair that maximises the SUM of contributions
    with that (b, c) -- i.e. the joint state of (L_t, R_t) that is the
    best witness for predicting R_{t+1}.
    """
    T = min(len(R), len(L))
    if T < 8:
        return dict(witness_lr="nan", witness_rl="nan")
    Rd = _discretise(R[:T])
    Ld = _discretise(L[:T])

    def _winner(src, cond, target_next):
        cnt_full = defaultdict(int)
        cnt_ac = defaultdict(int)
        cnt_c = defaultdict(int)
        cnt_bc = defaultdict(int)
        for k in range(T - 1):
            a, b, c = target_next[k], src[k], cond[k]
            cnt_full[(a, b, c)] += 1
            cnt_ac[(a, c)] += 1
            cnt_c[c] += 1
            cnt_bc[(b, c)] += 1
        n = float(T - 1)
        contrib = defaultdict(float)
        for (a, b, c), n_abc in cnt_full.items():
            p_abc = n_abc / n
            p_ac = cnt_ac[(a, c)] / n
            p_c = cnt_c[c] / n
            p_bc = cnt_bc[(b, c)] / n
            if p_abc > 0 and p_ac > 0 and p_bc > 0 and p_c > 0:
                p_a_bc = p_abc / p_bc
                p_a_c = p_ac / p_c
                if p_a_bc > 0 and p_a_c > 0:
                    contrib[(b, c)] += p_abc * (
                        math.log2(p_a_bc) - math.log2(p_a_c))
        if not contrib:
            return "nan"
        # each (b,c) can be at most one cell; we report the median b
        # and c index in the cell that contributes most.
        b, c = max(contrib, key=lambda k: contrib[k])
        return f"L{b}->R{c}={contrib[(b, c)]:.4f}bits"

    return dict(
        witness_lr=_winner(Ld[:-1], Rd[:-1], Rd[1:]),
        witness_rl=_winner(Rd[:-1], Ld[:-1], Ld[1:]),
    )


# ============================================================ data ====
def load_runs() -> list:
    runs = []
    d1 = json.load(open(os.path.join(RES, "drgrpo_vs_grpo.json")))
    for r in d1["runs"]:
        sl = r.get("step_log", [])
        if len(sl) < 8:
            continue
        L = np.array([s["mean_comp_len"] for s in sl], float)
        R = np.array([s["mean_reward"]  for s in sl], float)
        runs.append({"task": "arithmetic_easy", "algo": r["algo"],
                     "seed": r["seed"], "L": L, "R": R,
                     "n": int(len(sl))})
    d2 = json.load(open(os.path.join(RES, "drgrpo_gsm8k_cot_full.json")))
    for r in d2["runs"]:
        sl = r.get("step_log", [])
        if len(sl) < 8:
            continue
        L = np.array([s["mean_comp_len"] for s in sl], float)
        R = np.array([s["mean_reward"]  for s in sl], float)
        runs.append({"task": "gsm8k_cot", "algo": r["algo"],
                     "seed": r["seed"], "L": L, "R": R,
                     "n": int(len(sl))})
    return runs


# ========================================================== writers ====
def write_tsv(path: str, rows: list, header: list):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            line = []
            for h in header:
                v = r.get(h, "")
                if isinstance(v, float):
                    line.append(f"{v:.6f}" if math.isfinite(v) else "nan")
                else:
                    line.append(str(v))
            f.write("\t".join(line) + "\n")


# ============================================================== main ====
def analyse():
    runs = load_runs()
    perrun = []
    for r in runs:
        surr = te_surrogate_p(r["R"], r["L"])
        te = te_pair(r["R"], r["L"])
        rev = te_reversal_asymmetry(r["R"], r["L"])
        wit = witness_winner(r["R"], r["L"])
        perrun.append({
            "task":   r["task"],
            "algo":   r["algo"],
            "seed":   r["seed"],
            "n":      r["n"],
            "L_mean": float(r["L"].mean()),
            "R_mean": float(r["R"].mean()),
            "te_lr":  te["te_lr"],
            "te_rl":  te["te_rl"],
            "delta":  te["delta"],
            "ratio":  te["ratio"],
            "p_lr":   surr["p_lr"],
            "p_rl":   surr["p_rl"],
            "p_delta": surr["p_delta"],
            "a_lr":   rev["a_lr"],
            "a_rl":   rev["a_rl"],
            "witness_lr": wit["witness_lr"],
            "witness_rl": wit["witness_rl"],
        })
    return perrun


def paired_diff(perrun, task, key):
    by_key = defaultdict(dict)
    for row in perrun:
        if row["task"] != task:
            continue
        by_key[row["seed"]][row["algo"]] = row[key]
    pairs = []
    for s, d in by_key.items():
        if "grpo" in d and "dr_grpo" in d:
            a, b = d["dr_grpo"], d["grpo"]
            if not (math.isnan(a) or math.isnan(b)):
                pairs.append((a, b))
    if len(pairs) < 2:
        return {"task": task, "key": key, "n_pairs": len(pairs),
                "mean_diff": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan")}
    diffs = np.array([p[0] - p[1] for p in pairs], float)
    obs = float(diffs.mean())
    rng = np.random.default_rng(RNG_SEED)
    n = len(diffs)
    means = np.empty(B_BOOT, float)
    for b in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        means[b] = diffs[idx].mean()
    means.sort()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    if diffs.std(ddof=1) > 0:
        t = obs / (diffs.std(ddof=1) / np.sqrt(n))
        p = float(2.0 * (1.0 - scs.t.cdf(abs(t), df=n - 1)))
    else:
        p = 1.0
    return {"task": task, "key": key, "n_pairs": n,
            "mean_diff": obs, "ci_lo": lo, "ci_hi": hi, "p": p}


def main():
    perrun = analyse()
    base = os.path.join(RES, "length_bias_iter92_")
    write_tsv(base + "perrun.tsv", perrun,
              ["task", "algo", "seed", "n", "L_mean", "R_mean",
               "te_lr", "te_rl", "delta", "ratio",
               "p_lr", "p_rl", "p_delta",
               "a_lr", "a_rl",
               "witness_lr", "witness_rl"])

    paired_rows = []
    summary_rows = []
    keys = ["te_lr", "te_rl", "delta", "ratio", "a_lr", "a_rl"]
    for task in {"arithmetic_easy", "gsm8k_cot"}:
        for algo in ("grpo", "dr_grpo"):
            for k in keys:
                vs = [r[k] for r in perrun
                      if r["task"] == task and r["algo"] == algo
                      and math.isfinite(r[k])]
                if vs:
                    summary_rows.append({
                        "task": task, "algo": algo, "key": k,
                        "n_seeds": len(vs),
                        "mean": float(np.mean(vs)),
                        "std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                    })
        for k in keys:
            paired_rows.append(paired_diff(perrun, task, k))

    # Also compute per-task seed-share of significant TE_LR (p<0.05) and
    # significant TE_RL (p<0.05) -- the "directed-coupling detection rate"
    detect_rows = []
    for task in {"arithmetic_easy", "gsm8k_cot"}:
        for algo in ("grpo", "dr_grpo"):
            n_te = sum(1 for r in perrun
                        if r["task"] == task and r["algo"] == algo
                        and r["p_lr"] < 0.05)
            n_rl = sum(1 for r in perrun
                       if r["task"] == task and r["algo"] == algo
                       and r["p_rl"] < 0.05)
            n_d = sum(1 for r in perrun
                      if r["task"] == task and r["algo"] == algo
                      and r["p_delta"] < 0.05)
            n_tot = sum(1 for r in perrun
                        if r["task"] == task and r["algo"] == algo)
            detect_rows.append({
                "task": task, "algo": algo, "n_seeds": n_tot,
                "frac_p_lr_lt_05": n_te / max(n_tot, 1),
                "frac_p_rl_lt_05": n_rl / max(n_tot, 1),
                "frac_p_delta_lt_05": n_d / max(n_tot, 1),
            })

    write_tsv(base + "paired.tsv", paired_rows,
              ["task", "key", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p"])
    write_tsv(base + "summary.tsv", summary_rows,
              ["task", "algo", "key", "n_seeds", "mean", "std"])
    write_tsv(base + "detect.tsv", detect_rows,
              ["task", "algo", "n_seeds",
               "frac_p_lr_lt_05", "frac_p_rl_lt_05", "frac_p_delta_lt_05"])

    # write witness list
    wit_rows = [dict(task=r["task"], algo=r["algo"], seed=r["seed"],
                     witness_lr=r["witness_lr"], witness_rl=r["witness_rl"])
                for r in perrun]
    write_tsv(base + "witness.tsv", wit_rows,
              ["task", "algo", "seed", "witness_lr", "witness_rl"])

    meta = {
        "iter": 92,
        "pillar": "P4-LengthBias",
        "task": ("Pillar 4 (Length Bias / Dr.GRPO): Transfer-entropy "
                 "decomposition (TE_{L->R} vs TE_{R->L})"),
        "stats": ["te_lr", "te_rl", "delta", "ratio", "p_lr", "p_rl",
                  "p_delta", "witness_lr", "witness_rl"],
        "n_bins": N_BINS,
        "n_surrogates": N_SURR,
        "lag": 1,
        "null_model": "circular_shift_surrogate",
        "inputs": ["drgrpo_vs_grpo.json", "drgrpo_gsm8k_cot_full.json"],
        "n_runs": len(perrun),
        "n_paired_tests": len(paired_rows),
    }
    with open(base + "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"iter92 wrote {len(perrun)} per-run rows; "
          f"{len(paired_rows)} paired bootstrap tests")
    for d in paired_rows:
        if d["key"] in ("te_lr", "te_rl", "delta", "ratio", "a_lr", "a_rl"):
            print(f"  {d['task']:>15s}  {d['key']:>10s}  "
                  f"diff={d['mean_diff']:+.4f}  "
                  f"CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
                  f"p={d['p']:.4f}  n={d['n_pairs']}")
    print("--- per-task detection rate (fraction of seeds with p<0.05) ---")
    for d in detect_rows:
        print(f"  {d['task']:>15s}  {d['algo']:>8s}  "
              f"TE_LR={d['frac_p_lr_lt_05']:.2f}  "
              f"TE_RL={d['frac_p_rl_lt_05']:.2f}  "
              f"Delta={d['frac_p_delta_lt_05']:.2f}  (n={d['n_seeds']})")


if __name__ == "__main__":
    main()
