"""
SP25 L4 — Hajishirzi 'Unpacking DPO and PPO' pipeline-factor audit.
Maps Ivison et al. (arXiv:2406.09279, NeurIPS 2024) onto our Pillar 3 + 4 stacks.

Framework (Ivison et al., 2024): RL-from-feedback pipelines decompose into
four axes — preference data, learning algorithm, reward model, and policy
training prompts. For verifiable-reward RL (Tulu 3 RLVR, arXiv:2411.15124)
two axes (data, prompts) are pinned by construction; the remaining two
(algorithm, reward model) become the testable variance contributors.

Hypotheses:
  H1 ALGO axis vs residual (samestack_ppo_grpo): SS_algo / SS_total
     DECISIVE if <= 0.05 (i.e., algorithm-axis explains <5% of variance).
  H2 REWARD axis vs residual (variance_mitigation 9 methods x 5 seeds):
     DECISIVE if SS_method / SS_total <= 0.20; the 9 algorithmic variants
     are reward-interventions but should be dominated by seed noise.
  H3 IID hypothesis: |delta_grpo_minus_ppo| <= 0.005 holds (Tulu 3 RLVR
     equivalence claim).
  H4 CDH overlay (frontier synthesis): grad_norm_method / grad_norm_algo
     contribution is co-dominant with reward axis under our CDH row 12
     finding -- 'algorithm axis is at most competitive with reward axis'.
"""
from __future__ import annotations
import json, math, os
from collections import defaultdict
from statistics import fmean, pstdev, median

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES  = os.path.join(ROOT, "experiments", "results")
OUT  = os.path.join(RES, "berkeley")
os.makedirs(OUT, exist_ok=True)

# ----------------- helpers -----------------

def shannon(label):
    print(f"\n=== {label} ===")

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = fmean(a), fmean(b)
    sa, sb = pstdev(a), pstdev(b)
    sp = math.sqrt(((len(a)-1)*sa*sa + (len(b)-1)*sb*sb) / (len(a)+len(b)-2))
    return (ma - mb) / sp if sp > 1e-12 else float("nan")

def safe(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)

def axis_variance_fraction(rows, axis_key, value_key):
    """Compute SS_axis / SS_total for nested groups.

    rows: list[dict]. axis_key is the categorical axis (e.g. 'method'),
    value_key is the numeric scalar (e.g. 'last10_avg').
    """
    grand = []
    by_axis = defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand:
        return float("nan"), float("nan"), float("nan")
    grand_mean = fmean(grand)
    ss_total = sum((x - grand_mean) ** 2 for x in grand)
    ss_axis  = sum(len(vs) * (fmean(vs) - grand_mean) ** 2 for vs in by_axis.values())
    ss_within = ss_total - ss_axis
    eta2 = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    return eta2, ss_axis, ss_within

def extract_metric_rows(samestack_ppo_grpo_path):
    """Pull last10_avg per (algo, seed) from samestack_ppo_grpo.json."""
    out = []
    with open(samestack_ppo_grpo_path) as f:
        data = json.load(f)
    for run in data["runs"]:
        out.append({
            "algo": run["algo"],
            "seed": run["seed"],
            "heldout_acc": run.get("heldout_acc"),
            "last10_avg": run.get("last10_avg"),
        })
    return out

def extract_varmit_rows(variance_mitigation_path):
    """Per-method per-seed terminal scalars from variance_mitigation.tsv (5 seeds x 9 methods)."""
    out = []
    with open(variance_mitigation_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) < len(header):
                continue
            out.append({
                "method": cells[idx["method"]],
                "seed":   cells[idx["seed"]],
                "step":   int(cells[idx["step"]]),
                "zvf":    float(cells[idx["zvf"]]),
                "reward_mean": float(cells[idx["reward_mean"]]),
                "heldout_acc": float(cells[idx["heldout_acc"]]),
                "collapse": int(cells[idx["collapse"]]),
            })
    # reduce to terminal-per-(method, seed) using the LAST 5 steps mean
    terminal = defaultdict(list)
    for r in out:
        terminal[(r["method"], r["seed"])].append(r["heldout_acc"])
    terminal_rows = []
    for (m, s), vs in terminal.items():
        terminal_rows.append({
            "method": m,
            "seed":   s,
            "terminal_acc": fmean(vs[-5:]),
            "max_acc": max(vs),
        })
    return terminal_rows

def extract_group_size_rows(group_size_path):
    """Per-(G, seed) terminal-mean from group_size_advantage_variance.tsv."""
    terminal = defaultdict(list)
    with open(group_size_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) < len(header):
                continue
            G = int(cells[idx["G"]])
            seed = int(cells[idx["seed"]])
            step = int(cells[idx["step"]])
            ent = float(cells[idx["entropy"]])
            reward = float(cells[idx["mean_reward"]])
            gn = float(cells[idx["grad_norm"]])
            terminal[(G, seed)].append((step, reward, gn, ent))
    rows = []
    for (G, seed), traj in terminal.items():
        last10 = [r for s, r, *_ in traj if s >= 30]
        max_g  = max(g for _, _, g, _ in traj)
        rows.append({
            "G": G, "seed": seed,
            "last10_reward": fmean(last10) if last10 else float("nan"),
            "max_grad_norm": max_g,
            "mean_entropy": fmean(e for *_1, e in traj if e > 0),
        })
    return rows

# ----------------- computations -----------------

def main():
    findings = []

    # ----- H1: samestack_ppo_grpo algorithm-axis variance (Ivison axis #2) -----
    shannon("H1: ALGO axis vs residual (samestack PPO/GRPO, n=5 seeds)")
    rows_ss = extract_metric_rows(os.path.join(RES, "samestack_ppo_grpo.json"))
    rows_heldout = [r for r in rows_ss if r.get("heldout_acc") is not None]
    eta2_heldout, ss_axis, ss_within = axis_variance_fraction(rows_heldout, "algo", "heldout_acc")
    print(f"  eta^2(algo -> heldout_acc) = {eta2_heldout:.4f}  (SS_axis={ss_axis:.6f}, SS_within={ss_within:.6f})")

    grpo_acc = [r["heldout_acc"] for r in rows_heldout if r["algo"] == "grpo"]
    ppo_acc  = [r["heldout_acc"] for r in rows_heldout if r["algo"] == "ppo"]
    delta_acc = fmean(grpo_acc) - fmean(ppo_acc)
    d_h1 = cohens_d(grpo_acc, ppo_acc)
    print(f"  paired delta (grpo - ppo) heldout_acc = {delta_acc:+.4f}, Cohen's d = {d_h1:+.3f}")
    n_pairs = min(len(grpo_acc), len(ppo_acc))

    # exact p via paired permutation (10k iters)
    rng_state = []
    import random
    random.seed(7)
    diffs_obs = []
    pairs = []
    g_idx = [(i, r) for i, r in enumerate(rows_heldout) if r["algo"] == "grpo"]
    p_idx = [(i, r) for i, r in enumerate(rows_heldout) if r["algo"] == "ppo"]
    g_by_seed = {r["seed"]: r["heldout_acc"] for _, r in g_idx}
    p_by_seed = {r["seed"]: r["heldout_acc"] for _, r in p_idx}
    common = sorted(set(g_by_seed) & set(p_by_seed))
    diffs_obs = [g_by_seed[s] - p_by_seed[s] for s in common]
    obs_mean = fmean(diffs_obs) if diffs_obs else 0.0
    count = 0
    n_perm = 10000
    for _ in range(n_perm):
        signs = [random.choice([-1, 1]) for _ in diffs_obs]
        perm_mean = fmean(d * sg for d, sg in zip(diffs_obs, signs))
        if abs(perm_mean) >= abs(obs_mean):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    print(f"  paired permutation p (|delta|>=|obs|) = {p_perm:.4f}")
    h1_decisive = (eta2_heldout <= 0.05) and (abs(delta_acc) <= 0.005) and (p_perm > 0.10)
    h1_verdict  = "DECISIVE (algorithm axis <5% AND |delta|<=0.005)" if h1_decisive else \
                  "SUGGESTIVE" if eta2_heldout <= 0.10 else "NULL"
    print(f"  verdict: {h1_verdict}")

    findings.append({
        "hypothesis": "H1: ALGO axis variance is small (<5%) for samestack PPO vs GRPO",
        "n_pairs": len(common),
        "eta2": eta2_heldout,
        "delta_acc": delta_acc,
        "cohens_d": d_h1,
        "perm_p_two_sided": p_perm,
        "verdict": h1_verdict,
    })

    # ----- H2: variance_mitigation reward-intervention axis -----
    shannon("H2: METHOD axis (reward-intervention) variance")
    vm_path = os.path.join(RES, "variance_mitigation.tsv")
    if not os.path.exists(vm_path):
        print("  (variance_mitigation.tsv not found -- skipping H2)")
    else:
        rows_vm = extract_varmit_rows(vm_path)
        eta2_method, ss_m_axis, ss_m_within = axis_variance_fraction(rows_vm, "method", "terminal_acc")
        per_method = defaultdict(list)
        for r in rows_vm:
            per_method[r["method"]].append(r["terminal_acc"])
        method_means = {m: fmean(vs) for m, vs in per_method.items()}
        method_spread = (max(method_means.values()) - min(method_means.values()))
        print(f"  eta^2(method -> terminal_acc)  = {eta2_method:.4f}  (SS_axis={ss_m_axis:.4f}, SS_within={ss_m_within:.4f})")
        print(f"  per-method terminal_acc mean:  { {m: round(v, 4) for m, v in method_means.items()} }")
        print(f"  spread max-min method mean     = {method_spread:.4f}")

        # Pair-wise Cohen's d GRPO vs each method
        grpo_v = per_method.get("grpo", [])
        pair_d = []
        for m, vs in per_method.items():
            if m == "grpo":
                continue
            pair_d.append((m, cohens_d(grpo_v, vs)))
        pair_d.sort(key=lambda x: x[1])
        print("  Cohen's d (grpo - other):", {m: round(d, 3) for m, d in pair_d})

        h2_decisive = (eta2_method <= 0.20) and (method_spread <= 0.10)
        h2_verdict  = "DECISIVE (Ivison reward-axis is small fraction + small spread)" if h2_decisive else \
                      "SUGGESTIVE" if eta2_method <= 0.35 else "NULL"
        print(f"  verdict: {h2_verdict}")

        findings.append({
            "hypothesis": "H2: REWARD/INTERVENTION axis variance is small (<20%)",
            "n_methods": len(per_method),
            "n_terminal_rows": len(rows_vm),
            "eta2_method": eta2_method,
            "spread": method_spread,
            "pair_d_grpo_vs": {m: round(d, 4) for m, d in pair_d},
            "verdict": h2_verdict,
        })

    # ----- H3: RLVR/Match-axis equivalence (Tulu 3) -----
    shannon("H3: RLVR equivalence (Tulu 3 arXiv:2411.15124, |delta_grpo_minus_ppo|<=0.005)")
    print(f"  |delta|   = {abs(delta_acc):.4f}  (threshold 0.005)")
    h3_decisive = (abs(delta_acc) <= 0.005) and (p_perm > 0.10)
    h3_verdict = "DECISIVE" if h3_decisive else "NULL"
    print(f"  verdict: {h3_verdict}")
    findings.append({
        "hypothesis": "H3: |delta_grpo_minus_ppo|<=0.005 (Tulu 3 RLVR-equivalence)",
        "abs_delta": abs(delta_acc),
        "perm_p": p_perm,
        "verdict": h3_verdict,
    })

    # ----- H4: convergence-rate axis-variance decomposition -----
    shannon("H4: convergence-rate / entropy-rank axis dominates algorithm-rank axis")
    if not os.path.exists(vm_path):
        print("  (variance_mitigation missing -- skip)")
    else:
        # half-life to half-peak
        rows_vm_full = []
        with open(vm_path) as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {h: i for i, h in enumerate(header)}
            for line in f:
                cells = line.rstrip("\n").split("\t")
                if len(cells) < len(header):
                    continue
                rows_vm_full.append({
                    "method": cells[idx["method"]],
                    "seed":   int(cells[idx["seed"]]),
                    "step":   int(cells[idx["step"]]),
                    "zvf":    float(cells[idx["zvf"]]),
                    "heldout_acc": float(cells[idx["heldout_acc"]]),
                })
        # compute half-life per (method, seed)
        traj_idx = defaultdict(list)
        for r in rows_vm_full:
            traj_idx[(r["method"], r["seed"])].append((r["step"], r["heldout_acc"]))
        half_lives = []
        cv_by_method = defaultdict(list)
        for (m, s), traj in traj_idx.items():
            traj.sort()
            peak = max(v for _, v in traj)
            half = peak * 0.5
            reached = next((t for t, v in traj if v >= half), None)
            if reached is not None:
                half_lives.append({"method": m, "seed": s, "half_step": reached, "peak_acc": peak})
            # distribution of post-convergence rewards
            post = [v for t, v in traj if t >= 50]
            if post and len(post) > 5:
                cv_by_method[m].append(pstdev(post) / (fmean(post) + 1e-12))

        if half_lives:
            # coefficient of variation across methods (median half_step CV)
            per_method_median = defaultdict(list)
            for r in half_lives:
                per_method_median[r["method"]].append(r["half_step"])
            method_half_means = {m: fmean(vs) for m, vs in per_method_median.items()}
            method_half_stdev = {m: pstdev(vs) if len(vs) > 1 else 0.0 for m, vs in per_method_median.items()}
            cv_across_methods = (
                pstdev(list(method_half_means.values())) /
                (fmean(list(method_half_means.values())) + 1e-12)
            )
            cv_within_methods = fmean(method_half_stdev.values()) / (
                fmean(list(method_half_means.values())) + 1e-12
            )
            print(f"  CV(half_step across methods)  = {cv_across_methods:.3f}")
            print(f"  CV(half_step within methods)  = {cv_within_methods:.3f}")
            # eta^2 of method on half-step
            eta2_half, _, _ = axis_variance_fraction(half_lives, "method", "half_step")
            print(f"  eta^2(method -> half_step)    = {eta2_half:.4f}")
            h4_decisive = (eta2_half > 0.30) and (cv_across_methods > cv_within_methods)
            h4_verdict = "DECISIVE" if h4_decisive else "SUGGESTIVE" if eta2_half > 0.15 else "NULL"
            print(f"  verdict: {h4_verdict}")
            findings.append({
                "hypothesis": "H4: convergence-rate axis variance dominates algorithm axis",
                "cv_across_methods": cv_across_methods,
                "cv_within_methods": cv_within_methods,
                "eta2_method_half": eta2_half,
                "method_half_means": {m: round(v, 2) for m, v in method_half_means.items()},
                "verdict": h4_verdict,
            })

    # ----- H5: across-cell CDH overlay (CDH row 12 -- grad_norm channel) -----
    shannon("H5: grad_norm variance decomposition across G (group_size_advantage)")
    gs_path = os.path.join(RES, "group_size_advantage_variance.tsv")
    if not os.path.exists(gs_path):
        print("  (group_size_advantage_variance.tsv not found -- skipping H5)")
    else:
        rows_gs = extract_group_size_rows(gs_path)
        eta2_G_last10, ss_G, ss_W = axis_variance_fraction(rows_gs, "G", "last10_reward")
        eta2_G_grad, _, _       = axis_variance_fraction(rows_gs, "G", "max_grad_norm")
        per_G_last10 = defaultdict(list)
        per_G_grad   = defaultdict(list)
        for r in rows_gs:
            per_G_last10[r["G"]].append(r["last10_reward"])
            per_G_grad[r["G"]].append(r["max_grad_norm"])
        print(f"  eta^2(G -> last10_reward) = {eta2_G_last10:.4f}")
        print(f"  eta^2(G -> max_grad_norm)= {eta2_G_grad:.4f}")
        # Cohen's d on terminal reward G=2 vs G=16
        d_G = cohens_d(per_G_last10[2], per_G_last10[16])
        print(f"  Cohen's d last10_reward G=2 vs G=16 = {d_G:+.3f}")
        h5_decisive = (eta2_G_grad >= 0.40) and (eta2_G_last10 < 0.40)
        h5_verdict = "DECISIVE (grad_norm=G-driven, last10=G-shared)" if h5_decisive else \
                     "SUGGESTIVE" if eta2_G_grad > eta2_G_last10 else "NULL"
        print(f"  verdict: {h5_verdict}")
        findings.append({
            "hypothesis": "H5: grad_norm variance is G-axis-driven (CDH overlay)",
            "eta2_G_last10": eta2_G_last10,
            "eta2_G_grad": eta2_G_grad,
            "cohen_d_2v16": d_G,
            "per_G_last10_mean": {G: round(fmean(vs), 4) for G, vs in per_G_last10.items()},
            "verdict": h5_verdict,
        })

    # ----------------- save -----------------
    summary = {
        "ts": "2026-07-04",
        "iteration": 14,
        "pillar": "B-SP25",
        "lecture": "SP25 L4 — Hannaneh Hajishirzi (Tulu 3 / Unpacking DPO and PPO)",
        "papers": {
            "tulu3": {
                "title": "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
                "authors": "Lambert et al. (Allen AI / UW)",
                "arxiv": "2411.15124",
                "year": 2024,
                "venue": "arXiv (cs.CL) preprint",
            },
            "unpacking": {
                "title": "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback",
                "authors": "Ivison, Wang, Liu, Wu, Pyatkin, Lambert, Smith, Choi, Hajishirzi",
                "arxiv": "2406.09279",
                "year": 2024,
                "venue": "NeurIPS 2024 camera-ready",
            },
        },
        "framework": "Ivison 4-axis decomposition [data, algorithm, reward-model, prompts] -> verifiable-reward stacks pin (data, prompts). Testable axes: algorithm, reward-intervention.",
        "hypotheses": findings,
    }
    out_json = os.path.join(OUT, "unpacking_dpo_ppo_factorization.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # TSV row per hypothesis
    out_tsv = os.path.join(OUT, "unpacking_dpo_ppo_factorization.tsv")
    with open(out_tsv, "w") as f:
        f.write("hypothesis\tn_pairs\teta2\tabs_delta\tcohens_d\tperm_p\tverdict\n")
        for h in findings:
            f.write("\t".join([
                h.get("hypothesis", ""),
                str(h.get("n_pairs", "")),
                safe(h.get("eta2", h.get("eta2_method", h.get("eta2_G_last10", "")))),
                safe(h.get("abs_delta", h.get("spread", ""))),
                safe(h.get("cohens_d", "")),
                safe(h.get("perm_p", "")),
                h.get("verdict", ""),
            ]) + "\n")

    # ----------------- headline -----------------
    n_decisive = sum(1 for h in findings if h.get("verdict") == "DECISIVE (algorithm axis <5% AND |delta|<=0.005)" or h.get("verdict") == "DECISIVE")
    print(f"\n=== HEADLINE: {n_decisive}/{len(findings)} hypotheses DECISIVE")
    print(f"  outputs:")
    print(f"    {out_json}")
    print(f"    {out_tsv}")

if __name__ == "__main__":
    main()
