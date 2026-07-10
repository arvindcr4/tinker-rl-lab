"""Iter 19 — P7 vein (e): closed-form Beta-Binomial posterior-predictive
contrast-restoration on the N2 four-method reward tensors.

The four brief veins (a-d) are already covered:
  (a) counterfactual eval on N2            -> iter 03 (zvf-triage)
  (b) unify Dualformer-Auto + gamma*=0      -> iter 11 (Bayesian)
  (c) seed-robustness on N10                -> iter 07, iter 15
  (d) bootstrap CIs on every P7 headline    -> iter 14, iter 15

Vein (e) closes the empirical contrast-restoration loop on N2 data
itself (not the iter-15 synthetic Qwen/GSM8K sweep estimate
Delta_ZVF(8->16) = 0.059). For each observed prompt-step we form the
Beta(k+1, 9-k) posterior under a Beta(1,1) prior and compute the
posterior-predictive probability that escalating to G'=16 would
still yield a degenerate group. Closed-form via Beta-Binomial:

  P(Y' = y' | Y = k, n=8, G'=16, alpha=beta=1)
    = C(16, y') * B(k+1+y', 9-k+16-y') / B(k+1, 9-k)

  P(degenerate at G'=16) = P(Y'=0) + P(Y'=16)
  P(restored contrast)   = 1 - P(degenerate at G'=16)

Output:
  experiments/results/p5p8/p7_postpred_per_step.tsv
  experiments/results/p5p8/p7_postpred_summary.json
  experiments/results/p5p8/p7_postpred_summary.tsv
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
TENSOR_DIR = WORKTREE / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORKTREE / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ("grpo", "aero", "gift", "areal")
G_BASE, G_NEW = 8, 16
BOOT, RNG_SEED = 4000, 20260704


def betaln(a: float, b: float) -> float:
    """log B(a, b) = lgamma(a)+lgamma(b)-lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bb_postpred(k: int, n: int, yp: int, gp: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    """P(Y' = yp | Y=k, n=n, G'=gp) under Beta(alpha, beta) prior.

    Beta-Binomial closed form:
      P = C(gp, yp) * B(alpha+k+yp, beta+n-k+gp-yp) / B(alpha+k, beta+n-k)
    """
    log_p = (
        math.lgamma(gp + 1)
        - math.lgamma(yp + 1)
        - math.lgamma(gp - yp + 1)
        + betaln(alpha + k + yp, beta + n - k + gp - yp)
        - betaln(alpha + k, beta + n - k)
    )
    p = math.exp(log_p)
    # numerical safety: clamp to [0,1]
    return max(0.0, min(1.0, p))


def restore_prob(k: int, n: int = G_BASE, gp: int = G_NEW) -> float:
    """P(not degenerate at G'=gp) = 1 - P(Y'=0) - P(Y'=gp)."""
    p0 = bb_postpred(k, n, 0, gp)
    pgp = bb_postpred(k, n, gp, gp)
    return max(0.0, min(1.0, 1.0 - p0 - pgp))


def midrange_prob(k: int, n: int = G_BASE, alpha: float = 1.0, beta: float = 1.0) -> float:
    """m(k, n) = Pr(0.05 <= p <= 0.95 | Beta(k+alpha, n-k+beta)). Trapezoidal grid integration."""
    a = k + alpha
    b = n - k + beta
    lo, hi, grid = 0.05, 0.95, 1024
    dx = (hi - lo) / grid
    log_norm = betaln(a, b)
    total = 0.0
    prev = math.exp((a - 1.0) * math.log(lo) + (b - 1.0) * math.log(1.0 - lo) - log_norm)
    for i in range(1, grid + 1):
        x = lo + i * dx
        cur = math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_norm)
        total += 0.5 * (prev + cur) * dx
        prev = cur
    return total


def load_tensors(method: str) -> List[dict]:
    fp = TENSOR_DIR / f"{method}_s0_tensors.jsonl"
    out = []
    with fp.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def bootstrap_ci(values, boot=BOOT, seed=RNG_SEED):
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    pts = []
    for _ in range(boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        pts.append(statistics.mean(sample))
    pts.sort()
    return (statistics.mean(values), pts[int(0.025 * boot)], pts[int(0.975 * boot)])


def main():
    rows_per_step: List[dict] = []
    per_method: Dict[str, dict] = {}

    for method in METHODS:
        tensors = load_tensors(method)
        n_steps = len(tensors)
        per_prompt_restore = []
        per_prompt_midrange = []
        k_dist = {i: 0 for i in range(9)}
        n_degenerate = 0
        for step_rec in tensors:
            step_restore, step_midrange, step_degen = [], [], 0
            for rewards in step_rec["rewards"]:
                k = int(round(sum(rewards)))
                k_dist[k] += 1
                rp, mr = restore_prob(k), midrange_prob(k)
                step_restore.append(rp); step_midrange.append(mr)
                per_prompt_restore.append(rp); per_prompt_midrange.append(mr)
                if k == 0 or k == 8:
                    step_degen += 1
            n_degenerate += step_degen
            rows_per_step.append({
                "method": method, "step": step_rec["step"],
                "n_prompts": len(step_rec["rewards"]), "n_degenerate": step_degen,
                "zvf_obs": step_rec["zvf"], "reward_mean": step_rec["reward_mean"],
                "mean_restore_prob": statistics.mean(step_restore),
                "mean_midrange_prob": statistics.mean(step_midrange),
                "n_k0": k_dist[0], "n_k8": k_dist[8],
            })
        # Controller A: Bayesian at tau_post (per-prompt, currently degenerate)
        restore_bayes_pareto = {}
        for tau_post in (0.60, 0.65, 0.70, 0.80, 0.90):
            fires = 0
            saved_restore_sum = 0.0
            for step_rec in tensors:
                for rewards in step_rec["rewards"]:
                    k = int(round(sum(rewards)))
                    if k != 0 and k != 8:
                        continue
                    if midrange_prob(k) > tau_post:
                        fires += 1
                        saved_restore_sum += restore_prob(k)
            restore_bayes_pareto[tau_post] = {
                "fires": fires,
                "expected_restore_sum": saved_restore_sum,
            }
        # Controller B: zvf-triage step-level (escalate all 16 prompts in step)
        restore_zvf_triage = {}
        for tau in (0.50, 0.70, 0.90):
            fires = 0
            saved_restore_sum = 0.0
            for step_rec in tensors:
                if step_rec["zvf"] >= tau and step_rec["pcd"] <= 0.20:
                    fires += 1
                    for rewards in step_rec["rewards"]:
                        k = int(round(sum(rewards)))
                        saved_restore_sum += restore_prob(k)
            restore_zvf_triage[tau] = {"fires": fires, "expected_restore_sum": saved_restore_sum}
        # Controller C: Dualformer-Auto (Berkeley row 01: shrink when boundary)
        fires = 0
        saved_restore_sum = 0.0
        for step_rec in tensors:
            for rewards in step_rec["rewards"]:
                k = int(round(sum(rewards)))
                if k / 8.0 <= 0.125 or k / 8.0 >= 0.875:
                    fires += 1
                    saved_restore_sum += restore_prob(k)
        restore_dualformer = {"fires": fires, "expected_restore_sum": saved_restore_sum}

        per_method[method] = {
            "n_steps": n_steps,
            "n_degenerate_total": n_degenerate,
            "n_prompts_total": n_steps * 16,
            "mean_restore_prob_all": statistics.mean(per_prompt_restore),
            "mean_midrange_prob_all": statistics.mean(per_prompt_midrange),
            "k_distribution": k_dist,
            "bayes_tau_post": restore_bayes_pareto,
            "zvf_triage": restore_zvf_triage,
            "dualformer": restore_dualformer,
            "_all_restore": per_prompt_restore,
            "_all_midrange": per_prompt_midrange,
        }

    # ---------- write per-step TSV ----------
    per_step_fp = OUT_DIR / "p7_postpred_per_step.tsv"
    header = "method\tstep\tn_prompts\tn_degenerate\tzvf_obs\treward_mean\tmean_restore_prob\tmean_midrange_prob\tn_k0\tn_k8\n"
    lines = [header]
    for r in rows_per_step:
        lines.append(f"{r['method']}\t{r['step']}\t{r['n_prompts']}\t{r['n_degenerate']}\t{r['zvf_obs']:.4f}\t{r['reward_mean']:.4f}\t{r['mean_restore_prob']:.4f}\t{r['mean_midrange_prob']:.4f}\t{r['n_k0']}\t{r['n_k8']}\n")
    per_step_fp.write_text("".join(lines))
    print(f"wrote {per_step_fp}")

    # ---------- summary ----------
    summary = {
        "evidence_base": "N2 four-method reward tensors, 40 steps x 4 methods x 16 prompts = 2560 prompt-step obs",
        "method": "Beta-Binomial posterior predictive under Beta(1,1) prior",
        "tau_post_test_set": [0.60, 0.65, 0.70, 0.80, 0.90],
        "per_method": {},
        "iter15_sweep_estimate": {
            "delta_zvf_8_to_16": 0.0594,
            "ci": [0.0463, 0.0725],
            "evidence_base": "Qwen/Qwen3.5-4B GSM8K groupsize_zvf_sweep, n=3 seeds",
        },
    }
    for method in METHODS:
        pm = per_method[method]
        m_restore, lo_restore, hi_restore = bootstrap_ci(pm["_all_restore"])
        m_mid, lo_mid, hi_mid = bootstrap_ci(pm["_all_midrange"])
        cevs = {}
        for tau_post in (0.60, 0.65, 0.70, 0.80, 0.90):
            ev = pm["bayes_tau_post"][tau_post]
            rpf = round(ev["expected_restore_sum"] / ev["fires"], 4) if ev["fires"] else 0.0
            cevs[f"bayes_tau_post_{tau_post}"] = {"fires": ev["fires"], "expected_restore_sum": round(ev["expected_restore_sum"], 2), "restore_per_fire": rpf}
        for tau in (0.50, 0.70, 0.90):
            ev = pm["zvf_triage"][tau]
            rpf = round(ev["expected_restore_sum"] / ev["fires"], 4) if ev["fires"] else 0.0
            cevs[f"zvf_triage_tau_{tau}"] = {"fires": ev["fires"], "expected_restore_sum": round(ev["expected_restore_sum"], 2), "restore_per_fire": rpf}
        ev = pm["dualformer"]
        rpf = round(ev["expected_restore_sum"] / ev["fires"], 4) if ev["fires"] > 0 else 0.0
        cevs["dualformer_auto"] = {"fires": ev["fires"], "expected_restore_sum": round(ev["expected_restore_sum"], 2), "restore_per_fire": rpf}
        summary["per_method"][method] = {
            "n_steps": pm["n_steps"], "n_prompts_total": pm["n_prompts_total"],
            "n_degenerate_total": pm["n_degenerate_total"],
            "mean_restore_prob_all": round(m_restore, 4),
            "mean_restore_prob_ci": [round(lo_restore, 4), round(hi_restore, 4)],
            "mean_midrange_prob_all": round(m_mid, 4),
            "mean_midrange_prob_ci": [round(lo_mid, 4), round(hi_mid, 4)],
            "k_distribution": pm["k_distribution"],
            "controller_evaluations": cevs,
        }

    # ---------- write summary JSON ----------
    summary_fp = OUT_DIR / "p7_postpred_summary.json"
    with summary_fp.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {summary_fp}")

    # ---------- write summary TSV ----------
    summary_tsv_fp = OUT_DIR / "p7_postpred_summary.tsv"
    rows = ["method\tcontroller\ttau\tfires\texpected_restore_sum\trestore_per_fire"]
    for method in METHODS:
        pm = per_method[method]
        for tau_post in (0.60, 0.65, 0.70, 0.80, 0.90):
            ev = pm["bayes_tau_post"][tau_post]
            rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
            rows.append(f"{method}\tbayes\t{tau_post}\t{ev['fires']}\t{ev['expected_restore_sum']:.2f}\t{rpf:.4f}")
        for tau in (0.50, 0.70, 0.90):
            ev = pm["zvf_triage"][tau]
            rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
            rows.append(f"{method}\tzvf_triage\t{tau}\t{ev['fires']}\t{ev['expected_restore_sum']:.2f}\t{rpf:.4f}")
        ev = pm["dualformer"]
        rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
        rows.append(f"{method}\tdualformer\t-\t{ev['fires']}\t{ev['expected_restore_sum']:.2f}\t{rpf:.4f}")
    summary_tsv_fp.write_text("\n".join(rows) + "\n")
    print(f"wrote {summary_tsv_fp}")

    # ---------- console headline ----------
    print("\n=== HEADLINE (iter 19, P7 vein e) ===")
    print("Evidence base: N2 four-method tensors, 2560 prompt-step obs")
    print("Method: Beta-Binomial posterior predictive under Beta(1,1) prior\n")
    print(f"{'method':<8}{'mean_restore':>14}{'95% CI':>22}{'degen':>8}")
    for method in METHODS:
        m_restore, lo, hi = bootstrap_ci(per_method[method]["_all_restore"])
        print(f"{method:<8}{m_restore:>14.4f}  [{lo:.4f}, {hi:.4f}]   {per_method[method]['n_degenerate_total']:>8}")

    print("\n=== Controller compare: expected restore per fire (per method) ===")
    print(f"{'method':<8}{'controller':<14}{'tau':>6}{'fires':>8}{'restore_per_fire':>20}")
    for method in METHODS:
        pm = per_method[method]
        for tau_post in (0.60, 0.65, 0.70, 0.80, 0.90):
            ev = pm["bayes_tau_post"][tau_post]
            rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
            print(f"{method:<8}{'bayes':<14}{tau_post:>6.2f}{ev['fires']:>8}{rpf:>20.4f}")
        for tau in (0.50, 0.70, 0.90):
            ev = pm["zvf_triage"][tau]
            rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
            print(f"{method:<8}{'zvf_triage':<14}{tau:>6.2f}{ev['fires']:>8}{rpf:>20.4f}")
        ev = pm["dualformer"]
        rpf = (ev["expected_restore_sum"] / ev["fires"]) if ev["fires"] > 0 else 0.0
        print(f"{method:<8}{'dualformer':<14}{'-':>6}{ev['fires']:>8}{rpf:>20.4f}")
        print()


if __name__ == "__main__":
    main()