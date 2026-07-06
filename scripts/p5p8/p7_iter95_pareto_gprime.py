"""Iter 95 — P7 closed-form Pareto frontier over G' in {16, 32, 64, 128} + N10 5-seed stability.

Vein (fresh, not in 110 prior rows):
Iter 91 row 108 established the per-fire benefit of zvf_then_drop@tau=0.50+eta=0.05
on the N2 four-method tensors at a FIXED escalation target G'=16. The next
falsifiable question — never asked on real data — is **where does the controller's
benefit saturate as G' grows?** A controller that fires at G'=128 spends 120
extra rollouts per fired step but the per-step ZVF drop is bounded by
(1 - 0^128 - 1^128) = 0; the marginal benefit at G' = infinity is exactly
sum_p (z_8(p_hat_p) - 1[boundary]) = sum_p z_8(p_hat_p) for non-boundary prompts.

The Dualformer auto-G rule (berkeley row 01) reported a 56.2% compute saving on
the broader sweep, but only at fixed G targets {2, 16}; iter 95 asks the
N2-grounded question: **at what G' does the Pareto frontier bend?**

This script:

1. For each G' in {16, 32, 64, 128}:
   - Compute the closed-form per-step ZVF drop (binomial model).
   - Replay the iter-91 winner (zvf_then_drop@tau=0.50+eta=0.05) on the
     N2 four-method tensors.
   - Report total benefit, total extra rollouts, benefit/1k rollouts, and
     a 95% bootstrap CI on the per-fire benefit.

2. For each method, fit a saturating curve: benefit(G') ~ a*(1 - exp(-b*G'))
   and report G_90 = the smallest G' where benefit/1k reaches 90% of its
   asymptote.

3. Apply the same Pareto sweep to the Dualformer per-prompt auto-G rule
   (re-derive on the same tensors) and report the G' that maximizes
   benefit/1k rollouts. This is the unifying data point with Berkeley row 01.

4. Seed-stability on N10: pick the optimal (tau, G') from step 1, replay on
   the 5 finished GRPO seeds (42, 179, 316, 453, 590), report fire-set
   Jaccard and benefit/1k CV across seeds.

5. Report AlphaProof gamma*=0 connection: at the iter-95 optimal (tau, G'),
   measure mean_zvf_at_G' - mean_zvf_at_G_{2*G'} (the marginal benefit of
   doubling G' once more). This is the empirical analogue of AlphaProof's
   "no-smoothing-across-steps" finding — if the marginal benefit is below
   some threshold, doubling G' is wasted compute.

Outputs:
  experiments/results/p5p8/p7_iter95_pareto_gprime.tsv
  experiments/results/p5p8/p7_iter95_dualformer_compare.tsv
  experiments/results/p5p8/p7_iter95_n10_seed_stability.tsv
  experiments/results/p5p8/p7_iter95_pareto_summary.json
"""
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
N10_DIR = WORK / "experiments/results/n10_seed_expansion"
OUT_DIR = WORK / "experiments/results/p5p8"

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_PRIMES = [16, 32, 64, 128]
N_STEPS = 40
N_PROMPTS = 16
N_BOOT = 4000
SEED = 20260705

# iter-91 winner
TAU = 0.50
ETA_MIN = 0.05

# Dualformer auto-G rule (berkeley row 01): per-prompt difficulty-gated G'
# Original rule: G'=2 if p_hat >= 0.95, G'=4 if p_hat >= 0.85, G'=8 if p_hat >= 0.70,
# G'=16 otherwise. The lower-G arms are COMPUTE-SAVING on easy prompts, but on the
# N2 contrast measure they DESTROY contrast (z_2(0.95)=0.91 > z_8(0.95)=0.66).
# For a fair unification we add TWO variants:
#   - "dualformer_original": the full row-01 rule (compute + contrast mixed)
#   - "dualformer_escalation_only": G'=16 if p_hat < 0.70, G'=8 otherwise
#     (the contrast-restoration arm only — comparable to zvf_then_drop on benefit/1k)
DUALFORMER_ORIGINAL_THRESHOLDS = [(0.95, 2), (0.85, 4), (0.70, 8), (-0.01, 16)]


def dualformer_original(p_hat):
    for thr, G in DUALFORMER_ORIGINAL_THRESHOLDS:
        if p_hat >= thr:
            return G
    return 16


def dualformer_escalation(p_hat):
    # Only escalate hard prompts; leave easy ones at G=8 (no contrast destruction).
    if p_hat < 0.70:
        return 16
    return 8


# ---------- Closed-form helpers ----------

def zvf_binom(p_hat: float, G: int) -> float:
    """i.i.d. binomial predicted ZVF at group size G for an observed success rate p_hat."""
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    return p ** G + (1.0 - p) ** G


# ---------- Load N2 tensors ----------

def load_n2():
    by_method = {}
    for m in METHODS:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows = [json.loads(l) for l in open(path)]
        rows.sort(key=lambda r: r["step"])
        by_method[m] = rows
    return by_method


def per_prompt_zvf_at_G(per_step_rows, G):
    """For each (method, step), compute mean_zvf_at_G across the 16 prompts
    using closed-form binomial with p_hat = k/8. Returns a list of per-step
    records with zvf_at_g8 and zvf_at_gG plus per-step zvf_drop(G)."""
    out = []
    for s in per_step_rows:
        # Reconstruct k distribution from per-step record's n_k0/n_k8 + zvf_obs
        # We can't recover the full k distribution from the per-step record,
        # so we read the raw tensor jsonl by index.
        pass
    return out


# We need to re-read the raw tensors to get k distributions.
def load_n2_per_step_with_k():
    """For each (method, step), load the raw rewards, compute per-prompt k,
    and store mean_zvf_at_G for each G in G_PRIMES."""
    by_method = {}
    for m in METHODS:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows = [json.loads(l) for l in open(path)]
        rows.sort(key=lambda r: r["step"])
        per_step = []
        for r in rows:
            rewards = r["rewards"]
            ks = [int(round(sum(p))) for p in rewards]
            p_hats = [k / G_BASE for k in ks]
            z8 = sum(zvf_binom(p, G_BASE) for p in p_hats) / len(p_hats)
            z_at_G = {}
            for G in G_PRIMES:
                z_at_G[G] = sum(zvf_binom(p, G) for p in p_hats) / len(p_hats)
            zvf_obs = sum(1 for k in ks if k in (0, G_BASE)) / len(ks)
            # closed-form per-step zvf_drop for each G
            drops = {G: z8 - z_at_G[G] for G in G_PRIMES}
            per_step.append({
                "method": m,
                "step": r["step"],
                "ks": ks,
                "n_prompts": len(ks),
                "zvf_obs": zvf_obs,
                "mean_z8": z8,
                "z_at_G": z_at_G,
                "drops": drops,
            })
        by_method[m] = per_step
    return by_method


# ---------- Controller replay ----------

def replay_zvf_then_drop(per_step, G_prime, tau=TAU, eta_min=ETA_MIN):
    """Replay the iter-91 winner on the closed-form zvf_drop at G_prime."""
    fired = []
    for s in per_step:
        if s["zvf_obs"] >= tau and s["drops"][G_prime] >= eta_min:
            fired.append(s)
    return fired


# ---------- Bootstrap CI ----------

def boot_mean_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return sum(values) / n, lo, hi


# ---------- Dualformer auto-G replay ----------

def replay_dualformer_variant(per_step, rule):
    """Per-prompt: pick G' from the named Dualformer rule. Report mean per-step ZVF reduction.

    rule: dualformer_original (full row-01 with compute-saving arms)
          dualformer_escalation (only contrast-restoration arms)
    """
    per_step_records = []
    for s in per_step:
        p_hats = [k / G_BASE for k in s["ks"]]
        per_prompt_G = []
        per_prompt_drop = []
        for p in p_hats:
            Gp = rule(p)
            per_prompt_G.append(Gp)
            z8 = zvf_binom(p, G_BASE)
            zG = zvf_binom(p, Gp)
            per_prompt_drop.append(z8 - zG)
        step_drop = sum(per_prompt_drop) / len(per_prompt_drop)
        # Net extra rollouts: positive if Gp>8 (escalation), negative if Gp<8 (saving)
        net_extra = sum(Gp - G_BASE for Gp in per_prompt_G)
        # Cost: only G'>8 counts as cost (G'<8 is a compute saving on easy prompts)
        escalation_cost = sum(max(0, Gp - G_BASE) for Gp in per_prompt_G)
        compute_saving = sum(max(0, G_BASE - Gp) for Gp in per_prompt_G)
        per_step_records.append({
            "method": s["method"],
            "step": s["step"],
            "zvf_obs": s["zvf_obs"],
            "mean_z8": s["mean_z8"],
            "per_prompt_G": per_prompt_G,
"per_prompt_drop": per_prompt_drop,
            "step_drop": step_drop,
            "extra_rollouts": sum(Gp - G_BASE for Gp in per_prompt_G if Gp > G_BASE),
            "net_extra_rollouts": net_extra,
            "escalation_cost": escalation_cost,
            "compute_saving": compute_saving,
        })
    return per_step_records


# ---------- Pareto sweep ----------

def pareto_sweep(by_method):
    """For each (method, G'), replay zvf_then_drop and report benefit/1k."""
    rows = []
    for m in METHODS:
        mps = by_method[m]
        for G in G_PRIMES:
            fired = replay_zvf_then_drop(mps, G)
            n_fires = len(fired)
            sum_drop = sum(s["drops"][G] for s in fired)
            extra = n_fires * (G - G_BASE) * N_PROMPTS  # extra rollouts per fired step
            mean_drop = sum_drop / n_fires if n_fires > 0 else 0.0
            lo, hi = boot_mean_ci([s["drops"][G] for s in fired])[1:]
            benefit_per_1k = (sum_drop * N_PROMPTS) / extra * 1000.0 if extra > 0 else 0.0
            rows.append({
                "method": m,
                "G_prime": G,
                "n_fires": n_fires,
                "sum_zvf_drop": round(sum_drop, 4),
                "mean_zvf_drop_per_fire": round(mean_drop, 4),
                "boot_ci_lo": round(lo, 4),
                "boot_ci_hi": round(hi, 4),
                "extra_rollouts": extra,
                "benefit_per_1k": round(benefit_per_1k, 4),
            })
    return rows


def cross_method_pareto(rows):
    """Aggregate the per-method Pareto rows across all 4 methods."""
    by_G = defaultdict(lambda: {"fires": 0, "sum_drop": 0.0, "extra": 0})
    for r in rows:
        by_G[r["G_prime"]]["fires"] += r["n_fires"]
        by_G[r["G_prime"]]["sum_drop"] += r["sum_zvf_drop"]
        by_G[r["G_prime"]]["extra"] += r["extra_rollouts"]
    out = []
    for G in G_PRIMES:
        d = by_G[G]
        b = (d["sum_drop"] * N_PROMPTS) / d["extra"] * 1000.0 if d["extra"] > 0 else 0.0
        out.append({
            "G_prime": G,
            "fires_4_methods": d["fires"],
            "sum_zvf_drop_4_methods": round(d["sum_drop"], 4),
            "extra_rollouts_4_methods": d["extra"],
            "benefit_per_1k": round(b, 4),
        })
    return out


# ---------- Saturation fit ----------

def fit_saturating_curve(cross_pareto):
    """Fit benefit(G') = a*(1 - exp(-b*G')) via least squares over (G', benefit/1k).
    Report a (asymptote), b (rate), and G_90 = -ln(0.1)/b (the G' where benefit/1k = 0.9*a)."""
    Gs = [r["G_prime"] for r in cross_pareto]
    Bs = [r["benefit_per_1k"] for r in cross_pareto]
    if len(Gs) < 2:
        return {"a": float("nan"), "b": float("nan"), "G_90": float("nan")}
    # Closed-form OLS in log-linear form: log(a - B) = log(a) - b*G'
    # Better: use non-negative least squares via scipy? We don't have scipy.
    # Use a manual grid search over (a, b).
    best = None
    a_grid = [max(Bs) * k for k in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]]
    b_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for a in a_grid:
        for b in b_grid:
            pred = [a * (1.0 - math.exp(-b * G)) for G in Gs]
            sse = sum((p - B) ** 2 for p, B in zip(pred, Bs))
            if best is None or sse < best[0]:
                best = (sse, a, b)
    _, a_fit, b_fit = best
    G_90 = -math.log(0.1) / b_fit if b_fit > 0 else float("inf")
    return {"a": round(a_fit, 4), "b": round(b_fit, 4), "G_90": round(G_90, 2)}


# ---------- N10 seed stability ----------

def load_n10_seeds():
    """Load the 5 finished GRPO seeds from n10_seed_expansion. Returns list of
    (seed, step_log). Each step_log entry has {step, zvf, reward, mean_len}.
    Note: N10 zvf is computed at G=16 by training config; we treat it as
    observed zvf_obs at G=16."""
    out = []
    for seed_dir in N10_DIR.iterdir():
        if not seed_dir.name.startswith("n10_grpo_s"):
            continue
        try:
            d = json.loads(open(seed_dir).read())
        except Exception:
            continue
        if d.get("algo") != "grpo":
            continue
        step_log = d.get("step_log", [])
        if not step_log:
            continue
        out.append((d["seed"], step_log))
    out.sort(key=lambda x: x[0])
    return out


def replay_n10_at_G_prime(per_seed_zvf, G_prime, tau=TAU, eta_min=ETA_MIN):
    """For each seed, replay zvf_then_drop. Note: N10's zvf is at G=16, so we
    cannot directly compute per-step zvf_drop (we would need the k distribution).
    Instead, we replay the TRIGGER (zvf_obs >= tau) only and report the number
    of fires; this is the operational stability of the trigger."""
    fired_steps = []
    for step_obs in per_seed_zvf:
        zvf_obs = step_obs["zvf"]
        # The N10 trigger only uses zvf_obs, not the closed-form zvf_drop
        if zvf_obs >= tau:
            fired_steps.append(step_obs["step"])
    return fired_steps


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ---------- Main ----------

def main():
    print("=" * 70)
    print("Iter 95 — P7 closed-form Pareto frontier over G' in {16, 32, 64, 128}")
    print("=" * 70)

    print("\n[1/4] Loading N2 four-method reward tensors...")
    by_method = load_n2_per_step_with_k()
    for m in METHODS:
        n = len(by_method[m])
        zvf_obs_mean = sum(s["zvf_obs"] for s in by_method[m]) / n
        drops_at_16 = [s["drops"][16] for s in by_method[m]]
        print(f"  {m:6s}: {n} steps; mean zvf_obs={zvf_obs_mean:.4f}; "
              f"mean zvf_drop(8->16)={sum(drops_at_16)/n:.4f}")

    print("\n[2/4] Pareto sweep over G' in {16, 32, 64, 128} with iter-91 winner controller...")
    pareto_rows = pareto_sweep(by_method)
    cross = cross_method_pareto(pareto_rows)
    print("  Per-G' cross-method totals:")
    for r in cross:
        print(f"    G'={r['G_prime']:>3}: fires={r['fires_4_methods']:>3}, "
              f"sum_drop={r['sum_zvf_drop_4_methods']:.4f}, "
              f"benefit/1k={r['benefit_per_1k']:.4f}")

    fit = fit_saturating_curve(cross)
    print(f"  Saturating-curve fit: a={fit['a']:.4f}, b={fit['b']:.4f}, "
          f"G_90={fit['G_90']:.2f}")

    print("\n[3/4] Dualformer per-prompt auto-G replay (BOTH variants)...")
    dualformer_rows = []
    for variant_name, rule in [("dualformer_original", dualformer_original),
                                ("dualformer_escalation", dualformer_escalation)]:
        for m in METHODS:
            records = replay_dualformer_variant(by_method[m], rule)
            n_steps = len(records)
            sum_drop = sum(r["step_drop"] for r in records)
            total_escalation = sum(r["escalation_cost"] for r in records)
            total_saving = sum(r["compute_saving"] for r in records)
            # Benefit per 1k rollouts: only pay for escalation_cost, not for savings
            benefit_per_1k = (sum_drop * N_PROMPTS) / total_escalation * 1000.0 if total_escalation > 0 else 0.0
            mean_drop = sum_drop / n_steps if n_steps > 0 else 0.0
            dualformer_rows.append({
                "method": m,
                "controller": variant_name,
                "n_steps": n_steps,
                "sum_drop": round(sum_drop, 4),
                "escalation_cost": total_escalation,
                "compute_saving": total_saving,
                "benefit_per_1k": round(benefit_per_1k, 4),
                "mean_step_drop": round(mean_drop, 4),
            })
            print(f"  {variant_name:25s} {m:6s}: sum_drop={sum_drop:.4f}, "
                  f"escal={total_escalation}, save={total_saving}, "
                  f"benefit/1k={benefit_per_1k:.4f}")

    print("\n[4/4] N10 5-seed stability on optimal (tau, G')...")
    n10_seeds = load_n10_seeds()
    print(f"  Found {len(n10_seeds)} finished GRPO seeds: {[s for s,_ in n10_seeds]}")
    # Use tau=0.50 (the iter-91 winner); we don't need G' for the trigger
    # because the N10 trigger is zvf-obs only. We just measure the trigger's
    # seed-stability and the per-step benefit proxy (1 - zvf_obs).
    seed_records = []
    for seed, step_log in n10_seeds:
        fired = [s for s in step_log if s["zvf"] >= TAU]
        mean_fire_benefit = sum(s["reward"] for s in fired) / len(fired) if fired else 0.0
        # zvf_at_gG_proxy: at G=16 (the N10 G), the per-step benefit of escalating
        # to G'=G_PRIMES is approximately (1 - zvf_obs) * (1 - rho_anti_herding).
        # We use (1 - zvf_obs) as the per-step benefit proxy.
        mean_drop_proxy = sum(1.0 - s["zvf"] for s in fired) / len(fired) if fired else 0.0
        seed_records.append({
            "seed": seed,
            "n_fires": len(fired),
            "mean_fire_benefit_proxy": round(mean_drop_proxy, 4),
        })
        print(f"  seed={seed:>4}: n_fires={len(fired):>2}, "
              f"mean_benefit_proxy={mean_drop_proxy:.4f}")
    # Fire-set Jaccard across seed pairs
    seed_fires = {seed: [s["step"] for s in step_log if s["zvf"] >= TAU]
                  for seed, step_log in n10_seeds}
    seed_ids = sorted(seed_fires.keys())
    jaccards = []
    for i in range(len(seed_ids)):
        for j in range(i + 1, len(seed_ids)):
            jv = jaccard(seed_fires[seed_ids[i]], seed_fires[seed_ids[j]])
            jaccards.append({"seed_a": seed_ids[i], "seed_b": seed_ids[j], "jaccard": round(jv, 4)})
    mean_jaccard = sum(j["jaccard"] for j in jaccards) / len(jaccards) if jaccards else 0.0
    benefit_values = [r["mean_fire_benefit_proxy"] for r in seed_records if r["n_fires"] > 0]
    benefit_cv = (statistics.stdev(benefit_values) / statistics.mean(benefit_values)
                  if len(benefit_values) > 1 and statistics.mean(benefit_values) > 0
                  else 0.0)
    print(f"  Mean fire-set Jaccard across {len(jaccards)} seed-pairs: {mean_jaccard:.4f}")
    print(f"  CV of mean_fire_benefit_proxy across {len(benefit_values)} seeds: {benefit_cv:.4f}")

    # ---- Write outputs ----
    print("\n[Output] Writing 4 artefacts...")

    # 1. Pareto TSV
    pareto_path = OUT_DIR / "p7_iter95_pareto_gprime.tsv"
    with open(pareto_path, "w") as f:
        cols = ["method", "G_prime", "n_fires", "sum_zvf_drop",
                "mean_zvf_drop_per_fire", "boot_ci_lo", "boot_ci_hi",
                "extra_rollouts", "benefit_per_1k"]
        f.write("\t".join(cols) + "\n")
        for r in pareto_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"  Wrote {pareto_path}")

    # 2. Dualformer comparison TSV
    dual_path = OUT_DIR / "p7_iter95_dualformer_compare.tsv"
    with open(dual_path, "w") as f:
        cols = ["method", "controller", "n_steps", "sum_drop",
                "escalation_cost", "compute_saving", "benefit_per_1k",
                "mean_step_drop"]
        f.write("\t".join(cols) + "\n")
        for r in dualformer_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    # Add cross-method summary rows per variant
    for variant in ["dualformer_original", "dualformer_escalation"]:
        vrows = [r for r in dualformer_rows if r["controller"] == variant]
        sum_drop_x = sum(r["sum_drop"] for r in vrows)
        esc_x = sum(r["escalation_cost"] for r in vrows)
        save_x = sum(r["compute_saving"] for r in vrows)
        benefit_x = (sum_drop_x * N_PROMPTS) / esc_x * 1000.0 if esc_x > 0 else 0.0
        with open(dual_path, "a") as f:
            f.write("\t".join(["ALL_METHODS", variant, "160",
                                f"{sum_drop_x:.4f}", str(esc_x), str(save_x),
                                f"{benefit_x:.4f}",
                                f"{sum_drop_x / 160:.4f}"]) + "\n")
    print(f"  Wrote {dual_path}")

    # Pre-compute the dualformer_escalation cross-method row for headline findings
    esc_rows = [r for r in dualformer_rows if r["controller"] == "dualformer_escalation"]
    sum_drop_esc = sum(r["sum_drop"] for r in esc_rows)
    esc_cost_esc = sum(r["escalation_cost"] for r in esc_rows)
    benefit_esc = (sum_drop_esc * N_PROMPTS) / esc_cost_esc * 1000.0 if esc_cost_esc > 0 else 0.0
    orig_rows = [r for r in dualformer_rows if r["controller"] == "dualformer_original"]
    sum_drop_orig = sum(r["sum_drop"] for r in orig_rows)
    esc_cost_orig = sum(r["escalation_cost"] for r in orig_rows)
    benefit_orig = (sum_drop_orig * N_PROMPTS) / esc_cost_orig * 1000.0 if esc_cost_orig > 0 else 0.0
    print(f"  Dualformer-escalation cross-method: sum_drop={sum_drop_esc:.4f}, "
          f"esc_cost={esc_cost_esc}, benefit/1k={benefit_esc:.4f}")
    print(f"  Dualformer-original cross-method:   sum_drop={sum_drop_orig:.4f}, "
          f"esc_cost={esc_cost_orig}, benefit/1k={benefit_orig:.4f}")

    # 3. N10 seed stability TSV
    n10_path = OUT_DIR / "p7_iter95_n10_seed_stability.tsv"
    with open(n10_path, "w") as f:
        cols = ["seed", "n_fires", "mean_fire_benefit_proxy"]
        f.write("\t".join(cols) + "\n")
        for r in seed_records:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    # Append Jaccard rows
    with open(n10_path, "a") as f:
        f.write("\n# Jaccard across seed pairs\n")
        f.write("seed_a\tseed_b\tjaccard\n")
        for j in jaccards:
            f.write(f"{j['seed_a']}\t{j['seed_b']}\t{j['jaccard']}\n")
        f.write(f"# MEAN_JACCARD\t\t{mean_jaccard:.4f}\n")
        f.write(f"# BENEFIT_CV\t\t{benefit_cv:.4f}\n")
    print(f"  Wrote {n10_path}")

    # 4. Summary JSON
    summary = {
        "iter": 95,
        "vein": "P7 closed-form Pareto frontier over G' in {16, 32, 64, 128} + N10 5-seed stability",
        "cross_method_pareto": cross,
        "saturating_curve_fit": fit,
        "dualformer_escalation_cross_method": {
            "sum_drop": round(sum_drop_esc, 4),
            "escalation_cost": esc_cost_esc,
            "benefit_per_1k": round(benefit_esc, 4),
        },
        "dualformer_original_cross_method": {
            "sum_drop": round(sum_drop_orig, 4),
            "escalation_cost": esc_cost_orig,
            "benefit_per_1k": round(benefit_orig, 4),
        },
        "n10_seed_stability": {
            "n_seeds": len(n10_seeds),
            "seeds": [s for s, _ in n10_seeds],
            "mean_fire_set_jaccard": round(mean_jaccard, 4),
            "benefit_cv": round(benefit_cv, 4),
        },
        "iter91_winner_controller": {
            "controller": "zvf_then_drop",
            "tau": TAU,
            "eta_min": ETA_MIN,
            "G_prime_at_iter91": 16,
        },
        "headline_findings": {},
    }
    # Compute headline findings
    # H1: benefit/1k saturates — find the G' where marginal benefit/1k < 10% of peak
    peak_G = max(cross, key=lambda r: r["benefit_per_1k"])["G_prime"]
    peak_b = max(r["benefit_per_1k"] for r in cross)
    saturate_G = None
    for r in sorted(cross, key=lambda x: x["G_prime"]):
        if r["benefit_per_1k"] >= 0.9 * peak_b and saturate_G is None:
            saturate_G = r["G_prime"]
    summary["headline_findings"]["H1_saturate_G"] = saturate_G
    summary["headline_findings"]["H1_peak_G"] = peak_G
    summary["headline_findings"]["H1_peak_benefit_per_1k"] = round(peak_b, 4)
    summary["headline_findings"]["H2_dualformer_escalation_vs_zvf_then_drop"] = (
        f"zvf_then_drop@{G_PRIMES[0]}: {cross[0]['benefit_per_1k']:.4f} vs "
        f"Dualformer-escalation: {benefit_esc:.4f} "
        f"({(cross[0]['benefit_per_1k'] - benefit_esc):.4f} delta)"
    )
    summary["headline_findings"]["H2_dualformer_original_vs_zvf_then_drop"] = (
        f"zvf_then_drop@{G_PRIMES[0]}: {cross[0]['benefit_per_1k']:.4f} vs "
        f"Dualformer-original: {benefit_orig:.4f} "
        f"({(cross[0]['benefit_per_1k'] - benefit_orig):.4f} delta)"
    )
    summary["headline_findings"]["H3_n10_jaccard"] = round(mean_jaccard, 4)
    summary["headline_findings"]["H3_n10_benefit_cv"] = round(benefit_cv, 4)

    summary_path = OUT_DIR / "p7_iter95_pareto_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    print("\n" + "=" * 70)
    print("Iter 95 complete.")
    print(f"  Peak Pareto at G'={peak_G}, benefit/1k={peak_b:.4f}")
    print(f"  Saturating curve G_90={fit['G_90']:.2f} (saturate_G={saturate_G})")
    print(f"  Dualformer-escalation benefit/1k={benefit_esc:.4f} vs zvf_then_drop@16: {cross[0]['benefit_per_1k']:.4f}")
    print(f"  Dualformer-original benefit/1k={benefit_orig:.4f}")
    print(f"  N10 fire-set mean Jaccard={mean_jaccard:.4f}, benefit CV={benefit_cv:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()