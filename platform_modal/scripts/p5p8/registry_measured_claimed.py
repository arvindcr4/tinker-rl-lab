#!/usr/bin/env python3
"""Iter 18 P6 (Pillar 2) — measured-vs-claimed variant-delta reconciliation.

For each of the 11 variant-delta records in registry/entries/delta_*.json:

  (a) Read the *claimed* components (loss_form.*, reference_kl.*, etc.).
  (b) Pull measured proxies from:
      - N2 same-stack four-method tensors (aero/gift/areal/grpo, n=10 steps)
        via experiments/results/p5p8/registry_measured_deltas.json
      - zvf_iter130 risk index (9 methods, 5 seeds)
        via experiments/results/zvf_iter130_risk_index.tsv
      - registry stack records that *claim* the delta
        via registry/entries/<stack>_*.json
  (c) For each measurable proxy, decide:
        SUPPORT   — measured sign matches the predicted sign AND |Δ| is large
                    enough to be scientifically interesting (effect_size >=
                    paired-bootstrap floor)
        WEAK      — measured sign matches but |Δ| is within paired noise
        OPPOSE    — measured sign disagrees
        NO_DATA   — no measured proxy available for this claim

Writes:
  experiments/results/p5p8/registry_measured_claimed.tsv   — one row per delta
  experiments/results/p5p8/registry_measured_claimed.json  — machine-readable

Stdlib only. Run: python3 platform_modal/scripts/p5p8/registry_measured_claimed.py
"""

import csv
import json
import pathlib
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REG_ENTRIES = ROOT / "registry" / "entries"
N2_JSON = ROOT / "experiments" / "results" / "p5p8" / "registry_measured_deltas.json"
ZV130_TSV = ROOT / "experiments" / "results" / "zvf_iter130_risk_index.tsv"
OUT_TSV = ROOT / "experiments" / "results" / "p5p8" / "registry_measured_claimed.tsv"
OUT_JSON = ROOT / "experiments" / "results" / "p5p8" / "registry_measured_claimed.json"


def load_deltas():
    out = {}
    for p in sorted(REG_ENTRIES.glob("delta_*.json")):
        d = json.loads(p.read_text())
        out[d["id"]] = d
    return out


def load_n2():
    return json.loads(N2_JSON.read_text())


def load_zv130():
    rows = []
    with ZV130_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def n2_lookup(n2):
    """index measured_deltas_stepwise by (variant, metric)."""
    idx = {}
    for r in n2["measured_deltas_stepwise"]:
        idx[(r["variant"], r["metric"])] = r
    return idx


def zv130_per_method(zv130):
    """group risk-index rows by method; report mean_zvf across seeds + std."""
    grp = defaultdict(list)
    for r in zv130:
        if r["method"] in ("scaling_law_Qwen3.5-4B", "scaling_law_Llama-3.1-8B-Instruct",
                            "scaling_law_DeepSeek-V3.1", "scaling_law_Nemotron-120B",
                            "scaling_law_Qwen3-8B", "tool_use_qwen3-32b",
                            "tool_use_llama-8b-inst"):
            continue
        try:
            grp[r["method"]].append(float(r["mean_zvf"]))
        except (ValueError, KeyError):
            pass
    return {m: {"n": len(v), "mean_zvf": statistics.mean(v),
                "std_zvf": statistics.pstdev(v) if len(v) > 1 else 0.0,
                "min_zvf": min(v), "max_zvf": max(v)}
            for m, v in grp.items()}


def registry_field_check(deltas):
    """For each delta_id, look at stack records that *claim* the delta via
    variant_deltas_applied[*].delta_id == delta_id. Return the union of
    populated MIN-REPORT fields across those stack records (provenance of
    the registry-side claim)."""
    stacks_by_delta = defaultdict(list)
    for p in sorted(REG_ENTRIES.glob("*.json")):
        if p.name.startswith("delta_"):
            continue
        rec = json.loads(p.read_text())
        for vd in rec.get("variant_deltas_applied", []):
            stacks_by_delta[vd["delta_id"]].append({
                "stack_id": rec["id"],
                "status": vd["status"],
                "framework": rec.get("framework", {}).get("name"),
                "openness": rec.get("framework", {}).get("openness"),
                "label_claimed": rec.get("label_claimed"),
                "min_report": rec.get("min_report", {}),
            })
    return dict(stacks_by_delta)


def classify_n2(measured_delta, predicted_sign, ci_excludes_0):
    """Classify the measured N2 delta against a predicted qualitative sign.
    predicted_sign: +1, -1, or 0 (don't-care).
    Returns: SUPPORT/WEAK/OPPOSE/NO_DATA + reason."""
    if measured_delta is None:
        return "NO_DATA", "no N2 same-stack run for this method"
    d = measured_delta["paired_delta"]
    sign = 1 if d > 0 else (-1 if d < 0 else 0)
    excl = ci_excludes_0 == "yes"
    if predicted_sign == 0:
        return "WEAK", f"Δ={d:+.4g} (no predicted sign; within {('noise' if not excl else 'CI excludes 0')})"
    if sign == predicted_sign and excl:
        return "SUPPORT", f"Δ={d:+.4g} (CI excludes 0, sign matches predicted {predicted_sign:+d})"
    if sign == predicted_sign and not excl:
        return "WEAK", f"Δ={d:+.4g} (sign matches predicted {predicted_sign:+d} but CI contains 0)"
    if sign != predicted_sign and sign != 0 and excl:
        return "OPPOSE", f"Δ={d:+.4g} (CI excludes 0, sign OPPOSES predicted {predicted_sign:+d})"
    if sign != predicted_sign and sign != 0 and not excl:
        return "OPPOSE", f"Δ={d:+.4g} (sign OPPOSES predicted {predicted_sign:+d}; CI contains 0)"
    return "WEAK", f"Δ≈0 (within paired noise; predicted {predicted_sign:+d})"


# --- per-delta claim→proxy mapping ---
# Each entry maps a measurable proxy name to a (predicted_sign, predicted_reason)
# where predicted_sign is one of {+1, -1, 0}.
#   +1: variant claimed to INCREASE the proxy
#   -1: variant claimed to DECREASE the proxy
#   0: no clear directional claim

CLAIMS = {
    "delta_aero": {
        "summary": "off-policy reference rollouts inflate effective G; (le2025rlzvp)",
        "proxies": {
            "zvf":           (-1, "inflate effective G -> fewer zero-variance groups (predicted)"),
            "loss":          (0,  "no claim on loss magnitude"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "reward_mean":   (1,  "more samples + entropy guidance -> better reward (claimed)"),
        },
    },
    "delta_gift": {
        "summary": "gamma-style per-prompt likelihood prior subtracted from advantage (GIFT/UNA+DPO)",
        "proxies": {
            "loss":          (0,  "no claim; but a constant offset WILL shift loss magnitude"),
            "zvf":           (0,  "no claim on ZVF; constant offset cancels in std"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "reward_mean":   (0,  "no claim on reward"),
        },
    },
    "delta_areal": {
        "summary": "decouple rollout budget from optimizer step (single-batch same-stack run isolates label)",
        "proxies": {
            "zvf":           (0,  "single-batch static-G run: no rollout-vs-optimizer signal"),
            "reward_mean":   (0,  "single-batch run: no rollout-vs-optimizer signal"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss"),
        },
    },
    "delta_dapo": {
        "summary": "asymmetric clip + dynamic sampling + token-level loss + overlong-reward shaping + KL removed",
        "proxies": {
            "zvf":           (-1, "dynamic sampling zeroes degenerate groups -> lower ZVF"),
            "reward_mean":   (1,  "dynamic sampling + token-level loss -> better reward (claimed)"),
            "mean_len":      (-1, "overlong-reward shaping penalises long completions"),
            "loss":          (0,  "no claim on loss magnitude"),
            "cv_len":        (0,  "no claim on length variance"),
        },
    },
    "delta_drgrpo": {
        "summary": "remove length-normalization and advantage-std-normalization (Dr.GRPO)",
        "proxies": {
            "zvf":           (1,  "removing length norm exposes length bias -> more within-group variance"),
            "reward_mean":   (0,  "no claim on reward"),
            "mean_len":      (1,  "removing length norm -> longer completions (length bias recovered)"),
            "cv_len":        (1,  "removing length norm -> higher within-group length variance"),
            "loss":          (0,  "no claim on loss magnitude"),
        },
    },
    "delta_gspo": {
        "summary": "sequence-level importance ratio + sequence-level clip (GSPO)",
        "proxies": {
            "zvf":           (0,  "ratio level doesn't directly change reward-variance structure"),
            "reward_mean":   (0,  "no claim on reward"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss magnitude"),
        },
    },
    "delta_cppo": {
        "summary": "continuity penalty discourages large log-prob jumps (CPPO)",
        "proxies": {
            "zvf":           (0,  "no claim on ZVF; log-prob smoothness doesn't directly change within-group contrast"),
            "reward_mean":   (1,  "smoother optimisation -> better reward (claimed)"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss magnitude (penalty is a regulariser)"),
        },
    },
    "delta_ngrpo": {
        "summary": "normalize advantage by per-prompt gradient norm (NGraPO)",
        "proxies": {
            "zvf":           (0,  "no claim on ZVF; per-prompt norm re-weights but doesn't change zero-variance count"),
            "reward_mean":   (1,  "better advantage normalisation -> better reward (claimed)"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss"),
        },
    },
    "delta_mcgrpo": {
        "summary": "MCTS-augmented rollouts + per-prompt diversity bonus (MC-GRPO)",
        "proxies": {
            "zvf":           (0,  "MCTS boost could go either way; diversity bonus is meant to raise contrast"),
            "reward_mean":   (1,  "MCTS value guidance -> better reward (claimed)"),
            "mean_len":      (1,  "MCTS continuations are typically longer"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss"),
        },
    },
    "delta_es": {
        "summary": "replace policy-gradient with ES central-difference estimator (ES at scale)",
        "proxies": {
            "zvf":           (0,  "ES doesn't use within-group contrast -> ZVF irrelevant by construction"),
            "reward_mean":   (0,  "no claim on reward; estimator change not necessarily better"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "ES objective differs structurally from GRPO loss; not directly comparable"),
        },
    },
    "delta_scafgrpo": {
        "summary": "scaffold-completion-quality prior up-weights low-scaffold prompts (Scaf-GRPO)",
        "proxies": {
            "zvf":           (0,  "no direct claim on ZVF; re-weighting could go either way"),
            "reward_mean":   (1,  "up-weighting hard prompts -> better learning (claimed)"),
            "mean_len":      (0,  "no claim on length"),
            "cv_len":        (0,  "no claim on length variance"),
            "loss":          (0,  "no claim on loss"),
        },
    },
}

METHOD_OF = {  # delta_id -> method label used in N2 / zvf130
    "delta_aero":    "aero",
    "delta_gift":    "gift",
    "delta_areal":   "areal",
    "delta_dapo":    "dapo",
    "delta_drgrpo":  "drgrpo",
    "delta_gspo":    "gspo",
    "delta_cppo":    "cppo",
    "delta_ngrpo":   "ngrpo",
    "delta_mcgrpo":  "mcgrpo",
    "delta_es":      "es",
    "delta_scafgrpo":"scafgrpo",
}


def reconcile():
    deltas = load_deltas()
    n2 = load_n2()
    n2_idx = n2_lookup(n2)
    zv = zv130_per_method(load_zv130())
    stacks = registry_field_check(deltas)

    grpo_zv = zv.get("grpo", {}).get("mean_zvf", None)
    rows = []
    full = {"per_delta": [], "zvf130_per_method": zv, "n2_baseline_method": "grpo"}

    for delta_id in sorted(deltas.keys()):
        d = deltas[delta_id]
        method = METHOD_OF[delta_id]
        claim = CLAIMS.get(delta_id, {"summary": d["name"], "proxies": {}})
        # registry evidence
        claimers = stacks.get(delta_id, [])
        # N2 evidence
        n2_evidence = {}
        verdict_counts = defaultdict(int)
        for proxy, (pred_sign, pred_reason) in claim["proxies"].items():
            md = n2_idx.get((method, proxy))
            verdict, reason = classify_n2(md, pred_sign, md["ci_excludes_0"] if md else "no")
            verdict_counts[verdict] += 1
            n2_evidence[proxy] = {
                "predicted_sign": pred_sign,
                "predicted_reason": pred_reason,
                "measured_delta": md["paired_delta"] if md else None,
                "ci_lo": md["ci_lo"] if md else None,
                "ci_hi": md["ci_hi"] if md else None,
                "ci_excludes_0": md["ci_excludes_0"] if md else "no",
                "verdict": verdict,
                "reason": reason,
            }
        # zvf130 evidence
        zv_method = zv.get(method, None)
        zv_grpo = zv.get("grpo", None)
        zvf130_evidence = {}
        if zv_method is not None and zv_grpo is not None and grpo_zv is not None:
            delta_zv = zv_method["mean_zvf"] - grpo_zv
            pct_drop = (zv_method["mean_zvf"] - grpo_zv) / max(grpo_zv, 1e-6) * 100
            zvf130_evidence = {
                "method_mean_zvf": zv_method["mean_zvf"],
                "method_std_zvf": zv_method["std_zvf"],
                "method_n_seeds": zv_method["n"],
                "grpo_mean_zvf": grpo_zv,
                "delta_zvf": delta_zv,
                "pct_drop_vs_grpo": pct_drop,
                "verdict": ("ZVF_BELOW_GRPO" if delta_zv < 0
                            else "ZVF_AT_OR_ABOVE_GRPO"),
            }
        # registry evidence
        reg_evidence = {
            "n_claimers": len(claimers),
            "statuses": sorted({c["status"] for c in claimers}),
            "frameworks": sorted({c["framework"] for c in claimers if c["framework"]}),
            "opennesses": sorted({c["openness"] for c in claimers if c["openness"]}),
            "claimers": claimers,
        }
        # overall verdict
        if not n2_evidence:
            overall = "NO_DATA"
        elif verdict_counts.get("SUPPORT", 0) >= max(1, len(n2_evidence) // 2):
            overall = "SUPPORTED"
        elif verdict_counts.get("OPPOSE", 0) >= max(1, len(n2_evidence) // 2):
            overall = "OPPOSED"
        elif verdict_counts.get("SUPPORT", 0) > 0 or verdict_counts.get("WEAK", 0) > 0:
            overall = "MIXED"
        else:
            overall = "NULL"
        # row
        row = {
            "delta_id": delta_id,
            "method": method,
            "claim_summary": claim["summary"],
            "n2_verdict_counts": dict(verdict_counts),
            "n2_overall": overall,
            "n2_proxies_tested": len(n2_evidence),
            "n2_supports": verdict_counts.get("SUPPORT", 0),
            "n2_weak": verdict_counts.get("WEAK", 0),
            "n2_oppose": verdict_counts.get("OPPOSE", 0),
            "n2_no_data": verdict_counts.get("NO_DATA", 0),
            "zvf130_method_mean": (zv_method["mean_zvf"] if zv_method else ""),
            "zvf130_grpo_mean": grpo_zv if grpo_zv is not None else "",
            "zvf130_delta": zvf130_evidence.get("delta_zvf", ""),
            "zvf130_verdict": zvf130_evidence.get("verdict", "NO_DATA"),
            "registry_n_claimers": len(claimers),
            "registry_statuses": "|".join(reg_evidence["statuses"]),
            "registry_frameworks": "|".join(reg_evidence["frameworks"]),
            "registry_opennesses": "|".join(reg_evidence["opennesses"]),
        }
        rows.append(row)
        full["per_delta"].append({
            **row,
            "n2_evidence_per_proxy": n2_evidence,
            "zvf130_evidence": zvf130_evidence,
            "registry_evidence": reg_evidence,
        })

    # write TSV
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["delta_id", "method", "claim_summary",
            "n2_proxies_tested", "n2_supports", "n2_weak", "n2_oppose", "n2_no_data",
            "n2_overall", "n2_verdict_counts",
            "zvf130_method_mean", "zvf130_grpo_mean", "zvf130_delta", "zvf130_verdict",
            "registry_n_claimers", "registry_statuses", "registry_frameworks",
            "registry_opennesses"]
    with OUT_TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["n2_verdict_counts"] = json.dumps(r["n2_verdict_counts"], sort_keys=True)
            w.writerow(r)
    OUT_JSON.write_text(json.dumps(full, indent=2, sort_keys=True))
    # console summary
    print(f"wrote {OUT_TSV.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print("\nper-delta N2 verdict (SUPPORTED / MIXED / NULL / OPPOSED / NO_DATA):")
    for r in rows:
        print(f"  {r['delta_id']:20s} method={r['method']:9s} "
              f"N2={r['n2_overall']:10s} "
              f"supports={r['n2_supports']}/{r['n2_proxies_tested']} "
              f"zvf130={r['zvf130_verdict']:25s} "
              f"registry_claimers={r['registry_n_claimers']}")
    print("\nzvf130 vs grpo baseline:")
    for m, v in sorted(zv.items(), key=lambda kv: -kv[1]["mean_zvf"]):
        d = v["mean_zvf"] - grpo_zv
        print(f"  {m:25s} mean_zvf={v['mean_zvf']:.4f}  Δ_vs_grpo={d:+.4f}  "
              f"n_seeds={v['n']}")


if __name__ == "__main__":
    reconcile()