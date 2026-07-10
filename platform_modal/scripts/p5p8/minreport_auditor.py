#!/usr/bin/env python3
"""P5 MIN-REPORT-RL Auditor prototype.

Implements the "MIN-REPORT-RL Auditor (0-100 badge)" component of
paper/sections/p5_toolchain.tex. For each manifest in the worktree we
score the seven MIN-REPORT items of paper/sections/p5_stack.tex on a
weighted 0-100 badge:

  item 1 (loss form)               = 10 pts
  item 2 (reference policy & KL)   = 10 pts
  item 3 (sampler / backend)       = 20 pts
  item 4 (per-step ZVF/GU)         = 20 pts
  item 5 (group-size schedule)     = 10 pts
  item 6 (held-out split)          = 10 pts
  item 7 (decontam + parser probe) = 20 pts
                              total = 100 pts

Within each item, the per-cell score is:
  score = weight * base * (0.5 + 0.5 * subfield_coverage)
where base in {0.0, 0.25, 0.5, 1.0} is a coarse valuation:
  0.0  = missing key
  0.25 = key present but value unrecognized
  0.5  = honest n/a declaration (key present, value "n/a-*" matches)
  1.0  = key present, value validated against the item's regex list

We accept multiple alternate keys per item because the two corpus
families in the worktree use different naming (mega manifests use
"ref_policy_kl"; quick manifests use "ref_policy_kl_handling").

Outputs:
  experiments/results/p5p8/minreport_audit.tsv
  experiments/results/p5p8/minreport_audit_summary.json
  experiments/results/p5p8/figures/minreport_badge_dist.{png,pdf}
  experiments/results/p5p8/figures/minreport_per_item.{png,pdf}
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIRS = [
    ROOT / "experiments" / "results" / "mega_20260704" / "manifests",
    ROOT / "experiments" / "results" / "quick_20260704",
]
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
FIG_DIR = OUT_DIR / "figures"

# (item_no, name, alt-keys, validators, weight, subfield specs)
SCHEMA = [
    (1, "Loss form", ["loss_form"],
     [r"^(grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft|n/a-sampling)$"],
     10,
     [("ratio_level", r"(token|sequence)"),
      ("clip_range", r"clip[_ ]?(low|high|range)?\s*[:=]?\s*[0-9.]+"),
      ("advantage_normalization", r"(std|mean|sum|batch|group)"),
      ("dynamic_sampling", r"(dynamic[- ]sampling)"),
      ("token_mask", r"(token[- ]?mask|completion[- ]?only)"),
     ]),
    (2, "Reference policy & KL",
     ["ref_policy_kl", "ref_policy_kl_handling"],
     [r"^(kl-[a-z]+(\d+(\.\d+)?)?|kl-est-[a-z]+|no-kl|n/a(?:-[a-z]+)?)$"],
     10,
     [("ref_snapshot", r"(ref|snapshot|policy)"),
      ("kl_coefficient", r"kl[-_ ]?(coeff|coef|weight|beta)\s*[:=]?\s*[0-9.]+"),
      ("kl_estimator", r"(k1|k2|k3|mc|exact|approx)"),
     ]),
    (3, "Sampler / backend / precision", ["sampler_backend_precision"],
     [r"^(tinker-closed|vllm|sglang|hf|trtllm|openai|anthropic)[-@a-zA-Z0-9._/]*$"],
     20,
     [("backend", r"(tinker|vllm|sglang|hf|trtllm|openai|anthropic)"),
      ("precision", r"(bf16|fp16|fp32|fp8|int8)"),
      ("decoding_params", r"(temp|sampling|top[-_ ]?p|top[-_ ]?k)"),
     ]),
    (4, "Per-step ZVF/GU trajectory", ["per_step_zvf_path"],
     [r".*"],
     20,
     [("trajectory_key_zvf", r"\"(zvf|ZVF|zvf_traj|zvf_per_step)\""),
      ("trajectory_key_gu", r"\"(gu|GU|gradient_utilization|gradient_utilisation)\""),
      ("trajectory_length", r"\"(steps|trajectory|n_steps|length|n_groups)\""),
     ]),
    (5, "Group-size schedule", ["group_size_schedule"],
     [r"^(fixed-G=\d+|adaptive[-+a-zA-Z0-9=<>]*|escalating|decaying|constant G=\d+.*|paired phases.*|arm [A-Z]:.*|n/a.*)$"],
     10,
     [("G_value", r"G=\d+"),
      ("adaptive_rule", r"(adaptive|escalat|decay)"),
     ]),
    (6, "Held-out split", ["heldout_split"],
     [r".*"],
     10,
     [("split_identity", r"[a-z0-9_]+"),
      ("disjoint_flag", r"(disjoint|held[- ]?out|test)"),
     ]),
    (7, "Decontamination & parser probe", ["decontamination_notes"],
     [r".*"],
     20,
     [("contamination_check", r"(decontam|ngram|overlap|exact|check)"),
      ("parser_probe", r"(parser|probe|jitter|perturb)"),
      ("probe_quantified", r"(\d+\.\d+|\d+\s*[a-zA-Z]+|\beps\b|\\epsilon|10\^-?\d+)"),
     ]),
]

# Color coding for badges
BADGE_TIERS = [
    (90, "gold"),
    (75, "silver"),
    (50, "bronze"),
    (25, "wood"),
    (0, "fail"),
]


def tier_for(score: float) -> str:
    for thr, name in BADGE_TIERS:
        if score >= thr:
            return name
    return "fail"


def load_manifests() -> list[dict]:
    out = []
    for mdir in MANIFEST_DIRS:
        if not mdir.is_dir():
            continue
        for jf in sorted(mdir.glob("*.json")):
            try:
                with jf.open() as f:
                    d = json.load(f)
            except Exception as e:
                print(f"warn: bad json {jf}: {e}", file=sys.stderr)
                continue
            # Only score files that touch at least one MIN-REPORT key
            if not any(any(k in d for k in keys)
                       for _, _, keys, _, _, _ in SCHEMA):
                continue
            d["_path"] = jf.name
            d["_corpus"] = mdir.name
            m = re.match(
                r"^(?P<model>[^_]+)_(?P<task>[a-z0-9_]+)_G(?P<G>\d+)_t(?P<t>[\d.]+)_s(?P<s>\d+)_",
                jf.name,
            )
            if m:
                d["_model_id"] = m.group("model")
                d["_task_slice"] = m.group("task")
                d["_G"] = int(m.group("G"))
                d["_temperature"] = float(m.group("t"))
                d["_seed"] = int(m.group("s"))
            out.append(d)
    return out


def sub_coverage(value, trajectory_text, specs):
    n, k = 0, len(specs)
    if k == 0:
        return 0, 0
    hay = trajectory_text if trajectory_text is not None else (value or "")
    if hay is None:
        hay = ""
    for _, pat in specs:
        if re.search(pat, hay, re.IGNORECASE):
            n += 1
    return n, k


def score_manifest(m: dict) -> dict:
    out = {
        "cell_id": m.get("cell_id", m.get("exp", m.get("experiment", m.get("_path", "?")))),
        "_path": m.get("_path", ""),
        "_corpus": m.get("_corpus", ""),
    }
    total = 0.0
    per_item = []
    for item_no, name, keys, validators, weight, subs in SCHEMA:
        raw = None
        used_key = None
        for k in keys:
            if k in m and m[k] is not None:
                raw = m[k]
                used_key = k
                break
        present = raw is not None and str(raw).strip() != ""
        validated = bool(present) and any(
            re.match(v, str(raw), re.IGNORECASE) for v in validators)
        if not validated:
            sub_n, sub_k = 0, len(subs)
        else:
            trajectory_text = None
            if item_no == 4:
                zp = str(raw)
                zp_full = (ROOT / zp) if not zp.startswith("/") else Path(zp)
                if zp_full.is_file():
                    try:
                        trajectory_text = zp_full.read_text()
                    except Exception:
                        trajectory_text = ""
            sub_n, sub_k = sub_coverage(str(raw), trajectory_text, subs)
        sub_frac= (0.5 + 0.5 * sub_n / sub_k) if sub_k else 1.0
        is_na = isinstance(raw, str) and raw.strip().lower().startswith("n/a")
        honest_na = is_na and any(
            re.match(v, str(raw), re.IGNORECASE) for v in validators)
        if honest_na:
            base = 0.5
        elif present and validated:
            base = 1.0
        elif present:
            base = 0.25
        else:
            base = 0.0
        item_score = weight * base * sub_frac
        total += item_score
        per_item.append({
            "item_no": item_no,
            "name": name,
            "key": used_key or keys[0],
            "weight": weight,
            "present": int(present),
            "validated": int(validated),
            "honest_na": int(honest_na),
            "base": round(base, 3),
            "sub_n": sub_n,
            "sub_k": sub_k,
            "sub_frac": round(sub_frac, 3),
            "item_score": round(item_score, 2),
            "value": str(raw)[:80] if raw is not None else "",
        })
    out["per_item"] = per_item
    out["badge"] = round(total, 1)
    out["tier"] = tier_for(total)
    for k in ("_model_id", "_task_slice", "_G", "_temperature", "_seed"):
        if k in m:
            out[k.lstrip("_")] = m[k]
    return out


def write_outputs(scored):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tsv = OUT_DIR / "minreport_audit.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "cell_id", "corpus", "model_id", "task_slice", "G",
            "temperature", "seed",
            "item1_loss", "item2_kl", "item3_backend", "item4_zvf",
            "item5_G", "item6_heldout", "item7_decontam",
            "badge", "tier",
        ])
        for s in scored:
            pi = {p["item_no"]: p["item_score"] for p in s["per_item"]}
            w.writerow([
                s["cell_id"], s.get("_corpus", ""),
                s.get("model_id", ""), s.get("task_slice", ""),
                s.get("G", ""), s.get("temperature", ""),
                s.get("seed", ""),
                pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], pi[7],
                s["badge"], s["tier"],
            ])
    by_tier = defaultdict(int)
    by_task = defaultdict(list)
    by_model = defaultdict(list)
    by_G = defaultdict(list)
    by_temp = defaultdict(list)
    by_seed = defaultdict(list)
    by_corpus = defaultdict(list)
    item_score_total = defaultdict(float)
    item_score_max = defaultdict(float)
    badges = []
    for s in scored:
        by_tier[s["tier"]] += 1
        badges.append(s["badge"])
        by_task[s.get("task_slice", "?")].append(s["badge"])
        by_model[s.get("model_id", "?")].append(s["badge"])
        by_G[int(s.get("G", 0))].append(s["badge"])
        by_temp[float(s.get("temperature", 0))].append(s["badge"])
        by_seed[int(s.get("seed", 0))].append(s["badge"])
        by_corpus[s.get("_corpus", "?")].append(s["badge"])
        for p in s["per_item"]:
            item_score_total[p["item_no"]] += p["item_score"]
            item_score_max[p["item_no"]] += p["weight"]
    n = len(scored)
    summary = {
        "n_manifests": n,
        "corpus_sizes": {k: len(v) for k, v in by_corpus.items()},
        "badge_mean": round(sum(badges) / max(1, n), 2),
        "badge_median": round(sorted(badges)[n // 2], 2),
        "badge_min": round(min(badges), 2),
        "badge_max": round(max(badges), 2),
        "badge_std": round((sum((b - sum(badges) / n) ** 2 for b in badges)
                            / max(1, n)) ** 0.5, 2),
        "tier_counts": dict(by_tier),
        "stratified": {
            "by_corpus": {k: round(sum(v) / len(v), 2) for k, v in by_corpus.items()},
            "by_task_slice": {k: round(sum(v) / len(v), 2) for k, v in by_task.items()},
            "by_model": {k: round(sum(v) / len(v), 2) for k, v in by_model.items()},
            "by_G": {str(k): round(sum(v) / len(v), 2) for k, v in by_G.items()},
            "by_temperature": {str(k): round(sum(v) / len(v), 2) for k, v in by_temp.items()},
            "by_seed": {str(k): round(sum(v) / len(v), 2) for k, v in by_seed.items()},
        },
        "per_item_score_pct": {
            str(k): round(100.0 * item_score_total[k]
                          / max(1e-9, item_score_max[k]), 1)
            for k in sorted(item_score_total)
        },
    }
    (OUT_DIR / "minreport_audit_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def make_figure(scored, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warn: matplotlib not available; skipping figure",
              file=sys.stderr)
        return
    badges = [s["badge"] for s in scored]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(badges, bins=20, color="#4477AA", edgecolor="black")
    for thr, name in BADGE_TIERS[:-1]:
        ax.axvline(thr, color="grey", linestyle="--", alpha=0.5)
        ax.text(thr + 0.3, ax.get_ylim()[1] * 0.95, name,
                color="grey", fontsize=8, va="top")
    ax.set_xlabel("MIN-REPORT-RL Auditor badge (0-100)")
    ax.set_ylabel("# manifests")
    ax.set_title(f"MIN-REPORT-RL badge distribution (n={len(scored)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "minreport_badge_dist.png", dpi=150)
    fig.savefig(FIG_DIR / "minreport_badge_dist.pdf")
    plt.close(fig)
    items = sorted(SCHEMA, key=lambda x: x[0])
    labels = [f"item{i}\n{name.split()[0].lower()}"
              for i, name, *_ in items]
    weights = [w for *_, w, _ in items]
    pct = [summary["per_item_score_pct"][str(items[i][0])]
           for i in range(len(items))]
    x = list(range(len(items)))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x, pct,
            color=["#4477AA" if w == 10 else "#EE6677" for w in weights],
            edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("% of item weight achieved")
    ax2.set_title("Per-item MIN-REPORT coverage\n"
                  "red = high-leverage items (weight 20)")
    ax2.axhline(50, color="grey", linestyle=":", alpha=0.5)
    for i, p in enumerate(pct):
        ax2.text(i, p + 1, f"{p:.0f}%", ha="center", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "minreport_per_item.png", dpi=150)
    fig2.savefig(FIG_DIR / "minreport_per_item.pdf")
    plt.close(fig2)


def main():
    manifests = load_manifests()
    if not manifests:
        print(f"no manifests in {MANIFEST_DIRS}", file=sys.stderr)
        return 1
    scored = [score_manifest(m) for m in manifests]
    summary = write_outputs(scored)
    make_figure(scored, summary)
    print(f"manifests scored:    {summary['n_manifests']}")
    print(f"corpus sizes:        {summary['corpus_sizes']}")
    print(f"badge mean / median: {summary['badge_mean']} / {summary['badge_median']}")
    print(f"badge range:         [{summary['badge_min']}, {summary['badge_max']}]  "
          f"(std={summary['badge_std']})")
    print("tier counts:")
    for tier in ("gold", "silver", "bronze", "wood", "fail"):
        print(f"  {tier:>7s}: {summary['tier_counts'].get(tier, 0)}")
    print("per-item %:")
    for k, v in sorted(summary["per_item_score_pct"].items()):
        w = SCHEMA[int(k) - 1][4]
        print(f"  item {k} (w={w:>2}): {v:>5.1f}%")
    print("stratified:")
    for axis, vals in summary["stratified"].items():
        print(f"  {axis}: {vals}")
    print(f"figure: {FIG_DIR}/minreport_badge_dist.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())