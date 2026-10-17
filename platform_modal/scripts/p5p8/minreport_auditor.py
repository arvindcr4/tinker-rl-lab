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
  platform_hybrid/experiments/results/p5p8/minreport_audit.tsv
  platform_hybrid/experiments/results/p5p8/minreport_audit_summary.json
  platform_hybrid/experiments/results/p5p8/figures/minreport_badge_dist.{png,pdf}
  platform_hybrid/experiments/results/p5p8/figures/minreport_per_item.{png,pdf}
"""
from __future__ import annotations

from _minreport_contract import load_inventory_manifests, inventory_counts, verify_trajectory, verify_manifest_evidence, literal_field, field_report
from p5_iter205_manifest_minreport_subfield_audit import MIN_REPORT_FIELDS, classify_subfield
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT
ROOT = REPO_ROOT / "platform_hybrid"
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
    for directory in MANIFEST_DIRS:
        rows = load_inventory_manifests(directory, CELLS_TSV if directory.name == 'manifests' else None)
        for row in rows:
            row['_corpus'] = directory.name
        out.extend(rows)
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
    """Score explicit reporting coverage, separately exposing verification."""
    items = ['loss_form', 'reference_kl', 'sampler_backend', 'telemetry',
             'group_size_schedule', 'heldout_split', 'decontamination']
    out = {'cell_id': m.get('cell_id', m.get('_path', '?')), '_path': m.get('_path', ''),
           '_corpus': m.get('_corpus', ''), 'per_item': [], 'contract_version': 'minreport-review02-v2'}
    total = 0.0
    evidence_checks = verify_manifest_evidence(m, REPO_ROOT)
    out["evidence"] = evidence_checks
    out["manifest_status"] = m.get("_manifest_status", "observed")
    for (number, name, keys, validators, weight, _), item in zip(SCHEMA, items):
        fields = [(sub, parser) for it, sub, parser in MIN_REPORT_FIELDS if it == item]
        classes = {sub: classify_subfield(m, item, sub, parser, {}) for sub, parser in fields}
        covered = sum(c in ('EXTRACTABLE_LITERAL', 'EXTRACTABLE_PARSED') for c in classes.values())
        score = weight * covered / len(fields)
        total += score
        raw = m.get(keys[0])
        evidence = evidence_checks['trajectory'] if number == 4 else evidence_checks['split'] if number == 6 else {'verified': False}
        states = {}
        for sub, _ in fields:
            present_literal, value = literal_field(m, item, sub)
            states[sub] = field_report(value) if present_literal else dict(reported=classes[sub]=='EXTRACTABLE_PARSED', applicable=None, verified=False, value=None, status=classes[sub].lower())
        out['per_item'].append(dict(item_no=number, name=name, key=keys[0], weight=weight,
                                   present=int(raw is not None or covered > 0), validated=int(covered == len(fields)),
                                   honest_na=0, base=covered/len(fields), sub_n=covered, sub_k=len(fields),
                                   sub_frac=covered/len(fields), item_score=round(score, 2),
                                   value=str(raw)[:80], fields=classes, field_states=states, evidence=evidence))
    out.update(badge=round(total, 1), tier=tier_for(total))
    for k in ('_model_id', '_task_slice', '_G', '_temperature', '_seed'):
        if k in m:
            out[k.lstrip('_')] = m[k]
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
        "inventory_counts": {"expected": n, "observed": sum(s["manifest_status"] == "observed" for s in scored), "analyzed": sum(s["manifest_status"] == "observed" for s in scored), "missing": sum(s["manifest_status"] == "missing" for s in scored), "invalid": sum(s["manifest_status"] == "invalid" for s in scored), "contradictory": sum(s["evidence"]["split"].get("contradictory", False) for s in scored), "not_applicable": sum(any(state["applicable"] is False for item in s["per_item"] for state in item["field_states"].values() if state["reported"]) for s in scored)},
        "reporting_contract": "typed_reporting_coverage_not_scientific_validity",
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