#!/usr/bin/env python3
"""P6 iter-54: Add missing delta_*.json entries from real worktree data.

Vein (d) of the iter-54 brief: methods present in the worktree but missing
from the registry. Two new entries ship this iter:

  * delta_adaptiveg --- adaptive group-size schedule (4->6->8 ladder
    driven by per-step ZVF). Provenance = qp7_adaptive.tsv arm B vs arm A,
    paired bootstrap (B=2000, seed=20260704) on (reward_mean, zvf) per step.
    This is the live adaptive-g controller implemented in
    colab-open_grpo-adaptiveg_e3 (and the iter-47/51 P7 unified bank).

  * delta_reinforce --- policy-gradient with no baseline (REINFORCE).
    Provenance = iter-45 colab runs; measured block intentionally
    null because the worktree does NOT carry a same-stack REINFORCE
    arm (would require a new Tinker run that the iter budget does not
    permit). The entry closes the registry's "REINFORCE is mentioned
    in 2 stack records (reinforce on Qwen3-8B, GSM8K) but has no delta"
    gap.

  * delta_liteppo --- LitePPO (reduced-variant PPO without the value
    head / GAE). Provenance = iter-45 colab runs; same measured=null
    policy as delta_reinforce. LitePPO is mentioned in
    `experiments/results/EXPERIMENT_LEDGER.md` as `ppo_lite` but has
    no registry entry.

Plus a cross-reference audit (`missing_delta_audit.tsv`):

  For every (entry, variant_deltas_applied[*].delta_id) claim, check
  whether `registry/entries/<delta_id>.json` exists. Report any
  CLAIMED_BUT_MISSING delta_ids (the next iter's backlog).

Writes:
  experiments/results/p5p8/p6_new_deltas_audit.tsv        (one row per new delta)
  experiments/results/p5p8/p6_new_deltas_measured.tsv     (per (delta, metric, panel))
  experiments/results/p5p8/p6_new_deltas_summary.json     (headline numbers)
  experiments/results/p5p8/missing_delta_audit.tsv        (claimed but missing)
  registry/entries/delta_adaptiveg.json                   (new)
  registry/entries/delta_reinforce.json                   (new)
  registry/entries/delta_liteppo.json                     (new)

Stdlib + jsonschema only. Exit 0 iff every (old + new) entry still
parses against schema.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
import statistics as st

try:
    import jsonschema  # type: ignore
except ImportError:
    print("FATAL: jsonschema not installed", flush=True)
    raise SystemExit(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRIES = ROOT / "registry" / "entries"
SCHEMA = json.load(open(ROOT / "registry" / "schema.json"))
V = jsonschema.Draft202012Validator(SCHEMA)
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260704
N_BOOT = 2000
ADAPTIVE_TSV = ROOT / "experiments/results/quick_20260704/qp7_adaptive.tsv"


# ---------------------------------------------------------------------------
# 1. Compute measured deltas for delta_adaptiveg from qp7_adaptive.tsv
# ---------------------------------------------------------------------------
def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def paired_boot(deltas, n_boot=N_BOOT, seed=SEED):
    n = len(deltas)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return sum(deltas) / n, lo, hi, n


def adaptiveg_measured():
    """arm B (adaptive 4->6->8) vs arm A (fixed G=4). Paired by step."""
    rows = read_tsv(ADAPTIVE_TSV)
    a_by_step = {int(r["step"]): r for r in rows if r["arm"] == "A"}
    b_by_step = {int(r["step"]): r for r in rows if r["arm"] == "B"}
    common_steps = sorted(set(a_by_step) & set(b_by_step))
    if not common_steps:
        return []
    out = []
    for metric in ("reward_mean", "zvf"):
        deltas = []
        for s in common_steps:
            va = fnum(a_by_step[s][metric])
            vb = fnum(b_by_step[s][metric])
            if va is None or vb is None:
                continue
            deltas.append(vb - va)  # variant - base
        if len(deltas) < 3:
            continue
        d, lo, hi, n = paired_boot(deltas)
        out.append({
            "metric": metric,
            "panel": "qp7_adaptive_armB_vs_armA_paired",
            "base": "grpo",
            "delta": round(d, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "n": n,
            "significant": (lo > 0) or (hi < 0),
            "ci_method": {
                "method": "paired_step_bootstrap_pct",
                "n_boot": N_BOOT, "seed": SEED, "ci_level": 0.95,
                "source": "platform_modal/scripts/p5p8/p6_add_missing_deltas.py",
            },
            "source": "experiments/results/quick_20260704/qp7_adaptive.tsv",
            "note": f"arm B (adaptive 4->6->8) - arm A (fixed G=4); paired by step",
        })
    return out, common_steps


# ---------------------------------------------------------------------------
# 2. Sign-match + verdict classifier (mirrors p6_measured_vs_claim.py)
# ---------------------------------------------------------------------------
def sign_match(observed: float, predicted: str) -> bool:
    return {
        ">0": observed > 0, "<0": observed < 0,
        ">=0": observed >= 0, "<=0": observed <= 0,
        "=0": observed == 0, "==0": observed == 0,
    }[predicted]


def classify(observed, ci_low, ci_high, predicted):
    if predicted is None:
        return "UNCLAIMED", "no expected_effect declared"
    sig = (ci_low > 0) or (ci_high < 0)
    if not sig:
        return "NEUTRAL", f"CI=[{ci_low:+.4f},{ci_high:+.4f}] includes 0"
    sign_ok = sign_match(observed, predicted)
    if sign_ok:
        return "SUPPORTS", f"significant, matches predicted {predicted}"
    return "CONTRADICTS", f"significant, OPPOSITE predicted {predicted}"


# ---------------------------------------------------------------------------
# 3. New entries
# ---------------------------------------------------------------------------
def build_adaptiveg_entry():
    measured, common_steps = adaptiveg_measured()
    expected_effects = [
        {"metric": "reward_mean",
         "panel": "qp7_adaptive_armB_vs_armA_paired",
         "predicted_sign": ">=0",
         "rationale": "adaptive G aims to spend compute on contrast-rich prompts; "
                      "should be at least reward-neutral and ideally positive."},
        {"metric": "zvf",
         "panel": "qp7_adaptive_armB_vs_armA_paired",
         "predicted_sign": "<0",
         "rationale": "escalating G when ZVF is high reduces the fraction of "
                      "all-same groups on the next step (the iter-31 unified-band "
                      "controller's de-escalation branch in reverse)."},
    ]
    claim_validation = []
    expected_by_key = {(e["metric"], e["panel"]): e for e in expected_effects}
    for m in measured:
        key = (m["metric"], m["panel"])
        exp = expected_by_key.get(key)
        pred = exp["predicted_sign"] if exp else None
        v, r = classify(m["delta"], m["ci_low"], m["ci_high"], pred)
        claim_validation.append({
            "metric": m["metric"], "panel": m["panel"],
            "predicted_sign": pred, "observed_delta": m["delta"],
            "ci_low": m["ci_low"], "ci_high": m["ci_high"],
            "significant": m["significant"], "verdict": v, "rationale": r,
        })
    rec = {
        "record_type": "variant_delta",
        "schema_version": "0.1.0",
        "id": "delta_adaptiveg",
        "name": "Adaptive-G (ZVF-driven)",
        "base": "grpo",
        "citation": {
            "bibkey": "tinker2026adaptiveg",
            "arxiv": "",
            "title": "Adaptive group-size schedule driven by per-step ZVF (worktree "
                     "implementation, live in colab-open_grpo-adaptiveg_e3 and "
                     "iter-47/51 P7 unified controller bank)"
        },
        "deltas": [
            {"component": "zvf_driven_group_size_ladder",
             "field": "group_size_schedule.schedule",
             "change": "schedule='adaptive'; initial_g=4; adaptation_rule="
                      "'escalate 4->6->8 when ZVF>0.5 (cap 8); de-escalate "
                      "8->6->4 when ZVF<0.2 (floor 4); per-step ZVF measured "
                      "client-side from reward tensors' variance indicator."},
        ],
        "measured": measured,
        "expected_effects": expected_effects,
        "claim_validation": claim_validation,
        "notes": ("Iter-54 vein (d): added because the worktree carries a real "
                  "adaptive-G arm (qp7_adaptive.tsv, n=16 paired steps, G=4 fixed vs "
                  "4->6->8 adaptive ladder) but no delta_*.json existed. Measured "
                  f"on n={len(common_steps)} paired-by-step observations."),
    }
    return rec


def build_reinforce_entry():
    """REINFORCE without baseline; no same-stack measured data yet."""
    return {
        "record_type": "variant_delta",
        "schema_version": "0.1.0",
        "id": "delta_reinforce",
        "name": "REINFORCE",
        "base": "grpo",
        "citation": {
            "bibkey": "williams1992reinforce",
            "arxiv": "",
            "title": "Simple Statistical Gradient-Following Algorithms for "
                     "Connectionist Reinforcement Learning (REINFORCE)"
        },
        "deltas": [
            {"component": "no_baseline",
             "field": "reference_kl.reference_policy",
             "change": "no group-mean or value baseline; raw reward r_i is the "
                      "advantage (A_i = r_i); no per-token variance reduction "
                      "(equivalent to GRPO with no baseline term)."},
            {"component": "no_clipping",
             "field": "loss_form.clip_eps_low",
             "change": "no PPO-style ratio clipping (canonical leaf pinned at "
                      "clip_eps_low=null, clip_eps_high=null to mark reported-as-absent)."},
        ],
        "notes": ("Iter-54 vein (d): REINFORCE is mentioned in "
                  "experiments/results/EXPERIMENT_LEDGER.md (gsm8k-reinforce, "
                  "4 wandb runs on Qwen3-8B / Llama-3.1-8B-Instruct) but no "
                  "delta_*.json existed. Measured block intentionally null: "
                  "no same-stack REINFORCE arm exists in the worktree, so adding "
                  "a measured row would be fabricatory. To be measured once a "
                  "same-stack run lands (criterion: same model + task + sampler + "
                  "RLHF pipeline with only the baseline removed)."),
    }


def build_liteppo_entry():
    """LitePPO: reduced-variant PPO without value head / GAE."""
    return {
        "record_type": "variant_delta",
        "schema_version": "0.1.0",
        "id": "delta_liteppo",
        "name": "LitePPO",
        "base": "grpo",
        "citation": {
            "bibkey": "liteppo2024",
            "arxiv": "",
            "title": "LitePPO: a lightweight PPO variant without a learned value "
                     "head (worktree reference; no peer-reviewed citation — listed "
                     "here as a transparent placeholder so future measured blocks "
"have a stable id)"
        },
        "deltas": [
            {"component": "no_value_head",
             "field": "reference_kl.reference_policy",
             "change": "no learned V_head; advantages are estimated by an "
                      "external baseline (group mean) instead of GAE."},
            {"component": "ratio_clip",
             "field": "loss_form.clip_eps_low",
             "change": "PPO-style ratio clipping retained with eps_low=eps_high=0.2 "
                      "(symmetric clip, unlike DAPO's asymmetric 0.2/0.28)."},
        ],
        "notes": ("Iter-54 vein (d): LitePPO is mentioned in "
                  "experiments/results/EXPERIMENT_LEDGER.md as ppo_lite but had no "
                  "delta_*.json. Citation is a transparent placeholder (arxiv=null) "
                  "because no peer-reviewed LitePPO paper is verified; this is a "
                  "known limitation flagged on the entry itself, not hidden. "
                  "Measured block intentionally null for the same reason as "
                  "delta_reinforce: no same-stack arm in the worktree."),
    }


# ---------------------------------------------------------------------------
# 4. Cross-reference: CLAIMED_BUT_MISSING audit
# ---------------------------------------------------------------------------
def missing_delta_audit():
    """For every stack record, list every (entry, delta_id) it claims.
    Return rows where delta_*.json does not exist."""
    rows = []
    for p in sorted(ENTRIES.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["record_type"] != "stack":
            continue
        for vd in rec.get("variant_deltas_applied") or []:
            did = vd["delta_id"]
            target = ENTRIES / f"{did}.json"
            rows.append({
                "stack_id": rec["id"],
                "framework": rec["framework"]["name"],
                "delta_id": did,
                "component": vd["component"],
                "status": vd["status"],
                "target_exists": target.exists(),
                "verdict": ("OK" if target.exists() else "CLAIMED_BUT_MISSING"),
            })
    return rows


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------
def main():
    # 5a. Build + write the three new entries
    new_records = [
        build_adaptiveg_entry(),
        build_reinforce_entry(),
        build_liteppo_entry(),
    ]
    new_audit_rows = []
    written = 0
    for rec in new_records:
        errs = list(V.iter_errors(rec))
        assert not errs, (rec["id"], errs[0].message)
        path = ENTRIES / f"{rec['id']}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        written += 1
        n_measured = len(rec.get("measured") or [])
        n_cv = len(rec.get("claim_validation") or [])
        new_audit_rows.append({
            "id": rec["id"],
            "name": rec["name"],
            "base": rec["base"],
            "has_citation": bool(rec.get("citation", {}).get("bibkey")),
            "arxiv": rec.get("citation", {}).get("arxiv") or "",
            "n_deltas": len(rec.get("deltas") or []),
            "n_measured": n_measured,
            "n_claim_validation": n_cv,
            "notes_excerpt": (rec.get("notes") or "")[:80],
        })

    # 5b. Re-run the iter-50 registry health audit on the now-34-entry corpus
    schema_errors = []
    n_total = 0
    for p in sorted(ENTRIES.glob("*.json")):
        n_total += 1
        rec = json.loads(p.read_text())
        errs = list(V.iter_errors(rec))
        if errs:
            schema_errors.append((p.name, errs[0].message[:140]))

    # 5c. Missing-delta audit
    miss_rows = missing_delta_audit()
    n_missing = sum(1 for r in miss_rows if not r["target_exists"])

    # 5d. Write artifacts
    new_audit_tsv = OUT / "p6_new_deltas_audit.tsv"
    with open(new_audit_tsv, "w", newline="") as f:
        cols = ["id", "name", "base", "has_citation", "arxiv",
                "n_deltas", "n_measured", "n_claim_validation", "notes_excerpt"]
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
        w.writeheader()
        w.writerows(new_audit_rows)

    # measured rows from delta_adaptiveg
    measured_tsv = OUT / "p6_new_deltas_measured.tsv"
    with open(measured_tsv, "w", newline="") as f:
        f.write("delta_id\tmetric\tpanel\tdelta\tci_low\tci_high\t"
                "n\tsignificant\tsource\n")
        for rec in new_records:
            for m in (rec.get("measured") or []):
                f.write(f"{rec['id']}\t{m['metric']}\t{m['panel']}\t"
                        f"{m['delta']}\t{m['ci_low']}\t{m['ci_high']}\t"
                        f"{m['n']}\t{m['significant']}\t{m['source']}\n")

    miss_tsv = OUT / "missing_delta_audit.tsv"
    with open(miss_tsv, "w", newline="") as f:
        cols = ["stack_id", "framework", "delta_id", "component",
                "status", "target_exists", "verdict"]
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
        w.writeheader()
        w.writerows(miss_rows)

    summ = {
        "iter": 54,
        "pillar": "P6",
        "vein": "(d) add missing variant-delta entries + cross-reference audit",
        "n_new_entries_written": written,
        "new_entries": [r["id"] for r in new_records],
        "registry_total_after": n_total,
        "schema_pass_after": n_total - len(schema_errors),
        "schema_fail_after": len(schema_errors),
        "schema_fail_ids": [n for n, _ in schema_errors],
        "missing_delta_audit": {
            "n_claims_audited": len(miss_rows),
            "n_claimed_but_missing": n_missing,
            "missing_delta_ids": sorted({r["delta_id"] for r in miss_rows
                                         if not r["target_exists"]}),
        },
        "adaptiveg_panel": "qp7_adaptive_armB_vs_armA_paired",
        "adaptiveg_seed": SEED,
        "adaptiveg_n_boot": N_BOOT,
    }
    summ_path = OUT / "p6_new_deltas_summary.json"
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, sort_keys=True)

    # 5e. Print headline
    print(f"=== Iter 54 P6 — add missing variant-delta entries ===")
    print(f"  new entries written: {written} -> {[r['id'] for r in new_records]}")
    print(f"  registry after: {n_total} entries ({n_total - len(schema_errors)} "
          f"PASS, {len(schema_errors)} FAIL)")
    if schema_errors:
        for n, m in schema_errors:
            print(f"    FAIL {n}: {m}")
    print(f"  missing-delta audit: {n_missing}/{len(miss_rows)} "
          f"(claims missing delta_*.json)")
    if summ["missing_delta_audit"]["missing_delta_ids"]:
        print(f"    missing ids: {summ['missing_delta_audit']['missing_delta_ids']}")
    print(f"  outputs:")
    print(f"    {new_audit_tsv}")
    print(f"    {measured_tsv}")
    print(f"    {miss_tsv}")
    print(f"    {summ_path}")

    return 1 if schema_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
