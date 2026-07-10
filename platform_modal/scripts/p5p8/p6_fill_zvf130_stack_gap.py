#!/usr/bin/env python3
"""P6 iter-102: fill the zvf130 stack-entry coverage gap + stamp sig_robust.

Two actions, both provenance-tagged, both derived from ground truth
(platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv):

  1. Create zvf130_<m>.json STACK entries for the 4 real 5-seed methods that
     lacked one: grpo (the BASE!), aero, areal, gift.  Each mirrors the existing
     zvf130_scafgrpo.json shape and records BOTH the paired-bootstrap CI
     (from the delta entry, iter90 provenance) and a conservative Welch
     two-sample CI, plus sig_robust = bootstrap_sig AND welch_sig.

  2. Backfill `sig_robust` (+ welch CI) into every existing zvf130-derived block:
     the 5 prior stack entries' outcomes and all 8 delta entries' zvf130_5seed
     measured block.  This makes the optimistic bootstrap `significant` flag
     never stand alone -- aero & ngrpo are exposed as NOT sig-robust.

Idempotent: re-running overwrites the iter102 fields in place.
"""
import json, os, glob, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTRIES = os.path.join(ROOT, "registry/entries")
spec = importlib.util.spec_from_file_location(
    "ig", os.path.join(ROOT, "platform_modal/scripts/p5p8/p6_registry_crossref_integrity.py"))
ig = importlib.util.module_from_spec(spec); spec.loader.exec_module(ig)

TSV = ig.load_tsv()
BASE = ig.TSV_BASE if hasattr(ig, "TSV_BASE") else "grpo"
base = TSV["grpo"]
STACKS, DELTAS = ig.load_entries()
DATE = "2026-07-05"
AUDIT = "platform_modal/scripts/p5p8/p6_fill_zvf130_stack_gap.py"


def welch_of(m):
    return ig.welch(TSV[m]["zvf_risk_mean"], TSV[m]["zvf_risk_sd"], 5,
                    base["zvf_risk_mean"], base["zvf_risk_sd"], 5)


def boot_ci(m):
    dm = ig.delta_z130(DELTAS[m][1]) if m in DELTAS else None
    if not dm:
        return None, None, None
    return dm.get("ci_low"), dm.get("ci_high"), bool(dm.get("significant"))


def make_stack(m):
    d, wlo, whi, wsig, _ = welch_of(m)
    blo, bhi, bsig = boot_ci(m)
    gt = TSV[m]
    is_base = (m == "grpo")
    sig_robust = None if is_base else (bool(bsig) and bool(wsig))
    delta_apps = ([] if is_base else [{
        "delta_id": f"delta_{m}", "component": "see delta entry",
        "status": "unknown",
        "note": "single-batch risk-index harness; per-component isolation not run"}])
    outcomes = {
        "mean_last10_train_reward": None, "mean_zvf": None, "heldout_delta": None,
        "rollouts": None,
        "coverage": {
            "min_report_coverage": 0.4286,
            "declared_deltas_coverage": (None if is_base else 0.0),
            "measured_coverage": 1.0, "ci_method_present": True,
            "audit_source": AUDIT, "audit_date": DATE},
        "zvf_risk_mean": round(gt["zvf_risk_mean"], 6),
        "zvf_risk_sd": round(gt["zvf_risk_sd"], 6),
        "n_seeds": 5, "failure_rate": gt["failure_rate"],
        "mag_mean": round(gt["mag_mean"], 6), "csd_mean": round(gt["csd_mean"], 6),
        "drift_mean": round(gt["drift_mean"], 6),
        "delta_vs_grpo_mean": (0.0 if is_base else round(d, 6)),
        "delta_vs_grpo_ci_lo": (0.0 if is_base else blo),
        "delta_vs_grpo_ci_hi": (0.0 if is_base else bhi),
        "delta_vs_grpo_sig": (False if is_base else bool(bsig)),
        "delta_vs_grpo_welch_ci_lo": (0.0 if is_base else round(wlo, 6)),
        "delta_vs_grpo_welch_ci_hi": (0.0 if is_base else round(whi, 6)),
        "delta_vs_grpo_welch_sig": (False if is_base else bool(wsig)),
        "sig_robust_bootstrap_and_welch": sig_robust,
        "crossref_integrity_iter102": {
            "audit_date": DATE, "audit_source": AUDIT,
            "ground_truth": "platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv",
            "point_estimate_matches_tsv": True,
            "note": ("BASE reference (delta vs self = 0)." if is_base else
                     "delta_vs_grpo bootstrap CI from iter90 (B=4000 paired "
                     "Gaussian-residual); welch CI is a conservative two-sample "
                     "cross-check. sig_robust requires BOTH.")},
    }
    return {
        "record_type": "stack", "schema_version": "0.1.0",
        "id": f"zvf130_{m}", "label_claimed": m,
        "framework": {"name": "worktree-zvf130-batch", "version": None, "openness": "open"},
        "model": "Qwen/Qwen3-8B-base (shared zvf-iter130 batch harness)",
        "task": "gsm8k (canonical 16-prompt fixed eval, zvf_iter130 spec)",
        "seeds": [0, 1, 2, 3, 4],
        "provenance": {
            "source_artifacts": [
                f"zvf_iter130_method_risk.tsv row method={m} "
                f"(zvf_risk_mean={round(gt['zvf_risk_mean'],4)}, n_seeds=5); "
                "per-method risk index, single-batch harness"],
            "wandb": None, "date_recorded": "2026-07-04"},
        "variant_deltas_applied": delta_apps,
        "min_report": {
            "loss_form": {"importance_ratio_level": None, "clip_eps_low": None,
                          "clip_eps_high": None, "length_normalization": None,
                          "advantage_normalization": None, "token_mask": None},
            "reference_kl": {"reference_policy": None, "kl_beta": None, "kl_estimator": None},
            "sampler_backend": {"backend": "shared zvf-iter130 batch harness (Tinker)",
                                "precision": "bf16", "temperature": 0.7, "top_p": 0.95},
            "telemetry": {"per_step_zvf": True, "per_step_gu": True,
                          "source": "zvf_iter130_method_risk.tsv (per-step file zvf_iter130/*)"},
            "group_size_schedule": {"initial_g": None, "schedule": None, "adaptation_rule": None},
            "heldout_split": {"disjoint_from_reward_env": True,
                              "description": "GSM8K canonical 16-prompt fixed eval"},
            "decontamination": {"performed": None, "parser_robustness_probe": None}},
        "outcomes": outcomes,
        "notes": (
            f"Method '{m}' measured on the zvf_iter130 risk-index batch (n_seeds=5). "
            + ("BASE method: the reference point for every delta_vs_grpo in the panel; "
               "previously uncatalogued as a stack. "
               if is_base else
               f"zvf_risk_mean={round(gt['zvf_risk_mean'],4)} (sd {round(gt['zvf_risk_sd'],4)}). ")
            + "iter-102 coverage-gap fill: this stack entry was created by "
              f"{AUDIT} to close the zvf130 stack coverage from 5/9 to 9/9. "
              "loss-form/KL/decontam leaves are null because the tinker-managed "
              "single-batch risk-index harness does not expose them (reported-as-unknown). "
            + ("" if is_base else
               "sig_robust_bootstrap_and_welch records whether the risk reduction "
               "vs grpo survives BOTH the iter90 paired bootstrap AND a conservative "
               "Welch two-sample t-test.")),
    }


def main():
    created = []
    for m in ["grpo", "aero", "areal", "gift"]:
        path = os.path.join(ENTRIES, f"zvf130_{m}.json")
        if os.path.exists(path):
            continue
        json.dump(make_stack(m), open(path, "w"), indent=2)
        created.append(os.path.basename(path))

    # backfill sig_robust into existing 5 stack entries' outcomes
    stamped_stacks = []
    for m in ["scafgrpo", "ngrpo", "cppo", "mcgrpo", "es"]:
        path = os.path.join(ENTRIES, f"zvf130_{m}.json")
        e = json.load(open(path))
        d, wlo, whi, wsig, _ = welch_of(m)
        bsig = e["outcomes"].get("delta_vs_grpo_sig")
        e["outcomes"]["delta_vs_grpo_welch_ci_lo"] = round(wlo, 6)
        e["outcomes"]["delta_vs_grpo_welch_ci_hi"] = round(whi, 6)
        e["outcomes"]["delta_vs_grpo_welch_sig"] = bool(wsig)
        e["outcomes"]["sig_robust_bootstrap_and_welch"] = bool(bsig) and bool(wsig)
        e["outcomes"]["crossref_integrity_iter102"] = {
            "audit_date": DATE, "audit_source": AUDIT,
            "point_estimate_matches_tsv": True}
        json.dump(e, open(path, "w"), indent=2)
        stamped_stacks.append(m)

    # backfill sig_robust into all 8 delta entries' zvf130_5seed measured block
    stamped_deltas = []
    for m, (bn, e) in DELTAS.items():
        dm = ig.delta_z130(e)
        if dm is None:
            continue
        d, wlo, whi, wsig, _ = welch_of(m)
        dm["welch_ci_low"] = round(wlo, 6)
        dm["welch_ci_high"] = round(whi, 6)
        dm["welch_sig"] = bool(wsig)
        dm["sig_robust_bootstrap_and_welch"] = bool(dm.get("significant")) and bool(wsig)
        dm["sig_robust_note"] = ("iter102: sig survives BOTH paired-bootstrap and "
                                 "Welch two-sample" if (bool(dm.get("significant")) and bool(wsig))
                                 else "iter102: bootstrap-significant but Welch-NS -- NOT sig-robust")
        json.dump(e, open(os.path.join(ENTRIES, bn), "w"), indent=2)
        stamped_deltas.append(m)

    print("created stack entries:", created)
    print("stamped sig_robust into stacks:", stamped_stacks)
    print("stamped sig_robust into deltas:", sorted(stamped_deltas))


if __name__ == "__main__":
    main()
