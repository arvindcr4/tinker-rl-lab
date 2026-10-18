#!/usr/bin/env python3
"""Review17-v2 missing-method coverage proposals.
Only supplied source facts are retained. No seed identities, sampler metadata,
matched baselines, or verified TSV comparisons are fabricated. No original registry
entries are changed; these proposals require source-manifest review before application."""
import json
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT
ROOT = REPO_ROOT
ENTRIES = ROOT / "platform_hybrid/registry" / "entries"
TSV_130 = ROOT / "platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv"
META_130 = ROOT / "platform_hybrid/experiments/results/zvf_iter130_meta.json"
OUT_DIR = ROOT / "platform_hybrid/experiments/results/p5p8/review17_v2"


DATE = "2026-07-05"
AUDIT = "platform_modal/scripts/p5p8/p6_iter138_missing_method_audit.py"

GRPO_METHODS_KNOWN = {
    "grpo", "aero", "areal", "gift", "ngrpo", "cppo", "mcgrpo", "es", "scafgrpo",
    "adaptiveg", "dapo", "drgrpo", "gspo", "liteppo", "ppo", "reinforce",
    "tool_use_llama-8b-inst", "tool_use_qwen3-32b",
}

# anchor rows are *not* GRPO methods — they are scaling-law / extrapolation
# anchor points for the iter-130 risk index; they get NO registry entry.
ANCHOR_METHODS = {
    "scaling_law_Qwen3-8B", "scaling_law_Nemotron-120B",
    "scaling_law_Llama-3.1-8B-Instruct", "scaling_law_Qwen3.5-4B",
    "scaling_law_DeepSeek-V3.1",
}

# Already in registry as variant_delta or stack (resolved dynamically below).
REGISTRY_IDS = set()


def load_registry_ids():
    REGISTRY_IDS.clear()
    for p in sorted(ENTRIES.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            REGISTRY_IDS.add(d["id"])
        except Exception:
            pass


def load_zvf130_methods():
    """Return (per_method_rows, method_to_tsv_row)."""
    rows = json.loads(META_130.read_text())["per_method"]
    tsv_rows = {}
    with open(TSV_130) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header):
                continue
            d = dict(zip(header, cells))
            tsv_rows[d["method"]] = d
    return rows, tsv_rows


def classify(method):
    # A method is PRESENT if either zvf130_<m> or delta_<m> exists.
    if f"zvf130_{method}" in REGISTRY_IDS or f"delta_{method}" in REGISTRY_IDS:
        return "PRESENT"
    if method in ANCHOR_METHODS:
        return "ANCHOR_ROW"
    if method in GRPO_METHODS_KNOWN:
        return "MISSING_GRPO_METHOD"
    return "MISSING_GRPO_METHOD"


def make_stack(method, m):
    """Coverage proposal: preserve supplied facts, never reconstruct a run from n."""
    seed_ids = m.get("seed_ids")
    if seed_ids is not None and len(set(seed_ids)) != int(m["n_seeds"]):
        raise ValueError("Seed identities disagree with count")
    delta = make_delta(method, m)
    return {
        "record_type": "stack", "schema_version": "0.1.0",
        "id": f"zvf130_{method}", "label_claimed": method,
        "framework": {"name": "worktree-zvf130-batch", "version": None, "openness": "open"},
        "model": m.get("model"), "task": m.get("task"), "seeds": seed_ids,
        "provenance": {"source_artifacts": [str(TSV_130.relative_to(ROOT))]},
        "variant_deltas_applied": [] if method == "grpo" else [{
            "delta_id": delta["id"], "component": delta["deltas"][0]["component"],
            "status": "planned", "note": "Coverage only; application unverified"}],
        "min_report": {
            "loss_form": {}, "reference_kl": {},
            "sampler_backend": m.get("sampler_backend", {}), "telemetry": {},
            "group_size_schedule": {}, "heldout_split": m.get("heldout_split", {}),
            "decontamination": {}},
        "outcomes": {"zvf_risk_mean": float(m["zvf_risk_mean"]),
                     "n_seeds": int(m["n_seeds"]),
                     "delta_vs_grpo_mean": None,
                     "crossref_integrity_iter138": {"point_estimate_matches_tsv": None,
                         "status": "UNVERIFIED", "audit_source": AUDIT}},
        "notes": "review17-v2 coverage proposal; unknown run metadata remains unknown. No paired baseline or inference is asserted."
    }


def make_delta(method, m):
    """Unmatched task/control observations cannot establish a measured delta."""
    return {"record_type": "variant_delta", "schema_version": "0.1.0",
            "id": f"delta_{method}", "name": method, "base": "grpo",
            "citation": {"bibkey": None, "arxiv": None, "title": None},
            "deltas": [{"component": "task_substitution" if "tool_use" in method else "unverified",
                        "field": "unknown", "change": "Coverage proposal; requires explicit source manifests"}],
            "measured": [],
            "notes": f"Reported n_seeds={int(m['n_seeds'])}; no matched baseline or seed identities verified."}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_registry_ids()
    methods_meta, _ = load_zvf130_methods()

    # Build the missing-method audit TSV
    rows = []
    for m in methods_meta:
        method = m["method"]
        verdict = classify(method)
        rows.append({
            "method": method,
            "verdict": verdict,
            "zvf_risk_mean": m["zvf_risk_mean"],
            "n_seeds": m["n_seeds"],
            "failure_rate": m["failure_rate"],
            "registry_id_present": (f"zvf130_{method}" in REGISTRY_IDS
                                    or f"delta_{method}" in REGISTRY_IDS),
            "iter138_action": {
                "PRESENT": "none",
                "ANCHOR_ROW": "skip — not a GRPO method (scaling-law anchor)",
                "MISSING_GRPO_METHOD": (
                    f"create zvf130_{method}.json + delta_{method}.json"
                ),
            }[verdict],
        })

    out_tsv = OUT_DIR / "p6_iter138_missing_method_audit.tsv"
    with open(out_tsv, "w") as fh:
        cols = ["method", "verdict", "zvf_risk_mean", "n_seeds",
                "failure_rate", "registry_id_present", "iter138_action"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Create the missing entries (idempotent — skip if exists)
    created = []
    skipped = []
    for r in rows:
        if r["verdict"] != "MISSING_GRPO_METHOD":
            continue
        method = r["method"]
        m = next(mm for mm in methods_meta if mm["method"] == method)

        stack_path = OUT_DIR / f"zvf130_{method}.json"
        delta_path = OUT_DIR / f"delta_{method}.json"

        if not stack_path.exists():
            json.dump(make_stack(method, m), open(stack_path, "w"), indent=2)
            created.append(str(stack_path.relative_to(ROOT)))
        else:
            skipped.append(str(stack_path.relative_to(ROOT)))

        # grpo is the BASE method — no delta_grpo entry exists by design
        # (delta_X records what X CHANGES about grpo).
        if method == "grpo":
            continue

        if not delta_path.exists():
            json.dump(make_delta(method, m), open(delta_path, "w"), indent=2)
            created.append(str(delta_path.relative_to(ROOT)))
        else:
            skipped.append(str(delta_path.relative_to(ROOT)))

    summary = {
        "iter": 138,
        "pillar": "P6",
        "audit_date": DATE,
        "n_methods_in_data": len(methods_meta),
        "n_registry_entries_before": len(REGISTRY_IDS),
        "verdicts": {
            "PRESENT": sum(1 for r in rows if r["verdict"] == "PRESENT"),
            "ANCHOR_ROW": sum(1 for r in rows if r["verdict"] == "ANCHOR_ROW"),
            "MISSING_GRPO_METHOD": sum(1 for r in rows if r["verdict"] == "MISSING_GRPO_METHOD"),
        },
        "missing_methods": [r["method"] for r in rows if r["verdict"] == "MISSING_GRPO_METHOD"],
        "anchor_methods": [r["method"] for r in rows if r["verdict"] == "ANCHOR_ROW"],
        "files_created": created,
        "files_skipped_existing": skipped,
        "audit_source": AUDIT,
    }
    out_json = OUT_DIR / "p6_iter138_entry_summary.json"
    json.dump(summary, open(out_json, "w"), indent=2)

    # stdout summary
    print(f"iter-138 missing-method audit complete")
    print(f"  methods in zvf130 data: {len(methods_meta)}")
    print(f"  registry entries before: {len(REGISTRY_IDS)}")
    for k, v in summary["verdicts"].items():
        print(f"  {k}: {v}")
    print(f"  missing methods: {summary['missing_methods']}")
    print(f"  anchor methods: {summary['anchor_methods']}")
    print(f"  files created: {len(created)}")
    for f in created:
        print(f"    + {f}")
    print(f"  files skipped (existing): {len(skipped)}")
    print(f"  audit tsv: {out_tsv.relative_to(ROOT)}")
    print(f"  summary json: {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())