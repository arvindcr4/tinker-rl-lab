"""Pure historical v1 recomputation; never runs at import time.

This is a retrospective descriptive audit, not a declaration of prespecification.
CLI output goes to stdout only; historical stored result files are not modified.
"""
import argparse
import json
import math
from pathlib import Path


def load_runs(directory):
    from platform_hybrid.experiments.samestack_contract import strict_json_load
    return strict_json_load(Path(directory) / "samestack_gsm8k_cot_full.json")["runs"]


def validate_runs(records):
    seen = set()
    for record in records:
        identity = (record.get("arm"), record.get("seed"))
        if not isinstance(identity[0], str) or type(identity[1]) is not int:
            raise ValueError("invalid v1 arm/seed identity")
        if identity in seen:
            raise ValueError(f"duplicate v1 identity: {identity}")
        seen.add(identity)
        if record.get("n_eval") != 200:
            raise ValueError("v1 n_eval must be 200")
        for field in ("pre_correct", "post_correct"):
            values = record.get(field)
            if (not isinstance(values, list) or len(values) != 200 or
                    any(type(v) is not int or v not in (0, 1) for v in values)):
                raise ValueError(f"{field} must contain 200 binary integers")
        value = record.get("heldout_post_acc")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("stored post accuracy must be finite")
    return records


def analyze(records):
    try:
        from .stats_lib import holm, mcnemar_exact, mde, paired
        from .analysis_plan import PRIMARY, EXPLORATORY
    except ImportError:
        from stats_lib import holm, mcnemar_exact, mde, paired
        from analysis_plan import PRIMARY, EXPLORATORY
    validate_runs(records)
    by = {}
    for record in records:
        by.setdefault(record["arm"], {})[record["seed"]] = record
    out = {}
    for arm, runs in by.items():
        post = {seed: sum(run["post_correct"]) / 200 for seed, run in sorted(runs.items())}
        first_pre = next(iter(runs.values()))["pre_correct"]
        out[arm] = {"post_by_seed": post, "post_mean": sum(post.values()) / len(post),
                    "pre_mean": sum(sum(run["pre_correct"]) / 200 for run in runs.values()) / len(runs),
                    "pre_identical_across_seeds": all(run["pre_correct"] == first_pre for run in runs.values()),
                    "stored_mismatch_seeds": [seed for seed, run in runs.items() if not math.isclose(post[seed], run["heldout_post_acc"], abs_tol=1e-12, rel_tol=0)]}
    post = {arm: values["post_by_seed"] for arm, values in out.items()}
    primary, exploratory = {}, {}
    for family, target in ((PRIMARY, primary), (EXPLORATORY, exploratory)):
        for key, (a, b) in family.items():
            shared = set(post.get(a, {})) & set(post.get(b, {}))
            value = {"status": "unavailable", "n": len(shared), "mean": None, "p_t": 1.0, "p_signflip": 1.0, "tost_p": 1.0}
            if len(shared) >= 2:
                value = {"status": "available", **paired(post[a], post[b])}
                value["pilot_mde_n5"] = mde(value["sd"], 5) if value["sd"] > 0 else None
            target[key] = value
    for source, destination in (("p_t", "holm_p_t"), ("p_signflip", "holm_p_signflip"), ("tost_p", "holm_tost_p")):
        adjusted = holm({key: value[source] for key, value in primary.items()})
        for key, value in primary.items():
            value[destination] = adjusted[key]
    items = {}
    for key, (a, b) in PRIMARY.items():
        seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
        if not seeds:
            continue
        pairs = [(x, y) for seed in seeds for x, y in zip(by[a][seed]["post_correct"], by[b][seed]["post_correct"])]
        b_only, a_only = sum(x == 0 and y == 1 for x, y in pairs), sum(x == 1 and y == 0 for x, y in pairs)
        items[key] = {"x_only": a_only, "y_only": b_only, "naive_mcnemar_p": mcnemar_exact(a_only, b_only),
                      "scope": "Nonprimary pooled diagnostic; repeated seed-item observations are not independent"}
    return {"status": "historical_descriptive", "final_eligible": False,
            "provenance": "Historical v1 records have no independently verified environment/protocol/item manifest; this is not a canonical selected cohort",
            "family_note": "Retrospective fixed family with missing exposure-matched contrast p=1; G2-PPO exploratory. No retroactive prespecification claim",
            "arms": out, "primary_contrasts": primary, "exploratory_contrasts": exploratory,
            "item_level": items, "power_scope": "Pilot unadjusted t sensitivity only; not actual-design family or TOST power"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="platform_hybrid/experiments/results")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(load_runs(args.directory)), indent=2, allow_nan=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
