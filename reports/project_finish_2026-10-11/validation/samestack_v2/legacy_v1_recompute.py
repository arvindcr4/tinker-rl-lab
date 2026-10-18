# Historical review contract only; not interchangeable with canonical v3 evidence.
import json, sys, math
import numpy as np

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from legacy_stats_lib import paired, holm, mde, mcnemar_exact

R = sys.argv[1] if len(sys.argv) > 1 else "platform_hybrid/experiments/results/"
R = R.rstrip("/") + "/"
from legacy_v2_analysis import load

full = load(R + "samestack_gsm8k_cot_full.json")
by = {}
for r in full["runs"]:
    if (
        type(r.get("heldout_post_acc")) not in (int, float)
        or not math.isfinite(r["heldout_post_acc"])
        or not 0 <= r["heldout_post_acc"] <= 1
    ):
        raise ValueError("finite stored accuracy in [0,1] required")
    if r["seed"] in by.get(r["arm"], {}):
        raise ValueError(f"Duplicate experiment key: {(r['arm'], r['seed'])}")
    for phase in ("pre", "post"):
        values = r[phase + "_correct"]
        if len(values) != r["n_eval"] or any(
            type(v) not in (int, bool) or v not in (0, 1) for v in values
        ):
            raise ValueError("incomplete or nonbinary evaluation")
    pc, qc = np.array(r["pre_correct"], dtype=int), np.array(r["post_correct"], dtype=int)
    if len(qc) != 200:
        raise ValueError("historical n_eval must equal 200")
    by.setdefault(r["arm"], {})[r["seed"]] = dict(
        pre=pc,
        post=qc,
        stored_post=r["heldout_post_acc"],
        len=np.mean([s["mean_comp_len"] for s in r["step_log"]]),
        elapsed=r["elapsed_seconds"],
    )
out = {}
for a, d in by.items():
    mism = [s for s, x in d.items() if abs(x["post"].mean() - x["stored_post"]) > 1e-9]
    pre_same = all((x["pre"] == next(iter(d.values()))["pre"]).all() for x in d.values())
    out[a] = {
        "post_by_seed": {s: x["post"].mean() for s, x in sorted(d.items())},
        "pre_mean": np.mean([x["pre"].mean() for x in d.values()]),
        "pre_identical_across_seeds": pre_same,
        "post_mean": np.mean([x["post"].mean() for x in d.values()]),
        "stored_mismatch_seeds": mism,
        "train_comp_len_mean": np.mean([x["len"] for x in d.values()]),
        "elapsed_s_mean": np.mean([x["elapsed"] for x in d.values()]),
    }
post = {a: {s: x["post"].mean() for s, x in d.items()} for a, d in by.items()}
c = {
    k: paired(post[x], post[y])
    for k, (x, y) in {
        "g8-ppo": ("grpo_g8", "ppo"),
        "g8-g2": ("grpo_g8", "grpo_g2"),
        "g2-ppo": ("grpo_g2", "ppo"),
    }.items()
}
h = holm({k: v["p_t"] if v["p_t"] is not None else 1.0 for k, v in c.items()})
hs = holm({k: v["p_signflip"] for k, v in c.items()})
for k in c:
    c[k]["holm_t"], c[k]["holm_sf"] = h[k], hs[k]
    c[k]["mde_n5"] = mde(c[k]["sd"], 5) if c[k]["sd"] is not None and c[k]["sd"] > 0 else None
# item-level: pooled discordant counts g8 vs ppo

for x, y in [("grpo_g8", "ppo"), ("grpo_g8", "grpo_g2")]:
    b = sum(int(((by[x][s]["post"] == 1) & (by[y][s]["post"] == 0)).sum()) for s in by[x])
    cc = sum(int(((by[x][s]["post"] == 0) & (by[y][s]["post"] == 1)).sum()) for s in by[x])
    c[f"{x}-{y}_pooled_discordant"] = {
        "x_only": b,
        "y_only": cc,
        "discordant_pairs": b + cc,
        "status": "no_discordance" if b + cc == 0 else "available",
        "naive_mcnemar_p": mcnemar_exact(b, cc),
    }
print(
    json.dumps(
        {
            "schema_version": "historical-v1-recompute-v2-candidate",
            "evidence_level": "stored_binary_scores_only_no_item_identity_verification",
            "power_scope": "unadjusted paired-t planning sensitivity, not family or equivalence power",
            "arms": out,
            "contrasts": c,
        },
        indent=1,
        default=float,
        allow_nan=False,
    )
)
