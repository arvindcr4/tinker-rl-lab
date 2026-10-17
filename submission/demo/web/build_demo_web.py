"""Export repository evidence for the TinkerRL defense web demo (React app).

Reads stored results only (no training, no network) and writes src/data.json.
Then build the offline single-file page with `npm run build` (see README.md).
"""

import gzip
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
V2 = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s"
V1 = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/v1_recompute.json"
GOLD = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/gsm8k_test_gold.json"
AUDIT = ROOT / "reports/public_revision_2026-10-11/PUBLIC_CHECKOUT_NUMBER_AUDIT.json"
PLATFORM = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/platform_calibration.json"
ARMS = ["grpo_g8", "grpo_g2", "ppo", "ppo_8x8"]

sys.path.insert(0, str(ROOT / "platform_hybrid/registry"))
import query as registry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samestack():
    runs, sources = [], []
    for arm in ARMS:
        for path in sorted(V2.glob(f"{arm}_s*.json")):
            r = json.loads(path.read_text())
            if r["arm"] != arm or r.get("smoke"):
                continue
            runs.append({
                "arm": arm, "seed": r["seed"],
                "pre": r["heldout_pre_acc"], "post": r["heldout_post_acc"],
                "post_trunc": r["heldout_post_trunc"], "n_eval": r["n_eval"],
                "w2r": r["wrong_to_right"], "r2w": r["right_to_wrong"],
                "steps": [{"r": s["mean_reward"], "zvf": s["zvf"]} for s in r["step_log"]
                          if s.get("zvf") is not None and s["zvf"] == s["zvf"]],
            })
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    first = json.loads(next(V2.glob("grpo_g8_s*.json")).read_text())
    setup = {k: first[k] for k in ("model", "n_steps", "max_new", "lr", "k_epochs", "n_gen", "n_eval")}
    return {"runs": runs, "setup": setup, "sources": sources, "seeds_planned": 10}


def stackdiff():
    recs = registry.load("stack")
    entries = [{"id": rid, "label": r["label_claimed"],
                "framework": f"{r['framework']['name']}/{r['framework']['openness']}"}
               for rid, r in sorted(recs.items())]
    pairs = {}
    for (a, b), r in registry.all_pairs(recs).items():
        pairs[f"{a}|{b}"] = {
            "level": r["level"], "coverage": round(r["coverage"], 4),
            "underspecified": r["underspecified"], "same_label": r["same_label"],
            "differs": [[f, s, x, y] for f, s, x, y in sorted(r["differs"], key=lambda d: -d[1])],
            "unknown_n": len(r["unknown"]),
            "unknown_r5": sorted({u[0] for u in r["unknown"] if u[1] == 5}),
        }
    return {"entries": entries, "pairs": pairs, "risk": registry.RISK}


def examples(seed=42, per_pattern=4):
    """Stored greedy completions for one seed: base, GRPO G8 and PPO on the same test items.

    Items are grouped by outcome pattern (base, GRPO, PPO correct?) over all test items;
    the first `per_pattern` items by test index are taken from each pattern, so the sample
    is fixed by rule rather than chosen by hand.
    """
    gold = json.loads(GOLD.read_text())["test"]
    runs = {a: json.loads((V2 / f"{a}_s{seed}.json").read_text()) for a in ("grpo_g8", "ppo")}
    texts = {a: json.loads(gzip.open(V2 / f"{a}_s{seed}_texts.json.gz", "rt").read()) for a in runs}
    if texts["grpo_g8"]["pre"] != texts["ppo"]["pre"]:
        raise ValueError("base completions differ between arms")
    g, p = runs["grpo_g8"], runs["ppo"]
    patterns = {}
    for i in range(len(gold)):
        patterns.setdefault(f"{g['pre_correct'][i]}{g['post_correct'][i]}{p['post_correct'][i]}", []).append(i)
    items = []
    for key, idx in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
        for i in idx[:per_pattern]:
            items.append({
                "i": i, "pattern": key, "question": gold[i]["question"],
                "gold": gold[i]["answer"].split("####")[-1].strip().replace(",", ""),
                "base": {"text": texts["grpo_g8"]["pre"][i], "stored": g["pre_correct"][i], "trunc": g["pre_trunc"][i]},
                "grpo_g8": {"text": texts["grpo_g8"]["post"][i], "stored": g["post_correct"][i], "trunc": g["post_trunc"][i]},
                "ppo": {"text": texts["ppo"]["post"][i], "stored": p["post_correct"][i], "trunc": p["post_trunc"][i]},
            })
    return {"seed": seed, "n": len(gold), "pattern_counts": {k: len(v) for k, v in patterns.items()},
            "per_pattern": per_pattern, "items": items}


def grader_check():
    """Run the in-page grader over every stored completion (tests/grade_all.mjs) and record the result."""
    out = subprocess.run(["node", str(HERE / "tests/grade_all.mjs"), "--json"], capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def samestack_v1():
    """v1 (200-token cap, cold critic, 200 items): superseded, shown for the correction story."""
    if not V1.exists():
        return None
    d = json.loads(V1.read_text())
    arms = {a: {"pre": v["pre_mean"], "post_by_seed": v["post_by_seed"]} for a, v in d["arms"].items()}
    c = d["contrasts"]["g8-ppo"]
    return {"arms": arms, "g8_ppo": {k: c[k] for k in ("mean", "t_ci", "p_t", "p_signflip", "holm_sf")},
            "path": str(V1.relative_to(ROOT)), "sha256": sha(V1)}


def audit():
    d = json.loads(AUDIT.read_text())
    return {"status": d["status"], "summary": d["summary"], "token_counts": d["token_counts"],
            "path": str(AUDIT.relative_to(ROOT)), "sha256": sha(AUDIT)}


def platform():
    """Cross-platform calibration (Modal A10G vs Colab A100), if platform_calibration.py has been run."""
    if not PLATFORM.exists():
        return None
    d = json.loads(PLATFORM.read_text(), parse_constant=lambda _: None)
    return {**d, "path": str(PLATFORM.relative_to(ROOT)), "sha256": sha(PLATFORM)}


def main():
    data = {"samestack": samestack(), "samestack_v1": samestack_v1(), "examples": examples(), "grader_check": grader_check(), "stackdiff": stackdiff(), "audit": audit(), "platform": platform()}
    out = HERE / "src/data.json"
    out.write_text(json.dumps(data, separators=(",", ":"), default=str) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
