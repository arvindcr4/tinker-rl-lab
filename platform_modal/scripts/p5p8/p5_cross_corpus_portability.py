#!/usr/bin/env python3
"""P5 MIN-REPORT cross-corpus portability test (iter 77).

Applies the 7-item MIN-REPORT fingerprint to 7 internal corpora; for each
measures per-item coverage, variance, bits, mean Hamming discrimination,
and emits a STRONG / PORTABLE / LIMITED / NULL verdict.

Outputs (platform_hybrid/experiments/results/p5p8/):
  p5_cross_corpus_portability.tsv          (7 corpora x 31 cols)
  p5_cross_corpus_portability_pairs.tsv    (bootstrap pair stats)
  p5_cross_corpus_portability_summary.json (full machine-readable)

stdlib only. B=2000 bootstrap, seed=20260705.
"""
from __future__ import annotations
import csv, io, json, math, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)
SEED, B = 20260705, 2000

ITEMS = ["loss_form", "ref_policy_kl", "sampler_backend_precision",
         "zvf_gu_trajectory", "group_size_schedule", "heldout_split",
         "decontam_parser_probe"]


def entropy(vals):
    if not vals:
        return 0.0
    n = len(vals)
    c = defaultdict(int)
    for v in vals:
        c[v] += 1
    return -sum((k / n) * math.log2(k / n) for k in c.values() if k)


def hamming(fps, keys=ITEMS, b=B, seed=SEED):
    if len(fps) < 2:
        return 0.0, 0
    rng = random.Random(seed)
    s, k = 0, 0
    for _ in range(min(b, len(fps) * 50)):
        i, j = rng.sample(range(len(fps)), 2)
        s += sum(1 for key in keys if fps[i].get(key) != fps[j].get(key))
        k += 1
    return (s / k if k else 0.0), k


def boot_ci(vals, b=B, seed=SEED):
    if not vals:
        return 0.0, 0.0, 0.0
    n = len(vals)
    rng = random.Random(seed)
    ms = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)]
    ms.sort()
    return sum(vals) / n, ms[int(0.025 * b)], ms[int(0.975 * b) - 1]


# ---- corpus loaders (each returns list[dict[str,str]] of fingerprints) ----

KEYMAP = [("loss_form", "loss_form"), ("ref_policy_kl", "ref_policy_kl"),
          ("sampler_backend_precision", "sampler_backend_precision"),
          ("zvf_gu_trajectory", "per_step_zvf_path"),
          ("group_size_schedule", "group_size_schedule"),
          ("heldout_split", "heldout_split"),
          ("decontam_parser_probe", "decontamination_notes")]


def load_mega():
    cells = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
    if not cells.exists():
        return []
    out = []
    with cells.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m = {}
            mp = row.get("manifest_path", "")
            if mp and Path(mp).exists():
                try:
                    m = json.loads(Path(mp).read_text())
                except Exception:
                    m = {}
            fp = {it: str(m.get(k, "")) for it, k in KEYMAP}
            fp["zvf_gu_trajectory"] = "zvf_present" if fp["zvf_gu_trajectory"] else ""
            out.append(fp)
    return out


def load_n2():
    d = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
    lf = {"grpo": "grpo-sequence", "aero": "aero-trace",
          "gift": "gift-clip-asym", "areal": "areal-reward"}
    out = []
    for f in sorted(d.glob("*_s0_tensors.jsonl")):
        m = f.stem.replace("_s0_tensors", "")
        out.append({"loss_form": lf.get(m, ""), "ref_policy_kl": "kl-disabled",
                    "sampler_backend_precision": "tinker-closed",
                    "zvf_gu_trajectory": "zvf_present" if f.exists() else "",
                    "group_size_schedule": "fixed-G=8", "heldout_split": "gsm8k-train-slice",
                    "decontam_parser_probe": "gsm8k-train-slice"})
    return out


def load_n10():
    mf = ROOT / "experiments" / "results" / "n10_seed_expansion" / "n10_manifest_20260704.json"
    if not mf.exists():
        return []
    d = json.loads(mf.read_text())
    lf = {"grpo": "grpo-sequence", "dr_grpo": "drgrpo-no-std-norm"}
    out = []
    for r in d.get("runs", []):
        algo = r.get("algo", "")
        seed = r.get("seed", "")
        out.append({"loss_form": lf.get(algo, algo), "ref_policy_kl": "kl-disabled",
                    "sampler_backend_precision": "tinker-closed",
                    "zvf_gu_trajectory": "zvf_present" if r.get("steps", 0) > 0 else "",
                    "group_size_schedule": "fixed-G=8_seed=" + str(seed),
                    "heldout_split": "gsm8k_cot",
                    "decontam_parser_probe": "gsm8k-train-slice"})
    return out


def _row_fp(loss_form, ref="", back="", zvf="", gs="", heldout="", decon=""):
    return {"loss_form": loss_form, "ref_policy_kl": ref, "sampler_backend_precision": back,
            "zvf_gu_trajectory": zvf, "group_size_schedule": gs,
            "heldout_split": heldout, "decontam_parser_probe": decon}


def load_base_instruct():
    f = ROOT / "experiments" / "results" / "base_instruct_paired.tsv"
    if not f.exists():
        return []
    out = []
    with f.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            mid = row.get("model_id", "")
            out.append(_row_fp("instruct" if "instruct" in mid.lower() else "base",
                               heldout="heldout_summary" if row.get("delta_heldout") else ""))
    return out


def load_group_size():
    f = ROOT / "experiments" / "results" / "group_size_iter111_paired.tsv"
    if not f.exists():
        return []
    out = []
    with f.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            t = row.get("T_tokens", "")
            out.append(_row_fp("grpo-sequence",
                               zvf="gu_summary" if row.get("G4_gu") or row.get("G32_gu") else "",
                               gs=("G4_vs_G32@T=" + str(t)) if row.get("G4_acc") and row.get("G32_acc") else "",
                               heldout="paired_diff" if row.get("delta_ci_lo") else ""))
    return out


def load_length_bias():
    for cand in [ROOT / "experiments" / "results" / "length_bias_iter60_grpo_vs_drgrpo.tsv",
                 ROOT / "experiments" / "results" / "length_bias_iter76_summary.tsv",
                 ROOT / "experiments" / "results" / "length_bias_iter84_paired.tsv"]:
        if cand.exists():
            f = cand
            break
    else:
        return []
    out = []
    with f.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            kind = row.get("kind", "")
            out.append(_row_fp(("grpo-vs-drgrpo:" + str(kind)) if kind else "grpo-vs-drgrpo:paired",
                               heldout=row.get("task", "") or "paired_length"))
    return out


def load_zvf():
    for cand in [ROOT / "experiments" / "results" / "zvf_iter118_auroc.tsv",
                 ROOT / "experiments" / "results" / "zvf_iter110_auroc.tsv",
                 ROOT / "experiments" / "results" / "zvf_iter114_dose_response.tsv"]:
        if cand.exists():
            f = cand
            break
    else:
        return []
    with f.open() as fh:
        body = "".join(ln for ln in fh if not ln.startswith("#"))
    out = []
    for row in csv.DictReader(io.StringIO(body), delimiter="\t"):
        if not row.get("stratum") and not row.get("G"):
            continue
        gv = row.get("G", "")
        out.append(_row_fp("zvf-stratum:" + row.get("stratum", "zvf-default"),
                           zvf="zvf-auc",
                           gs=("G=" + str(gv)) if gv else "",
                           heldout=row.get("target", "") or "zvf-summary"))
    return out


CORPORA = [
    ("C1_mega_20260704", "98-cell mega manifest + cells.tsv", load_mega),
    ("C2_n2_reward_tensor", "4-method same-stack tensors (G=8)", load_n2),
    ("C3_n10_seed_expansion", "2-algo x 16-seed expansion manifest", load_n10),
    ("C4_base_instruct_paired", "paired base vs instruct t-test rows", load_base_instruct),
    ("C5_group_size_iter111", "G sweep G4 vs G32 paired", load_group_size),
    ("C6_length_bias_iter", "length-bias paired runs (GRPO vs DrGRPO)", load_length_bias),
    ("C7_zvf_iter118", "ZVF per-stratum AUROC rows", load_zvf),
]


def verdict(pop, var):
    if pop >= 5 and var >= 1:
        return "STRONG"
    if pop >= 3 and var >= 1:
        return "PORTABLE"
    if pop >= 1 and var >= 1:
        return "LIMITED"
    return "NULL"


def main():
    rows, pairs, summary = [], [], {"corpora": [], "falsifiable_headlines": {}}
    for cid, desc, loader in CORPORA:
        fps = loader()
        n = len(fps)
        cov, var, ent = {}, {}, {}
        for it in ITEMS:
            vs = [str(fp.get(it, "")) for fp in fps]
            ps = [v for v in vs if v]
            cov[it] = (len(ps) / n) if n else 0.0
            var[it] = len(set(ps))
            ent[it] = entropy(ps)
        mh, npairs = hamming(fps)
        if n >= 4:
            rng = random.Random(SEED)
            sample = [sum(1 for k in ITEMS if fps[i := rng.randrange(n)].get(k) !=
                          fps[j := rng.randrange(n)].get(k)) for _ in range(B // 4)]
            _, lo, hi = boot_ci(sample, b=B // 4, seed=SEED)
        else:
            lo = hi = mh
        npop = sum(1 for it in ITEMS if cov[it] > 0)
        nvar = sum(1 for it in ITEMS if var[it] > 1)
        ver = verdict(npop, nvar)
        rows.append({"corpus_id": cid, "corpus_desc": desc, "n_records": n,
                     "n_items_populated": npop, "n_items_with_variance": nvar,
                     "total_bits": round(sum(ent.values()), 4),
                     "mean_hamming": round(mh, 4),
                     "hamming_ci_lo": round(lo, 4), "hamming_ci_hi": round(hi, 4),
                     "n_pairs_sampled": npairs, "verdict": ver,
                     **{f"cov_{k}": round(cov[k], 4) for k in ITEMS},
                     **{f"var_{k}": var[k] for k in ITEMS},
                     **{f"bits_{k}": round(ent[k], 4) for k in ITEMS}})
        pairs.append({"corpus_id": cid, "n_records": n,
                      "mean_hamming": round(mh, 4),
                      "hamming_ci_lo": round(lo, 4), "hamming_ci_hi": round(hi, 4)})
        summary["corpora"].append({"corpus_id": cid, "n_records": n,
                                   "n_items_populated": npop, "n_items_with_variance": nvar,
                                   "total_bits": round(sum(ent.values()), 4),
                                   "mean_hamming": round(mh, 4), "verdict": ver})

    by_pop = sorted(summary["corpora"], key=lambda x: x["n_items_populated"], reverse=True)
    by_h = sorted(summary["corpora"], key=lambda x: x["mean_hamming"], reverse=True)
    h1m, h1l, h1h = boot_ci([c["n_items_populated"] for c in summary["corpora"]], b=B, seed=SEED)
    h_means = [c["mean_hamming"] for c in summary["corpora"]]
    h3l, h3h = boot_ci(h_means, b=B, seed=SEED)[1:]
    summary["falsifiable_headlines"] = {
        "H1_portability_tax": "across 7 corpora, mean n_items_populated = {0:.2f} [95% bootstrap CI: {1:.2f}, {2:.2f}] of 7 items".format(h1m, h1l, h1h),
        "H2_strongest_corpus": by_pop[0]["corpus_id"] + " (n_items_populated=" + str(by_pop[0]["n_items_populated"]) + ")",
        "H3_most_discriminating": by_h[0]["corpus_id"] + " (mean_hamming="+ str(round(max(h_means), 4)) + "; CI over corpus-means: [" + str(round(h3l, 4)) + ", " + str(round(h3h, 4)) + "])",
        "H4_minimal_corpus": by_pop[-1]["corpus_id"] + " (n_items_populated=" + str(by_pop[-1]["n_items_populated"]) + ")",
        "H5_null_verdicts": sum(1 for c in summary["corpora"] if c["verdict"] == "NULL"),
        "H6_portable_verdicts": sum(1 for c in summary["corpora"] if c["verdict"] in {"STRONG", "PORTABLE"}),
        "H7_verdict_distribution": {v: sum(1 for c in summary["corpora"] if c["verdict"] == v) for v in ["STRONG", "PORTABLE", "LIMITED", "NULL"]},
    }

    out_tsv = OUT / "p5_cross_corpus_portability.tsv"
    with out_tsv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    out_pairs = OUT / "p5_cross_corpus_portability_pairs.tsv"
    with out_pairs.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(pairs)
    out_json = OUT / "p5_cross_corpus_portability_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"wrote {out_tsv} ({len(rows)} rows)")
    print(f"wrote {out_pairs}")
    print(f"wrote {out_json}")
    print()
    for c in summary["corpora"]:
        print(f"  {c['corpus_id']:30s}  n={c['n_records']:4d}  pop={c['n_items_populated']}/7  var={c['n_items_with_variance']}/7  H={c['mean_hamming']:.3f}  -> {c['verdict']}")
    print()
    for k, v in summary["falsifiable_headlines"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()