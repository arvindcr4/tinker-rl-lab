#!/usr/bin/env python3
"""P6 Registry Validation: schema check + measured variant deltas from N2 tensors.

Three deliverables:
  1. platform_hybrid/experiments/results/p5p8/registry_schema_check.tsv
     - one row per registry/entries/*.json
     - PASS / FAIL on schema validation + a leaf-coverage table per MIN-REPORT item
  2. platform_hybrid/experiments/results/p5p8/registry_measured_deltas.tsv
     - one row per (method_pair) on N2 reward tensors (same stack, G=8, seed=0)
     - measured delta_reward_mean, delta_zvf, delta_loss, paired bootstrap CI
  3. platform_hybrid/experiments/results/p5p8/registry_measured_deltas.json
     - machine-readable dump (also consumed by paper_P6 §measured-evidence patch)

This is the P6 T3 item: validate entries against what the N2 four-method run
actually logged. Per the P5P8 brief, prototype on real data.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

import jsonschema

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT  # noqa: E402
WORKTREE = REPO_ROOT
REGISTRY = WORKTREE / "platform_hybrid/registry"
RESULTS = WORKTREE / "platform_hybrid/experiments" / "results" / "p5p8"
N2 = WORKTREE / "platform_hybrid/experiments" / "results" / "n2_reward_tensor_resume"
MIN_REPORT_ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
                    "group_size_schedule", "heldout_split", "decontamination"]


# ---------------------------------------------------------------------------
# 1. Schema validation + per-leaf coverage table
# ---------------------------------------------------------------------------
def leaf_values(d):
    """Yield (path, value) for every leaf in a (possibly nested) dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in leaf_values(v):
                yield f"{k}.{k2}", v2
        else:
            yield k, v


def coverage(d):
    """Fraction of leaf fields in a MIN-REPORT item that are non-null."""
    leaves = list(leaf_values(d))
    if not leaves:
        return 0.0, 0, 0
    nonnull = sum(1 for _, v in leaves if v is not None)
    return nonnull / len(leaves), nonnull, len(leaves)


def schema_check():
    schema = json.loads((REGISTRY / "schema.json").read_text())
    out_rows = []
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        rid = rec.get("id", p.stem)
        rec_type = rec.get("record_type", "?")
        # try schema
        try:
            jsonschema.validate(rec, schema)
            ok = "PASS"
            err = ""
        except jsonschema.ValidationError as e:
            ok = "FAIL"
            err = str(e.message)[:80]
        # coverage (stack records only)
        cov_per_item = {}
        if rec_type == "stack" and "min_report" in rec:
            for it in MIN_REPORT_ITEMS:
                c, n, t = coverage(rec["min_report"].get(it, {}))
                cov_per_item[it] = (c, n, t)
            total = sum(c for c, _, _ in cov_per_item.values()) / len(MIN_REPORT_ITEMS)
            badge = round(100 * total)
        else:
            cov_per_item = {it: (1.0, 0, 0) for it in MIN_REPORT_ITEMS}
            badge = 100 if rec_type == "variant_delta" else 0
        out_rows.append({
            "id": rid,
            "record_type": rec_type,
            "schema": ok,
            "badge": badge,
            "error": err,
            **{f"cov_{it}": f"{cov_per_item[it][1]}/{cov_per_item[it][2]}" for it in MIN_REPORT_ITEMS},
        })
    return out_rows


# ---------------------------------------------------------------------------
# 2. Measured variant deltas from N2 tensors
# ---------------------------------------------------------------------------
def load_n2_metrics():
    """Parse the n2_metrics.tsv file: per-(method,seed,step) scalar rollouts."""
    metrics = {}
    with (N2 / "n2_metrics.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["method"], int(row["seed"]), int(row["step"]))
            metrics[key] = {
                "zvf": float(row["zvf"]),
                "reward_mean": float(row["reward_mean"]),
                "loss": float(row["loss"]),
                "mean_len": float(row["mean_len"]),
                "cv_len": float(row["cv_len"]),
                "frac_all_zero": float(row["frac_all_zero"]),
                "frac_all_one": float(row["frac_all_one"]),
                "pcd": float(row["pcd"]),
                "larq": float(row["larq"]),
            }
    return metrics


def load_n2_tensor_prompts():
    """For each method/seed/step, re-derive per-step *per-prompt* mean reward from
    the JSONL tensors (this is what each method actually saw as a training signal
    summary, NOT the scalar reward_mean column). Returns a dict keyed by
    (method, seed, step) -> list[float] of per-prompt mean rewards.

    Why this is useful: the same-stack delta is more honestly measured at the
    per-prompt reward level than at the per-step scalar level, because the
    scalar already pools over groups with different contrast structures.
    """
    out = {}
    for method in ("grpo", "aero", "gift", "areal"):
        path = N2 / f"{method}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["method"], rec["seed"], rec["step"])
                # per-prompt mean over G rollouts
                ids = rec.get("prompt_ids")
                rewards = rec["rewards"]
                if not isinstance(ids, list) or len(ids) != len(rewards) or len(set(ids)) != len(ids):
                    raise ValueError("tensor rows require unique explicit prompt_ids aligned to rewards")
                if key in out: raise ValueError("duplicate method/seed/step")
                out[key] = {pid: statistics.fmean(g) for pid,g in zip(ids,rewards)}
    return out


from review22_contract import bootstrap_paired_diff, PROTOCOL_ID


def measured_deltas(metrics):
    """Paired seed means; within-seed last-ten steps are never replications."""
    fields = ("reward_mean", "zvf", "loss", "mean_len", "cv_len", "frac_all_zero", "frac_all_one", "pcd", "larq")
    out=[]
    for other in ("aero", "gift", "areal"):
        seeds={k[1] for k in metrics if k[0]=="grpo"}
        if seeds != {k[1] for k in metrics if k[0]==other} or not seeds:
            raise ValueError("matched nonempty seed identities required")
        for field in fields:
            a=[]; b=[]; counts={}
            for seed in sorted(seeds):
                steps=sorted(k[2] for k in metrics if k[:2]==("grpo",seed))
                if steps != sorted(k[2] for k in metrics if k[:2]==(other,seed)):
                    raise ValueError("same ordered step identities required")
                selected=steps[-10:]; counts[str(seed)]=len(selected)
                a.append(statistics.fmean(metrics[("grpo",seed,step)][field] for step in selected))
                b.append(statistics.fmean(metrics[(other,seed,step)][field] for step in selected))
            delta=statistics.fmean(x-y for x,y in zip(a,b))
            lo=hi=None
            if len(seeds)>=2:
                _,lo,hi,_=bootstrap_paired_diff(a,b,2000)
            out.append(dict(metric=field,baseline="grpo",variant=other,n_seeds=len(seeds),
                n_steps=sum(counts.values()),steps_per_seed=counts,grpo_mean=statistics.fmean(a),
                variant_mean=statistics.fmean(b),paired_delta=delta,ci_lo=lo,ci_hi=hi,
                ci_excludes_0=None if lo is None else lo>0 or hi<0,
                uncertainty_unit="independent seed; independence must be justified",
                status="descriptive_single_seed" if len(seeds)<2 else "exploratory_seed_bootstrap"))
    return out


def measured_per_prompt(prompts):
    """Join explicit (seed,prompt ID), retaining repeated step membership."""
    grouped={m:{} for m in ("grpo","aero","gift","areal")}
    for (method,seed,step), values in prompts.items():
        if not isinstance(values,dict):
            raise ValueError("explicit prompt IDs required; array position is not identity")
        for prompt_id,reward in values.items():
            if not isinstance(prompt_id,str) or not prompt_id or not isinstance(reward,(int,float)) or not math.isfinite(reward):
                raise ValueError("valid prompt identity and finite reward required")
            grouped[method].setdefault((seed,prompt_id),{})[step]=reward
    rows=[]
    for other in ("aero","gift","areal"):
        keys=set(grouped['grpo'])
        if not keys or keys != set(grouped[other]):
            raise ValueError("same explicit seed/prompt identities required")
        per_seed={}; counts={}
        for key in sorted(keys):
            av,bv=grouped['grpo'][key],grouped[other][key]
            if av.keys()!=bv.keys(): raise ValueError("same prompt step observations required")
            per_seed.setdefault(key[0],[]).append((statistics.fmean(av.values()),statistics.fmean(bv.values())))
            counts[str(key)]=len(av)
        a=[statistics.fmean(x for x,_ in v) for v in per_seed.values()]
        b=[statistics.fmean(y for _,y in v) for v in per_seed.values()]
        lo=hi=None
        if len(a)>=2: _,lo,hi,_=bootstrap_paired_diff(a,b,2000)
        rows.append(dict(scope="seed_aggregated_prompt_means",baseline="grpo",variant=other,
            n_prompts=len(keys),n_seeds=len(a),steps_per_prompt=counts,
            grpo_mean=statistics.fmean(a),variant_mean=statistics.fmean(b),paired_delta=statistics.fmean(x-y for x,y in zip(a,b)),
            ci_lo=lo,ci_hi=hi,ci_excludes_0=None if lo is None else lo>0 or hi<0))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write outputs under platform_hybrid/experiments/results/p5p8/")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("# 1. Schema validation + MIN-REPORT coverage")
    rows = schema_check()
    cols = ["id", "record_type", "schema", "badge"] + \
           [f"cov_{it}" for it in MIN_REPORT_ITEMS] + ["error"]
    if args.write:
        with (RESULTS / "registry_schema_check.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    cov_keys = [f"cov_{it}" for it in MIN_REPORT_ITEMS]
    for r in rows:
        cov_str = "  ".join(f"{it}={r[k]}" for it, k in zip(MIN_REPORT_ITEMS, cov_keys))
        print(f"  {r['id']:40s} {r['schema']:4s} badge={r['badge']:3d}  {cov_str}")
    print(f"  ({len(rows)} entries; "
          f"{sum(r['schema']=='PASS' for r in rows)}/{len(rows)} pass)")

    print()
    print("# 2. Measured variant deltas from N2 tensors (seed 0, last 10 steps)")
    metrics = load_n2_metrics()
    deltas = measured_deltas(metrics)
    if args.write:
        with (RESULTS / "registry_measured_deltas.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()), delimiter="\t")
            w.writeheader()
            for d in deltas:
                w.writerow(d)
    for d in deltas:
        print(f"  {d['metric']:18s} grpo vs {d['variant']:5s}  "
              f"Δ={d['paired_delta']:+.4f}  CI=[{d['ci_lo']},{d['ci_hi']}]  "
              f"sig={d['ci_excludes_0']}")

    print()
    print("# 3. Per-prompt pooled reward-mean deltas")
    prompts = load_n2_tensor_prompts()
    pp = measured_per_prompt(prompts)
    for d in pp:
        print(f"  {d['baseline']:5s} vs {d['variant']:5s} (n={d['n_prompts']:4d} prompts)  "
              f"Δ={d['paired_delta']:+.4f}  CI=[{d['ci_lo']},{d['ci_hi']}]  "
              f"sig={d['ci_excludes_0']}")

    if args.write:
        out_json = {
            "protocol_id": PROTOCOL_ID,
            "schema_check_summary": {
                "n_total": len(rows),
                "n_pass": sum(r["schema"] == "PASS" for r in rows),
                "n_stack": sum(r["record_type"] == "stack" for r in rows),
                "n_variant_delta": sum(r["record_type"] == "variant_delta" for r in rows),
                "by_record": rows,
            },
            "measured_deltas_stepwise": deltas,
            "measured_deltas_per_prompt": pp,
            "source": "platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl",
            "note": "Descriptive or exploratory contrasts; matching seed/step IDs alone does not establish equivalent executed stacks or causal isolation.",
        }
        (RESULTS / "registry_measured_deltas.json").write_text(
            json.dumps(out_json, indent=2))
        print(f"\nwrote {RESULTS}/registry_schema_check.tsv")
        print(f"wrote {RESULTS}/registry_measured_deltas.{{tsv,json}}")


if __name__ == "__main__":
    sys.exit(main())