#!/usr/bin/env python3
"""GRPO-Registry reference CLI: list / query / badge / stackdiff / drift.

Stdlib only. Field convention (see schema.json): JSON null = unreported;
explicit false / 0.0 / "none" = reported-as-absent. The MIN-REPORT badge
scores reporting coverage, not configuration virtue.

Usage:
  python3 registry/query.py list
  python3 registry/query.py query --item reference_kl [--full]
  python3 registry/query.py badge [entry_id]
  python3 registry/query.py stackdiff <entry_id_a> <entry_id_b>
  python3 registry/query.py drift  # iter-42: schema-anchored `field:` claims
"""
import argparse
import json
import re
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]
# stackdiff severity ladder.  A level is assigned ONLY from fields that are
# known on both sides and differ; fields unknown on either side never raise
# the level -- they lower `coverage` and are reported as `underspecified`.
RISK = {0: "R0 no known differences",
        1: "R1 nuisance-only differences (seeds, trainer version, reporting)",
        2: "R2 runtime differences (trainer, sampler backend, precision, implementation hash)",
        3: "R3 hyperparameter / budget differences (decoding, batch shape, steps, optimizer, LoRA, data subset)",
        4: "R4 algorithm differences (loss form, advantage, clipping, KL, variant-delta set)",
        5: "R5 outcome-critical differences (checkpoint, reward, prompt/chat template, dataset, eval protocol, execution mode)"}
# Canonical comparison catalogue: field -> severity.  Values come from the
# entry's `run_config` block when present (generated from run sources with
# per-field provenance), otherwise from the hand-entered top-level fields and
# `min_report` (see legacy_view).  Order here is the print order.
FIELDS = {
    # R5 outcome-critical
    "execution.mode": 5, "checkpoint.id": 5, "checkpoint.variant": 5,
    "checkpoint.revision": 5, "reward.id": 5, "prompt.template_sha256": 5,
    "prompt.chat_template": 5, "data.dataset": 5, "eval.metric": 5,
    "eval.heldout": 5,
    # R4 algorithm
    "loss.objective": 4, "loss.importance_ratio": 4, "loss.clip": 4,
    "loss.aggregation": 4, "loss.advantage_norm": 4, "loss.token_mask": 4,
    "loss.length_normalization": 4, "loss.dynamic_sampling": 4,
    "loss.reward_shaping": 4, "loss.inner_epochs": 4,
    "kl.reference_policy": 4, "kl.beta": 4, "kl.estimator": 4,
    # R3 hyperparameters / budget
    "decoding.temperature": 3, "decoding.top_p": 3, "decoding.max_tokens": 3,
    "batch.prompts_per_step": 3, "batch.samples_per_prompt": 3,
    "batch.group_schedule": 3, "train.steps": 3, "train.sampler_refresh": 3,
    "data.subset": 3, "data.selection": 3,
    "optimizer.name": 3, "optimizer.lr": 3, "optimizer.betas": 3,
    "optimizer.eps": 3, "optimizer.weight_decay": 3,
    "lora.rank": 3, "lora.alpha": 3, "lora.dropout": 3, "lora.targets": 3,
    # R2 runtime
    "trainer.name": 2, "runtime.backend": 2, "runtime.precision": 2,
    "reward.fn_sha256": 2,
    # R1 nuisance
    "trainer.version": 1, "seeds": 1,
}
# Underspecified flag: coverage below this, or any R5 field unknown, except
# fields in FLAG_EXEMPT (checkpoint.revision is unpinned in every run script in
# this repo; it is still listed among the unknowns but cannot trigger the flag).
COVERAGE_FLOOR = 0.8
FLAG_EXEMPT = {"checkpoint.revision"}
# run_config status vocabulary; only "unknown" counts as not known.
KNOWN_STATUSES = ("logged", "code", "default", "inferred")


def load(kind=None):
    recs = {}
    for p in sorted((HERE / "entries").glob("*.json")):
        r = json.loads(p.read_text())
        # Amendment sidecars intentionally do not implement the registry-record
        # schema.  They live next to records for provenance, so ignore them in
        # commands that enumerate queryable stack/delta records.
        if "record_type" not in r or "id" not in r:
            continue
        if kind is None or r["record_type"] == kind:
            recs[r["id"]] = r
    return recs


def item_score(d):
    """Fraction of leaf fields in one MIN-REPORT item that are non-null."""
    leaves = list(d.values())
    return sum(v is not None for v in leaves) / len(leaves) if leaves else 0.0




def badge(rec):
    scores = {it: item_score(rec["min_report"][it]) for it in ITEMS}
    return round(100 * sum(scores.values()) / len(ITEMS)), scores


def _variant(model):
    """Base/instruct from the checkpoint name; None when the name is silent."""
    if not model:
        return None
    name = model.lower().split("/")[-1]
    if "base" in name:
        return "base"
    if "instruct" in name or name.endswith("-inst") or name.endswith("-it"):
        return "instruct"
    return None


def legacy_view(rec):
    """Map hand-entered top-level + min_report fields onto FIELDS names.

    Every value is tagged status="registry" (hand-entered, no per-field
    provenance).  JSON null -> unknown, per the registry convention."""
    mr = rec.get("min_report", {})
    lf, kl = mr.get("loss_form", {}), mr.get("reference_kl", {})
    sb, gs = mr.get("sampler_backend", {}), mr.get("group_size_schedule", {})
    clip = None
    if lf.get("clip_eps_low") is not None and lf.get("clip_eps_high") is not None:
        clip = f"ratio[{1 - lf['clip_eps_low']:g},{1 + lf['clip_eps_high']:g}]"
    seeds = rec.get("seeds")
    raw = {
        "checkpoint.id": rec.get("model"),
        "checkpoint.variant": _variant(rec.get("model")),
        "data.dataset": rec.get("task"),
        "trainer.name": rec.get("framework", {}).get("name"),
        "trainer.version": rec.get("framework", {}).get("version"),
        "seeds": sorted(seeds) if seeds else None,
        "loss.importance_ratio": lf.get("importance_ratio_level"),
        "loss.clip": clip,
        "loss.aggregation": lf.get("token_aggregation"),
        "loss.advantage_norm": lf.get("advantage_normalization"),
        "loss.token_mask": lf.get("token_mask"),
        "loss.length_normalization": lf.get("length_normalization"),
        "loss.dynamic_sampling": lf.get("sampling_dynamic_filter"),
        "loss.reward_shaping": lf.get("reward_shaping_type"),
        "kl.reference_policy": kl.get("reference_policy"),
        "kl.beta": kl.get("kl_beta"),
        "kl.estimator": kl.get("kl_estimator"),
        "decoding.temperature": sb.get("temperature"),
        "decoding.top_p": sb.get("top_p"),
        "runtime.backend": sb.get("backend"),
        "runtime.precision": sb.get("precision"),
        "batch.samples_per_prompt": gs.get("initial_g"),
        "batch.group_schedule": gs.get("schedule"),
    }
    return {k: {"value": v, "status": "registry" if v is not None else "unknown",
                "source": "registry entry (hand-entered)"} for k, v in raw.items()}


def field_view(rec):
    """Canonical {field: {value, status, source}} over every FIELDS name.

    run_config (generated, with provenance) overrides the legacy view."""
    view = legacy_view(rec)
    view.update(rec.get("run_config") or {})
    unknown = {"value": None, "status": "unknown", "source": "not recorded"}
    return {f: view.get(f, unknown) for f in FIELDS}


def is_known(cell):
    return cell.get("status") != "unknown" and cell.get("value") is not None


def spec_coverage(rec):
    """Fraction of FIELDS known for one entry (what stackdiff can compare)."""
    return sum(is_known(c) for c in field_view(rec).values()) / len(FIELDS)


# Comparison keys for identifier fields whose hand-entered values carry
# free-text annotations ("Qwen/Qwen3-8B-base (shared ... harness)") or
# aliases ("gsm8k" vs "openai/gsm8k:main").  Displayed values are unchanged.
DATASET_ALIASES = {"gsm8k": "openai/gsm8k:main"}


def _strip_note(v):
    return re.sub(r"\s*\(.*\)\s*$", "", v).strip().lower()


CANON = {
    "checkpoint.id": _strip_note,
    "data.dataset": lambda v: DATASET_ALIASES.get(_strip_note(v), _strip_note(v)),
}


def _key(field, value):
    f = CANON.get(field)
    return _norm(f(value) if f and isinstance(value, str) else value)


def _norm(x):
    if isinstance(x, bool):
        return ("bool", x)  # keep False distinct from 0.0
    if isinstance(x, (int, float)):
        return round(float(x), 12)
    if isinstance(x, list):
        return [_norm(i) for i in x]
    return x


def stackdiff(a, b):
    """Pure comparison of two stack records.

    Returns a dict with:
      level           0..5: max severity over fields KNOWN on both sides that differ
      differs         [(field, severity, value_a, value_b)]
      unknown         [(field, severity, side)], side in {"a", "b", "both"}
      undeclared      [(delta row, side missing it)]
      coverage        fraction of compared fields known on both sides
      hidden_max      max severity among unknown fields (what could still differ)
      underspecified  coverage < COVERAGE_FLOOR or any R5 field unknown
    Variant-delta rows are compared too: differing known statuses are R4;
    "unknown" on either side is reported as unknown; a row declared on one
    side only is listed in `undeclared` (never a difference, not in coverage).
    """
    va, vb = field_view(a), field_view(b)
    differs, unknown, known_both = [], [], 0
    for f, sev in FIELDS.items():
        ka, kb = is_known(va[f]), is_known(vb[f])
        if ka and kb:
            known_both += 1
            if _key(f, va[f]["value"]) != _key(f, vb[f]["value"]):
                differs.append((f, sev, va[f]["value"], vb[f]["value"]))
        else:
            side = "both" if not (ka or kb) else ("a" if not ka else "b")
            unknown.append((f, sev, side))
    da = {(d["delta_id"], d["component"]): d["status"]
          for d in a.get("variant_deltas_applied", [])}
    db = {(d["delta_id"], d["component"]): d["status"]
          for d in b.get("variant_deltas_applied", [])}
    undeclared, n_rows = [], 0
    for key in sorted(set(da) | set(db)):
        name = f"delta.{key[0]}:{key[1]}"
        if key not in da or key not in db:
            # Declared on one side only: not evidence of absence (a baseline
            # lists no deltas even when, e.g., its loss also has no KL term).
            # Reported, but neither a difference nor counted in coverage.
            undeclared.append((name, "a" if key not in da else "b"))
            continue
        n_rows += 1
        sa, sb = da[key], db[key]
        if "unknown" in (sa, sb):
            side = "both" if sa == sb else ("a" if sa == "unknown" else "b")
            unknown.append((name, 4, side))
        else:
            known_both += 1
            if sa != sb:
                differs.append((name, 4, sa, sb))
    coverage = known_both / (len(FIELDS) + n_rows)
    return {"level": max((d[1] for d in differs), default=0),
            "differs": differs, "unknown": unknown, "undeclared": undeclared,
            "coverage": coverage,
            "hidden_max": max((u[1] for u in unknown), default=0),
            "underspecified": coverage < COVERAGE_FLOOR or any(
                u[1] == 5 and u[0] not in FLAG_EXEMPT for u in unknown),
            "same_label": a["label_claimed"] == b["label_claimed"]}


def all_pairs(recs=None):
    """stackdiff over every unordered pair of stack records."""
    import itertools
    recs = recs if recs is not None else load("stack")
    return {(x, y): stackdiff(recs[x], recs[y])
            for x, y in itertools.combinations(sorted(recs), 2)}


def cmd_list(_):
    for r in load("stack").values():
        print(f"{r['id']:40s} label={r['label_claimed']:14s} "
              f"framework={r['framework']['name']}/{r['framework']['openness']}")


def cmd_badge(args):
    recs = load("stack")
    if args.entry:
        recs = {args.entry: recs[args.entry]}
    for rid, r in recs.items():
        total, scores = badge(r)
        detail = " ".join(f"{k}={v:.2f}" for k, v in scores.items())
        src = "run_config" if r.get("run_config") else "hand-entered"
        print(f"{rid:40s} badge={total:3d}  spec_coverage={spec_coverage(r):.2f} "
              f"({src})  {detail}")


def cmd_query(args):
    """Which stacks fully report a MIN-REPORT item? (e.g. --item reference_kl)"""
    for rid, r in load("stack").items():
        s = item_score(r["min_report"][args.item])
        status = "FULL" if s == 1.0 else ("PARTIAL" if s > 0 else "UNREPORTED")
        line = f"{rid:40s} {args.item}={status} ({s:.2f})"
        if args.full and status != "UNREPORTED":
            line += "  " + json.dumps(r["min_report"][args.item])
        print(line)


def cmd_stackdiff(args):
    from collections import Counter
    recs = load("stack")
    if getattr(args, "all", False):
        if getattr(args, "generated", False):
            recs = {k: v for k, v in recs.items() if v.get("run_config")}
        res = all_pairs(recs)
        print(f"stackdiff --all: {len(recs)} stacks, {len(res)} pairs")
        groups = [("all pairs", res),
                  ("same claimed label", {k: v for k, v in res.items() if v["same_label"]})]
        for title, sub in groups:
            n = len(sub)
            if not n:
                continue
            lv = Counter(r["level"] for r in sub.values())
            und = sum(r["underspecified"] for r in sub.values())
            covs = sorted(r["coverage"] for r in sub.values())
            print(f"{title} (n={n}):")
            for k in range(6):
                print(f"  R{k}: {lv[k]:4d}   {RISK[k]}")
            print(f"  underspecified: {und}/{n}   coverage median={covs[n // 2]:.2f} "
                  f"min={covs[0]:.2f} max={covs[-1]:.2f}")
        return 0
    if not (args.a and args.b):
        sys.exit("stackdiff needs two entry ids (or --all)")
    a, b = recs[args.a], recs[args.b]
    r = stackdiff(a, b)
    va, vb = field_view(a), field_view(b)
    print(f"stackdiff {a['id']} vs {b['id']}")
    if r["same_label"]:
        print(f"  shared claimed label: '{a['label_claimed']}'")
    for f, sev, x, y in sorted(r["differs"], key=lambda d: -d[1]):
        src = ""
        if getattr(args, "sources", False) and f in va:
            src = f"\n        a: {va[f]['source']}\n        b: {vb[f]['source']}"
        print(f"  [R{sev}] {f}: {x!r} -> {y!r}{src}")
    print(f"verdict: {RISK[r['level']]}")
    print(f"coverage: {r['coverage']:.2f} of compared fields known on both sides; "
          f"{len(r['unknown'])} unknown (max hidden severity R{r['hidden_max']})")
    if r["underspecified"]:
        crit = [u[0] for u in r["unknown"] if u[1] == 5 and u[0] not in FLAG_EXEMPT]
        print("UNDERSPECIFIED: " + (f"outcome-critical fields unknown: {', '.join(crit)}"
                                     if crit else f"coverage < {COVERAGE_FLOOR}"))
    if r["undeclared"]:
        print(f"undeclared delta rows (one side only, not compared): "
              + ", ".join(f"{n} [missing on {s}]" for n, s in r["undeclared"]))
    if getattr(args, "verbose", False):
        for f, sev, side in sorted(r["unknown"], key=lambda u: -u[1]):
            print(f"  [unknown R{sev}] {f} (missing on {side})")
    if r["same_label"] and r["level"] >= 4:
        print("label-flip risk: the shared label does not pin the stack;"
              "\n  do not attribute outcome differences to the label.")
    return r["level"]


def cmd_implementations(args):
    """Cross-reference: which stacks claim to implement a (delta, component)?

    Usage:
      python3 registry/query.py implementations --delta delta_dapo
      python3 registry/query.py implementations --status implemented
      python3 registry/query.py implementations --status unknown --framework worktree-zvf130-batch
    """
    recs = load("variant_delta") if args.delta else {}
    stacks = load("stack")
    # Build reverse index: (did, comp) -> [(stack_id, status)]
    by_pair = {}
    for sid, s in stacks.items():
        for vd in s.get("variant_deltas_applied", []):
            by_pair.setdefault((vd["delta_id"], vd["component"]), []).append(
                (sid, vd["status"]))
    if args.delta:
        if args.delta not in recs:
            print(f"unknown delta_id: {args.delta}"); return 1
        d = recs[args.delta]
        print(f"delta_id={args.delta} name='{d.get('name')}'")
        for comp in d["deltas"]:
            key = (args.delta, comp["component"])
            claims = by_pair.get(key, [])
            if args.status:
                claims = [(s, st) for s, st in claims if st == args.status]
            if args.framework:
                claims = [(s, st) for s, st in claims
                          if stacks[s]["framework"]["name"] == args.framework]
            print(f"  {comp['component']:30s} {len(claims)} claim(s): "
                  + ", ".join(f"{s}={st}" for s, st in claims) if claims
                  else f"  {comp['component']:30s} 0 claims")
    else:
        # Aggregate: print per-delta, per-component counts
        from collections import Counter
        ctr = Counter()
        for (did, comp), claims in by_pair.items():
            for _, st in claims:
                ctr[(did, comp, st)] += 1
        for (did, comp, st), n in sorted(ctr.items()):
            line = f"{did:18s} {comp:30s} {st:12s} {n}"
            if args.status and st != args.status:
                continue
            print(line)
    return 0


def cmd_drift(args):
    """Iter-42: walk every delta_*.json component's `field:` claim, classify
    it against the actual schema MIN-REPORT surface. Verdict counts and
    per-row table mirror scripts/p5p8/p6_delta_field_drift_audit.py."""
    schema = json.load(open(HERE / "schema.json"))
    defs = schema.get("$defs", {})

    def deref(node):
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                if node["$ref"].startswith("#/$defs/"):
                    return deref(defs[node["$ref"].split("/")[-1]])
            return {k: deref(v) for k, v in node.items() if k != "$ref"}
        if isinstance(node, list):
            return [deref(x) for x in node]
        return node

    stack_def = deref(schema["$defs"]["stack_record"])
    min_report = stack_def["properties"]["min_report"]["properties"]
    block_leaves = {}
    for item_name, item_def in min_report.items():
        out = set()
        def walk(o):
            if not isinstance(o, dict):
                return
            for k, v in o.items():
                if k == "properties" and isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, dict) and "properties" in vv:
                            walk(vv)
                        elif isinstance(vv, dict):
                            out.add(kk)
        walk(item_def)
        block_leaves[item_name] = out

    counts = {"OK": 0, "SEE_CITATION": 0,
              "BLOCK_NOT_IN_MIN_REPORT": 0,
              "LEAF_NOT_IN_SCHEMA": 0, "AMBIGUOUS_REFERENCE": 0}
    n_rows = 0
    drifts = []
    for p in sorted((HERE / "entries").glob("delta_*.json")):
        d = json.load(open(p))
        for c in d.get("deltas", []):
            n_rows += 1
            field = c.get("field", "")
            comp = c["component"]
            if not field:
                v, r = "EMPTY", "no field declared"
            elif field == "see delta-list and citation":
                v, r = "SEE_CITATION", "deferred to source paper"
            elif field == "task":
                v, r = "OUT_OF_SCOPE", "top-level task field, not a MIN-REPORT loss lever"
            elif "." not in field:
                v, r = "AMBIGUOUS_REFERENCE", f"block-only reference: {field!r}"
            else:
                block, leaf = field.split(".", 1)
                if block not in block_leaves:
                    v, r = "BLOCK_NOT_IN_MIN_REPORT", f"block {block!r} not in MIN-REPORT"
                elif leaf not in block_leaves[block]:
                    v, r = "LEAF_NOT_IN_SCHEMA", f"leaf {leaf!r} missing from MIN-REPORT.{block}"
                else:
                    v, r = "OK", f"{block}.{leaf}"
            counts[v] = counts.get(v, 0) + 1
            if v not in ("OK", "SEE_CITATION", "OUT_OF_SCOPE"):
                drifts.append((d["id"], comp, field, v, r))
    print(f"delta-schema drift audit (iter-42): {n_rows} (delta, component) pairs")
    drift_n = sum(
        c for k, c in counts.items() if k not in ("OK", "SEE_CITATION", "OUT_OF_SCOPE")
    )
    print(f"  drift rate: {drift_n/max(1,n_rows):.3f}  ({drift_n}/{n_rows})")
    print()
    print(f"  {'verdict':24s} count")
    for k, n in sorted(counts.items()):
        print(f"  {k:24s} {n}")
    if drifts:
        print()
        print(f"  drift rows ({len(drifts)}):")
        for d, c, f, v, r in drifts:
            print(f"    {d:14s} {c:30s} field={f!r:40s} -> {v}")
    return 0


def cmd_claim_validation(args):
    """Iter-46: print the (delta, metric, panel) claim-validation verdict table
    for every variant_delta record. Optional --delta filters to one entry.
    Verdict codes: SUPPORTS / NEUTRAL / CONTRADICTS / UNCLAIMED."""
    from collections import Counter
    counts = Counter()
    records = load("variant_delta")
    for rec in records.values():
        rows = rec.get("claim_validation", []) or []
        if args.delta and rec["id"] != args.delta:
            continue
        for r in rows:
            counts[r["verdict"]] += 1
            if args.delta:
                def number(value):
                    return "NR" if value is None else f"{value:+.4f}"
                print(f"  {rec['id']:18s} {r['metric']:18s} {r['panel']:24s} "
                      f"verdict={r['verdict']:11s} "
                      f"observed={number(r['observed_delta'])} "
                      f"CI=[{number(r.get('ci_low'))},{number(r.get('ci_high'))}]")
    if not args.delta:
        print(f"=== Iter 46 P6 — claim-validation verdict counts (across all {len(records)} variant-delta records) ===")
        for k, n in sorted(counts.items()):
            print(f"  {k:12s} {n}")
    return 0


def cmd_validate(_):
    """Iter-50: CI-style schema validation pass. Returns exit code 0 iff every
    entry parses against registry/schema.json. Stdout is one PASS/FAIL line per
    entry; stderr is empty on full pass."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("FATAL: jsonschema not installed", file=sys.stderr)
        return 2
    schema = json.load(open(HERE / "schema.json"))
    fails = 0
    paths = sorted((HERE / "entries").glob("*.json"))
    queryable = 0
    for p in paths:
        rec = json.loads(p.read_text())
        if "record_type" not in rec:
            print(f"SKIP  {p.name}  provenance sidecar")
            continue
        queryable += 1
        try:
            jsonschema.validate(rec, schema)
            print(f"PASS  {p.name}")
        except jsonschema.ValidationError as e:
            fails += 1
            print(f"FAIL  {p.name}  {str(e.message)[:140]}")
    print(f"--- {queryable - fails}/{queryable} queryable records pass ---")
    return 1 if fails else 0


def cmd_health(_):
    """Iter-50: shell out to scripts/p5p8/p6_registry_health.py for the
    full coverage + null-rate + verdict-signature audit. Passes the exit
    code through unchanged so CI sees schema failures."""
    import subprocess
    target = (
        HERE.parents[1] / "platform_modal/scripts/p5p8/p6_registry_health.py"
    ).resolve()
    rc = subprocess.run([sys.executable, str(target)], cwd=HERE.parent).returncode
    return rc


def cmd_validate_strict(args):
    """Iter-122: integrate the iter-118 strict validator as a CI gate.

    Returns exit code 0 if and only if there are zero ERROR-severity findings
    (orphan delta_id, missing source on disk, bad arxiv id, bad bibkey, schema
    validation failures). WARN and INFO findings (fully-unknown MIN-REPORT
    items, missing optional items, etc.) print to stdout but do NOT block.

    The gate contract: ``orphan_deltas == 0 AND missing_sources == 0 AND
    schema_pass == True`` ⇒ exit 0; otherwise exit 1.

    Use::

        python3 registry/query.py validate-strict      # CI gate (default)
        python3 registry/query.py validate-strict --include-warn  # also fail on warn
    """
    import subprocess
    # First: schema validation (same as cmd_validate).
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("FATAL: jsonschema not installed", file=sys.stderr)
        return 2
    schema = json.load(open(HERE / "schema.json"))
    schema_fails = []
    for p in sorted((HERE / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        if "record_type" not in rec:
            continue
        try:
            jsonschema.validate(rec, schema)
        except jsonschema.ValidationError as e:
            schema_fails.append((p.name, str(e.message)[:140]))
    # Second: shell out to p6_iter118_strict_validator.py for orphan/source checks.
    target = (
        HERE.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    ).resolve()
    rc = subprocess.run(
        [sys.executable, str(target)], cwd=HERE.parent, capture_output=True, text=True
    )
    sys.stdout.write(rc.stdout)
    # Parse the JSON the script wrote.
    audit_json = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_iter118_strict_audit.json"
    audit_json = audit_json.resolve()
    audit = json.loads(audit_json.read_text())
    error_findings = [f for f in audit["findings"] if f.get("severity") == "error"]
    warn_findings = [f for f in audit["findings"] if f.get("severity") == "warn"]
    info_findings = [f for f in audit["findings"] if f.get("severity") == "info"]
    # CI gate contract.
    n_errors = len(error_findings) + len(schema_fails)
    print(f"--- validate-strict summary ---")
    print(f"  n_entries           {audit['n_stack_entries']}")
    print(f"  n_deltas            {audit['n_deltas']}")
    print(f"  schema_fails        {len(schema_fails)}")
    print(f"  error_findings      {len(error_findings)}")
    print(f"  warn_findings       {len(warn_findings)}")
    print(f"  info_findings       {len(info_findings)}")
    if n_errors == 0:
        print("CI GATE: PASS")
    else:
        print(f"CI GATE: FAIL ({n_errors} blocking findings)")
    if args.include_warn:
        return 1 if (n_errors + len(warn_findings)) else 0
    return 1 if n_errors else 0


def cmd_measured_coverage(args):
    """Iter-58: print the measured-block provenance & coverage audit.

    For every variant_delta record, audit the measured / expected_effects /
    claim_validation block presence, source path resolution on disk, and the
    cross-panel sign agreement (N2 zvf vs ZVF130 risk for entries with both).
    Exit 0 always — this is a reporting audit, not a gate."""
    audit_path = HERE / "measured_block_audit.json"
    if not audit_path.exists():
        # Self-rerun if no cache: shell out to the script.
        import subprocess
        target = (
            HERE.parents[1] / "platform_modal/scripts/p5p8/p6_measured_coverage.py"
        ).resolve()
        subprocess.run([sys.executable, str(target)], cwd=HERE.parent, check=True)
    audit = json.load(open(audit_path))
    if args.delta:
        rows = [r for r in audit["per_entry"] if r["delta_id"] == args.delta]
        if not rows:
            print(f"no such delta_id: {args.delta}", file=sys.stderr)
            return 1
        for r in rows:
            print(f"=== {r['delta_id']}  ({r['name']}) ===")
            for k, v in r.items():
                if k in ("delta_id", "name"):
                    continue
                print(f"  {k:24s} {v}")
        # Print cross-panel agreement if available
        for c in audit.get("cross_panel_agreement", []):
            if c["delta_id"] == args.delta:
                print(f"  cross_panel_agreement:")
                for k, v in c.items():
                    if k == "delta_id":
                        continue
                    print(f"    {k:18s} {v}")
        return 0
    headline = {
        "n_deltas": audit["n_entries"],
        "with_measured": sum(1 for r in audit["per_entry"]
                             if r["measured_count"] > 0),
        "empty_measured": sum(1 for r in audit["per_entry"]
                              if r["measured_count"] == 0),
        "verdict_totals": audit["verdict_totals"],
        "n_cross_panel_pairs": len(audit.get("cross_panel_agreement", [])),
        "n_cross_panel_same_sign": sum(
            1 for r in audit.get("cross_panel_agreement", [])
            if r["same_sign"]),
        "n_missing_sources": audit.get("n_mismatch_audit", 0),
    }
    print(f"=== Iter 58 P6 — measured-block coverage audit ({headline['n_deltas']} deltas) ===")
    for k, v in headline.items():
        print(f"  {k:24s} {v}")
    print("--- per-entry ---")
    for r in audit["per_entry"]:
        print(f"  {r['delta_id']:22s} m={r['measured_count']:2d} sig={r['n_significant']:2d} "
              f"S={r['supports']} C={r['contradicts']} N={r['neutral']} U={r['unclaimed']}")
    if audit.get("cross_panel_agreement"):
        print("--- cross-panel (N2 zvf  vs  ZVF130 risk) ---")
        for c in audit["cross_panel_agreement"]:
            flag = "✓" if c["same_sign"] else "✗"
            print(f"  {flag} {c['delta_id']:14s}  N2 zvf={c['n2_zvf_delta']:+.4f}  "
                  f"Z130 risk={c['zvf130_risk_delta']:+.4f}")
    return 0


def cmd_coverage(args):
    """Iter-62: print the outcomes.coverage self-report block per stack entry.

    Reads the audit TSV cached by scripts/p5p8/p6_outcomes_coverage_block.py
    if present, otherwise shells out to regenerate it. Exit 0 always.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_outcomes_coverage_audit.tsv"
    audit_path = audit_path.resolve()
    if not audit_path.exists():
        import subprocess
        target = (
            HERE.parents[1] / "platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py"
        ).resolve()
        subprocess.run([sys.executable, str(target)], cwd=HERE.parent, check=True)
    # parse the TSV
    with audit_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if args.entry:
        rows = [r for r in rows if r[idx["entry_id"]] == args.entry]
        if not rows:
            print(f"no such entry_id: {args.entry}", file=sys.stderr)
            return 1
    # headline
    n_total = len(rows)
    n_stack = sum(1 for r in rows if r[idx["record_type"]] == "stack")
    n_with_cov_block = 0
    recs = load()
    for r in rows:
        eid = r[idx["entry_id"]]
        if eid not in recs:
            continue
        cov = recs[eid].get("outcomes", {})
        if isinstance(cov, dict) and "coverage" in cov:
            n_with_cov_block += 1
    print(f"=== Iter 62 P6 — outcomes.coverage self-report ({n_total} entries) ===")
    print(f"  n_with_coverage_block: {n_with_cov_block}/{n_total}")
    print(f"  n_stack: {n_stack}, n_delta: {n_total - n_stack}")
    print("--- per entry ---")
    if args.entry:
        # full dict for one entry
        eid = args.entry
        rec = recs.get(eid)
        if rec:
            out = rec.get("outcomes", {}) or {}
            cov = out.get("coverage") or {}
            print(f"  {eid}")
            for k, v in cov.items():
                print(f"    {k:30s} {v}")
        return 0
    for r in rows:
        eid = r[idx["entry_id"]]
        mr = float(r[idx["min_report_coverage"]])
        dd = float(r[idx["declared_deltas_coverage"]])
        me = float(r[idx["measured_coverage"]])
        ci = r[idx["ci_method_present"]]
        rt = r[idx["record_type"]]
        print(f"  {eid:38s} {rt:14s} min={mr:.3f} decl={dd:.3f} meas={me:.3f} ci={ci}")
    return 0


def cmd_antiherding(args):
    """Iter-66: print outcomes.zvf_antiherding per stack + measured_yield_residual per delta.

    Reads experiments/results/p5p8/p6_zvf_antiherding_summary.tsv (regenerated
    by scripts/p5p8/p6_zvf_antiherding.py if missing). Exit 0 always.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_zvf_antiherding_summary.tsv"
    audit_path = audit_path.resolve()
    if not audit_path.exists():
        import subprocess
        target = (
            HERE.parents[1] / "platform_modal/scripts/p5p8/p6_zvf_antiherding.py"
        ).resolve()
        subprocess.run([sys.executable, str(target)], cwd=HERE.parent, check=True)
    with audit_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if args.method:
        rows = [r for r in rows if r[idx["method"]] == args.method]
        if not rows:
            print(f"no such method: {args.method}", file=sys.stderr)
            return 1
    print(f"  {'method':8s} {'G':>3s} {'zvf_obs':>8s} {'zvf_iid':>8s} {'delta_div':>9s} {'Y_obs':>7s}  vs_grpo [lo,hi]  sig  p")
    for r in rows:
        m = r[idx["method"]]
        G = r[idx["G"]]
        zo = float(r[idx["zvf_obs_mean"]])
        zi = float(r[idx["zvf_iid_mean"]])
        dd = float(r[idx["delta_div_mean"]])
        yo = float(r[idx["y_obs_mean"]])
        dv = float(r[idx["delta_div_vs_grpo"]])
        lo = float(r[idx["ci_low"]])
        hi = float(r[idx["ci_high"]])
        sig = r[idx["significant"]]
        p = float(r[idx["p_two_sided"]])
        print(f"  {m:8s} {G:>3s} {zo:>8.4f} {zi:>8.4f} {dd:>9.4f} {yo:>7.4f}  "
              f"{dv:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig[:1]}   {p:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pb = sub.add_parser("badge")
    pb.add_argument("entry", nargs="?", default=None)
    pq = sub.add_parser("query")
    pq.add_argument("--item", choices=ITEMS, required=True)
    pq.add_argument("--full", action="store_true")
    pd = sub.add_parser("stackdiff")
    pd.add_argument("a", nargs="?")
    pd.add_argument("b", nargs="?")
    pd.add_argument("--all", action="store_true",
                    help="R-level distribution over all stack pairs")
    pd.add_argument("--exit-level", action="store_true",
                    help="exit with the R-level (0..5) instead of 0, for scripted gating")
    pd.add_argument("--generated", action="store_true",
                    help="with --all: only entries that carry a generated run_config")
    pd.add_argument("--sources", action="store_true",
                    help="print per-field provenance next to each difference")
    pd.add_argument("-v", "--verbose", action="store_true",
                    help="also list every unknown field")
    pi = sub.add_parser("implementations")
    pi.add_argument("--delta", default=None,
                    help="Show stacks claiming this delta_id (e.g. delta_dapo).")
    pi.add_argument("--status", choices=("implemented", "surrogate", "absent",
                                          "unknown"), default=None)
    pi.add_argument("--framework", default=None,
                    help="Filter to stacks whose framework.name == X.")
    sub.add_parser("drift")
    pcv = sub.add_parser("claim-validation",
                          help="Iter-46: print (delta, metric, panel) claim-validation verdicts.")
    pcv.add_argument("--delta", default=None,
                     help="Filter to one delta_id (e.g. delta_aero).")
    sub.add_parser("validate",
                   help="Iter-50: CI-style schema validation pass")
    sub.add_parser("health",
                   help="Iter-50: full registry health audit")
    pvs = sub.add_parser("validate-strict",
                         help="Iter-122: schema + orphan-delta + missing-source CI gate")
    pvs.add_argument("--include-warn", action="store_true",
                     help="Also fail on warn-severity findings (default: warn/info are non-blocking)")
    pmc = sub.add_parser("measured-coverage",
                         help="Iter-58: measured-block provenance & coverage audit.")
    pmc.add_argument("--delta", default=None,
                     help="Filter to one delta_id (e.g. delta_aero).")
    pcov = sub.add_parser("coverage",
                          help="Iter-62: print outcomes.coverage self-report block per entry.")
    pcov.add_argument("--entry", default=None,
                      help="Filter to one entry_id.")
    pah = sub.add_parser("antiherding",
                         help="Iter-66: print Contrastive Yield / anti-herding residual per N2 method.")
    pah.add_argument("--method", default=None,
                     help="Filter to one method (grpo, aero, areal, gift).")
    args = ap.parse_args()
    if args.cmd == "stackdiff":
        level = cmd_stackdiff(args)
        # Default exit 0: a difference is a finding, not a tool failure.
        # --exit-level restores the pre-2026-10-16 behaviour (exit code = R-level).
        return level if args.exit_level and not args.all else 0
    rc = {"list": cmd_list, "badge": cmd_badge,
          "query": cmd_query, "stackdiff": cmd_stackdiff,
          "implementations": cmd_implementations, "drift": cmd_drift,
          "claim-validation": cmd_claim_validation,
          "validate": cmd_validate, "health": cmd_health,
          "validate-strict": cmd_validate_strict,
          "measured-coverage": cmd_measured_coverage,
          "coverage": cmd_coverage,
          "antiherding": cmd_antiherding}[args.cmd](args)
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main() or 0)
