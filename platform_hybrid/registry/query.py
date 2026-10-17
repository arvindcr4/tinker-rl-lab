#!/usr/bin/env python3
"""GRPO-Registry reference CLI: list / query / badge / stackdiff / drift.

Requires jsonschema for centralized ingestion. Field convention (see schema.json): JSON null = unreported;
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
import hashlib
import json
import math
import re
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
try:
    from .json_protocol import strict_json_loads
except ImportError:  # Script/spec-loader invocation.
    sys.path.insert(0, str(HERE))
    from json_protocol import strict_json_loads
ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]
# stackdiff severity ladder.  A level is assigned ONLY from fields that are
# known on both sides and differ; fields unknown on either side never raise
# the level -- they lower `coverage` and are reported as `underspecified`.
RISK = {0: "R0 no known differences",
        1: "R1 seed / version / reporting differences; semantic review may be required",
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
    "eval.heldout": 5, "tokenizer.id": 5, "tokenizer.revision": 5,
    "data.revision": 5, "data.ordered_example_identity": 5,
    # R4 algorithm
    "loss.objective": 4, "loss.importance_ratio": 4, "loss.clip": 4,
    "loss.aggregation": 4, "loss.advantage_norm": 4, "loss.token_mask": 4,
    "loss.length_normalization": 4, "loss.dynamic_sampling": 4,
    "loss.reward_shaping": 4, "loss.inner_epochs": 4,
    "kl.reference_policy": 4, "kl.beta": 4, "kl.estimator": 4,
    # R3 hyperparameters / budget
    "decoding.temperature": 3, "decoding.top_p": 3, "decoding.max_tokens": 3,
    "decoding.top_k": 3, "decoding.repetition_penalty": 3,
    "batch.prompts_per_step": 3, "batch.samples_per_prompt": 3,
    "batch.group_schedule": 3, "batch.adaptation_rule": 3, "train.steps": 3, "train.sampler_refresh": 3,
    "data.subset": 3, "data.selection": 3,
    "optimizer.name": 3, "optimizer.lr": 3, "optimizer.betas": 3,
    "optimizer.eps": 3, "optimizer.weight_decay": 3,
    "lora.rank": 3, "lora.alpha": 3, "lora.dropout": 3, "lora.targets": 3,
    # R2 runtime
    "trainer.name": 2, "runtime.backend": 2, "runtime.precision": 2,
    "reward.fn_sha256": 2, "runtime.gpu": 2, "runtime.environment": 2,
    "implementation.identity": 2,
    # R1 nuisance
    "trainer.version": 1, "seeds": 1,
}
# All unknown outcome-critical fields, including checkpoint revision, flag
# underspecification. R0 is not evidence of a matched stack.
COVERAGE_FLOOR = 0.8
COMPARISON_COVERAGE_VERSION = "stackdiff_known_both_v2"
SPEC_COVERAGE_VERSION = "stack_spec_known_fields_v2"
MATCHING_REQUIREMENTS_VERSION = "all_catalogue_fields_v2"
# run_config status vocabulary; only "unknown" counts as not known.
KNOWN_STATUSES = ("logged", "code", "default", "inferred")


VALIDATOR_VERSION = "registry-strict-v3"
COVERAGE_VERSION = "reported-substantive-leaf-mean-v3"


def _sidecars(registry):
    path = registry / "sidecars.json"
    if not path.exists():
        return {}
    doc = strict_json_loads(path.read_text())
    if doc.get("version") != "registry-sidecars-v1" or not isinstance(doc.get("documents"), dict):
        raise ValueError("invalid sidecar catalogue")
    return doc["documents"]


def validate_record(rec, schema):
    import jsonschema
    jsonschema.Draft202012Validator(schema).validate(rec)
    seen = set()
    for row in rec.get("variant_deltas_applied", []):
        key = row["delta_id"], row["component"]
        if key in seen:
            raise ValueError(f"duplicate delta declaration: {key}")
        seen.add(key)
    lf = rec.get("min_report", {}).get("loss_form", {})
    if lf.get("clipping_applied") is False and any(lf.get(k) is not None for k in ("clip_eps_low", "clip_eps_high")):
        raise ValueError("disabled clipping cannot declare clipping epsilon")


def load(kind=None, registry=None):
    """Single schema boundary for all commands; unknown documents fail closed."""
    registry = HERE if registry is None else pathlib.Path(registry)
    schema = strict_json_loads((registry / "schema.json").read_text())
    sidecars = _sidecars(registry)
    recs = {}
    for path in sorted((registry / "entries").glob("*.json")):
        raw = path.read_bytes()
        rec = strict_json_loads(raw)
        if path.name in sidecars:
            declaration = sidecars[path.name]
            if (declaration.get("record_type") != "historical_amendment"
                    or declaration.get("sha256") != hashlib.sha256(raw).hexdigest()
                    or not isinstance(rec, dict) or "record_type" in rec):
                raise ValueError(f"unrecognized or modified sidecar: {path.name}")
            continue
        try:
            validate_record(rec, schema)
        except Exception as exc:
            raise ValueError(f"invalid registry record {path.name}: {exc}") from exc
        if rec["id"] in recs:
            raise ValueError(f"duplicate registry id: {rec['id']}")
        recs[rec["id"]] = rec
    return {k: v for k, v in recs.items() if kind is None or v["record_type"] == kind}


def item_score(d, item=None):
    """Fixed schema denominator, excluding explanatory provenance metadata."""
    definitions = strict_json_loads((HERE / "schema.json").read_text())["$defs"]["min_report"]["properties"]
    if item is None:
        candidates = [k for k, v in definitions.items() if set(d) <= set(v["properties"])]
        if len(candidates) != 1:
            raise ValueError("item name required for an ambiguous MIN-REPORT block")
        item = candidates[0]
    leaves = set(definitions[item]["properties"]) - {"source", "description"}
    if item == "loss_form" and d.get("clipping_applied") is False:
        leaves -= {"clip_eps_low", "clip_eps_high"}
    return sum(d.get(key) is not None for key in leaves) / len(leaves) if leaves else 0.0


def badge(rec):
    scores = {item: item_score(rec.get("min_report", {}).get(item, {}), item) for item in ITEMS}
    return round(100 * sum(scores.values()) / len(ITEMS)), scores


def _variant(model):
    """Base/instruct from the checkpoint name; None when the name is silent."""
    if not model:
        return None
    name = model.lower().split("/")[-1]
    if re.search(r"(?:^|[-_.])base(?:$|[-_.])", name):
        return "base"
    if re.search(r"(?:^|[-_.])instruct(?:$|[-_.])", name) or name.endswith(("-inst", "-it")):
        return "instruct"
    return None


def legacy_view(rec):
    """Map hand-entered top-level + min_report fields onto FIELDS names.

    Every value is tagged status="registry" (hand-entered, no per-field
    provenance).  JSON null -> unknown, per the registry convention."""
    mr = rec.get("min_report", {})
    lf, kl = mr.get("loss_form", {}), mr.get("reference_kl", {})
    sb, gs = mr.get("sampler_backend", {}), mr.get("group_size_schedule", {})
    clip = "none" if lf.get("clipping_applied") is False else None
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
        "batch.adaptation_rule": gs.get("adaptation_rule"),
    }
    return {k: {"value": v, "status": "registry" if v is not None else "unknown",
                "source": "registry entry (hand-entered)"} for k, v in raw.items()}


# Explicit mappings only: no model-name guesses, library defaults or absent
# values are promoted to observations. Both flat catalogue keys and these
# nested effective-manifest paths are supported.
MANIFEST_FIELDS = {
    "checkpoint.id": ("revisions.model.id", "model.id", "model_id"),
    "checkpoint.revision": ("revisions.model.revision", "model.revision", "model_revision"),
    "tokenizer.id": ("revisions.tokenizer.id", "tokenizer.id", "tokenizer_id"),
    "tokenizer.revision": ("revisions.tokenizer.revision", "tokenizer.revision", "tokenizer_revision"),
    "runtime.gpu": ("hardware.gpu_names", "runtime.gpu", "gpu"),
    "runtime.environment": ("environment_hash", "runtime.environment"),
    "runtime.precision": ("hardware.model_dtype", "runtime.precision", "precision"),
    "data.dataset": ("revisions.dataset.id", "dataset.id", "dataset_id"),
    "data.revision": ("revisions.dataset.revision", "dataset.revision", "dataset_revision"),
    "data.ordered_example_identity": ("data.ordered_example_identity",
                                      "ordered_example_identity", "ordered_example_ids_sha256"),
    "implementation.identity": ("implementation.identity", "implementation_identity", "source"),
    "decoding.top_k": ("generation.train.top_k", "generation.top_k", "generation_config.top_k"),
    "decoding.repetition_penalty": ("generation.train.repetition_penalty", "generation.repetition_penalty", "generation_config.repetition_penalty"),
    "decoding.temperature": ("generation.train.temperature", "generation.temperature", "generation_config.temperature"),
    "decoding.top_p": ("generation.train.top_p", "generation.top_p", "generation_config.top_p"),
    "decoding.max_tokens": ("generation.train.max_new_tokens", "generation.max_new_tokens", "generation_config.max_new_tokens"),
}


def manifest_view(manifest, source="entry#effective_manifest"):
    """Extract observed configuration from an effective run manifest."""
    view = {}
    if not isinstance(manifest, dict):
        return view
    missing = object()
    for field in FIELDS:
        for path in (field, *MANIFEST_FIELDS.get(field, ())):
            value = manifest
            if path in manifest:
                value = manifest[path]
            else:
                for part in path.split("."):
                    if not isinstance(value, dict) or part not in value:
                        value = missing
                        break
                    value = value[part]
            if value is not missing:
                if (field == "implementation.identity" and path == "source"
                        and isinstance(value, dict)
                        and (value.get("status") == "unknown"
                             or not (value.get("source_files") or value.get("commit")))):
                    value = None
                view[field] = {"value": value, "status": "logged" if value is not None else "unknown",
                               "source": f"{source}.{path}"}
                break
    if "data.ordered_example_identity" not in view:
        ids = {k: manifest[k] for k in ("train_item_ids", "test_item_ids") if k in manifest}
        if len(ids) == 2:
            # Compare observed ordered lists, not an assumed shared dataset split.
            view["data.ordered_example_identity"] = {
                "value": ids, "status": "logged",
                "source": f"{source}.train_item_ids,test_item_ids"}
    return view


def field_view(rec):
    """Canonical {field: {value, status, source}} over every FIELDS name.

    run_config (generated, with provenance) overrides the legacy view."""
    view = {} if "run_config" in rec else legacy_view(rec)
    view.update(rec.get("run_config") or {})
    view.update(manifest_view(rec.get("effective_manifest")))
    unknown = {"value": None, "status": "unknown", "source": "not recorded"}
    return {f: view.get(f, unknown) for f in FIELDS}


def is_known(cell):
    return (isinstance(cell, dict) and cell.get("status") in KNOWN_STATUSES + ("registry",)
            and cell.get("value") is not None and bool(cell.get("source")))


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
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("configuration number must be finite")
        return x
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm(i) for i in x]
    return x


def evidence_coverage(rec):
    """Disjoint evidence classes, each divided by the full catalogue."""
    groups = {"observed": {"logged"}, "source_defined": {"code"},
              "assumed": {"default", "inferred", "registry"}}
    cells = list(field_view(rec).values())
    result = {group: sum(is_known(c) and c["status"] in statuses for c in cells) / len(FIELDS)
              for group, statuses in groups.items()}
    result["unresolved"] = sum(not is_known(c) for c in cells) / len(FIELDS)
    return result


def stackdiff(a, b, required_fields=None):
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
    side only is unknown R4 evidence and belongs to the coverage denominator.
    """
    required = set(FIELDS if required_fields is None else required_fields)
    if not required <= set(FIELDS):
        raise ValueError(f"unknown matching requirements: {sorted(required - set(FIELDS))}")
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
    def delta_view(record):
        result = {}
        for row in record.get("variant_deltas_applied", []):
            key = row["delta_id"], row["component"]
            if key in result or row.get("status") not in {"implemented", "surrogate", "absent", "unknown", "single-seed-surrogate"}:
                raise ValueError(f"invalid or duplicate delta declaration: {key}")
            result[key] = row["status"]
        return result
    da, db = delta_view(a), delta_view(b)
    undeclared, n_rows = [], 0
    for key in sorted(set(da) | set(db)):
        name = f"delta.{key[0]}:{key[1]}"
        n_rows += 1
        if key not in da or key not in db:
            # Declared on one side only: not evidence of absence (a baseline
            # lists no deltas even when, e.g., its loss also has no KL term).
            # Unknown R4 evidence lowers coverage without fabricating absence.
            side = "a" if key not in da else "b"
            undeclared.append((name, side))
            unknown.append((name, 4, side))
            continue
        sa, sb = da[key], db[key]
        if "unknown" in (sa, sb):
            side = "both" if sa == sb else ("a" if sa == "unknown" else "b")
            unknown.append((name, 4, side))
        else:
            known_both += 1
            if sa != sb:
                differs.append((name, 4, sa, sb))
    coverage = known_both / (len(FIELDS) + n_rows)
    conditional = [f for f in FIELDS if is_known(va[f]) and is_known(vb[f])
                   and any(c["status"] in {"default", "inferred", "registry"} for c in (va[f], vb[f]))]
    semantic_review = [f for f, *_ in differs if f in {"trainer.version", "reward.fn_sha256", "implementation.identity"}]
    comparable = not unknown and not undeclared and not conditional and not semantic_review
    return {"level": max((d[1] for d in differs), default=0),
            "differs": differs, "unknown": unknown, "undeclared": undeclared,
            "coverage": coverage, "coverage_version": COMPARISON_COVERAGE_VERSION,
            "no_known_differences": not differs,
            "comparable": comparable, "conditional_fields": conditional,
            "semantic_review_required": semantic_review,
            "evidence_coverage": {"a": evidence_coverage(a), "b": evidence_coverage(b)},
            "evidence_complete": not unknown and not undeclared,
            "matching_requirements": {
                "version": MATCHING_REQUIREMENTS_VERSION,
                "required_fields": sorted(required),
                "unknown_fields": [f for f, _, _ in unknown if f in required],
                "differing_fields": [f for f, *_ in differs if f in required],
                "satisfied": (not any(f in required for f, *_ in unknown + differs)
                              and not undeclared and not conditional and not semantic_review
                              and not any(f.startswith("delta.") for f, *_ in unknown + differs)),
            },
            "hidden_max": max((u[1] for u in unknown), default=0),
            "underspecified": coverage < COVERAGE_FLOOR or any(
                u[1] == 5 for u in unknown),
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
              f"({SPEC_COVERAGE_VERSION}; {src})  {detail}")


def cmd_query(args):
    """Which stacks fully report a MIN-REPORT item? (e.g. --item reference_kl)"""
    for rid, r in load("stack").items():
        s = item_score(r["min_report"][args.item], args.item)
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
        if getattr(args, "exit_level", False):
            level = max((r["level"] for r in res.values()), default=0)
            return level or (6 if not res or any(not r["comparable"] for r in res.values()) else 0)
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
    print(f"coverage: {r['coverage']:.2f} ({r['coverage_version']}) of compared fields known on both sides; "
          f"{len(r['unknown'])} unknown (max hidden severity R{r['hidden_max']})")
    print(f"evidence complete: {r['evidence_complete']}; "
          f"matching requirements satisfied: {r['matching_requirements']['satisfied']}")
    if r["underspecified"]:
        crit = [u[0] for u in r["unknown"] if u[1] == 5]
        print("UNDERSPECIFIED: " + (f"outcome-critical fields unknown: {', '.join(crit)}"
                                     if crit else f"coverage < {COVERAGE_FLOOR}"))
    if r["undeclared"]:
        print("undeclared delta rows (one side only, unknown R4 evidence): "
              + ", ".join(f"{n} [missing on {s}]" for n, s in r["undeclared"]))
    if getattr(args, "verbose", False):
        for f, sev, side in sorted(r["unknown"], key=lambda u: -u[1]):
            print(f"  [unknown R{sev}] {f} (missing on {side})")
    if r["same_label"] and r["level"] >= 4:
        print("label-flip risk: the shared label does not pin the stack;"
              "\n  do not attribute outcome differences to the label.")
    print(f"comparable: {r['comparable']}; conditional fields: {len(r['conditional_fields'])}; "
          f"semantic review: {r['semantic_review_required']}")
    return r["level"] if r["level"] or r["comparable"] or not getattr(args, "exit_level", False) else 6


def cmd_implementations(args):
    """Cross-reference: which stacks claim to implement a (delta, component)?

    Usage:
      python3 registry/query.py implementations --delta delta_dapo
      python3 registry/query.py implementations --status implemented
      python3 registry/query.py implementations --status unknown --framework worktree-zvf130-batch
    """
    recs = load("variant_delta") if args.delta else {}
    stacks = load("stack")
    if args.framework:
        stacks = {k: v for k, v in stacks.items() if v["framework"]["name"] == args.framework}
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
        def walk(o, out=out):
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
        for d, c, f, v, _r in drifts:
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
    try:
        records = load()
    except (ValueError, OSError, ImportError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {len(records)} registry records validated")
    return 0


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


def registry_input_manifest(registry):
    """Content-address the exact registry inputs, including the schema and sidecars."""
    paths = [registry / "schema.json", *sorted((registry / "entries").glob("*.json"))]
    if (registry / "sidecars.json").exists():
        paths.append(registry / "sidecars.json")
    return {p.relative_to(registry).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def registry_manifest_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        import jsonschema  # noqa: F401
    except ImportError:
        print("FATAL: jsonschema not installed", file=sys.stderr)
        return 2
    manifest = registry_input_manifest(HERE)
    schema_fails = []
    try:
        load()
    except (ValueError, OSError) as exc:
        print(f"CI GATE: FAIL ({exc})", file=sys.stderr)
        return 1
    # Second: shell out to p6_iter118_strict_validator.py for orphan/source checks.
    target = (
        HERE.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    ).resolve()
    # A fresh, unique output prevents a failed producer from blessing a prior audit.
    import tempfile
    if registry_input_manifest(HERE) != manifest:
        print("CI GATE: FAIL (registry inputs changed during schema validation)", file=sys.stderr)
        return 1
    digest = registry_manifest_sha256(manifest)
    try:
        producer_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        print(f"CI GATE: FAIL (validator unavailable: {exc})", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="registry-strict-") as work:
        audit_json = pathlib.Path(work) / "audit.json"
        try:
            rc = subprocess.run(
                [sys.executable, str(target), "--registry", str(HERE),
                 "--output", str(audit_json), "--input-sha256", digest],
                cwd=HERE.parent, capture_output=True, text=True,
            )
        except OSError as exc:
            print(f"CI GATE: FAIL (validator could not start: {exc})", file=sys.stderr)
            return 1
        sys.stdout.write(rc.stdout)
        sys.stderr.write(rc.stderr)
        if rc.returncode:
            print(f"CI GATE: FAIL (validator exited {rc.returncode})", file=sys.stderr)
            return rc.returncode if rc.returncode > 0 else 1
        try:
            audit = strict_json_loads(audit_json.read_text())
            if not isinstance(audit, dict):
                raise ValueError("audit must be an object")
            if audit.get("validator_version") != VALIDATOR_VERSION:
                raise ValueError("unrecognized validator version")
            if (audit.get("validator_sha256") != producer_digest
                    or hashlib.sha256(target.read_bytes()).hexdigest() != producer_digest):
                raise ValueError("validator implementation identity mismatch")
            if (audit.get("input_manifest") != manifest
                    or audit.get("input_sha256") != digest
                    or registry_input_manifest(HERE) != manifest):
                raise ValueError("audit does not match the current registry input manifest")
            if not isinstance(audit.get("findings"), list) or not all(
                    isinstance(f, dict) and f.get("severity") in {"error", "warn", "info"}
                    for f in audit["findings"]):
                raise ValueError("invalid audit findings")
            for key in ("n_stack_entries", "n_deltas"):
                if type(audit.get(key)) is not int or audit[key] < 0:
                    raise ValueError(f"invalid audit {key}")
        except (OSError, ValueError, TypeError) as exc:
            print(f"CI GATE: FAIL (missing, malformed, or stale audit: {exc})", file=sys.stderr)
            return 1
    error_findings = [f for f in audit["findings"] if f.get("severity") == "error"]
    warn_findings = [f for f in audit["findings"] if f.get("severity") == "warn"]
    info_findings = [f for f in audit["findings"] if f.get("severity") == "info"]
    # CI gate contract.
    n_errors = len(error_findings) + len(schema_fails)
    print("--- validate-strict summary ---")
    print(f"  n_entries           {audit['n_stack_entries']}")
    print(f"  n_deltas            {audit['n_deltas']}")
    print(f"  schema_fails        {len(schema_fails)}")
    print(f"  error_findings      {len(error_findings)}")
    print(f"  warn_findings       {len(warn_findings)}")
    print(f"  info_findings       {len(info_findings)}")
    if n_errors == 0 and not (args.include_warn and warn_findings):
        print("CI GATE: PASS")
    else:
        print(f"CI GATE: FAIL ({n_errors + (len(warn_findings) if args.include_warn else 0)} blocking findings)")
    if args.include_warn:
        return 1 if (n_errors + len(warn_findings)) else 0
    return 1 if n_errors else 0


def audit_input_manifest(target):
    """Bind reporting caches to registry, producer, and referenced local artifacts."""
    root = HERE.parents[1]
    manifest = {f"registry/{k}": v for k, v in registry_input_manifest(HERE).items()}
    manifest["producer"] = hashlib.sha256(target.read_bytes()).hexdigest()
    paths = set((root / "platform_hybrid/experiments/results/n2_reward_tensor_resume").glob("*_tensors.jsonl"))
    paths.update(path for path in (HERE / "query.py", HERE / "json_protocol.py") if path.is_file())
    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"source", "audit_source"} and isinstance(child, str):
                    raw = child.split("#", 1)[0].split(" sha256:", 1)[0]
                    if pathlib.Path(raw).is_absolute() and not pathlib.Path(raw).resolve().is_relative_to(root):
                        raise ValueError("external reporting source is outside the bound repository")
                    if ".." in pathlib.Path(raw).parts:
                        raise ValueError("parent traversal is not a bound reporting source")
                    for base in (root, root / "platform_hybrid"):
                        path = (base / raw).resolve()
                        if path.is_relative_to(root) and path.is_file():
                            paths.add(path)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for rec in load().values():
        collect(rec)
    for path in sorted(paths):
        manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def audit_text(audit_path, target, args):
    """Read a bound cache, or refresh in an isolated copy without editing evidence."""
    import shutil
    import subprocess
    import tempfile
    manifest = audit_input_manifest(target)
    digest = registry_manifest_sha256(manifest)
    if getattr(args, "refresh", False):
        root = HERE.parents[1]
        with tempfile.TemporaryDirectory(prefix="registry-report-") as temporary:
            staged = pathlib.Path(temporary)
            stage_registry = staged / "platform_hybrid/registry"
            shutil.copytree(HERE / "entries", stage_registry / "entries")
            for name in ("schema.json", "sidecars.json", "query.py", "json_protocol.py"):
                if (HERE / name).exists():
                    shutil.copy2(HERE / name, stage_registry / name)
            script = staged / target.relative_to(root)
            script.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, script)
            for name in manifest:
                if name.startswith(("registry/", "producer")):
                    continue
                source = root / name
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            (staged / "platform_hybrid/experiments/results/p5p8").mkdir(parents=True, exist_ok=True)
            result = subprocess.run([sys.executable, str(script)], cwd=staged,
                                    capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f"report refresh failed: {result.stderr or result.stdout}")
            text = (staged / audit_path.relative_to(root)).read_text()
        if audit_input_manifest(target) != manifest:
            raise ValueError("report inputs changed during refresh")
        print(f"snapshot: {digest}; fresh isolated recomputation; producer sha256={manifest['producer']}")
        return text
    receipt_path = audit_path.with_name(audit_path.name + ".receipt.json")
    try:
        raw = audit_path.read_bytes()
        receipt = strict_json_loads(receipt_path.read_text())
        if (receipt.get("version") != "registry-report-cache-v1"
                or receipt.get("input_manifest") != manifest
                or receipt.get("input_sha256") != digest
                or receipt.get("output_sha256") != hashlib.sha256(raw).hexdigest()):
            raise ValueError("report receipt does not match current inputs")
    except (OSError, ValueError) as exc:
        raise ValueError(f"unverified/stale cached report {audit_path.name}; use --refresh ({exc})") from exc
    print(f"snapshot: {digest}; verified cache")
    return raw.decode()


def cmd_measured_coverage(args):
    """Iter-58: print the measured-block provenance & coverage audit.

    For every variant_delta record, audit the measured / expected_effects /
    claim_validation block presence, source path resolution on disk, and the
    cross-panel sign agreement (N2 zvf vs ZVF130 risk for entries with both).
    Missing or stale caches fail; --refresh recomputes without editing evidence."""
    audit_path = HERE / "measured_block_audit.json"
    target = HERE.parents[1] / "platform_modal/scripts/p5p8/p6_measured_coverage.py"
    audit = strict_json_loads(audit_text(audit_path, target, args))
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
                print("  cross_panel_agreement:")
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

    Reads only content-bound audit snapshots; --refresh recomputes in isolation.
    MIN-REPORT coverage is shared v3; other legacy metrics are explicitly labelled.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_outcomes_coverage_audit.tsv"
    audit_path = audit_path.resolve()
    target = HERE.parents[1] / "platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py"
    lines = audit_text(audit_path, target, args).splitlines()
    header = lines[0].split("\t")
    idx = {h: i for i, h in enumerate(header)}
    rows = [line.split("\t") for line in lines[1:]]
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
    print(f"=== outcomes.coverage report ({n_total} entries); {COVERAGE_VERSION} ===")
    print(f"  n_with_coverage_block: {n_with_cov_block}/{n_total}")
    print(f"  n_stack: {n_stack}, n_delta: {n_total - n_stack}")
    print("--- per entry ---")
    if args.entry:
        # Print the freshly verified/recomputed row rather than an old entry block.
        for key, index in idx.items():
            label = "legacy_item_presence" if key == "min_report_coverage" else key
            print(f"    {label:30s} {rows[0][index]}")
        rec = recs[args.entry]
        if rec["record_type"] == "stack":
            scores = badge(rec)[1]
            print(f"    reported_leaf_coverage_v3      {sum(scores.values()) / len(ITEMS)}")
            print(f"    item_presence_coverage_v3      {sum(v > 0 for v in scores.values()) / len(ITEMS)}")
        return 0
    for r in rows:
        eid = r[idx["entry_id"]]
        scores = badge(recs[eid])[1] if recs[eid]["record_type"] == "stack" else None
        leaf = f"{sum(scores.values()) / len(ITEMS):.3f}" if scores else "n/a"
        presence = f"{sum(v > 0 for v in scores.values()) / len(ITEMS):.3f}" if scores else "n/a"
        dd = float(r[idx["declared_deltas_coverage"]])
        me = float(r[idx["measured_coverage"]])
        ci = r[idx["ci_method_present"]]
        rt = r[idx["record_type"]]
        print(f"  {eid:38s} {rt:14s} reported_leaf_v3={leaf} item_presence_v3={presence} "
              f"legacy_decl={dd:.3f} legacy_meas={me:.3f} ci={ci}")
    return 0


def cmd_antiherding(args):
    """Iter-66: print outcomes.zvf_antiherding per stack + measured_yield_residual per delta.

    Requires a content-bound snapshot or explicit --refresh in a temporary copy.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_zvf_antiherding_summary.tsv"
    audit_path = audit_path.resolve()
    target = HERE.parents[1] / "platform_modal/scripts/p5p8/p6_zvf_antiherding.py"
    lines = audit_text(audit_path, target, args).splitlines()
    header = lines[0].split("\t")
    idx = {h: i for i, h in enumerate(header)}
    rows = [line.split("\t") for line in lines[1:]]
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
                    help="exit with R-level when different; exit 6 for R0 with unresolved comparability")
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
    for report_parser in (pmc, pcov, pah):
        report_parser.add_argument("--refresh", action="store_true",
                                   help="recompute in an isolated temporary copy; preserve stored evidence")
    args = ap.parse_args()
    if args.cmd == "stackdiff":
        level = cmd_stackdiff(args)
        # Default exit 0: a difference is a finding, not a tool failure.
        # --exit-level restores the pre-2026-10-16 behaviour (exit code = R-level).
        return level if args.exit_level else 0
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
    try:
        sys.exit(main() or 0)
    except (ValueError, OSError, KeyError) as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        sys.exit(1)
