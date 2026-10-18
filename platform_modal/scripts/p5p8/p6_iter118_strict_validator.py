#!/usr/bin/env python3
"""P6 iter-118 strict registry validation.

Extends platform_modal/scripts/p5p8/registry_validate.py with a ``--strict`` mode
that flags:

* orphan delta_id references (variant_deltas_applied[].delta_id must
  match a real ``delta_*.json`` file);
* empty or placeholder rationale strings;
* sign-convention drift between entries (registry uses variant-base,
  registry_validate.py uses base-variant);
* MIN-REPORT items where every leaf is null (i.e. the entry is
  reporting-as-unknown on every leaf of that item).

Output: platform_hybrid/experiments/results/p5p8/p6_iter118_strict_audit.json
"""
import json
import hashlib
from datetime import datetime, timezone
import pathlib
import sys

WORKTREE = pathlib.Path(__file__).resolve().parents[3]
PROJECT_ROOT = WORKTREE / "platform_hybrid"
REGISTRY = PROJECT_ROOT / "registry"
RESULTS = PROJECT_ROOT / "experiments" / "results" / "p5p8" / "review17_v2"

MIN_REPORT_ITEMS = ["loss_form", "reference_kl", "sampler_backend",
                    "telemetry", "group_size_schedule", "heldout_split",
                    "decontamination"]


def leaves(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, child in leaves(v):
                yield f"{k}.{k2}", child
        else:
            yield k, v


def fully_unknown(mr_item):
    """Return True if every leaf in a MIN-REPORT item is None."""
    leaves_it = list(leaves(mr_item or {}))
    if not leaves_it:
        return False
    return all(v is None for _, v in leaves_it)


def source_exists(raw_path):
    """Registry sources may be repo-root- or platform_hybrid-relative."""
    path = pathlib.Path(raw_path)
    if path.is_absolute():
        return path.exists()
    return (WORKTREE / path).exists() or (PROJECT_ROOT / path).exists()


def main(argv=None):
    global REGISTRY
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--registry", type=pathlib.Path, default=PROJECT_ROOT / "registry")
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--input-sha256", help="expected content hash of registry inputs")
    args = ap.parse_args(argv)
    REGISTRY = args.registry.resolve()
    if args.output is None:
        if REGISTRY != (PROJECT_ROOT / "registry").resolve():
            ap.error("--output is required for a non-default --registry")
        args.output = RESULTS / "p6_iter118_strict_audit.json"
    sys.path.insert(0, str(WORKTREE / "platform_hybrid" / "registry"))
    from query import registry_input_manifest, registry_manifest_sha256, VALIDATOR_VERSION, load
    try:
        validated_records = load(registry=REGISTRY)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    manifest = registry_input_manifest(REGISTRY)
    digest = registry_manifest_sha256(manifest)
    if args.input_sha256 is not None and args.input_sha256 != digest:
        print("registry input manifest mismatch", file=sys.stderr)
        return 1
    deltas = {record_id for record_id, rec in validated_records.items()
              if rec["record_type"] == "variant_delta"}

    findings = []
    if not validated_records:
        return 1
    for rid, rec in validated_records.items():
        for claim in rec.get("expected_effects", []):
            rationale = claim.get("rationale")
            if not isinstance(rationale, str) or rationale.strip().lower() in {"", "todo", "tbd", "unknown", "n/a"}:
                findings.append({"kind": "missing_rationale", "entry": rid, "severity": "error"})
        for measurement in rec.get("measured", []):
            convention = measurement.get("sign_convention")
            if convention is not None and convention != "variant_minus_baseline":
                findings.append({"kind": "sign_convention_drift", "entry": rid, "severity": "error"})

    # === orphan delta_id references ===
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("record_type") != "stack":
            continue
        for vda in rec.get("variant_deltas_applied", []):
            did = vda.get("delta_id")
            if did and did not in deltas:
                findings.append({
                    "kind": "orphan_delta_id",
                    "entry": p.stem,
                    "delta_id": did,
                    "severity": "error",
                    "fix": f"create entries/{did}.json or correct the reference",
                })

    # === fully-unknown MIN-REPORT items ===
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("record_type") != "stack":
            continue
        mr = rec.get("min_report", {})
        for it in MIN_REPORT_ITEMS:
            if it not in mr:
                findings.append({
                    "kind": "missing_min_report_item",
                    "entry": p.stem,
                    "item": it,
                    "severity": "warn",
                    "fix": f"add min_report.{it}",
                })
                continue
            if fully_unknown(mr[it]):
                findings.append({
                    "kind": "fully_unknown_item",
                    "entry": p.stem,
                    "item": it,
                    "severity": "info",
                    "fix": f"populate at least one leaf of min_report.{it}",
                })

    # === sign convention drift ===
    # Variant-delta entries store measured.delta as (variant - base).
    # registry_validate.py stores its paired_step_bootstrap diff as
    # (variant - base) but labels the table "grpo vs variant" and prints
    # the raw diff. Both directions are correct mathematically but if
    # someone reads one column "grpo vs aero: +0.025" and the entry
    # "delta_aero: delta=-0.025" they may miss that these mean the same
    # thing. flag any sign-flips for human review.
    # === source path existence ===
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        for m in rec.get("measured", []):
            src = m.get("source")
            if src and not source_exists(src):
                findings.append({
                    "kind": "missing_source",
                    "entry": p.stem,
                    "source": src,
                    "severity": "error",
                    "fix": "recreate the source artifact or update the path",
                })

    # === citation resolution ===
    for p in sorted((REGISTRY / "entries").glob("delta_*.json")):
        rec = json.loads(p.read_text())
        cit = rec.get("citation", {}) or {}
        arxiv = cit.get("arxiv")
        bibkey = cit.get("bibkey")
        if arxiv and not isinstance(arxiv, str):
            findings.append({
                "kind": "bad_arxiv_id",
                "entry": p.stem,
                "severity": "error",
                "fix": "arxiv must be a string like 2509.21880",
            })
        if bibkey and not isinstance(bibkey, str):
            findings.append({
                "kind": "bad_bibkey",
                "entry": p.stem,
                "severity": "error",
                "fix": "bibkey must be a string",
            })

    out_json = {
        "validator_version": VALIDATOR_VERSION,
        "validator_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "input_manifest": manifest,
        "input_sha256": digest,
        "n_findings": len(findings),
        "findings": findings,
        "n_deltas": len(deltas),
        "n_stack_entries": sum(
            1 for p in (REGISTRY / "entries").glob("*.json")
            if json.loads(p.read_text()).get("record_type") == "stack"),
        "audit_date": datetime.now(timezone.utc).isoformat(),
        "audit_source": "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py",
    }
    if registry_input_manifest(REGISTRY) != manifest:
        print("registry inputs changed during validation", file=sys.stderr)
        return 1
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_json, indent=2))

    by_kind = {}
    for f in findings:
        by_kind.setdefault(f["kind"], 0)
        by_kind[f["kind"]] += 1
    print(f"# strict audit: {len(findings)} findings across "
          f"{out_json['n_stack_entries']} stacks / {len(deltas)} deltas")
    for k, n in sorted(by_kind.items()):
        print(f"  {k:30s} {n}")
    print(f"  wrote {out_path}")

    return int(any(f["severity"] == "error" for f in findings))


if __name__ == "__main__":
    sys.exit(main())
