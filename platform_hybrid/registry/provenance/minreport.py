#!/usr/bin/env python3
"""MIN-REPORT-RL/2.0: authenticated publication gate and unsigned diagnostics.

``verify RECORD --trusted-key KEY`` requires every evidence check and Ed25519
trust verification. ``--diagnostic`` never accepts a record for publication.
The reporting percentage is completeness, not statistical power or authenticity.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

try:
    from . import sign
except ImportError:
    import sign

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_SCHEMA = Path(__file__).resolve().parents[1] / "schema.json"
SPEC_VERSION = "MIN-REPORT-RL/2.0"
MIN_REPORT_SCHEMA = "registry-v3#/$defs/min_report"
SEVEN = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]
REWARD_FN_NAMES = ("reward", "reward_fn", "reward_function")
FULL_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
FULL_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def object_digest(value):
    return digest(sign.canonical_bytes(value))


def resolve_source_path(raw_path):
    """Resolve repo-local sources only, including the historical layout alias."""
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        return None
    root = REPO_ROOT.resolve()
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [root / path, root / "platform_hybrid" / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                return resolved
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def sha256_file(path):
    resolved = resolve_source_path(path)
    return digest(resolved.read_bytes()) if resolved else None


def reward_fn_sha256(path):
    resolved = resolve_source_path(path)
    if resolved is None:
        return None
    try:
        source = resolved.read_text(encoding="utf-8")
        functions = [node for node in ast.walk(ast.parse(source))
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in REWARD_FN_NAMES]
        # Ambiguity is not evidence of which reward implementation executed.
        if len(functions) != 1:
            return None
        return digest(ast.get_source_segment(source, functions[0]).encode("utf-8"))
    except (SyntaxError, UnicodeError, OSError):
        return None


def _git(*args):
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True)
    return result.stdout


def git_sha():
    try:
        value = _git("rev-parse", "HEAD").decode().strip()
        return value if FULL_COMMIT.fullmatch(value) else None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def worktree_identity():
    """Tracked worktree identity; selected source bytes are independently hashed."""
    try:
        patch = _git("-c", "color.ui=false", "diff", "--binary", "--full-index",
                     "--no-ext-diff", "--no-textconv", "--no-renames", "HEAD", "--")
        return {"format": "git-tracked-worktree/1.0",
                "git_tree": _git("rev-parse", "HEAD^{tree}").decode().strip(),
                "tracked_diff_hash": digest(patch), "dirty": bool(patch)}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _shared_schema():
    registry = sign.load_json(REGISTRY_SCHEMA)
    definitions = registry["$defs"]
    return {"$schema": registry["$schema"], "$defs": definitions,
            "$ref": "#/$defs/min_report"}


def _schema_digest():
    definitions = _shared_schema()["$defs"]
    # Only the referenced MIN-REPORT contract, not unrelated registry fields.
    return object_digest({key: value for key, value in definitions.items()
                          if key == "min_report" or key.startswith("nullable_")})


def canonical_min_report(config):
    """Adapt known aliases; leave every unsupported claim as JSON null."""
    definition = _shared_schema()["$defs"]["min_report"]
    supplied = config.get("min_report_rl", config)
    if type(supplied) is not dict:
        raise ValueError("min_report_rl must be an object")
    result = {}
    for item in SEVEN:
        fields = definition["properties"][item]["properties"]
        value = supplied.get(item)
        target = {field: None for field in fields}
        if type(value) is dict:
            target.update(copy.deepcopy(value))
        elif item == "sampler_backend" and type(value) is str:
            target["backend"] = value
        elif value is not None and not (item == "telemetry" and type(value) is list):
            raise ValueError(f"{item} must be an object")
        if item == "reference_kl" and "beta" in target:
            target["kl_beta"] = target.pop("beta")
        if item == "group_size_schedule":
            if "G" in target:
                target["initial_g"] = target.pop("G")
            elif target["initial_g"] is None and "G" in config:
                target["initial_g"] = config["G"]
            if target["schedule"] == "constant":
                target["schedule"] = "fixed"
        if item == "heldout_split":
            if "disjoint" in target:
                target["disjoint_from_reward_env"] = target.pop("disjoint")
            target.setdefault("n", config.get("heldout_n"))
        result[item] = target
    return result


def generate(run, config_path, out):
    # An explicitly supplied missing or malformed config is always an error.
    config = sign.load_json(config_path) if config_path is not None else {}
    if type(config) is not dict:
        raise ValueError("configuration must be a JSON object")
    runtime_config = {key: value for key, value in config.items()
                      if key not in {"rigor", "results", "evidence"}}
    code = runtime_config.get("code_file")
    record = {
        "run_id": run, "spec_version": SPEC_VERSION,
        "min_report_schema": MIN_REPORT_SCHEMA, "min_report_schema_hash": _schema_digest(),
        "min_report_rl": canonical_min_report(config),
        "provenance": {"git_sha": git_sha(), "code_file": code,
                       "code_hash": sha256_file(code), "reward_fn_hash": reward_fn_sha256(code),
                       "data_ref": config.get("data_ref", config.get("data")),
                       "resolved_config": copy.deepcopy(runtime_config), "config_hash": object_digest(runtime_config),
                       "worktree": worktree_identity()},
        "rigor": copy.deepcopy(config.get("rigor", {})),
        "results": copy.deepcopy(config.get("results", {})),
        "evidence": copy.deepcopy(config.get("evidence", {})),
    }
    # Creating a draft must never rewrite a historical record or source file.
    sign._exclusive_write(Path(out), (json.dumps(record, indent=2, allow_nan=False) + "\n").encode(), 0o644)
    return record


def _schema_errors(record):
    try:
        sign.canonical_bytes(record)
        from jsonschema import Draft202012Validator
        schema = {
            "type": "object",
            "required": ["run_id", "spec_version", "min_report_schema", "min_report_schema_hash",
                         "min_report_rl", "provenance", "rigor", "results", "evidence"],
            "properties": {
                "run_id": {"type": "string", "minLength": 1},
                "spec_version": {"const": SPEC_VERSION},
                "min_report_schema": {"const": MIN_REPORT_SCHEMA},
                "min_report_schema_hash": {"const": _schema_digest()},
                "min_report_rl": {"$ref": "#/$defs/min_report"},
                **{key: {"type": "object"} for key in ("provenance", "rigor", "results", "evidence")},
            },
            "$defs": _shared_schema()["$defs"],
        }
        errors = [error.message for error in Draft202012Validator(schema).iter_errors(record)]
        min_report = _object(_object(record).get("min_report_rl"))
        for item, definition in schema["$defs"]["min_report"]["properties"].items():
            values = _object(min_report.get(item))
            for field in definition["properties"]:
                value = values.get(field)
                if field not in {"source", "description"} and type(value) is str and not value.strip():
                    errors.append(f"{item}.{field}: empty string is not substantive reporting")
        return errors
    except (ValueError, ImportError, OSError, KeyError, TypeError) as exc:
        return [str(exc)]


def reporting_coverage(min_report):
    """Use the registry's shared, schema-denominated reporting-leaf metric."""
    try:
        from platform_hybrid.registry import query
    except ImportError:  # Direct script invocation needs the repository import root.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from platform_hybrid.registry import query
    sanitized = {item: _object(_object(min_report).get(item)) for item in SEVEN}
    score, item_scores = query.badge({"min_report": sanitized})
    return {"coverage_version": query.COVERAGE_VERSION,
            "reported_leaf_coverage_v3": score / 100,
            "item_presence_coverage_v3": round(sum(value > 0 for value in item_scores.values()) / len(SEVEN), 4),
            "items": item_scores}


def _object(value):
    return value if type(value) is dict else {}


def _full_hash(value):
    return type(value) is str and FULL_HASH.fullmatch(value) is not None


def _number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def _positive_int(value):
    return type(value) is int and value > 0


def _artifact(reference):
    reference = _object(reference)
    path = resolve_source_path(reference.get("path"))
    if path is None or not _full_hash(reference.get("sha256")):
        raise ValueError("artifact requires a repo-local path and full SHA-256")
    raw = path.read_bytes()
    if digest(raw) != reference["sha256"]:
        raise ValueError(f"artifact hash mismatch: {reference['path']}")
    document = sign.strict_json_loads(raw)
    if type(document) is not dict:
        raise ValueError("evidence artifact must be a JSON object")
    return document


def _split_evidence(record):
    evidence = _object(record.get("evidence"))
    split = _artifact(evidence.get("split"))
    train, heldout = split.get("train_ids"), split.get("heldout_ids")
    for ids in (train, heldout):
        if type(ids) is not list or not ids or not all(type(item) is str and item for item in ids):
            raise ValueError("saved train/heldout identifiers must be nonempty string lists")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate split identifiers")
    if set(train) & set(heldout):
        raise ValueError("train/heldout identifiers overlap")
    held = _object(_object(record.get("min_report_rl")).get("heldout_split"))
    if (not _positive_int(held.get("n")) or held["n"] != len(heldout)
            or held.get("disjoint_from_reward_env") is not True):
        raise ValueError("heldout count/disjointness contradicts saved split")
    data_ref = _object(record.get("provenance")).get("data_ref")
    if not isinstance(data_ref, str) or not data_ref or split.get("data_ref") != data_ref:
        raise ValueError("saved split does not identify the declared dataset")
    if _object(record.get("rigor")).get("heldout_disjoint") is not True:
        raise ValueError("rigor heldout disjointness is missing or contradictory")
    return split


def _run_evidence(record):
    rigor, evidence, pv = (_object(record.get(key)) for key in ("rigor", "evidence", "provenance"))
    seeds, references = rigor.get("seeds"), evidence.get("runs")
    if type(seeds) is not list or type(references) is not list or len(references) != len(seeds):
        raise ValueError("one completed run artifact is required per distinct seed")
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("run seeds must be distinct integers")
    paths = [resolve_source_path(_object(ref).get("path")) for ref in references]
    if None in paths or len(set(paths)) != len(paths):
        raise ValueError("run artifact paths must be distinct")
    split_hash = _object(evidence.get("split")).get("sha256")
    metric = _object(record.get("results")).get("metric")
    if not isinstance(metric, str) or not metric:
        raise ValueError("results must identify a held-out metric")
    values = {}
    for reference in references:
        run = _artifact(reference)
        seed = run.get("seed")
        if type(seed) is not int or seed not in seeds or seed in values:
            raise ValueError("completed run seed does not match the declared distinct seeds")
        if run.get("status") != "completed" or run.get("run_id") != record.get("run_id"):
            raise ValueError("completed run identity/status is missing or inconsistent")
        for field in ("code_hash", "config_hash"):
            if run.get(field) != pv.get(field) or not _full_hash(run.get(field)):
                raise ValueError(f"run {field} differs from the record anchor")
        if run.get("split_sha256") != split_hash or not _full_hash(split_hash):
            raise ValueError("run is not bound to the saved held-out split")
        value = _object(run.get("heldout_metrics")).get(metric)
        if not _number(value):
            raise ValueError("run lacks a finite held-out metric")
        values[seed] = value
    if set(values) != set(seeds) or rigor.get("reports_heldout") is not True:
        raise ValueError("held-out reporting claim is missing or inconsistent")
    return values


def _analysis_evidence(record):
    evidence, results, rigor = (_object(record.get(key)) for key in ("evidence", "results", "rigor"))
    values = _run_evidence(record)
    analysis = _artifact(evidence.get("analysis"))
    expected_inputs = [_object(ref).get("sha256") for ref in evidence["runs"]]
    if (analysis.get("input_run_sha256") != expected_inputs
            or analysis.get("metric") != results.get("metric")):
        raise ValueError("analysis must bind the exact ordered run digests and metric")
    if not isinstance(analysis.get("method"), str) or not analysis["method"].strip():
        raise ValueError("analysis method is unreported")
    if type(analysis.get("n")) is not int or analysis["n"] != len(values):
        raise ValueError("analysis sample count contradicts the runs")
    if len(values) < 2:
        raise ValueError("sample dispersion requires at least two runs")
    mean, sd = statistics.mean(values.values()), statistics.stdev(values.values())
    for field, expected in (("mean", mean), ("sd", sd)):
        if not _number(analysis.get(field)) or not math.isclose(analysis[field], expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"analysis {field} does not recompute from completed runs")
        if results.get(field) != analysis[field] or not _number(results.get(field)):
            raise ValueError(f"reported {field} contradicts the saved analysis")
    interval, confidence = analysis.get("ci"), analysis.get("confidence")
    if (type(interval) is not list or len(interval) != 2 or not all(_number(x) for x in interval)
            or not interval[0] <= mean <= interval[1]
            or not _number(confidence) or not 0 < confidence < 1):
        raise ValueError("analysis lacks a finite ordered interval, confidence, or contains a contradictory mean")
    if (type(results.get("ci")) is not list or not all(_number(value) for value in results["ci"])
            or results["ci"] != interval or not _number(results.get("confidence"))
            or results["confidence"] != confidence or rigor.get("reports_ci") is not True):
        raise ValueError("reported uncertainty contradicts the saved analysis")
    expected_per_seed = [{"seed": seed, "value": values[seed]} for seed in rigor["seeds"]]
    per_seed = results.get("per_seed")
    if (type(per_seed) is not list or any(type(row) is not dict or type(row.get("seed")) is not int
                                        or not _number(row.get("value")) for row in per_seed)
            or per_seed != expected_per_seed):
        raise ValueError("reported per-seed results contradict completed runs")
    return analysis


def _decontamination_evidence(record):
    split = _split_evidence(record)
    evidence = _object(record.get("evidence"))
    report = _artifact(evidence.get("decontamination"))
    claim = _object(_object(record.get("min_report_rl")).get("decontamination"))
    # This verifies the declared identifier-disjoint check only, not semantic
    # dataset decontamination or the statistical adequacy of a parser probe.
    if (report.get("method") != "split-id-disjoint/1.0" or report.get("verified") is not True
            or report.get("performed") is not True or claim.get("performed") is not True
            or claim.get("verified", True) is not True
            or report.get("split_sha256") != _object(evidence.get("split")).get("sha256")):
        raise ValueError("decontamination claim needs a verified, split-bound check artifact")
    overlap = len(set(split["train_ids"]) & set(split["heldout_ids"]))
    if type(report.get("overlap_count")) is not int or report["overlap_count"] != overlap:
        raise ValueError("decontamination overlap count does not recompute")
    if (type(claim.get("parser_robustness_probe")) is not bool
            or report.get("parser_robustness_probe") is not claim["parser_robustness_probe"]):
        raise ValueError("parser probe reporting is missing or contradictory")
    return report


def _checks(record):
    checks = []
    def add(condition, weight, ok, bad):
        checks.append({"level": "PASS" if condition else "FAIL", "weight": weight,
                       "msg": ok if condition else bad})
    def evidence_check(function, weight, label):
        try:
            function(record)
            add(True, weight, label, "")
        except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
            add(False, weight, "", label + ": " + str(exc))

    errors = _schema_errors(record)
    add(not errors, 1, "versioned record schema valid", "record schema invalid: " + "; ".join(errors))
    record = _object(record)
    mr, pv, rigor = (_object(record.get(key)) for key in ("min_report_rl", "provenance", "rigor"))
    try:
        coverage = reporting_coverage(mr)
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        coverage = {"coverage_version": "reported-substantive-leaf-mean-v3",
                    "reported_leaf_coverage_v3": 0, "item_presence_coverage_v3": 0,
                    "items": {item: 0 for item in SEVEN}}
    for item in SEVEN:
        add(coverage["items"].get(item) == 1, 1, f"MIN-REPORT item complete: {item}",
            f"MIN-REPORT item incomplete: {item}")
    source = resolve_source_path(pv.get("code_file"))
    label = str(source.relative_to(REPO_ROOT.resolve())) if source else pv.get("code_file")
    add(source is not None, 2, f"provenance source exists: {label}", f"provenance source missing: {label}")
    live_hash = sha256_file(pv.get("code_file"))
    add(_full_hash(pv.get("code_hash")) and pv["code_hash"] == live_hash, 2,
        "full code_hash verified", "full code_hash missing or mismatched")
    reward_hash = reward_fn_sha256(pv.get("code_file"))
    add(_full_hash(pv.get("reward_fn_hash")) and pv["reward_fn_hash"] == reward_hash, 1,
        "full reward_fn_hash verified", "full reward_fn_hash missing or mismatched")
    commit = pv.get("git_sha")
    add(type(commit) is str and FULL_COMMIT.fullmatch(commit) is not None and commit == git_sha(), 2,
        "full git_sha verified against repository HEAD", "git_sha missing, abbreviated, or not this repository HEAD")
    worktree = pv.get("worktree")
    worktree_ok = (type(worktree) is dict and type(worktree.get("dirty")) is bool
                   and type(worktree.get("git_tree")) is str
                   and FULL_COMMIT.fullmatch(worktree["git_tree"]) is not None
                   and _full_hash(worktree.get("tracked_diff_hash"))
                   and worktree == worktree_identity())
    add(worktree_ok, 1,
        "tracked worktree identity verified", "tracked worktree identity missing or mismatched")
    config = pv.get("resolved_config")
    try:
        config_ok = type(config) is dict and _full_hash(pv.get("config_hash")) and pv["config_hash"] == object_digest(config)
    except ValueError:
        config_ok = False
    add(config_ok, 2, "full config_hash verified against embedded resolved configuration", "resolved configuration or full config_hash missing/mismatched")
    try:
        claims_ok = (type(config) is dict and mr == canonical_min_report(config)
                     and pv.get("code_file") == config.get("code_file")
                     and pv.get("data_ref") == config.get("data_ref", config.get("data")))
    except (ValueError, TypeError, KeyError):
        claims_ok = False
    add(claims_ok, 1, "record declarations match resolved configuration", "record declarations contradict resolved configuration")
    loss, kl, sampler = (_object(mr.get(item)) for item in ("loss_form", "reference_kl", "sampler_backend"))
    clipping = loss.get("clipping_applied")
    eps = [loss.get("clip_eps_low"), loss.get("clip_eps_high")]
    loss_ok = ((clipping is False and eps == [None, None])
               or (clipping is True and all(_number(value) and value >= 0 for value in eps)))
    kl_ok = (_number(kl.get("kl_beta")) and kl["kl_beta"] >= 0
             and (kl.get("reference_policy") is True
                  or kl.get("reference_policy") is False and kl["kl_beta"] == 0))
    sampler_ok = (_number(sampler.get("temperature")) and sampler["temperature"] >= 0
                  and _number(sampler.get("top_p")) and 0 < sampler["top_p"] <= 1)
    add(loss_ok and kl_ok and sampler_ok, 1, "loss/KL/sampler semantics consistent",
        "loss/KL/sampler settings missing or contradictory")
    group = _object(mr.get("group_size_schedule"))
    group_ok = (_positive_int(group.get("initial_g")) and group.get("schedule") in ("fixed", "adaptive")
                and (group["schedule"] != "adaptive" or isinstance(group.get("adaptation_rule"), str)
                     and bool(group["adaptation_rule"].strip())))
    add(group_ok, 1, "group-size schedule values valid", "group-size schedule missing or invalid")
    seeds, count = rigor.get("seeds"), rigor.get("n_seeds")
    seeds_ok = (type(count) is int and count >= 3 and type(seeds) is list
                and all(type(seed) is int for seed in seeds) and count == len(seeds) == len(set(seeds)))
    add(seeds_ok, 3, "minimum distinct-seed requirement met (at least 3)", "minimum distinct-seed requirement not met (distinct integer seeds, exact count, at least 3)")
    evidence_check(_run_evidence, 3, "completed run artifacts and held-out metrics verified")
    evidence_check(_split_evidence, 2, "saved split identifiers and held-out count verified")
    evidence_check(_decontamination_evidence, 1, "decontamination artifact verified (identifier disjointness only)")
    evidence_check(_analysis_evidence, 2, "saved analysis inputs, results and uncertainty verified")
    return checks, coverage


def _finish(record, checks, coverage, authenticated, authentication_message):
    evidence_valid = all(check["level"] == "PASS" for check in checks)
    checks.append({"level": "PASS" if authenticated else "FAIL", "weight": 2,
                   "msg": authentication_message})
    accepted = evidence_valid and authenticated
    pct = 100 * coverage["reported_leaf_coverage_v3"]
    return {"run_id": _object(record).get("run_id"), "spec_version": SPEC_VERSION,
            "accepted": accepted, "authenticated": authenticated,
            "assurance": "trusted-ed25519" if authenticated else "unauthenticated",
            "schema_valid": checks[0]["level"] == "PASS", "evidence_valid": evidence_valid,
            "record_digest_sha256": object_digest(record) if not _schema_errors(record) else None,
            "score_pct": round(pct, 1), "grade": "A" if accepted else "REJECTED",
            "coverage": coverage, "checks": checks,
            "limitations": ["Signatures establish approval by the configured key, not that execution occurred.",
                            "Three distinct completed seeds are a replication rule, not a power claim.",
                            "Intervals are checked against saved analysis; statistical adequacy requires review."]}


def verify(record, strict=False):
    """Unsigned diagnostics. ``strict`` remains accepted for caller compatibility."""
    checks, coverage = _checks(record)
    return _finish(record, checks, coverage, False,
                   "UNAUTHENTICATED diagnostic: publication requires audit_file with a trusted key")


def audit_file(record_path, trusted_key, allowed_algorithms=("ed25519",)):
    """The only publication gate: strict JSON, caller trust, schema, all evidence."""
    try:
        record = sign.load_json(record_path)
    except (ValueError, OSError) as exc:
        record = {}
        checks, coverage = _checks(record)
        checks[0]["msg"] = "record loading failed: " + str(exc)
        return _finish(record, checks, coverage, False, "record authentication failed")
    checks, coverage = _checks(record)
    try:
        envelope = sign.load_json(sign.sig_path(record_path))
        authenticated = sign.verify_record(record, envelope, trusted_key, allowed_algorithms)
    except (ValueError, OSError):
        authenticated = False
    return _finish(record, checks, coverage, authenticated,
                   "trusted Ed25519 signature verified" if authenticated else "trusted Ed25519 signature missing, invalid, or disallowed")


def _write_audit(report, record_path, output, trusted_key=None):
    record_path, output = Path(record_path), Path(output)
    # Resolve symlinks and check hardlinks before creating/replacing any output.
    if output.resolve() == record_path.resolve() or (output.exists() and os.path.samefile(output, record_path)):
        raise ValueError("audit output must not overwrite its input record")
    if output.is_symlink():
        raise ValueError("audit output must not be a symlink")
    # Do not overwrite signatures, trusted keys, source, or referenced evidence.
    protected = [sign.sig_path(record_path)]
    if trusted_key is not None:
        protected.append(Path(trusted_key))
    try:
        record = sign.load_json(record_path)
        pv, evidence = _object(record.get("provenance")), _object(record.get("evidence"))
        source = resolve_source_path(pv.get("code_file"))
        if source:
            protected.append(source)
        refs = [evidence.get(key) for key in ("split", "analysis", "decontamination")]
        if type(evidence.get("runs")) is list:
            refs += evidence["runs"]
        protected.extend(path for ref in refs if (path := resolve_source_path(_object(ref).get("path"))))
    except (ValueError, OSError, AttributeError):
        pass
    if any(output.resolve() == path.resolve() or (output.exists() and path.exists() and os.path.samefile(output, path)) for path in protected):
        raise ValueError("audit output must not overwrite evidence or a signature")
    fd, temporary = tempfile.mkstemp(prefix=".audit-", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--run", required=True)
    gen.add_argument("--config")
    gen.add_argument("--out", required=True)
    audit = commands.add_parser("verify")
    audit.add_argument("record", type=Path)
    audit.add_argument("--trusted-key", type=Path)
    audit.add_argument("--strict", action="store_true", help="compatibility alias; publication is always strict")
    audit.add_argument("--diagnostic", action="store_true", help="unsigned diagnostics; never publication accepted")
    audit.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "generate":
            generate(args.run, args.config, args.out)
            print(f"wrote unsigned draft {args.out}")
            return 0
        if args.diagnostic and args.strict:
            raise ValueError("--diagnostic and --strict cannot be combined")
        report = (verify(sign.load_json(args.record)) if args.diagnostic
                  else audit_file(args.record, args.trusted_key))
        output = args.out or args.record.with_name(args.record.name + ".audit.json")
        _write_audit(report, args.record, output, args.trusted_key)
        for check in report["checks"]:
            print(f"[{check['level']}] {check['msg']}")
        print(f"PUBLICATION: {'ACCEPT' if report['accepted'] else 'REJECT'}; "
              f"reporting completeness: {report['score_pct']}%; assurance: {report['assurance']}")
        return 0 if report["accepted"] or args.diagnostic else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
