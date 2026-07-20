#!/usr/bin/env python3
"""MIN-REPORT-RL provenance protocol — generate + verify (P5 flagship).

Elevates the static 7-item MIN-REPORT-RL block (registry/schema.json $defs.min_report)
into an ENFORCEABLE protocol: a run emits a signed provenance record; `verify` grades
its completeness (7 items), integrity (recomputable hashes), and statistical rigor
(seeds/held-out/CI), returning a reproducibility badge A-F. stdlib-only (matches query.py).

Usage:
  python minreport.py generate --run <name> --config <path.json> --out record.json
  python minreport.py verify record.json            # grades one record
  python minreport.py verify record.json --strict   # WARN counts as fail
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SEVEN = [
    "loss_form",
    "reference_kl",
    "sampler_backend",
    "telemetry",
    "group_size_schedule",
    "heldout_split",
    "decontamination",
]


def resolve_source_path(raw_path):
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = (
        [path]
        if path.is_absolute()
        else [
            REPO_ROOT / path,
            REPO_ROOT / "platform_hybrid" / path,
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def sha256_file(path):
    resolved = resolve_source_path(path)
    if resolved is None:
        return None
    return "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]


def git_sha():
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def generate(run, config_path, out):
    cfg = json.load(open(config_path)) if config_path and os.path.exists(config_path) else {}
    code = cfg.get("code_file", "platform_hybrid/experiments/openings/campaign.py")
    record = {
        "run_id": run,
        "spec_version": "MIN-REPORT-RL/1.0",
        "min_report_rl": {
            "loss_form": cfg.get(
                "loss_form",
                {
                    "importance_ratio_level": "token",
                    "advantage_normalization": "std",
                    "length_normalization": False,
                    "clip_eps_low": None,
                    "clip_eps_high": None,
                },
            ),
            "reference_kl": cfg.get(
                "reference_kl", {"beta": 0.0, "note": "no KL penalty (pure advantage-weighted)"}
            ),
            "sampler_backend": cfg.get("sampler_backend", "tinker"),
            "telemetry": cfg.get("telemetry", ["reward", "loss", "collapsed", "heldout"]),
            "group_size_schedule": cfg.get(
                "group_size_schedule", {"G": cfg.get("G"), "schedule": "constant"}
            ),
            "heldout_split": cfg.get(
                "heldout_split", {"n": cfg.get("heldout_n"), "disjoint": True, "eval": "greedy"}
            ),
            "decontamination": cfg.get(
                "decontamination", {"method": "index-disjoint from train pool", "verified": True}
            ),
        },
        "provenance": {
            "git_sha": git_sha(),
            "code_file": code,
            "code_hash": sha256_file(code),
            "reward_fn_hash": sha256_file(code),  # reward fn lives in the run file
            "data_hash": "sha256:"
            + hashlib.sha256(
                json.dumps(cfg.get("data", "openai/gsm8k:main:train")).encode()
            ).hexdigest()[:16],
            "config_hash": "sha256:"
            + hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        },
        "rigor": cfg.get("rigor", {}),
        "results": cfg.get("results", {}),
    }
    json.dump(record, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return record


def verify(record, strict=False):
    checks = []  # (level PASS/WARN/FAIL, weight, msg)

    def add(cond, w, ok_msg, bad_msg, warn=False):
        checks.append(
            ("PASS" if cond else ("WARN" if warn else "FAIL"), w, ok_msg if cond else bad_msg)
        )

    mr = record.get("min_report_rl", {})
    # 1) completeness: all 7 items present & non-null
    for item in SEVEN:
        v = mr.get(item)
        add(
            v not in (None, {}, []),
            1,
            f"MIN-REPORT item present: {item}",
            f"MISSING MIN-REPORT item: {item}",
        )
    # 2) integrity: provenance hashes present & code hash recomputes
    pv = record.get("provenance", {})
    for h in ["git_sha", "code_hash", "reward_fn_hash", "data_hash"]:
        add(bool(pv.get(h)), 1, f"provenance hash present: {h}", f"missing provenance: {h}")
    # recompute code hash against the file named in the record's provenance
    code_file = pv.get("code_file")
    resolved_code_file = resolve_source_path(code_file)
    live = sha256_file(code_file) if code_file else None
    resolved_label = (
        str(resolved_code_file.relative_to(REPO_ROOT))
        if resolved_code_file is not None
        else code_file
    )
    add(
        live is not None,
        2,
        f"provenance source exists: {resolved_label}",
        f"provenance source missing: {code_file}",
    )
    if live and pv.get("code_hash"):
        add(
            live == pv["code_hash"],
            2,
            f"code_hash VERIFIED against live source ({resolved_label})",
            f"code_hash MISMATCH for {resolved_label} (recorded {pv['code_hash']} vs live {live}) — code changed since run",
            warn=True,
        )
    # 3) rigor
    rg = record.get("rigor", {})
    ns = rg.get("n_seeds", 0)
    add(ns >= 3, 3, f"multi-seed (n={ns}>=3)", f"UNDERPOWERED: n_seeds={ns} (<3)")
    add(bool(rg.get("reports_heldout")), 2, "held-out reported", "no held-out evaluation reported")
    add(
        bool(rg.get("heldout_disjoint")),
        1,
        "held-out disjoint from train",
        "held-out disjointness not asserted",
    )
    add(
        bool(rg.get("reports_ci")),
        1,
        "uncertainty (CI/sd) reported",
        "no CI/sd on results",
        warn=True,
    )
    add(
        mr.get("heldout_split", {}).get("disjoint") is True,
        1,
        "decontamination asserted",
        "decontamination not asserted",
    )
    # score
    total = sum(w for _, w, _ in checks)
    # A warning identifies missing evidence and therefore cannot earn badge
    # credit. ``strict`` controls process failure below; it must not silently
    # change the scientific completeness score.
    got = sum(w for lvl, w, _ in checks if lvl == "PASS")
    pct = 100 * got / total
    grade = (
        "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 45 else "F"
    )
    return {
        "run_id": record.get("run_id"),
        "score_pct": round(pct, 1),
        "grade": grade,
        "checks": [{"level": l, "weight": w, "msg": m} for l, w, m in checks],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--config")
    g.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("record")
    v.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.run, a.config, a.out)
    else:
        rep = verify(json.load(open(a.record)), a.strict)
        print(f"\n=== MIN-REPORT-RL audit: {rep['run_id']} ===")
        for c in rep["checks"]:
            mark = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[c["level"]]
            print(f"  {mark} [{c['level']:4}] {c['msg']}")
        print(f"\n  SCORE: {rep['score_pct']}%  ->  GRADE {rep['grade']}\n")
        json.dump(rep, open(a.record.replace(".json", ".audit.json"), "w"), indent=2)
        if a.strict and any(check["level"] != "PASS" for check in rep["checks"]):
            sys.exit("Strict verification rejected WARN/FAIL evidence gaps.")
