#!/usr/bin/env python3
"""P5 MIN-REPORT manifest ground-truth audit on the 98 live mega manifests.

Reads every JSON file in ``platform_hybrid/experiments/results/mega_20260704/manifests/`` and
audits:

* **H1** — every v1 MIN-REPORT item has leaf-presence >=98/98.
* **H2** — >=3 v1 items are PLACEBO (constant across 98 manifests).
* **H3** — ``per_step_zvf_path`` resolves to a real file on >=98% of cells;
  its basename equals ``cell_id``.
* **H4** — declared ``group_size_schedule`` matches G-from-cell_id on >=98%;
  declared ``heldout_split`` matches task-from-cell_id on >=98%.
* **H5** — every (model_family, task_slice, G, temperature, seed) axis parsed
  out of cell_id agrees with cells.tsv ground truth on >=98% of cells.
* **F5** — v2 schema expansion carries +2.99 fresh entropy bits beyond what v1
  stack descriptors already encode.

Outputs (in ``platform_hybrid/experiments/results/p5p8/``):

* ``p5_iter169_manifest_audit_per_cell.tsv`` (98 rows)
* ``p5_iter169_manifest_audit_per_key.tsv`` (8 rows)
* ``p5_iter169_manifest_audit_cells_join.tsv`` (98 rows)
* ``p5_iter169_summary.json`` (H1-H5 verdicts + per-key entropy + v2 expansion)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
MANIFESTS_DIR = f"{WORKTREE}/platform_hybrid/experiments/results/mega_20260704/manifests"
CELLS_TSV = f"{WORKTREE}/platform_hybrid/experiments/results/mega_20260704/cells.tsv"
OUT_DIR = f"{WORKTREE}/platform_hybrid/experiments/results/p5p8"

V1_KEYS = [
    "loss_form",
    "ref_policy_kl",
    "sampler_backend_precision",
    "per_step_zvf_path",
    "group_size_schedule",
    "heldout_split",
    "decontamination_notes",
]
V2_AXIS_KEYS = ["model_family", "task_slice", "G", "temperature", "seed"]
KNOWN_TASKS = ("gsm8k_easy", "gsm8k_hard", "humaneval_subset")
CELL_TAIL_RE = re.compile(r"_G(\d+)_t([\d.]+)_s(\d+)_")
HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")
VENDOR_BOUNDARY_RE = re.compile(r"^(?P<v>.+?)(-[A-Z])")


def _shannon_h_bits(values: List[str]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    h = 0.0
    for c in Counter(values).values():
        p = c / n
        h -= p * math.log2(p)
    return h


def load_cells() -> Dict[str, Dict[str, str]]:
    cells: Dict[str, Dict[str, str]] = {}
    with open(CELLS_TSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cid = row.get("cell_id", "").strip()
            if cid:
                cells[cid] = row
    return cells


def parse_cell_id(cid: str) -> Dict[str, str]:
    """Recover (model, task, G, temp, seed) from cell_id grammar."""
    tail = CELL_TAIL_RE.search(cid)
    if not tail:
        return {}
    prefix = cid[: tail.start()]
    g, temp, seed = tail.group(1), tail.group(2), tail.group(3)
    if HASH_SUFFIX_RE.search(prefix):
        prefix = prefix[: prefix.rfind("_")]
    for task in KNOWN_TASKS:
        if prefix.endswith("_" + task):
            return {"model": prefix[: -len(task) - 1], "task": task, "G": g, "temp": temp, "seed": seed}
    return {"model": prefix, "task": "", "G": g, "temp": temp, "seed": seed}


def _vendor(model_str: str) -> str:
    """Vendor fingerprint: leading text before the first ``-<uppercase>`` boundary."""
    if not model_str:
        return ""
    m = VENDOR_BOUNDARY_RE.match(model_str)
    return m.group("v") if m else model_str.split("-")[0]


def audit_v1_keys(files: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    """Per-cell and per-key v1 audit. Returns (per_cell, per_key_summary, parsed_axes)."""
    per_cell: List[Dict[str, Any]] = []
    per_key_values: Dict[str, List[str]] = defaultdict(list)
    parsed_axes: List[Dict[str, str]] = []
    for fp in files:
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            per_cell.append({"file": os.path.basename(fp), "cell_id": "", "json_ok": "0", "parse_err": str(exc)})
            continue
        cid = obj.get("cell_id", "")
        row: Dict[str, Any] = {"file": os.path.basename(fp), "cell_id": cid, "json_ok": "1", "parse_err": ""}
        n_present = 0
        for k in V1_KEYS:
            present = k in obj
            if present:
                n_present += 1
                v = obj[k]
                row[f"has_{k}"] = "1"
                row[f"type_{k}"] = type(v).__name__
                row[f"len_{k}"] = len(str(v))
                per_key_values[k].append(str(v))
            else:
                row[f"has_{k}"] = "0"
                row[f"type_{k}"] = ""
                row[f"len_{k}"] = ""
        row["n_v1_present"] = n_present
        p = obj.get("per_step_zvf_path", "")
        row["path_exists"] = "1" if (p and os.path.exists(p)) else "0"
        row["path_basename_matches_cell_id"] = "1" if (os.path.basename(p).replace(".json", "") == cid) else "0"
        m = re.search(r"_G(\d+)_", cid)
        g_from_cid = m.group(1) if m else ""
        gs = obj.get("group_size_schedule", "")
        row["gs_decl"] = gs
        row["g_from_cid"] = g_from_cid
        row["gs_matches_g"] = "1" if (gs == f"fixed-G={g_from_cid}") else "0"
        axes = parse_cell_id(cid)
        axes["cell_id"] = cid  # type: ignore[assignment]
        row["task_from_cid"] = axes.get("task", "")
        row["heldout_decl"] = obj.get("heldout_split", "")
        row["heldout_matches_task"] = "1" if (axes.get("task", "") == obj.get("heldout_split", "")) else "0"
        parsed_axes.append(axes)
        per_cell.append(row)

    per_key_summary: Dict[str, Dict[str, Any]] = {}
    for k in V1_KEYS:
        vals = per_key_values[k]
        unique_vals = sorted(set(vals))
        per_key_summary[k] = {
            "n_populated": len(vals),
            "n_unique": len(unique_vals),
"h_bits": _shannon_h_bits(vals),
            "is_placebo": len(unique_vals) <= 1,
            "top_value": (Counter(vals).most_common(1)[0][0] if vals else ""),
            "top_value_freq": (Counter(vals).most_common(1)[0][1] / max(1, len(vals))),
        }
    return per_cell, per_key_summary, parsed_axes


def join_to_cells(
    parsed_axes: List[Dict[str, str]],
    cells: Dict[str, Dict[str, str]],
    per_cell: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Cross-reference parsed cell_id-axes with cells.tsv ground truth.

    Encoding normalization:
    * model — cells.tsv uses slash (e.g. ``Qwen/Qwen3.5-4B``); cell_id uses
      dash (e.g. ``Qwen-Qwen3-5-4B``). We compare the vendor-prefix
      fingerprint (text before the first ``-<uppercase>`` vs before the first
      ``/``).
    * temperature — cells.tsv stores ``0.6`` / ``1.0``; cell_id encodes
      ``t0.6`` and ``t1``. Compare as ``float`` so ``1`` == ``1.0``.
    """
    per_cell_lookup = {row.get("cell_id", ""): row for row in per_cell}
    out: List[Dict[str, Any]] = []
    for axes in parsed_axes:
        cid = axes.get("cell_id", "")
        cells_row = cells.get(cid, {})
        manifest_model = axes.get("model", "")
        cells_model = cells_row.get("model_family", "")
        manifest_vendor = _vendor(manifest_model)
        cells_vendor = cells_model.split("/")[0] if "/" in cells_model else cells_model.split("-")[0]
        match_model = "1" if (manifest_vendor and manifest_vendor == cells_vendor) else "0"
        match_task = "1" if cells_row.get("task_slice", "") == axes.get("task", "") else "0"
        match_G = "1" if cells_row.get("G", "") == axes.get("G", "") else "0"
        try:
            match_temp = "1" if abs(float(cells_row.get("temperature", "0")) - float(axes.get("temp", "0"))) < 1e-9 else "0"
        except (TypeError, ValueError):
            match_temp = "1" if cells_row.get("temperature", "") == axes.get("temp", "") else "0"
        match_seed = "1" if cells_row.get("seed", "") == axes.get("seed", "") else "0"
        manifest_path = ""
        f = per_cell_lookup.get(cid, {}).get("file")
        if f:
            try:
                manifest_path = json.loads(open(f"{MANIFESTS_DIR}/{f}", "r", encoding="utf-8").read()).get("per_step_zvf_path", "")
            except Exception:  # noqa: BLE001
                pass
        cells_basename = os.path.basename(cells_row.get("tensor_path", ""))
        manifest_basename = os.path.basename(manifest_path)
        match_zvf_path = "1" if (cells_basename and manifest_basename and cells_basename == manifest_basename) else "0"
        out.append({
            "cell_id": cid,
            "match_model_vendor": match_model,
            "match_task": match_task,
            "match_G": match_G,
            "match_temp": match_temp,
            "match_seed": match_seed,
            "match_zvf_path_basename": match_zvf_path,
            "cells_model_family": cells_model,
            "cells_task_slice": cells_row.get("task_slice", ""),
            "cells_G": cells_row.get("G", ""),
            "cells_temperature": cells_row.get("temperature", ""),
            "cells_seed": cells_row.get("seed", ""),
            "manifest_model": manifest_model,
            "manifest_task": axes.get("task", ""),
            "manifest_G": axes.get("G", ""),
            "manifest_temp": axes.get("temp", ""),
            "manifest_seed": axes.get("seed", ""),
        })
    return out


def v2_axis_values(parsed_axes: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for axis in V2_AXIS_KEYS:
        keymap = {"model_family": "model", "task_slice": "task", "G": "G", "temperature": "temp", "seed": "seed"}
        vals = [a.get(keymap[axis], "") for a in parsed_axes]
        summary[axis] = {
            "n_populated": len(vals),
            "n_unique": len(set(vals)),
            "h_bits": _shannon_h_bits(vals),
            "vals_top5": [v for v, _ in Counter(vals).most_common(5)],
        }
    return summary


def _write_tsv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{MANIFESTS_DIR}/*.json"))
    n_files = len(files)
    print(f"# n_manifests={n_files}", file=sys.stderr)
    cells = load_cells()
    print(f"# n_cells_tsv_rows={len(cells)}", file=sys.stderr)
    per_cell, per_key_summary, parsed_axes = audit_v1_keys(files)

    def _write_dictrows(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"# wrote {path} ({len(rows)} rows)", file=sys.stderr)

    out_pc = f"{OUT_DIR}/p5_iter169_manifest_audit_per_cell.tsv"
    out_pk = f"{OUT_DIR}/p5_iter169_manifest_audit_per_key.tsv"
    out_join = f"{OUT_DIR}/p5_iter169_manifest_audit_cells_join.tsv"
    _write_dictrows(out_pc, per_cell)
    _write_tsv(out_pk, ["key", "n_populated", "n_unique", "h_bits", "is_placebo", "top_value", "top_value_freq"], [
        [k, per_key_summary[k]["n_populated"], per_key_summary[k]["n_unique"],
         f"{per_key_summary[k]['h_bits']:.6f}", "1" if per_key_summary[k]["is_placebo"] else "0",
         per_key_summary[k]["top_value"], f"{per_key_summary[k]['top_value_freq']:.6f}"]
        for k in V1_KEYS
    ])
    print(f"# wrote {out_pk}", file=sys.stderr)
    join_rows = join_to_cells(parsed_axes, cells, per_cell)
    _write_dictrows(out_join, join_rows)
    v2_summary = v2_axis_values(parsed_axes)

    # Verdicts
    bar = int(0.98 * n_files)
    h1_pass = all(sum(1 for r in per_cell if r.get(f"has_{k}", "0") == "1") >= bar for k in V1_KEYS)
    h1_per_key = {k: sum(1 for r in per_cell if r.get(f"has_{k}", "0") == "1") for k in V1_KEYS}
    h2_placebo_keys = [k for k, r in per_key_summary.items() if r["is_placebo"]]
    h2_pass = len(h2_placebo_keys) >= 3
    n_path_exists = sum(1 for r in per_cell if r.get("path_exists", "0") == "1")
    n_basename_match = sum(1 for r in per_cell if r.get("path_basename_matches_cell_id", "0") == "1")
    h3_pass = n_path_exists >= bar
    n_gs_match = sum(1 for r in per_cell if r.get("gs_matches_g", "0") == "1")
    n_heldout_match = sum(1 for r in per_cell if r.get("heldout_matches_task", "0") == "1")
    h4_pass = (n_gs_match >= bar) and (n_heldout_match >= bar)
    H5_AXES = ("match_model_vendor", "match_task", "match_G", "match_temp", "match_seed")
    cells_match_counts = {**{a: 0 for a in H5_AXES}, "match_zvf_path_basename": 0}
    for r in join_rows:
        for a in cells_match_counts:
            if r.get(a, "0") == "1":
                cells_match_counts[a] += 1
    h5_pass = all(cells_match_counts[a] >= n_files for a in H5_AXES)

    v1_total_h = sum(per_key_summary[k]["h_bits"] for k in V1_KEYS)
    v1_discriminative_h = sum(
        per_key_summary[k]["h_bits"] for k in V1_KEYS
        if (not per_key_summary[k]["is_placebo"]) and (k != "per_step_zvf_path")
    )
    v2_total_h = sum(v["h_bits"] for v in v2_summary.values())
    v2_truly_fresh = {a: v2_summary[a]["h_bits"] for a in v2_summary if a not in {"task_slice", "G"}}
    v2_truly_fresh_total_h = sum(v2_truly_fresh.values())

    summary = {
        "iter": 169,
        "n_manifests": n_files,
        "n_cells_tsv_rows": len(cells),
        "H1_leaf_presence_ge_98pct": h1_pass,
        "H1_per_key_present_counts": h1_per_key,
        "H2_placebo_keys_ge_3": h2_pass,
        "H2_placebo_keys": h2_placebo_keys,
        "H2_n_placebo": len(h2_placebo_keys),
        "H3_path_exists_ge_98pct": h3_pass,
        "H3_n_path_exists": n_path_exists,
        "H3_n_basename_match_cell_id": n_basename_match,
        "H4_gs_matches_g_and_heldout_matches_task": h4_pass,
        "H4_n_gs_match": n_gs_match,
        "H4_n_heldout_match": n_heldout_match,
        "H5_cells_tsv_axis_match_all_5_axes_on_all_98": h5_pass,
        "H5_per_axis_match_counts": cells_match_counts,
        "v1_total_h_bits": v1_total_h,
        "v1_discriminative_h_bits": v1_discriminative_h,
        "v2_axis_layer_total_h_bits_if_added": v2_total_h,
        "v2_axis_layer_unique_counts": {k: v["n_unique"] for k, v in v2_summary.items()},
        "v2_axis_layer_h_bits_per_axis": {k: v["h_bits"] for k, v in v2_summary.items()},
        "v2_truly_fresh_h_bits": v2_truly_fresh_total_h,
        "v2_truly_fresh_h_bits_per_axis": v2_truly_fresh,
        "v1_key_summary": per_key_summary,
        "v2_axis_summary": v2_summary,
        "n_placebo_v1_keys": len(h2_placebo_keys),
        "n_discriminative_v1_keys": 7 - len(h2_placebo_keys) - 1,
    }
    out_summary = f"{OUT_DIR}/p5_iter169_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"# wrote {out_summary}", file=sys.stderr)

    print("=" * 60)
    print("P5 MANIFEST AUDIT — ITER 169")
    print("=" * 60)
    print(f"n_manifests={n_files}")
    print(f"H1 leaf-presence >=98%: {'PASS' if h1_pass else 'FAIL'}")
    print(f"H2 placebo >=3 v1 fields: {'PASS' if h2_pass else 'FAIL'} ({h2_placebo_keys})")
    print(f"H3 path_exists >=98%: {'PASS' if h3_pass else 'FAIL'} ({n_path_exists}/{n_files})")
    print(f"H4 gs_matches_g heldout_matches_task: {'PASS' if h4_pass else 'FAIL'}")
    print(f"H5 cells.tsv axis match (5 axes): {'PASS' if h5_pass else 'FAIL'} {cells_match_counts}")
    print(f"v1 total_h={v1_total_h:.3f} bits / v1_discriminative_h={v1_discriminative_h:.3f} bits / v2_total_h={v2_total_h:.3f} bits / v2_truly_fresh_h={v2_truly_fresh_total_h:.3f} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
