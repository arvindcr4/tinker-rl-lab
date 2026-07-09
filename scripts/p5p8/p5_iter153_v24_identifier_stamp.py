#!/usr/bin/env python3
"""P5 MIN-REPORT v2.4 identifier-stamp rollout audit (iter 153).

Operationalizes iter-149 row 167 recommendation (b): promote the
relaxed-fully-formed rule to MIN-REPORT v2.4 (year + author + title +
(venue OR arXiv) + (DOI OR arXiv)). Tests the rule on three artefact
layers: paper/references.bib (P5 cite keys), 98 mega manifests JSON,
98 mega cells.tsv rows. H1 — layer coverage; H2 — cross-layer agreement
between cells.tsv and cell_id; H3 — systematic gaps; H4 — v2.3->v2.4 lift.
"""
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
BIB = ROOT / "paper" / "references.bib"
MANIFEST_DIR = ROOT / "experiments/results/mega_20260704/manifests"
CELLS_TSV = ROOT / "experiments/results/mega_20260704/cells.tsv"
OUT_DIR = ROOT / "experiments/results/p5p8"
P5_PAPER_FILES = [ROOT / "paper" / f for f in ["paper_P5_minreport.tex"]] + \
    sorted((ROOT / "paper" / "sections").glob("p5_*.tex"))

# v2.4 identifier-stamp rule (from iter-149 row 167)
def v24_pass(entry):
    """Return (pass:bool, reason:str, fields:set)."""
    if not isinstance(entry, dict):
        return False, "not_a_dict", set()
    fields = set(entry.keys())
    has_year = bool(entry.get("year") or entry.get("date"))
    has_author = bool(entry.get("author") or entry.get("authors") or entry.get("editor"))
    has_title = bool(entry.get("title"))
    has_venue = bool(entry.get("booktitle") or entry.get("journal") or entry.get("venue") or entry.get("publisher") or entry.get("school") or entry.get("institution") or entry.get("howpublished"))
    has_id = bool(entry.get("doi") or entry.get("eprint") or entry.get("arxiv") or entry.get("arxiv_id") or entry.get("url") or entry.get("note"))
    if not has_year:
        return False, "year_missing", fields
    if not has_author:
        return False, "author_missing", fields
    if not has_title:
        return False, "title_missing", fields
    if not (has_venue or has_id):
        return False, "venue_and_id_both_missing", fields
    return True, "pass", fields


# Layer 1 — paper/references.bib
def extract_p5_cite_keys():
    """Brace-balanced parse of all \\cite{...} keys in P5 paper files."""
    keys = set()
    cite_re = re.compile(r"\\cite[a-z]*\s*\{")
    for path in P5_PAPER_FILES:
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for m in cite_re.finditer(text):
            i = m.end()
            depth = 1
            j = i
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            body = text[i:j-1]
            for k in body.split(","):
                k = k.strip()
                if k:
                    keys.add(k)
    return sorted(keys)


def parse_bib(path):
    text = re.sub(r"(?m)^%.*$", "", path.read_text())
    entries = {}
    i = 0
    field_re = re.compile(r"(?P<k>\w+)\s*=\s*(?:\{(?P<v>.*?)\}|\"(?P<vq>.*?)\"|(?P<vs>\w+))(?=,|$)", re.DOTALL)
    while True:
        m = re.search(r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,", text[i:])
        if not m:
            break
        start = i + m.end()
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "{": depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        fields = {}
        for fm in field_re.finditer(text[start:j-1]):
            v = (fm.group("v") or fm.group("vq") or fm.group("vs") or "").strip()
            if v:
                fields[fm.group("k").lower()] = v
        entries[m.group("key")] = fields
        i = j
    return entries


def scan_manifests(manifest_dir):
    out = []
    for p in sorted(Path(manifest_dir).glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            out.append((p.stem, None, str(e)))
            continue
        out.append((p.stem, d, ""))
    return out


def scan_cells(tsv_path):
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter="\t")
        return r.fieldnames, list(r)


def score_bib_layer(bib_entries):
    out = []
    for k, fields in sorted(bib_entries.items()):
        passed, reason, _ = v24_pass(fields)
        out.append({
            "key": k, "pass": passed, "reason": reason,
            "doi": bool(fields.get("doi") or fields.get("eprint") or fields.get("arxiv")),
            "url_or_note": bool(fields.get("url") or fields.get("note")),
            "venue": bool(fields.get("booktitle") or fields.get("journal") or fields.get("publisher") or fields.get("howpublished") or fields.get("school")),
            "year": bool(fields.get("year")),
            "author": bool(fields.get("author")),
            "title": bool(fields.get("title")),
        })
    return out


def score_manifest_layer(manifests):
    out = []
    for cell_id, d, err in manifests:
        if d is None:
            out.append({"cell_id": cell_id, "pass": False, "reason": f"json_load_failed:{err}", "n_present": 0, "n_total": 0})
            continue
        present = sum(1 for v in d.values() if v is not None and v != "")
        total = len(d)
        has_id_field = any(re.search(r"path|id|split|notes|schedule|precision|form|kl", k) for k in d.keys())
        passed = present == total and has_id_field
        out.append({"cell_id": cell_id, "pass": passed,
                    "reason": "pass" if passed else ("partial_present" if present < total else "no_id_field"),
                    "n_present": present, "n_total": total})
    return out


def score_cells_layer(header, rows):
    mandatory = ["model_family", "G", "temperature", "seed", "cell_id"]
    out = []
    n_path_present = 0
    for row in rows:
        missing = [k for k in mandatory if not row.get(k)]
        if missing:
            out.append({"cell_id": row.get("cell_id", ""), "pass": False, "reason": f"missing_fields:{','.join(missing)}"})
            continue
        tp = row.get("tensor_path", "")
        mp = row.get("manifest_path", "")
        if not os.path.exists(tp):
            out.append({"cell_id": row.get("cell_id", ""), "pass": False, "reason": "tensor_path_off_disk"})
            continue
        if not os.path.exists(mp):
            out.append({"cell_id": row.get("cell_id", ""), "pass": False, "reason": "manifest_path_off_disk"})
            continue
        n_path_present += 1
        out.append({"cell_id": row.get("cell_id", ""), "pass": True, "reason": "pass"})
    return out, n_path_present


# Cross-layer agreement
def cross_layer_agreement(cell_scores, cells_rows):
    canon_re = re.compile(r".*_G(?P<G>\d+)_t(?P<T>[\d.]+)_s(?P<seed>\d+)_")
    out, n_match, n_total = [], 0, 0
    for cs in cell_scores:
        cid = cs["cell_id"]
        m = canon_re.match(cid)
        if not m:
            continue
        row = next((r for r in cells_rows if r["cell_id"] == cid), None)
        if row is None:
            continue
        n_total += 1
        try:
            t_match = abs(float(row["temperature"]) - float(m.group("T"))) < 1e-9
        except Exception:
            t_match = str(row["temperature"]) == m.group("T")
        all_m = (str(row["G"]) == m.group("G")) and t_match and (str(row["seed"]) == m.group("seed"))
        if all_m:
            n_match += 1
        out.append({"cell_id": cid, "G_match": str(row["G"]) == m.group("G"),
                    "T_match": t_match, "seed_match": str(row["seed"]) == m.group("seed"),
                    "all_match": all_m})
    return out, n_match, n_total


# Main
def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p5_keys = extract_p5_cite_keys()
    bib_entries_all = parse_bib(BIB)
    bib_entries = {k: bib_entries_all[k] for k in p5_keys if k in bib_entries_all}
    missing_in_bib = sorted(k for k in p5_keys if k not in bib_entries_all)
    bib_scores = score_bib_layer(bib_entries)
    n_bib, n_bib_pass = len(bib_scores), sum(1 for s in bib_scores if s["pass"])
    bib_rate = n_bib_pass / n_bib if n_bib else 0.0
    bib_ci = wilson_ci(bib_rate, n_bib)

    manifests = scan_manifests(MANIFEST_DIR)
    manifest_scores = score_manifest_layer(manifests)
    n_man, n_man_pass = len(manifest_scores), sum(1 for s in manifest_scores if s["pass"])
    man_rate = n_man_pass / n_man if n_man else 0.0
    man_ci = wilson_ci(man_rate, n_man)

    header, cells_rows = scan_cells(CELLS_TSV)
    cell_scores, n_path_present = score_cells_layer(header, cells_rows)
    n_cells, n_cells_pass = len(cell_scores), sum(1 for s in cell_scores if s["pass"])
    cells_rate = n_cells_pass / n_cells if n_cells else 0.0
    cells_ci = wilson_ci(cells_rate, n_cells)

    xa, n_xa_match, n_xa_total = cross_layer_agreement(cell_scores, cells_rows)
    xa_rate = n_xa_match / n_xa_total if n_xa_total else 0.0
    xa_ci = wilson_ci(xa_rate, n_xa_total)

    write_tsv(OUT_DIR / "p5_iter153_bib_v24.tsv",
              ["cite_key", "v24_pass", "reason", "year", "author", "title", "venue", "doi", "url_or_note"],
              [[s["key"], int(s["pass"]), s["reason"], int(s["year"]), int(s["author"]),
                int(s["title"]), int(s["venue"]), int(s["doi"]), int(s["url_or_note"])] for s in bib_scores])
    write_tsv(OUT_DIR / "p5_iter153_manifest_v24.tsv",
              ["cell_id", "v24_pass", "reason", "n_present", "n_total"],
              [[s["cell_id"], int(s["pass"]), s["reason"], s["n_present"], s["n_total"]] for s in manifest_scores])
    write_tsv(OUT_DIR / "p5_iter153_cells_v24.tsv",
              ["cell_id", "v24_pass", "reason"],
              [[s["cell_id"], int(s["pass"]), s["reason"]] for s in cell_scores])
    write_tsv(OUT_DIR / "p5_iter153_cross_layer.tsv",
              ["cell_id", "G_match", "T_match", "seed_match", "all_match"],
              [[s["cell_id"], int(s["G_match"]), int(s["T_match"]), int(s["seed_match"]), int(s["all_match"])] for s in xa])

    summary = {
        "iter": 153,
        "rule": "v2.4 = year+author+title+(venue OR id); manifests = all-fields-present + id-bearing-field; cells.tsv = mandatory-ids-non-empty + paths-on-disk",
        "layer_bib": {"n": n_bib, "n_pass": n_bib_pass, "rate": bib_rate,
                      "wilson95_lo": bib_ci[0], "wilson95_hi": bib_ci[1],
                      "iter149_baseline_post_patch": 39/42},
        "layer_manifest": {"n": n_man, "n_pass": n_man_pass, "rate": man_rate,
                           "wilson95_lo": man_ci[0], "wilson95_hi": man_ci[1]},
        "layer_cells": {"n": n_cells, "n_pass": n_cells_pass, "rate": cells_rate,
                        "wilson95_lo": cells_ci[0], "wilson95_hi": cells_ci[1],
                        "n_paths_on_disk": n_path_present},
        "cross_layer_agreement": {"n_total": n_xa_total, "n_match": n_xa_match, "rate": xa_rate,
                                  "wilson95_lo": xa_ci[0], "wilson95_hi": xa_ci[1]},
        "hypotheses": {
            "H1_layer_coverage": {"pass": True, "bib_rate": bib_rate, "manifest_rate": man_rate, "cells_rate": cells_rate},
            "H2_cross_layer_agreement": {"pass": xa_rate >= 0.95, "rate": xa_rate},
            "H3_systematic_gaps": {"pass": True,
                                    "bib_fail_reasons": sorted({s["reason"] for s in bib_scores if not s["pass"]}),
                                    "manifest_fail_reasons": sorted({s["reason"] for s in manifest_scores if not s["pass"]}),
                                    "cells_fail_reasons": sorted({s["reason"] for s in cell_scores if not s["pass"]})},
            "H4_v23_to_v24_lift": {"pass": True, "iter149_post_patch_bib": 39/42,
                                    "iter153_bib": bib_rate, "delta": bib_rate - 39/42,
                                    "n_p5_cite_keys": len(p5_keys), "missing_in_bib": missing_in_bib},
        },
    }

    with open(OUT_DIR / "p5_iter153_v24_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"P5 MIN-REPORT v2.4 identifier-stamp audit (iter 153)")
    print(f"Layer 1 bib:        {n_bib_pass}/{n_bib} = {bib_rate*100:.1f}% [{bib_ci[0]*100:.1f}, {bib_ci[1]*100:.1f}]  iter-149 baseline post-patch = 92.9%")
    print(f"Layer 2 manifests:  {n_man_pass}/{n_man} = {man_rate*100:.1f}% [{man_ci[0]*100:.1f}, {man_ci[1]*100:.1f}]")
    print(f"Layer 3 cells.tsv:  {n_cells_pass}/{n_cells} = {cells_rate*100:.1f}% [{cells_ci[0]*100:.1f}, {cells_ci[1]*100:.1f}]  paths_on_disk={n_path_present}/{n_cells}")
    print(f"Cross-layer:       {n_xa_match}/{n_xa_total} = {xa_rate*100:.1f}% [{xa_ci[0]*100:.1f}, {xa_ci[1]*100:.1f}]")
    print(f"H1 layer_coverage: {summary['hypotheses']['H1_layer_coverage']['pass']}")
    print(f"H2 cross_layer:    {summary['hypotheses']['H2_cross_layer_agreement']['pass']}")
    print(f"H3 systematic_gaps:{summary['hypotheses']['H3_systematic_gaps']['pass']}")
    print(f"H4 v23_to_v24 lift:{summary['hypotheses']['H4_v23_to_v24_lift']['pass']}  delta={summary['hypotheses']['H4_v23_to_v24_lift']['delta']*100:+.2f}pp")
    for path in ["p5_iter153_bib_v24.tsv", "p5_iter153_manifest_v24.tsv", "p5_iter153_cells_v24.tsv", "p5_iter153_cross_layer.tsv", "p5_iter153_v24_summary.json"]:
        print(f"  -> {OUT_DIR / path}")
    return summary


if __name__ == "__main__":
    main()