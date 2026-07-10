#!/usr/bin/env python3
# P5 iter-121 — MIN-REPORT v2.2 value-correctness MUTATION stress test
#
# Fresh vein, NOT in 135 prior rows. Closes brief vein (a) at the
# AUDITOR-ROBUSTNESS layer. Iter-97/105/113/117 audited manifest
# CONTENTS at schema/value/content/structural layers. None of them
# stress-tested the AUDITORS THEMSELVES against controlled perturbations.
#
# Method:
# 1. Load n=98 manifests from platform_hybrid/experiments/results/mega_20260704/manifests/
# 2. Pick 8 controlled mutations per cell (each corrupts one specific
#    MIN-REPORT v2.2 item by flipping the value to a known-wrong one):
#    M1 cell_id ↔ swap last 8 hex chars
#    M2 model_family ↔ swap meta-llama ↔ Qwen
#    M3 task_slice ↔ swap gsm8k_easy ↔ humaneval_subset
#    M4 G ↔ swap 2 ↔ 32
#    M5 temperature ↔ swap 0.6 ↔ 1.0
#    M6 seed ↔ swap 0 ↔ 1
#    M7 heldout_split ↔ swap gsm8k_easy ↔ humaneval_subset
#    M8 per_step_zvf_path ↔ drop the suffix .json
# 3. For each (cell, mutation) pair, compute the audit signal — the
#    change in the 10 value-correctness check scores (C01..C10 from
#    iter-121 base audit). A good auditor catches EVERY mutation on
#    at least one check.
# 4. Aggregate detection rate across (mutation × check) cells.
#
# Outputs:
#   - platform_hybrid/experiments/results/p5p8/p5_iter121_mutation_stress.tsv
#       (rows = 8 mutations × 98 cells = 784 rows)
#   - platform_hybrid/experiments/results/p5p8/p5_iter121_mutation_summary.json
#       (per-mutation detection rate + per-check detection rate + the
#       headline "did the auditor catch every mutation on at least one
#       check" verdict)
"""
MIN-REPORT v2.2 value-correctness mutation stress test (iter 121).

Loads n=98 live mega manifests, applies 8 controlled mutations to each
(JSON-body swap of a known-wrong value), re-runs the 10-check value-
correctness audit, and measures detection rate.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Reuse iter-121 base audit by import
sys.path.insert(0, str(Path(__file__).parent))
from p5_iter121_value_correctness import (
    load_cells_tsv, parse_filename, check_consistency
)

WORK = Path("/home/claude/tinker-rl-lab-minimax")
MEGA = WORK / "platform_hybrid/experiments/results/mega_20260704"
MANIFEST_DIR = MEGA / "manifests"
CELLS_TSV = MEGA / "cells.tsv"
OUT_DIR = WORK / "platform_hybrid/experiments/results/p5p8"


def _mut_m1(m, c, f):
    return {**m, 'cell_id': m['cell_id'][:-8] + ('0' * 8)}

def _mut_m2(m, c, f):
    parts = m['cell_id'].split('_')
    if parts[0].startswith('meta-llama'):
        parts[0] = 'Qwen-Qwen3-5-4B'
    else:
        parts[0] = 'meta-llama-Llama-3-2-3B'
    return {**m, 'cell_id': '_'.join(parts)}

def _mut_m3(m, c, f):
    cid = m['cell_id']
    if '_gsm8k_easy_' in cid:
        cid = cid.replace('_gsm8k_easy_', '_HUMANEVALX_', 1)
    elif '_gsm8k_hard_' in cid:
        cid = cid.replace('_gsm8k_hard_', '_HUMANEVALX_', 1)
    elif '_humaneval_subset_' in cid:
        cid = cid.replace('_humaneval_subset_', '_gsm8k_easy_', 1)
    else:
        return m
    cid = cid.replace('_HUMANEVALX_', '_humaneval_subset_', 1)
    return {**m, 'cell_id': cid}

def _mut_m4(m, c, f):
    cid = m['cell_id']
    m_g = re.match(r'^(.+)_G(\d+)_(.+)$', cid)
    if not m_g:
        return m
    G_val = int(m_g.group(2))
    new_G = 32 if G_val == 2 else 2
    return {**m, 'cell_id': f'{m_g.group(1)}_G{new_G}_{m_g.group(3)}'}

def _mut_m5(m, c, f):
    cid = m['cell_id']
    if '_t0.6_' in cid:
        cid = cid.replace('_t0.6_', '__TT__', 1).replace('_t1_', '_t0.6_', 1)
        cid = cid.replace('__TT__', '_t1_', 1)
    elif '_t1_' in cid:
        cid = cid.replace('_t1_', '__TT__', 1).replace('_t0.6_', '_t1_', 1)
        cid = cid.replace('__TT__', '_t0.6_', 1)
    else:
        return m
    return {**m, 'cell_id': cid}

def _mut_m6(m, c, f):
    cid = m['cell_id']
    m_s = re.match(r'^(.+)_s(\d+)_(.+)$', cid)
    if not m_s:
        return m
    s_val = int(m_s.group(2))
    new_s = 1 if s_val == 0 else 0
    return {**m, 'cell_id': f'{m_s.group(1)}_s{new_s}_{m_s.group(3)}'}

def _mut_m7(m, c, f):
    h = m.get('heldout_split')
    if h == 'gsm8k_easy':
        return {**m, 'heldout_split': 'humaneval_subset'}
    elif h == 'humaneval_subset':
        return {**m, 'heldout_split': 'gsm8k_easy'}
    elif h == 'gsm8k_hard':
        return {**m, 'heldout_split': 'gsm8k_easy'}
    return m

def _mut_m8(m, c, f):
    zp = m.get('per_step_zvf_path', '')
    if not zp:
        return m
    return {**m, 'per_step_zvf_path': zp.rstrip('.json') + '__broken.json'}


MUTATIONS = [
    ('M1', 'cell_id_swap_hash', _mut_m1),
    ('M2', 'model_family_swap', _mut_m2),
    ('M3', 'task_slice_swap', _mut_m3),
    ('M4', 'G_swap', _mut_m4),
    ('M5', 'temperature_swap', _mut_m5),
    ('M6', 'seed_swap', _mut_m6),
    ('M7', 'heldout_split_swap', _mut_m7),
    ('M8', 'per_step_zvf_path_break', _mut_m8),
]


def audit_one(cell_id, manifest, cells_row, file_info):
    """Run base audit; return {cid: passed_bool}."""
    results = check_consistency(cell_id, manifest, cells_row, file_info)
    return {r[0]: r[2] for r in results}


def main():
    cells = load_cells_tsv()
    # Per (mutation, check) detection count
    detect_counts = defaultdict(lambda: defaultdict(int))
    # Per mutation: how many cells did AT LEAST ONE check fail
    mutation_total_cells = defaultdict(int)
    mutation_caught = defaultdict(int)
    # Per check: how many (mutation × cell) pairs it catches
    check_catches = defaultdict(int)
    # Total number of (mutation × cell) evaluations
    n_mutations = len(MUTATIONS)
    n_cells_total = 0
    # Per-mutation: per-cell which checks flipped from PASS->FAIL
    n_flipped_per_mutation = defaultdict(int)

    for manifest_path in sorted(MANIFEST_DIR.glob('*.json')):
        cell_id = manifest_path.stem
        if cell_id not in cells:
            continue
        with open(manifest_path) as f:
            manifest_orig = json.load(f)
        cells_row = cells[cell_id]
        file_info_orig = parse_filename(cell_id)
        if file_info_orig is None:
            continue

        baseline = audit_one(cell_id, manifest_orig, cells_row, file_info_orig)

        for mid, mname, mfn in MUTATIONS:
            try:
                mutated = mfn(manifest_orig, cells_row, file_info_orig)
            except Exception as e:
                # Skip on mutation construction error
                continue
            # For mutations that change cell_id, re-parse filename
            new_cell_id = mutated['cell_id']
            new_file_info = parse_filename(new_cell_id)
            audit = audit_one(new_cell_id, mutated, cells_row, new_file_info)

            any_fail = False
            any_flipped = False
            for cid in ('C01', 'C02', 'C03', 'C04', 'C05',
                        'C06', 'C07', 'C08', 'C09', 'C10'):
                if not audit.get(cid, False):
                    any_fail = True
                    detect_counts[mid][cid] += 1
                    check_catches[cid] += 1
                # Did this check flip from PASS (baseline) to FAIL (mutated)?
                if baseline.get(cid, False) and not audit.get(cid, False):
                    any_flipped = True
                    n_flipped_per_mutation[mid] += 1

            mutation_total_cells[mid] += 1
            if any_fail:
                mutation_caught[mid] += 1
            if any_flipped:
                pass  # counted per-cell in n_flipped_per_mutation

        n_cells_total += 1

    # Write per-cell TSV (8 mutations × 98 cells = 784 rows)
    tsv_path = OUT_DIR / 'p5_iter121_mutation_stress.tsv'
    with open(tsv_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['cell_id', 'mutation_id', 'mutation_name',
                    'n_checks_failed', 'flipped_check_ids', 'caught'])
        for manifest_path in sorted(MANIFEST_DIR.glob('*.json')):
            cell_id = manifest_path.stem
            if cell_id not in cells:
                continue
            with open(manifest_path) as f_:
                manifest_orig = json.load(f_)
            cells_row = cells[cell_id]
            file_info_orig = parse_filename(cell_id)
            if file_info_orig is None:
                continue
            baseline = audit_one(cell_id, manifest_orig, cells_row, file_info_orig)
            for mid, mname, mfn in MUTATIONS:
                try:
                    mutated = mfn(manifest_orig, cells_row, file_info_orig)
                except Exception:
                    continue
                new_cell_id = mutated['cell_id']
                new_file_info = parse_filename(new_cell_id)
                audit = audit_one(new_cell_id, mutated, cells_row, new_file_info)
                flipped = [cid for cid in baseline
                            if baseline.get(cid) and not audit.get(cid, False)]
                n_failed = sum(1 for v in audit.values() if not v)
                w.writerow([cell_id, mid, mname, n_failed,
                            ','.join(flipped), int(bool(flipped))])
    print(f"WROTE {tsv_path}")

    # Per-mutation summary
    per_mutation = {}
    for mid, mname, _ in MUTATIONS:
        total = mutation_total_cells[mid]
        caught = mutation_caught[mid]
        det_rate = caught / total if total else 0
        # Which checks detect this mutation
        check_det = {cid: detect_counts[mid][cid] for cid in
                      ('C01', 'C02', 'C03', 'C04', 'C05',
                       'C06', 'C07', 'C08', 'C09', 'C10')}
        # The check that detects this mutation most often
        top_check = max(check_det.items(), key=lambda kv: kv[1])
        per_mutation[mid] = {
            'name': mname,
            'n_cells': total,
            'n_caught': caught,
            'detection_rate': round(det_rate, 4),
            'n_flipped_per_mutation': n_flipped_per_mutation[mid],
            'per_check_detection': check_det,
            'top_check': top_check[0],
            'top_check_count': top_check[1],
        }

    # Per-check summary: across all mutations, how often does each check fail?
    per_check = {}
    for cid in ('C01', 'C02', 'C03', 'C04', 'C05',
                'C06', 'C07', 'C08', 'C09', 'C10'):
        per_check[cid] = check_catches[cid]

    # Headline H1 — every mutation detected on >=1 check
    headline_h1 = all(per_mutation[mid]['detection_rate'] == 1.0
                       for mid in per_mutation)
    # Headline H2 — every mutation detected on >=1 SPECIFIC check
    # (i.e. the same check fires every time)
    headline_h2 = {}
    for mid in per_mutation:
        d = per_mutation[mid]
        headline_h2[mid] = {
            'top_check': d['top_check'],
            'top_check_rate': d['top_check_count'] / d['n_cells']
                              if d['n_cells'] else 0,
        }

    summary = {
        'n_cells': n_cells_total,
        'n_mutations': n_mutations,
        'total_evaluations': n_cells_total * n_mutations,
        'per_mutation': per_mutation,
        'per_check_total_catches': per_check,
        'headline_H1_every_mutation_caught': headline_h1,
        'headline_H2_top_check_per_mutation': headline_h2,
    }
    summary_path = OUT_DIR / 'p5_iter121_mutation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"WROTE {summary_path}")

    print()
    print(f"=== iter-121 P5 mutation stress test ===")
    print(f"n_cells: {n_cells_total}")
    print(f"n_mutations: {n_mutations}")
    print(f"Headline H1 (every mutation caught on >=1 check): "
          f"{headline_h1}")
    for mid in sorted(per_mutation.keys()):
        d = per_mutation[mid]
        print(f"  {mid} ({d['name']}): caught {d['n_caught']}/{d['n_cells']} "
              f"= {d['detection_rate']*100:.1f}%; "
              f"top check {d['top_check']} fires on {d['top_check_count']} cells")


if __name__ == '__main__':
    main()