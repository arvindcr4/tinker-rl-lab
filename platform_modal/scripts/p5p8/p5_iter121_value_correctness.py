#!/usr/bin/env python3
# P5 iter-121 — MIN-REPORT v2.2 value-correctness audit (fresh vein)
#
# Tests whether the VALUE of each explicit-json-key item in the live
# manifest JSON body is *semantically consistent* with the cells.tsv
# ground-truth and the cell_id filename. Iter-97 audited schema-level
# presence; iter-105 audited per-value-class presence; iter-113 audited
# declared-vs-emitted-vs-derivable content; iter-117 audited per-location
# encoding mode. None of them audited whether the *value itself* agrees
# with the canonical ground-truth.
#
# Vein (fresh, not in 135 prior rows): MIN-REPORT v2.2 VALUE-CORRECTNESS
# audit. For each of the 7 explicit-json-key items + 2 absent-from-JSON
# items (model_family, temperature) derivable from filename+cells.tsv,
# check the value-consistency with the canonical ground-truth in cells.tsv.
# A manifest that passes presence audits but has wrong values is a WORSE
# false-positive than a manifest that declares the item absent.
"""
P5 MIN-REPORT v2.2 value-correctness audit (iter 121).

Loads the live n=98 mega corpus (manifests/ + cells.tsv), and for each
manifest runs 10 value-consistency checks (one per MIN-REPORT v2.2 item
that has ground-truth in cells.tsv or filename):

  C01 cell_id_json == filename_basename (no .json)
  C02 model_family_filename ∈ {Qwen, meta-llama} canonical names
  C03 task_slice_filename == cells.tsv task_slice
  C04 G_filename ∈ {2,4,8,16,32} matches cells.tsv G
  C05 temperature_filename ∈ {0.6, 1.0} matches cells.tsv temperature
  C06 seed_filename ∈ {0, 1, ...} matches cells.tsv seed
  C07 heldout_split_json == cells.tsv task_slice
  C08 group_size_schedule_json contains the cells.tsv G value
  C09 decontamination_notes_json contains task_slice prefix
  C10 per_step_zvf_path_json exists on disk

Per-cell aggregate:
  total_checks, passed_checks, score = passed_checks / total_checks
  tier ∈ {perfect (10/10), high (>=8), mid (>=6), low (<6)}

Per-item aggregate across 98 cells:
  pass_count, fail_count, fail_rate

Outputs:
  - platform_hybrid/experiments/results/p5p8/p5_iter121_value_correctness.tsv (per-cell)
  - platform_hybrid/experiments/results/p5p8/p5_iter121_value_correctness_per_item.tsv
  - platform_hybrid/experiments/results/p5p8/p5_iter121_summary.json
  - docs/p5p8_improvements/136_p5_value_correctness.md
"""
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict

WORK = Path("/home/claude/tinker-rl-lab-minimax")
MEGA = WORK / "platform_hybrid/experiments/results/mega_20260704"
MANIFEST_DIR = MEGA / "manifests"
CELLS_TSV = MEGA / "cells.tsv"
OUT_DIR = WORK / "platform_hybrid/experiments/results/p5p8"

# Canonical valid sentinels per MIN-REPORT v2.2 schema
LOSS_FORM_VALID = {"n/a-sampling", "n/a-pretrain", "policy-only",
                   "kl-augmented", "policy+value", "policy+kl"}
SAMPLER_VALID = {"tinker-closed", "vllm-fp32", "vllm-bf16", "hf-bf16",
                 "hf-fp32", "openai-api"}
KL_VALID_SENTINELS = {"n/a", "n/a-no-ref", "n/a-sampling"}


def load_cells_tsv():
    """Load cells.tsv as {cell_id: row_dict}."""
    cells = {}
    with open(CELLS_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cells[row['cell_id']] = row
    return cells


def parse_filename(cell_id):
    """Parse the cell_id filename into canonical fields."""
    # Format: model_task_slice_G{g}_t{t}_s{seed}_{hash}
    # E.g. Qwen-Qwen3-5-4B_gsm8k_easy_G2_t0.6_s0_923b060d59
    pat = re.compile(
        r'^(?P<model>.+?)_(?P<task>[a-z0-9_]+?)_G(?P<G>\d+)_t(?P<temp>[\d.]+)_s(?P<seed>\d+)_(?P<hash>[0-9a-f]+)$'
    )
    m = pat.match(cell_id)
    if not m:
        return None
    return m.groupdict()


def check_consistency(cell_id, manifest, cells_row, file_info):
    """Run 10 value-consistency checks. Returns list of (check_id, name, passed, details)."""
    results = []

    # C01 cell_id JSON matches filename basename (no .json)
    fname = cell_id  # manifests are stored as <cell_id>.json
    passed = manifest.get('cell_id') == fname
    results.append(('C01', 'cell_id_json_eq_filename', passed,
                    f"json={manifest.get('cell_id')!r} file={fname!r}"))

    if file_info is None:
        # Can't run remaining checks; mark them as errored.
        for cid in ('C02', 'C03', 'C04', 'C05', 'C06'):
            results.append((cid, f'{cid}_skipped', False, 'filename_parse_failed'))
        for cid in ('C07', 'C08', 'C09', 'C10'):
            results.append((cid, f'{cid}_skipped', False, 'filename_parse_failed'))
        return results

    # C02 model_family filename token ∈ canonical set
    model_token = file_info['model']
    passed = model_token in ('meta-llama-Llama-3-2-3B', 'Qwen-Qwen3-5-4B')
    results.append(('C02', 'model_family_filename_canonical', passed,
                    f"model={model_token!r}"))

    # C03 task_slice filename matches cells.tsv task_slice
    passed = file_info['task'] == cells_row['task_slice']
    results.append(('C03', 'task_slice_filename_eq_cells', passed,
                    f"file={file_info['task']!r} cells={cells_row['task_slice']!r}"))

    # C04 G filename matches cells.tsv G (numeric)
    passed = file_info['G'] == cells_row['G']
    results.append(('C04', 'G_filename_eq_cells', passed,
                    f"file={file_info['G']!r} cells={cells_row['G']!r}"))

    # C05 temperature filename matches cells.tsv temperature
    # Canonical encoding: filename stores 't1' (no trailing .0) for
    # temperature=1.0; 't0.6' for 0.6. cells.tsv stores '1.0' / '0.6'.
    # Normalize: strip trailing .0 from cells.tsv value.
    file_temp = file_info['temp']
    cell_temp = cells_row['temperature']
    if '.' in cell_temp and cell_temp.endswith('.0'):
        cell_temp_norm = cell_temp[:-2]
    else:
        cell_temp_norm = cell_temp
    passed = file_temp == cell_temp_norm
    results.append(('C05', 'temperature_filename_eq_cells', passed,
                    f"file={file_temp!r} cells={cell_temp!r} norm={cell_temp_norm!r}"))

    # C06 seed filename matches cells.tsv seed
    passed = file_info['seed'] == cells_row['seed']
    results.append(('C06', 'seed_filename_eq_cells', passed,
                    f"file={file_info['seed']!r} cells={cells_row['seed']!r}"))

    # C07 heldout_split JSON matches cells.tsv task_slice
    passed = manifest.get('heldout_split') == cells_row['task_slice']
    results.append(('C07', 'heldout_split_json_eq_task_slice', passed,
                    f"json={manifest.get('heldout_split')!r} cells={cells_row['task_slice']!r}"))

    # C08 group_size_schedule JSON contains the G value
    gs = manifest.get('group_size_schedule', '')
    G_val = cells_row['G']
    passed = f'G={G_val}' in gs or f'fixed-G={G_val}' in gs or f'G:{G_val}' in gs
    results.append(('C08', 'group_size_schedule_contains_G', passed,
                    f"schedule={gs!r} G={G_val!r}"))

    # C09 decontamination_notes JSON contains task_slice prefix
    # E.g. gsm8k_easy → 'gsm8k-train-slice', humaneval_subset → 'humaneval-...'
    dc = manifest.get('decontamination_notes', '')
    task = cells_row['task_slice']
    # task_slice's first token (e.g. 'gsm8k' from 'gsm8k_easy')
    task_prefix = task.split('_')[0]
    passed = task_prefix in dc
    results.append(('C09', 'decontamination_contains_task_prefix', passed,
                    f"notes={dc!r} task_prefix={task_prefix!r}"))

    # C10 per_step_zvf_path JSON file exists on disk
    zp = manifest.get('per_step_zvf_path', '')
    # Manifest stores an absolute path; we may have a different worktree root
    # Try both as-is and the relative path under our worktree
    candidates = []
    if zp:
        candidates.append(zp)
        # Try rewriting the prefix /home/claude/tinker-rl-lab/ → our worktree root
        if '/home/claude/tinker-rl-lab/' in zp:
            rel = zp.split('/home/claude/tinker-rl-lab/', 1)[1]
            candidates.append(str(WORK / rel))
    exists = any(os.path.exists(c) for c in candidates)
    results.append(('C10', 'per_step_zvf_path_exists_on_disk', exists,
f"raw_path={zp!r} resolved={exists}"))

    return results


def main():
    cells = load_cells_tsv()
    per_cell = []  # list of dicts for TSV
    per_item = defaultdict(lambda: {'pass_count': 0, 'fail_count': 0,
                                     'fail_examples': []})
    tier_counts = defaultdict(int)
    score_sum = 0.0
    n_cells = 0

    for manifest_path in sorted(MANIFEST_DIR.glob('*.json')):
        cell_id = manifest_path.stem  # filename without .json
        if cell_id not in cells:
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        cells_row = cells[cell_id]
        file_info = parse_filename(cell_id)

        results = check_consistency(cell_id, manifest, cells_row, file_info)

        passed_count = sum(1 for r in results if r[2])
        total = len(results)
        score = passed_count / total if total else 0.0
        if passed_count == total:
            tier = 'perfect'
        elif passed_count >= 8:
            tier = 'high'
        elif passed_count >= 6:
            tier = 'mid'
        else:
            tier = 'low'
        tier_counts[tier] += 1
        score_sum += score
        n_cells += 1

        # Per-cell row
        per_cell.append({
            'cell_id': cell_id,
            'passed': passed_count,
            'total': total,
            'score': round(score, 4),
            'tier': tier,
            'fail_check_ids': ','.join(r[0] for r in results if not r[2]),
        })

        # Per-item aggregation
        for cid, cname, passed, details in results:
            if passed:
                per_item[cid]['pass_count'] += 1
            else:
                per_item[cid]['fail_count'] += 1
                if len(per_item[cid]['fail_examples']) < 3:
                    per_item[cid]['fail_examples'].append({
                        'cell_id': cell_id, 'details': details
                    })

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # per-cell TSV
    tsv_path = OUT_DIR / 'p5_iter121_value_correctness.tsv'
    with open(tsv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cell_id', 'passed', 'total',
                                          'score', 'tier', 'fail_check_ids'],
                            delimiter='\t')
        w.writeheader()
        for row in per_cell:
            w.writerow(row)
    print(f"WROTE {tsv_path} ({len(per_cell)} rows)")

    # per-item TSV
    item_tsv = OUT_DIR / 'p5_iter121_value_correctness_per_item.tsv'
    with open(item_tsv, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['check_id', 'check_name', 'pass_count', 'fail_count',
                    'fail_rate', 'fail_examples_json'])
        for cid in sorted(per_item.keys()):
            d = per_item[cid]
            total = d['pass_count'] + d['fail_count']
            fr = d['fail_count'] / total if total else 0
            w.writerow([cid, '', d['pass_count'], d['fail_count'],
                        round(fr, 4), json.dumps(d['fail_examples'])])
    print(f"WROTE {item_tsv}")

    # summary JSON
    summary = {
        'n_cells': n_cells,
        'mean_score': round(score_sum / n_cells if n_cells else 0, 4),
        'tier_counts': dict(tier_counts),
        'tier_pct': {k: round(v / n_cells * 100, 1) if n_cells else 0
                      for k, v in tier_counts.items()},
        'per_item': {cid: {'pass': d['pass_count'], 'fail': d['fail_count']}
                     for cid, d in sorted(per_item.items())},
        'hypotheses': {
            'H1_value_correctness_strict': (
                f"{(sum(1 for r in per_cell if r['score'] == 1.0)) / n_cells * 100:.1f}%"
                f" of cells pass ALL 10 value-consistency checks"
                if n_cells else 'n/a'),
            'H2_perfect_rate_floor': (
                f"perfect-rate = {(tier_counts.get('perfect', 0)) / n_cells * 100:.1f}%"
                if n_cells else 'n/a'),
            'H3_silent_inconsistency': (
                'silently-wrong cells present (score < 1.0)' if n_cells > 0 else 'n/a'),
        },
    }
    summary_path = OUT_DIR / 'p5_iter121_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"WROTE {summary_path}")
    print()
    print(f"=== iter-121 P5 value-correctness audit ===")
    print(f"n_cells: {n_cells}")
    print(f"mean_score: {summary['mean_score']}")
    print(f"tier_counts: {summary['tier_counts']}")
    print(f"H1 strict-pass rate: {summary['hypotheses']['H1_value_correctness_strict']}")
    for cid in sorted(per_item.keys()):
        d = per_item[cid]
        print(f"  {cid}: pass={d['pass_count']} fail={d['fail_count']}")


if __name__ == '__main__':
    main()