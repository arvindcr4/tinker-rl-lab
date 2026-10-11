#!/usr/bin/env python3
"""Frozen, CPU-only exploratory provenance reanalysis. No network or training."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

MANIFEST_SHA256 = 'f2290f4cc8c45cdd28d90190bc9f426c615893ecde84a96ab91234f4d0d01bd9'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_inputs(input_dir):
    raw = (input_dir / 'manifest.json').read_bytes()
    if sha(raw) != MANIFEST_SHA256:
        raise ValueError('Frozen manifest hash mismatch')
    manifest = json.loads(raw)
    protocol = input_dir.parent / 'PROTOCOL.md'
    if not protocol.is_file() or sha(protocol.read_bytes()) != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol missing or hash mismatch')
    for source in manifest['sources']:
        name = source['retained_file']
        if Path(name).name != name or sha((input_dir / name).read_bytes()) != source['retained_sha256']:
            raise ValueError('Retained input hash mismatch: ' + name)
    return manifest


def read_tsv(path):
    return list(csv.DictReader(io.StringIO(''.join(line for line in path.read_text().splitlines(True)
                                                  if not line.startswith('#'))), delimiter='\t'))


def num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    good = [x for x in values if x is not None]
    return float(np.mean(good)) if good else None


def association(xs, ys, method='spearman'):
    if len(xs) < 3:
        return {'n': len(xs), 'coefficient': None, 'reason': 'fewer_than_three_pairs'}
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return {'n': len(xs), 'coefficient': None, 'reason': 'constant_axis'}
    value = spearmanr(xs, ys).statistic if method == 'spearman' else np.corrcoef(xs, ys)[0, 1]
    return {'n': len(xs), 'coefficient': float(value), 'method': method,
            'inference': 'descriptive_only_no_p_value_or_confidence_interval'}


def pool(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['experiment'], row['model'], int(row['group_size']))].append(row)
    cells = []
    for key, values in groups.items():
        cell = dict(zip(('experiment', 'model', 'group_size'), key))
        cell.update({field: mean([num(r[field]) for r in values])
                     for field in ('mean_zvf', 'last10_avg', 'peak', 'mean_reward')})
        cell['source_row_ids'] = [r['source_row_id'] for r in values]
        cell['source_row_count'] = len(values)
        cells.append(cell)
    return cells


def collapse(cell):
    last, peak = cell['last10_avg'], cell['peak']
    return float(last <= .001 or (peak > .7 and last < .35))


def provenance(experiment):
    return {
        'variance_mitigation': 'projection_backed_synthetic',
        'groupsize_zvf_sweep': 'measured_training_seed_aggregates',
        'drgrpo_vs_grpo': 'measured_training_algorithms_collapsed_by_historical_pool_key',
        'tinker_gsm8k_zvf': 'sampling_only_seed_rows_plus_summary',
        'samestack_ppo_grpo': 'measured_training_missing_summary_zvf',
        'scaling_law_three_phase': 'no_zvf_not_eligible',
    }.get(experiment, 'cross_tool_G1_mechanically_degenerate_zvf_raw_lineage_not_verified')


def historical(input_dir):
    rows = read_tsv(input_dir / 'zvf_summary.tsv')
    for i, row in enumerate(rows):
        row['source_row_id'] = i + 1  # data-row ordinal, excludes comments/header
    if len(rows) != 80:
        raise ValueError('Expected frozen 80-row historical summary')
    candidates = pool(rows)
    included, rejected = [], []
    for cell in candidates:
        cell['provenance'] = provenance(cell['experiment'])
        if cell['last10_avg'] is None or cell['peak'] is None:
            rejected.append({**cell, 'reason': 'missing_endpoint_or_peak'})
        elif cell['mean_zvf'] is None:
            rejected.append({**cell, 'reason': 'missing_zvf'})
        else:
            included.append(cell)
    if len(candidates) != 23 or len(included) != 17 or len(rejected) != 6:
        raise ValueError('Unexpected historical pooling/filtering cohort')
    xs = [c['mean_zvf'] for c in included]
    ys = [c['last10_avg'] for c in included]
    last_assoc = association(xs, ys)
    collapse_assoc = association(xs, [collapse(c) for c in included])
    published = read_tsv(input_dir / 'zvf_failure_correlation.tsv')
    for test, computed in [('mean_zvf vs last10_avg', last_assoc),
                           ('mean_zvf vs is_collapse', collapse_assoc)]:
        row = next(r for r in published if r['test'] == test and r['method'].startswith('Spearman'))
        if abs(float(row['rho']) - computed['coefficient']) > .00005:
            raise ValueError('Historical Spearman reproduction mismatch: ' + test)
    residual_rows = [r for r in rows if all(num(r[f]) is not None
                     for f in ('mean_zvf', 'last10_avg', 'mean_reward'))]
    residual_cells = pool(residual_rows)
    rx = np.array([c['mean_zvf'] for c in residual_cells])
    ry = np.array([c['last10_avg'] for c in residual_cells])
    reward = np.array([c['mean_reward'] for c in residual_cells])
    design = np.column_stack([np.ones(len(reward)), reward])
    residual = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    semi = association(rx.tolist(), residual.tolist(), 'pearson')
    reported = next(r for r in read_tsv(input_dir / 'zvf_iter26_residual.tsv')
                    if r['slice'] == 'pooled_all_per_experiment')
    if abs(float(reported['residual_r']) - semi['coefficient']) > .0005:
        raise ValueError('Historical semipartial reproduction mismatch')
    return {
        'source_rows': len(rows), 'source_rows_by_experiment': dict(Counter(r['experiment'] for r in rows)),
        'published_n': 23, 'candidate_cells': len(candidates), 'effective_pairs': len(included),
        'synthetic_source_rows': sum(r['experiment'] == 'variance_mitigation' for r in rows),
        'synthetic_effective_cells': sum(c['experiment'] == 'variance_mitigation' for c in included),
        'metric_identity': 'last10_avg is heterogeneous, not a common heldout accuracy; arithmetic uses training reward',
        'last10_spearman': last_assoc, 'collapse_spearman': collapse_assoc,
        'residual_source_rows': len(residual_rows), 'residual_effective_cells': len(residual_cells),
        'outcome_only_semipartial_pearson': semi,
        'residual_semantics': 'OLS residualizes last10_avg on mean_reward only; ZVF remains unresidualized',
        'included_cells': included, 'rejected_cells': rejected,
        'interpretation': 'reproduction of mixed-provenance historical statistic; not measured predictive evidence',
    }



def synthetic_proof(input_dir):
    """Independent transcription of the fixed source recipe, never execute input code."""
    methods = ('grpo', 'aero', 'cppo', 'ngrpo', 'scafgrpo', 'mcgrpo', 'gift', 'areal', 'es')
    targets = dict(zip(methods, [(.412, .418), (.448, .441), (.439, .433), (.431, .427),
                                (.460, .452), (.445, .438), (.451, .444), (.454, .446), (.438, .431)]))
    plateaus = dict(zip(methods, [.88, .72, .78, .75, None, .65, .60, .58, .25]))
    mids = dict(zip(methods, [45, 70, 62, 60, None, 78, 82, 80, 55]))
    quotas = dict(zip(methods, [3, 2, 2, 3, 1, 2, 1, 1, 0]))
    generated = {}
    for method_index, method in enumerate(methods):
        for seed in range(5):
            rng = random.Random(10000 * (method_index + 1) + seed)
            flagged = 0
            for step in range(100):
                if method == 'scafgrpo':
                    z = max(0., min(1., .22 + .002 * step + rng.gauss(0, .03)))
                else:
                    sigmoid = 1. / (1. + math.exp(-(step - mids[method]) / 8.))
                    z = max(0., min(1., plateaus[method] * sigmoid + rng.gauss(0, .03)))
                flagged = int(bool(flagged) or z > .9)
                reward = targets[method][0] * (1 - math.exp(-step / 30.)) + rng.gauss(0, .025)
                heldout = targets[method][1] * (1 - math.exp(-step / 40.)) + rng.gauss(0, .003)
                generated[(method, seed, step)] = (round(z, 4), round(reward, 4), round(heldout, 4),
                                                  flagged if seed < quotas[method] else 0)
    rows = read_tsv(input_dir / 'variance_mitigation.tsv')
    keys = []
    for row in rows:
        key = (row['method'], int(row['seed']), int(row['step']))
        observed = tuple(float(row[field]) for field in ('zvf', 'reward_mean', 'heldout_acc', 'collapse'))
        if key not in generated or observed != generated[key]:
            raise ValueError('Projection generator mismatch at ' + str(key))
        keys.append(key)
    if len(rows) != 5540 or set(keys) != set(generated):
        raise ValueError('Unexpected synthetic source cohort')
    counts = Counter(keys)
    return {'source_rows': len(rows), 'exact_generator_matches': len(rows),
            'unique_method_seed_step_keys': len(counts),
            'duplicated_keys': sum(n > 1 for n in counts.values()),
            'excess_duplicate_rows': len(rows) - len(counts),
            'conflicting_duplicate_rows': 0, 'method_seed_trajectories': 45,
            'interpretation': 'All source rows exactly reproduce deterministic dry-run projections; no training evidence'}


def measured(input_dir):
    studies = []
    specs = [('groupsize_zvf_sweep', (2, 4, 8, 16), (42, 123, 456), 'group_size'),
             ('drgrpo_vs_grpo', ('grpo', 'dr_grpo'), (42, 123, 456, 789, 1024), 'algo')]
    for name, arms, seeds, arm_field in specs:
        runs = json.loads((input_dir / (name + '_selected.json')).read_text())['runs']
        expected = {(arm, seed) for arm in arms for seed in seeds}
        actual = [(r[arm_field], r['seed']) for r in runs]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError('Missing, duplicate or unexpected measured run: ' + name)
        output_rows, strata = [], []
        for index, row in enumerate(runs):
            if row['source_run_index'] != index or row['experiment'] != name:
                raise ValueError('Source run identity mismatch')
            log = row['step_log']
            if [s['step'] for s in log] != list(range(len(log))) or len(log) != 40:
                raise ValueError('Unexpected measured trajectory')
            for field, value in [('mean_zvf', np.mean([s['zvf'] for s in log])),
                                 ('last10_avg', np.mean([s['mean_reward'] for s in log[-10:]]))]:
                if num(row[field]) is None or abs(row[field] - value) > 1e-12:
                    raise ValueError('Metric reconstruction failed: ' + field)
            if num(row['heldout_acc']) is None or not 0 <= row['heldout_acc'] <= 1:
                raise ValueError('Invalid recorded evaluation accuracy')
            output_rows.append({k: v for k, v in row.items() if k != 'step_log'})
        for arm in arms:
            subset = [r for r in runs if r[arm_field] == arm]
            strata.append({'stratum': {arm_field: arm}, 'seeds': [r['seed'] for r in subset],
                           **association([r['mean_zvf'] for r in subset], [r['heldout_acc'] for r in subset])})
        studies.append({'study': name, 'n_runs': len(runs), 'rows': output_rows,
                        'stratified_mean_zvf_vs_recorded_posttraining_evaluation': strata,
                        'metric_checks': 'mean_zvf and last10_avg recomputed from complete step logs',
                        'rejected_rows': [], 'baseline_heldout_gain': None,
                        'baseline_unavailable_reason': 'source contains no pretraining evaluation',
                        'evaluation': 'fresh 200 arithmetic draws; train disjointness not certified',
                        'scope': 'descriptive whole-trajectory association; no prediction or cross-study pooling'})
    return studies


def analyze(input_dir):
    manifest = verify_inputs(input_dir)
    return {'schema_version': 1, 'status': 'PASS', 'study_type': 'exploratory_existing_data_reanalysis_no_new_training',
            'manifest_sha256': MANIFEST_SHA256, 'protocol_sha256': manifest['protocol_sha256'],
            'analysis_script_sha256': sha(Path(__file__).read_bytes()),
            'historical_lineage': historical(input_dir), 'synthetic_source_proof': synthetic_proof(input_dir),
            'measured_studies': measured(input_dir),
            'source_mapping': manifest['sources'],
            'limitations': ['No independent predictive or early warning validation',
                            'No inference from heterogeneous historical pooling',
                            'Small measured strata and near-ceiling evaluation accuracy',
                            'Original source hashes identify local snapshots; remote verifies retained bytes only']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'effective_pairs': result['historical_lineage']['effective_pairs'],
                      'synthetic_cells': result['historical_lineage']['synthetic_effective_cells'],
                      'measured_runs': sum(s['n_runs'] for s in result['measured_studies'])}))


if __name__ == '__main__':
    main()
