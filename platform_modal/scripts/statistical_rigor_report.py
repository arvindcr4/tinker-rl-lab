#!/usr/bin/env python3
"""Versioned endpoint/replication audit, not a certificate of statistical power.

Requires genuine per-seed observations. Summary rows and fixed token budgets
are never treated as repeated seeds. Paired differences use an explicit
(experiment, configuration, seed, endpoint) join. Archived outputs unchanged.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT
from _review02_contracts import STATISTIC_VERSION

RESULTS_DIR = REPO_ROOT / 'platform_hybrid/experiments/results'
OUT_TSV = RESULTS_DIR / 'statistical_rigor_report.tsv'
MIN_SEEDS = 5
MIN_STEPS = 100


def cohens_d(a, b):
    """Independent-group sample-pooled d; not used for paired comparisons."""
    if min(len(a), len(b)) < 2:
        return (float('nan'),)*3
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1-1)*np.var(a, ddof=1)+(n2-1)*np.var(b, ddof=1))/(n1+n2-2))
    if not pooled:
        return (float('nan'),)*3
    d = (np.mean(a)-np.mean(b))/pooled
    se = math.sqrt((n1+n2)/(n1*n2)+d*d/(2*(n1+n2)))
    return float(d), float(d-1.96*se), float(d+1.96*se)


def endpoint_records(rows, arm_key, metric_keys, default_experiment):
    """Read numeric endpoints and reject ambiguous duplicates; never row-order last."""
    records = {}
    excluded = []
    for number, row in enumerate(rows, 2):
        try:
            arm = str(row[arm_key])
            seed = int(row['seed'])
            step = float(row.get('step', row.get('training_step', row.get('n_steps', ''))))
            metric = next(row[k] for k in metric_keys if k in row and row[k] != '')
            accuracy = float(metric)
            if not math.isfinite(step) or step < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError('invalid step or accuracy')
            if int(row.get('n_seeds', 1)) != 1:
                raise ValueError('aggregate rows are not seed observations')
            experiment = row.get('experiment', default_experiment)
            configuration = (row.get('configuration_id', ''), row.get('model', ''),
                             row.get('task', ''), row.get('budget_tokens', ''))
            key = (experiment, configuration, arm, seed, step)
            if key in records:
                raise ValueError(f'duplicate endpoint identity {key}')
            records[key] = dict(experiment=experiment, configuration=configuration,
                                arm=arm, seed=seed, step=step, accuracy=accuracy,
                                init_id=row.get('initialization_id', row.get('initial_checkpoint', '')),
                                source_row=number)
        except (KeyError, ValueError, StopIteration) as exc:
            if 'duplicate endpoint' in str(exc):
                raise
            excluded.append(dict(row=number, reason=str(exc)))
    final = {}
    for key, record in records.items():
        identity = key[:-1]
        if identity not in final or record['step'] > final[identity]['step']:
            final[identity] = record
    return list(final.values()), excluded


def _read(path):
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def _load_variance_mitigation_last_step():
    return endpoint_records(_read(RESULTS_DIR/'variance_mitigation.tsv'), 'method',
                            ['heldout_acc'], 'variance_mitigation')


def _load_base_instruct():
    return endpoint_records(_read(RESULTS_DIR/'base_instruct_paired.tsv'), 'init',
                            ['accuracy', 'heldout_acc'], 'base_instruct')


def _load_group_size():
    rows = _read(RESULTS_DIR/'group_size_token_normalized.tsv')
    canonical = [dict(r, G=r.get('G', r.get('group_size', ''))) for r in rows]
    return endpoint_records(canonical, 'G', ['heldout_acc', 'accuracy', 'heldout_acc_mean'], 'group_size')


def paired_comparison(records, arm_a, arm_b, comparison_id):
    def index(arm):
        return {(r['experiment'], r['configuration'], r['seed'], r['step']): r
                for r in records if r['arm'] == arm}
    a, b = index(arm_a), index(arm_b)
    keys = sorted(set(a) & set(b))
    differences = np.array([a[k]['accuracy']-b[k]['accuracy'] for k in keys], float)
    n = len(keys)
    point = float(differences.mean()) if n else None
    low = high = standardized = None
    if n >= 2:
        sd = float(differences.std(ddof=1))
        half = float(stats.t.ppf(.975, n-1))*sd/math.sqrt(n)
        low, high = point-half, point+half
        standardized = point/sd if sd else None
    matched_init = ('yes' if keys and all(a[k]['init_id'] and a[k]['init_id'] == b[k]['init_id'] for k in keys)
                    else 'no' if keys and all(a[k]['init_id'] and b[k]['init_id'] for k in keys) else 'unknown')
    enough_steps = bool(keys) and all(k[-1] >= MIN_STEPS for k in keys)
    return dict(comparison_id=comparison_id, claim=f'{arm_a} minus {arm_b} at matched numeric endpoint',
                n_seeds_A=len(a), n_seeds_B=len(b), n_pairs=n,
                unmatched_A=len(set(a)-set(b)), unmatched_B=len(set(b)-set(a)),
                matched_init=matched_init, meets_replication_minimum=n >= MIN_SEEDS,
                meets_step_minimum=enough_steps, power_status='not_assessed_no_prespecified_effect_or_margin',
                estimand='mean paired heldout accuracy difference', estimate=point, ci_low=low, ci_high=high,
                unit='accuracy', method='paired_seed_t_interval' if n >= 2 else 'unavailable',
                standardized_effect_dz=standardized,
                standardized_definition='mean paired difference / sample SD of paired differences',
                uncertainty_source='training seeds conditional on fixed evaluation examples',
                multiplicity='descriptive; no confirmatory family specified',
                statistic_version=STATISTIC_VERSION,
                recommendation='insufficient_data' if n < 2 else 'report_descriptively_with_design_limitations')


def generate_report():
    report = []
    datasets = [('variance_mitigation', _load_variance_mitigation_last_step()),
                ('base_instruct', _load_base_instruct()), ('group_size', _load_group_size())]
    for name, (records, exclusions) in datasets:
        # Keep fixed budgets/configurations separate rather than treating them as seeds.
        configurations = sorted({(r['experiment'], r['configuration']) for r in records})
        if not records:
            report.append(dict(comparison_id=name, recommendation='insufficient_data',
                               exclusion_count=len(exclusions), exclusions=str(exclusions),
                               power_status='not_assessed', statistic_version=STATISTIC_VERSION))
        for exp, config in configurations:
            subset = [r for r in records if r['experiment'] == exp and r['configuration'] == config]
            arms = sorted({r['arm'] for r in subset})
            pairs = [(a, 'grpo') for a in arms if a != 'grpo'] if name == 'variance_mitigation' and 'grpo' in arms else itertools.combinations(arms, 2)
            for a, b in pairs:
                row = paired_comparison(subset, a, b, f'{name}:{config}:{a}_vs_{b}')
                row.update(exclusion_count=len(exclusions), exclusions=str(exclusions))
                report.append(row)
    return report


def write_report(rows, path=OUT_TSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(OUT_TSV))
    args = parser.parse_args(argv)
    rows = generate_report()
    write_report(rows, Path(args.out))
    print(f'{len(rows)} descriptive comparisons; power not assessed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
