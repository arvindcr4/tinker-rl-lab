#!/usr/bin/env python3
"""
P6 registry field-level coverage audit (iter 78, vein (b)).

Reads every registry/entries/*.json, classifies each entry by record_type and
framework, and emits:
  - registry_field_coverage_matrix.tsv   (entries x fields; cell = 1/0/None)
  - registry_field_coverage_summary.json (per-record-type and per-framework
                                         counts of populated optional blocks,
                                         plus framework x method matrix)
  - registry_field_coverage_gaps.tsv     (entries with >=2 optional blocks
                                         entirely null, sorted by #null blocks)
  - registry_method_coverage_matrix.tsv  (framework x method registration
                                         matrix; cells = entries-present count)
  - registry_method_coverage_summary.json (machine-readable summary)

Stdlib only (json, glob, os, collections, statistics).
"""
import json
import glob
import os
import sys
from collections import defaultdict, Counter
import csv
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTRIES = os.path.join(ROOT, 'registry', 'entries')
RESULTS = os.path.join(ROOT, 'experiments', 'results', 'p5p8')
os.makedirs(RESULTS, exist_ok=True)

# Optional blocks we audit per record_type. The schema marks these as either
# nullable or object/array (so the absence is informative, not just a stylistic
# choice). Required fields are NOT audited (a missing required field is a
# schema violation, not a coverage gap).
OPTIONAL_FIELDS = {
    'variant_delta': [
        'measured',
        'expected_effects',
        'claim_validation',
        'measured_yield_residual',
        'controller_predicted_savings_per_rollout',
    ],
    'stack': [
        'outcomes',
        'min_report',
        'variant_deltas_applied',
        'notes',
    ],
}

# Which ledger file to cross-reference for the framework x method audit
LEDGER_TSV = os.path.join(ROOT, 'experiments', 'results', 'experiment_ledger.tsv')

# Methods whose variants should be present if they appear in the ledger
KNOWN_GRPO_FAMILY = {
    'grpo', 'GRPO', 'TRL-GRPO', 'ppo_reinforce', 'reinforce', 'PPO',
    'per-group regression; continuous reward; population-standardized advantage',
}

def is_populated(value):
    """Heuristic: a block is 'populated' if it has at least one non-null sub-value."""
    if value is None:
        return False
    if isinstance(value, (list, str)) and len(value) == 0:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False
    if isinstance(value, dict):
        # If every value is None, treat as null
        nonnull = [v for v in value.values() if v is not None]
        if not nonnull:
            return False
    return True

def audit_entry(entry_path):
    """Return a dict with entry_id, record_type, framework, and per-field pop flags."""
    e = json.load(open(entry_path))
    rt = e.get('record_type', 'unknown')
    eid = e.get('id', os.path.basename(entry_path).replace('.json', ''))

    # Framework derivation: for stack records use framework.name; for variants
    # the framework is implicit (the variant is framework-agnostic).
    if rt == 'stack':
        fw = e.get('framework', {}).get('name', 'unknown')
    else:
        # Variants: tag with their base algorithm and provenance where possible
        fw = e.get('base', 'unknown') + '_variant'
    label = e.get('label_claimed', e.get('name', ''))

    fields = OPTIONAL_FIELDS.get(rt, [])
    field_pop = {}
    for f in fields:
        v = e.get(f, None)
        field_pop[f] = 1 if is_populated(v) else 0

    return {
        'id': eid,
        'record_type': rt,
        'framework': fw,
        'label': label,
        'field_pop': field_pop,
        'citation_arxiv': (e.get('citation') or {}).get('arxiv', ''),
        'has_measured': field_pop.get('measured', 0) == 1,
        'n_expected': len(e.get('expected_effects') or []),
        'n_validated': len(e.get('claim_validation') or []),
    }

def load_ledger():
    """Return list of dicts with one per ledger row."""
    if not os.path.exists(LEDGER_TSV):
        return []
    with open(LEDGER_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)

def framework_method_matrix(entries_data, ledger_rows):
    """Compute (framework, method) registration matrix from entries and ledger.
    Returns: (all_frameworks, registry_methods_normalized, fw_method_present, ledger_methods_normalized)
    """
    def norm(s):
        return s.strip().lower().replace('-', '').replace('_', '').replace(' ', '')

    all_frameworks = set()
    registry_methods = set()  # methods that appear in registry entries
    ledger_methods = set()    # methods that appear in the experiment ledger
    fw_method_present = defaultdict(set)

    for ed in entries_data:
        n = norm(ed['label'])
        registry_methods.add(n)
        if ed['record_type'] == 'stack':
            all_frameworks.add(ed['framework'])
            fw_method_present[ed['framework']].add(n)

    for row in ledger_rows:
        algo = row.get('algo', 'UNKNOWN')
        source = row.get('source', 'unknown')
        n = norm(algo)
        ledger_methods.add(n)
        if source:
            all_frameworks.add(source)
            fw_method_present[source].add(n)

    return all_frameworks, registry_methods, fw_method_present, ledger_methods

def main():
    entry_paths = sorted(glob.glob(os.path.join(ENTRIES, '*.json')))
    entries_data = [audit_entry(p) for p in entry_paths]

    # ----- coverage matrix -----
    matrix_rows = []
    fields = sorted({f for ed in entries_data for f in ed['field_pop']})
    for ed in entries_data:
        row = {
            'id': ed['id'],
            'record_type': ed['record_type'],
            'framework': ed['framework'],
            'label': ed['label'],
        }
        for f in fields:
            row[f] = ed['field_pop'].get(f, '')
        row['n_populated'] = sum(ed['field_pop'].values())
        row['n_total'] = sum(1 for f in OPTIONAL_FIELDS[ed['record_type']])
        row['coverage_pct'] = round(100.0 * row['n_populated'] / max(row['n_total'], 1), 1)
        row['citation_arxiv'] = ed['citation_arxiv']
        matrix_rows.append(row)

    fields_ordered = ['id', 'record_type', 'framework', 'label'] + fields + [
        'n_populated', 'n_total', 'coverage_pct', 'citation_arxiv']
    with open(os.path.join(RESULTS, 'registry_field_coverage_matrix.tsv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=fields_ordered, delimiter='\t')
        w.writeheader()
        for r in matrix_rows:
            w.writerow({k: r.get(k, '') for k in fields_ordered})

    # ----- gaps -----
    gaps = []
    for ed in entries_data:
        null_fields = [f for f, v in ed['field_pop'].items() if v == 0]
        if len(null_fields) >= 2:
            gaps.append({
                'id': ed['id'],
                'record_type': ed['record_type'],
                'framework': ed['framework'],
                'n_populated': sum(ed['field_pop'].values()),
                'n_total': len(ed['field_pop']),
                'null_fields': ';'.join(null_fields),
                'n_null': len(null_fields),
                'citation_arxiv': ed['citation_arxiv'],
            })
    gaps.sort(key=lambda r: -r['n_null'])
    gap_fields = ['id', 'record_type', 'framework', 'n_populated', 'n_total',
                  'null_fields', 'n_null', 'citation_arxiv']
    with open(os.path.join(RESULTS, 'registry_field_coverage_gaps.tsv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=gap_fields, delimiter='\t')
        w.writeheader()
        for r in gaps:
            w.writerow(r)

    # ----- framework x method matrix -----
    ledger_rows = load_ledger()
    ledger_methods_raw = set()
    for r in ledger_rows:
        ledger_methods_raw.add(r.get('algo', 'UNKNOWN'))

    fws_in_registry, registry_methods_norm, fw_method_in_registry, ledger_methods_norm = framework_method_matrix(entries_data, ledger_rows)
    methods_in_ledger_norm = ledger_methods_norm  # alias for clarity

    # Compute ledger-derived framework -> method (split source string)
    fw_method_in_ledger = defaultdict(set)
    for r in ledger_rows:
        src = r.get('source', '')
        algo = r.get('algo', 'UNKNOWN')
        if src:
            fw_method_in_ledger[src].add(algo)

    # Combine
    all_fws = sorted(set(list(fw_method_in_registry.keys()) + list(fw_method_in_ledger.keys())))
    all_methods_norm = sorted(registry_methods_norm | methods_in_ledger_norm)
    matrix_rows2 = []
    for fw in all_fws:
        row = {'framework': fw}
        reg_methods = fw_method_in_registry.get(fw, set())
        led_methods = fw_method_in_ledger.get(fw, set())
        for m in all_methods_norm:
            row[m] = '+' if m in reg_methods else ('' if m not in led_methods else '*')
        row['n_in_registry'] = len(reg_methods)
        row['n_in_ledger'] = len(led_methods)
        row['n_overlap'] = len(reg_methods & led_methods)
        row['n_ledger_only'] = len(led_methods - reg_methods)
        matrix_rows2.append(row)

    fw_method_fields = ['framework'] + all_methods_norm + [
        'n_in_registry', 'n_in_ledger', 'n_overlap', 'n_ledger_only']
    with open(os.path.join(RESULTS, 'registry_method_coverage_matrix.tsv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=fw_method_fields, delimiter='\t')
        w.writeheader()
        for r in matrix_rows2:
            w.writerow(r)

    # ----- summary JSON -----
    # Per-record-type aggregate
    rt_counts = Counter(ed['record_type'] for ed in entries_data)
    rt_avg_pop = {}
    for rt in OPTIONAL_FIELDS:
        ed = [e for e in entries_data if e['record_type'] == rt]
        if not ed:
            rt_avg_pop[rt] = None
            continue
        total_fields = sum(len(OPTIONAL_FIELDS[rt]) for _ in ed)
        total_pop = sum(sum(e['field_pop'].values()) for e in ed)
        rt_avg_pop[rt] = round(100.0 * total_pop / max(total_fields, 1), 1)

    # Per-framework aggregate
    fw_counts = Counter(ed['framework'] for ed in entries_data if ed['record_type'] == 'stack')

    # Headline numbers
    n_entries = len(entries_data)
    n_variants = sum(1 for e in entries_data if e['record_type'] == 'variant_delta')
    n_stacks = sum(1 for e in entries_data if e['record_type'] == 'stack')
    n_measured = sum(1 for e in entries_data if e['has_measured'])
    n_with_expected = sum(1 for e in entries_data if e['n_expected'] > 0)
    n_with_validated = sum(1 for e in entries_data if e['n_validated'] > 0)

    # Framework-method coverage stats (using normalized names)
    methods_only_in_ledger = sorted(methods_in_ledger_norm - registry_methods_norm)
    methods_in_both = sorted(registry_methods_norm & methods_in_ledger_norm)
    methods_only_in_registry = sorted(registry_methods_norm - methods_in_ledger_norm)

    summary = {
        'audit_date': datetime.date.today().isoformat(),
        'iter': 78,
        'n_entries': n_entries,
        'n_stacks': n_stacks,
        'n_variants': n_variants,
        'n_variants_measured': n_measured,
        'n_variants_with_expected_effects': n_with_expected,
        'n_variants_with_claim_validation': n_with_validated,
        'record_type_counts': dict(rt_counts),
        'record_type_avg_field_population_pct': rt_avg_pop,
        'framework_counts_stack': dict(fw_counts),
        'methods_only_in_ledger_normalized': methods_only_in_ledger,
        'methods_only_in_registry_normalized': methods_only_in_registry,
        'methods_in_both_normalized': methods_in_both,
        'n_methods_in_ledger_raw': len(ledger_methods_raw),
        'n_methods_in_registry_normalized': len(registry_methods_norm),
        'n_methods_overlap_normalized': len(methods_in_both),
        'all_frameworks': sorted(fws_in_registry),
        'all_methods_in_ledger_raw': sorted(ledger_methods_raw),
        'all_methods_in_registry_normalized': sorted(registry_methods_norm),
    }

    with open(os.path.join(RESULTS, 'registry_field_coverage_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RESULTS, 'registry_method_coverage_summary.json'), 'w') as f:
        json.dump({
            'audit_date': datetime.date.today().isoformat(),
            'iter': 78,
            'frameworks': all_fws,
            'methods_in_ledger_raw': sorted(ledger_methods_raw),
            'methods_in_registry_normalized': sorted(registry_methods_norm),
            'methods_in_both_normalized': methods_in_both,
            'methods_only_in_ledger_normalized': methods_only_in_ledger,
            'methods_only_in_registry_normalized': methods_only_in_registry,
            'matrix_rows': matrix_rows2,
            'matrix_methods': all_methods_norm,
        }, f, indent=2)

    # Console summary
    print(f'Iter 78 P6 registry field-level coverage audit complete.')
    print(f'  Entries: {n_entries} ({n_stacks} stack + {n_variants} variant)')
    print(f'  Variants with measured evidence: {n_measured}/{n_variants}')
    print(f'  Variants with expected_effects: {n_with_expected}/{n_variants}')
    print(f'  Variants with claim_validation: {n_with_validated}/{n_variants}')
    print(f'  Methods in ledger (raw): {sorted(ledger_methods_raw)}')
    print(f'  Methods in registry (normalized): {sorted(registry_methods_norm)}')
    print(f'  Methods overlap (normalized): {sorted(methods_in_both)}')
    if methods_only_in_ledger:
        print(f'  Methods in LEDGER but NOT in registry: {methods_only_in_ledger}')
    if methods_only_in_registry:
        print(f'  Methods in REGISTRY but NOT in ledger: {methods_only_in_registry}')
    print(f'  Avg optional-block coverage: {rt_avg_pop}')
    print(f'  Output written to {RESULTS}')

if __name__ == '__main__':
    main()