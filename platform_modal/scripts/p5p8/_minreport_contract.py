"""Canonical typed MIN-REPORT declarations and evidence checks (v2).

Reported, applicable and verified are separate. False and 0 are valid reports;
nonempty prose is never proof of disjointness or performed checks.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

VERSION = "minreport-review02-v2.1"
ITEM_ALIASES = {'reference_kl': 'ref_policy_kl', 'sampler_backend': 'sampler_backend_precision',
                'telemetry': 'per_step_zvf_path', 'decontamination': 'decontamination_notes'}
BOOL_FIELDS = {'disjoint_from_reward_env', 'performed', 'parser_robustness_probe', 'per_step_zvf',
               'per_step_gu', 'dynamic_sampling', 'disjointness', 'decontam_check', 'parser_probe'}
NUMBER_FIELDS = {'clip_eps_low', 'clip_eps_high', 'clip_low', 'clip_high', 'kl_beta', 'kl_coef',
                 'temperature', 'top_p'}
INTEGER_FIELDS = {'initial_g', 'G_value', 'split_size'}
PRECISIONS = {'bf16', 'bfloat16', 'fp16', 'float16', 'fp32', 'float32', 'fp8', 'int8', 'int4'}


def field_report(value):
    """Wrap a typed literal; applicability requires an explicit reason."""
    if isinstance(value, dict) and value.get('applicable') is False:
        reason = value.get('reason')
        return dict(reported=isinstance(reason, str) and bool(reason.strip()), applicable=False,
                    verified=False, value=None, status='not_applicable' if reason else 'invalid')
    if isinstance(value, dict) and 'value' in value:
        value = value['value']
    reported = value is not None and not (isinstance(value, str) and value.strip().lower() in {'', 'unknown', 'unspecified', 'tbd', 'null'})
    return dict(reported=reported, applicable=True, verified=False, value=value,
                status='reported' if reported else 'missing')


def valid_literal(subfield, value):
    report = field_report(value)
    if not report['reported'] or report['applicable'] is False:
        return report['reported']
    value = report['value']
    if subfield in BOOL_FIELDS:
        return isinstance(value, bool)
    if subfield in NUMBER_FIELDS:
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if subfield in INTEGER_FIELDS:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if subfield == 'precision':
        return value in PRECISIONS if isinstance(value, str) else False
    return isinstance(value, str) and bool(value.strip())


def literal_field(manifest, item, subfield):
    """Canonical min_report.<item> first, then explicit legacy object fields."""
    mr = manifest.get('min_report', {})
    candidates = []
    for root in (mr, manifest):
        if isinstance(root, dict):
            candidates += [root.get(item), root.get(ITEM_ALIASES.get(item, item))]
    for block in candidates:
        if isinstance(block, dict):
            if subfield in block:
                return True, block[subfield]
            if block.get('applicable') is False:
                return True, block
    if subfield in manifest:
        return True, manifest[subfield]
    return False, None


def verify_split(heldout_ids, reward_ids):
    if not isinstance(heldout_ids, list) or not isinstance(reward_ids, list) or not heldout_ids or not reward_ids:
        return dict(verified=False, disjoint=None, status='insufficient_data')
    if any(not isinstance(x, (str, int)) or isinstance(x, bool) for x in heldout_ids+reward_ids):
        return dict(verified=False, disjoint=None, status='invalid')
    a, b = set(heldout_ids), set(reward_ids)
    return dict(verified=True, disjoint=not bool(a & b), overlap_count=len(a & b), status='verified')


def verify_trajectory(filename, root):
    result = dict(reported=isinstance(filename, str) and bool(filename.strip()), applicable=True,
                  verified=False, status='missing', per_step_zvf=None, per_step_gu=None)
    if not result['reported']:
        return result
    path = Path(filename)
    if not path.is_absolute():
        path = Path(root)/path
    if not path.is_file():
        return result
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows = data.get('step_log', data.get('trajectory', data.get('steps')))
        else:
            rows = data
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError('expected nonempty list of step records')
        steps = [row['step'] for row in rows]
        if any(not isinstance(i, (int, float)) or isinstance(i, bool) or not math.isfinite(i) for i in steps) or len(set(steps)) != len(steps) or steps != sorted(steps):
            raise ValueError('unique increasing numeric steps required')
        for name, field in [('per_step_zvf', 'zvf'), ('per_step_gu', 'gu')]:
            valid = all(isinstance(r.get(field), (float, int)) and not isinstance(r.get(field), bool)
                        and math.isfinite(r[field]) and 0 <= r[field] <= 1 for r in rows)
            result[name] = valid
        if not result['per_step_zvf']:
            raise ValueError('finite ZVF in [0,1] required at each step')
        result.update(verified=True, status='verified', n_steps=len(rows))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status='invalid', reason=str(exc))
    return result


def load_inventory_manifests(directory, inventory=None):
    """Eligibility comes from the independent cell inventory, never audited keys.

    With no inventory, every JSON file is eligible; absent files cannot be
    inferred, and this limitation is explicit in each record.
    """
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob('*.json')):
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise ValueError('manifest must be an object')
            d = dict(d)
            d['_manifest_status'] = 'observed'
        except (OSError, ValueError) as exc:
            d = dict(_manifest_status='invalid', _manifest_error=str(exc))
        d.setdefault('cell_id', path.stem)
        d['_path'] = str(path)
        rows.append(d)
    if inventory is not None and Path(inventory).is_file():
        with Path(inventory).open() as f:
            expected = list(csv.DictReader(f, delimiter='\t'))
        by_id = {r['cell_id']: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError('duplicate manifest cell identity')
        return [dict(by_id.get(r['cell_id'], dict(cell_id=r['cell_id'], _manifest_status='missing')),
                     _inventory_status='expected') for r in expected]
    for r in rows:
        r['_inventory_status'] = 'file_inventory_only_missing_files_unknown'
    return rows


def inventory_counts(records):
    counts = dict(expected=len(records), observed=0, analyzed=0, missing=0, invalid=0,
                  contradictory=0, not_applicable=0)
    for record in records:
        status = record.get('_manifest_status', 'observed')
        counts[status] += 1
        counts['analyzed'] += status == 'observed'
    return counts


def verify_manifest_evidence(manifest, root):
    """Check available artifacts separately from claims of reporting completeness."""
    found, source = literal_field(manifest, 'telemetry', 'source')
    source = field_report(source)['value'] if found else manifest.get('per_step_zvf_path')
    trajectory = verify_trajectory(source, root)
    block = manifest.get('min_report', {}).get('heldout_split', {}) if isinstance(manifest.get('min_report'), dict) else manifest.get('heldout_split', {})
    block = block if isinstance(block, dict) else {}
    split = verify_split(block.get('heldout_example_ids'), block.get('reward_example_ids'))
    found, declaration = literal_field(manifest, 'heldout_split', 'disjoint_from_reward_env')
    declared = field_report(declaration)['value'] if found else None
    split['contradictory'] = bool(split['verified'] and isinstance(declared, bool) and declared != split['disjoint'])
    return dict(trajectory=trajectory, split=split)
