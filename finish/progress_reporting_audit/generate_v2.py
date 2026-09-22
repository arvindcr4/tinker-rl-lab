"""Version 2: canonical scope labels and explicit local continuation closeouts.

Retains v1 and prior snapshots. No parent-state writes, cloud calls or model work.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import generate_v1 as v1

need = v1.need
RESOLVED_NAMES = {'sdab_eval':'SDAB','mlagentbench_eval':'MLAgentBench',
                  'banker_toolbench_eval':'BankerToolBench','binaryaudit_eval':'BinaryAudit',
                  'cybergym_eval':'CyberGym','appbench_eval':'AppBench','visual_agent_bench_eval':'VisualAgentBench'}


def resolve_labels(rows, mapping, replacements, registry):
    entries = mapping['lanes']
    need(len(entries) == 14 and len({x['historical_lane'] for x in entries}) == 14, 'canonical mapping needs 14 unique lanes')
    indexed = {x['historical_lane']:x for x in entries}
    for row in rows:
        item = indexed[row['lane']]
        original = row['original']['suite_id']
        public = row['portfolio']['suite_id']
        need(original == item['historical_suite'] and public == item['open_portfolio_suite'], 'canonical scope mapping mismatch')
        need(original in registry, 'canonical original suite missing')
        if original != public:
            need(replacements[original]['new_suite_id'] == public, 'override scope mismatch')
        elif 'RETAINED_SUITE' not in item['status']:
            raise v1.EvidenceError('same-suite scope requires retained declaration')
        if original in RESOLVED_NAMES: row['original']['name'] = RESOLVED_NAMES[original]
        if public in RESOLVED_NAMES: row['portfolio']['name'] = RESOLVED_NAMES[public]
        # Display names come from reviewed v1 suite-ID mapping, not ambiguous parent E-labels.
        row['display_label'] = row['lane'] + ' — original: ' + row['original']['name'] + '; portfolio: ' + row['portfolio']['name']
        row['scope_resolution'] = {'original_suite_id': original, 'portfolio_suite_id': public,
                                   'same_suite': original == public, 'public_result_applies_to_original': original == public,
                                   'original_private_or_full_scope_from_public_subset': None if original != public else 'requires matching run provenance',
                                   'mapping_is_execution_evidence': False}
    return rows


def audit_tau_snapshot(snapshot, raw_episodes, allocated_ids):
    started = snapshot['started_ids_in_original_order']
    graded = snapshot['graded']
    ungraded = snapshot['started_without_grade']
    never = snapshot['never_started_ids']
    need(snapshot['full_suite_denominator'] == 97 and snapshot['subset_denominator'] == len(allocated_ids), 'native10 denominator mismatch')
    need(len(set(allocated_ids)) == len(allocated_ids), 'duplicate allocation task')
    need(len(started) == len(set(started)) == snapshot['started_unique'], 'duplicate/incorrect started count')
    need(len(never) == len(set(never)) and len(ungraded) == len(set(ungraded)), 'duplicate task state')
    need(set(started).isdisjoint(never) and set(started) | set(never) == set(allocated_ids), 'allocation state partition mismatch')
    need(set(graded).isdisjoint(ungraded) and set(graded) | set(ungraded) == set(started), 'started/graded partition mismatch')
    need(snapshot['prior37_refs_hash_verified'] is True and snapshot['no_overlap_prior37'] is True
         and snapshot['no_duplicate_attempt_intents'] is True, 'prior-attempt reconciliation missing')
    need(snapshot['full_suite_score'] is None, 'partial native10 cannot have full-suite score')
    raw = {e['task_id']: e for e in raw_episodes}
    need(len(raw) == len(raw_episodes), 'duplicate raw native10 task')
    for task, reward in graded.items():
        need(task in raw and raw[task]['reward_info']['reward'] == reward, 'coverage snapshot/raw native grade mismatch')
    selected = [raw[t] for t in graded]
    m = v1.tau_run('E5-native-20260912-10', selected, allocated_ids)
    m.update({'label':'Native10 coverage_snapshot completed-only subset',
              'allocation_graded_coverage':v1.ratio(len(graded),len(allocated_ids)),
              'started_allocation_coverage':v1.ratio(len(started),len(allocated_ids)),
              'started_without_grade':ungraded, 'never_started_count':len(never),
              'snapshot_at_epoch':snapshot['at'], 'snapshot_raw_grade_count':len(raw),
              'failure_event_counts_not_task_outcomes':snapshot.get('known_failure_attempt_types',{}),
              'selection_caveat':'8 or other ungraded starts are neither automatic failures nor zeros. Prior37 are starts, not 37 grades. No cross-run score pooling.'})
    return m


def infra_zero(receipt, count_key, expected_status, label, denominator):
    need(receipt['status'] == expected_status and receipt[count_key] == 0 and receipt['model_calls'] == 0, 'infrastructure-only receipt contradicted')
    m = v1.measure(label,0,denominator,kind='infrastructure_only')
    m['accuracy_or_success'] = None
    return m


def build(reader):
    result = v1.build(reader)
    reader.raw('finish/progress_reporting_audit/generate_v2.py')
    result['schema_version'] = 'evidence-progress-report-v2'
    mapping = reader.load('outputs/e1_e14_results_2026-09-05/public_portfolio_mapping.json')
    overrides = reader.load('zvf-program/flagship/pavlovs_open_portfolio_overrides.json')
    registry = reader.load('zvf-program/flagship/pavlovs_domain_contract.json')['suite_registry']
    resolve_labels(result['rows'],mapping,overrides['replacements'],registry)
    # E5: selected authoritative continuation snapshot plus direct native records.
    candidates = reader.glob(v1.FINISH + '/e5_runtime/native10_execution/coverage_snapshot_*.json')
    need(bool(candidates),'native10 coverage snapshot missing')
    path = max(candidates)
    snapshot = reader.load(path)
    runbase = v1.FINISH + '/e5_runtime/controller_runs/E5-native-20260912-10/files'
    contract = reader.load(runbase + '/native_ready.json')['contract']
    raw = [reader.load(p) for p in reader.glob(runbase + '/journal/episodes/*.native.json')]
    measurement = audit_tau_snapshot(snapshot,raw,contract['task_ids'])
    measurement['snapshot_source'] = path
    measurement['allocated_task_count'] = len(contract['task_ids'])
    ms = result['rows'][4]['portfolio']['measurements']
    old = next((m for m in ms if m.get('run_id') == 'E5-native-20260912-10'),None)
    need(old is not None,'native10 raw measurement absent')
    measurement['model_identity'] = old['model_identity']
    ms[ms.index(old)] = measurement
    # E2: closeout of the v13 infrastructure allocation, not an actor benchmark run.
    e2path = v1.FINISH + '/e2_completion/direct_vm_v13/ledger29/completion_handoff_v13_ledger29.json'
    e2 = reader.load(e2path)
    e2m = infra_zero(e2,'native_episodes','INFRASTRUCTURE_AND_SYNTHETIC_ACTOR_PASS_NOT_NATIVE_EVALUATION',
                    'CORE-Bench v13 infrastructure closeout; no native episodes',45)
    need(e2['native_score'] is None and e2['task_code_executed'] is False and e2['controller_and_watchdog_exited'] is True
         and e2['cleanup'] == 'INDEPENDENT_VM_AND_BOTH_DISKS_NOT_FOUND', 'E2 closure/scope mismatch')
    for path, sha in e2['bound_receipts'].items(): reader.raw(path,sha)
    e2m.update({'cleanup':e2['cleanup'],'status':e2['status'],'source':e2path,'capsule_id':e2['capsule_id'],
                'prepared_file_count_verified':e2['prepared_file_count_verified'],'other44_runtime_compatibility':e2['other44_runtime_compatibility']})
    result['rows'][1]['portfolio']['measurements'] = [e2m]
    # E9: missed cutoff is a prevented launch, not an attempted zero score.
    e9base = v1.FINISH + '/e9_completion/continuation05/execution01'
    deadline = reader.load(e9base+'/deadline_fail_closed.json')
    closure = reader.load(e9base+'/independent_cleanup_closeout.json')
    e9m = infra_zero(deadline,'native_runs','FAILED_CLOSED_LATEST_ACTOR_START_MISSED',
                    'MLDevBench released scope; cutoff missed, no native run',34)
    need(closure['instances'] == [] and closure['disks'] == [] and closure['firewall-rules'] == []
         and closure['actor_session_absent'] is True and closure['ephemeral_ssh_keys_absent'] is True
         and closure['original_snapshot_preserved'] is True, 'E9 cleanup not established')
    reader.ref(deadline['ledger30_ref']); reader.ref(deadline['amendment_ref'])
    e9m.update({'status':deadline['status'],'cleanup_verified_at':closure['verified_at'],
                'source':e9base+'/deadline_fail_closed.json','scope_limitation':'0/34 released configurations; private full-scope percentage N/A.'})
    result['rows'][8]['portfolio']['measurements'] = [e9m]
    result['scope_label_audit'] = [result['rows'][i-1]['scope_resolution'] | {'lane':f'E{i}','display_label':result['rows'][i-1]['display_label']}
                                  for i in (3,4,7,12)]
    result['publication_recommendations'] = [
        'E1 portfolio: native-graded coverage 35/300 (11.67%); separately terminal-attempt coverage 110/300 (36.67%). Do not label 110 native evaluations.',
        'E1 native-report conditional success 4/35 (11.43%); terminal-attempt success including errors 4/110 (3.64%). Neither is a completed 300-task score.',
        'E5 native10: show graded/full97 and graded/allocated60 separately; started-without-grade is pending/unknown, not zero. Preserve other runs separately.',
        'Use N/A for an unavailable or unestablished original/private scope; name public subsets before displaying their own percentages.',
        'Use 0/45 for E2 CORE-Bench native episodes and 0/34 for E9 released MLDevBench scope when citing these no-native closeouts. Accuracy stays N/A.',
        'Always name both portfolios: E3 SDAB / MLAgentBench; E4 BankerToolBench retained; E7 BinaryAudit / CyberGym; E12 AppBench / VisualAgentBench.'
    ]
    # Avoid hardcoding future E1 counts into a reusable publication recommendation.
    swe = result['rows'][0]['portfolio']['measurements'][0]
    result['publication_recommendations'][0] = 'E1 portfolio: native-graded coverage '+v1.fmt(swe['coverage'])+'; separately terminal-attempt coverage '+v1.fmt(swe['terminal_attempt_coverage'])+'. Do not label terminal attempts native evaluations.'
    result['publication_recommendations'][1] = 'E1 conditional native-report success '+v1.fmt(swe['accuracy_or_success'])+'; terminal-attempt success including errors '+v1.fmt(swe['terminal_attempt_success_rate_errors_as_zero'])+'. Neither is a completed 300-task score.'
    result['as_of_utc'] = datetime.now(timezone.utc).isoformat()
    v1.validate_report(result)
    reader.stable()
    result['sources'] = reader.sources
    result['discovery_membership'] = reader.discoveries
    return result


def render(report):
    # Dedicated short publication table; earlier v1 detail is retained below it.
    lines = ['# Refreshed reporting audit', '', report['as_of_utc'], '',
             'N/A means scope unavailable/unestablished or no measured accuracy. An infrastructure-only closeout may prove zero executions without proving zero accuracy.', '',
             '| Lane | Original scope (not renamed) | Named portfolio scope | Native evaluation coverage | Accuracy / success |',
             '|---|---|---|---|---|']
    for row in report['rows']:
        orig,pub = row['original'],row['portfolio']
        original = orig['name'] + ': ' + (str(orig['denominator']) if orig['denominator'] else 'N/A — heldout/full denominator unestablished')
        ms = pub['measurements']
        coverage = '<br>'.join(m['label']+': '+v1.fmt(m['coverage']) for m in ms) or 'N/A — no measured receipt selected'
        score = '<br>'.join(v1.fmt(m.get('accuracy_or_success')).replace('unverified','N/A') for m in ms) or 'N/A'
        lines.append(f"| {row['lane']} | {original} | {pub['name']} ({pub['declared_scope']}) | {coverage} | {score} |")
    lines += ['', '## Publication recommendations', '']
    lines += ['- '+s for s in report['publication_recommendations']]
    m = next(m for m in report['rows'][4]['portfolio']['measurements'] if m.get('run_id') == 'E5-native-20260912-10')
    lines += ['', 'Native10 allocation coverage: '+v1.fmt(m['allocation_graded_coverage'])+
              '; started: '+v1.fmt(m['started_allocation_coverage'])+
              '; started without grade: '+str(len(m['started_without_grade']))+
              '; never started in this allocation: '+str(m['never_started_count'])+'.', '',
              'E2 v13: infrastructure and synthetic actor only, closed with VM/disks absent. E9: latest actor start missed; saved closeout verifies owned compute/disk/firewalls absent and snapshots retained. These are local receipt observations, not fresh cloud queries.', '',
              '## Canonical label resolution', '', '| Lane | Original suite | Portfolio suite | Same suite? |', '|---|---|---|---|']
    for s in report['scope_label_audit']:
        lines.append(f"| {s['lane']} | {s['original_suite_id']} | {s['portfolio_suite_id']} | {s['same_suite']} |")
    lines += ['', '## Detailed preserved evidence', '', v1.render(report)]
    return '\n'.join(lines)


def publish(root, dest, report):
    root=Path(root).resolve();dest=Path(dest).resolve()
    allowed=[root/'.codex-run/finish_20260912/progress_reporting',root/'finish/progress_reporting_audit']
    need(any(dest.is_relative_to(a) and dest != a for a in allowed),'output outside owned reporting subdirectories')
    need(not dest.exists(),'refuse overwrite')
    payloads={'report.json':json.dumps(report,indent=2)+'\n','report.md':render(report)}
    need(min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)>=v1.FLOOR,'DISK_BELOW_6_GIB')
    dest.mkdir(parents=True)
    for name,text in payloads.items():
        need(min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)>=v1.FLOOR,'DISK_BELOW_6_GIB')
        (dest/name).write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        report=build(v1.Reader(a.root));publish(a.root,a.out,report)
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(json.dumps({'status':'BLOCKED','error':str(e)}));return 2
    print(json.dumps({'status':'NEW_SNAPSHOT_PUBLISHED','sources':len(report['sources']),'out':str(a.out)}));return 0


if __name__ == '__main__': raise SystemExit(main())
