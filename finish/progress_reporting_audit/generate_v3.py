"""Versioned terminal receipt adapter; local reads and new owned snapshots only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import generate_v1 as v1
import generate_v2 as v2

need = v1.need
E5 = v1.FINISH + '/e5_runtime/native10_execution'
E9 = v1.FINISH + '/e9_completion/continuation06/revision02/execution01'


def terminal_tau(audit, episodes, allocated):
    need(audit['status'] == 'INDEPENDENT_NATIVE10_TERMINAL_AUDIT_PASS', 'unknown E5 terminal schema')
    need(audit['full_suite_denominator'] == 97 and audit['subset_denominator'] == len(allocated) == 60, 'E5 denominator drift')
    need(audit['full_suite_score'] is None, 'partial run promoted to full score')
    groups = [[e['task_id'] for e in episodes],
              [e['task_id'] for e in audit['native_failure_records']], audit['interrupted_ids']]
    flat = sum(groups, [])
    need(len(set(flat)) == len(flat), 'overlap or duplicate terminal task disposition')
    started = audit['current_started_ids']; never = audit['remaining_never_started']
    need(len(set(started)) == len(started) and set(flat) == set(started), 'terminal starts partition mismatch')
    need(len(set(never)) == len(never) and set(never).isdisjoint(started)
         and set(never) | set(started) == set(allocated), 'allocation partition mismatch')
    need(len(episodes) == audit['graded_episodes'], 'terminal raw grade count mismatch')
    need(len(episodes) + len(groups[1]) == audit['native_batch_records'], 'native batch disposition mismatch')
    need(all(e['termination_reason'] == 'infrastructure_error' for e in audit['native_failure_records']), 'unknown failure category')
    m = v1.tau_run('E5-native-20260912-10', episodes, allocated)
    need(m['accuracy_or_success']['numerator'] == audit['passes'], 'terminal reward mismatch')
    m.update(label='Native10 terminal completed-only grades',
             allocation_graded_coverage=v1.ratio(len(episodes), len(allocated)),
             started_allocation_coverage=v1.ratio(len(started), len(allocated)),
             started_without_grade=groups[1] + groups[2],
             infrastructure_failure_ids=groups[1], interrupted_ids=groups[2],
             never_started_count=len(never), allocated_task_count=len(allocated),
             disposition='TERMINAL_PARTIAL_RUN',
             selection_caveat='Infrastructure failures and interrupted tasks have no native reward; do not impute zeros or pool runs.')
    return m


def e9_terminal(receipt, result):
    need(receipt['status'] == 'NATIVE_ATTEMPT_TERMINAL_INFRASTRUCTURE_FAILURE_CLEANED', 'unknown E9 terminal schema')
    need(receipt['native_attempts'] == 1 and receipt['task_model_calls'] == 0
         and receipt['native_success'] is False, 'E9 pre-agent disposition contradicted')
    rows = result['results']
    need(len(rows) == 1 and rows[0]['task_id'] == receipt['native_task'], 'E9 task identity mismatch')
    runs = rows[0]['runs']
    need(len(runs) == 1 and runs[0]['success'] is False and runs[0]['agent_output'] is None
         and runs[0]['error'] == receipt['error'], 'E9 infrastructure failure mismatch')
    m = v1.measure('Continuation06 revision02: pre-agent infrastructure failure', 0, 34, kind='infrastructure_only')
    m.update(run_id='E9-continuation06-revision02-execution01', terminal_attempt_coverage=v1.ratio(1, 34),
             task_id=receipt['native_task'], task_model_calls=0, actor_smoke_calls=receipt['actor_smoke_calls'],
             disposition=receipt['status'], error=receipt['error'],
             raw_harness_success=v1.ratio(0, 1),
             raw_harness_success_is_capability_score=False,
             scope_limitation='One terminal harness attempt; zero capability-graded outcomes. Original/private scope percentage N/A.')
    return m


def build(reader):
    report = v2.build(reader)
    reader.raw('finish/progress_reporting_audit/generate_v3.py')
    # Discover terminal receipts so new/changed evidence cannot disappear silently.
    paths = reader.glob(v1.FINISH + '/e5_runtime/*/terminal_audit*.json')
    need(paths == [E5 + '/terminal_audit01.json'], 'new E5 terminal adapter review required')
    audit = reader.load(paths[0])
    episodes = []
    for row in audit['episodes']:
        episode = reader.ref(row['source_ref'])
        need(episode['task_id'] == row['task_id'] and episode['reward_info']['reward'] == row['reward'], 'E5 terminal grade binding mismatch')
        episodes.append(episode)
    for ref in audit['all_prior_start_refs']: reader.ref(ref)
    for key in ('native_results_ref', 'recovery_manifest_ref', 'cleanup_ref'): reader.ref(audit[key])
    runbase = v1.FINISH + '/e5_runtime/controller_runs/E5-native-20260912-10/files'
    raw = [reader.load(p) for p in reader.glob(runbase + '/journal/episodes/*.native.json')]
    need({e['task_id'] for e in raw} == {e['task_id'] for e in episodes}, 'new E5 raw grades need terminal reconciliation')
    m = terminal_tau(audit, episodes, reader.load(runbase + '/native_ready.json')['contract']['task_ids'])
    reconciliations = reader.glob(E5 + '/reconciliation_*.json')
    need(bool(reconciliations), 'E5 cleanup reconciliation missing')
    cleanup = reader.load(max(reconciliations))
    need(cleanup['status'] == 'TERMINAL_UNCHANGED_CLEANUP_REVERIFIED'
         and cleanup['native_graded'] == len(episodes) and cleanup['passes'] == audit['passes']
         and cleanup['native_failure_records'] == len(audit['native_failure_records'])
         and cleanup['interrupted_ids'] == audit['interrupted_ids'], 'E5 cleanup/count drift')
    need(all(cleanup[k] is True for k in ('actor_app_stopped_zero_tasks', 'owned_container_stopped_no_oom', 'port18016_no_listener')), 'E5 cleanup incomplete')
    m.update(source=paths[0], cleanup_source=max(reconciliations), cleanup_observed_at=cleanup['at'],
             cleanup='Saved reconciliation: actor stopped, container stopped, owned PIDs absent, port closed',
             historical_unique_started_not_graded=cleanup['unique_started'])
    ms = report['rows'][4]['portfolio']['measurements']
    old = next(x for x in ms if x.get('run_id') == m['run_id'])
    m['model_identity'] = old['model_identity']; ms[ms.index(old)] = m
    completions = reader.glob(v1.FINISH + '/e9_completion/**/completion_receipt.json')
    selected = E9 + '/completion_receipt.json'
    need(selected in completions, 'E9 terminal receipt missing')
    # Every newly found completion must be explicitly classified, never silently ignored.
    known = {v1.FINISH + '/e9_completion/' + s for s in (
        'completion_receipt.json', 'continuation01/completion_receipt.json',
        'continuation04/completion_receipt.json', 'continuation05/completion_receipt.json')}
    need(set(completions) == known | {selected}, 'new E9 terminal adapter review required')
    dispositions = [{'source':p, 'status':reader.load(p).get('status')} for p in completions]
    receipt = reader.load(selected)
    bound = {k:reader.ref(v) for k,v in receipt.items() if k.endswith('_ref')}
    m9 = e9_terminal(receipt, bound['native_result_ref'])
    c9 = bound['cleanup_ref']; actor = bound['actor_cleanup_ref']
    need(all(c9[k] == [] for k in ('instances', 'disks', 'firewall-rules'))
         and actor['stop_verified'] is True and actor['app']['state'] == 'stopped'
         and actor['app']['app_id'] == receipt['actor_app_id'] and str(actor['app']['tasks']) == '0', 'E9 cleanup identity/state mismatch')
    m9.update(source=selected, cleanup_observed_at=c9['at'], retained_snapshots=receipt['retained_snapshots'],
              cleanup='Saved receipts: VM, disk, firewalls absent; exact actor stopped with zero tasks')
    historical = report['rows'][8]['portfolio']['measurements'][0]
    historical.update(run_id='E9-continuation05', label='Historical continuation05: cutoff missed, no native attempt')
    report['rows'][8]['portfolio']['measurements'].append(m9)
    report['terminal_disposition_inventory'] = dispositions
    report['schema_version'] = 'evidence-progress-report-v3'
    report['publication_recommendations'][2] = 'E5 native10 terminal: 20/97 graded (20.62%), 20/60 allocated (33.33%), success 0/20; 33 starts comprise 20 grades, 10 infrastructure failures and 3 interrupted. Preserve each run separately.'
    report['publication_recommendations'][4] = 'E2 v13 remains infrastructure-only, 0/45 native. E9 continuation05 had no attempt; continuation06 revision02 has 1/34 terminal attempts but 0/34 capability-graded coverage and accuracy N/A. Two actor smoke calls are not task calls.'
    report['as_of_utc'] = datetime.now(timezone.utc).isoformat()
    v1.validate_report(report); reader.stable()
    report['sources'] = reader.sources; report['discovery_membership'] = reader.discoveries
    return report


def render(report):
    text = v1.render(report).replace('unverified', 'N/A — unestablished')
    return text + '\n## Current terminal dispositions\n\n' + '\n'.join('- ' + x for x in report['publication_recommendations']) + '\n\nCleanup statements are saved local receipt observations, not fresh provider queries. E2/E13 lifecycle amendments remain pending; local reviews add no native coverage.\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = build(v1.Reader(a.root))
    # Reuse the owned-path, exclusive-output and disk-floor guards with this renderer.
    v2.render = render
    v2.publish(a.root, a.out, report)
    print(json.dumps({'status':'NEW_SNAPSHOT_PUBLISHED','sources':len(report['sources']), 'out':str(a.out)}))


if __name__ == '__main__': main()
