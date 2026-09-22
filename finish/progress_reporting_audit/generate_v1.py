"""Read-only receipt reporting. Writes only a new directory under owned output roots."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
from pathlib import Path
import shutil

FLOOR = 6 * 1024**3
BASE = "outputs/PES_Phase2_Review_2026-09-12"
FINISH = BASE + "/finish"
ORIGINAL = [
    ("SWE-bench Pro", "swe_bench_pro_eval"), ("FrontierSWE", "frontier_swe_eval"),
    ("SDAB", "sdab_eval"), ("BankerToolBench", "banker_toolbench_eval"),
    ("APEX-Agents", "apex_agents_eval"), ("WebBench", "webbench_eval"),
    ("BinaryAudit", "binaryaudit_eval"), ("LifeSciBench", "lifescibench_eval"),
    ("MLE-bench", "mle_bench_eval"), ("AgentHarm", "agentharm_eval"),
    ("VerilogEval", "verilog_eval"), ("AppBench", "appbench_eval"),
    ("OpenReward Games", "openreward_games_eval"), ("FrontierMath", "frontiermath_eval")]
EXPECTED_ORIGINAL = [731, 17, 80, 100, 480, None, 28, 750, 75, None, 312, None, None, None]
EXPECTED_PUBLIC = [300, 45, 104, 100, 97, 812, 1507, 1967, 34, 97, 312, 746, 255, 4428]
PUBLIC_SUITES = ['swe_bench_multilingual_eval','core_bench_eval','mlagentbench_eval','banker_toolbench_eval',
                 'tau3_banking_eval','webarena_eval','cybergym_eval','lab_bench_public_eval','ml_dev_bench_eval',
                 'agentdojo_eval','verilog_eval','visual_agent_bench_eval','balrog_eval','omni_math_eval']
BOUNDARIES = [
    '300-task Multilingual replacement, not 731-task SWE-bench Pro. Native reports, terminal errors and success rates are separate.',
    '45 hard capsules containing 79 questions. Acquisition and setup are not evaluated capsules or answered questions.',
    '13 MLAgentBench task configurations with 8 repeats = 104 episodes; not the private 80-task SDAB suite.',
    '100 tasks with 15,054 rubric criteria; prepared directories and a recovered one-task metric are not a full clean run.',
    '97 Tau3 banking tasks; retain each run and selected completed subset. Not the 480-task APEX suite.',
    '812 WebArena episodes; setup and canonical website availability are not native grades and do not cover original WebBench.',
    '1,507 CyberGym tasks; no selected native result. Distinct from BinaryAudit and its 28-task primary split.',
    'All 1,967 public LAB-Bench MCQs only; no private LAB-Bench or original 750-task LifeSciBench completion claim.',
    '34 released MLDevBench configurations, not full private scope and not the 75 MLE-bench competitions.',
    '97 AgentDojo benign utility episodes; no prompt-injection security or original AgentHarm test_private claim.',
    '312 attempts across two framings of 156 Verilog problems; retained original result, not a new run.',
    '746 VisualAgentBench paper tasks; 462 released records do not establish full execution. Not original heldout AppBench.',
    '255 BALROG episodes / 58 configurations. BabyAI completed subset is not all families or original OpenReward Games.',
    '4,428 Omni-MATH dispositions; accepted-report accuracy excludes two parser failures. Not private FrontierMath.'
]


class EvidenceError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise EvidenceError(message)


def integer(n):
    return type(n) is int and n >= 0


def ratio(n, d):
    need(integer(n) and type(d) is int and 0 < d and n <= d, "invalid count/denominator")
    return {"numerator": n, "denominator": d, "fraction": n / d}


def measure(label, covered, total, passes=None, score_denominator=None, kind="native_evaluation", complete=False):
    coverage = ratio(covered, total)
    score = None if passes is None else ratio(passes, covered if score_denominator is None else score_denominator)
    need(not complete or covered == total, "partial scope cannot be complete")
    return {"label": label, "kind": kind, "coverage": coverage, "accuracy_or_success": score,
            "complete_named_scope": complete, "full_original_suite_claim": False}


def validate_report(report):
    rows = report['rows']
    need([r['lane'] for r in rows] == [f'E{i}' for i in range(1,15)], 'ordered 14 unique lanes required')
    for i, row in enumerate(rows):
        need(row['original']['suite_id'] == ORIGINAL[i][1], 'original suite substitution')
        need(row['original']['denominator'] == EXPECTED_ORIGINAL[i], 'original denominator drift')
        need(row['portfolio']['suite_id'] == PUBLIC_SUITES[i], 'portfolio suite drift')
        need(row['portfolio']['declared_denominator'] == EXPECTED_PUBLIC[i], 'portfolio denominator drift')
        for m in row['portfolio']['measurements']:
            c = m['coverage']
            need(c == ratio(c['numerator'], c['denominator']), 'coverage fraction mismatch')
            need(c['denominator'] == EXPECTED_PUBLIC[i], 'measured scope mismatch')
            a = m.get('accuracy_or_success')
            if a is not None:
                need(a == ratio(a['numerator'], a['denominator']), 'score fraction mismatch')
            need(not m.get('full_original_suite_claim'), 'replacement evidence promoted to original')
            need(not m['complete_named_scope'] or c['numerator'] == c['denominator'], 'partial complete flag')
    need(report['all_original_suites_complete'] is False, 'original campaign completion unsupported')
    derived = [r['lane'] for r in rows if any(m.get('accuracy_or_success') is not None for m in r['portfolio']['measurements'])]
    need(report['replacement_score_lanes'] == derived and report['replacement_score_lane_count'] == len(derived), 'score-progress count mismatch')


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.discoveries = {}
        self.cache = {}

    def guard(self):
        need(shutil.disk_usage(self.root).free >= FLOOR and shutil.disk_usage('/').free >= FLOOR, "DISK_BELOW_6_GIB")

    def resolve(self, path):
        p = Path(path)
        # Only this documented historical alias is relocated, never arbitrary suffixes.
        old = Path('/Users/arvind/Developer/agentic_repos/tinker-rl-lab')
        if p.is_absolute() and p.is_relative_to(old):
            p = self.root / p.relative_to(old)
        elif not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        need(p.is_relative_to(self.root), "source outside repository")
        return p

    def raw(self, path, expected=None):
        self.guard()
        p = self.resolve(path)
        need(p.is_file() and p.stat().st_size <= 64 * 1024**2, f"missing/oversized evidence: {p}")
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        need(expected is None or expected == sha, f"receipt hash mismatch: {p}")
        key = str(p.relative_to(self.root))
        if key in self.sources:
            need(self.sources[key]['sha256'] == sha, f"concurrent source change: {key}")
        self.sources[key] = {"sha256": sha, "bytes": len(raw)}
        return raw

    def load(self, path, expected=None):
        return json.loads(self.raw(path, expected))

    def ref(self, ref):
        return self.load(ref['path'], ref['sha256'])

    def glob(self, pattern):
        self.guard()
        files = sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file())
        self.discoveries[pattern] = files
        return files

    def stable(self):
        for path, ref in list(self.sources.items()):
            self.raw(path, ref['sha256'])
        for pattern, prior in self.discoveries.items():
            now = sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file())
            need(now == prior, f"new receipt appeared during snapshot: {pattern}; rerun")


def aggregate_swe(audits):
    """Union disjoint task identities, keeping terminal errors out of native coverage."""
    outcomes = {}
    for audit in audits:
        need(audit['full_suite_denominator'] == 300, "SWE Multilingual denominator drift")
        rows = audit['outcomes']
        need(len(rows) == audit['terminal_attempts'], "SWE terminal count mismatch")
        for row in rows:
            key = row['instance_id']
            need(key not in outcomes, f"duplicate SWE task across selected receipts: {key}")
            need(row['status'] in ('NATIVE_REPORT_VERIFIED', 'NATIVE_PATCH_APPLY_FAILED', 'NATIVE_EMPTY_PATCH'), "unknown SWE terminal status")
            need(type(row['resolved']) is bool, "invalid SWE resolution")
            need(not row['resolved'] or row['status'] == 'NATIVE_REPORT_VERIFIED', "error cannot be resolved")
            outcomes[key] = row
        need(sum(r['resolved'] for r in rows) == audit['native_resolved_count'], "SWE pass count mismatch")
    native = sum(r['status'] == 'NATIVE_REPORT_VERIFIED' for r in outcomes.values())
    passes = sum(r['resolved'] for r in outcomes.values())
    result = measure('SWE-bench Multilingual native reports', native, 300, passes if native else None)
    result['terminal_attempt_coverage'] = ratio(len(outcomes), 300)
    result['terminal_attempt_success_rate_errors_as_zero'] = ratio(passes, len(outcomes)) if outcomes else None
    result['patch_application_errors'] = sum(r['status'] == 'NATIVE_PATCH_APPLY_FAILED' for r in outcomes.values())
    result['empty_patch_errors'] = sum(r['status'] == 'NATIVE_EMPTY_PATCH' for r in outcomes.values())
    result['task_ids'] = sorted(outcomes)
    return result


def tau_run(run_id, episodes, allowed_ids=None):
    ids = set()
    passes = 0
    for e in episodes:
        need(e['task_id'] not in ids, "duplicate Tau3 task within run")
        need(allowed_ids is None or e['task_id'] in allowed_ids, 'Tau3 task outside native run contract')
        need(e.get('end_time') and e.get('termination_reason'), "nonterminal Tau3 episode")
        reward = e['reward_info']['reward']
        need(type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1), "nonbinary Tau3 reward")
        ids.add(e['task_id'])
        passes += int(reward)
    r = measure(run_id + ' only', len(ids), 97, passes if ids else None)
    r['run_id'] = run_id
    r['task_ids'] = sorted(ids)
    r['selection_caveat'] = 'Completed-only native records from this run; no cross-run score pooling or live-status assertion.'
    return r


def original_denominators(reader):
    def load(path): return reader.load('outputs/' + path)
    e1 = load('modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/receipt.json')
    e2 = load('e2_frontier_swe/e2_terminal_attempt_receipt_2026-08-22.json')
    e3 = load('e3_sdab/preflight_receipt_2026-08-22.json')
    e4 = load('e4_banker_toolbench/split_manifest_100.json')
    e5 = load('e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json')
    e7 = load('e7_binaryaudit/split_manifest.json')
    e8 = load('e8_lifescibench/terminal_receipt_2026-08-22.json')
    e9 = load('e9_mle_bench/terminal_receipt_2026-08-22.json')
    e11 = load('modal_e1_e14/2026-08-16/e11_full_receipt.json')
    values = [e1['coverage']['expected_tasks'], e2['scope']['official_suite_task_count'], e3['benchmark']['task_count'],
              e4['split']['task_count'], e5['dataset']['official_suite_tasks'], None,
              len(e7['split_manifest']['primary_eval']), e8['source']['officially_reported_tasks'], e9['split_manifest']['eval_split']['count'], None,
              e11['pass_at_1']['raw']['denominator'], None, None, None]
    need(values == EXPECTED_ORIGINAL, 'original denominator changed: requires contract review')
    need(len(e7['task_ids']) == 46 and len(e7['split_manifest']['train']) == 8
         and len(e7['split_manifest']['receipt_proven_heldout']) == 10, 'E7 split mismatch')
    return values, (e1, e2, e5, e7, e11)


def build(reader):
    reader.raw('finish/progress_reporting_audit/generate_v1.py')
    registry = reader.load('zvf-program/flagship/pavlovs_domain_contract.json')['suite_registry']
    totals, (e1, e2, e5, e7, e11) = original_denominators(reader)
    old = reader.load('outputs/e1_e14_results_2026-09-05/results.json')
    oldrows = {x['lane']: x for x in old['lanes']}
    public = reader.load(BASE + '/results.json')
    pubrows = {x['experiment']: x for x in public['rows']}
    need(set(oldrows) == set(pubrows) == {f'E{i}' for i in range(1, 15)}, '14 unique lanes required')
    rows = []
    for i, (name, suite) in enumerate(ORIGINAL, 1):
        need(suite in registry, f'original suite missing: {suite}')
        p = pubrows[f'E{i}']
        need(p['suite_id'] == PUBLIC_SUITES[i-1], 'replacement suite mapping changed')
        number = re.match(r'([0-9,]+)', str(p['denominator']))
        need(number is not None and int(number[1].replace(',','')) == EXPECTED_PUBLIC[i-1], 'replacement denominator changed')
        rows.append({'lane': f'E{i}', 'original': {'name': name, 'suite_id': suite, 'denominator': totals[i-1],
                     'contract_split': registry[suite]['split'], 'denominator_status': 'receipt_bound' if totals[i-1] else 'not_established_for_original_scope',
                     'historical_status': oldrows[f'E{i}']['benchmark_status'], 'measurements': []},
                     'portfolio': {'name': p['benchmark'], 'suite_id': p['suite_id'], 'declared_denominator': EXPECTED_PUBLIC[i-1],
                                   'declared_scope': p['denominator'], 'is_different_suite': suite != p['suite_id'], 'measurements': [],
                                   'scope_caveat': BOUNDARIES[i-1], 'historical_context_only': p['note']},
                     'findings': []})
    def orig(i, m): rows[i-1]['original']['measurements'].append(m)
    def pub(i, m): rows[i-1]['portfolio']['measurements'].append(m)
    orig(1, measure('Historical seed1818; mixed backends; failures/losses retained', e1['coverage']['native_evaluations'], 731,
                    e1['coverage']['resolved'], 731))
    rows[0]['original']['measurements'][0]['terminal_attempt_coverage'] = ratio(e1['coverage']['attempted_generations'], 731)
    rows[0]['original']['measurements'][0]['score_metric'] = 'canonical pass@1; not completed-only native accuracy'
    orig(2, {'label': 'Historical frozen-artifact native verifier replay, one task', 'kind': 'recovery_only',
             'recovery_coverage': ratio(e2['scope']['tasks_executed'], 17), 'accuracy_or_success': None,
             'native_task_reward': e2['result']['native_task_reward'], 'normalized_task_metric': e2['result']['leaderboard_normalized_task_score'],
             'complete_named_scope': False, 'full_original_suite_claim': False})
    r4 = reader.load('outputs/modal_e1_e14/2026-08-16/e4_recovery_evidence_boundary_2026-08-22.json')
    orig(4, {'label': 'Historical replayed artifact; not clean campaign score', 'kind': 'recovery_only',
             'recovery_coverage': ratio(1,100), 'native_task_metric': r4['attempt_metric'], 'accuracy_or_success': None,
             'complete_named_scope': False, 'full_original_suite_claim': False})
    orig(5, measure('Historical exact APEX sequential prefix', e5['selection']['native_scored_tasks'], 480))
    rows[4]['original']['measurements'][0].update({'attempted_coverage': ratio(e5['selection']['attempted_tasks'],480),
          'scored_receipt_mean': e5['metrics']['scored_receipt_mean'], 'metric_caveat': 'rubric score mean, not binary accuracy; unscored failures are not native zeros'})
    a7 = reader.load('outputs/e7_binaryaudit/2026-08-22_e7_paid_attempt_receipt.json')
    task = next(x for x in e7['tasks'] if x['raw_task_id'] == a7['attempt']['task'])
    need(task['split'] == a7['attempt']['split'] == 'primary_eval', 'E7 attempt not primary')
    orig(7, {'label': 'Historical errored task, no replay', 'kind': 'errored_historical_attempt', 'terminal_attempt_coverage': ratio(1,28),
             'accuracy_or_success': None, 'native_verifier_reward': a7['attempt']['native_verifier_reward'],
             'agent_exception': a7['attempt']['agent_exception'], 'complete_named_scope': False, 'full_original_suite_claim': False})
    rows[6]['findings'].append('Original denominator is 28 lane-constructed primary tasks; 46 is inventory (28 primary + 10 heldout-labelled + 8 train). Historical 1/46 wording is corrected here only.')
    e9csv = list(csv.DictReader(io.StringIO(reader.raw('outputs/e1_e14_results_2026-09-05/e9_competition_receipts.csv').decode())))
    ids = set()
    for item in e9csv:
        if item['arm'] == 'modal_streaming' and item['status'] == 'NATIVE_SINGLE_COMPETITION_GRADED':
            d = reader.load(item['receipt'])
            need(d['status'] == item['status'] and d['competition_id'] == item['competition_id']
                 and d['native_grade']['valid_submission'] is True, 'E9 receipt mismatch')
            ids.add(item['competition_id'])
    orig(9, measure('Historical legacy Tinker arm; unique native-graded competitions', len(ids), 75))
    rows[8]['original']['measurements'][0]['competition_ids'] = sorted(ids)
    rows[8]['findings'].append('Legacy MLE-bench grades are coverage, not accuracy; fresh merged-arm pilots are not unioned. MLDevBench 34 is released scope, not full private scope.')
    v = e11['pass_at_1']['raw']
    need(v['canonical'] is True and math.isclose(v['pass_at_1'], v['passes']/312), 'Verilog canonical mismatch')
    vm = measure('Retained canonical 312 framings of 156 problems',312,312,v['passes'],complete=True)
    orig(11, vm); pub(11, dict(vm))
    rows[10]['findings'].append('129/311 is noncanonical sensitivity and must not replace raw 129/312.')
    # Discover latest immutable audit per batch, not a union of repeated snapshots.
    audits = []
    patterns = [FINISH + '/e1_completion/partial_verified_*.json']
    wavefiles = reader.glob(FINISH + '/e1_completion/continuation/wave*/partial_verified_*.json')
    groups = {}
    for p in wavefiles: groups.setdefault(str(Path(p).parent), []).append(p)
    selections = [max(reader.glob(pattern)) for pattern in patterns if reader.glob(pattern)]
    selections += [max(paths) for paths in groups.values()]
    for path in selections:
        audit = reader.load(path)
        for ref in audit.get('reports', []): reader.raw(ref['path'], ref['sha256'])
        audits.append(audit)
    pub(1, aggregate_swe(audits))
    # Preserve each Tau run separately, including historical repeated starts.
    paths = reader.glob(FINISH + '/e5_runtime/controller_runs/E5-native-*/files/journal/episodes/*.native.json')
    groups = {}
    for p in paths: groups.setdefault(Path(p).parts[-5], []).append(p)
    for run, paths in sorted(groups.items()):
        contract = reader.load(FINISH + '/e5_runtime/controller_runs/' + run + '/files/native_ready.json')['contract']
        need(contract['source']['revision'] == 'a2c024725189473d2d7cea3a5cfdbcc67478e41f', 'Tau3 revision drift')
        ids = contract['task_ids']
        need(len(ids) == len(set(ids)) == contract['expected_episodes'] and
             contract.get('full_suite_denominator',len(ids)) == 97, 'Tau3 task scope drift')
        m = tau_run(run, [reader.load(p) for p in paths], ids)
        m['allocated_task_count'] = len(ids)
        m['model_identity'] = contract['model_identity']
        pub(5, m)
    rows[4]['findings'].append('Tau3 scores and coverage remain per-run. No pooled score, no sum of repeated task counts, no completion inferred from dispatches.')
    lab = reader.load('outputs/public_portfolio_2026-09-05/labbench_native_receipt.json')
    need(lab['expected_total'] == lab['evaluated'] == 1967 and lab['missing'] == 0, 'LAB public count mismatch')
    passes = round(lab['score']*1967)
    need(math.isclose(passes/1967,lab['score']), 'LAB score/count mismatch')
    pub(8, measure('Full public LAB-Bench MCQ split only',1967,1967,passes,complete=True))
    rows[7]['findings'].append('Native LAB metric coverage measures answer behavior; evaluated/expected=100% is evaluation coverage. Neither completes private LifeSciBench.')
    dojo = reader.load('outputs/public_portfolio_2026-09-05/agentdojo_continuation01_collection/native_summary.json')
    need(dojo['completed_episodes'] == dojo['score_denominator'] == 97 and dojo['protocol']['security_evaluation'] is False, 'AgentDojo scope mismatch')
    pub(10, measure('Native benign utility only; no injection-security or holdout claim',97,97,dojo['utility_passes'],complete=True))
    baby = reader.load(FINISH + '/e13_control/E13-native-20260912-01-completed13-native-score.json')
    need(baby['full_suite_episodes'] == 255 and baby['full_native_score'] is None, 'BALROG scope mismatch')
    ids = [e['episode_id'] for e in baby['episodes']]
    need(len(ids) == len(set(ids)) == baby['completed_episodes'], 'BALROG episode mismatch')
    need(sum(e['native_progression'] == 1 for e in baby['episodes']) == baby['successes'], 'BALROG successes mismatch')
    for ep in baby['episodes']:
        reader.raw(ep['native_json_ref']['path'],ep['native_json_ref']['sha256'])
        reader.raw(ep['terminal_ref']['path'],ep['terminal_ref']['sha256'])
    pub(13, measure('Completed BabyAI subset of BALROG only',baby['completed_episodes'],255,baby['successes']))
    rows[12]['findings'].append(baby['selection_caveat'])
    omni = reader.load(BASE + '/e14_scoring_recheck.json')
    need(omni['attempted'] == 4428 and omni['accepted'] + omni['skipped'] == 4428 and omni['strict_score'] is None, 'Omni exclusions/scope mismatch')
    need(math.isclose(omni['correct']/omni['accepted'],omni['official_accuracy']), 'Omni accuracy mismatch')
    pub(14, measure('Official parser-accepted Omni-MATH reports',omni['accepted'],4428,omni['correct']))
    rows[13]['portfolio']['measurements'][0].update({'excluded_reports':omni['skipped'], 'strict_full_scope_score':None})
    rows[13]['findings'].append('Two exclusions are not wrong answers or missing attempts; original FrontierMath remains separate. All 4428 dispositions reviewed in source receipt.')
    rows[11]['findings'].append('Original AppBench: six public task artifacts do not establish heldout denominator. Replacement VisualAgentBench 746 is paper scope, with 284 assets historically unrecovered.')
    for idx in (3,6,8,10,12,13,14):
        rows[idx-1]['original']['noncompletion_reason'] = oldrows[f'E{idx}'].get('remaining_requirement')
    # Parent state is comparison-only, never the source of measured counts or scores.
    parent = reader.load(FINISH + '/progress_reporting_state.json')
    score_lanes = [r['lane'] for r in rows if any(m.get('accuracy_or_success') is not None for m in r['portfolio']['measurements'])]
    full = [r['lane'] for r in rows if any(m.get('complete_named_scope') for m in r['portfolio']['measurements'])]
    result = {'schema_version':'evidence-progress-report-v1', 'as_of_utc':datetime.now(timezone.utc).isoformat(),
              'rows':rows, 'replacement_score_lanes':score_lanes, 'replacement_score_lane_count':len(score_lanes),
              'total_lanes':14, 'complete_named_portfolio_scopes':full, 'all_original_suites_complete':False,
              'rule':'Count a verified partial score for replacement progress; never call that full-suite completion. Zero is a score. No aggregate accuracy across suites or runs.',
              'parent_state_comparison_only':parent, 'verification_level':'Local receipts and selected hash-bound child artifacts; not a full transitive rerun or live-job audit.',
              'cloud_calls':0,'model_calls':0,'launch_authorized':False}
    swe = rows[0]['portfolio']['measurements'][0]
    result['audit_findings'] = [
        {'code':'E1_TERMINAL_VS_NATIVE', 'terminal':swe['terminal_attempt_coverage'], 'native':swe['coverage'],
         'detail':'Parent E1 coverage must name terminal attempts if it uses the larger count; accuracy requires its own denominator.'},
        {'code':'E7_PRIMARY_SPLIT', 'primary':28, 'inventory':46, 'heldout_labelled':10, 'train':8,
         'detail':'Historical 1/46 report wording is not primary-split coverage. Errored native reward is not accuracy.'},
        {'code':'NO_CROSS_PORTFOLIO_COMPLETION', 'detail':'E-label equality does not establish suite equality. Public completion never promotes an unavailable private suite.'},
        {'code':'NO_CROSS_RUN_TAU_POOLING', 'detail':'Each native run keeps its own task set, denominator and result. Allocation/setup/HTTP requests do not count as completed episodes.'},
        {'code':'NO_AGGREGATE_ACCURACY', 'detail':'The /14 score-progress count is a count of lanes with a result; it is neither mean accuracy nor completed original suites.'}
    ]
    validate_report(result)
    reader.stable()
    result['sources'] = reader.sources
    result['discovery_membership'] = reader.discoveries
    return result


def fmt(r):
    return 'unverified' if r is None else f"{r['numerator']}/{r['denominator']} ({100*r['fraction']:.2f}%)"


def render(report):
    lines = ['# Evidence-backed 14-lane report', '', report['as_of_utc'], '',
             f"Replacement score progress: {report['replacement_score_lane_count']}/14 (partial scores count). Complete named scopes: {', '.join(report['complete_named_portfolio_scopes'])}. Original campaign completion: false.", '',
             'Coverage below means native-graded outcomes / named scope. Accuracy or success uses its own explicit denominator. Unknown is not zero. Separate runs are not pooled.', '',
             '| Lane | Original suite denominator | Named portfolio evidence | Native evaluation coverage | Accuracy / success |',
             '|---|---|---|---|---|']
    for row in report['rows']:
        o, p = row['original'],row['portfolio']
        ms = p['measurements']
        label = '<br>'.join(m['label'] for m in ms) or p['name'] + ': no measured receipt selected'
        coverage = '<br>'.join(fmt(m.get('coverage')) for m in ms) or 'unverified'
        score = '<br>'.join(fmt(m.get('accuracy_or_success')) for m in ms) or 'unverified'
        lines.append(f"| {row['lane']} | {o['name']}: {o['denominator'] or 'unestablished'} | {label} | {coverage} | {score} |")
    lines += ['', '## Reporting audit', '']
    for item in report['audit_findings']: lines.append('- ' + item['code'] + ': ' + item['detail'])
    lines += ['', '## Original receipts and qualifications', '']
    for row in report['rows']:
        lines += [f"### {row['lane']} — {row['original']['name']}", '']
        for m in row['original']['measurements']:
            lines.append(f"- {m['label']}: {json.dumps(m,ensure_ascii=False)}")
        for f in row['findings']: lines.append('- ' + f)
        lines += ['', 'Portfolio boundary: ' + row['portfolio']['scope_caveat'], '']
    lines += ['## Provenance', '', report['verification_level'], '',
              'Every consumed file SHA-256 and discovery membership is stored in report.json. Parent live reporting files were read for comparison only. No source receipts were modified.', '']
    return '\n'.join(lines)


def publish(root, destination, report):
    root = Path(root).resolve()
    dest = Path(destination).resolve()
    allowed = [root/'.codex-run/finish_20260912/progress_reporting',root/'finish/progress_reporting_audit']
    need(any(dest.is_relative_to(p) and dest != p for p in allowed), 'output must be a NEW subdirectory in owned reporting roots')
    need(not dest.exists(), 'refuse overwrite of existing report')
    need(shutil.disk_usage(root).free >= FLOOR and shutil.disk_usage('/').free >= FLOOR, 'DISK_BELOW_6_GIB')
    payloads = {'report.json':json.dumps(report,indent=2)+'\n', 'report.md':render(report)}
    dest.mkdir(parents=True)
    for name, text in payloads.items():
        need(shutil.disk_usage(root).free >= FLOOR and shutil.disk_usage('/').free >= FLOOR, 'DISK_BELOW_6_GIB')
        (dest/name).write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:
        r=build(Reader(args.root))
        publish(args.root,args.out,r)
    except (EvidenceError,OSError,ValueError,KeyError,TypeError) as e:
        print(json.dumps({'status':'BLOCKED','error':str(e),'published':False}))
        return 2
    print(json.dumps({'status':'PUBLISHED_NEW_SNAPSHOT','score_lanes':r['replacement_score_lanes'],'sources':len(r['sources']),'out':str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
