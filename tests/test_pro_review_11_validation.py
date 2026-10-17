"""Offline synthetic regressions for review 11; no provider/model calls."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SAME = ROOT / 'reports/project_finish_2026-10-11/validation/samestack_v2'
AUDIT = ROOT / 'finish/progress_reporting_audit'
FIXES = ROOT / 'platform_hybrid/analysis_publication_fixes'
for path in (ROOT, SAME, AUDIT, FIXES): sys.path.insert(0, str(path))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


v2 = module('review11_samestack', SAME / 'v2_analysis.py')
sl = module('review11_stats', SAME / 'stats_lib.py')
gate = module('review11_gate', ROOT / 'reports/project_finish_2026-10-11/validation/colab_followup/accept_archive.py')
v1 = module('generate_v1', AUDIT / 'generate_v1.py')
g2 = module('generate_v2', AUDIT / 'generate_v2.py')
g3 = module('generate_v3', AUDIT / 'generate_v3.py')
e12 = module('review11_e12', ROOT / 'finish/e12_admission/validate_v1.py')
validation = module('review11_zvf', FIXES / 'analysis_validation.py')
exporter = module('review11_history', FIXES / 'pull_zvf_audit_histories.py')
from research.public_analysis import parser_review, parser_v2, parser_v3, parser_v4, reference_grader_v2, rescue_statistics


@pytest.fixture(autouse=True)
def synthetic_disk_floor(monkeypatch):
    # Test-only miniature fixtures do not relax the production disk-floor gate.
    monkeypatch.setattr(v1, 'FLOOR', 0)
    monkeypatch.setattr(e12, 'FLOOR', 0)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def put(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return sha(raw)


@pytest.fixture
def same_fixture(tmp_path):
    gold = {'test':[{'id':f'item{i}', 'question':f'Synthetic {i}', 'answer':'#### 42'} for i in range(6)]}
    gold_path = tmp_path / 'gold.json'; gold_sha = put(gold_path, gold)
    plan = {'schema_version':'samestack-publication-protocol-v3-candidate',
            'primary_contrasts':v2.PRIMARY, 'expected_seeds':[1,2,3],
            'endpoint':'post_accuracy', 'stopping_rule':'fixed_expected_seeds', 'equivalence_margin':.02,
            'dataset':{'sha256':gold_sha,'revision':'synthetic-revision'},
            'environment_fingerprint':'synthetic-full-environment-digest', 'run_protocol_version':'synthetic-v1',
            'selected_runs':[]}
    plan['environment'] = {field:'synthetic-' + field for field in ('provider','hardware','software','precision','checkpoint_revision','tokenizer_revision','evaluation_protocol')}
    plan['environment_fingerprint'] = v2.object_hash(plan['environment'])
    ids = [x['id'] for x in gold['test']]
    for arm in ('grpo_g8','grpo_g2','ppo','ppo_8x8'):
        for seed in (1,2,3):
            name = f'{arm}_s{seed}.json'
            texts = {phase:[r'\boxed{42}'] * 6 for phase in ('pre','post')}
            texts.update(dataset_revision='synthetic-revision', eval_item_ids=ids)
            text_path = tmp_path / name.replace('.json','_texts.json.gz')
            text_path.write_bytes(gzip.compress(json.dumps(texts).encode()))
            run = {'arm':arm,'seed':seed,'n_eval':6,'run_id':name,'attempt_id':'attempt-'+name,
                   'protocol_version':'synthetic-v1','environment_fingerprint':plan['environment_fingerprint'],
                   'dataset_revision':'synthetic-revision','eval_item_ids':ids,
                   'pre_correct':[1]*6,'post_correct':[1]*6,'pre_trunc':[0]*6,'post_trunc':[0]*6,
                   'step_log':[{'trunc_frac':0.,'mean_comp_len':1.}], 'elapsed_seconds':1.,'eval_seconds':.2}
            digest = put(tmp_path/name, run)
            plan['selected_runs'].append({'file':name, 'sha256':digest,'texts_sha256':sha(text_path.read_bytes()),
                                         **{k:run[k] for k in ('run_id','attempt_id','protocol_version','environment_fingerprint')}})
    return tmp_path, gold_path, plan


def mutate_run(fixture, mutate, name='grpo_g8_s1.json', texts_mutate=None):
    root, _, plan = fixture
    path = root / name; run = json.loads(path.read_text()); mutate(run)
    choice = next(x for x in plan['selected_runs'] if x['file'] == name)
    choice['sha256'] = put(path, run)
    if texts_mutate:
        text_path = path.with_name(path.stem + '_texts.json.gz')
        texts = json.loads(gzip.decompress(text_path.read_bytes())); texts_mutate(texts)
        text_path.write_bytes(gzip.compress(json.dumps(texts).encode())); choice['texts_sha256'] = sha(text_path.read_bytes())


def test_truncating_both_texts_and_correctness_is_rejected(same_fixture):
    mutate_run(same_fixture, lambda r:r['post_correct'].pop(), texts_mutate=lambda t:t['post'].pop())
    with pytest.raises(ValueError, match='Incomplete'): v2.validated_inputs(*same_fixture)


@pytest.mark.parametrize('change', [lambda r:r['eval_item_ids'].reverse(), lambda r:r.update(dataset_revision='other'),
                                   lambda r:r['pre_correct'].__setitem__(0, 2), lambda r:r.update(environment_fingerprint='other')])
def test_identity_revision_binary_environment_rejected(same_fixture, change):
    mutate_run(same_fixture, change)
    with pytest.raises(ValueError): v2.validated_inputs(*same_fixture)


def test_duplicate_run_cannot_replace_seed(same_fixture):
    root, _, plan = same_fixture
    old = root / 'grpo_g8_s1.json'; name='zz_grpo_g8_s1.json'
    (root/name).write_bytes(old.read_bytes())
    text = old.with_name(old.stem+'_texts.json.gz')
    (root/name.replace('.json','_texts.json.gz')).write_bytes(text.read_bytes())
    choice=deepcopy(plan['selected_runs'][0]);choice['file']=name;plan['selected_runs'].append(choice)
    with pytest.raises(ValueError,match='Duplicate experiment key'): v2.validated_inputs(*same_fixture)


def test_full_null_analysis_counts_strict_json_and_missing_family(same_fixture):
    root,gold,plan=same_fixture
    result=v2.analyze(root,gold,plan)
    assert all(x['items_graded'] == 18 for x in result['arms'].values())
    assert result['item_level']['g8-ppo']['status'] == 'no_discordance'
    assert result['item_level']['g8-ppo']['pooled_mcnemar_naive_p'] == 1
    assert result['contrasts']['stored']['g8-ppo']['p_t'] is None
    assert result['power']['g8-ppo']['status'] == 'unavailable'
    json.dumps(result,allow_nan=False)
    assert gate.accept_publication(result,deepcopy(result))['status']=='PASS'
    for choice in list(plan['selected_runs']):
        if choice['file'].startswith('ppo_8x8'):
            (root/choice['file']).unlink();plan['selected_runs'].remove(choice)
    incomplete=v2.analyze(root,gold,plan)
    assert incomplete['holm_family']==v2.PRIMARY
    assert incomplete['family_status']=='INCOMPLETE_PRIMARY_FAMILY'
    assert incomplete['confirmatory_publication_allowed'] is False
    with pytest.raises(ValueError):gate.accept_publication(incomplete,incomplete)


@pytest.mark.parametrize('bad', [[math.nan,math.nan],[1,math.inf],[[1,2],[3,4]]])
def test_signflip_nonfinite_or_nondimensional_rejected(bad):
    with pytest.raises(ValueError): sl.signflip(bad)


@pytest.mark.parametrize('bad',[math.nan,math.inf,-.1,1.1,True])
def test_holm_nonfinite_or_invalid_rejected(bad):
    with pytest.raises(ValueError):sl.holm({'a':bad})


def test_mcnemar_zero_discordance_and_exact_signflip_limit():
    assert sl.mcnemar_exact(0,0)==1
    assert sl.signflip([1]*5)==.0625
    assert sl.holm(dict.fromkeys('abc',.0625))==dict.fromkeys('abc',.1875)


def test_two_tails_adaptive_mde_and_search_exhaustion():
    delta=sl.mde(.1,2)
    assert sl.paired_t_power(.1,2,delta)==pytest.approx(.8)
    critical=stats.t.ppf(.975,4); ncp=.05/(.1/math.sqrt(5))
    expected=stats.nct.sf(critical,4,ncp)+stats.nct.cdf(-critical,4,ncp)
    assert sl.paired_t_power(.1,5,.05)==pytest.approx(expected)
    with pytest.raises(ValueError,match='max_n'):sl.n_needed(.1,.00001,max_n=3)
    with pytest.raises(ValueError,match='strictly positive'):sl.mde(0,5)


@pytest.mark.parametrize(('text','expected','reason'),[(r'\boxed{\frac{1}{0}}',0,'invalid_fraction'),
    (r'\boxed{\frac{.}{1}}',0,'invalid_fraction'),(r'\boxed{\text{42}}',1,'ok'),
    (r'\boxednotacommand then {42}',0,'nobox'),(r'\boxed{42',0,'unclosed_box')])
def test_independent_grader_total_and_command_boundary(text,expected,reason):
    assert v2.grade(text,'42')==(expected,reason)


def test_recursive_archive_gate_rejects_all_publication_mutations():
    original={'contrasts':{'stored':{'g8-ppo':{'t_ci':[.1,.2],'holm_p_t':.04,'seeds':[1,2], 'diffs':[.1,.2]},
                                   'g8-ppo_8x8':{'mean':.1}}},
              'power':{'tost':.01},'arms':{'g8':{'item_disagreements_pre_total':0}},'platform_calibrated':{'mean':.3}}
    for path in [('contrasts','stored','g8-ppo','t_ci',0),('contrasts','stored','g8-ppo','holm_p_t'),
                 ('contrasts','stored','g8-ppo','seeds',0),('contrasts','stored','g8-ppo','diffs',0),
                 ('contrasts','stored','g8-ppo_8x8','mean'),('power','tost'),('arms','g8','item_disagreements_pre_total'),
                 ('platform_calibrated','mean')]:
        mutated=deepcopy(original);current=mutated
        for key in path[:-1]:current=current[key]
        current[path[-1]]+=1
        with pytest.raises(gate.AuditError):gate.compare_publication(original,mutated)
    with pytest.raises(gate.AuditError):gate.compare_publication({'p':math.nan},{'p':math.nan})
    with pytest.raises(gate.AuditError):gate.compare_publication({'n':2},{'n':2.})


def test_archive_checks_survive_python_optimization():
    path=ROOT/'reports/project_finish_2026-10-11/validation/colab_followup/accept_archive.py'
    code=f"import runpy; m=runpy.run_path({str(path)!r}); m['compare_publication']({{'p':.01}},{{'p':.9}})"
    result=subprocess.run([sys.executable,'-O','-c',code],capture_output=True,text=True)
    assert result.returncode!=0 and 'finite numeric mismatch' in result.stderr
    # Frozen remote payload is not run; both integrity predicates remain explicit.
    remote=(path.parent/'remote_job.py').read_text()
    import ast
    assert not any(isinstance(x,ast.Assert) for x in ast.walk(ast.parse(remote)))


@pytest.mark.parametrize('digest',[None,'','garbage','g'*64,'a'*63,'a'*65])
def test_ref_requires_real_digest(tmp_path,digest):
    (tmp_path/'child.json').write_text('{}')
    with pytest.raises(v1.EvidenceError,match='SHA256_REQUIRED'):
        v1.Reader(tmp_path).ref({'path':'child.json','sha256':digest})


def test_ref_missing_and_wrong_digest(tmp_path):
    (tmp_path/'child.json').write_text('{}')
    with pytest.raises(v1.EvidenceError):v1.Reader(tmp_path).ref({'path':'child.json'})
    with pytest.raises(v1.EvidenceError,match='hash mismatch'):v1.Reader(tmp_path).raw_ref({'path':'child.json','sha256':'0'*64})


def swe_fixture(root):
    binding={'suite_id':'swe_bench_multilingual_eval','dataset_revision':'synthetic','model_id':'synthetic-model','run_id':'synthetic-run','evaluator_version':'synthetic-v1'}
    def ref(name,value):return {'path':name,'sha256':put(root/name,value)}
    manifest=ref('tasks.json',{'schema_version':'swe-canonical-task-manifest-v1','suite_id':binding['suite_id'],'dataset_revision':'synthetic','task_ids':[f't{i}' for i in range(300)]})
    report=ref('native.json',{'schema_version':'swe-native-task-report-v1','binding':binding,'instance_id':'t0','resolved':True})
    rows=[{'instance_id':'t0','status':'NATIVE_REPORT_VERIFIED','resolved':True,'run_id':'synthetic-run'},
          {'instance_id':'t1','status':'NATIVE_PATCH_APPLY_FAILED','resolved':False,'run_id':'synthetic-run'}]
    return {'full_suite_denominator':300,'terminal_attempts':2,'native_resolved_count':1,'outcomes':rows,'task_manifest':manifest,'reports':[report],'binding':binding}


def test_swe_native_report_derived_and_errors_separate(tmp_path):
    audit=swe_fixture(tmp_path);result=v1.aggregate_swe([audit],v1.Reader(tmp_path))
    assert result['coverage']['numerator']==1
    assert result['terminal_attempt_coverage']['numerator']==2
    assert result['accuracy_or_success']['fraction']==1
    assert result['terminal_attempt_success_rate_errors_as_zero']['fraction']==.5


@pytest.mark.parametrize('mutation',['foreign','missing','reuse','contradiction','binding'])
def test_swe_fake_native_claims_rejected(tmp_path,mutation):
    audit=swe_fixture(tmp_path)
    if mutation=='foreign':audit['outcomes'][0]['instance_id']='not-a-suite-id'
    elif mutation=='missing':audit['reports']=[]
    elif mutation=='reuse':audit['reports']*=2
    elif mutation=='binding':audit['binding']['model_id']='other'
    else:
        report=json.loads((tmp_path/'native.json').read_text());report['resolved']=False
        audit['reports'][0]['sha256']=put(tmp_path/'native.json',report)
    with pytest.raises(v1.EvidenceError):v1.aggregate_swe([audit],v1.Reader(tmp_path))


def empty_report():
    rows=[]
    for i in range(14):
        rows.append({'lane':f'E{i+1}','original':{'suite_id':v1.ORIGINAL[i][1],'denominator':v1.EXPECTED_ORIGINAL[i], 'name':v1.ORIGINAL[i][0],'measurements':[]},
                     'portfolio':{'suite_id':v1.PUBLIC_SUITES[i],'declared_denominator':v1.EXPECTED_PUBLIC[i], 'name':v1.PUBLIC_SUITES[i], 'scope_caveat':'synthetic','measurements':[]},'findings':[]})
    return {'rows':rows,'total_lanes':14,'complete_named_portfolio_scopes':[],
            'all_original_suites_complete':False,'replacement_score_lanes':[],'replacement_score_lane_count':0,
            'as_of_utc':'synthetic','audit_findings':[],'verification_level':'synthetic only','publication_recommendations':['']*6}


@pytest.mark.parametrize('mutation',['coverage_score','original','completion','total','ratio','policy'])
def test_final_gate_rejects_contradictory_claims(mutation):
    report=empty_report();m=v1.measure('synthetic',1967,1967,100,complete=True)
    report['rows'][7]['portfolio']['measurements']=[m]
    report.update(replacement_score_lanes=['E8'],replacement_score_lane_count=1,complete_named_portfolio_scopes=['E8'])
    v1.validate_report(report)
    if mutation=='coverage_score':m['coverage']=v1.ratio(1,1967);m['complete_named_scope']=False;report['complete_named_portfolio_scopes']=[]
    elif mutation=='original':report['rows'][0]['original']['measurements']=[dict(v1.measure('original',1,731,1), accuracy_or_success={'numerator':999,'denominator':1,'fraction':999})]
    elif mutation=='completion':report['complete_named_portfolio_scopes'].append('E7')
    elif mutation=='total':report['total_lanes']=999
    elif mutation=='ratio':m['coverage']['fraction']=.5
    else:m['score_denominator_policy']='arbitrary'
    with pytest.raises(v1.EvidenceError):v1.validate_report(report)


def test_snapshot_cutoff_and_later_disclosure():
    ids=['a','b'];episode=lambda task,time:{'task_id':task,'end_time':time,'termination_reason':'done','reward_info':{'reward':0}}
    snapshot={'at':10,'snapshot_mode':'historical','started_ids_in_original_order':ids,'started_unique':2,'graded':{'a':0},'started_without_grade':['b'], 'never_started_ids':[], 'full_suite_denominator':97,'subset_denominator':2,'prior37_refs_hash_verified':True,'no_overlap_prior37':True,'no_duplicate_attempt_intents':True,'full_suite_score':None}
    episodes=[episode('a',9),episode('b',11)]
    result=g2.audit_tau_snapshot(snapshot,episodes,ids)
    assert result['later_graded_task_ids']==['b']
    episodes[1]['end_time']=9
    with pytest.raises(ValueError,match='cutoff'):g2.audit_tau_snapshot(snapshot,episodes,ids)
    snapshot['snapshot_mode']='current'
    with pytest.raises(ValueError,match='current snapshot'):g2.audit_tau_snapshot(snapshot,episodes,ids)


def test_parser_duplicate_request_ids_rejected_every_entry_point():
    record={'request':{'logical_request_id':'same','prompt_id':'prompt'},'raw_completion_text':'42',
            'v2':{'status':'bad','value':None,'source':None},'v3':{'status':'ok','value':'42','source':'hash'}}
    records=[record,dict(deepcopy(record),raw_completion_text='other')]
    for call in [lambda:parser_review.packet_rows(records),lambda:parser_review.validate_packet([],{},records),
                 lambda:parser_review.decide(records,{},[]),lambda:parser_review.review_and_decide(records,[],{},[])]:
        with pytest.raises(ValueError,match='logical_request_id'):call()


def test_review_orchestration_digest_and_complete_scope():
    record={'request':{'logical_request_id':'one','prompt_id':'prompt'},'raw_completion_text':'#### 42',
            'v2':{'status':'bad','value':None,'source':None},'v3':{'status':'ok','value':'42','source':'hash'}}
    packet,mapping=parser_review.packet_rows([record]);digest=parser_review.packet_digest(packet)
    reviews=[{'reviewer_id':str(i),'packet_sha256':digest,'independent_output_only_review':True,
              'completed_at_utc':'2000-01-01T00:00:00Z','cases':[{'case_id':packet[0]['case_id'],'verdict':'single_numeric','value':'42','rationale':'synthetic'}]} for i in range(2)]
    result=parser_review.review_and_decide([record],packet,mapping,reviews)
    assert result['output_denominator']==result['reviewed_changed_count']==1
    reviews[1]['packet_sha256']='0'*64
    with pytest.raises(ValueError):parser_review.review_and_decide([record],packet,mapping,reviews)


@pytest.mark.parametrize('text',['#### <answer>\n5','#### <answer>\n#### 5'])
def test_explicit_candidate_closes_historical_xml_gap(text):
    assert parser_v2.parse_answer(text)['status']=='ok'
    assert parser_v3.parse_answer(text)['status']=='ok'
    assert parser_v4.parse_answer(text)['status']=='malformed_answer_tag'
    assert parser_v4.parse_answer('#### <answer>\n#### 5\n#### </answer>')['value']=='5'


@pytest.mark.parametrize(('text','expected'),[('#### 1/2','1/2'),('#### 1e3','1000'),('#### 12\n#### 13',None),('#### 9007199254740993','9007199254740993')])
def test_exact_reference_candidates(text,expected):assert reference_grader_v2.extract_answer(text)==expected


def test_exact_reference_disagreements_keep_item_identity():
    report=reference_grader_v2.compare_items([{'item_id':'fraction','text':'#### 1/2'}])
    assert report['disagreements']==[{'item_id':'fraction','historical_value':'1','exact_value':'1/2'}]


@pytest.mark.parametrize(('k','n'),[(1,0),(.5,2),(1,2.5),(True,2),(-1,2)])
def test_rescue_invalid_counts(k,n):
    with pytest.raises(ValueError):rescue_statistics.binomial_summary(k,n)


@pytest.mark.parametrize(('k','n','m','prior'),[(0,8,-1,1),(0,0,1,1),(0,8,.5,1),(0,8,1,0),(0,8,1,math.inf)])
def test_rescue_invalid_prediction(k,n,m,prior):
    with pytest.raises(ValueError):rescue_statistics.rescue_prediction(k,n,m,prior)


def test_empirical_jensen_sign_counterexample_and_mean_preserving_boundary():
    result=validation.empirical_delta_expectation([.49,.51],8)
    assert result['latent_probability_delta']==pytest.approx(-.000087587514,abs=1e-10)
    assert result['empirical_plugin_expectation']==pytest.approx(.0204289,abs=1e-7)
    for pb in (.05,.5,.95):assert sum(validation.mean_preserving_points(pb))/2==pytest.approx(pb)


def test_trajectory_bootstrap_retains_whole_seeds():
    rows=[{'seed':seed,'G':8,'id':step,'delta_div':float(seed)} for seed in (0,1) for step in range(10)]
    result=validation.trajectory_summary(rows,samples=100)
    assert result['n_independent_seed_clusters']==2 and result['n_rows']==20
    assert result['ci']==[0.,1.]
    with pytest.raises(ValueError):validation.trajectory_summary(rows+[rows[0]])


def test_robust_incremental_uses_hc3_and_actual_cohort():
    import pandas as pd
    smf = pytest.importorskip('statsmodels.formula.api', reason='statsmodels absent from offline review venv')
    rng=np.random.default_rng(4);x=np.linspace(-2,2,100)
    data=pd.DataFrame({'zvf':x+rng.normal(size=100)*(1+x*x),'r':x,'r2':x*x,'run':[f'r{i}' for i in range(100)]})
    data.loc[0,'r']=math.nan
    actual=validation.robust_incremental(data,'zvf ~ 1','zvf ~ r + r2')
    expected=smf.ols('zvf ~ r + r2',data.dropna()).fit(cov_type='HC3').wald_test('r = 0, r2 = 0',use_f=True,scalar=True)
    assert actual['p']==pytest.approx(float(expected.pvalue))
    assert actual['covariance_type']=='HC3' and actual['n']==99 and 'r0' not in actual['run_ids']


def test_registry_r0_nontransitive_chain_and_even_median():
    query=SimpleNamespace(field_view=lambda r:r,is_known=lambda c:c['value'] is not None,_key=lambda _,v:v)
    records={name:{'x':{'value':value}} for name,value in [('a',1),('b',None),('c',2)]}
    assert validation.equivalence_classes(records,query)==[['a'],['b'],['c']]
    records['d']={'x':{'value':1}}
    assert validation.equivalence_classes(records,query)==[['a','d'],['b'],['c']]
    assert validation.coverage_median([.1,.9])==.5
    with pytest.raises(ValueError):validation.coverage_median([])


def test_history_export_missing_steps_and_identity_errors_are_explicit():
    record={'run_id':'run','display_name':'synthetic','model':'synthetic','config':{'steps':2}}
    response={'data':{'project':{'run':{'name':'run','state':'finished','config':{'steps':2},'summaryMetrics':{},'sampledHistory':[[{'step':0,'reward':.2,'zvf':.3}]]}}}}
    opener=lambda *a,**k:io.BytesIO(json.dumps(response).encode())
    result=exporter.get(record,'synthetic-not-a-credential',opener)
    assert result['retrieval_status']=='incomplete' and 'expected-step' in result['error']
    response['data']['project']['run']['sampledHistory'][0].append({'step':1,'reward':.2,'zvf':.3})
    result=exporter.get(record,'synthetic-not-a-credential',opener)
    assert result['retrieval_status']=='limited_sampled_history'
    assert result['snapshot_status']=='single_response_not_immutable_atomic_snapshot'


def test_insufficient_pairs_report_unavailable_not_nan():
    result=sl.paired({1:.2},{1:.1})
    assert result['status']=='insufficient_pairs' and result['p_t'] is None and result['sd'] is None
    json.dumps(result,allow_nan=False)


def e12_fixture(root,monkeypatch):
    def ref(name,value):return {'path':name,'sha256':put(root/name,value)}
    ids=[f't{i}' for i in range(6)]
    assets={'csv':ref('csv',{'synthetic':True}),'split':ref('split',{'tasks':[{'task_id':t} for t in ids]}),
            'disjointness':ref('disjointness',{})}
    monkeypatch.setattr(e12,'PINS',{name:asset['sha256'] for name,asset in assets.items()})
    model={'id':'synthetic','revision':'a'*40}
    binding={'suite_id':'appbench_eval','dataset_revision':e12.REVISION,'split_sha256':assets['split']['sha256'],'model':model}
    attempt={'task_id':'t0','ordinal':1,'run_id':'next'}
    manifest={'schema_version':'e12-offline-admission-v1','suite_id':'appbench_eval','dataset_revision':e12.REVISION,'scope':'original_heldout','model':model,'attempt':attempt,'assets':assets}
    env={'issuer':'AfterQuery','official_exact_environment':True,'image_digest':'sha256:'+'b'*64}
    for key in ('template','runtime_reset','deployment','artifact_verification','side_effect_verification','credentials_policy'):env[key]=ref(key,{'synthetic':True})
    docs={
      'permission':{'issuer':'AfterQuery','evaluation_allowed':True,'aggregate_publication_allowed':True},
      'holdout':{'issuer':'AfterQuery','held_out_from_model_training':True,'task_ids':ids},
      'environment':env,
      'grading':{'issuer':'AfterQuery','graders':[{'id':'a','qualified_full_stack':True},{'id':'b','qualified_full_stack':True}],
                 'independent_then_consensus':True,'binary_per_item':True,'attempts_per_task':3,'aggregation':'sum_best_of_three_per_task_over_151','rubric_counts':[24,33,22,25,23,24],'website_23_vs_csv_24_resolved':True},
      'attempt_ledger':{'authoritative_complete':True,'attempts':[{'task_id':t,'ordinal':n,'state':'not_started'} for t in ids for n in (1,2,3)]}}
    def seal():
        for name,value in docs.items():value['binding']=deepcopy(binding);assets[name]=ref(name,value)
        assets['parent_review']=ref('parent',{'binding':binding,'attempt':attempt,'authenticity_and_history_verified':True,'reviewer':'synthetic',
                                             'evidence_sha256':{n:assets[n]['sha256'] for n in e12.EVIDENCE}})
    seal()
    return manifest,docs,binding,ref,seal


@pytest.mark.parametrize('mutation',['empty','duplicate','wrong_task','wrong_ordinal','wrong_state','wrong_binding','bool_ordinal'])
def test_e12_receipts_bound_to_distinct_typed_ledger_slots(tmp_path,monkeypatch,mutation):
    manifest,docs,binding,ref,seal=e12_fixture(tmp_path,monkeypatch)
    rows=docs['attempt_ledger']['attempts']
    for index in (1,2):
        row=rows[index];row.update(state='completed',run_id=f'historical-{index}')
        value={'schema_version':'e12-attempt-receipt-v2-candidate','binding':binding,
               'task_id':row['task_id'],'ordinal':row['ordinal'],'run_id':row['run_id'],'state':row['state']}
        row['receipt']=ref(f'attempt-{index}',value)
    seal();assert e12.validate(manifest,tmp_path)['blockers']==[]
    row=rows[1];value=json.loads((tmp_path/row['receipt']['path']).read_text())
    if mutation=='empty':value={}
    elif mutation=='duplicate':rows[2]['run_id']=row['run_id'];rows[2]['receipt']=deepcopy(row['receipt'])
    elif mutation=='wrong_task':value['task_id']='t5'
    elif mutation=='wrong_ordinal':value['ordinal']=3
    elif mutation=='wrong_state':value['state']='failed'
    elif mutation=='wrong_binding':value['binding']['model']['id']='different'
    else:rows[0]['ordinal']=True
    row['receipt']=ref('mutated',value);seal()
    result=e12.validate(manifest,tmp_path)
    assert result['status']=='BLOCKED' and result['launch_authorized'] is False


@pytest.mark.parametrize(('passes','smoke_calls'),[(0,2),(1,3)])
def test_v3_full_build_and_render_use_finalized_counts(monkeypatch,passes,smoke_calls):
    ids=[f't{i}' for i in range(60)]
    episodes=[{'task_id':t,'end_time':'synthetic','termination_reason':'done','reward_info':{'reward':int(i<passes)}} for i,t in enumerate(ids[:20])]
    audit={'status':'INDEPENDENT_NATIVE10_TERMINAL_AUDIT_PASS','full_suite_denominator':97,'subset_denominator':60,
           'full_suite_score':None,'graded_episodes':20,'passes':passes,'native_batch_records':30,
           'current_started_ids':ids[:33],'remaining_never_started':ids[33:],
           'native_failure_records':[{'task_id':t,'termination_reason':'infrastructure_error'} for t in ids[20:30]],
           'interrupted_ids':ids[30:33], 'episodes':[{'task_id':e['task_id'],'reward':e['reward_info']['reward'],'source_ref':{'path':e['task_id']}} for e in episodes],
           'all_prior_start_refs':[],'native_results_ref':{'path':'empty'},'recovery_manifest_ref':{'path':'empty'},'cleanup_ref':{'path':'empty'}}
    cleanup={'status':'TERMINAL_UNCHANGED_CLEANUP_REVERIFIED','native_graded':20,'passes':passes,'native_failure_records':10,
             'interrupted_ids':ids[30:33], 'actor_app_stopped_zero_tasks':True,'owned_container_stopped_no_oom':True,'port18016_no_listener':True,'at':'synthetic','unique_started':33}
    receipt={'status':'NATIVE_ATTEMPT_TERMINAL_INFRASTRUCTURE_FAILURE_CLEANED','native_attempts':1,'task_model_calls':0,
             'native_success':False,'native_task':'synthetic','actor_smoke_calls':smoke_calls,'error':'synthetic infrastructure error',
             'native_result_ref':{'path':'e9-result'},'cleanup_ref':{'path':'e9-cleanup'},'actor_cleanup_ref':{'path':'actor'},'actor_app_id':'actor-id','retained_snapshots':[]}
    e9result={'results':[{'task_id':'synthetic','runs':[{'success':False,'agent_output':None,'error':receipt['error']}]}]}
    base=empty_report();base['rows'][4]['portfolio']['measurements']=[dict(v1.measure('old',16,97,0),run_id='E5-native-20260912-10',model_identity='synthetic')]
    base['rows'][8]['portfolio']['measurements']=[v1.measure('old infra',0,34,kind='infrastructure_only')]
    base.update(replacement_score_lanes=['E5'],replacement_score_lane_count=1)
    monkeypatch.setattr(g3.v2,'build',lambda reader:deepcopy(base))
    runbase=v1.FINISH+'/e5_runtime/controller_runs/E5-native-20260912-10/files'
    completion_paths=[v1.FINISH+'/e9_completion/'+x for x in ('completion_receipt.json','continuation01/completion_receipt.json','continuation04/completion_receipt.json','continuation05/completion_receipt.json')]+[g3.E9+'/completion_receipt.json']
    loads={g3.E5+'/terminal_audit01.json':audit,g3.E5+'/reconciliation_01.json':cleanup,
           runbase+'/native_ready.json':{'contract':{'task_ids':ids}},g3.E9+'/completion_receipt.json':receipt,
           'empty':{},'e9-result':e9result,'e9-cleanup':{'instances':[],'disks':[],'firewall-rules':[],'at':'synthetic'},
           'actor':{'stop_verified':True,'app':{'state':'stopped','app_id':'actor-id','tasks':0}},
           **{e['task_id']:e for e in episodes}}
    for p in completion_paths[:-1]:loads[p]={'status':'synthetic-historical'}
    class Reader:
        sources={};discoveries={}
        def raw(self,path):return b'synthetic'
        def stable(self):pass
        def load(self,path):return deepcopy(loads[path])
        def ref(self,ref):return self.load(ref['path'])
        def glob(self,pattern):
            if pattern.endswith('terminal_audit*.json'):return [g3.E5+'/terminal_audit01.json']
            if pattern.endswith('*.native.json'):return [e['task_id'] for e in episodes]
            if pattern.endswith('reconciliation_*.json'):return [g3.E5+'/reconciliation_01.json']
            if pattern.endswith('**/completion_receipt.json'):return completion_paths
            raise AssertionError(pattern)
    report=g3.build(Reader());rendered=g3.render(report)
    assert f'success {passes}/20' in rendered
    assert f'{smoke_calls} actor smoke calls' in rendered
    assert report['rows'][4]['portfolio']['measurements'][0]['accuracy_or_success']['numerator']==passes
    assert report['claim_receipts'][0]['evidence_level']=='native_episode_outcomes_recomputed'
    json.dumps(report,allow_nan=False)


def test_incomplete_last_box_cannot_rescue_an_earlier_answer():
    assert v2.grade(r'\boxed{42} then \boxed{', '42') == (0,'unclosed_box')


def test_absent_primary_arm_has_incomplete_status(same_fixture):
    root,gold,plan=same_fixture
    for choice in list(plan['selected_runs']):
        if choice['file'].startswith('grpo_g8'):
            (root/choice['file']).unlink();plan['selected_runs'].remove(choice)
    result=v2.analyze(root,gold,plan)
    assert result['family_status']=='INCOMPLETE_PRIMARY_FAMILY'
    assert result['contrasts']['stored']['g8-ppo']['status']=='insufficient_pairs'
    json.dumps(result,allow_nan=False)


def test_direct_parser_decision_is_packet_bound():
    record={'request':{'logical_request_id':'one','prompt_id':'p'},'raw_completion_text':'#### 42',
            'v2':{'status':'bad','value':None,'source':None},'v3':{'status':'ok','value':'42','source':'hash'}}
    packet,mapping=parser_review.packet_rows([record])
    reviews=[{'reviewer_id':str(i),'packet_sha256':'0'*64,'independent_output_only_review':True,
              'completed_at_utc':'2000-01-01T00:00:00Z','cases':[{'case_id':packet[0]['case_id'],'verdict':'single_numeric','value':'42','rationale':'synthetic'}]} for i in range(2)]
    with pytest.raises(ValueError,match='packet binding'):parser_review.decide([record],mapping,reviews)


def test_history_export_cli_returns_nonzero_for_partial(tmp_path,monkeypatch):
    registry=tmp_path/'registry.jsonl';registry.write_text(json.dumps({'project':'zvf-audit','run_id':'synthetic'})+'\n')
    monkeypatch.setattr(exporter,'REG',registry)
    monkeypatch.setattr(exporter.netrc,'netrc',lambda:SimpleNamespace(authenticators=lambda host:('synthetic',None,'test-only')))
    monkeypatch.setattr(exporter,'get',lambda record,authorization:{'run_id':record['run_id'],'error':'incomplete','retrieval_status':'incomplete'})
    output=tmp_path/'new-export.json'
    assert exporter.main([str(output)])==2
    assert json.loads(output.read_text())[0]['retrieval_status']=='incomplete'


def test_historical_v1_duplicate_run_rejected_under_optimization(tmp_path):
    run={'arm':'grpo_g8','seed':1,'pre_correct':[0]*200,'post_correct':[0]*200,'n_eval':200,
         'heldout_post_acc':0,'step_log':[{'mean_comp_len':1}], 'elapsed_seconds':1}
    put(tmp_path/'samestack_gsm8k_cot_full.json',{'runs':[run,deepcopy(run)]})
    result=subprocess.run([sys.executable,'-O',str(SAME/'v1_recompute.py'),str(tmp_path)],text=True,capture_output=True)
    assert result.returncode!=0 and 'Duplicate experiment key' in result.stderr


def test_correctly_hash_bound_wrong_ci_is_rejected(same_fixture):
    root,gold,plan=same_fixture;replay=v2.analyze(root,gold,plan)
    shipped=deepcopy(replay);shipped['contrasts']['stored']['g8-ppo']['t_ci']=[.4,.5]
    artifact=root/'shipped-result.json';expected=put(artifact,shipped)
    assert sha(artifact.read_bytes())==expected  # Integrity succeeds; statistical replay must fail.
    with pytest.raises(gate.AuditError):gate.accept_publication(replay,json.loads(artifact.read_text()))
    replay['arms']['ppo']['item_disagreements_pre_total']=1
    with pytest.raises(gate.AuditError,match='pre independent regrading disagreement'):
        gate.accept_publication(replay,deepcopy(replay))


@pytest.mark.parametrize('bad',[r'\boxed{4.2,000}',r'\boxed{42.0,000}',r'\boxed{1,00}',r'\boxed{1,000,00}'])
def test_malformed_decimal_grouping_is_not_a_correct_reward(bad):
    assert v2.grade(bad,'4.2')[0]==0
    assert v2.grade(bad,'42')[0]==0


def test_valid_grouped_integer_and_fraction_remain_exact():
    assert v2.grade(r'\boxed{1,000.25}','1000.25')[0]==1
    assert v2.grade(r'\boxed{\frac{1,000}{4}}','250')[0]==1


def test_selected_unstarted_slot_cannot_hide_completed_receipt(tmp_path,monkeypatch):
    manifest,docs,binding,ref,seal=e12_fixture(tmp_path,monkeypatch)
    row=docs['attempt_ledger']['attempts'][0]
    row['run_id']='already-completed'
    row['receipt']=ref('done',{'schema_version':'e12-attempt-receipt-v2-candidate','binding':binding,'task_id':row['task_id'],'ordinal':row['ordinal'],'run_id':row['run_id'],'state':'completed'})
    seal()
    result=e12.validate(manifest,tmp_path)
    assert 'NOT_STARTED_CANNOT_HAVE_EXECUTION_EVIDENCE' in result['blockers']


@pytest.mark.parametrize('mutation',['terminal_coverage','terminal_score','negative_errors','error_partition'])
def test_auxiliary_terminal_counts_must_reconcile(mutation):
    report=empty_report();m=v1.measure('synthetic native',35,300,4)
    m.update(terminal_attempt_coverage=v1.ratio(40,300),terminal_attempt_success_rate_errors_as_zero=v1.ratio(4,40),patch_application_errors=3,empty_patch_errors=2)
    report['rows'][0]['portfolio']['measurements']=[m];report.update(replacement_score_lanes=['E1'],replacement_score_lane_count=1)
    v1.validate_report(report)
    if mutation=='terminal_coverage':m['terminal_attempt_coverage']=v1.ratio(1,300)
    elif mutation=='terminal_score':m['terminal_attempt_success_rate_errors_as_zero']=v1.ratio(99,100)
    elif mutation=='negative_errors':m['empty_patch_errors']=-1
    else:m['empty_patch_errors']=999
    with pytest.raises(v1.EvidenceError):v1.validate_report(report)


@pytest.mark.parametrize('raw',[b'{"value":NaN}',b'{"value":Infinity}',b'{"value":1e400}'])
def test_all_json_nonfinite_encodings_rejected(tmp_path,raw):
    p=tmp_path/'bad.json';p.write_bytes(raw)
    with pytest.raises(ValueError):v2.load(p)
    with pytest.raises(ValueError):v1.Reader.decode(raw)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,1.2])
def test_historical_v1_cli_rejects_invalid_stored_accuracy(tmp_path,bad):
    runs=[{'arm':arm,'seed':seed,'n_eval':200,'pre_correct':[0]*200,'post_correct':[0]*200,'heldout_post_acc':bad,'step_log':[{'mean_comp_len':1.}],'elapsed_seconds':1.} for arm in ['grpo_g8','grpo_g2','ppo'] for seed in [1,2]]
    (tmp_path/'samestack_gsm8k_cot_full.json').write_text(json.dumps({'runs':runs}))
    result=subprocess.run([sys.executable,str(SAME/'v1_recompute.py'),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0 and 'stored_mismatch_seeds' not in result.stdout
