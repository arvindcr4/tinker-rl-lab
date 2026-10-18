"""Synthetic-only review27-v3 diagnostic and evidence-contract regressions."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'platform_modal/scripts'
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table(path):
    return list(csv.DictReader((line for line in path.read_text().splitlines() if not line.startswith('#')), delimiter='\t'))


def observation(**kw):
    return dict(experiment='study', model='model', task='task', group_size=8,
                seed=1, algorithm='grpo', mean_zvf=.2, peak=.8, last10_avg=.7, **kw)


def test_condition_aggregate_and_algorithm_isolation(tmp_path):
    m=load('zvf_diagnostic')
    a=observation(); b=dict(a,seed=2); agg=dict(a,seed='agg',mean_zvf=.9)
    other=dict(a,algorithm='drgrpo')
    rows=m.write_by_library([a,b,agg,other],tmp_path/'lib.tsv')['rows']
    assert len(rows)==2
    grpo=next(r for r in rows if r['algorithm']=='grpo')
    assert grpo['n_rows']==2 and grpo['n_seeds']==2 and grpo['mean_zvf']==.2
    m.write_summary([other],tmp_path/'summary.tsv')
    exported=table(tmp_path/'summary.tsv')[0]
    assert exported['algorithm']=='drgrpo' and exported['failure_label']=='converged'


def test_collapse_denominators_and_unknown(tmp_path):
    m=load('zvf_diagnostic')
    a=observation(collapse_steps=1,observed_steps=30)
    b=dict(a,seed=2,peak=None,last10_avg=None,collapse_steps=0,observed_steps=10)
    r=m.write_by_library([a,b],tmp_path/'lib.tsv')['rows'][0]
    assert r['per_step_collapse_rate']==1/40
    assert r['run_ever_collapsed_rate']==.5
    assert r['n_classified']==1 and r['converged_rate']==1


def test_complete_cases_not_mixed_between_seeds(tmp_path):
    m=load('zvf_diagnostic')
    a=observation(); rows=[dict(a,mean_zvf=None),dict(a,seed=2,last10_avg=None)]
    result=m.write_correlation(rows,tmp_path/'corr.tsv')
    assert result['n_pooled_rows']==0
    for record in table(tmp_path/'corr.tsv'):
        assert record['n_pooled_rows']=='0' and record['ci_lo']=='NA'


def test_static_sampling_no_training_metrics(tmp_path):
    m=load('zvf_diagnostic');m.RESULTS=tmp_path;m.REPO_ROOT=tmp_path
    summary=dict(seeds=[42],zvf_per_seed=[.5],sampling_accuracy_per_seed=[.6],group_size=8,n_seeds=1,mean_zvf=.5,sampling_accuracy_mean=.6,n_problems_total=2)
    (tmp_path/'tinker_gsm8k_zvf_summary.json').write_text(json.dumps(summary))
    (tmp_path/'tinker_gsm8k_zvf_s42.json').write_text(json.dumps({'per_problem':[{'zvf':0},{'zvf':1}]}))
    rows=m.load_tinker_gsm8k()
    assert all(r['peak'] is None and r['last10_avg'] is None and r['n_steps'] is None for r in rows)
    assert rows[0]['sampling_accuracy']==.6 and m.classify(rows[0])=='unknown'
    assert len(m.independent_rows(rows))==1


def test_sweep_se_not_zvf_or_peak(tmp_path):
    m=load('zvf_diagnostic');m.RESULTS=tmp_path
    (tmp_path/'groupsize_zvf_sweep.tsv').write_text('G\tn\ta\tse\tlast\tzvf\ttrain\n8\t3\t.7\t.1\t.5\t.4\t.6\n')
    row=m.load_groupsize_sweep()[0]
    assert row['min_zvf'] is None and row['max_zvf'] is None and row['peak'] is None
    assert row['heldout_accuracy_se']==.1


def test_residuals_never_infer_dependence(tmp_path):
    m=load('zvf_antiherding_elevation')
    for sign in [-1,1]:
        rows=[dict(source='synthetic',delta_div=sign*.1) for _ in range(20)]
        result=m.write_falsification_tsv(rows,tmp_path/'res.tsv')['synthetic']
        assert result['verdict'].startswith('UNIDENTIFIED') and math.isnan(result['ci_lo'])
    rows=[dict(source='synthetic',p_x=1.,delta_div=0.,G=8,Y_obs=0.,zvf_obs=1.)]
    m.write_empirical_isog_tsv(rows,tmp_path/'grid.tsv')
    assert all(r['status']=='not_observed_feasible' for r in table(tmp_path/'grid.tsv'))


def canonical():
    return dict(run_id='r',paper_claim_ids=[],model='m',model_short='m',method='grpo',platform='p',task='t',split='test',seed=0,steps_completed=2,group_size=8,learning_rate=.01,metric_type='reward',peak_reward=1.,last10_avg=.5,reward_trace=[0.,1.],status='complete',source_files=['trace.json'])


def test_schema_validation_never_migrates(tmp_path):
    m=load('validate_master_results_schema');row=canonical();row['experiment_id']=row.pop('run_id')
    src=tmp_path/'legacy.json';src.write_text(json.dumps([row]));original=src.read_bytes()
    assert m.main(['validate','--input',str(src)])==1
    dst=tmp_path/'migrated.json'
    assert m.main(['migrate','--input',str(src),'--output',str(dst)])==0
    assert src.read_bytes()==original and 'run_id' in json.loads(dst.read_text())[0]
    assert m.main(['migrate','--input',str(src),'--output',str(dst)])==1


@pytest.mark.parametrize('key,value',[('seed',True),('group_size',0),('steps_completed',-1),('learning_rate',float('inf')),('source_files',[])])
def test_schema_rejects_invalid_fields(key,value):
    m=load('validate_master_results_schema');r=canonical();r[key]=value
    assert m.validate_row(r,1)


def test_schema_explicit_unavailable_contract():
    m=load('validate_master_results_schema');r=canonical();r.update(status='unavailable',steps_completed=0,peak_reward=None,last10_avg=None,reward_trace=None)
    assert not m.validate_row(r,1)
    r['peak_reward']=0.
    assert m.validate_row(r,1)


def claim_row(run_id='r',**kw):
    return dict(run_id=run_id,model='m',task='t',split='test',stack='trl',algorithm='grpo',seed=0,protocol_id='frozen-test-fixture',**kw)


def evidence(tmp_path,claim,rows):
    m=load('verify_claims_offline');m.EXPERIMENTS_DIR=tmp_path
    root=tmp_path/'verification_v3';root.mkdir(exist_ok=True)
    p=root/(claim+'.json');p.write_text(json.dumps(dict(schema_version='review27-v3',protocol_id='frozen-test-fixture',expected_run_ids=[r['run_id'] for r in rows],rows=rows)))
    return m,p


def test_verifier_recomputes_confusion_and_rejects_duplicate(tmp_path):
    rows=[claim_row(str(i),predicted_collapse=i<2,observed_collapse=i<2) for i in range(3)]
    m,p=evidence(tmp_path,'zvf_triage',rows)
    assert m.verify_zvf_triage()[0]
    rows[2]['predicted_collapse']=True
    m,p=evidence(tmp_path,'zvf_triage',rows)
    assert not m.verify_zvf_triage()[0]
    rows[2]['run_id']='0';m,p=evidence(tmp_path,'zvf_triage',rows)
    assert not m.verify_zvf_triage()[0]


def test_verifier_trl_and_group_size(tmp_path):
    m,_=evidence(tmp_path,'trl_baseline',[claim_row(last10_avg=.734)])
    assert m.verify_trl_baseline()[0]
    rows=[claim_row(str(g),group_size=g,last10_avg=.844 if g==8 else .5) for g in [4,8]]
    m,_=evidence(tmp_path,'group_size',rows)
    assert m.verify_group_size()[0]
    rows[1]['split']='different';m,_=evidence(tmp_path,'group_size',rows)
    assert not m.verify_group_size()[0]


def test_verifier_saturation_recomputes_predictions(tmp_path):
    rows=[claim_row(steps=[0,1,2],rewards=[0,.5,1.],initial=0.,asymptote=1.,rate=1.,trace_provenance='measured',fit_family='exponential_saturation')]
    m,_=evidence(tmp_path,'saturation_fit',rows)
    ok,detail=m.verify_saturation_fit()
    assert not ok and 'individual_R2=' in detail
    rows[0]['trace_provenance']='reconstructed';m,_=evidence(tmp_path,'saturation_fit',rows)
    assert 'actual measured' in m.verify_saturation_fit()[1]


def test_verifier_correlation_constant_unavailable(tmp_path):
    rows=[claim_row(str(i),mean_zvf=.2,last10_avg=i/3,independent_unit_id=str(i)) for i in range(3)]
    m,_=evidence(tmp_path,'zvf_correlation',rows)
    assert not m.verify_zvf_correlation()[0]


def test_weave_anchor_uniqueness_and_version_updates(monkeypatch):
    m=load('weave_addendum')
    with pytest.raises(ValueError):m.insert_before('## X\n## X\n',r'^## X','text')
    with pytest.raises(ValueError):m.append_to_section('## B\n## A\n',r'^## A',r'^## B','text')
    report='\n'.join(['## 3. Methodology','### 4.2 Tools','### 4.3 Next','### 4.5.8 Stack','### 4.6 Next','### 5.8 Scope','### 5.9 Next','### 5.12 ZVF','### 5.13 Next','### 5.16 Evidence','## 6. Summary of Findings','### 7.1 Gaps'])+'\n'
    fragments={name:'### 10 Title\nBody\n' for name in m.FRAGMENT_NAMES}
    fragments['08-related-work-integration.md']='### 10.8 Related\n#### 10.8.1 A\nA\n#### 10.8.2 B\nB\n'
    fragments['06-heldout-base-instruct.md']='### 10.6 Heldout\n#### 10.6.1 A\nA\n#### 10.6.2 B\nB\n'
    monkeypatch.setattr(m,'read_fragment',lambda name:fragments[name])
    first=m.transform(report)
    assert m.transform(first)==first
    fragments['01-zvf-formalization.md']+='New content\n'
    updated=m.transform(first)
    assert updated.count('New content')==1 and m.transform(updated)==updated
    assert updated.count('### 5.17 ')==1


def test_correlation_stratifies_incompatible_metrics(tmp_path):
    m=load('zvf_diagnostic')
    a=observation(metric_definition='training_reward')
    b=dict(a,seed=2,metric_definition='heldout_accuracy')
    result=m.write_correlation([a,b],tmp_path/'corr.tsv')
    assert len(result['rows'])==8 and result['n_pooled_rows'] is None
    assert {r['metric_definition'] for r in result['rows']}=={'training_reward','heldout_accuracy'}


def test_matched_stack_reversal_and_incomplete_pairs(tmp_path):
    rows=[]
    for model,values in [('a',(.7,.5)),('b',(.4,.8))]:
        for algo,value in zip(['grpo','ppo'],values):
            r=claim_row(model+algo,last10_avg=value);r.update(model=model,algorithm=algo)
            rows.append(r)
    m,_=evidence(tmp_path,'stack_sensitivity',rows)
    assert m.verify_stack_sensitivity()[0]
    rows.pop();m,_=evidence(tmp_path,'stack_sensitivity',rows)
    assert not m.verify_stack_sensitivity()[0]


def test_duplicate_observed_steps_rejected(tmp_path):
    m=load('zvf_diagnostic');m.RESULTS=tmp_path
    (tmp_path/'variance_mitigation.tsv').write_text('method\tseed\tstep\tzvf\treward_mean\theldout_acc\tcollapse\ngrpo\t0\t0\t.5\t.2\t.3\t0\ngrpo\t0\t0\t.5\t.2\t.3\t0\n')
    with pytest.raises(ValueError,match='duplicate step'):m.load_variance_mitigation()


def test_decomposition_rejects_duplicate_and_nonfinite(tmp_path):
    m=load('zvf_antiherding_elevation');p=tmp_path/'data.tsv'
    row='sourceA\t1\tproblem1\t8\t.5\t0\t.0078125\t.0078125\t1\n'
    p.write_text(row+row)
    with pytest.raises(ValueError,match='duplicate'):m.load_decomposition(p)
    p.write_text(row.replace('.0078125','nan'))
    with pytest.raises(ValueError,match='nonfinite'):m.load_decomposition(p)
