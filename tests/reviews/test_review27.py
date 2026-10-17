import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
from _review27_contracts import exact_iso_g

@pytest.mark.parametrize('p,target',[(.5,.6),(.5,.8),(.1,.95),(.9,.6),(.01,.99)])
def test_iso_g_is_minimal_and_meets_target(p,target):
    g=exact_iso_g(p,target)
    assert 1-p**g-(1-p)**g >= target-1e-14
    assert g==1 or 1-p**(g-1)-(1-p)**(g-1) < target

@pytest.mark.parametrize('p,target',[(0,.6),(1,.6),(.5,1)])
def test_iso_g_infeasible(p,target):
    assert math.isinf(exact_iso_g(p,target))

def verifier(tmp_path):
    m=load('verify_claims_offline');m.BASE_DIR=tmp_path;m.EXPERIMENTS_DIR=tmp_path/'experiments';m.REPORTS_DIR=tmp_path/'reports/final'
    m.EXPERIMENTS_DIR.mkdir();m.REPORTS_DIR.mkdir(parents=True)
    return m

def test_checksum_is_recomputed(tmp_path):
    m=verifier(tmp_path);p=tmp_path/'evidence';p.write_text('actual')
    assert m.verify_checksum_manifest()[0] is False
    manifest=tmp_path/'SHA256SUMS.txt';manifest.write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  evidence\n')
    assert m.verify_checksum_manifest()[0] is True
    p.write_text('tampered')
    assert m.verify_checksum_manifest()[0] is False
    manifest.write_text('0'*64+'  ../escape\n')
    assert m.verify_checksum_manifest()[0] is False

def test_empty_evidence_never_passes(tmp_path):
    m=verifier(tmp_path);(m.EXPERIMENTS_DIR/'master_results.json').write_text('[]')
    for name,(description,check) in m.CHECKS.items():
        assert check()[0] is False, name

@pytest.mark.parametrize('value',[0,None,float('nan')])
def test_headline_missing_zero_nonfinite_fail(value,tmp_path):
    m=verifier(tmp_path)
    rows=[{'model':'Qwen/Qwen3-8B','method':algo,'last10_avg':value} for algo in ['grpo','ppo']]
    (m.EXPERIMENTS_DIR/'master_results.json').write_text(json.dumps(rows))
    assert not m.verify_qwen3_8b_headline_reward()[0]

def test_exact_model_required(tmp_path):
    m=verifier(tmp_path)
    rows=[{'model':'Qwen/Qwen3.5-4B','method':algo,'last10_avg':v} for algo,v in [('grpo',.344),('ppo',.225)]]
    p=m.EXPERIMENTS_DIR/'master_results.json';p.write_text(json.dumps(rows))
    assert not m.verify_qwen3_8b_headline_reward()[0]
    for row in rows:row['model']='Qwen/Qwen3-8B'
    p.write_text(json.dumps(rows));assert m.verify_qwen3_8b_headline_reward()[0]

def test_heldout_dedup_zero_and_missing_stats(tmp_path):
    m=verifier(tmp_path);root=m.REPORTS_DIR
    (root/'gsm8k_base_control_200.json').write_text(json.dumps({'correct':164,'total':200}))
    for seed,c in [(42,0),(137,170)]:
        (root/f'gsm8k_heldout_seed{seed}.json').write_text(json.dumps({'correct':c,'total':200}))
    ok,detail=m.verify_gsm8k_heldout()
    assert not ok and 'n_unique_seeds=2' in detail and 'mean=0.425000' in detail
    (root/'gsm8k_heldout_seed042.json').write_text(json.dumps({'correct':0,'total':200}))
    assert 'duplicate seed' in m.verify_gsm8k_heldout()[1]

@pytest.mark.parametrize('record',[{'step':1,'rewards':[0,1,0],'group_size':2},{'step':1,'rewards':[0,'bad',1,0],'group_size':2},{'step':1,'rewards':[0,float('nan')],'group_size':2},{'step':1,'rewards':[0,1]}])
def test_trl_never_trims_or_invents_groups(record):
    with pytest.raises(ValueError): load('zvf_compute_cross_framework').parse_trl(record)

def test_ragged_groups_remain_intact():
    m=load('zvf_compute_cross_framework')
    record={'step':1,'group_rewards':[[0,0],[0,0,1]]}
    row=m.compute_time_series([record],'tinker')[0]
    assert row['zvf']==.5 and row['batch_reward_mean']==.2
    assert row['group_sizes']==[2,3] and row['K'] is None
    with pytest.raises(ValueError):m.zvf([[0,float('inf')]])
    with pytest.raises(ValueError):m.zvf([])

def test_summary_reconstruction_cannot_pass_as_trace():
    m=load('scaling_law_iter45')
    assert m.fit_saturation(10,0,.5,.3,.5,.1,.1,.5,0)[2] is False

def test_schema_conflicts_nulls_and_semantic_aliases():
    m=load('validate_master_results_schema')
    for row in [{'run_id':'A','experiment_id':'B'},{'experiment_id':'B','run_id':'A'}]:
        with pytest.raises(ValueError):m.normalize_row(row)
    assert m.normalize_row({'final_reward':.1,'peak_reward':.8})['peak_reward']==.8
    assert m.normalize_row({'url':'https://example.org'})['source_files']==['https://example.org']
    assert m.validate_row(dict.fromkeys(m.CANONICAL_SCHEMA),1)

def test_extract_preserves_heading_and_idempotent_marker():
    m=load('weave_addendum')
    assert m._extract_between('##### 10.6.1 Held-out\nText A\n##### 10.6.2 Next\nB','10.6.1','10.6.2')=='##### 10.6.1 Held-out\nText A\n'
    text=m.MARKER+'\nreport'
    assert m.transform(text)==text

@pytest.mark.parametrize('script',['zvf_contrastive_yield','zvf_diagnostic'])
def test_self_test_never_loads_evidence_or_writes(monkeypatch,tmp_path,script):
    m=load(script)
    for name in dir(m):
        if name.startswith('load_'):
            monkeypatch.setattr(m,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError('loaded real evidence')))
    monkeypatch.setattr(sys,'argv',[script,'--self-test'])
    assert m.main()==0
    assert not list(tmp_path.iterdir())
