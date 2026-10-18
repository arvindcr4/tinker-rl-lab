"""Review17 P0 reproductions, using only synthetic observations."""
import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'platform_modal/scripts/p5p8'))
from review17_contract import paired_result, load_seed_metric


def test_pairing_and_rng():
    base = {str(i): i*.2 for i in range(5)}
    method = {k: v+.1 for k, v in reversed(list(base.items()))}
    r = paired_result(base, method)
    assert r['delta'] == pytest.approx(.1)
    assert r['ci_low'] == pytest.approx(.1)
    assert r['ci_high'] == pytest.approx(.1)
    for n in (2, 4):
        r = paired_result(dict.fromkeys(range(n), 0), {i: int(i == n-1) for i in range(n)})
        assert r['ci_low'] < r['ci_high']
    for a,b in [({},method), ({0:0},{1:1}), ([0,0],[0,1])]:
        with pytest.raises(ValueError):
            paired_result(a,b)


def test_source_semantics(tmp_path):
    p=tmp_path/'risk.tsv'
    p.write_text('method\tseed\tmean_zvf\ng\tagg\t.8\ng\t1\t.2\ng\t2\t.3\n')
    assert load_seed_metric(p,'mean_zvf') == {'g': {'1': .2, '2': .3}}
    with pytest.raises(ValueError, match='explicit source'):
        load_seed_metric(p,'mag_mean')
    p.write_text(p.read_text()+'g\t1\t.2\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_seed_metric(p,'mean_zvf')


def test_migration_preserves_unknown_method():
    mod=importlib.import_module('p6_iter134_patch_ci_shape')
    assert mod.migrate_method('welch') == dict(method='welch',n_boot=None,seed=None,ci_level=None,source=None)


def test_complete_result_updates_atomically():
    mod=importlib.import_module('p6_iter128_patch_registry')
    entry={'measured':[{'metric':'mean_zvf','panel':'zvf130_5seed','delta':99,'n':99,'significant':False}]}
    result=paired_result({1:0,2:0},{1:1,2:1})
    r=mod.proposed_entry(entry,dict(result=result,metric='mean_zvf',units='fraction',source_sha256='abc'))
    assert r['measured'][0]['delta']==1 and r['measured'][0]['n']==2
    assert r['measured'][0]['significant'] is True
    assert entry['measured'][0]['delta']==99
    assert mod.proposed_entry(r,dict(result=result,metric='mean_zvf',units='fraction',source_sha256='abc'))==r


def test_joint_identity_units_and_missing_ci(monkeypatch):
    mod=importlib.import_module('p6_controller_joint_extension')
    monkeypatch.setattr(mod,'METHODS',['grpo','gift']); monkeypatch.setattr(mod,'TAUS',[.03])
    rows=[dict(method=m,tau='.03',n_contrast='1',n_fired_steps='1',n_zvf_saved='0',rollout_saves='1',net_saves='1',g_total='8',cost_ratio='.875') for m in mod.METHODS]
    monkeypatch.setattr(mod,'load_joint_tsv',lambda:rows)
    r=mod.build_predictions()
    assert {p['method'] for p in r}=={'grpo','gift'}
    assert all(p['savings_per_rollout_pt']==.125 and p['ci_method'] is None for p in r)
    rows.append(rows[0])
    with pytest.raises(ValueError, match='Duplicate'):
        mod.build_predictions()
    rows.pop(); rows.pop()
    with pytest.raises(ValueError, match='Incomplete'):
        mod.build_predictions()


def test_validator_real_schema_and_empty_corpus(tmp_path, monkeypatch):
    mod=importlib.import_module('p6_iter130_schema_ci')
    bad=dict(record_type='variant_delta',id=7,name='x',base='grpo',deltas=[],citation='wrong')
    assert mod.valid_delta(bad, [])
    monkeypatch.setattr(mod,'ENTRIES',tmp_path/'missing')
    monkeypatch.setattr(mod,'OUT_DIR',tmp_path/'out')
    assert mod.main()==1
    for lo,hi in [(1,-1),(float('nan'),2),(0,float('inf'))]:
        assert mod.ci_overlaps_zero(dict(ci_low=lo,ci_high=hi)) is None


def test_missing_field_distinct_from_null(tmp_path):
    import json
    mod=importlib.import_module('p6_iter134_measured_field_completeness')
    p=tmp_path/'entry.json';p.write_text(json.dumps({'measured':[{'ci_low':None}]}))
    row=mod.audit_entry(p)[0]
    assert row['ci_low_state']=='NULL' and row['ci_high_state']=='MISSING'


def test_step_identity_and_sample_count(tmp_path,monkeypatch):
    mod=importlib.import_module('p6_iter106_claim_evidence_ledger')
    p=tmp_path/'n2.tsv';monkeypatch.setattr(mod,'N2',p)
    p.write_text('method\tstep\tx\n'+''.join(f'{m}\t{s}\t{int(m=="aero")}\n' for m in ['aero','grpo'] for s in reversed(range(30,40))))
    row=mod.compute_n2_delta('aero','x')
    assert row['n']==10 and row['delta']==1 and row['n_independent_runs']==1
    p.write_text('method\tstep\tx\naero\t30\t1\ngrpo\t30\t0\n')
    with pytest.raises(ValueError,match='Incomplete'):
        mod.compute_n2_delta('aero','x')


def test_crosspanel_reordered_steps_and_missing(monkeypatch):
    mod=importlib.import_module('p6_iter110_n2_zvf130_xpanel')
    monkeypatch.setattr(mod,'METHODS',['aero']);monkeypatch.setattr(mod,'N2_METRICS',['x'])
    data={'grpo':[dict(step=i,x=i) for i in range(40)],'aero':[dict(step=i,x=i+1) for i in reversed(range(40))]}
    assert mod.n2_panel(data)[0][2]==1
    data['aero'].pop()
    with pytest.raises(ValueError):
        mod.n2_panel(data)


def test_exact_method_matching(tmp_path):
    import json
    mod=importlib.import_module('p6_iter118_coverage_audit')
    p=tmp_path/'tinker_ngrpo_model.json';p.write_text(json.dumps(dict(id=p.stem,label_claimed='ngrpo')))
    assert not mod.match_stack('tinker','grpo',p)
    assert mod.match_stack('tinker','ngrpo',p)


def test_population_does_not_invent_run_metadata():
    mod=importlib.import_module('p6_iter138_missing_method_audit')
    row=dict(n_seeds=3,zvf_risk_mean=.4)
    stack=mod.make_stack('tool_use_example',row)
    delta=mod.make_delta('tool_use_example',row)
    assert stack['seeds'] is None
    assert stack['min_report']['sampler_backend']=={}
    assert stack['outcomes']['crossref_integrity_iter138']['point_estimate_matches_tsv'] is None
    assert stack['variant_deltas_applied'][0]['component']==delta['deltas'][0]['component']
    assert delta['measured']==[]


def test_closure_filters_planned_and_rejects_conflict():
    mod=importlib.import_module('p5_schema_bump_closure')
    entry={'variant_deltas_applied':[dict(delta_id='delta_dapo',status='planned'),dict(delta_id='delta_gspo',status='implemented')]}
    assert mod.closure_values(entry,['token_aggregation'])=={'token_aggregation':'sequence'}
    entry['variant_deltas_applied'][0]['status']='implemented'
    with pytest.raises(ValueError,match='Conflicting'):
        mod.closure_values(entry,['token_aggregation'])


def test_no_significance_quality_promotion():
    mod=importlib.import_module('p6_iter126_measured_evidence_tier')
    assert mod.tier(0,1,1)==mod.tier(100,100,10)=='QUALITY_UNASSESSED'
    mod=importlib.import_module('p6_iter114_robustness_ledger')
    assert mod.classify(dict(delta=0,ci_low=-.01,ci_high=.01,n=100))[0]=='DESCRIPTIVE_INTERVAL'
    assert mod.classify(dict(delta=99,ci_low=98,ci_high=100,n=100,significant=True))[0]=='DESCRIPTIVE_INTERVAL'
    assert mod.classify(dict(ci_low=float('nan'),ci_high=1))[0]=='INVALID_OR_MISSING'


def test_mag_repair_proposal_is_idempotent_and_read_only(tmp_path, monkeypatch):
    import json
    mod=importlib.import_module('p6_iter130_patch_stale_mag')
    monkeypatch.setattr(mod,'ROOT',tmp_path);monkeypatch.setattr(mod,'ENTRIES',tmp_path)
    source=tmp_path/'risk.tsv';source.write_text('explicit synthetic mag_mean observations')
    monkeypatch.setattr(mod,'SOURCE',source)
    path=tmp_path/'delta_x.json'
    path.write_text(json.dumps({'name':'x','measured':[{'metric':'mag_mean','panel':'zvf130_5seed','delta':999}]}))
    before=path.read_bytes(); data={'grpo':{'a':0,'b':0},'x':{'a':1,'b':2}}
    first=mod.patch_entry('delta_x',data)
    assert path.read_bytes()==before
    path.write_text(json.dumps(first))
    second=mod.patch_entry('delta_x',data)
    assert second==first and len(second['measured'])==1


def test_claim_main_fails_invalid_interval(tmp_path, monkeypatch):
    import json
    mod=importlib.import_module('p6_iter118_claim_validation')
    registry=tmp_path/'reg';(registry/'entries').mkdir(parents=True)
    monkeypatch.setattr(mod,'REGISTRY',registry);monkeypatch.setattr(mod,'RESULTS',tmp_path/'out')
    monkeypatch.setattr(sys,'argv',['audit'])
    claim=dict(metric='x',panel='p',predicted_sign='>0',rationale='test')
    row=dict(metric='x',panel='p',delta=1,ci_low=2,ci_high=-2)
    (registry/'entries/delta_x.json').write_text(json.dumps(dict(id='delta_x',expected_effects=[claim],measured=[row])))
    assert mod.main()==1
