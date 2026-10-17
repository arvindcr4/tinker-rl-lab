import sys
import random
import numpy as np
import pytest
from test_pro_review_19_regressions import functions, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
from _review_calibration import crossfit, stacked_crossfit, isotonic_predict


def test_identity_oof_alignment():
    scores=[.1*i for i in range(1,11)]
    ids=[8,3,1,4,7,0,9,6,2,5]
    assert crossfit(scores,[0,1]*5,ids,'identity') == [scores[i] for i in ids]


def test_isotonic_monotonic_ties_and_bounds():
    assert isotonic_predict([.1,.9],[0,1],[.1,.9]) == [0,1]
    assert isotonic_predict([.1,.9],[1,0],[.1,.9]) == [.5,.5]
    p=isotonic_predict([.1,.1,.9],[0,1,1],[-1,.1,.5,.9,2])
    assert p==sorted(p) and p[0]==.5 and p[-1]==1


@pytest.mark.parametrize('method',['isotonic','rate'])
def test_held_label_cannot_change_own_prediction(method):
    scores=np.linspace(.01,.99,20).tolist(); labels=[0,1]*10
    base=crossfit(scores,labels,list(range(20)),method)
    for i in range(20):
        changed=labels[:];changed[i]=1-changed[i]
        assert crossfit(scores,changed,list(range(20)),method)[i] == base[i]


def test_stacked_outer_label_exclusion():
    scores=np.linspace(.01,.99,20).tolist(); labels=[0,1]*10; cohorts=[0]*10+[1]*10
    base=stacked_crossfit(scores,labels,cohorts)
    for i in range(20):
        changed=labels[:];changed[i]=1-changed[i]
        assert stacked_crossfit(scores,changed,cohorts)[i] == base[i]


def test_escalation_precision_not_squared():
    ns=functions('p8_iter156_disagreement_escalation',np=np)
    point,lo,hi=ns['bootstrap_escalation'](10,90,1000,n_boot=2000)
    assert point==.1 and lo < .1 < hi
    assert np.isnan(ns['bootstrap_escalation'](0,0,100)[0])


def test_perturbation_policy_paired_identity_and_bounded_loss():
    ns=functions('p8_iter128_feature_perturbation',np=np,G_THR=.001,K_PCT=50)
    y=np.array([1,1,0,0]); a=np.array([.9,.8,.2,.1]); b=1-a
    assert ns['boot_ci_paired'](a,a,a,a,y,400,42)[:3] == (0,0,0)
    pt,lo,hi,_=ns['boot_ci_paired'](a,a,b,b,y,400,42)
    assert pt == -1 and -1 <= lo <= hi <= 1


def test_routing_changes_outcome_and_requires_scores():
    ns=functions('p8_iter112_cost_cis_realistic_rates',np=np,K_PCT=50,G_THR=.2,W_ABS=2,COST_XGB=.001,COST_LLM=.01)
    raw=np.array([.9,.8,.2,.1]); full=np.array([.1,.99,.3,.2]); y=np.array([0,1,0,0])
    with pytest.raises(ValueError,match='second_stage'):
        ns['apply_rules'](raw,y,25)
    r=ns['apply_rules'](raw,y,25,full)
    assert r['xgb_only'][1]==0 and r['absolute_band'][1]==1
    assert r['absolute_band'][0]==4
    assert r['absolute_band'][2]==pytest.approx(.011)


def test_low_threshold_candidate_and_row_noise():
    ns=functions('p8_iter188_cost_asymmetric_transfer',np=np,N_TH=100)
    _,costs,_=ns['cost_curve'](np.array([.001,.002]),np.array([1,0]),1000)
    assert min(costs)==.5
    ns=functions('p8_iter88_noise_cost_frontier',np=np)
    out=ns['add_noise'](np.zeros((100,1)),[0],1,np.ones(1),random.Random(42))
    assert len(np.unique(out[:,0]))==100


def test_tied_auc_and_ap_are_permutation_invariant():
    for name,fn in [('p8_single_sensor_ablation','auc_roc'),('p8_single_sensor_seed_stability','auc_roc'),('p8_iter192_vmean_decile_audit','auc_quick')]:
        ns=functions(name,np=np)
        assert ns[fn]([.5,.5],[1,0]) == .5
    ns=functions('p8_iter140_sensor_feat_realistic',np=np,G_THR=.001)
    assert ns['pr_auc_at_mask'](np.array([.5,.5]),np.array([1,0]),1) == .5


def test_calibration_slope_binomial_contract():
    ns=functions('p8_iter180_calibration_slope_curvature',np=np)
    y=np.array([1]+[0]*9+[0]+[1]*9)
    slope,intercept=ns['reliability_slope'](y,np.array([.1]*10+[.9]*10))
    assert slope==pytest.approx(1,abs=1e-5) and intercept==pytest.approx(0,abs=1e-5)
    assert np.isnan(ns['reliability_slope'](np.array([0,1]),np.array([.5,.5]))[0])


def test_input_inference_cost_and_uniform_alert_fairness():
    ns=functions('p8_single_sensor_ablation',np=np)
    assert ns['cost_per_caught']([1]+[0]*99,[1]+[0]*99,2,.01)[0]==1
    ns=functions('p8_subgroup_alert_fairness',np=np,N_BOOT=100,BOOT_SEED=42)
    rows=[dict(n_alert_bin=2,recall_bin=.5,precision_bin=.5,p_bin=.02,alert_rate_bin=.02,n_pos_bin=2,n_bin=100)]*5
    assert ns['heterogeneity_summary'](rows)['l1_alert_rate_gap_vs_global']==0


def test_schema_failure_preserves_original(tmp_path):
    import json
    import jsonschema
    entries=tmp_path/'entries'; entries.mkdir()
    schema={'type':'object','required':['required_key'],'$defs':{'stack_record':{'properties':{'variant_deltas_applied':{'items':{'properties':{'delta_id':{'type':'string'}}}}}}}}
    target=tmp_path/'schema.json'; original=json.dumps(schema); target.write_text(original)
    (entries/'delta_a.json').write_text('{}')
    ns=functions('regenerate_schema_delta_enum',json=json,jsonschema=jsonschema,SCH=target,ENT=entries,sys=sys)
    with pytest.raises(SystemExit): ns['main']()
    assert target.read_text()==original
    ns['jsonschema']=None
    with pytest.raises(RuntimeError): ns['main']()
    assert target.read_text()==original
