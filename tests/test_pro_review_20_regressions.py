import sys
import json
from collections import Counter
import numpy as np
import pytest
from test_pro_review_19_regressions import functions, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
from _review_policy import optimal_threshold, alert_threshold


def test_beta_binomial_normalization_and_contrast():
    import math
    lb=lambda a,b:math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    ns=functions('p7_iter203_bucket_gstar', lbeta=lb, TARGET_GU=.85, G_CANDIDATES=(2,4,8,12,16,24,32))
    assert ns['beta_binom_cdf'](0,1,1,1) == pytest.approx(.5)
    assert ns['beta_binom_cdf'](8,8,5,5) == pytest.approx(1)
    assert ns['gu_post'](.5,8,8) == pytest.approx(.9592760181)
    assert ns['optimal_gstar_post'](4, target=.5) == 4


def test_cannot_manufacture_contrast_across_prompts():
    ns=functions('p7_iter203_empirical_gprime')
    rows=[[0]*8,[1]*8]*8
    assert ns['emp_gu_at_gpr'](rows,8) == 0
    with pytest.raises(ValueError,match='same-prompt'):
        ns['emp_gu_at_gpr'](rows,16)


def test_saturation_summary_matches_all_zero_steps(tmp_path):
    rows=[dict(step=i,rewards=[[0]*8]*16) for i in range(40)]
    (tmp_path/'synthetic_s0_tensors.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ns=functions('p7_satband_per_prompt', json=json, Counter=Counter, METHODS=['synthetic'], N2=tmp_path, OUT=tmp_path, SAT_BAND=.9, TAU=.7, DELTA=.2)
    ns['main']()
    import csv
    with (tmp_path/'p7_satband_per_prompt_summary.tsv').open() as f:
        row=next(csv.DictReader(f,delimiter='\t'))
    for k in ['zvf_baseline_mean','zvf_zvf70_mean','zvf_dual_mean','zvf_hybrid_mean']:
        assert float(row[k]) == 1


def test_cost_policy_direction_and_ties():
    assert alert_threshold(.5,100) == .005
    r=optimal_threshold([.5,.5],[1,0],.5,100)
    assert r['kstar']==2 and r['cost_per_dec']==.5
    assert sum(s>=r['tau'] for s in [.5,.5])==r['kstar']
    r=optimal_threshold([.9,.1],[0,1],5,5)
    assert r['kstar']==0 and r['cost_per_dec']==2.5


def test_hybrid_uses_paid_second_stage():
    ns=functions('p8_cost_adjusted_curve', np=np,RANDOM_SEED=42,N_BOOT=100,BOOT_SEED=42,COST_REVIEW_PER_ROW=1,COST_XGB_PER_ROW=.001,COST_LLM_PER_ROW=.01)
    raw=np.array([.9,.8]); y=np.array([0,1])
    a=ns['hybrid_op_point'](raw,raw,y,.5,1)
    b=ns['hybrid_op_point'](raw,np.array([.1,.99]),y,.5,1)
    assert a['tp']==0 and b['tp']==1


def test_actual_cost_denominator_and_zero_catches():
    ns=functions('p8_alert_volume_pareto',np=np,C_INV=.5,C_SENSE=.0035,N_BOOT=100,BOOT_SEED=42)
    m=dict(n_rows=100,tp=1,fp=9,fn=1,alert_rate=.1)
    assert ns['compute_dollar_at_k'](m,'24full')[0] == pytest.approx(1.0535)
    m['tp']=0
    assert ns['compute_dollar_at_k'](m,'24full')[1] is None
