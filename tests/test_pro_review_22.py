"""Synthetic regression tests for the partial review22."""
import importlib
import math
import sys
from pathlib import Path

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'platform_modal/scripts/p5p8'
sys.path.insert(0,str(SCRIPTS))
from review22_contract import bootstrap_paired_diff, density_ci, ratio_ci, highest_density, score_gradient_selection


def test_bootstrap_resamples_full_vectors():
    point,lo,hi,n=bootstrap_paired_diff([0,0,0,100],[0]*4,10000,17)
    assert point==25 and n==4 and lo==0 and hi<100


@pytest.mark.parametrize('a,b', [([],[]),([1],[1,2]),([math.nan],[0]),([math.inf],[0])])
def test_bootstrap_invalid(a,b):
    with pytest.raises(ValueError): bootstrap_paired_diff(a,b)


def test_explicit_prompt_identity_and_steps_required():
    m=importlib.import_module('registry_validate')
    rows={(arm,0,step):{'b':.5,'a':.1} for arm in ['grpo','aero','gift','areal'] for step in [0,1]}
    out=m.measured_per_prompt(rows)
    assert all(r['paired_delta']==0 and r['ci_lo'] is None and r['n_seeds']==1 for r in out)
    rows[('aero',0,1)]={'different':.5,'a':.1}
    with pytest.raises(ValueError): m.measured_per_prompt(rows)
    with pytest.raises(ValueError): m.measured_per_prompt({('grpo',0,0):[.1,.5]})


def test_seed_not_steps_is_inference_unit():
    m=importlib.import_module('registry_validate')
    fields=['reward_mean','zvf','loss','mean_len','cv_len','frac_all_zero','frac_all_one','pcd','larq']
    rows={(arm,0,step):dict.fromkeys(fields,step/20) for arm in ['grpo','aero','gift','areal'] for step in range(20)}
    result=m.measured_deltas(rows)
    assert all(r['n_seeds']==1 and r['n_steps']==10 and r['ci_lo'] is None for r in result)
    del rows[('aero',0,1)]
    with pytest.raises(ValueError):m.measured_deltas(rows)


def test_sorted_gradient_indices_and_original_mask():
    order,g,top,fire=score_gradient_selection([.1,.9,.8,.2],2,.2)
    assert order.tolist()==[1,2,3,0]
    assert top.tolist()==pytest.approx([1.,.1])
    assert fire.tolist()==[False,False,True,False]


def test_interval_decision_not_practical_threshold():
    r=ratio_ci(density_ci(500,1000),density_ci(200,1000))
    assert r['excludes_1.0'] and not r['outside_practical_band']
    assert not r['confirmatory']
    zero=ratio_ci(density_ci(0,10),density_ci(0,10))
    assert math.isnan(zero['ratio']) and math.isinf(zero['hi']) and not zero['excludes_1.0']


@pytest.mark.parametrize('counts',[(-1,5),(6,5),(0,0),(1.2,5)])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):density_ci(*counts)


def test_domain_id_ranking():
    assert highest_density({'D1':.1,'D4':.9})
    assert not highest_density({'D1':.9,'D4':.1})


def test_simulation_cannot_be_measured_domain():
    m=importlib.import_module('synth_iter160_twelve_domain_density')
    with pytest.raises(ValueError,match='Raw rollout'):m.compute_d12({('a',0):.5},.05)
    r=m.compute_d12({('a',0):.5},.05,simulation=True)
    assert r[0]['evidence_kind']=='model_based_bernoulli_simulation'


@pytest.mark.parametrize('name',['synth_iter108_invariance','synth_iter124_three_domain_density','synth_iter132_four_domain_density','synth_iter136_five_domain_density','synth_iter140_six_domain_density','synth_iter152_ten_domain_density','synth_iter160_twelve_domain_density','synth_iter180_d17_paper_reproducibility'])
def test_unsupported_publication_fails_before_read_or_write(name):
    with pytest.raises(ValueError,match='Publication blocked'):importlib.import_module(name).main()
