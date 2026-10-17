import math
import pytest
from test_review23 import load

def test_seed_observations_are_distinct_and_reject_duplicates():
    m=load('length_bias_iter136_fig')
    rows=[{'task':'t','algo':'grpo','seed':s,'value':v} for s,v in [(1,.1),(2,.9)]]
    result=m.seed_observations(rows, {'h':('value','smaller')})
    assert result[('t','grpo',1,'h')] == .1
    assert result[('t','grpo',2,'h')] == .9
    with pytest.raises(ValueError): m.seed_observations(rows+rows, {'h':('value','smaller')})

def test_separate_output_grains_and_sparse_row(tmp_path):
    m=load('length_bias_iter56'); m.RES=tmp_path
    m.write_grains('cumulative_tax',[{'seed':1,'tax':0},{'seed':2,'tax':1,'R_first':.3}],[{'metric':'tax','n_pairs':2,'ci_lo':0}])
    run=(tmp_path/'length_bias_iter56_cumulative_tax_per_run_v2.tsv').read_text()
    comparison=(tmp_path/'length_bias_iter56_cumulative_tax_comparison_v2.tsv').read_text()
    assert 'R_first' in run and 'n_pairs' not in run
    assert 'n_pairs' in comparison and 'seed' not in comparison
    assert not list(tmp_path.glob('*.tmp'))

@pytest.mark.parametrize('a,b,expected',[([1,1],[2,2],-1),([2,2],[1,1],1),([1,1],[1,1],0)])
def test_signed_area(a,b,expected):
    assert load('length_bias_iter68').trajectory_auc(a,b) == expected

@pytest.mark.parametrize('reward,identified', [([4,1,0,1,4],False), ([-4,-1,0,-1,-4],True), ([1,1,1,1,1],False)])
def test_quadratic_requires_negative_curvature(reward,identified):
    fit=load('length_bias_iter60').fit_quadratic_R_of_L(reward,[-2,-1,0,1,2])
    assert fit['optimum_identified'] is identified
    assert math.isfinite(fit['L_star']) is identified

@pytest.mark.parametrize('z',[float('nan'),float('inf'),-.1,1.1])
def test_missing_zvf_not_high(z):
    assert load('length_bias_iter52')._zvf_bin(z) == 'unavailable'
