import numpy as np
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

from _review24_statistics import conditional_ccf, first_crossing, stable_seed

def test_dry_run_cannot_enter_training(monkeypatch):
    m=load('hyperparam_sensitivity')
    monkeypatch.setattr(m,'HAS_TORCH',True)
    monkeypatch.setattr(m,'set_global_seed',lambda *_: (_ for _ in ()).throw(AssertionError('training entered')))
    assert 0 <= m.train_and_evaluate({},42,dry_run=True) <= 1
    monkeypatch.setattr(m,'HAS_TORCH',False)
    with pytest.raises(RuntimeError): m.train_and_evaluate({},42)

@pytest.mark.parametrize('x,y,target,result',[([2,0,1],[.8,.2,.5],.5,1),([1,0],[.5,.5],.5,0),([0,1,2],[.2,.8,.3],.5,.5),([0,1],[.2,.3],.7,None)])
def test_frontier_crossings(x,y,target,result):
    assert first_crossing(x,y,target) == (None if result is None else pytest.approx(result))

def test_iso_cost_interpolates():
    m=load('group_size_iter91')
    result=m.iso_cost([{'G':4,'T_M':100,'acc':.8},{'G':4,'T_M':1,'acc':.2}], [.5])[0]
    assert result['T_needed_M_star'] == pytest.approx(10)

def test_missing_cross_cells():
    m=load('group_size_iter87')
    with pytest.raises(ValueError,match='common'): m.compute_crossover({(1,4):(.3,.2,.4,3),(2,32):(.4,.3,.5,3)},4,32)

def test_permutation_preserves_actual_group_labels(monkeypatch):
    m=load('group_size_iter99'); observed=[]
    monkeypatch.setattr(m,'_between_group_stat',lambda pool,labels,pairs: observed.append(tuple(labels)) or 1)
    labels=[4,4,4,16,16]
    assert m._perm_pvalue_curves([[1,2]]*5, labels, n_perm=10) == 1
    assert observed[0] == tuple(labels)
    assert all(sorted(x)==sorted(labels) for x in observed)

def test_bootstrap_refits_same_estimator(monkeypatch):
    m=load('group_size_iter75');calls=[]
    monkeypatch.setattr(m,'fit_power_law',lambda g,a,lo,hi: calls.append(a) or (.8,.2,float(np.mean(a)),0))
    m.bootstrap_ci([2,4,8],[.3,.4,.5],[.2,.3,.4],[.4,.5,.6],n_boot=7)
    assert len(calls)==8
    assert m.test_null(.5,(.4,.6),0)[0] is False

def test_quantile_mask_preserves_original_time():
    x=np.arange(12,dtype=float); mask=np.arange(12)%2==0
    cc,counts=conditional_ccf(x,x,mask,2)
    assert counts == [5,0,6,0,5]
    assert np.isnan(cc[1]) and np.isnan(cc[3])
    assert cc[0] == pytest.approx(1)

def test_stable_distinct_seeds():
    assert stable_seed(1,'task',0,'fwd') == stable_seed(1,'task',0,'fwd')
    assert stable_seed(1,'task',0,'fwd') != stable_seed(2,'task',0,'fwd')

def test_ceiling_failures_remain_counted(monkeypatch):
    m=load('group_size_iter95')
    monkeypatch.setattr(m,'curve_fit',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('optimizer failed')))
    rows=[{'G':4,'T':t,'acc':.5,'lo':.4,'hi':.6} for t in [1,2,3]]
    result=m.fit_ceiling(4,rows=rows)
    assert result['status']=='failed' and result['reason']=='optimizer failed'
    assert np.isnan(result['a'])
    with pytest.raises(ValueError,match='point fit failed'):m.analyse(rows,B=2)

def test_ceiling_sensitivity_fixed_schema_and_no_claim():
    m=load('group_size_iter95')
    rows=[{'G':g,'T':t,'acc':a*t/(t+2),'lo':a*t/(t+2)-.001,'hi':a*t/(t+2)+.001} for g,a in [(4,.6),(32,.8)] for t in [1,4,16,64]]
    result=m.analyse(rows,B=8)
    assert result['ratio']['estimate']==pytest.approx(.75,abs=1e-4)
    assert result['ratio']['successful']==8
    assert result['sampling']=='independent_normal_summary_sensitivity'

def test_fevd_one_step_is_contemporaneous():
    m=load('length_bias_iter100');rng=np.random.default_rng(19)
    states=np.zeros((500,2));noise=rng.normal(size=(500,2))
    K=np.array([[.2,.4],[0,.1]])
    for i in range(1,len(states)):states[i]=K@states[i-1]+noise[i]
    result=m.fit_var1(states[:,0],states[:,1])
    assert result['fevd_L_from_R_h1']==pytest.approx(0,abs=1e-14)
    assert result['fevd_L_from_R_h4']>.01

def test_missing_ccf_pairs_have_explicit_coverage():
    from _review24_statistics import paired_available
    result=paired_available([1,np.nan,2],[2,3,3],B=10,seed=1)
    assert result['n']==2 and result['n_excluded']==1 and result['delta']==1
    result=paired_available([np.nan],[1],B=10,seed=1)
    assert result['n']==0 and np.isnan(result['p'])
