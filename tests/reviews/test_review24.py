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
