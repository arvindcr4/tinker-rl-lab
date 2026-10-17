"""Synthetic CPU-only regressions for the seven visible review-16 findings."""
import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'platform_modal/scripts/p5p8'
sys.path.insert(0, str(SCRIPTS))

def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schedule_charges_each_step_once():
    m = load('controller_counterfactual')
    for triggers in itertools.product([0., 1.], repeat=3):
        steps = [dict(step=i, zvf=z, pcd=0., per_p=[0.] * (i+1)) for i,z in enumerate(triggers)]
        rows = m.triage_schedule(steps, .5)
        assert [r['g_applied'] for r in rows] == [8] + [16 if z else 8 for z in triggers[:-1]]
        assert sum(r['rollout_cost'] for r in rows) == sum((i+1)*g for i,g in enumerate([8]+[16 if z else 8 for z in triggers[:-1]]))


def test_recovery_is_not_fabricated():
    m = load('controller_counterfactual')
    assert m.would_save([[0]*8, [1]*8, [0,1]*4], 16) == (None,None)
    assert not hasattr(m, 'oracle_decide')


def test_zvf_never_uses_accuracy_se():
    m = load('p5_headline_cis')
    rows = [dict(G=g, n_seeds=3, mean_zvf=z, heldout_acc_se=999) for g,z in [(2,.9),(4,.5),(8,.6),(16,.1)]]
    out = m.headline_h1_zvf_monotone(rows)
    assert all(r['ci_lo'] == 'n/a' and r['verdict'] == 'DESCRIPTIVE_ONLY' for r in out)
    assert 'False' in out[-1]['note']


def test_standard_omega_and_no_additive_claim():
    m = load('mega_eta2')
    rows = [dict(axis=a,y=y) for a,ys in [('a',[0.,2.]),('b',[4.,6.])] for y in ys]
    assert m.omega2_by_axis(rows,'axis','y') == pytest.approx(14/22)
    assert np.isnan(m.omega2_by_axis([dict(axis=0,y=0),dict(axis=1,y=1)],'axis','y'))
    assert 'stack_eta2_sum' not in (SCRIPTS/'mega_eta2.py').read_text()


def test_all_three_classification_axes():
    m = load('p5_3axis_triangulation_per_cell')
    for a,b,c in itertools.product([False,True],repeat=3):
        out = m.classify_cell(100*a,1.*b,1.*c)
        assert (out == 'honest_and_informative') == (a and b and c)
    assert m.classify_cell(100,0,1) == 'informative_but_unaudited'
    assert m.classify_cell(100,0,0) == 'alignment_pass_registry_fail'


def test_auc_ties_permutation_and_undefined():
    m = load('_p8_common')
    for scores,want in [([1,1],.5),([0,1],1),([1,0],0)]:
        assert m.auc_p8(scores,[0,1]) == want
        assert m.auc_p8(scores[::-1],[1,0]) == want
    for scores,y in [([0,1],[1,1]),([0,float('nan')],[0,1]),([],[])]:
        with pytest.raises(ValueError): m.auc_p8(scores,y)


def test_unmeasured_declaration_and_model_mismatch_no_credit():
    m = load('p5_claim_measurement_alignment')
    assert m.axis_match('decontam','none',None) == (0.,'DECLARED_UNVERIFIED')
    assert m.axis_match('model','Llama','meta-llama/Llama-3.2-3B')[0] == 0
    assert m.axis_match('model','model-3.2','model-32')[0] == 0
