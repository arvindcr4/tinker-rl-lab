import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'platform_modal/scripts'
sys.path.insert(0, str(SCRIPTS))
def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize('budget', [1000000, 4000000, 16000000, 64000000])
@pytest.mark.parametrize('old_block', ['', '\n# ITER_63_RETENTION_FITS\nold data\n'])
def test_iter63_preserves_every_historical_byte(tmp_path, budget, old_block):
    m = load('group_size_iter63')
    m.EFFECT_TSV = tmp_path/'historical.tsv'
    original = f'budget\tmetric\n{budget}.0\t0.4\n' + old_block
    m.EFFECT_TSV.write_text(original)
    m.append_to_effect_tsv([(budget, .8, .7, .9)], [0], (.8, 0, 1, (0,0,0)), (.8, 0, 1, (0,0,0)), (.8,.8,1e6,1,(0,0,0),(0,0,0)), False,False)
    assert m.EFFECT_TSV.read_text() == original
    output = tmp_path/'group_size_iter63_retention_fits_v2.tsv'
    assert len(output.read_text().splitlines()) == 2

@pytest.mark.parametrize('destination', ['existing', 'source', 'child'])
def test_anonymizer_refuses_unsafe_destinations(tmp_path, destination):
    dest = tmp_path/'valuable'
    dest.mkdir(); (dest/'keep').write_text('precious')
    out = {'existing': dest, 'source': SCRIPTS.parents[1], 'child': SCRIPTS.parents[1]/'unsafe-new-copy'}[destination]
    proc = subprocess.run(['bash', str(SCRIPTS/'anonymize.sh'), str(out)], cwd=tmp_path, capture_output=True)
    assert proc.returncode != 0
    assert (dest/'keep').read_text() == 'precious'

def test_tost_direction():
    m = load('group_size_iter43')
    assert m.tost_pvalue(0, -.01, .01, .05) < .05
    assert m.tost_pvalue(-.1, -.11, -.09, .05) > .95

@pytest.mark.parametrize('small,expected', [(.4,False), (.79,True)])
def test_native_retention_direction(small, expected):
    m = load('group_size_iter135')
    runs = [{'seed': seed, 'group_size': g, 'heldout_acc': value} for seed in [1,2,3] for g,value in [(2,small),(16,.8)]]
    result = m.native_wu_claim({'runs': runs})
    assert f'wu_native_claim_holds={expected}' in result[1]['headline']

def test_nonmonotone_crossing():
    m = load('group_size_iter59')
    assert 1 < m.find_T_for_acc([{'G':4,'T':t,'acc':a} for t,a in [(1,.4),(4,.8),(16,.6)]],4,.7) < 4

def test_no_threshold_crossing():
    m=load('group_size_iter39')
    assert m.t_critical_exponential(.99,1) == float('inf')

def test_impossible_fixed_marginal():
    m=load('group_size_iter23')
    rows=m.anti_herding_simulation()
    assert rows.loc[rows.p_eff==.8, 'feasible_alpha'].iloc[0] == 'unattainable_at_fixed_marginal_p'

def test_contamination_requires_real_inputs(tmp_path):
    result=subprocess.run([sys.executable,str(SCRIPTS/'contamination_check.py')],cwd=tmp_path,capture_output=True)
    assert result.returncode != 0
    assert b'actual --train-texts' in result.stderr
