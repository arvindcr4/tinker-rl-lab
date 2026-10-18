"""CPU-only controlled review28 reproductions; no provider or stored data."""
import importlib
import sys
from pathlib import Path
import numpy as np
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'platform_modal'/'scripts'
sys.path.insert(0,str(SCRIPTS))


def test_tied_statistics():
    from _review28_statistics import auc,spearman,average_precision
    assert np.isnan(spearman([1]*4,[0,0,1,1]))
    for labels in ([0,0,1,1],[1,0,1,0],[1,1,0,0]):
        assert auc([.5]*4,labels)==.5
        assert average_precision([.5]*4,labels)==.5


def test_signed_rank_exact():
    from zvf_iter22 import wilcoxon_signed_rank
    assert wilcoxon_signed_rank([0]*5)==(0,1)
    assert wilcoxon_signed_rank([1,2,3,4,5])[1]==.0625
    assert wilcoxon_signed_rank([-1,-2,-3,-4,-5])[1]==.0625


def test_ks_and_linkage():
    from zvf_iter66_distshape import ks_2samp,hclust_average,cut_tree
    assert ks_2samp([.5]*10,[.5]*10)[0]==0
    assert ks_2samp([0,0],[1,1])[0]==1
    d=np.array([[0,.1,.8,.9],[.1,0,.7,.8],[.8,.7,0,.1],[.9,.8,.1,0]])
    tree=hclust_average(d,['a','b','c','d'])
    for k in (1,2,3,4):
        result=cut_tree(tree,4,k)
        assert set(result)==set(range(4)) and len(set(result.values()))==k


def test_singleton_class_not_available_to_fold():
    from zvf_iter34_phasescan import loo_classify
    pred=loo_classify(np.array([[0.,1.],[1.,0.]]),np.array(['plateau','plateau']),np.array([[0.,0.]]),['collapse','plateau'])
    assert pred.tolist()==['plateau']


def test_future_labels_censoring_and_step_units():
    from zvf_iter30_earliness import _featurise_trajectory
    import inspect
    print(inspect.signature(_featurise_trajectory))
    rows=_featurise_trajectory([(i,.2,0) for i in range(10)],horizons=[3])
    assert [r['y_K3'] for r in rows]==[0]*7+[None]*3
    rows=_featurise_trajectory([(i,.2,int(i==4)) for i in range(7)],horizons=[1])
    assert rows[3]['y_K1']==1 and rows[4]['y_K1'] is None
    rows=_featurise_trajectory([(0,.2,0),(10,.2,1)],horizons=[3])
    assert rows[0]['y_K3']==0


def test_mastery_not_collapse():
    for name in ['zvf_diagnostic_iter106','zvf_discrimination_iter110']:
        assert importlib.import_module(name)._classify_row(0,.99)=='mastery'


def test_event_precedes_alarm_and_recovery():
    from zvf_dynamics import _lead_time
    from zvf_iter22 import find_first_collapse
    result=_lead_time({'zvf_series':[.2,.8,.8,.8],'collapse_series':[1,1,1,1]})
    assert result['first_collapse_step']==0 and result['lead_steps']<0 and result['status']=='late'
    rows=[{'step':i,'zvf':.5,'collapse':int(i==1),'heldout_acc':acc} for i,acc in enumerate([.4,.1,.9])]
    assert find_first_collapse(rows)[1]==1


def test_risk_provenance_and_frozen_threshold():
    from _review28_risk import evaluate,score
    base=dict(experiment='e',method='m',seed='1',mean_zvf=.4,rolling_lag1=.6,drift_slope=.01,
              outcome_observed=True,failure=1)
    assert score(base) is None
    row=dict(base,**{name+'_measured':True for name in ('mean_zvf','rolling_lag1','drift_slope')})
    measured,excluded,threshold=evaluate([row],threshold=.95,calibrate=False)
    assert not excluded and threshold==.95
    assert measured[0]['predicted_failure']==int(score(row)>=.95)
    assert evaluate([row],calibrate=False)[2] is None
    with pytest.raises(ValueError):evaluate([row,row])
    for n in (130,134):
        module=importlib.import_module(f'zvf_diagnostic_iter{n}')
        assert module.score(row)==score(row)


def test_partial_auc_refits_and_constants():
    from _review28_statistics import partial_auc_bootstrap
    point,lo,hi=partial_auc_bootstrap([.5]*8,list(range(8)),[0,0,0,0,1,1,1,1],B=50)
    assert point==lo==hi==.5


def test_iso_yield_boundaries_and_interior():
    from zvf_iter38_crosslibrary import iso_yield_iid,iso_yield_empirical
    from zvf_iter46_isog import yield_emp
    from zvf_iso_yield_iter122 import _iso_yield_curve
    assert iso_yield_iid(.05,.8)==32
    assert np.isinf(iso_yield_iid(0,.8))
    assert iso_yield_empirical(.5,.2,.1)>=2
    assert yield_emp(.5,1,.2)==0
    assert yield_emp(.5,2,-.1)<yield_emp(.5,2,0)
    row=_iso_yield_curve([2,4,8],[.8,.4,.6],[.5])[0]
    assert row['g_min']==4 and row['achieved_zvf']==.4 and row['status']=='observed'


def test_quantile_bins_are_contiguous_and_nonempty():
    from _review28_statistics import quantile_partitions
    from zvf_diagnostic_iter114 import dose_response
    values=[.9,.1,.8,.2,.7,.3]
    rows=[dict(mean_zvf=v,last10_avg=.4,peak=.5,failure_label='plateau') for v in values]
    bins=quantile_partitions(rows,'mean_zvf',3)
    assert [np.mean([r['mean_zvf'] for r in b]) for b in bins]==pytest.approx([.15,.5,.85])
    assert len(dose_response(rows,n_bins=10))==6
    bins=quantile_partitions([{'z':i} for i in range(17)],'z',5)
    assert [r['z'] for b in bins for r in b]==list(range(17))


def test_paired_seed_join_order_and_unmatched(tmp_path):
    from zvf_iter118_diagnostic import write_aero_grpo_gap
    rows=[]
    for seed,a,g in [(1,.2,.3),(2,.6,.7)]:
        for method,z in [('AERO',a),('GRPO',g)]:
            rows.append(dict(experiment='variance_mitigation',model=method,seed=seed,mean_zvf=z,last10_avg=.7,failure='plateau'))
    a=write_aero_grpo_gap(rows,tmp_path/'a.tsv')
    b=write_aero_grpo_gap(list(reversed(rows)),tmp_path/'b.tsv')
    assert a==b
    assert a['gap_zvf']==pytest.approx(-.1)
    assert a['ci_zvf']==pytest.approx((-.1,-.1))
    extra=dict(rows[0],seed=3,mean_zvf=.99)
    c=write_aero_grpo_gap(rows+[extra],tmp_path/'c.tsv')
    assert c['gap_zvf']==pytest.approx(-.1) and c['unmatched_aero']==1


def test_last_ten_sorted_before_seed_pool(tmp_path,monkeypatch):
    import zvf_iter26_residual as m
    monkeypatch.setattr(m,'RESULTS',tmp_path)
    lines=['method\tseed\tstep\tzvf\theldout_acc\treward_mean']
    lines += [f'm\t1\t{i}\t.3\t{int(i>=10)}\t.4' for i in reversed(range(20))]
    (tmp_path/'variance_mitigation.tsv').write_text('\n'.join(lines))
    assert m._library_level_pool([])[0]['last10']==1


def test_table_roundtrip_and_run_identity(tmp_path,monkeypatch):
    import zvf_iter54_doom_threshold as m
    path=tmp_path/'table.tsv'
    m._write_tsv(path,['method','seed','value'],[['a',1,.2],['b',2,.5]])
    header,rows=m._read_tsv(path)
    assert header==['method','seed','value'] and len(rows)==2
    monkeypatch.setattr(m,'RESULTS',tmp_path)
    (tmp_path/'variance_mitigation.tsv').write_text('method\tseed\tzvf\na\t1\t.2\nb\t1\t.9\n')
    assert m._per_step_max_zvf_variance_mitigation('1','a')==.2


def test_real_figure_negative_values_and_changed_verdict(tmp_path):
    from zvf_iter50_fig import render
    from PIL import Image
    values,text=render([{'method':'grpo','lag':1,'r':-.8}],[{'id':'P4','pass':'False','value':-10}],tmp_path)
    assert values[('grpo',1)]==-.8 and 'False' in text
    with Image.open(tmp_path/'zvf_iter50.png') as im:
        im.load();assert im.width>100 and im.height>100
    assert (tmp_path/'zvf_iter50.pdf').read_bytes().startswith(b'%PDF-')
    import shutil,subprocess
    if shutil.which('pdfinfo'):
        assert subprocess.run(['pdfinfo',str(tmp_path/'zvf_iter50.pdf')],capture_output=True).returncode==0
    from zvf_iter46_fig import make_pdf
    _,png=make_pdf(tmp_path)
    with Image.open(png) as im:
        im.load();assert im.width>100


def test_prediction_missing_comparator_absolute_lag(tmp_path):
    import csv,json
    import zvf_iter50 as m
    lagged={('grpo',lag):{'mean':0.,'n':1} for lag in m.LAGS}
    lagged[('grpo',-10)]={'mean':-.95,'n':1}
    lagged[('grpo',1)]={'mean':.1,'n':1}
    output=tmp_path/'predictions.tsv'
    m.write_predictions(output,lagged,{}, {},{('grpo',1):[{'collapse':1}]},{('grpo',1):.25})
    rows={row['id']:row for row in csv.DictReader(output.open(),delimiter='\t')}
    assert rows['P3_grpo_p2_int_coll_minus_noncoll']['pass']=='NOT_EVALUABLE'
    assert rows['P4_grpo_peak_lag_positive']['value']=='-10'
    assert rows['P4_grpo_peak_lag_positive']['pass']=='FAIL'
    for row in rows.values():json.loads(row['extras_json'])


def test_summary_numeric_and_contrast_roundtrip(tmp_path,monkeypatch):
    import csv
    import zvf_iter66_distshape as m
    monkeypatch.setattr(m,'RES',str(tmp_path));monkeypatch.setattr(m,'FIG',str(tmp_path))
    data={name:[(.1+.1*i,.5,.4)]*4 for i,name in enumerate(['grpo','aero','gift','es'])}
    monkeypatch.setattr(m,'load_per_step',lambda:data)
    m.main()
    with (tmp_path/'zvf_iter66_summary_review28_v2.tsv').open() as fh:
        rows=list(csv.DictReader((l for l in fh if not l.startswith('#')),delimiter='\t'))
    for row in rows:
        assert float(row['mean_zvf'])==pytest.approx(data[row['method']][0][0],abs=1e-4)
        assert float(row['delta_obs_minus_pred'])==pytest.approx(float(row['mean_zvf'])-float(row['iid_pred_zvf']),abs=2e-4)


def test_aggregate_plugin_requires_explicit_group_size():
    from zvf_iter102 import calibration_table
    row=dict(library='tool_use',model='m',mean_zvf=1,mean_last10=.5,mean_peak=.5)
    assert calibration_table([row],[])==[]
    result=calibration_table([dict(row,group_size=8)],[])[0]
    assert result['delta_calibration']==pytest.approx(.9922)
    assert result['interpretation']=='not_identified_dependence'


def test_empty_tables_replace_stale_data(tmp_path,monkeypatch):
    from zvf_dynamics import _write_tsv
    import zvf_iso_yield_iter122 as m
    path=tmp_path/'empty.tsv';path.write_text('old\nrecord\n')
    _write_tsv(path,[])
    assert path.read_text().splitlines()[-1]=='status'
    monkeypatch.setattr(m.zd,'load_variance_mitigation',lambda:[])
    m.write_aero_quantile(path)
    assert path.read_text()=='seed\taero_zvf\tgrpo_zvf\tgap_zvf\n'


def test_same_group_plugin_is_descriptive_not_null_evidence(tmp_path,monkeypatch):
    from itertools import product
    import json
    import zvf_diagnostic_iter106 as m
    monkeypatch.setattr(m,'RES',tmp_path)
    data=dict(group_size=8,per_problem=[dict(problem_id=i,rewards=list(r)) for i,r in enumerate(product([0,1],repeat=8))])
    (tmp_path/'tinker_gsm8k_zvf_s42.json').write_text(json.dumps(data))
    rows=m.load_per_problem_phase_points()
    assert len(rows)==256
    assert np.mean([r['delta'] for r in rows])==pytest.approx(-.055872, abs=1e-6)
    assert {r['estimator'] for r in rows}=={'same_group_plugin_descriptive_not_dependence'}


def test_aero_contrast_direction(tmp_path,monkeypatch):
    import zvf_iso_yield_iter122 as m
    rows=[dict(model=method,seed=seed,mean_zvf=z,last10_avg=.7)
          for seed in (1,2) for method,z in [('AERO',.2),('GRPO',.5)]]
    monkeypatch.setattr(m.zd,'load_variance_mitigation',lambda:rows)
    result=m.write_aero_quantile(tmp_path/'gap.tsv')
    assert result['mean_gap_zvf']==pytest.approx(-.3)


def test_reported_stable_seed():
    import hashlib
    from zvf_iter102 import per_library_delta_summary
    rows=[dict(library='lib',delta_calibration=.1,rho_overdispersion=1.2,zvf_iid_pred=.2,zvf_emp=.3)]
    result=per_library_delta_summary(rows)[0]
    assert result['bootstrap_seed']==int.from_bytes(hashlib.sha256(b'iter102:lib').digest()[:8],'big')


def test_tex_escaping():
    from zvf_gradient_coupling import tex_escape
    assert tex_escape('a_b%')==r'a\_b\%'


def test_iter50_producer_renderer_chain(tmp_path, monkeypatch):
    import zvf_iter50 as producer
    import zvf_iter50_fig as renderer
    root = SCRIPTS.parents[1]
    assert Path(producer.INPUT) == root/'platform_hybrid/experiments/results/variance_mitigation.tsv'
    assert Path(producer.OUT_LAGGED).parent == renderer.RESULTS
    assert renderer.analysis is producer
    source = tmp_path/'input.tsv'
    source.write_text('method\tseed\tstep\tzvf\treward_mean\theldout_acc\tcollapse\n' +
                     '\n'.join(f'grpo\t1\t{i}\t{.9-i*.01}\t{.1+i*.01}\t{.8-i*.001}\t0' for i in range(80)))
    monkeypatch.setattr(producer, 'INPUT', str(source))
    out = tmp_path/'results'
    monkeypatch.setattr(producer, 'OUT_DIR', str(out))
    for key in ('OUT_LAGGED', 'OUT_INTEGRALS', 'OUT_SUMMARY', 'OUT_PREDICTIONS'):
        monkeypatch.setattr(producer, key, str(out/Path(getattr(producer, key)).name))
    monkeypatch.setattr(renderer, 'FIG_DIR', tmp_path/'figures')
    producer.main()
    renderer.main()
    assert (tmp_path/'figures/zvf_iter50.png').is_file()
    assert float(renderer.read_rows(producer.OUT_LAGGED)[0]['r']) < 0


def test_average_precision_nonfinite_scores_do_not_hang():
    from _review28_statistics import average_precision
    assert average_precision([.5, np.nan], [1, 0]) == 1.
    assert average_precision([np.nan, .5], [0, 1]) == 1.
    assert np.isnan(average_precision([np.nan], [1]))
    assert average_precision([.5, np.inf, .1], [1, 1, 2]) == 1.
    with pytest.raises(ValueError): average_precision([.5], [1, 0])
