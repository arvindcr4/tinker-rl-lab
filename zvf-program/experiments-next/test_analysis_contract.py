"""Synthetic offline regressions; no historical result files are opened or written."""
import contextlib
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# These standalone scripts use local absolute imports. Isolate their module
# names during collection so other analysis packages cannot replace them.
import sys
_LOCAL_NAMES = ['analysis_contract', 'build_pool', 'common', 'analyze_rollout_quality', 'aggregate_seed_audits', 'analyze_t1_ci', 'analyze_t1_correlated_fix', 'analyze_t2_floor', 'analyze_t3_gstar', 'fill_t3_prediction']
_previous = {name: sys.modules.pop(name, None) for name in _LOCAL_NAMES}
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import analysis_contract as c
    import build_pool
    import common
    import analyze_rollout_quality as quality
    import aggregate_seed_audits as aggregate
    import analyze_t1_ci as t1
    import analyze_t1_correlated_fix as correlated
    import analyze_t2_floor as t2
    import analyze_t3_gstar as t3
    import fill_t3_prediction as weighted
finally:
    sys.path.pop(0)
    for _name, _module in _previous.items():
        sys.modules.pop(_name, None)
        if _module is not None:
            sys.modules[_name] = _module


def pool():
    return {'kind':'zvf_pool','status':'complete','tag':'synthetic','model':'test',
            'n_prompts':8,'planned_prompts':8,'rollouts_per_prompt':4,'seed':1,
            'sampling_completeness':'full','exploratory_only':False,
            'analysis_contract':{'version':c.VERSION, **{k:'test' for k in c.IDENTITY}},
            'prompts':[{'idx':i,'rewards':[0,0,1,1], 'p_hat':.5,'token_counts':[1]*4} for i in range(8)]}


class ContractTests(unittest.TestCase):
    def test_nonfinite_bool_range_and_singleton_rejected(self):
        for rs in ([0,float('nan')],[0,float('inf')],[False,1],[0,2],[1], [0,1e-10]):
            with self.assertRaises(ValueError): c.validate_rewards(rs)
            with self.assertRaises(ValueError): quality.core_metrics([{'rewards':rs}])
        self.assertTrue(c.zero_variance([0,0]))
        self.assertFalse(c.zero_variance([0,1]))
    def test_truncation_cannot_become_confirmatory(self):
        p=pool(); p['truncated']=True
        self.assertTrue(c.completeness(p)['exploratory_only'])
        with self.assertRaises(ValueError): c.validate_full_pool(p)
        audit=quality.analyze(p,n_bootstrap=10,seed=0)
        with self.assertRaises(ValueError): aggregate.aggregate([audit],seed=0,n_bootstrap=10)
    def test_legacy_status_complete_is_not_sampling_proof(self):
        p=pool(); del p['sampling_completeness']
        with self.assertRaises(ValueError): c.validate_full_pool(p)
    def test_full_count_and_token_lengths(self):
        p=pool(); c.validate_full_pool(p)
        p['prompts'][0]['token_counts']=[1]
        with self.assertRaises(ValueError): c.validate_full_pool(p)
        p=pool(); p['prompts'].pop()
        with self.assertRaises(ValueError): c.validate_full_pool(p)
    def test_every_identity_field_gates_aggregation(self):
        first=quality.analyze(pool(),n_bootstrap=10,seed=0)
        for key in c.IDENTITY:
            other=quality.analyze(pool(),n_bootstrap=10,seed=0); other['source']['seed']=2
            other['source']['analysis_contract']=dict(other['source']['analysis_contract'])
            other['source']['analysis_contract'][key]='changed'
            with self.assertRaisesRegex(ValueError,'incompatible'): aggregate.aggregate([first,other],seed=0,n_bootstrap=10)
            del other['source']['analysis_contract'][key]
            with self.assertRaisesRegex(ValueError,'missing'): aggregate.aggregate([other],seed=0,n_bootstrap=10)
    def test_resume_missing_sampler_refused_before_any_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous=pool();previous.update(status='started',requested_sampler_path='tinker://trained',
                                           sampler_path='tinker://trained',max_prompt_tokens=1024,lora_rank=4,
                                           tokenizer_revision='rev')
            Path(tmp,'pool_test.json').write_text(json.dumps(previous))
            argv=['build_pool','--model','test','--tag','test','--resume',
                  '--model-revision','rev','--dataset-revision','rev',
                  '--tokenizer-revision','rev','--checkpoint-identity','rev']
            with patch.object(build_pool,'RESULTS_DIR',Path(tmp)),patch('sys.argv',argv),\
                 patch.object(build_pool,'load_repo_env') as env, patch.object(build_pool,'make_sampler') as sampler,\
                 patch.object(build_pool,'load_gsm8k') as data, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'requested_sampler_path'): build_pool.main()
                env.assert_not_called();sampler.assert_not_called();data.assert_not_called()
    def test_short_dataset_refused_before_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv=['build_pool','--model','test','--prompts','2','--model-revision','rev',
                  '--dataset-revision','rev','--tokenizer-revision','rev','--checkpoint-identity','rev']
            with patch.object(build_pool,'RESULTS_DIR',Path(tmp)),patch('sys.argv',argv),\
                 patch.object(build_pool,'load_gsm8k',return_value=[('q','a')]),\
                 patch.object(build_pool,'make_sampler') as sampler,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'shortfall'):build_pool.main()
                sampler.assert_not_called()
    def test_exact_finite_pool_probability(self):
        self.assertAlmostEqual(correlated.prompt_zv_prob(random.Random(0),[0,0,1,1],2),1/3)
    def test_strata_cover_remainder_and_weighted_ci(self):
        strata=c.strata_all(list(range(11)),3)
        self.assertEqual(sum(strata,[]),list(range(11)))
        lo,hi=c.stratified_interval([0,1,0],[len(s)/11 for s in strata])
        self.assertLessEqual(lo,4/11);self.assertGreaterEqual(hi,4/11)
    def test_censoring_and_nearest_rank(self):
        trial=t2.rollouts_until_mixed(random.Random(0),[{'rewards':[0]*4}],2,cap_groups=3)
        self.assertEqual(trial,{'rollouts':6,'censored':True})
        self.assertEqual(t2.quantile([1,2,3,4],.5),2)
        self.assertEqual(t2.quantile([1,2,3,4],1),4)
        with self.assertRaises(ValueError):t2.floor_rollouts(.8,1,2)
    def test_proxy_tie_for_points_and_mixtures(self):
        for ps in ([0],[1],[.01],[.5],[.9],[.01,.2,.7,.99]):
            rows=[{'p_hat':p} for p in ps]
            self.assertAlmostEqual(weighted.analytic_J(rows,2),weighted.analytic_J(rows,3),places=14)
    def test_design_domains(self):
        for kw in ({'groups':[1]}, {'groups':[2],'counts':[0]}, {'groups':[8],'available':4}, {'groups':[]}):
            with self.assertRaises(ValueError):c.validate_design(**kw)
    def test_synthetic_offline_analysis_entrypoints(self):
        for module,flags in [(t1,['--group-sizes','2','--batch-sizes','4','--batches','5']),
                             (correlated,['--group-size','2','--batch-size','3','--batches','5']),
                             (t2,['--group-size','2','--n-strata','2','--trials','5']),
                             (t3,['--grid','2','3','--n-strata','2','--draws','5']),
                             (weighted,['--n-strata','2'])]:
            with patch('sys.argv',['test','--pool','synthetic',*flags]),patch.object(module,'load_pool',return_value=pool()),\
                 patch.object(module,'write_result') as write,contextlib.redirect_stdout(io.StringIO()):
                module.main()
                payload=write.call_args.args[1]
                json.dumps(payload,allow_nan=False)
                self.assertEqual(payload['status'],'complete')

if __name__ == '__main__':unittest.main()
