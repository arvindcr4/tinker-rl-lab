"""Offline 75-minute profile, exact selection and client cleanup tests."""
import ast
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('modal_public_portfolio_group_75_runtime.py')
FAST=SOURCE.with_name('modal_public_portfolio_fast_runtime.py')
spec=importlib.util.spec_from_file_location('actor75_validation_fixture',ROOT/'.codex-run/public_colab_runtime_fast.py')
actor=importlib.util.module_from_spec(spec);spec.loader.exec_module(actor)


def helpers():
    tree=ast.parse(FAST.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_profile_reservation')
    ns={'load_runtime':lambda:actor,'INCLUSIVE_HOURLY_USD':5.751216};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(FAST),'exec'),ns)
    frozen=types.SimpleNamespace(validate_profile_reservation=ns['validate_profile_reservation'])
    tree=ast.parse(SOURCE.read_text());names={'runtime_profile','validate_profile_reservation','validate_inputs','main','stop_marker_path'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for fn in funcs:fn.decorator_list=[]
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'ORIGINAL_EXPORT_SHA256','RESUME_HELPER_SHA256','RESUME_COLLECTOR_SHA256','LIMITS'} for t in n.targets)]
    ns={'frozen':frozen,'load_runtime':lambda:actor,'Path':Path,'json':json,'hashlib':hashlib,'dt':dt,
        'source_manifest':lambda:{'EXPLICIT_FIXTURE_ONLY':'a'*64},'__file__':str(SOURCE)}
    exec(compile(ast.Module(body=constants+funcs,type_ignores=[]),str(SOURCE),'exec'),ns)
    ns['PROFILE']=ns['runtime_profile']();return ns


def fixture(ns):
    now=dt.datetime.now(dt.timezone.utc)
    rows=[{'task_id':'explicit-fixture-0','kind':'actor','api_path':'/v1/chat/completions','contract_sha256':'a'*64,'payload':{'messages':[]}}]
    raw=json.dumps(rows[0])+'\n';digest=hashlib.sha256(raw.encode()).hexdigest()
    selection={'schema':'omni-actor-resume-review-v1','status':'PREPARED_NOT_DISPATCHED','score':None,'native_ingestion_performed':False,
        'original_export_sha256':ns['ORIGINAL_EXPORT_SHA256'],'source_sha256':{'preparer':ns['RESUME_HELPER_SHA256'],'collector':ns['RESUME_COLLECTOR_SHA256']},
        'full_expected_total':4428,'automatic_resampling_allowed':False,'byte_identical_original_rows':True,
        'new_native_claims':0,'model_calls':0,'provider_calls':0,'continuation_requests_sha256':digest,
        'continuation_task_ids':['explicit-fixture-0'],'continuation_count':1,'contract_sha256':'a'*64,
        'excluded_started_task_ids':[f'prior-fixture-{i}' for i in range(4427)],
        'provider_observed_at':now.isoformat(),'review_valid_until':(now+dt.timedelta(seconds=300)).isoformat()}
    funds={'status':'RESERVED','reservation_id':'LOCAL_FIXTURE_NO_REAL_RESERVATION','provider':'modal','unit':'USD','reserved_units':9,
        'max_hourly_units':7.2,'max_wall_seconds':4500,'max_requests':3,'runtime_profile':'group-fullbatch75',
        'expires_at':(now+dt.timedelta(hours=4)).isoformat(),'source_dependencies_sha256':ns['source_manifest'](),'explicit_resource_limits':ns['LIMITS'],
        'requests_sha256':digest,'original_export_sha256':ns['ORIGINAL_EXPORT_SHA256'],
        'resume_receipt_sha256':hashlib.sha256((json.dumps(selection,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()}
    prior={'mode':'online','initialized_before_model_work':True,'run_id':'EXPLICIT_FIXTURE_NO_REAL_WANDB'}
    return raw,selection,funds,prior


class Launcher75Tests(unittest.TestCase):
    def test_exact_profile_reservation_and_fresh_selection(self):
        ns=helpers();raw,selection,funds,prior=fixture(ns)
        ns['validate_inputs'](raw,funds,prior,False,selection)
        self.assertEqual(ns['PROFILE'],{'name':'group-fullbatch75','purpose':'fullbatch75','reserved_usd':9.,'total_seconds':4500,'startup_seconds':60,'function_seconds':4440,'max_requests':4430})
        for key,value in [('max_wall_seconds',4499),('max_wall_seconds',4501),('reserved_units',8.99),('reserved_units',9.01),('max_hourly_units',7),
                           ('source_dependencies_sha256',{}),('explicit_resource_limits',{'cpu':[4,8],'memory_mib':[131072,131072]}),('requests_sha256','0'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):ns['validate_inputs'](raw,{**funds,key:value},prior,False,selection)
    def test_scope_or_freshness_drift_rejected_before_dispatch(self):
        ns=helpers();raw,selection,funds,prior=fixture(ns)
        for key,value in [('source_sha256',{}),('full_expected_total',4427),('continuation_count',2),('automatic_resampling_allowed',True),
                          ('byte_identical_original_rows',False),('provider_observed_at','2020-01-01T00:00:00+00:00'),('native_ingestion_performed',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):ns['validate_inputs'](raw,funds,prior,False,{**selection,key:value})
        with self.assertRaises(ValueError):ns['validate_inputs'](raw,funds,prior,True,selection)
        with self.assertRaises(ValueError):ns['validate_inputs'](raw,funds,prior,False,None)
    def test_hard_resource_single_use_and_same_actor_command(self):
        tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_batch')
        opts={v.arg:ast.unparse(v.value) for v in fn.decorator_list[0].keywords}
        self.assertEqual(opts['cpu'],'(4, 4)');self.assertEqual(opts['memory'],'(131072, 131072)')
        self.assertEqual(opts['retries'],'0');self.assertEqual(opts['max_containers'],'1');self.assertEqual(opts['single_use_containers'],'True')
        source=SOURCE.read_text();self.assertIn('"--request-seconds", "180", "--max-model-len", "32768", "--max-num-seqs", "32"',source)
        self.assertNotIn('def execute_prepared',source);self.assertIn('"/root/public_colab_runtime_group.py"',source)
    def test_cleanup_all_failures_including_uncertain_spawn_and_disk(self):
        for failure in ('spawn','get','keyboard','success-write','failure-write','cancel'):
            ns=helpers();raw,selection,funds,prior=fixture(ns)
            call=types.SimpleNamespace(get=Mock(return_value={'receipt':{'status':'FIXTURE'},'responses':[]}),cancel=Mock())
            spawn=Mock(return_value=call);stop=Mock()
            if failure=='spawn':spawn.side_effect=TimeoutError('uncertain spawn')
            elif failure in ('get','cancel','failure-write'):call.get.side_effect=TimeoutError('fixture')
            elif failure=='keyboard':call.get.side_effect=KeyboardInterrupt()
            if failure=='cancel':call.cancel.side_effect=RuntimeError('fixture cancel failure')
            ns.update(run_batch=types.SimpleNamespace(spawn=spawn),TOTAL_RESERVED_SECONDS=4500,app=types.SimpleNamespace(app_id='fixture-app'),
                      subprocess=types.SimpleNamespace(run=stop),sys=types.SimpleNamespace(executable='fixture-python'))
            with tempfile.TemporaryDirectory() as td:
                p=Path(td);out=p/'output.json'
                for name,obj in [('funds',funds),('prior',prior),('selection',selection)]: (p/name).write_text(json.dumps(obj))
                (p/'rows').write_text(raw)
                old_open=Path.open
                def guarded_open(path,*args,**kwargs):
                    if path==out and failure in ('success-write','failure-write'):raise OSError('fixture disk failure')
                    return old_open(path,*args,**kwargs)
                with patch.object(Path,'open',guarded_open),self.subTest(failure=failure),self.assertRaises(BaseException):
                    ns['main'](str(p/'funds'),str(p/'prior'),str(p/'rows'),str(p/'selection'),str(out))
            self.assertEqual(stop.call_count,1)
            self.assertEqual(stop.call_args.args[0],['fixture-python','-m','modal','app','stop','--yes','fixture-app'])
            if failure!='spawn':call.cancel.assert_called_once_with(terminate_containers=True)

if __name__=='__main__':unittest.main()
