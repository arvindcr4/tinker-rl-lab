import asyncio
import json
import sys
from types import SimpleNamespace as NS, ModuleType
from pathlib import Path

import pytest
from platform_hybrid.experiments.webarena import aggregate, react_eval as re


def row(seed=42, score=1):
    return dict(env_id='browsergym/webarena.1', seed=seed, evaluation_id='v2', score=score,
                num_steps=3, valid_action_count=2, wall_time_sec=1)


def test_aggregate_preserves_seeds_and_rejects_corruption(tmp_path):
    p=tmp_path/'rows.jsonl'
    p.write_text('\n'.join(map(json.dumps,[row(),row(43,0)])))
    assert aggregate.summarize(aggregate.parse_results([p])).success_rate == .5
    p.write_text(json.dumps(row())+'\ninvalid\n'+json.dumps(row(43,0)))
    with pytest.raises(ValueError,match=':2:'):
        aggregate.parse_results([p])


def test_aggregate_rejects_conflicts_and_nonfinite(tmp_path):
    p=tmp_path/'rows.jsonl'
    for rows in ([row(),row(score=0)], [row(score=float('nan'))]):
        p.write_text('\n'.join(map(json.dumps,rows)))
        with pytest.raises(ValueError): aggregate.parse_results([p])


def test_episode_action_errors_and_partial_reward(monkeypatch):
    monkeypatch.setattr(re,'_import_benchmark',lambda _:None)
    monkeypatch.setattr(re,'_axtree_to_str',lambda obs,n:'tree')
    env=NS(reset=lambda **k:({'goal':'x'},{}), close=lambda:None,
           step=lambda action:({'last_action_error':'invalid'},.4,False,False,{}))
    monkeypatch.setattr(re,'_make_env',lambda _:env)
    result=re.run_episode(NS(sample=lambda _:"Thought: x\nAction: click('1')"),'task',42,3,100,'webarena')
    assert result.score == .4 and result.valid_action_count == 0
    assert result.truncated and all(s.error == 'invalid' for s in result.steps)


def test_environment_creation_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(re,'_import_benchmark',lambda _:None)
    monkeypatch.setattr(re,'_make_env',lambda _:(_ for _ in ()).throw(RuntimeError('offline')))
    result=re.run_episode(None,'task',42,3,100,'webarena')
    assert result.score == 0 and 'offline' in result.error


def args(path, shard='0/1', resume=False):
    return NS(benchmark='webarena',tasks='x',model='model',seed=42,max_steps=3,max_axtree_chars=100,
              action_tokens=12,temperature=0.,concurrency=1,shard=shard,limit=0,wandb_project=None,
              out=str(path),resume=resume,hf_repo=None)


def test_real_driver_sharding_and_resume_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(re,'_import_benchmark',lambda _:None)
    monkeypatch.setattr(re,'_parse_tasks',lambda *a:['a','b','c','d'])
    monkeypatch.setattr(re,'TinkerChatSampler',lambda **k:None)
    monkeypatch.setattr(re,'run_episode',lambda sampler,env_id,seed,**k:re.EpisodeResult(env_id,seed,1,1,True,False,1,1,[],1))
    whole=tmp_path/'whole'; shard=tmp_path/'shard'
    re._main_sync(args(whole)); re._main_sync(args(shard,'1/2'))
    full={r['env_id']:r for r in map(json.loads,whole.read_text().splitlines())}
    for r in map(json.loads,shard.read_text().splitlines()):
        assert r['seed']==full[r['env_id']]['seed']
        assert r['evaluation_id']==full[r['env_id']]['evaluation_id']
    re._main_sync(args(whole,resume=True))
    assert len(whole.read_text().splitlines())==4
    changed=args(whole,resume=True);changed.model='other'
    with pytest.raises(ValueError,match='mismatch'):re._main_sync(changed)


def vast_module(monkeypatch):
    monkeypatch.setitem(sys.modules,'asyncssh',ModuleType('asyncssh'))
    from platform_hybrid.skyrl.backends import vastai_runner
    return vastai_runner


def test_vast_new_contract_is_owned_immediately(tmp_path,monkeypatch):
    v=vast_module(monkeypatch); launcher=v.VastAILauncher(api_key='fake')
    launcher.ownership_log=tmp_path/'owned'
    monkeypatch.setattr(launcher,'run_command',lambda *a,**k:NS(returncode=0,stdout=json.dumps({'success':True,'new_contract':123})))
    instance=asyncio.run(launcher.launch_instance(9))
    assert instance.id==123 and launcher.instances==[instance]
    assert json.loads(launcher.ownership_log.read_text())['instance_id']==123


def test_vast_failed_setup_skips_training_and_cleans_every_rental(tmp_path,monkeypatch):
    v=vast_module(monkeypatch); launcher=v.VastAILauncher(api_key='fake')
    launcher.ownership_log=tmp_path/'owned';events=[]
    async def search(*a):return [{'id':1}]
    async def launch(*a):
        instance=v.VastInstance(1,'host',22,'gpu',1,1,'ready');launcher.instances.append(instance);return instance
    async def ready(*a):return True
    async def setup(*a):return False
    async def train(*a):events.append('train');return True
    async def clean(*a):events.append('clean')
    for name,fn in [('search_instances',search),('launch_instance',launch),('wait_for_ready',ready),('setup_instance',setup),('run_training',train),('cleanup_instance',clean)]: monkeypatch.setattr(launcher,name,fn)
    with pytest.raises(RuntimeError,match='Setup failed'):asyncio.run(launcher.run('model'))
    assert events==['clean']


def test_actual_train_loop_sampler_versions_and_save_awaits(monkeypatch):
    from platform_hybrid.skyrl import run_skyrl_tinker as trainer
    samples=[];saves=[]
    def future(value=None,callback=lambda:None):
        def result():callback();return value
        return NS(result=result)
    class Client:
        model_id='mock'; version=0
        def save_weights_for_sampler(self,name):return future(NS(path=str(self.version)))
        def create_sampling_client(self,model_path):
            version=int(model_path)
            def sample(*a,**k):
                samples.append((version,self.version));return future(NS(sequences=[NS(tokens=[1]),NS(tokens=[2])]))
            return NS(sample=sample)
        def forward_backward_custom(self,**kwargs):
            import numpy as np
            loss, metrics = kwargs['loss_fn'](kwargs['data'], [np.array([100.,1.]),np.array([200.,2.])])
            assert loss == pytest.approx(.5, abs=1e-6)  # prompt positions excluded
            return future(NS(metrics=metrics))
        def optim_step(self,*a):return future(callback=lambda:setattr(self,'version',self.version+1))
        def save_state(self,name):return future(callback=lambda:saves.append(name))
    client=Client();tinker=ModuleType('tinker');types=ModuleType('tinker.types')
    tinker.ServiceClient=lambda **k:NS(create_lora_training_client=lambda **k:client)
    for name in ['SamplingParams','Datum','TensorData','AdamParams']:setattr(types,name,lambda **k:NS(**k))
    types.ModelInput=NS(from_ints=lambda ids:ids)
    monkeypatch.setitem(sys.modules,'tinker',tinker);monkeypatch.setitem(sys.modules,'tinker.types',types)
    transformers=ModuleType('transformers');transformers.AutoTokenizer=NS(from_pretrained=lambda *a,**k:NS(encode=lambda *a,**k:[4,5],decode=lambda tokens,**k:str(tokens[0])))
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    import numpy as np
    torch=ModuleType('torch');torch.stack=np.stack
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setattr(trainer,'upload_to_hub',lambda *a:None)
    cfg=dict(execution_protocol=trainer.PROTOCOL_ID,model=dict(name='mock',tokenizer='mock'),tinker=dict(lora_rank=1,learning_rate=.1),training=dict(steps=11,group_size=2,prompts_per_step=1,save_every=10,sampling=dict(max_tokens=3,temperature=1,top_p=1)))
    env=NS(name='mock',sample_batch=lambda n:[{}],make_prompt=lambda _:"prompt",reward=lambda text,item:float(text=='1'))
    trainer.train(cfg,env,None)
    assert samples==[(i,i) for i in range(11)]
    assert saves==['state_10','final']


def test_disabled_wandb_does_not_import_dependency(monkeypatch):
    from platform_hybrid.skyrl import run_skyrl_tinker as trainer
    monkeypatch.setitem(sys.modules,'wandb',None)
    assert trainer.init_wandb({'logging':{'backend':'none'}},'mock') is None


def test_card_requires_receipt_and_no_errors():
    from platform_hybrid.experiments.webarena.push_model_card import _render_model_card
    kwargs=dict(base_model='model',benchmark='webarena',dataset_repo='data',run_id='run')
    with pytest.raises(ValueError):_render_model_card(final={},**kwargs)
    final={'completion_status':'complete','evaluation_receipt':{'evaluation_id':'v2','effective_config':{'model':'model','benchmark':'webarena','temperature':.7,'action_tokens':99},'actual_concurrency':1}}
    import hashlib
    config = final['evaluation_receipt']['effective_config']
    config.update(tasks=['task'], seed=42)
    eid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    final['evaluation_receipt'].update(evaluation_id=eid, coverage=[dict(evaluation_id=eid,
        env_id='task', seed=int.from_bytes(hashlib.sha256(b'42:task').digest()[:4], 'big'))])
    final['overall'] = dict(n=1, episodes_with_error=0)
    card=_render_model_card(final=final,**kwargs)
    assert 'temperature 0.7' in card and 'budget 99' in card
    assert '50-way' not in card and 'sample_async' not in card


def test_causal_unknown_argument_rejected_before_allocation(tmp_path):
    import subprocess
    script=Path(__file__).resolve().parents[2]/'platform_hybrid/experiments/run_causal_zvf.sh'
    result=subprocess.run(['bash',str(script),'--unknown'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 2 and not list(tmp_path.iterdir())


def test_vast_unsupported_epochs_rejected_before_search(monkeypatch):
    v=vast_module(monkeypatch);launcher=v.VastAILauncher(api_key='fake')
    with pytest.raises(ValueError,match='epochs'):asyncio.run(launcher.run('model',epochs=3))


def test_failed_vast_cleanup_is_not_success(monkeypatch):
    v=vast_module(monkeypatch);launcher=v.VastAILauncher(api_key='fake')
    monkeypatch.setattr(launcher,'run_command',lambda *a,**k:NS(returncode=1,stdout='{}'))
    with pytest.raises(RuntimeError,match='Cleanup not confirmed'):
        asyncio.run(launcher.cleanup_instance(v.VastInstance(12,'host',22,'gpu',1,1,'ready')))
