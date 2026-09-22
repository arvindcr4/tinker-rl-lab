#!/usr/bin/env python3
"""Offline tau3 Banking preparation and native configuration bridge.

This module never launches a simulation or model. prepare binds the actual 97
released tasks, unchanged native policy/tool schemas, user prompts and scorers.
plan binds the saved Qwen checkpoint to the native LiteLLM actor interface and
writes an explicit future command. All output scores remain null.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shlex
import socket
import sys
import tempfile
from urllib.parse import urlparse

SCHEMA = 'public-tau3-native-preparation-v1'
REVISION = 'a2c024725189473d2d7cea3a5cfdbcc67478e41f'
SOURCE_MANIFEST_SHA256 = 'd2bb123860f30d1ae4f6bab2661fa5134c50981b8a892bb248c965fb6190f536'
BASE_MODEL = 'Qwen/Qwen3.6-35B-A3B'
BASE_REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'
ADAPTER_REPO = 'arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6'
ADAPTER_REVISION = '64444133c55d88c3f1bf0df8a2f5d7ac646125c8'
USER_MODEL = 'gpt-4.1-2025-04-14'
TOTAL = 97
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / 'outputs/public_portfolio_2026-09-05/tau3_setup'
SCORER_PATHS = ['src/tau2/evaluator/evaluator.py', 'src/tau2/evaluator/evaluator_env.py',
                'src/tau2/evaluator/evaluator_action.py', 'src/tau2/evaluator/evaluator_communicate.py',
                'src/tau2/evaluator/evaluator_nl_assertions.py', 'src/tau2/metrics/agent_metrics.py']

class ContractError(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise ContractError(message)

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def fingerprint(value):
    return sha(canonical(value))

def read_json(path):
    return json.loads(Path(path).read_text())

def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)
    if path.exists():
        require(path.read_bytes() == raw, f'immutable artifact differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.tau3-publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == raw, f'immutable artifact differs: {path}')
    finally:
        os.unlink(temporary)

def validate_source_blob(item, raw):
    require(len(raw) == item['bytes'], f'source size mismatch: {item["path"]}')
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    require(blob == item['git_blob_sha1'] and sha(raw) == item['sha256'], f'source content mismatch: {item["path"]}')

def verify_sources(setup):
    setup = Path(setup).resolve()
    rows = read_json(setup / 'source_receipts.json')
    require(len(rows) == len({r['path'] for r in rows}), 'duplicate source paths')
    manifest = {'revision': REVISION, 'files': [{'path': r['path'], 'git_blob_sha1': r['git_blob_sha1'], 'size': r['bytes']} for r in rows]}
    require(fingerprint(manifest) == SOURCE_MANIFEST_SHA256, 'pinned source inventory mismatch')
    root = setup / 'source'
    for row in rows:
        require(row['source_revision'] == REVISION, 'source revision mismatch')
        path = root / row['path']
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'unsafe source path')
        validate_source_blob(row, path.read_bytes())
    return {'revision': REVISION, 'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
            'source_files': len(rows), 'source_bytes': sum(r['bytes'] for r in rows),
            'license': 'MIT', 'license_sha256': sha((root / 'LICENSE').read_bytes())}

@contextlib.contextmanager
def offline_guard():
    """Block all socket connection attempts while native metadata is inspected."""
    old = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    def deny(*args, **kwargs):
        raise ContractError('network disabled during offline preparation')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = old

def load_native(setup):
    setup = Path(setup).resolve()
    src = setup / 'source' / 'src'
    os.environ['TAU2_DATA_DIR'] = str(setup / 'source' / 'data')
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    os.environ['LOGURU_LEVEL'] = 'ERROR'
    sys.path.insert(0, str(src)) if str(src) not in sys.path else None
    package = importlib.import_module('tau2')
    require(Path(package.__file__).resolve().is_relative_to(src), 'tau2 import came from a different checkout')
    require(importlib.metadata.version('tau2') == '1.0.1', 'native tau2 package version mismatch')
    env = importlib.import_module('tau2.domains.banking_knowledge.environment')
    utils = importlib.import_module('tau2.utils.utils')
    require(utils.DATA_DIR.resolve() == setup / 'source' / 'data', 'native data path already bound to another setup')
    return env

class UninitializedRetrieval:
    """Metadata-only sentinel. It cannot supply a retrieval or shell result."""
    def __getattr__(self, name):
        raise ContractError('retrieval/shell was not initialized; offline metadata inspection only')

def metadata_environment(native):
    from tau2.domains.banking_knowledge.retrieval import resolve_variant, build_policy
    from tau2.domains.banking_knowledge.retrieval_toolkits import KnowledgeToolsAllTools
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools
    from tau2.environment.environment import Environment
    db = native.get_db()
    kb = native.get_knowledge_base()
    variant = resolve_variant('alltools')
    require(variant.kb_search_dense.embedder_model == 'text-embedding-3-large', 'dense backend drift')
    require(variant.shell.allow_writes is False, 'native shell write policy drift')
    guard = UninitializedRetrieval()
    tools = KnowledgeToolsAllTools(db, guard, guard, guard)
    policy = build_policy(variant, kb)
    env = Environment(domain_name='banking_knowledge', policy=policy, tools=tools, user_tools=KnowledgeUserTools(db))
    return env, kb

def validate_identity(identity):
    expected = {'model_id': BASE_MODEL, 'model_revision': BASE_REVISION,
                'hf_repo': ADAPTER_REPO, 'hf_commit': ADAPTER_REVISION}
    for key, value in expected.items():
        require(identity.get(key) == value, f'exact saved actor mismatch: {key}')
    require(isinstance(identity.get('served_model_id'), str) and bool(re.fullmatch(r'[A-Za-z0-9_.:/-]+', identity['served_model_id'])), 'missing or unsafe served model ID')
    return fingerprint(identity)

def make_native_config(identity, actor_base_url, campaign_id, max_tokens=None):
    """Native config only. This does not create a client, simulator or runtime."""
    validate_identity(identity)
    u = urlparse(actor_base_url)
    require(u.scheme in {'http', 'https'} and u.hostname in {'127.0.0.1', 'localhost', '::1'}, 'actor endpoint must be a guarded loopback OpenAI-compatible service')
    require(not u.username and not u.password and not u.query and not u.fragment and u.path.rstrip('/') == '/v1', 'invalid actor endpoint URL')
    require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+', campaign_id)), 'unsafe campaign ID')
    from tau2.data_model.simulation import TextRunConfig
    kwargs = {'temperature': 0.0, 'api_base': actor_base_url.rstrip('/'), 'api_key': 'public-local-runtime'}
    if max_tokens is not None:
        require(isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0, 'actor max tokens must be positive')
        kwargs['max_tokens'] = max_tokens
    config = TextRunConfig(domain='banking_knowledge', llm_agent='openai/' + identity['served_model_id'],
                           llm_args_agent=kwargs, llm_user=USER_MODEL, llm_args_user={'temperature': 0.0},
                           retrieval_config='alltools', task_split_name='base', num_trials=1,
                           seed=300, max_steps=200, max_errors=10, max_concurrency=3,
                           save_to=campaign_id, verbose_logs=True)
    return config

def prepare(setup, output):
    setup, output = Path(setup).resolve(), Path(output)
    verified = verify_sources(setup)
    with offline_guard():
        native = load_native(setup)
        from tau2.runner.build import build_agent, build_user, _derive_read_log_allowlist
        from tau2.data_model.simulation import TextRunConfig
        from tau2.metrics.agent_metrics import pass_hat_k
        tasks = native.get_tasks('base')
        paths = sorted((setup / 'source/data/tau2/domains/banking_knowledge/tasks').glob('task_*.json'))
        require(len(tasks) == len(paths) == TOTAL, 'native loader did not load all 97 task files')
        require(len({t.id for t in tasks}) == TOTAL, 'duplicate task ID')
        require([t.id for t in tasks] == [p.stem for p in paths], 'native task ordering/IDs differ from filenames')
        env, kb = metadata_environment(native)
        agent = build_agent('llm_agent', env, llm='openai/ACTOR_AT_RUNTIME', llm_args={'temperature': 0.0})
        schemas = [t.openai_schema for t in env.get_tools()]
        require({'KB_search_bm25', 'KB_search_dense', 'shell'} <= {t.name for t in env.get_tools()}, 'alltools schema incomplete')
        write_once(output / 'agent_tools.json', schemas)
        write_once(output / 'agent_prompt.json', {'domain_policy': env.get_policy(), 'agent_system_prompt': agent.system_prompt})
        defaults = TextRunConfig(domain='banking_knowledge').model_dump(mode='json')
        write_once(output / 'native_defaults.json', defaults)
        rows = []
        trial_seed = random.Random(defaults['seed']).randint(0, 1000000)
        for task, path in zip(tasks, paths):
            raw = read_json(path)
            require(raw['id'] == task.id, 'task ID mismatch')
            required = raw.get('required_documents') or []
            require(set(required) <= set(kb.documents), f'missing required documents: {task.id}')
            user = build_user('user_simulator', env, task, llm=USER_MODEL, llm_args={'temperature': 0.0})
            user_schema = [tool.openai_schema for tool in (user.tools or [])]
            expected_user_names = set(task.user_tools) if task.user_tools is not None else {t.name for t in env.get_user_tools()}
            require({tool.name for tool in (user.tools or [])} == expected_user_names, f'user tool filtering failed: {task.id}')
            user_binding = {'system_prompt': user.system_prompt, 'tools': user_schema, 'model': USER_MODEL, 'llm_args': {'temperature': 0.0}}
            write_once(output / 'user_bindings' / f'{task.id}.json', user_binding)
            criteria = task.evaluation_criteria.model_dump(mode='json')
            rows.append({'task_id': task.id, 'evaluation_id': f'tau3:{REVISION}:base:{task.id}:trial0:seed{trial_seed}',
                         'trial': 0, 'seed': trial_seed, 'task_file': str(path.relative_to(setup / 'source')),
                         'task_sha256': sha(path.read_bytes()), 'native_task_sha256': fingerprint(task.model_dump(mode='json')),
                         'user_binding_sha256': fingerprint(user_binding), 'initial_state_sha256': fingerprint(raw.get('initial_state')),
                         'evaluation_criteria_sha256': fingerprint(criteria), 'reward_basis': criteria['reward_basis'],
                         'golden_action_count': len(criteria.get('actions') or []),
                         'nl_assertion_count': len(criteria.get('nl_assertions') or []),
                         'required_documents': required, 'read_log_allowlist': sorted(_derive_read_log_allowlist(task)),
                         'score': None})
        require(sum(r['golden_action_count'] for r in rows) == 955, 'golden action count drift')
        require([r['task_id'] for r in rows if r['nl_assertion_count']] == ['task_102'], 'NL grader task membership drift')
        require(pass_hat_k(1, 1, 1) == 1.0 and pass_hat_k(1, 0, 1) == 0.0, 'native pass^1 definition changed')
        documents = [{'document_id': d.id, 'content_sha256': sha(d.content.encode()), 'title': d.title} for d in kb.documents.values()]
        write_once(output / 'knowledge_manifest.json', documents)
        binding = {'agent_class': 'tau2.agent.llm_agent.LLMAgent', 'actor_transport': 'tau2.utils.llm_utils.generate -> litellm.completion',
                   'agent_prompt_sha256': fingerprint({'domain_policy': env.get_policy(), 'agent_system_prompt': agent.system_prompt}),
                   'agent_tools_sha256': fingerprint(schemas), 'agent_tool_count': len(schemas),
                   'user_class': 'tau2.user.user_simulator.UserSimulator', 'user_model': USER_MODEL,
                   'retrieval_config': 'alltools', 'embedding_model': 'text-embedding-3-large',
                   'nl_assertion_model': USER_MODEL, 'nl_assertion_tasks': ['task_102'],
                   'scorers': {p: sha((setup / 'source' / p).read_bytes()) for p in SCORER_PATHS},
                   'metric': 'native compute_metrics(...).pass_hat_ks[1]',
                   'retrieval_status': 'NOT_INITIALIZED; only native policy and tool schema extracted',
                   'metadata_sentinel_note': 'A sentinel occupies retrieval/sandbox handles during schema inspection. It raises if used; it never provides substitute retrieval results.',
                   'native_default_config_sha256': fingerprint(defaults)}
        write_once(output / 'native_binding.json', binding)
    manifest = {'schema_version': SCHEMA, 'source': verified, 'suite_id': 'tau3_banking_eval', 'domain': 'banking_knowledge',
                'task_split_argument': 'base', 'split_note': 'Native banking loader ignores task_split_name and loads the 97 task_*.json files.',
                'tasks': rows, 'task_count': TOTAL, 'num_trials': 1, 'native_trial_seed': trial_seed,
                'native_binding_sha256': fingerprint(binding), 'knowledge_manifest_sha256': fingerprint(documents),
                'knowledge_document_count': len(documents), 'status': 'PREPARED_NOT_EXECUTED',
                'runtime_ready': False, 'score': None, 'model_calls': 0, 'provider_calls': 0,
                'decontamination': {'status': 'NOT_PERFORMED_BY_THIS_BRIDGE', 'heldout_claim': False}}
    write_once(output / 'manifest.json', manifest)
    return manifest

def plan(setup, manifest_dir, identity, actor_base_url, campaign_id, output, max_tokens=None):
    verified = verify_sources(setup)
    manifest = read_json(Path(manifest_dir) / 'manifest.json')
    require(manifest['source'] == verified and manifest['task_count'] == TOTAL and len(manifest['tasks']) == TOTAL, 'manifest source/count mismatch')
    require(len({r['task_id'] for r in manifest['tasks']}) == TOTAL, 'duplicate task identity')
    root = Path(setup).resolve() / 'source'
    expected_paths = sorted((root / 'data/tau2/domains/banking_knowledge/tasks').glob('task_*.json'))
    require([r['task_id'] for r in manifest['tasks']] == [p.stem for p in expected_paths], 'task identity drift')
    for row, path in zip(manifest['tasks'], expected_paths):
        require(row['task_file'] == str(path.relative_to(root)) and row['task_sha256'] == sha(path.read_bytes()), 'task source binding drift')
        require(row['score'] is None, 'preparation cannot contain a model score')
        user_binding = read_json(Path(manifest_dir) / 'user_bindings' / f'{row["task_id"]}.json')
        require(fingerprint(user_binding) == row['user_binding_sha256'], 'native user binding drift')
    binding = read_json(Path(manifest_dir) / 'native_binding.json')
    require(fingerprint(binding) == manifest['native_binding_sha256'], 'native binding drift')
    require(binding['retrieval_config'] == 'alltools' and binding['embedding_model'] == 'text-embedding-3-large', 'retrieval condition drift')
    for field, name in [('agent_tools_sha256', 'agent_tools.json'), ('agent_prompt_sha256', 'agent_prompt.json'), ('native_default_config_sha256', 'native_defaults.json')]:
        require(fingerprint(read_json(Path(manifest_dir) / name)) == binding[field], f'native metadata drift: {name}')
    require(fingerprint(read_json(Path(manifest_dir) / 'knowledge_manifest.json')) == manifest['knowledge_manifest_sha256'], 'knowledge manifest drift')
    with offline_guard():
        load_native(setup)
        config = make_native_config(identity, actor_base_url, campaign_id, max_tokens)
    cfg = config.model_dump(mode='json')
    argv = ['tau2', 'run', '--domain', 'banking_knowledge', '--agent', 'llm_agent', '--user', 'user_simulator',
            '--agent-llm', cfg['llm_agent'], '--agent-llm-args', json.dumps(cfg['llm_args_agent'], separators=(',', ':')),
            '--user-llm', USER_MODEL, '--user-llm-args', '{"temperature":0.0}', '--retrieval-config', 'alltools',
            '--task-split-name', 'base', '--num-trials', '1', '--max-steps', '200', '--max-errors', '10',
            '--max-concurrency', '3', '--seed', '300', '--save-to', campaign_id, '--verbose-logs', '--llm-log-mode', 'all']
    result = {'schema_version': SCHEMA, 'status': 'CONFIGURATION_PREPARED_REQUIRES_GUARDED_RUNTIME', 'score': None,
              'source': verified, 'manifest_sha256': fingerprint(manifest), 'model_identity': identity,
              'model_identity_sha256': validate_identity(identity), 'native_config': cfg,
              'argv': argv, 'command': shlex.join(argv), 'llm_log_mode': 'all',
              'inference_condition': {'temperature': 0.0, 'actor_max_tokens': max_tokens,
                                      'max_tokens_note': 'Unset preserves native kwargs; an explicit cap is a declared inference condition, not an upstream default.',
                                      'served_runtime_limits': identity.get('runtime_limits')},
              'required_environment': {'TAU2_DATA_DIR': str(Path(setup).resolve() / 'source/data'),
                                       'OPENAI_API_KEY': 'native OpenAI user, judge and embedding credentials; secret value omitted',
                                       'OPENAI_BASE_URL': 'must be unset for the native public OpenAI user/judge/embedding routes; actor api_base is per-call'},
              'before_execution': ['Start an online W&B run and bind source, task manifest, exact served checkpoint and limits before any model call.',
                                   'Use a dedicated runtime with sandbox-runtime 0.0.23 and rg; Linux also needs bwrap and socat.',
                                   'Verify actual Qwen service weights and tool-call parsing; no actor service was contacted by this plan.',
                                   'Install a durable shared spending/request guard covering actor, user simulator, NL judge and embedding requests. Native task retries (3) and LiteLLM request retries (3) otherwise create additional billable work.',
                                   'Reconcile prior intents/receipts before starting or resuming. A native infrastructure-error placeholder is incomplete evaluation, not a model failure score.',
                                   'Keep all 97 original task, policy, tools, retrieval and evaluator bindings. Capture raw native records and per-call provider usage; no retrieval substitute.',
                                   'Bind the actual source files independently of native Info.git_commit, which reflects the process working directory and may be the surrounding project commit for an extracted source bundle.'],
              'execution_performed': False}
    write_once(output, result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path)
    p = sub.add_parser('plan'); p.add_argument('--manifest-dir', type=Path); p.add_argument('--model-identity', type=Path, required=True)
    p.add_argument('--actor-base-url', required=True); p.add_argument('--campaign-id', required=True)
    p.add_argument('--actor-max-tokens', type=int); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.setup, args.output or args.setup / 'prepared')
        print(json.dumps({'status': result['status'], 'task_count': result['task_count'], 'knowledge_documents': result['knowledge_document_count'], 'score': None}))
    else:
        result = plan(args.setup, args.manifest_dir or args.setup / 'prepared', read_json(args.model_identity), args.actor_base_url,
                      args.campaign_id, args.output, args.actor_max_tokens)
        print(json.dumps({'status': result['status'], 'output': str(args.output), 'score': None}))

if __name__ == '__main__':
    main()
