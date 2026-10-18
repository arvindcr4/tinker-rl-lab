"""Offline synthetic counterexamples, never authentic execution evidence."""
import ast
import json
import socket
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import mle_bench_eval as mle
import modal_e1_swe_bench_pro_full as runner


def test_sampling_request_excludes_every_poisoned_gold_field():
    public = ('instance_id', 'repo', 'base_commit', 'problem_statement', 'requirements', 'interface', 'repo_language', 'dockerhub_tag')
    forbidden = ('patch', 'test_patch', 'fail_to_pass', 'pass_to_pass', 'selected_test_files_to_run', 'before_repo_set_cmd')
    row = {key: 'public-' + key for key in public}
    row.update({key: 'SENSITIVE_' + key for key in forbidden})
    task = runner._sanitize_task(row)
    tree = ast.parse(Path(runner.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_candidate')
    node.decorator_list = []
    namespace = dict(vars(runner))
    captured = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return json.dumps(messages)
        def encode(self, value, **kwargs): return list(value.encode())
        def decode(self, value, **kwargs): return ''
    class Sampler:
        def sample(self, request, **kwargs):
            captured.append(bytes(request).decode())
            return types.SimpleNamespace(result=lambda: types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[])]))
    run = Mock(id='fixture', url='https://fixture.invalid', summary={})
    tinker = types.ModuleType('tinker'); ttypes = types.ModuleType('tinker.types')
    ttypes.ModelInput = types.SimpleNamespace(from_ints=lambda values: values)
    ttypes.SamplingParams = lambda **kwargs: kwargs
    tinker.types = ttypes
    wandb = types.SimpleNamespace(init=lambda **kwargs: run)
    namespace['_verified_tokenizer'] = Tokenizer
    namespace['_shared_tinker_sampler'] = lambda *args: Sampler()
    exec(compile(ast.Module(body=[node], type_ignores=[]), runner.__file__, 'exec'), namespace)
    with patch.dict(sys.modules, {'tinker': tinker, 'tinker.types': ttypes, 'wandb': wandb}):
        result = namespace['generate_candidate']({'task': task, 'sources': {}, 'seed': 1, 'index': 0, 'max_tokens': 1, 'temperature': 0})
    assert result['sample_completed'] is True
    assert len(captured) == 1
    for key in forbidden:
        assert 'SENSITIVE_' + key not in captured[0]


@pytest.mark.parametrize('revision,status', [('0'*40, ''), (mle.UPSTREAM_COMMIT, ' M mlebench/grade.py'), (mle.UPSTREAM_COMMIT, '?? forged.py')])
def test_actual_source_revision_and_modifications_fail_closed(tmp_path, revision, status):
    with patch.object(mle.subprocess, 'run', side_effect=[types.SimpleNamespace(stdout=revision), types.SimpleNamespace(stdout=status)]):
        assert not mle.verify_source_checkout(tmp_path)['verified']


def test_absent_source_evidence_fails_closed(tmp_path):
    assert not mle.verify_source_checkout(tmp_path)['verified']


@pytest.mark.parametrize('name', ['pavlov_browsergym_receipt', 'pavlov_lifescibench_eval_adapter'])
def test_validator_import_and_empty_input_without_network(name):
    # Fresh process: install failure guards before importing the target or helpers.
    code = '''import socket, importlib, urllib.request
from unittest.mock import patch
with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")), patch.object(urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
    module = importlib.import_module(NAME)
    result = module.validate_receipt({}) if hasattr(module, "validate_receipt") else module.validate_e8_boundary({})
'''.replace('NAME', repr(name))
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
