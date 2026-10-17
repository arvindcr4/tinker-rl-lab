"""Offline security regressions; synthetic data and fixture executables only."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location('review48_' + Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('value', ['0', '17', "1'];__import__('os').system('false');#", 'nan'])
def test_colab_launch_rejects_invalid_concurrency(value):
    ctl = load('platform_colab/publication_fixes/colab_ctl.py')
    with pytest.raises(ValueError):
        ctl.launch_code('/tmp/jobs.json', value)


def test_colab_launch_path_stays_literal(monkeypatch):
    ctl = load('platform_colab/publication_fixes/colab_ctl.py')
    value = "a'\n);raise RuntimeError('injected');#"
    tree = ast.parse(ctl.launch_code(value, 2))
    assert len(tree.body) == 3
    call = tree.body[1].value
    assert ast.literal_eval(call.args[0]) == ['python', '/content/pf/vm_queue.py', value, '2']


def test_colab_status_metacharacters_are_not_shell(tmp_path, monkeypatch, capsys):
    ctl = load('platform_colab/publication_fixes/colab_ctl.py')
    directory = tmp_path / "logs; echo injected; '$(false)'"
    directory.mkdir()
    (directory / 'status.json').write_text('{"state":"fixture"}')
    (directory / 'worker.log').write_text('fixture output')
    calls = []
    def fake(command, **kwargs):
        assert not kwargs.get('shell')
        assert kwargs['timeout'] == 30
        calls.append(command)
        return SimpleNamespace(stdout='fixture GPU')
    monkeypatch.setattr(subprocess, 'run', fake)
    exec(compile(ctl.status_code(str(directory)), '<fixture>', 'exec'), {})
    assert len(calls) == 1
    assert 'fixture output' in capsys.readouterr().out


def test_colab_cli_failure_is_not_success(monkeypatch):
    ctl = load('platform_colab/publication_fixes/colab_ctl.py')
    def fake(command, **kwargs):
        assert kwargs['check'] and kwargs['timeout'] == 720
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(subprocess, 'run', fake)
    with pytest.raises(subprocess.CalledProcessError):
        ctl.exec_py('fixture', 'pass')


def test_queue_concurrency_ceiling():
    queue = load('platform_colab/publication_fixes/vm_queue.py')
    with pytest.raises(ValueError):
        queue.validate_jobs([], 17)


@pytest.mark.parametrize('duration', ['25h', '86401s', '1441m'])
def test_gcp_rejects_excessive_runtime(duration):
    gcp = load('platform_gcp/run_unified_dispatch.py')
    with pytest.raises(ValueError, match='ceiling'):
        gcp.build_startup_script('pass', max_run_duration=duration)


def test_gcp_all_lifecycle_calls_have_deadlines(monkeypatch, tmp_path):
    gcp = load('platform_gcp/run_unified_dispatch.py')
    calls = []
    state = {}
    def fake(cmd, **kwargs):
        assert kwargs['timeout'] == gcp.GCLOUD_TIMEOUT_S
        calls.append(cmd)
        if 'create' in cmd:
            state['run_id'] = next(v.split('=', 2)[2] for v in cmd if v.startswith('--labels='))
            return SimpleNamespace(stdout='[{"id":"owned"}]', returncode=0)
        if 'describe' in cmd:
            return SimpleNamespace(stdout=json.dumps({'id': 'owned', 'labels': {'dispatch-run': state['run_id']}}), returncode=0)
        return SimpleNamespace(stdout='', returncode=0)
    monkeypatch.setattr(gcp, 'gcloud_binary', lambda: 'fixture')
    monkeypatch.setattr(gcp.subprocess, 'run', fake)
    monkeypatch.setattr(gcp, '_wait_for_receipt', lambda *a: {'exit_code': 0, 'status': 'process_completed'})
    args = gcp.parse_args(['--framework','tinker','--repo-revision','a'*40,'--receipt-dir',str(tmp_path)])
    assert gcp.run_unit(args)['exit_code'] == 0
    assert len(calls) == 3


def test_gcp_ambiguous_timeout_does_not_delete(monkeypatch):
    gcp = load('platform_gcp/run_unified_dispatch.py')
    calls = []
    def timeout(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
    monkeypatch.setattr(gcp, 'gcloud_binary', lambda: 'fixture')
    monkeypatch.setattr(gcp.subprocess, 'run', timeout)
    args = gcp.parse_args(['--framework','tinker','--repo-revision','a'*40])
    with pytest.raises(subprocess.TimeoutExpired):
        gcp.run_unit(args)
    assert len(calls) == 1 and 'create' in calls[0]
    startup = next(v.split('=',2)[2] for v in calls[0] if v.startswith('--metadata-from-file='))
    assert not Path(startup).exists()


@pytest.mark.parametrize('bucket', ['bad"; touch /tmp/should-not-exist; #', '$(false)', 'bare-bucket'])
def test_builder_rejects_bucket_source_injection_before_cloud(tmp_path, bucket):
    # A fixture executable is mandatory even though rejection precedes invocation.
    fake = tmp_path / 'gcloud'
    sentinel = tmp_path / 'cloud-called'
    fake.write_text('#!/bin/sh\ntouch "$SENTINEL"\nexit 99\n')
    fake.chmod(0o700)
    env = {**os.environ, 'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],
           'SENTINEL':str(sentinel), 'GCP_PROJECT':'fixture-project', 'WEBARENA_TAR_GCS_BUCKET':bucket}
    proc = subprocess.run(['bash', str(ROOT/'platform_gcp/infra/gcp/build_webarena_image.sh')], env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 2
    assert not sentinel.exists()


def test_builder_supply_chain_and_lifetime_guards():
    text = (ROOT/'platform_gcp/infra/gcp/build_webarena_image.sh').read_text()
    assert 'get.docker.com' not in text
    assert '--max-run-duration=4h' in text
    assert '--instance-termination-action=DELETE' in text
    assert 'current" = "$BUILDER_ID' in text
    assert 'cat > "$SETUP_FILE"' in text
    assert 'timeout --kill-after=30s 7200s gcloud' in text


@pytest.mark.parametrize('setting,value', [('NUM_WORKERS','11'), ('DISK_GB','1001'), ('MAX_RUN_SECONDS','86401'), ('MACHINE_TYPE','a3-highgpu-8g')])
def test_mig_resource_caps_precede_cloud(tmp_path, setting, value):
    import hashlib
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps({'schema':'webarena-service-contract-v1',
        'benchmark':'webarena_verified', 'tasks':[{'id':'fixture.0','required_sites':['shopping']}]}))
    fake = tmp_path / 'gcloud'
    sentinel = tmp_path / 'cloud-called'
    fake.write_text('#!/bin/sh\ntouch "$SENTINEL"\nexit 99\n')
    fake.chmod(0o700)
    env = {**os.environ, 'PATH':str(tmp_path)+':/tmp/tinker-review-venv/bin:/usr/bin:/bin',
        'SENTINEL':str(sentinel), 'GCP_PROJECT':'fixture', 'BUCKET':'fixture',
        'WEBARENA_TASK_MANIFEST':str(manifest),
        'WEBARENA_TASK_MANIFEST_SHA256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'TINKER_REPO_REVISION':'a'*40, setting:value}
    proc = subprocess.run(['bash',str(ROOT/'platform_gcp/infra/gcp/deploy_mig.sh')],
        env=env,capture_output=True,text=True,timeout=10)
    assert proc.returncode == 2
    assert not sentinel.exists()


def test_builder_generated_setup_and_owned_cleanup_offline(tmp_path):
    import hashlib
    import sys
    log = tmp_path/'calls.jsonl'
    setup = tmp_path/'setup.sh'
    fake = tmp_path/'gcloud'
    fake.write_text(f'''#!{sys.executable}
import json,pathlib,sys
args=sys.argv[1:]
with open({str(log)!r},'a') as f: f.write(json.dumps(args)+'\\n')
if 'describe' in args: print('owned-fixture-id')
if 'scp' in args and any(a.endswith(':/tmp/webarena_setup.sh') for a in args):
    pathlib.Path({str(setup)!r}).write_text(pathlib.Path(args[2]).read_text())
''')
    fake.chmod(0o700)
    manifest=tmp_path/'hashes.txt'
    manifest.write_text('fixture manifest\n')
    env={**os.environ, 'PATH':str(tmp_path)+':/tmp/tinker-review-venv/bin:/usr/bin:/bin',
        'GCP_PROJECT':'fixture', 'WEBARENA_TAR_GCS_BUCKET':'gs://fixture-bucket/reviewed-prefix',
        'WEBARENA_SHA256_MANIFEST':str(manifest),
        'WEBARENA_SHA256_ANCHOR':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'KIWIX_IMAGE':'ghcr.io/kiwix/kiwix-serve@sha256:'+'a'*64,
        'TINKER_REPO_REVISION':'a'*40, 'WEBARENA_REPO_REVISION':'b'*40}
    subprocess.run(['bash',str(ROOT/'platform_gcp/infra/gcp/build_webarena_image.sh')],
        env=env,capture_output=True,text=True,timeout=10,check=True)
    subprocess.run(['bash','-n',str(setup)],check=True)
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    create=next(call for call in calls if call[:3]==['compute','instances','create'])
    assert '--max-run-duration=4h' in create and '--instance-termination-action=DELETE' in create
    assert calls[-1][:3]==['compute','instances','delete']
    uploaded=next(call[2] for call in calls if 'scp' in call and any(a.endswith(':/tmp/webarena_setup.sh') for a in call))
    assert not Path(uploaded).exists()
