#!/usr/bin/env python3
"""E5 successor27 controller: drive the 27 never-started Tau3 banking tasks.

Rebuild of the expired sealed successor27 packet (outputs/PES_Phase2_Review_2026-09-12/
finish/e5_runtime/successor27_v6_27/, sealed 2026-09-12, expired 2026-09-13, controller
code under .codex-run/finish_20260912/e5_runtime_v6_27 deleted). This module lives in a
tracked path and performs NO network access and NO paid calls by itself: every actor
dispatch and native evaluation goes through a runtime adapter that the lead binds at
launch. Without an adapter the controller refuses to dispatch.

Per task, in original native order: readiness check -> actor request -> generation
(intent) receipt with the exact schema of the existing episode receipts -> native
evaluation -> append to the run journal. Concurrency is bounded at 3 and wall-clock
deadlines are enforced (host wall 21600s, actor function 21540s, native dispatch
stops 200s before the actor provider deadline, readiness window 800s).

Offline subcommands: verify / plan / gate. launch requires a lead authorization
receipt plus a runtime adapter module.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

SCHEMA = 'e5-successor27-controller-v1'
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- sealed packet pins (successor27_v6_27, still intact on disk) -------------
INVENTORY_PATH = REPO_ROOT / ('outputs/PES_Phase2_Review_2026-09-12/finish/'
                              'e5_runtime/successor27_v6_27/inventory.json')
INVENTORY_SHA256 = 'ad80e45e5da84fbcf988f572e2ad837a7573a2c70381af6874906988dd498368'
OLD_REQUEST_PATH = REPO_ROOT / ('outputs/PES_Phase2_Review_2026-09-12/finish/'
                                'e5_runtime/successor27_v6_27/resource_request.json')
OLD_REQUEST_SHA256 = 'a7739b1ea3b0234ade4100ae940da7149cb016759471b5c7df3d938496caf135'

# --- immutable native/model pins (identical to the lane runner) ---------------
LANE_RUNNER_PATH = REPO_ROOT / 'zvf-program/flagship/public_tau3_native.py'
PREPARED_MANIFEST_PATH = (REPO_ROOT / 'outputs/public_portfolio_2026-09-05/'
                          'tau3_setup/prepared/manifest.json')
PREPARED_MANIFEST_FINGERPRINT = ('5d73c1a0646298e9a5556f8b146aa88e6258ec928d74e657'
                                 'f4cec1f3e95a26eb')
TRIAL_SEED = 626729
NATIVE_SEED = 300
TOTAL = 97
IDENTITY = {
    'model_id': 'Qwen/Qwen3.6-35B-A3B',
    'model_revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
    'hf_repo': 'arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6',
    'hf_commit': '64444133c55d88c3f1bf0df8a2f5d7ac646125c8',
    'served_model_id': 'pavlov-public-portfolio-bf16',
}

# --- fresh request pins (resource_request_2026-09-19.json in this directory) --
REQUEST_PATH = Path(__file__).resolve().parent / 'resource_request_2026-09-19.json'
FRESH_ACTOR_ID = 'public0919-e5-actor11'
FRESH_NATIVE_ID = 'public0919-e5-native11'
FRESH_RUN_ID = 'E5-native-20260919-11'

# --- poison/native10 namespace that must never be reused ----------------------
PRIOR_ALLOCATION_PREFIX = 'public0912-e5-'
PRIOR_RUN_PREFIX = 'E5-native-20260912-'
FORBIDDEN_HELD_REQUEST_IDS = (1902, 1903)  # host-held requests inside native10
FORBIDDEN_RUN_RE = re.compile(r'^E5-native-20260912-\d+$')

# --- finite bounds (per-run; unlimited budget mode does not lift these) -------
HOST_WALL_SECONDS = 21600
ACTOR_FUNCTION_SECONDS = 21540
ACTOR_READY_SECONDS = 800
NATIVE_MARGIN_SECONDS = 200
MAX_NATIVE_CONCURRENCY = 3
MAX_ATTEMPTS_PER_TASK = 4
MAX_STEPS = 200
MAX_ERRORS = 10
TRIALS = 1
HOST_FLOOR_BYTES = 6442450944
REQUEST_VALIDITY_SECONDS = 86400

LIVE27 = [f'task_{n:03d}' for n in range(76, 103)]


class ControllerError(RuntimeError):
    """Base class for every controller refusal."""


class ScopeError(ControllerError):
    pass


class OutsideSuiteError(ScopeError):
    pass


class DuplicateTaskError(ScopeError):
    pass


class StartedTaskError(ScopeError):
    pass


class PartialSetError(ScopeError):
    pass


class ReorderedSetError(ScopeError):
    pass


class ChangedBindingError(ControllerError):
    pass


class FreshStartError(ControllerError):
    pass


class AuthorizationError(ControllerError):
    pass


class DeadlineExceeded(ControllerError):
    pass


class RuntimeBindingError(ControllerError):
    pass


class NativeEvaluationError(ControllerError):
    def __init__(self, message, exception_type=None):
        super().__init__(message)
        self.exception_type = exception_type or type(self).__name__


# --- primitives (same discipline as the lane runner) --------------------------
def require(ok, message):
    if not ok:
        raise ControllerError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(canonical(value))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    """Immutable canonical artifact; refuses to overwrite differing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)
    if path.exists():
        require(path.read_bytes() == raw, f'immutable artifact differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.e5s27-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == raw, f'immutable artifact differs: {path}')
    finally:
        os.unlink(temporary)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


class SystemClock:
    def now(self):
        return time.time()


class FakeClock:
    def __init__(self, start):
        self._now = float(start)

    def now(self):
        return self._now

    def advance(self, seconds):
        self._now += float(seconds)


# --- inventory scope ----------------------------------------------------------
def load_inventory(path=INVENTORY_PATH, expect_sha=INVENTORY_SHA256):
    path = Path(path)
    require(path.exists(), f'inventory missing: {path}')
    digest = sha(path.read_bytes())
    require(digest == expect_sha,
            f'inventory sha256 drift: {digest} != {expect_sha}')
    inv = json.loads(path.read_text())
    original = inv['original_task_ids']
    require(len(original) == TOTAL and len(set(original)) == TOTAL,
            'original suite must enumerate exactly 97 unique tasks')
    require(original == [t for t in original if t in original], 'original order malformed')
    excluded33 = inv['native10_excluded33']
    prior37 = inv['prior_excluded37']
    require(len(excluded33) == 33 and len(prior37) == 37, 'exclusion cardinality drift')
    require(not (set(excluded33) & set(prior37)), 'exclusion sets overlap')
    require(set(excluded33) | set(prior37) == set(inv['all_excluded70'])
            and len(inv['all_excluded70']) == 70, 'excluded70 partition drift')
    never = inv['never_started27']
    require(never == LIVE27, f'never_started27 drift: {never[:3]}...{never[-1:]}')
    require(set(original) == set(never) | set(excluded33) | set(prior37),
            'live27+excluded70 must partition the original 97')
    require([t for t in original if t in set(never)] == never,
            'live27 order must follow the original 97 order')
    cats = inv['native10_exclusion_categories']
    require(sorted(cats['graded20'] + cats['ungraded_infra10'] + cats['interrupted3'])
            == sorted(excluded33), 'native10 exclusion categories do not partition the 33')
    require(len(inv['all_start_refs']) == 178, 'all_start_refs cardinality drift (178 pinned)')
    require(len({r['path'] for r in inv['all_start_refs']}) == 178, 'duplicate start ref path')
    held = {h['id'] for h in inv.get('host_held_requests', [])}
    require(held <= set(FORBIDDEN_HELD_REQUEST_IDS), 'unexpected host-held request ids')
    return inv


def select_live(inventory, requested=None):
    """Exact live-27 selection with old/started/duplicate/outside/reorder/partial refusal."""
    original = inventory['original_task_ids']
    excluded = set(inventory['all_excluded70'])
    if requested is None:
        requested = list(inventory['never_started27'])
    require(isinstance(requested, list) and all(isinstance(t, str) for t in requested),
            'requested scope must be a list of task IDs')
    if not requested:
        raise PartialSetError('requested scope must be exactly 27 tasks, got 0')
    seen = set()
    for task_id in requested:
        if task_id not in set(original):
            raise OutsideSuiteError(f'task outside original 97-task suite: {task_id}')
        if task_id in seen:
            raise DuplicateTaskError(f'duplicate task ID in requested scope: {task_id}')
        seen.add(task_id)
        if task_id in excluded:
            category = 'native10_started' if task_id in set(inventory['native10_excluded33']) else 'prior37_started'
            raise StartedTaskError(f'requested task was already started ({category}): {task_id}')
    if len(requested) != 27:
        raise PartialSetError(f'requested scope must be exactly 27 tasks, got {len(requested)}')
    expected = [t for t in original if t in set(requested)]
    if requested != expected:
        raise ReorderedSetError('requested scope must preserve the original 97 order')
    if requested != list(inventory['never_started27']):
        raise ScopeError('requested scope differs from the sealed never-started 27')
    return list(requested)


def verify_start_refs(inventory, repo_root=REPO_ROOT):
    """Rescan every original native start; any change or new start invalidates the packet."""
    repo_root = Path(repo_root)
    for ref in inventory['all_start_refs']:
        path = Path(ref['path'])
        require(path.is_relative_to(repo_root) and not path.is_symlink(),
                f'unsafe start ref path: {path}')
        if not path.exists():
            raise ChangedBindingError(f'original start receipt vanished: {path}')
        digest = sha(path.read_bytes())
        if digest != ref['sha256']:
            raise ChangedBindingError(f'original start receipt changed: {path}')
    runs_root = repo_root / ('outputs/PES_Phase2_Review_2026-09-12/finish/e5_runtime/'
                             'controller_runs')
    refs = {str(Path(r['path']).resolve()) for r in inventory['all_start_refs']}
    on_disk = set()
    for run_dir in sorted(runs_root.glob(f'{PRIOR_RUN_PREFIX}*')):
        if not run_dir.is_dir():
            continue
        for path in run_dir.glob('files/journal/episodes/*.intent.json'):
            on_disk.add(str(path.resolve()))
    new_starts = on_disk - refs
    require(not new_starts,
            f'new native start(s) since seal invalidate the packet: {sorted(new_starts)[:3]}')
    missing = refs - on_disk
    require(not missing, f'pinned start receipts missing from prior runs: {sorted(missing)[:3]}')
    return {'start_refs_verified': len(refs), 'episode_starts_on_disk': len(on_disk)}


def load_prepared_manifest(path=PREPARED_MANIFEST_PATH,
                           expect_fingerprint=PREPARED_MANIFEST_FINGERPRINT):
    manifest = read_json(path)
    require(fingerprint(manifest) == expect_fingerprint, 'prepared manifest fingerprint drift')
    require(manifest['task_count'] == TOTAL and len(manifest['tasks']) == TOTAL,
            'prepared manifest task count drift')
    require(manifest['native_trial_seed'] == TRIAL_SEED, 'trial seed drift')
    require(all(row['score'] is None for row in manifest['tasks']),
            'prepared manifest cannot contain scores')
    return manifest


def build_plan(inventory, manifest, requested=None):
    live = select_live(inventory, requested)
    rows = {row['task_id']: row for row in manifest['tasks']}
    return [{'task_id': t, 'evaluation_id': rows[t]['evaluation_id'],
             'task_sha256': rows[t]['task_sha256'],
             'initial_state_sha256': rows[t]['initial_state_sha256'],
             'seed': TRIAL_SEED} for t in live]


# --- deadline math ------------------------------------------------------------
class Deadlines:
    """Host wall 21600s; actor function 21540s; native dispatch stops 200s earlier."""

    def __init__(self, launch_epoch):
        self.launch_epoch = float(launch_epoch)
        self.host_wall = self.launch_epoch + HOST_WALL_SECONDS
        self.actor_function = self.launch_epoch + ACTOR_FUNCTION_SECONDS
        self.native_dispatch = self.actor_function - NATIVE_MARGIN_SECONDS
        self.actor_ready = self.launch_epoch + ACTOR_READY_SECONDS
        require(self.native_dispatch < self.actor_function < self.host_wall,
                'deadline invariant violated')
        require(self.actor_function - self.native_dispatch == NATIVE_MARGIN_SECONDS,
                'native margin must stay 200s')

    def remaining_host(self, clock):
        return self.host_wall - clock.now()

    def remaining_native(self, clock):
        return self.native_dispatch - clock.now()

    def assert_within(self, clock, deadline_name='native_dispatch'):
        limit = getattr(self, deadline_name)
        now = clock.now()
        if now >= limit:
            raise DeadlineExceeded(
                f'{deadline_name} exceeded: now={now:.3f} limit={limit:.3f}')

    def as_dict(self):
        return {'launch_epoch': self.launch_epoch, 'host_wall': self.host_wall,
                'actor_function': self.actor_function,
                'native_dispatch': self.native_dispatch, 'actor_ready': self.actor_ready}


# --- fresh-start guard --------------------------------------------------------
def check_fresh_start(request, repo_root=REPO_ROOT):
    """Refuse any reuse of the consumed native10 supervisor/IDs/poison state."""
    repo_root = Path(repo_root)
    run_id = request['requested_run_id']
    ids = request['requested_allocation_ids']
    require(not FORBIDDEN_RUN_RE.match(run_id) and not run_id.startswith(PRIOR_RUN_PREFIX),
            f'refusing native10 run namespace reuse: {run_id}')
    require(run_id == FRESH_RUN_ID, f'unexpected run id: {run_id}')
    require(ids.get('actor') == FRESH_ACTOR_ID and ids.get('native') == FRESH_NATIVE_ID,
            'allocation ids must be the fresh 0919 pair')
    for role, alloc_id in ids.items():
        require(not alloc_id.startswith(PRIOR_ALLOCATION_PREFIX),
                f'refusing consumed native10/actor allocation reuse: {alloc_id}')
    claims_dir = repo_root / ('outputs/PES_Phase2_Review_2026-09-12/finish/e5_runtime/'
                             'allocation_claims')
    if claims_dir.is_dir():
        for claim in claims_dir.glob('*.claim.json'):
            text = claim.read_text()
            require(FRESH_ACTOR_ID not in text and FRESH_NATIVE_ID not in text,
                    f'fresh allocation id already claimed: {claim.name}')
    runs_root = repo_root / ('outputs/PES_Phase2_Review_2026-09-12/finish/e5_runtime/'
                             'controller_runs')
    require(not (runs_root / run_id).exists(),
            f'native output already exists for fresh run: {runs_root / run_id}')
    paths = request['proposed_output_paths']
    for name in ('actor_output', 'native_output'):
        require(not Path(paths[name]).exists(), f'{name} must be absent before launch: {paths[name]}')
    require(request.get('resume') is not True and not request.get('automatic_resume_authorized'),
            'no resume: every started ID stays excluded')
    return True


# --- authorization gating -----------------------------------------------------
def assess_launch_authorization(request, request_path, authorization_path, clock):
    """Launch only under a fresh lead authorization binding this exact request."""
    require(request.get('status') == 'SEALED_REQUEST_NOT_AUTHORIZED' and
            request.get('authorized') is False,
            'request must remain a sealed, unauthorized request at gate time')
    now = clock.now()
    valid_until = request['request_valid_until_epoch']
    require(request['at'] + REQUEST_VALIDITY_SECONDS == valid_until,
            'request validity must be 24h from seal')
    if now >= valid_until:
        raise AuthorizationError(f'sealed request expired at {valid_until} (now={now:.3f})')
    if now < request['at']:
        raise AuthorizationError('authorization cannot precede the sealed request')
    auth_path = Path(authorization_path)
    if not auth_path.exists():
        raise AuthorizationError(f'missing lead authorization receipt: {auth_path}')
    auth = read_json(auth_path)
    request_digest = sha(Path(request_path).read_bytes())
    if auth.get('authorized') is not True:
        raise AuthorizationError('lead authorization receipt is not authorized:true')
    if auth.get('request_sha256') != request_digest:
        raise AuthorizationError('authorization does not bind this request sha256')
    if auth.get('requested_run_id') != request['requested_run_id'] or \
            auth.get('requested_allocation_ids') != request['requested_allocation_ids']:
        raise AuthorizationError('authorization ids/run mismatch')
    issued = auth.get('issued_at_epoch')
    require(isinstance(issued, (int, float)) and not isinstance(issued, bool),
            'authorization issued_at_epoch missing')
    if issued < request['at']:
        raise AuthorizationError('authorization issued before the request was sealed')
    if now >= issued + REQUEST_VALIDITY_SECONDS:
        raise AuthorizationError('lead authorization expired (24h validity)')
    if auth.get('resume') is True or auth.get('replay') is True:
        raise AuthorizationError('authorization may not permit resume/replay')
    return {'admitted': True, 'request_sha256': request_digest,
            'authorization_path': str(auth_path), 'issued_at_epoch': issued,
            'valid_until_epoch': issued + REQUEST_VALIDITY_SECONDS}


# --- runtime adapter (the only place execution can bind) ----------------------
class NativeRuntimeAdapter:
    """Base adapter: refuses everything. The lead binds a real adapter at launch.

    A concrete adapter must implement readiness(), actor_request() and
    evaluate_native(); it owns the actor serving path (guarded loopback
    OpenAI-compatible route for the exact BF16 checkpoint) and the native tau2
    evaluation inside the retained E5 runtime. This module alone never
    dispatches.
    """

    def readiness(self, deadline):
        raise RuntimeBindingError('actor/native runtime adapter not bound')

    def actor_request(self, task_id, attempt, deadline):
        raise RuntimeBindingError('actor serving path not bound; no dispatch possible')

    def evaluate_native(self, task_id, attempt, deadline):
        raise RuntimeBindingError('native evaluation path not bound')


# --- run journal --------------------------------------------------------------
class RunJournal:
    """Append-only JSONL journal with fsync before returning."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, event, clock, **fields):
        record = {'seq': self._next_seq(), 'at': clock.now(), 'event': event}
        record.update(fields)
        raw = canonical(record)
        with self._lock:
            with open(self.path, 'ab') as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
        return record

    def _next_seq(self):
        with self._lock:
            self._seq += 1
            return self._seq

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


# --- controller ---------------------------------------------------------------
class ControllerConfig:
    def __init__(self, request_path=REQUEST_PATH, inventory_path=INVENTORY_PATH,
                 inventory_sha=INVENTORY_SHA256, repo_root=REPO_ROOT,
                 authorization_path=None, adapter=None, clock=None,
                 concurrency=MAX_NATIVE_CONCURRENCY):
        self.request_path = Path(request_path)
        self.inventory_path = Path(inventory_path)
        self.inventory_sha = inventory_sha
        self.repo_root = Path(repo_root)
        self.authorization_path = Path(authorization_path) if authorization_path else None
        self.adapter = adapter
        self.clock = clock or SystemClock()
        require(isinstance(concurrency, int) and 1 <= concurrency <= MAX_NATIVE_CONCURRENCY,
                f'native concurrency must stay within 1..{MAX_NATIVE_CONCURRENCY}')
        self.concurrency = concurrency


class Successor27Controller:
    def __init__(self, config):
        self.config = config
        self.request = read_json(config.request_path)
        self.deadlines = None
        self.stop = threading.Event()

    # -- offline verification --------------------------------------------------
    def verify_packet(self):
        inventory = load_inventory(self.config.inventory_path, self.config.inventory_sha)
        refs = verify_start_refs(inventory, self.config.repo_root)
        manifest = load_prepared_manifest()
        check_fresh_start(self.request, self.config.repo_root)
        plan = build_plan(inventory, manifest)
        return {'schema': SCHEMA, 'status': 'PACKET_VERIFIED',
                'inventory_sha256': self.config.inventory_sha,
                'live_tasks': [t['task_id'] for t in plan],
                'subset_denominator': len(plan), 'full_suite_denominator': TOTAL,
                'start_refs': refs, 'full_suite_score': None}

    # -- launch ----------------------------------------------------------------
    def launch(self):
        packet = self.verify_packet()
        if self.config.authorization_path is None:
            raise AuthorizationError('launch requires a lead authorization receipt')
        gate = assess_launch_authorization(self.request, self.config.request_path,
                                           self.config.authorization_path, self.config.clock)
        if self.config.adapter is None:
            raise RuntimeBindingError(
                'launch requires a runtime adapter (actor serving path + native evaluation); '
                'refusing to dispatch without one')
        self.deadlines = Deadlines(self.config.clock.now())
        inventory = load_inventory(self.config.inventory_path, self.config.inventory_sha)
        manifest = load_prepared_manifest()
        plan = build_plan(inventory, manifest)
        return self.execute(plan, packet, gate)

    def execute(self, plan, packet=None, gate=None):
        require(self.deadlines is not None, 'deadlines must be set before execution')
        native_output = Path(self.request['proposed_output_paths']['native_output'])
        require(not native_output.exists(), f'native output must be absent: {native_output}')
        journal_dir = native_output / 'files' / 'journal'
        episodes_dir = journal_dir / 'episodes'
        journal = RunJournal(journal_dir / 'run_journal.jsonl')
        write_once(journal_dir / 'contract.json', {
            'schema': SCHEMA, 'automatic_resume_authorized': False,
            'expected_episodes': len(plan), 'subset_denominator': len(plan),
            'full_suite_denominator': TOTAL, 'model_identity': IDENTITY,
            'seed': NATIVE_SEED, 'trial_seed': TRIAL_SEED,
            'max_attempts_per_task': MAX_ATTEMPTS_PER_TASK,
            'max_native_concurrency': self.config.concurrency,
            'deadlines': self.deadlines.as_dict(),
            'authorization': gate, 'packet': packet and {
                'inventory_sha256': packet['inventory_sha256'],
                'start_refs': packet['start_refs']},
            'generation_receipt_schema': ['attempt', 'recorded_at', 'seed', 'task_id'],
            'model_calls': 0, 'provider_calls': 0})
        journal.append('launch', self.config.clock,
                       run_id=self.request['requested_run_id'],
                       allocation_ids=self.request['requested_allocation_ids'],
                       deadlines=self.deadlines.as_dict())
        outcomes = {}
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = {task['task_id']: pool.submit(self._execute_task, task,
                                                    journal, episodes_dir)
                       for task in plan}
            for task_id, future in futures.items():
                outcomes[task_id] = future.result()
        summary = self._summarize(plan, outcomes, journal)
        write_once(native_output / 'run_summary.json', summary)
        journal.append('run_complete', self.config.clock,
                       run_id=self.request['requested_run_id'],
                       counts=summary['counts'], full_suite_score=None)
        return summary

    def _execute_task(self, task, journal, episodes_dir):
        task_id = task['task_id']
        if self.stop.is_set():
            journal.append('task_skipped_halted', self.config.clock, task_id=task_id)
            return {'task_id': task_id, 'status': 'skipped_halted'}
        clock = self.config.clock
        for attempt in range(1, MAX_ATTEMPTS_PER_TASK + 1):
            if self.stop.is_set():
                journal.append('task_skipped_halted', clock, task_id=task_id, attempt=attempt)
                return {'task_id': task_id, 'status': 'skipped_halted'}
            try:
                self.deadlines.assert_within(clock, 'native_dispatch')
                journal.append('readiness_check', clock, task_id=task_id, attempt=attempt)
                self.config.adapter.readiness(
                    min(self.deadlines.actor_ready, self.deadlines.native_dispatch))
                journal.append('readiness_ok', clock, task_id=task_id, attempt=attempt)
                # generation (intent) receipt barrier: durable BEFORE any dispatch
                receipt = {'attempt': attempt, 'recorded_at': utc_now_iso(),
                           'seed': task['seed'], 'task_id': task_id}
                write_once(episodes_dir / f'{task_id}.{attempt}.intent.json', receipt)
                journal.append('intent_recorded', clock, task_id=task_id, attempt=attempt,
                               receipt_sha256=fingerprint(receipt))
                self.deadlines.assert_within(clock, 'native_dispatch')
                dispatch = self.config.adapter.actor_request(
                    task_id, attempt, self.deadlines.native_dispatch)
                journal.append('actor_dispatched', clock, task_id=task_id, attempt=attempt,
                               dispatch=dispatch)
                native = self.config.adapter.evaluate_native(
                    task_id, attempt, self.deadlines.native_dispatch)
                write_once(episodes_dir / f'{task_id}.{attempt}.native.json', native)
                projection = project_native(task, attempt, native)
                journal.append('native_recorded', clock, **projection)
                return {'task_id': task_id, 'status': 'recorded', 'attempt': attempt,
                        'projection': projection}
            except DeadlineExceeded as exc:
                self.stop.set()
                journal.append('deadline_halt', clock, task_id=task_id, attempt=attempt,
                               error=str(exc))
                return {'task_id': task_id, 'status': 'halted_deadline', 'attempt': attempt,
                        'error': str(exc)}
            except Exception as exc:  # receipt then bounded retry; deadlines handled above
                exception_type = getattr(exc, 'exception_type', None) or type(exc).__name__
                write_once(episodes_dir / f'{task_id}.{attempt}.known_failure.json',
                           {'exception_type': exception_type,
                            'recorded_at': utc_now_iso()})
                journal.append('known_failure', clock, task_id=task_id, attempt=attempt,
                               exception_type=exception_type)
                if attempt == MAX_ATTEMPTS_PER_TASK:
                    return {'task_id': task_id, 'status': 'failed_terminal',
                            'attempts': attempt, 'exception_type': exception_type}
        raise ControllerError('unreachable attempt loop')

    def _summarize(self, plan, outcomes, journal):
        recorded = sum(1 for o in outcomes.values() if o['status'] == 'recorded')
        counts = {
            'planned': len(plan), 'recorded': recorded,
            'failed_terminal': sum(1 for o in outcomes.values()
                                   if o['status'] == 'failed_terminal'),
            'halted_deadline': sum(1 for o in outcomes.values()
                                   if o['status'] == 'halted_deadline'),
            'skipped_halted': sum(1 for o in outcomes.values()
                                  if o['status'] == 'skipped_halted'),
        }
        require(counts['planned'] == 27, 'plan cardinality must stay 27')
        return {'schema': SCHEMA, 'run_id': self.request['requested_run_id'],
                'status': 'TERMINAL_INCOMPLETE_IF_ANY_NOT_RECORDED' if recorded != 27
                          else 'ALL_27_RECORDED_PENDING_COLLECTION',
                'counts': counts, 'subset_denominator': 27,
                'full_suite_denominator': TOTAL, 'full_suite_score': None,
                'subset_score': None,
                'score_note': 'Scores stay null until the terminal native collection '
                              'promotes grades; this controller never pools scores.',
                'journal_records': len(journal.records())}


def project_native(task, attempt, native):
    """Native projection for the journal; scores stay null until collection."""
    require(isinstance(native, dict), 'native evaluation must return a record')
    return {'task_id': task['task_id'], 'attempt': attempt,
            'evaluation_id': task['evaluation_id'],
            'native_duration': native.get('duration'),
            'exception_type': native.get('exception_type'),
            'score': None}


# --- CLI ----------------------------------------------------------------------
def _cmd_verify(args):
    controller = Successor27Controller(ControllerConfig(
        request_path=args.request, inventory_path=args.inventory))
    print(json.dumps(controller.verify_packet(), indent=1, sort_keys=True))
    return 0


def _cmd_plan(args):
    controller = Successor27Controller(ControllerConfig(
        request_path=args.request, inventory_path=args.inventory))
    inventory = load_inventory(args.inventory)
    plan = build_plan(inventory, load_prepared_manifest())
    launch_epoch = args.launch_epoch if args.launch_epoch is not None else time.time()
    deadlines = Deadlines(launch_epoch)
    print(json.dumps({'schema': SCHEMA, 'status': 'PLAN_READY_NOT_EXECUTED',
                      'tasks': plan, 'task_count': len(plan),
                      'deadlines': deadlines.as_dict(), 'score': None,
                      'model_calls': 0, 'provider_calls': 0}, indent=1, sort_keys=True))
    return 0


def _cmd_gate(args):
    request = read_json(args.request)
    gate = assess_launch_authorization(request, args.request, args.authorization,
                                       SystemClock())
    print(json.dumps(gate, indent=1, sort_keys=True))
    return 0


def _cmd_launch(args):
    adapter = None
    if args.runtime_adapter:
        module_dir = Path(args.runtime_adapter).resolve().parent
        module_name = Path(args.runtime_adapter).resolve().stem
        sys.path.insert(0, str(module_dir)) if str(module_dir) not in sys.path else None
        module = importlib.import_module(module_name)
        adapter = module.create_adapter()
    config = ControllerConfig(request_path=args.request, inventory_path=args.inventory,
                              authorization_path=args.authorization, adapter=adapter)
    controller = Successor27Controller(config)
    summary = controller.launch()
    print(json.dumps({'status': summary['status'], 'counts': summary['counts'],
                      'full_suite_score': None}, indent=1, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=REQUEST_PATH)
    parser.add_argument('--inventory', type=Path, default=INVENTORY_PATH)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    plan_cmd = sub.add_parser('plan')
    plan_cmd.add_argument('--launch-epoch', type=float, default=None)
    gate_cmd = sub.add_parser('gate')
    gate_cmd.add_argument('--authorization', type=Path, required=True)
    launch_cmd = sub.add_parser('launch')
    launch_cmd.add_argument('--authorization', type=Path, required=True)
    launch_cmd.add_argument('--runtime-adapter', default=None,
                            help='module path providing create_adapter(); '
                                 'the actor serving path binds here')
    args = parser.parse_args(argv)
    handlers = {'verify': _cmd_verify, 'plan': _cmd_plan,
                'gate': _cmd_gate, 'launch': _cmd_launch}
    try:
        return handlers[args.command](args)
    except ControllerError as exc:
        print(json.dumps({'status': 'REFUSED', 'error_type': type(exc).__name__,
                          'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
