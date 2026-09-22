#!/usr/bin/env python3
"""14 offline test classes for the rebuilt E5 successor27 controller.

Reimplementation of the sealed successor27 packet's tests.json (4 new-scope
tests + 10 subset-projection/diagnostic tests) against the rebuilt controller
in this directory. Everything runs offline: sockets are blocked for the whole
run, no paid calls, no writes outside a temp directory. Run with:

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests_offline -v
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import unittest

import controller as ctrl

HERE = Path(__file__).resolve().parent
REAL_INVENTORY = ctrl.INVENTORY_PATH
REAL_REQUEST = ctrl.REQUEST_PATH
REAL_EPISODES_ROOT = (ctrl.REPO_ROOT / 'outputs/PES_Phase2_Review_2026-09-12/finish/'
                      'e5_runtime/controller_runs')
REAL_PRIOR_RECEIPT = (REAL_EPISODES_ROOT / 'E5-native-20260912-10/files/journal/episodes/'
                      'task_075.1.intent.json')
FIXTURE_AT = 1789800000.0


def block_sockets(test):
    """Guarantee offline execution: any connect attempt fails the test."""

    def deny(*args, **kwargs):
        raise AssertionError('network access attempted during offline tests')

    original = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)

    def wrapped(self):
        socket.socket.connect = deny
        socket.socket.connect_ex = deny
        socket.create_connection = deny
        try:
            test(self)
        finally:
            socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original

    return wrapped


class RepoFixture:
    """Temp repository root with the sealed inventory's start refs materialized."""

    def __init__(self, at=FIXTURE_AT):
        self.root = Path(tempfile.mkdtemp(prefix='e5s27-test-'))
        self.runs_root = self.root / ('outputs/PES_Phase2_Review_2026-09-12/finish/'
                                      'e5_runtime/controller_runs')
        self.claims_root = self.root / ('outputs/PES_Phase2_Review_2026-09-12/finish/'
                                        'e5_runtime/allocation_claims')
        self.claims_root.mkdir(parents=True)
        (self.claims_root / 'prior.claim.json').write_text(json.dumps(
            {'allocation_ids': ['public0912-e5-actor10', 'public0912-e5-native10']}))
        self.inventory_path = self.root / 'inventory.json'
        self.inventory_sha = self._materialize_inventory()
        self.request_path = self.root / 'request.json'
        self.request = self._make_request(at)
        self.write_request(self.request)
        self.auth_path = self.root / 'authorization.json'

    def _materialize_inventory(self):
        inv = json.loads(REAL_INVENTORY.read_text())
        refs = []
        for ref in inv['all_start_refs']:
            suffix = ref['path'].split('controller_runs/', 1)[1]
            target = self.runs_root / suffix
            name = target.name
            task_id, attempt = name.split('.')[0], int(name.split('.')[1])
            record = {'attempt': attempt, 'recorded_at': '2026-09-12T15:00:00+00:00',
                      'seed': ctrl.TRIAL_SEED, 'task_id': task_id}
            raw = ctrl.canonical(record)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            refs.append({'path': str(target), 'sha256': ctrl.sha(raw)})
        inv['all_start_refs'] = refs
        raw = ctrl.canonical(inv)
        self.inventory_path.write_bytes(raw)
        return ctrl.sha(raw)

    def _make_request(self, at):
        return {
            'schema': 'e5-successor27-root-review-v2', 'at': at,
            'status': 'SEALED_REQUEST_NOT_AUTHORIZED', 'authorized': False,
            'request_valid_until_epoch': at + ctrl.REQUEST_VALIDITY_SECONDS,
            'requested_allocation_ids': {'actor': ctrl.FRESH_ACTOR_ID,
                                         'native': ctrl.FRESH_NATIVE_ID},
            'requested_run_id': ctrl.FRESH_RUN_ID,
            'proposed_output_paths': {
                'actor_output': str(self.root / 'actor_session11'),
                'native_output': str(self.runs_root / ctrl.FRESH_RUN_ID)},
            'reservations_usd': {'actor': 48, 'native': 32, 'total': 80},
            'subset_denominator': 27, 'full_suite_denominator': 97,
            'full_suite_score': None, 'task_ids': list(ctrl.LIVE27),
            'resume': False, 'automatic_resume_authorized': False,
        }

    def write_request(self, request):
        self.request_path.write_bytes(ctrl.canonical(request))
        self.request = request

    def write_auth(self, request_path=None, authorized=True, issued_at=FIXTURE_AT + 60,
                   request_sha=None, run_id=ctrl.FRESH_RUN_ID, allocation_ids=None):
        auth = {'schema': 'e5-successor27-lead-authorization-v1',
                'authorized': authorized, 'issued_at_epoch': issued_at,
                'issued_by': 'lead',
                'request_sha256': request_sha or ctrl.sha(Path(
                    request_path or self.request_path).read_bytes()),
                'requested_run_id': run_id,
                'requested_allocation_ids': allocation_ids or
                {'actor': ctrl.FRESH_ACTOR_ID, 'native': ctrl.FRESH_NATIVE_ID},
                'resume': False, 'replay': False}
        self.auth_path.write_bytes(ctrl.canonical(auth))
        return self.auth_path

    def config(self, **kwargs):
        kwargs.setdefault('request_path', self.request_path)
        kwargs.setdefault('inventory_path', self.inventory_path)
        kwargs.setdefault('inventory_sha', self.inventory_sha)
        kwargs.setdefault('repo_root', self.root)
        return ctrl.ControllerConfig(**kwargs)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.fixtures = []

    def tearDown(self):
        for fixture in self.fixtures:
            fixture.cleanup()

    def fixture(self, **kwargs):
        fixture = RepoFixture(**kwargs)
        self.fixtures.append(fixture)
        return fixture


class FakeAdapter(ctrl.NativeRuntimeAdapter):
    """Offline adapter: proves the intent barrier and tracks dispatches."""

    def __init__(self, clock=None, fail=None, halt_after_ready=False):
        self.clock = clock
        self.fail = dict(fail or {})  # (task_id, attempt) -> exception_type
        self.halt_after_ready = halt_after_ready
        self.dispatches = []
        self.barrier_violations = []
        self.episodes_dir = None
        self.lock = threading.Lock()

    def readiness(self, deadline):
        if self.halt_after_ready and self.clock is not None:
            # Halt BEFORE any intent/dispatch: the readiness probe itself
            # observes the dispatch deadline as already passed. Raising (not
            # merely advancing the clock) is the only mechanism that reaches
            # the halt path with zero intent files: any task passing the
            # first assert would otherwise write its intent receipt before
            # the second assert can fire.
            raise ctrl.DeadlineExceeded(
                "readiness halt: dispatch deadline already passed")

    def actor_request(self, task_id, attempt, deadline):
        intent = Path(self.episodes_dir) / f'{task_id}.{attempt}.intent.json'
        if not intent.exists():
            self.barrier_violations.append(str(intent))
        with self.lock:
            self.dispatches.append((task_id, attempt))
        return {'status': 'dispatched', 'task_id': task_id, 'attempt': attempt}

    def evaluate_native(self, task_id, attempt, deadline):
        failure = self.fail.get((task_id, attempt))
        if failure:
            raise ctrl.NativeEvaluationError(f'native failure {task_id} #{attempt}',
                                             exception_type=failure)
        return {'duration': 1.25, 'exception_type': None, 'score': None,
                'messages': 4}


def execute_offline(fixture, adapter, clock=None, plan=None):
    """Drive Successor27Controller.execute with deadlines set, gates already owned."""
    clock = clock or ctrl.FakeClock(FIXTURE_AT + 10)
    config = fixture.config(adapter=adapter, clock=clock)
    controller = ctrl.Successor27Controller(config)
    controller.deadlines = ctrl.Deadlines(clock.now())
    if plan is None:
        plan = ctrl.build_plan(ctrl.load_inventory(fixture.inventory_path,
                                                   fixture.inventory_sha),
                               ctrl.load_prepared_manifest())
    native_output = Path(controller.request['proposed_output_paths']['native_output'])
    adapter.episodes_dir = native_output / 'files' / 'journal' / 'episodes'
    summary = controller.execute(plan)
    return controller, summary, native_output


# --- 1. exact live-27 selection ------------------------------------------------
class TestLive27Selection(OfflineTestCase):
    @block_sockets
    def test_default_selection_is_exact_live27(self):
        inventory = ctrl.load_inventory()
        live = ctrl.select_live(inventory)
        self.assertEqual(live, ctrl.LIVE27)
        self.assertEqual(live, [f'task_{n:03d}' for n in range(76, 103)])
        self.assertEqual(len(live), 27)

    @block_sockets
    def test_explicit_request_matches_sealed_never_started(self):
        inventory = ctrl.load_inventory()
        self.assertEqual(ctrl.select_live(inventory, list(ctrl.LIVE27)),
                         inventory['never_started27'])

    @block_sockets
    def test_plan_binds_prepared_manifest_evaluation_ids(self):
        plan = ctrl.build_plan(ctrl.load_inventory(), ctrl.load_prepared_manifest())
        self.assertEqual([t['task_id'] for t in plan], ctrl.LIVE27)
        manifest = ctrl.load_prepared_manifest()
        rows = {r['task_id']: r for r in manifest['tasks']}
        for entry in plan:
            self.assertEqual(entry['evaluation_id'], rows[entry['task_id']]['evaluation_id'])
            self.assertEqual(entry['seed'], ctrl.TRIAL_SEED)
            self.assertEqual(rows[entry['task_id']]['score'], None)

    @block_sockets
    def test_verify_packet_reports_verified_scope(self):
        fixture = self.fixture()
        controller = ctrl.Successor27Controller(fixture.config())
        packet = controller.verify_packet()
        self.assertEqual(packet['status'], 'PACKET_VERIFIED')
        self.assertEqual(packet['inventory_sha256'], fixture.inventory_sha)
        self.assertEqual(packet['subset_denominator'], 27)
        self.assertEqual(packet['full_suite_denominator'], 97)
        self.assertIsNone(packet['full_suite_score'])
        self.assertEqual(packet['start_refs']['start_refs_verified'], 178)


# --- 2. old60/started rejection -----------------------------------------------
class TestStartedExclusionRejection(OfflineTestCase):
    @block_sockets
    def test_any_native10_started_task_rejected(self):
        inventory = ctrl.load_inventory()
        for task_id in ('task_043', 'task_061', 'task_075'):
            with self.assertRaises(ctrl.StartedTaskError):
                ctrl.select_live(inventory, list(ctrl.LIVE27) + [task_id])

    @block_sockets
    def test_any_prior37_started_task_rejected(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.StartedTaskError):
            ctrl.select_live(inventory, list(ctrl.LIVE27) + ['task_001'])
        with self.assertRaises(ctrl.StartedTaskError):
            ctrl.select_live(inventory, list(ctrl.LIVE27) + ['task_041'])

    @block_sockets
    def test_old60_scope_rejected(self):
        inventory = ctrl.load_inventory()
        old60 = inventory['original60_task_ids']
        self.assertEqual(len(old60), 60)
        with self.assertRaises(ctrl.StartedTaskError):
            ctrl.select_live(inventory, old60)

    @block_sockets
    def test_interrupted_and_held_tasks_rejected(self):
        inventory = ctrl.load_inventory()
        for task_id in inventory['native10_exclusion_categories']['interrupted3']:
            with self.assertRaises(ctrl.StartedTaskError):
                ctrl.select_live(inventory, list(ctrl.LIVE27) + [task_id])
        held = {h['task_id'] for h in inventory['host_held_requests']}
        self.assertEqual(held, {'task_073', 'task_075'})
        self.assertTrue(held <= set(inventory['all_excluded70']))


# --- 3. duplicate rejection ----------------------------------------------------
class TestDuplicateRejection(OfflineTestCase):
    @block_sockets
    def test_duplicate_id_rejected(self):
        inventory = ctrl.load_inventory()
        requested = list(ctrl.LIVE27) + ['task_076']
        with self.assertRaises(ctrl.DuplicateTaskError):
            ctrl.select_live(inventory, requested)

    @block_sockets
    def test_repeated_id_rejected_even_in_isolation(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.DuplicateTaskError):
            ctrl.select_live(inventory, ['task_076', 'task_076'])


# --- 4. outside-suite rejection ----------------------------------------------
class TestOutsideSuiteRejection(OfflineTestCase):
    @block_sockets
    def test_unknown_ids_rejected(self):
        inventory = ctrl.load_inventory()
        for bad in ('task_000', 'task_103', 'task_999', 'task_07x', 'other'):
            with self.assertRaises(ctrl.OutsideSuiteError):
                ctrl.select_live(inventory, list(ctrl.LIVE27) + [bad])

    @block_sockets
    def test_non_list_scope_rejected(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.ControllerError):
            ctrl.select_live(inventory, 'task_076')


# --- 5. reorder rejection ------------------------------------------------------
class TestReorderRejection(OfflineTestCase):
    @block_sockets
    def test_reordered_scope_rejected(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.ReorderedSetError):
            ctrl.select_live(inventory, list(reversed(ctrl.LIVE27)))

    @block_sockets
    def test_swap_rejected_even_with_identical_members(self):
        inventory = ctrl.load_inventory()
        swapped = list(ctrl.LIVE27)
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        with self.assertRaises(ctrl.ReorderedSetError):
            ctrl.select_live(inventory, swapped)


# --- 6. partial-set rejection --------------------------------------------------
class TestPartialSetRejection(OfflineTestCase):
    @block_sockets
    def test_26_of_27_rejected(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.PartialSetError):
            ctrl.select_live(inventory, ctrl.LIVE27[:-1])

    @block_sockets
    def test_empty_scope_rejected(self):
        inventory = ctrl.load_inventory()
        with self.assertRaises(ctrl.PartialSetError):
            ctrl.select_live(inventory, [])


# --- 7. changed-binding rejection ---------------------------------------------
class TestChangedBindingRejection(OfflineTestCase):
    @block_sockets
    def test_changed_start_receipt_rejects(self):
        fixture = self.fixture()
        inventory = ctrl.load_inventory(fixture.inventory_path, fixture.inventory_sha)
        victim = Path(inventory['all_start_refs'][10]['path'])
        record = json.loads(victim.read_text())
        record['recorded_at'] = '2026-09-13T00:00:00+00:00'
        victim.write_bytes(ctrl.canonical(record))
        with self.assertRaises(ctrl.ChangedBindingError):
            ctrl.verify_start_refs(inventory, fixture.root)

    @block_sockets
    def test_missing_start_receipt_rejects(self):
        fixture = self.fixture()
        inventory = ctrl.load_inventory(fixture.inventory_path, fixture.inventory_sha)
        Path(inventory['all_start_refs'][5]['path']).unlink()
        with self.assertRaises(ctrl.ChangedBindingError):
            ctrl.verify_start_refs(inventory, fixture.root)

    @block_sockets
    def test_new_native_start_rejects_packet(self):
        fixture = self.fixture()
        inventory = ctrl.load_inventory(fixture.inventory_path, fixture.inventory_sha)
        run_dir = sorted(fixture.runs_root.glob('E5-native-20260912-*'))[0]
        new_start = run_dir / 'files/journal/episodes/task_076.1.intent.json'
        new_start.parent.mkdir(parents=True, exist_ok=True)
        new_start.write_bytes(ctrl.canonical(
            {'attempt': 1, 'recorded_at': '2026-09-19T00:00:00+00:00',
             'seed': ctrl.TRIAL_SEED, 'task_id': 'task_076'}))
        with self.assertRaises(ctrl.ControllerError):
            ctrl.verify_start_refs(inventory, fixture.root)

    @block_sockets
    def test_inventory_sha_drift_rejects(self):
        fixture = self.fixture()
        tampered = json.loads(fixture.inventory_path.read_text())
        tampered['never_started27'] = tampered['never_started27'][:26]
        fixture.inventory_path.write_bytes(ctrl.canonical(tampered))
        with self.assertRaises(ctrl.ControllerError):
            ctrl.load_inventory(fixture.inventory_path, fixture.inventory_sha)


# --- 8. fresh-start guard ------------------------------------------------------
class TestFreshStartGuard(OfflineTestCase):
    @block_sockets
    def test_existing_native_output_rejected(self):
        fixture = self.fixture()
        request = dict(fixture.request)
        Path(request['proposed_output_paths']['native_output']).mkdir(parents=True)
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(request, fixture.root)

    @block_sockets
    def test_existing_actor_output_rejected(self):
        fixture = self.fixture()
        request = dict(fixture.request)
        Path(request['proposed_output_paths']['actor_output']).mkdir(parents=True)
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(request, fixture.root)

    @block_sockets
    def test_native10_run_namespace_reuse_rejected(self):
        fixture = self.fixture()
        request = dict(fixture.request)
        request['requested_run_id'] = 'E5-native-20260912-11'
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(request, fixture.root)

    @block_sockets
    def test_consumed_allocation_ids_rejected(self):
        fixture = self.fixture()
        request = dict(fixture.request)
        request['requested_allocation_ids'] = {'actor': 'public0912-e5-actor11',
                                               'native': ctrl.FRESH_NATIVE_ID}
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(request, fixture.root)

    @block_sockets
    def test_fresh_id_already_claimed_rejected(self):
        fixture = self.fixture()
        (fixture.claims_root / 'stale.claim.json').write_text(json.dumps(
            {'allocation_ids': [ctrl.FRESH_NATIVE_ID]}))
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(fixture.request, fixture.root)

    @block_sockets
    def test_resume_request_rejected(self):
        fixture = self.fixture()
        request = dict(fixture.request)
        request['automatic_resume_authorized'] = True
        with self.assertRaises(ctrl.ControllerError):
            ctrl.check_fresh_start(request, fixture.root)

    @block_sockets
    def test_clean_fresh_state_accepted(self):
        fixture = self.fixture()
        self.assertTrue(ctrl.check_fresh_start(fixture.request, fixture.root))


# --- 9. authorization gating ---------------------------------------------------
class TestAuthorizationGating(OfflineTestCase):
    @block_sockets
    def test_sealed_request_alone_cannot_launch(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.root / 'absent.json', clock)

    @block_sockets
    def test_unauthorized_receipt_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        fixture.write_auth(authorized=False)
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_wrong_request_binding_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        fixture.write_auth(request_sha='0' * 64)
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_id_or_run_mismatch_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        fixture.write_auth(run_id='E5-native-20260919-99')
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_expired_request_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + ctrl.REQUEST_VALIDITY_SECONDS + 1)
        fixture.write_auth()
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_authorization_issued_before_seal_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        fixture.write_auth(issued_at=FIXTURE_AT - 1)
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_expired_authorization_rejected(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + ctrl.REQUEST_VALIDITY_SECONDS + 200)
        fixture.write_auth(issued_at=FIXTURE_AT + 60)
        with self.assertRaises(ctrl.AuthorizationError):
            ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                             fixture.auth_path, clock)

    @block_sockets
    def test_valid_lead_authorization_admits(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        fixture.write_auth()
        gate = ctrl.assess_launch_authorization(fixture.request, fixture.request_path,
                                                fixture.auth_path, clock)
        self.assertTrue(gate['admitted'])
        self.assertEqual(gate['request_sha256'],
                         ctrl.sha(fixture.request_path.read_bytes()))

    @block_sockets
    def test_launch_refuses_without_authorization_or_adapter(self):
        fixture = self.fixture()
        # In-validity fake clock: the sealed request's 24h window is
        # wall-clock anchored, and this test gates on missing auth/adapter,
        # not on expiry (expiry has its own test).
        clock = ctrl.FakeClock(FIXTURE_AT + 100)
        controller = ctrl.Successor27Controller(fixture.config(clock=clock))
        with self.assertRaises(ctrl.AuthorizationError):
            controller.launch()
        fixture.write_auth()
        controller = ctrl.Successor27Controller(
            fixture.config(authorization_path=fixture.auth_path, clock=clock))
        with self.assertRaises(ctrl.RuntimeBindingError):
            controller.launch()
        self.assertFalse(Path(fixture.request['proposed_output_paths']
                              ['native_output']).exists())

    @block_sockets
    def test_real_sealed_request_shape(self):
        self.assertTrue(REAL_REQUEST.exists(), 'fresh sealed request must exist')
        request = ctrl.read_json(REAL_REQUEST)
        self.assertEqual(request['status'], 'SEALED_REQUEST_NOT_AUTHORIZED')
        self.assertFalse(request['authorized'])
        self.assertEqual(request['requested_run_id'], ctrl.FRESH_RUN_ID)
        self.assertEqual(request['requested_allocation_ids'],
                         {'actor': ctrl.FRESH_ACTOR_ID, 'native': ctrl.FRESH_NATIVE_ID})
        self.assertEqual(request['request_valid_until_epoch'] - request['at'],
                         ctrl.REQUEST_VALIDITY_SECONDS)


# --- 10. deadline math ---------------------------------------------------------
class TestDeadlineMath(OfflineTestCase):
    @block_sockets
    def test_pinned_bounds(self):
        deadlines = ctrl.Deadlines(0.0)
        self.assertEqual(deadlines.host_wall, 21600)
        self.assertEqual(deadlines.actor_function, 21540)
        self.assertEqual(deadlines.native_dispatch, 21340)
        self.assertEqual(deadlines.actor_ready, 800)
        self.assertEqual(deadlines.actor_function - deadlines.native_dispatch, 200)

    @block_sockets
    def test_assert_within_raises_past_limit(self):
        deadlines = ctrl.Deadlines(1000.0)
        clock = ctrl.FakeClock(1000.0 + 21339)
        deadlines.assert_within(clock)
        clock.advance(2)
        with self.assertRaises(ctrl.DeadlineExceeded):
            deadlines.assert_within(clock)
        self.assertLess(deadlines.remaining_native(clock), 0)

    @block_sockets
    def test_concurrency_bounded_to_three(self):
        with self.assertRaises(ctrl.ControllerError):
            ctrl.ControllerConfig(concurrency=4)
        self.assertEqual(ctrl.ControllerConfig().concurrency, 3)

    @block_sockets
    def test_execution_halts_before_dispatch_past_deadline(self):
        fixture = self.fixture()
        clock = ctrl.FakeClock(FIXTURE_AT + 10)
        adapter = FakeAdapter(halt_after_ready=True, clock=clock)
        controller, summary, native_output = execute_offline(fixture, adapter, clock=clock)
        self.assertEqual(summary['counts']['recorded'], 0)
        halted = (summary['counts']['halted_deadline'] +
                  summary['counts']['skipped_halted'])
        self.assertEqual(halted, 27)
        self.assertEqual(adapter.dispatches, [])
        self.assertEqual(adapter.barrier_violations, [])
        episodes = native_output / 'files/journal/episodes'
        self.assertEqual(list(episodes.glob('*.intent.json')), [])
        journal = ctrl.RunJournal(native_output / 'files/journal/run_journal.jsonl')
        self.assertTrue(any(r['event'] == 'deadline_halt' for r in journal.records()))


# --- 11. bounded concurrency ---------------------------------------------------
class TestBoundedConcurrency(OfflineTestCase):
    @block_sockets
    def test_exactly_three_never_more(self):
        fixture = self.fixture()

        class ConcurrencyAdapter(FakeAdapter):
            def __init__(self):
                super().__init__()
                self.barrier = threading.Barrier(3, timeout=30)
                self.live = 0
                self.max_live = 0

            def actor_request(self, task_id, attempt, deadline):
                with self.lock:
                    self.live += 1
                    self.max_live = max(self.max_live, self.live)
                try:
                    self.barrier.wait()
                finally:
                    with self.lock:
                        self.live -= 1
                return super().actor_request(task_id, attempt, deadline)

        adapter = ConcurrencyAdapter()
        controller, summary, _ = execute_offline(fixture, adapter)
        self.assertEqual(summary['counts']['recorded'], 27)
        self.assertEqual(adapter.max_live, 3)
        self.assertEqual(len(adapter.dispatches), 27)
        self.assertEqual(adapter.barrier_violations, [])

    @block_sockets
    def test_contract_pins_concurrency(self):
        fixture = self.fixture()
        adapter = FakeAdapter()
        controller, summary, native_output = execute_offline(fixture, adapter)
        contract = ctrl.read_json(native_output / 'files/journal/contract.json')
        self.assertEqual(contract['max_native_concurrency'], 3)
        self.assertEqual(contract['expected_episodes'], 27)
        self.assertFalse(contract['automatic_resume_authorized'])
        self.assertEqual(contract['deadlines']['host_wall'] -
                         contract['deadlines']['launch_epoch'], 21600)


# --- 12. generation receipt schema --------------------------------------------
class TestGenerationReceiptSchema(OfflineTestCase):
    @block_sockets
    def test_receipts_match_prior_episode_schema(self):
        fixture = self.fixture()
        adapter = FakeAdapter()
        controller, summary, native_output = execute_offline(fixture, adapter)
        prior = ctrl.read_json(REAL_PRIOR_RECEIPT)
        episodes = native_output / 'files/journal/episodes'
        receipts = sorted(episodes.glob('*.intent.json'))
        self.assertEqual(len(receipts), 27)
        for path in receipts:
            record = json.loads(path.read_text())
            self.assertEqual(set(record), set(prior))
            self.assertEqual(set(record), {'attempt', 'recorded_at', 'seed', 'task_id'})
            self.assertEqual(record['attempt'], 1)
            self.assertEqual(record['seed'], ctrl.TRIAL_SEED)
            self.assertRegex(record['recorded_at'],
                             r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00$')
            self.assertEqual(path.read_bytes(), ctrl.canonical(record))

    @block_sockets
    def test_intent_receipt_durable_before_dispatch(self):
        fixture = self.fixture()
        adapter = FakeAdapter()
        controller, summary, _ = execute_offline(fixture, adapter)
        self.assertEqual(adapter.barrier_violations, [])
        self.assertEqual(len(adapter.dispatches), 27)

    @block_sockets
    def test_native_and_failure_receipts_use_prior_schema(self):
        fixture = self.fixture()
        adapter = FakeAdapter(fail={('task_090', 1): 'RemoteProtocolError'})
        controller, summary, native_output = execute_offline(fixture, adapter)
        episodes = native_output / 'files/journal/episodes'
        native = ctrl.read_json(episodes / 'task_076.1.native.json')
        self.assertIn('duration', native)
        self.assertIn('exception_type', native)
        failure = ctrl.read_json(episodes / 'task_090.1.known_failure.json')
        self.assertEqual(set(failure), {'exception_type', 'recorded_at'})
        self.assertEqual(failure['exception_type'], 'RemoteProtocolError')


# --- 13. attempt/retry policy --------------------------------------------------
class TestAttemptPolicy(OfflineTestCase):
    @block_sockets
    def test_bounded_retry_then_success(self):
        fixture = self.fixture()
        adapter = FakeAdapter(fail={('task_090', 1): 'RemoteProtocolError',
                                    ('task_090', 2): 'ContextWindowExceededError',
                                    ('task_090', 3): 'RemoteProtocolError'})
        controller, summary, native_output = execute_offline(fixture, adapter)
        episodes = native_output / 'files/journal/episodes'
        self.assertEqual(summary['counts']['recorded'], 27)
        for attempt in (1, 2, 3):
            self.assertTrue((episodes / f'task_090.{attempt}.known_failure.json').exists())
        self.assertTrue((episodes / 'task_090.4.intent.json').exists())
        self.assertTrue((episodes / 'task_090.4.native.json').exists())
        self.assertFalse((episodes / 'task_090.5.intent.json').exists())
        journal = ctrl.RunJournal(native_output / 'files/journal/run_journal.jsonl')
        failures = [r for r in journal.records()
                    if r['event'] == 'known_failure' and r['task_id'] == 'task_090']
        self.assertEqual([f['attempt'] for f in failures], [1, 2, 3])

    @block_sockets
    def test_terminal_failure_after_four_attempts(self):
        fixture = self.fixture()
        adapter = FakeAdapter(fail={(f'task_{n:03d}', a): 'RemoteProtocolError'
                                    for n in range(90, 91) for a in range(1, 5)})
        controller, summary, native_output = execute_offline(fixture, adapter)
        episodes = native_output / 'files/journal/episodes'
        self.assertEqual(summary['counts']['recorded'], 26)
        self.assertEqual(summary['counts']['failed_terminal'], 1)
        for attempt in range(1, 5):
            self.assertTrue((episodes / f'task_090.{attempt}.known_failure.json').exists())
        self.assertEqual(ctrl.MAX_ATTEMPTS_PER_TASK, 4)

    @block_sockets
    def test_attempt_numbers_are_sequential_from_one(self):
        fixture = self.fixture()
        adapter = FakeAdapter(fail={('task_076', 1): 'RemoteProtocolError',
                                    ('task_076', 2): 'RemoteProtocolError'})
        controller, summary, native_output = execute_offline(fixture, adapter)
        episodes = native_output / 'files/journal/episodes'
        self.assertTrue(all((episodes / f'task_076.{a}.intent.json').exists()
                            for a in (1, 2, 3)))


# --- 14. native projection -----------------------------------------------------
class TestNativeProjection(OfflineTestCase):
    @block_sockets
    def test_projection_keeps_scores_null_and_denominators(self):
        fixture = self.fixture()
        adapter = FakeAdapter()
        controller, summary, native_output = execute_offline(fixture, adapter)
        self.assertEqual(summary['counts']['recorded'], 27)
        self.assertEqual(summary['subset_denominator'], 27)
        self.assertEqual(summary['full_suite_denominator'], 97)
        self.assertIsNone(summary['full_suite_score'])
        self.assertIsNone(summary['subset_score'])
        journal = ctrl.RunJournal(native_output / 'files/journal/run_journal.jsonl')
        projected = [r for r in journal.records() if r['event'] == 'native_recorded']
        self.assertEqual(len(projected), 27)
        rows = {r['task_id']: r for r in ctrl.load_prepared_manifest()['tasks']}
        for event in projected:
            self.assertIsNone(event['score'])
            self.assertEqual(event['evaluation_id'],
                             rows[event['task_id']]['evaluation_id'])
            self.assertEqual(event['attempt'], 1)

    @block_sockets
    def test_summary_is_terminal_and_never_pools_scores(self):
        fixture = self.fixture()
        adapter = FakeAdapter(fail={(f'task_{n:03d}', a): 'RemoteProtocolError'
                                    for n in range(76, 79) for a in range(1, 5)})
        controller, summary, native_output = execute_offline(fixture, adapter)
        self.assertEqual(summary['status'], 'TERMINAL_INCOMPLETE_IF_ANY_NOT_RECORDED')
        self.assertEqual(summary['counts']['failed_terminal'], 3)
        self.assertEqual(summary['counts']['recorded'], 24)
        self.assertIsNone(summary['full_suite_score'])

    @block_sockets
    def test_journal_sequence_per_task(self):
        fixture = self.fixture()
        adapter = FakeAdapter()
        controller, summary, native_output = execute_offline(fixture, adapter)
        journal = ctrl.RunJournal(native_output / 'files/journal/run_journal.jsonl')
        records = journal.records()
        self.assertEqual([r['seq'] for r in records], list(range(1, len(records) + 1)))
        task_events = [r['event'] for r in records if r.get('task_id') == 'task_076']
        self.assertEqual(task_events, ['readiness_check', 'readiness_ok',
                                       'intent_recorded', 'actor_dispatched',
                                       'native_recorded'])
        self.assertEqual(records[0]['event'], 'launch')
        self.assertEqual(records[-1]['event'], 'run_complete')


if __name__ == '__main__':
    unittest.main()
