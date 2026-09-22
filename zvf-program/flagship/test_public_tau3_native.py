"""Meaningful offline checks against the pinned tau3 code; never provider calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import public_tau3_native as bridge

class NativeTau3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('tau2') is None:
            raise unittest.SkipTest('Run with the pinned tau3_setup/.venv Python after uv sync --frozen --extra knowledge')
        cls.setup = bridge.DEFAULT_SETUP
        cls.guard = bridge.offline_guard(); cls.guard.__enter__()
        cls.native = bridge.load_native(cls.setup)
        cls.tasks = cls.native.get_tasks('base')
        cls.env, cls.kb = bridge.metadata_environment(cls.native)
        cls.identity = {'model_id': bridge.BASE_MODEL, 'model_revision': bridge.BASE_REVISION,
                        'hf_repo': bridge.ADAPTER_REPO, 'hf_commit': bridge.ADAPTER_REVISION,
                        'served_model_id': 'pavlov-public-portfolio-bf16'}
        cls.prepared = cls.setup / 'prepared'

    @classmethod
    def tearDownClass(cls):
        cls.guard.__exit__(None, None, None)

    def test_exact_native_inventory_and_required_documents(self):
        result = bridge.verify_sources(self.setup)
        self.assertEqual(result['source_files'], 1093)
        self.assertEqual(len(self.tasks), 97)
        self.assertEqual(len(self.kb.documents), 698)
        self.assertEqual(sum(len(t.evaluation_criteria.actions) for t in self.tasks), 955)
        manifest = bridge.read_json(self.prepared / 'manifest.json')
        self.assertEqual([t.id for t in self.tasks], [r['task_id'] for r in manifest['tasks']])
        for row in manifest['tasks']:
            self.assertLessEqual(set(row['required_documents']), set(self.kb.documents))
            self.assertIsNone(row['score'])
        self.assertIsNone(manifest['score'])
        self.assertFalse(manifest['runtime_ready'])

    def test_modified_native_source_is_rejected(self):
        row = bridge.read_json(self.setup / 'source_receipts.json')[0]
        raw = (self.setup / 'source' / row['path']).read_bytes()
        with self.assertRaises(bridge.ContractError):
            bridge.validate_source_blob(row, raw + b'changed')

    def test_exact_checkpoint_and_loopback_actor_boundary(self):
        bad = dict(self.identity, hf_commit='0' * 40)
        with self.assertRaises(bridge.ContractError):
            bridge.make_native_config(bad, 'http://127.0.0.1:8080/v1', 'test')
        with self.assertRaises(bridge.ContractError):
            bridge.make_native_config(self.identity, 'https://api.openai.com/v1', 'test')
        config = bridge.make_native_config(self.identity, 'http://127.0.0.1:8080/v1', 'test', 4096)
        self.assertEqual(config.llm_args_agent['api_base'], 'http://127.0.0.1:8080/v1')
        self.assertEqual(config.llm_args_agent['max_tokens'], 4096)
        self.assertEqual(config.llm_user, 'gpt-4.1-2025-04-14')
        self.assertEqual(config.llm_args_user, {'temperature': 0.0})
        self.assertEqual(config.retrieval_config, 'alltools')
        self.assertEqual(config.max_steps, 200)
        self.assertEqual(config.num_trials, 1)

    def test_uninitialized_metadata_retrieval_cannot_return_fake_results(self):
        self.assertEqual(len(self.env.get_tools()), 17)
        with self.assertRaises(bridge.ContractError):
            self.env.tools.KB_search_dense('credit card')
        with self.assertRaises(bridge.ContractError):
            self.env.tools.KB_search_bm25('credit card')
        with self.assertRaises(bridge.ContractError):
            self.env.tools.shell('ls')

    def test_native_actor_serializes_tools_and_preserves_endpoint(self):
        from tau2.runner.build import build_agent
        from tau2.data_model.message import UserMessage
        from litellm import ModelResponse
        import tau2.utils.llm_utils as llm
        config = bridge.make_native_config(self.identity, 'http://127.0.0.1:8080/v1', 'test', 4096)
        actor = build_agent('llm_agent', self.env, llm=config.llm_agent, llm_args=config.llm_args_agent)
        captured = []
        def fake_completion(**kwargs):
            captured.append(kwargs)
            return ModelResponse(model='gpt-4.1-2025-04-14', choices=[{'index': 0, 'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'fixture-call', 'type': 'function', 'function': {'name': 'KB_search_dense', 'arguments': '{"query":"credit card"}'}}]}, 'finish_reason': 'tool_calls'}],
                usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        with patch.object(llm, 'completion', fake_completion):
            message, _ = actor.generate_next_message(UserMessage(role='user', content='I need a credit card'), actor.get_init_state())
        self.assertEqual(len(captured), 1)
        self.assertEqual(message.tool_calls[0].name, 'KB_search_dense')
        self.assertEqual(captured[0]['api_base'], config.llm_args_agent['api_base'])
        self.assertEqual(captured[0]['model'], config.llm_agent)
        self.assertEqual(captured[0]['messages'][0]['content'], actor.system_prompt)
        self.assertEqual(captured[0]['tools'], [t.openai_schema for t in self.env.get_tools()])
        self.assertEqual(captured[0]['tool_choice'], 'auto')
        self.assertEqual(captured[0]['num_retries'], 3)

    def test_native_user_remains_separate_and_task_tools_filtered(self):
        from tau2.runner.build import build_user
        from tau2.data_model.message import AssistantMessage
        from litellm import ModelResponse
        import tau2.utils.llm_utils as llm
        task = self.tasks[0]
        user = build_user('user_simulator', self.env, task, llm=bridge.USER_MODEL, llm_args={'temperature': 0.0})
        self.assertEqual([t.name for t in user.tools], ['apply_for_credit_card'])
        captured = []
        def fake_completion(**kwargs):
            captured.append(kwargs)
            return ModelResponse(model=bridge.USER_MODEL, choices=[{'index': 0, 'message': {'role': 'assistant', 'content': 'I need a credit card.'}, 'finish_reason': 'stop'}], usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        with patch.object(llm, 'completion', fake_completion):
            user.generate_next_message(AssistantMessage(role='assistant', content='How can I help?'), user.get_init_state())
        self.assertEqual(captured[0]['model'], bridge.USER_MODEL)
        self.assertNotIn('api_base', captured[0])
        self.assertNotIn('api_key', captured[0])
        self.assertIn(str(task.user_scenario), captured[0]['messages'][0]['content'])

    def test_one_native_nl_judge_uses_exact_default_model(self):
        from tau2.data_model.message import AssistantMessage
        from tau2.evaluator import evaluator_nl_assertions as nl
        task = next(t for t in self.tasks if t.id == 'task_102')
        captured = []
        def fake_generate(**kwargs):
            captured.append(kwargs)
            return AssistantMessage(role='assistant', content=json.dumps({'results': [{'expectedOutcome': task.evaluation_criteria.nl_assertions[0], 'reasoning': 'offline test fixture', 'metExpectation': False}]}))
        with patch.object(nl, 'generate', fake_generate):
            result = nl.NLAssertionsEvaluator.calculate_reward(task, [AssistantMessage(role='assistant', content='Fixture only')])
        self.assertEqual(captured[0]['model'], bridge.USER_MODEL)
        self.assertEqual(captured[0]['temperature'], 0.0)
        self.assertEqual(len(result.nl_assertions), 1)
        self.assertFalse(result.nl_assertions[0].met)
        self.assertEqual(result.reward, 0.0)

    def test_plan_keeps_native_scope_unscored_and_requires_runtime_guard(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'plan.json'
            plan = bridge.plan(self.setup, self.prepared, self.identity, 'http://127.0.0.1:8080/v1', 'offline-plan', output, 4096)
            self.assertIsNone(plan['score'])
            self.assertFalse(plan['execution_performed'])
            self.assertNotIn('--num-tasks', plan['argv'])
            self.assertNotIn('--task-ids', plan['argv'])
            self.assertEqual(plan['native_config']['retrieval_config'], 'alltools')
            self.assertEqual(plan['native_config']['llm_user'], bridge.USER_MODEL)
            self.assertTrue(any('shared spending/request guard' in x for x in plan['before_execution']))
            self.assertEqual(bridge.read_json(output), plan)

    def test_partial_manifest_and_overwrite_are_rejected(self):
        manifest = bridge.read_json(self.prepared / 'manifest.json')
        partial = copy.deepcopy(manifest); partial['tasks'].pop()
        original_read = bridge.read_json
        with patch.object(bridge, 'read_json', side_effect=lambda path: partial if Path(path).name == 'manifest.json' else original_read(path)):
            with self.assertRaises(bridge.ContractError):
                bridge.plan(self.setup, self.prepared, self.identity, 'http://127.0.0.1:8080/v1', 'test', '/unused')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'receipt.json'
            bridge.write_once(p, {'score': None})
            bridge.write_once(p, {'score': None})
            with self.assertRaises(bridge.ContractError):
                bridge.write_once(p, {'score': 1})

if __name__ == '__main__':
    unittest.main()
