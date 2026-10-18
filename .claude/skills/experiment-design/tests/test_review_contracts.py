"""Regression tests for the cross-skill review contracts (stdlib only)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILLS = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('design', SKILLS / 'experiment-design/scripts/design_experiments.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


class Contracts(unittest.TestCase):
    def test_default_is_unallocated_and_descriptive(self):
        result = design.generate_design({})
        self.assertIsNone(result['num_seeds'])
        self.assertIsNone(result['estimated_total_runs'])
        self.assertEqual(result['evaluation_protocol']['statistical_test'], 'none')
        self.assertIn('incomplete_seed_plan', design.format_markdown(result))

    def test_seed_plan_validated_and_preserved(self):
        plan = {'type': 'precision', 'target': 'half-width 0.1', 'required_num_seeds': 16,
                'calculation': 'ceil((1.96 * .2 / .1)^2)', 'experimental_unit': 'training seed'}
        result = design.generate_design({'seed_plan': plan})
        self.assertEqual(result['num_seeds'], 16)
        self.assertEqual(result['seed_plan'], plan)
        self.assertEqual(result['evaluation_protocol']['statistical_test'], 'none')
        self.assertIn('half-width 0.1', design.format_markdown(result))
        for bad in ({'num_seeds': 3}, {'seed_plan': plan, 'num_seeds': 15},
                    {'seed_plan': {}}, {'seed_plan': plan, 'num_seeds': True}):
            with self.assertRaises(ValueError):
                design.generate_design(bad)

    def test_completion_independent_of_improvement(self):
        for stage in design.STAGE_TEMPLATES:
            self.assertIn('completed', stage['completion_criteria'].lower())
            self.assertNotIn('improvement', stage['completion_criteria'])

    def test_placeholder_fix_stays_invalid(self):
        script = SKILLS / 'citation-management/scripts/validate_citations.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tex, bib = root / 'paper.tex', root / 'refs.bib'
            tex.write_text(r'\cite{paper}')
            bib.write_text('')
            def run(path, *extra):
                return subprocess.run([sys.executable, str(script), '--tex', str(tex),
                                       '--bib', str(path), *extra], capture_output=True, text=True)
            self.assertEqual(run(bib, '--fix').returncode, 1)
            fixed = root / 'refs_fixed.bib'
            result = run(fixed)
            self.assertEqual(result.returncode, 1)
            self.assertIn('UNRESOLVED BIB PLACEHOLDERS', result.stdout)
            for marker in ('TODO', 'TBD', '20XX', 'placeholder'):
                bib.write_text('@misc{paper, title={' + marker + '}}')
                self.assertEqual(run(bib).returncode, 1)
            bib.write_text('% Auto-generated placeholder entries\n@misc{paper, title={Verified title}, year={2025}}')
            self.assertEqual(run(bib).returncode, 0)

    def test_readability_gate_precedes_completion(self):
        text = (SKILLS / 'hyperresearch-16-readability-audit/SKILL.md').read_text()
        self.assertLess(text.index('## Final review contract v2'), text.index('## Exit criterion'))
        self.assertLess(text.index('its receipt is preserved'), text.index('## Pipeline complete'))

    def test_polish_gate_rejects_failed_envelope(self):
        text = (SKILLS / 'hyperresearch-15-polish/SKILL.md').read_text()
        code = text.split("<<'PY_CHECK'\n", 1)[1].split('\nPY_CHECK', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'research/runs/test'
            root.mkdir(parents=True)
            for name in ('polish-log', 'patch-log'):
                (root / f'{name}.json').write_text(json.dumps({'run_id': 'test'}))
            for critic in ('dialectic', 'depth', 'width', 'instruction'):
                (root / f'critic-findings-{critic}.json').write_text(json.dumps({
                    'run_id': 'test', 'critic': critic, 'status': 'completed', 'findings': [],
                    'query_sha256': 'q', 'corpus_sha256': 'c', 'draft_sha256': 'd'}))
            def run():
                return subprocess.run([sys.executable, '-c', code, 'test', 'full'], cwd=directory,
                                      capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            (root / 'critic-findings-depth.json').write_text(json.dumps({
                'status': 'failed', 'findings': None, 'error': 'unavailable'}))
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('review_status: incomplete', result.stderr)

    def test_gap_and_tension_schemas(self):
        text = (SKILLS / 'hyperresearch-8-corpus-critic/SKILL.md').read_text()
        self.assertIn('"id": "preflight-', text)
        self.assertIn('nonempty unique stable id (agent-', text)
        self.assertLess(text.index('dispatch and await the targeted fetchers'), text.index('1. **Spawn ONE'))
        text = (SKILLS / 'hyperresearch-7-source-tensions/SKILL.md').read_text()
        self.assertIn('"resolution_status": "resolved|unresolved"', text)
        self.assertIn('"evidence_needed":', text)


if __name__ == '__main__':
    unittest.main()
