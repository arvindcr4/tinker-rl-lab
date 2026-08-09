from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e11_model_run as driver  # noqa: E402


_MODULE = "module TopModule (\n  output zero\n);\n  assign zero = 1'b0;\nendmodule"


class ExtractionTests(unittest.TestCase):
    def test_plain_fenced_block(self) -> None:
        self.assertEqual(extract(f"Here you go:\n```verilog\n{_MODULE}\n```"), _MODULE)

    def test_unfenced_module(self) -> None:
        self.assertEqual(extract(f"Sure.\n\n{_MODULE}\n"), _MODULE)

    def test_thinking_trace_is_stripped_even_without_opening_tag(self) -> None:
        # The chat template pre-opens <think>, so responses usually contain only
        # the closing tag. A draft inside the trace must not win.
        draft = "module TopModule (output zero);\n  assign zero = 1'b1;\nendmodule"
        response = f"I'll try this.\n```verilog\n{draft}\n```\nNo wait.\n</think>\n\n```verilog\n{_MODULE}\n```"
        self.assertEqual(extract(response), _MODULE)
        self.assertNotIn("1'b1", extract(response) or "")

    def test_matched_think_tags_are_stripped(self) -> None:
        response = f"<think>\nreasoning about zero\n</think>\n\n```systemverilog\n{_MODULE}\n```"
        self.assertEqual(extract(response), _MODULE)

    def test_last_fenced_block_wins(self) -> None:
        first = "module TopModule (output zero);\n  assign zero = 1'b1;\nendmodule"
        response = f"```verilog\n{first}\n```\nOn reflection:\n```verilog\n{_MODULE}\n```"
        self.assertEqual(extract(response), _MODULE)

    def test_wrong_module_name_is_not_extracted(self) -> None:
        # Test benches instantiate TopModule by name; anything else is unusable.
        self.assertIsNone(extract("```verilog\nmodule Foo (output zero);\nendmodule\n```"))

    def test_prose_only_response_yields_none(self) -> None:
        self.assertIsNone(extract("I'm not sure how to implement this."))

    def test_truncated_module_yields_none(self) -> None:
        # A response that hit max_tokens mid-module has no endmodule.
        self.assertIsNone(extract("```verilog\nmodule TopModule (output zero);\n  assign ze"))


def extract(text: str) -> str | None:
    return driver.extract_module(text)


class SampleLayoutTests(unittest.TestCase):
    def test_writes_sample_and_generate_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build = Path(temp_dir)
            driver.write_sample(
                build, "Prob001_zero", _MODULE, prompt_tokens=210, resp_tokens=180, cost_usd=0.00036
            )
            sample = build / "Prob001_zero" / "Prob001_zero_sample01.sv"
            log = build / "Prob001_zero" / "Prob001_zero_sample01-sv-generate.log"

            self.assertTrue(sample.is_file())
            self.assertIn("module TopModule", sample.read_text())
            text = log.read_text()
            for needle in ("prompt_tokens = 210", "resp_tokens = 180", "cost = 0.000360"):
                self.assertIn(needle, text)
            self.assertIn(driver.MODEL_REVISION, text)

    def test_failed_extraction_still_writes_both_files(self) -> None:
        # sv-iv-analyze opens the generate log unconditionally; a missing sample
        # would break the make target rather than scoring as a miss.
        with tempfile.TemporaryDirectory() as temp_dir:
            build = Path(temp_dir)
            driver.write_sample(
                build, "Prob002_x", None, prompt_tokens=10, resp_tokens=0, cost_usd=0.0
            )
            self.assertEqual((build / "Prob002_x" / "Prob002_x_sample01.sv").read_text(), "")
            self.assertTrue((build / "Prob002_x" / "Prob002_x_sample01-sv-generate.log").is_file())


class ScoringTests(unittest.TestCase):
    def test_reports_both_denominators(self) -> None:
        results = {f"verilog_eval/spec-to-rtl/P{i}": True for i in range(9)}
        results["verilog_eval/spec-to-rtl/Prob099_m2014_q6c"] = False
        scored = driver.score_pass_at_1(results)

        self.assertEqual(scored["raw"]["denominator"], 10)
        self.assertEqual(scored["raw"]["passes"], 9)
        self.assertEqual(scored["raw"]["pass_at_1"], 0.9)

        self.assertEqual(scored["corrected"]["denominator"], 9)
        self.assertEqual(scored["corrected"]["passes"], 9)
        self.assertEqual(scored["corrected"]["pass_at_1"], 1.0)
        self.assertIn("verilog_eval/spec-to-rtl/Prob099_m2014_q6c", scored["corrected"]["excluded"])

    def test_absent_defect_task_leaves_denominators_equal(self) -> None:
        results = {"verilog_eval/code-complete-iccad2023/A": True}
        scored = driver.score_pass_at_1(results)
        self.assertEqual(scored["raw"]["denominator"], scored["corrected"]["denominator"])
        self.assertEqual(scored["unscoreable_tasks"], [])

    def test_parses_official_summary_csv(self) -> None:
        parsed = driver.parse_summary_csv("Prob001_zero,1,1,1.0,.\nProb002_x,0,1,0.0,R\n")
        self.assertEqual(parsed, {"Prob001_zero": True, "Prob002_x": False})


class BudgetTests(unittest.TestCase):
    def test_projection_matches_pinned_prices(self) -> None:
        projection = driver.project_cost(208264, 312, 4096)
        self.assertTrue(projection["within_gate"])
        self.assertLess(projection["projected_usd"], 2.0)

    def test_large_max_tokens_trips_the_gate(self) -> None:
        self.assertFalse(driver.project_cost(208264, 312, 16384)["within_gate"])


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get(driver.AUTHORIZATION_ENV)
        os.environ.pop(driver.AUTHORIZATION_ENV, None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(driver.AUTHORIZATION_ENV, None)
        else:
            os.environ[driver.AUTHORIZATION_ENV] = self._saved

    def test_sampling_refuses_without_authorization(self) -> None:
        def generate(_: str):  # pragma: no cover - must never be reached
            raise AssertionError("generate() was called without authorization")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(driver.PaidRunNotAuthorized):
                driver.sample_all([("Prob001_zero", "prompt")], generate, Path(temp_dir))

    def test_wrong_token_is_rejected(self) -> None:
        os.environ[driver.AUTHORIZATION_ENV] = "true"
        with self.assertRaises(driver.PaidRunNotAuthorized):
            driver.require_authorization()

    def test_sampling_takes_exactly_one_sample_per_prompt(self) -> None:
        os.environ[driver.AUTHORIZATION_ENV] = driver.AUTHORIZATION_TOKEN
        calls: list[str] = []

        def generate(prompt: str):
            calls.append(prompt)
            return (f"```verilog\n{_MODULE}\n```", 100, 50)

        with tempfile.TemporaryDirectory() as temp_dir:
            report = driver.sample_all(
                [("Prob001_zero", "p1"), ("Prob002_x", "p2")], generate, Path(temp_dir)
            )

        self.assertEqual(len(calls), 2, msg="pass@1 must call the model once per prompt")
        self.assertEqual(report["samples_per_problem"], 1)
        self.assertEqual(report["problems"], 2)
        self.assertEqual(report["extraction_failures"], 0)
        self.assertEqual(report["actual_prompt_tokens"], 200)
        self.assertEqual(report["actual_resp_tokens"], 100)

    def test_extraction_failure_is_recorded_not_retried(self) -> None:
        os.environ[driver.AUTHORIZATION_ENV] = driver.AUTHORIZATION_TOKEN
        calls = {"n": 0}

        def generate(_: str):
            calls["n"] += 1
            return ("I cannot help with that.", 10, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            report = driver.sample_all([("Prob001_zero", "p")], generate, Path(temp_dir))

        self.assertEqual(calls["n"], 1, msg="a weak answer must not be re-rolled")
        self.assertEqual(report["extraction_failures"], 1)


if __name__ == "__main__":
    unittest.main()
