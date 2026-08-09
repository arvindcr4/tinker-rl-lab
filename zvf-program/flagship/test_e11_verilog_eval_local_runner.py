from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from . import e11_verilog_eval_local_runner as runner  # noqa: F401
else:
    try:
        from . import e11_verilog_eval_local_runner as runner
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        runner = import_module("e11_verilog_eval_local_runner")


_REFERENCE = "module RefModule (\n  output zero\n);\n\n  assign zero = 1'b0;\n\nendmodule\n"
_PINNED_CHECKOUT = (
    Path(__file__).resolve().parents[2]
    / "outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d"
)


class E11VerilogEvalLocalRunnerTests(unittest.TestCase):
    def test_reference_candidate_uses_interface_without_mutating_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interface = root / "Prob001_zero_ifc.txt"
            reference = root / "Prob001_zero_ref.sv"
            candidate = root / "reference_only_top.sv"
            interface.write_text("module TopModule (output zero);\n", encoding="utf-8")
            reference.write_text(
                "module RefModule (output zero);\nassign zero = 1'b0;\nendmodule\n",
                encoding="utf-8",
            )
            original_reference = reference.read_text(encoding="utf-8")

            runner.build_reference_candidate(interface, reference, candidate)

            self.assertEqual(reference.read_text(encoding="utf-8"), original_reference)
            self.assertEqual(
                candidate.read_text(encoding="utf-8"),
                "module TopModule (output zero);\nassign zero = 1'b0;\nendmodule\n",
            )

    def test_receipt_contract_labels_reference_smoke_as_not_a_model_score(self) -> None:
        self.assertEqual(runner.SCHEMA_VERSION, "e11-verilog-eval-rerun-receipt-v2")
        self.assertEqual(runner.UPSTREAM_COMMIT, "c498220d0a52248f8e3fdffe279075215bde2da6")

    @unittest.skipUnless(
        (_PINNED_CHECKOUT.parent / "e11_verilog_eval_rerun_receipt.json").is_file(),
        "live E11 receipt not present",
    )
    def test_live_receipt_schema_matches_runner_contract(self) -> None:
        receipt_path = _PINNED_CHECKOUT.parent / "e11_verilog_eval_rerun_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema_version"], runner.SCHEMA_VERSION)

    def test_default_compiler_points_at_the_locally_built_v12(self) -> None:
        # Upstream documents Icarus v12 and says v13 is unsupported; the v13
        # install also present on this host must never be the default.
        self.assertIn("iverilog-12", str(runner.DEFAULT_IVERILOG))

    def test_selected_compiler_uses_its_sibling_vvp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            compiler = bin_dir / "iverilog"
            sibling_vvp = bin_dir / "vvp"
            compiler.touch()
            sibling_vvp.touch()
            compiler.chmod(0o755)
            sibling_vvp.chmod(0o755)

            resolved_vvp, selection = runner.resolve_icarus_runtime(compiler)

            self.assertEqual(resolved_vvp, sibling_vvp.resolve())
            self.assertEqual(selection, "compiler_sibling")

    def test_mismatched_or_missing_vvp_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler = root / "iverilog"
            runtime = root / "vvp"
            compiler.touch()
            runtime.touch()
            compiler.chmod(0o755)
            runtime.chmod(0o755)

            with self.assertRaisesRegex(ValueError, "missing or not executable"):
                runner.resolve_icarus_runtime(compiler, root / "not-vvp")

            with patch.object(
                runner,
                "_tool_version",
                side_effect=[
                    {"exit_code": 0, "stdout": "Icarus Verilog version 12.0"},
                    {"exit_code": 0, "stdout": "Icarus Verilog runtime version 13.0"},
                ],
            ):
                with self.assertRaisesRegex(ValueError, "version mismatch"):
                    runner.validate_icarus_pair(
                        compiler, runtime, cwd=root, selection="explicit"
                    )

    def test_matching_versions_are_recorded_as_a_verified_pair(self) -> None:
        with patch.object(
            runner,
            "_tool_version",
            side_effect=[
                {"exit_code": 0, "stdout": "Icarus Verilog version 12.0 (stable) (v12_0)"},
                {"exit_code": 0, "stdout": "Icarus Verilog runtime version 12.0 (stable) (v12_0)"},
            ],
        ):
            pair = runner.validate_icarus_pair(
                Path("/x/iverilog"), Path("/x/vvp"), cwd=Path("."), selection="compiler_sibling"
            )
        self.assertTrue(pair["pair_verified"])
        self.assertEqual(pair["compiler_version"], "12.0")
        self.assertEqual(pair["runtime_version"], "12.0")


class InterfaceDerivationTests(unittest.TestCase):
    def test_derives_top_module_header_from_reference(self) -> None:
        self.assertEqual(
            runner.derive_interface_from_reference(_REFERENCE),
            "module TopModule (\n  output zero\n);",
        )

    def test_rejects_a_source_that_is_not_a_reference_module(self) -> None:
        with self.assertRaises(ValueError):
            runner.derive_interface_from_reference("module TopModule (\n  output zero\n);\n")

    def test_candidate_without_an_interface_file_derives_its_own_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "Prob001_zero_ref.sv"
            candidate = root / "top.sv"
            reference.write_text(_REFERENCE, encoding="utf-8")

            runner.build_reference_candidate(None, reference, candidate)

            text = candidate.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("module TopModule ("))
            self.assertIn("assign zero = 1'b0;", text)
            self.assertNotIn("RefModule", text)

    @unittest.skipUnless(_PINNED_CHECKOUT.is_dir(), "pinned NVlabs checkout not present")
    def test_shipped_interfaces_match_the_derivation_rule(self) -> None:
        # The spec-to-rtl sweep has no _ifc.txt to read and relies on this
        # equivalence, so it is asserted against every problem that does ship one.
        dataset = _PINNED_CHECKOUT / "dataset_code-complete-iccad2023"
        problems = runner.read_problem_ids(_PINNED_CHECKOUT, "code-complete-iccad2023")
        self.assertEqual(len(problems), 156)
        for problem in problems:
            reference = (dataset / f"{problem}_ref.sv").read_text(encoding="utf-8")
            shipped = (dataset / f"{problem}_ifc.txt").read_text(encoding="utf-8")
            self.assertEqual(
                runner.derive_interface_from_reference(reference).strip(),
                shipped.strip(),
                msg=problem,
            )


class ProblemListTests(unittest.TestCase):
    def test_reads_the_upstream_problem_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset_spec-to-rtl"
            dataset.mkdir()
            (dataset / "problems.txt").write_text("ProbA\n\nProbB\n", encoding="utf-8")
            self.assertEqual(runner.read_problem_ids(root, "spec-to-rtl"), ["ProbA", "ProbB"])

    def test_missing_problem_list_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                runner.read_problem_ids(Path(temp_dir), "spec-to-rtl")

    @unittest.skipUnless(_PINNED_CHECKOUT.is_dir(), "pinned NVlabs checkout not present")
    def test_pinned_checkout_lists_156_problems_per_dataset(self) -> None:
        for dataset_name in runner.DATASETS:
            self.assertEqual(
                len(runner.read_problem_ids(_PINNED_CHECKOUT, dataset_name)), 156, msg=dataset_name
            )


class VerdictTests(unittest.TestCase):
    @staticmethod
    def _ok(stdout: str = "") -> dict[str, object]:
        return {"exit_code": 0, "stdout": stdout, "stderr": "", "timed_out": False, "command": "c"}

    def test_pass_requires_the_no_mismatch_verdict_string(self) -> None:
        self.assertEqual(
            runner._verdict(self._ok(), self._ok("Mismatches: 0 in 20 samples"))["status"], "PASS"
        )
        self.assertEqual(
            runner._verdict(self._ok(), self._ok("Mismatches: 3 in 20 samples"))["verdict"],
            "missing_no_mismatches_verdict",
        )

    def test_failure_modes_are_distinguished(self) -> None:
        failed = {"exit_code": 1, "stdout": "", "stderr": "boom", "timed_out": False, "command": "c"}
        self.assertEqual(runner._verdict(failed, None)["verdict"], "compile_or_build_error")
        self.assertEqual(runner._verdict(self._ok(), None)["verdict"], "executable_not_produced")
        timed_out = {"exit_code": None, "stdout": "", "stderr": "", "timed_out": True, "command": "c"}
        self.assertEqual(runner._verdict(self._ok(), timed_out)["verdict"], "simulation_timeout")


class CompactResultTests(unittest.TestCase):
    @staticmethod
    def _result(status: str) -> dict[str, object]:
        execute = {
            "exit_code": 0,
            "stdout": "Mismatches: 0 in 20 samples\n",
            "stderr": "",
            "command": "vvp out.vvp",
            "timed_out": False,
        }
        compile_step = {
            "exit_code": 0 if status == "PASS" else 1,
            "stdout": "",
            "stderr": "unable to bind tb_mismatch",
            "command": "iverilog ...",
            "timed_out": False,
        }
        return {
            "task_id": "Prob001_zero",
            "dataset": "code-complete-iccad2023",
            "interface_source": "upstream_ifc_txt",
            "source_bundle": {"reference": {"sha256": "a" * 64}, "test": {"sha256": "b" * 64}},
            "iverilog": {"status": status, "verdict": "v", "compile": compile_step, "execute": execute},
            "verilator": {"status": status, "verdict": "v", "build": compile_step, "execute": execute},
        }

    def test_passing_rows_stay_terse(self) -> None:
        compact = runner.compact_result(self._result("PASS"))
        self.assertEqual(compact["iverilog"]["mismatch_line"], "Mismatches: 0 in 20 samples")
        self.assertNotIn("compile_stderr", compact["iverilog"])

    def test_failing_rows_keep_the_diagnostic_output(self) -> None:
        compact = runner.compact_result(self._result("ERROR"))
        self.assertIn("unable to bind tb_mismatch", compact["iverilog"]["compile_stderr"])
        self.assertIn("command", compact["iverilog"])


class LangchainShimTests(unittest.TestCase):
    def test_shim_satisfies_the_exact_import_sv_iv_analyze_makes(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as temp_dir:
            root = runner._write_langchain_shim(Path(temp_dir) / "shims")
            completed = subprocess.run(
                [sys.executable, "-c", "from langchain.schema import SystemMessage, HumanMessage"],
                cwd=temp_dir,
                env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    @unittest.skipUnless(_PINNED_CHECKOUT.is_dir(), "pinned NVlabs checkout not present")
    def test_upstream_scorer_imports_langchain_but_never_uses_it(self) -> None:
        # Justifies shimming instead of installing the real dependency tree.
        source = (_PINNED_CHECKOUT / "scripts" / "sv-iv-analyze").read_text(encoding="utf-8")
        self.assertIn("from langchain.schema", source)
        for symbol in ("SystemMessage", "HumanMessage"):
            self.assertEqual(source.count(symbol), 1, msg=f"{symbol} is used, not just imported")


class FindingsTests(unittest.TestCase):
    @staticmethod
    def _sweep(iverilog_failures: list[str], verilator_failures: list[str]) -> dict[str, object]:
        return {
            "datasets": {
                "spec-to-rtl": {
                    "problem_count": 156,
                    "iverilog_failures": iverilog_failures,
                    "verilator_failures": verilator_failures,
                    "results": [
                        {
                            "task_id": "Prob099_m2014_q6c",
                            "iverilog": {
                                "verdict": "compile_or_build_error",
                                "compile_stderr": "port `Y2' is not a port of good1.",
                            },
                            "verilator": {"verdict": "compile_or_build_error"},
                        }
                    ],
                }
            }
        }

    def test_no_sweep_yields_no_findings(self) -> None:
        self.assertEqual(runner.summarize_findings(None), [])

    def test_a_reference_failing_both_simulators_is_reported_as_unscoreable(self) -> None:
        findings = runner.summarize_findings(
            self._sweep(["Prob099_m2014_q6c"], ["Prob099_m2014_q6c"])
        )
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["finding"], "reference_fails_its_own_test_bench")
        self.assertEqual(
            finding["canonical_task_id"], "verilog_eval/spec-to-rtl/Prob099_m2014_q6c"
        )
        self.assertTrue(finding["confirmed_by_both_simulators"])
        self.assertIn("unscoreable", finding["scoreability"])
        self.assertIn("1/156", finding["impact_on_a_real_run"])

    def test_a_single_simulator_failure_is_reported_as_degraded(self) -> None:
        findings = runner.summarize_findings(self._sweep(["Prob099_m2014_q6c"], []))
        self.assertFalse(findings[0]["confirmed_by_both_simulators"])
        self.assertIn("degraded", findings[0]["scoreability"])

    def test_a_clean_sweep_yields_no_findings(self) -> None:
        self.assertEqual(runner.summarize_findings(self._sweep([], [])), [])


class RealRunRequirementTests(unittest.TestCase):
    def test_requirements_are_specified_without_making_a_paid_call(self) -> None:
        requirements = runner.real_run_requirements(Path("/checkout"))
        self.assertEqual(requirements["status"], "SPECIFIED_NOT_EXECUTED")
        self.assertFalse(requirements["paid_call_cost_estimate"]["made_paid_call"])
        self.assertEqual(requirements["paid_call_cost_estimate"]["prompt_count"], 312)
        self.assertTrue(requirements["paid_call_cost_estimate"]["estimates_usd"])
        self.assertIn("TopModule", requirements["artifact_format"]["content"])
        self.assertIn("sv-iv-analyze", requirements["sample_layout"]["scoring_target"])

    def test_receipt_requirements_name_every_mandatory_field(self) -> None:
        text = " ".join(runner.real_run_requirements(Path("/checkout"))["receipt_requirements"])
        for needle in ("model identity", "task IDs", "verifier output", "W&B run identity"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
