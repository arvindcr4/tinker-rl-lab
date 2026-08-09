#!/usr/bin/env python3
"""Create an auditable local VerilogEval reference-toolchain receipt.

This runner deliberately does *not* evaluate a language model.  It derives a
temporary ``TopModule`` implementation from the public reference implementation
only to test the local HDL tools against the exact interface/test/reference
bundle used by VerilogEval.  The resulting receipt is therefore labelled
``harness_validation`` with ``is_model_score: false`` and leaves every model
metric null.  A reference implementation passing its own test bench proves the
plumbing works for that problem; it is never a benchmark score.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "e11-verilog-eval-rerun-receipt-v2"
UPSTREAM_COMMIT = "c498220d0a52248f8e3fdffe279075215bde2da6"
TASK_NAME = "Prob001_zero"
DATASETS = ("code-complete-iccad2023", "spec-to-rtl")
PASS_MARKER = "Mismatches: 0"

_ICARUS_VERSION_RE = re.compile(r"Icarus Verilog (?:runtime )?version ([0-9]+(?:\.[0-9]+)*)")
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Upstream README documents Icarus Verilog v12 and states v13 is unsupported.
# v13 fails to bind ``tb_mismatch`` on the pinned test benches.  The default
# compiler is the locally built v12 and the runtime is resolved as its sibling,
# so a v12 compile can never be paired with a v13 ``vvp``.
DEFAULT_IVERILOG = _REPO_ROOT / "outputs/e11_verilog_eval/toolchain/iverilog-12/bin/iverilog"
DEFAULT_VERILATOR = Path("/opt/homebrew/bin/verilator")

# The official Makefile needs GNU Make >= 4.0 (the ``!=`` shell-assignment
# operator and ``$(file ...)``) and GNU ``seq`` (``--format``).  Apple ships GNU
# Make 3.81 and BSD ``seq``, which together expand every ``*_sv_samples``
# variable to empty and leave the per-problem targets with no prerequisites.
DEFAULT_GNU_MAKE = Path("/opt/homebrew/bin/gmake")
DEFAULT_GNU_COREUTILS_BIN = Path("/opt/homebrew/opt/coreutils/libexec/gnubin")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_text(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one native command and retain its complete, JSON-safe evidence."""

    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "command": _command_text(command),
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": _command_text(command),
            "cwd": str(cwd),
            "exit_code": None,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    return path


def derive_interface_from_reference(reference_text: str) -> str:
    """Return the ``TopModule`` header implied by a ``RefModule`` reference.

    ``dataset_code-complete-iccad2023`` ships an explicit ``_ifc.txt`` for every
    problem; ``dataset_spec-to-rtl`` ships none.  Across all 156 pinned
    code-complete problems the shipped ``_ifc.txt`` is byte-for-byte the
    reference module header with ``RefModule`` renamed to ``TopModule``, so the
    same rename reconstructs the interface for spec-to-rtl without editing any
    pinned benchmark source.
    """

    if not reference_text.lstrip().startswith("module RefModule"):
        raise ValueError("reference implementation does not begin with RefModule declaration")
    module_end = reference_text.find(");")
    if module_end < 0:
        raise ValueError("reference implementation has no complete module header")
    return reference_text[: module_end + 2].replace("module RefModule", "module TopModule", 1)


def build_reference_candidate(
    interface_path: Path | None, reference_path: Path, output_path: Path
) -> None:
    """Build a temporary TopModule from a public interface and reference body.

    VerilogEval keeps the interface separately because a normal generated sample
    receives that declaration from ``sv-generate``.  The reference source is a
    ``RefModule``.  Reusing only the body after its module declaration produces
    a temporary ``TopModule`` without editing any pinned benchmark source.  When
    ``interface_path`` is ``None`` (spec-to-rtl, which ships no ``_ifc.txt``),
    the header is derived from the reference itself.
    """

    reference = _require_file(reference_path, "reference implementation").read_text(
        encoding="utf-8"
    )
    module_end = reference.find(");")
    if module_end < 0 or not reference.lstrip().startswith("module RefModule"):
        raise ValueError("reference implementation does not begin with RefModule declaration")

    if interface_path is None:
        interface = derive_interface_from_reference(reference)
    else:
        interface = _require_file(interface_path, "interface declaration").read_text(
            encoding="utf-8"
        )
        if "module TopModule" not in interface or not interface.rstrip().endswith(");"):
            raise ValueError("interface declaration does not define a complete TopModule header")

    output_path.write_text(
        interface.rstrip() + "\n" + reference[module_end + 2 :].lstrip(), encoding="utf-8"
    )


def _tool_version(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    result = _run(command, cwd=cwd, timeout=30)
    return {
        "path": str(command[0]),
        "command": result["command"],
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _icarus_version(tool: dict[str, Any], *, role: str) -> str:
    if tool["exit_code"] != 0:
        raise ValueError(f"selected {role} cannot report its version")
    version_match = _ICARUS_VERSION_RE.search(
        "\n".join((tool.get("stdout", ""), tool.get("stderr", "")))
    )
    if version_match is None:
        raise ValueError(f"selected {role} did not report an Icarus version")
    return version_match.group(1)


def resolve_icarus_runtime(iverilog: Path, explicit_vvp: Path | None = None) -> tuple[Path, str]:
    """Resolve the runtime for a compiler without silently mixing installations."""

    compiler = _require_file(iverilog, "selected iverilog compiler").resolve()
    if not os.access(compiler, os.X_OK):
        raise ValueError(f"selected iverilog compiler is not executable: {compiler}")
    runtime = (explicit_vvp if explicit_vvp is not None else compiler.parent / "vvp").resolve()
    selection = "explicit" if explicit_vvp is not None else "compiler_sibling"
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise ValueError(
            f"{selection} vvp runtime is missing or not executable for selected compiler: {runtime}"
        )
    return runtime, selection


def validate_icarus_pair(
    iverilog: Path, vvp: Path, *, cwd: Path, selection: str
) -> dict[str, Any]:
    """Fail closed unless the compiler and runtime report the same Icarus version."""

    compiler = _tool_version([str(iverilog), "-V"], cwd=cwd)
    runtime = _tool_version([str(vvp), "-V"], cwd=cwd)
    compiler_version = _icarus_version(compiler, role="iverilog compiler")
    runtime_version = _icarus_version(runtime, role="vvp runtime")
    if compiler_version != runtime_version:
        raise ValueError(
            "selected iverilog/vvp version mismatch: "
            f"compiler {compiler_version}, runtime {runtime_version}"
        )
    return {
        "compiler": compiler,
        "runtime": runtime,
        "compiler_version": compiler_version,
        "runtime_version": runtime_version,
        "pair_verified": True,
        "runtime_selection": selection,
    }


def _tool_environment(*prepend: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    prefix = os.pathsep.join(str(item) for item in prepend if item is not None)
    environment["PATH"] = prefix + os.pathsep + environment.get("PATH", "")
    return environment


def _configure(checkout: Path, iverilog: Path, *, task: str = DATASETS[0]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="e11_verilog_eval_configure_") as temp_dir:
        build = Path(temp_dir) / "build"
        build.mkdir()
        command = [
            str(checkout / "configure"),
            f"--with-task={task}",
            "--with-model=manual-rtl-coder",
            "--with-examples=0",
            "--with-samples=1",
            "--with-temperature=0",
            "--with-top-p=0.01",
        ]
        result = _run(command, cwd=build, timeout=120, env=_tool_environment(Path(iverilog).parent))
        result["cwd"] = "<fresh-temporary-build>"
        result["parameters"] = {
            "task": task,
            "model": "manual-rtl-coder",
            "examples": 0,
            "samples": 1,
            "temperature": 0,
            "top_p": 0.01,
        }
        return result


def _write_langchain_shim(root: Path) -> Path:
    """Satisfy the unused ``langchain`` import in upstream's ``sv-iv-analyze``.

    ``scripts/sv-iv-analyze`` does ``from langchain.schema import SystemMessage,
    HumanMessage`` at module scope and then never references either name in its
    367 lines -- a copy-paste leftover from ``sv-generate``, which does need
    langchain.  Without it the official scorer dies with ModuleNotFoundError
    before reading a single log.  A three-line shim satisfies the import exactly;
    installing the real dependency tree would pull ~100 MB onto shared disk to
    service a symbol that is never used.  The pinned checkout is not modified.
    """

    package = root / "langchain"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "schema.py").write_text(
        '"""Import-only shim: upstream sv-iv-analyze imports these and never uses them."""\n\n'
        "class SystemMessage:  # pragma: no cover - never instantiated\n"
        "    pass\n\n\n"
        "class HumanMessage:  # pragma: no cover - never instantiated\n"
        "    pass\n",
        encoding="utf-8",
    )
    return root


def _official_makefile_probe(
    checkout: Path,
    *,
    iverilog: Path,
    gnu_make: Path = DEFAULT_GNU_MAKE,
    gnu_coreutils_bin: Path = DEFAULT_GNU_COREUTILS_BIN,
) -> dict[str, Any]:
    """Drive the official ``<Prob>-sv-iv-test`` target end to end in a fresh build.

    ``sv-generate`` is responsible for putting the interface into a generated
    sample; the Makefile rule itself lists only sample, test, and reference.
    This probe places the temporary reference-only sample where that rule expects
    it, never changes the checkout, and then runs the real target and the real
    ``sv-iv-analyze`` scorer.
    """

    dataset_name = DATASETS[0]
    dataset = checkout / f"dataset_{dataset_name}"
    interface = dataset / f"{TASK_NAME}_ifc.txt"
    reference = dataset / f"{TASK_NAME}_ref.sv"

    make_binary = str(gnu_make) if Path(gnu_make).is_file() else (shutil.which("gmake") or "make")
    environment = _tool_environment(Path(iverilog).parent, Path(gnu_coreutils_bin))

    with tempfile.TemporaryDirectory(prefix="e11_verilog_eval_makefile_") as temp_dir:
        build = Path(temp_dir) / "build"
        build.mkdir()
        configure_command = [
            str(checkout / "configure"),
            f"--with-task={dataset_name}",
            "--with-model=manual-rtl-coder",
            "--with-examples=0",
            "--with-samples=1",
            "--with-temperature=0",
            "--with-top-p=0.01",
        ]
        configured = _run(configure_command, cwd=build, timeout=120, env=environment)

        sample = build / TASK_NAME / f"{TASK_NAME}_sample01.sv"
        sample.parent.mkdir(parents=True, exist_ok=True)
        build_reference_candidate(interface, reference, sample)

        # sv-iv-analyze reads a per-sample sv-generate log for token/cost
        # telemetry and raises FileNotFoundError without it. No model was
        # called, so the recorded counts are a truthful zero rather than
        # invented usage.
        generate_log = build / TASK_NAME / f"{TASK_NAME}_sample01-sv-generate.log"
        generate_log.write_text(
            "# harness validation: reference implementation, no model call was made\n"
            "prompt_tokens = 0\n"
            "resp_tokens = 0\n"
            "cost = 0.000000\n",
            encoding="utf-8",
        )

        make_version = _run([make_binary, "--version"], cwd=build, timeout=30, env=environment)
        bash_version = _run(["/bin/bash", "--version"], cwd=build, timeout=30, env=environment)
        seq_probe = _run(
            ["seq", "--format", "%02g", "1", "1"], cwd=build, timeout=30, env=environment
        )
        expansion = _run(
            [make_binary, f"debug-{TASK_NAME}_sv_iv_test_logs"],
            cwd=build,
            timeout=60,
            env=environment,
        )
        made = _run(
            [make_binary, f"{TASK_NAME}-sv-iv-test", "VERBOSE=1"],
            cwd=build,
            timeout=180,
            env=environment,
        )

        log = build / TASK_NAME / f"{TASK_NAME}_sample01-sv-iv-test.log"
        log_text = log.read_text(encoding="utf-8") if log.is_file() else None

        shim = _write_langchain_shim(Path(temp_dir) / "shims")
        analyze_environment = dict(environment)
        analyze_environment["PYTHONPATH"] = (
            str(shim) + os.pathsep + analyze_environment.get("PYTHONPATH", "")
        )
        analyzed = _run(
            [str(checkout / "scripts" / "sv-iv-analyze"), "--csv=summary.csv", TASK_NAME],
            cwd=build,
            timeout=60,
            env=analyze_environment,
        )
        summary_csv = build / "summary.csv"
        summary_text = summary_csv.read_text(encoding="utf-8") if summary_csv.is_file() else None

        scored = analyzed["exit_code"] == 0 and summary_text is not None
        passed = (
            made["exit_code"] == 0
            and log_text is not None
            and PASS_MARKER in log_text
            and scored
        )
        return {
            "status": "OFFICIAL_MAKE_TARGET_PASSES" if passed else "OFFICIAL_MAKE_TARGET_FAILED",
            "official_scoring_path": "COMPLETE" if scored else "INCOMPLETE",
            "bash_version": bash_version,
            "verbose_is_mandatory_on_this_host": (
                "The non-verbose recipe appends with '&>>' and tests ${PIPESTATUS[0]} with '[[ ]]'. "
                "GNU make runs recipes under /bin/sh, and Apple ships bash 3.2.57 as both /bin/sh "
                "and /bin/bash; '&>>' was added in bash 4.0, so the append line dies with 'syntax "
                "error near unexpected token >'. The recipe is prefixed with '-', so make IGNORES "
                "the error and leaves a zero-byte log. sv-iv-analyze then scores every sample 'R' "
                "and reports pass_rate 0.00. A default 'gmake sv-iv-test' therefore returns a "
                "SILENT 0% on this host. VERBOSE=1 switches the redirect to '2>&1 | tee' and is "
                "mandatory here; SHELL=/bin/bash does not help because that bash is also 3.2."
            ),
            "summary_csv_note": (
                "This summary.csv scores the REFERENCE implementation, not a model. pass_rate 1.0 "
                "here means the scoring path is intact end to end. It is harness validation and "
                "must never be reported as a benchmark result."
            ),
            "explanation": (
                "The official per-problem target runs end to end once GNU Make >= 4.0 and GNU "
                "coreutils seq are on PATH. Apple's system make is GNU Make 3.81, which predates "
                "the '!=' shell-assignment operator, and BSD seq has no '--format'; together they "
                "expand every *_sv_samples variable to empty, which is what produced the earlier "
                "HOST_MAKE_INCOMPATIBLE_NO_SAMPLE_TARGETS finding. The Makefile is not at fault."
            ),
            "make_binary": make_binary,
            "make_version": make_version,
            "gnu_seq_probe": seq_probe,
            "configure": configured,
            "sample_target_expansion": expansion,
            "make": made,
            "sv_iv_analyze": analyzed,
            "sv_iv_analyze_note": (
                "scripts/sv-iv-analyze imports langchain.schema at module scope and never uses "
                "SystemMessage or HumanMessage anywhere in the file. Without langchain installed "
                "the official scorer aborts with ModuleNotFoundError before reading any log. This "
                "probe supplies an import-only shim on PYTHONPATH; the pinned checkout is unchanged."
            ),
            "summary_csv": summary_text,
            "iverilog_log": log_text,
        }


def _verdict(compile_or_build: dict[str, Any], execute: dict[str, Any] | None) -> dict[str, str]:
    if compile_or_build["exit_code"] != 0:
        return {"status": "ERROR", "verdict": "compile_or_build_error"}
    if execute is None:
        return {"status": "ERROR", "verdict": "executable_not_produced"}
    if execute.get("timed_out"):
        return {"status": "ERROR", "verdict": "simulation_timeout"}
    if execute["exit_code"] != 0:
        return {"status": "ERROR", "verdict": "simulation_error"}
    if PASS_MARKER in execute["stdout"]:
        return {"status": "PASS", "verdict": "no_mismatches"}
    return {"status": "ERROR", "verdict": "missing_no_mismatches_verdict"}


def read_problem_ids(checkout: Path, dataset_name: str) -> list[str]:
    """Return the upstream-shipped authoritative problem list for one dataset.

    ``problems.txt`` is committed at the pinned revision, so this is upstream's
    own task list rather than a local filename inventory.
    """

    listing = _require_file(
        checkout / f"dataset_{dataset_name}" / "problems.txt",
        f"{dataset_name} problems.txt",
    )
    return [
        line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_reference_problem(
    checkout: Path,
    task: str = TASK_NAME,
    *,
    dataset_name: str = DATASETS[0],
    iverilog: Path,
    vvp: Path,
    verilator: Path,
    simulation_timeout: int = 45,
) -> dict[str, Any]:
    """Compile one problem's reference implementation against its own test bench."""

    dataset = checkout / f"dataset_{dataset_name}"
    interface_path = dataset / f"{task}_ifc.txt"
    interface = interface_path if interface_path.is_file() else None
    reference = _require_file(dataset / f"{task}_ref.sv", f"{task} reference")
    test = _require_file(dataset / f"{task}_test.sv", f"{task} test bench")

    with tempfile.TemporaryDirectory(prefix="e11_verilog_eval_smoke_") as temp_dir:
        work = Path(temp_dir)
        candidate = work / f"{task}_reference_only_top.sv"
        build_reference_candidate(interface, reference, candidate)

        iverilog_output = work / f"{task}_iverilog.vvp"
        iverilog_compile = _run(
            [
                str(iverilog),
                "-Wall",
                "-Winfloop",
                "-Wno-timescale",
                "-g2012",
                "-s",
                "tb",
                "-o",
                str(iverilog_output),
                str(candidate),
                str(reference),
                str(test),
            ],
            cwd=work,
            timeout=120,
        )
        iverilog_execute = (
            _run([str(vvp), str(iverilog_output)], cwd=work, timeout=simulation_timeout)
            if iverilog_compile["exit_code"] == 0
            else None
        )
        verilator_build = _run(
            [
                str(verilator),
                "--binary",
                "--timing",
                "-Wno-fatal",
                "--top-module",
                "tb",
                "--Mdir",
                str(work / "obj_dir"),
                str(candidate),
                str(reference),
                str(test),
            ],
            cwd=work,
            timeout=300,
        )
        verilator_binary = work / "obj_dir" / "Vtb"
        verilator_execute = (
            _run([str(verilator_binary)], cwd=work, timeout=simulation_timeout)
            if verilator_build["exit_code"] == 0 and verilator_binary.is_file()
            else None
        )

        return {
            "label": "harness_validation",
            "is_model_score": False,
            "task_id": task,
            "dataset": dataset_name,
            "pass_at_1": None,
            "interface_source": (
                "upstream_ifc_txt" if interface is not None else "derived_from_reference_header"
            ),
            "source_bundle": {
                "interface": (
                    {"path": str(interface), "sha256": _sha256_file(interface)}
                    if interface is not None
                    else None
                ),
                "reference": {"path": str(reference), "sha256": _sha256_file(reference)},
                "test": {"path": str(test), "sha256": _sha256_file(test)},
                "temporary_reference_only_top": {"sha256": _sha256_file(candidate)},
            },
            "iverilog": {
                **_verdict(iverilog_compile, iverilog_execute),
                "compile": iverilog_compile,
                "execute": iverilog_execute,
            },
            "verilator": {
                **_verdict(verilator_build, verilator_execute),
                "build": verilator_build,
                "execute": verilator_execute,
            },
        }


# Retained name for the single-problem smoke used by the receipt.
def run_reference_smoke(
    checkout: Path, *, iverilog: Path, vvp: Path, verilator: Path
) -> dict[str, Any]:
    """Run the reference-only Prob001 smoke through both native simulators."""

    return run_reference_problem(
        checkout,
        TASK_NAME,
        dataset_name=DATASETS[0],
        iverilog=iverilog,
        vvp=vvp,
        verilator=verilator,
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep per-problem evidence small on PASS and complete on failure."""

    def summarize(simulator: str, compile_key: str) -> dict[str, Any]:
        block = result[simulator]
        execute = block.get("execute")
        compact: dict[str, Any] = {
            "status": block["status"],
            "verdict": block["verdict"],
            "compile_exit_code": block[compile_key]["exit_code"],
            "execute_exit_code": execute["exit_code"] if execute else None,
        }
        if execute is not None:
            mismatch_lines = [
                line for line in execute["stdout"].splitlines() if line.startswith("Mismatches:")
            ]
            compact["mismatch_line"] = mismatch_lines[-1] if mismatch_lines else None
        if block["status"] != "PASS":
            compact["command"] = block[compile_key]["command"]
            compact["compile_stdout"] = block[compile_key]["stdout"][-4000:]
            compact["compile_stderr"] = block[compile_key]["stderr"][-4000:]
            if execute is not None:
                compact["execute_stdout"] = execute["stdout"][-4000:]
                compact["execute_stderr"] = execute["stderr"][-4000:]
        return compact

    return {
        "task_id": result["task_id"],
        "dataset": result["dataset"],
        "interface_source": result["interface_source"],
        "reference_sha256": result["source_bundle"]["reference"]["sha256"],
        "test_sha256": result["source_bundle"]["test"]["sha256"],
        "iverilog": summarize("iverilog", "compile"),
        "verilator": summarize("verilator", "build"),
    }


def sweep_all_problems(
    checkout: Path,
    *,
    iverilog: Path,
    vvp: Path,
    verilator: Path,
    datasets: Iterable[str] = DATASETS,
    workers: int = 6,
    progress: bool = False,
) -> dict[str, Any]:
    """Run every pinned problem's reference implementation through both simulators.

    This is harness validation, not a model score: the reference answer is the
    input, so a pass proves only that the compile/simulate/verdict path is intact
    for that problem.  A reference that fails its own test bench is a toolchain
    finding, never a benchmark result.
    """

    started = time.time()
    per_dataset: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    for dataset_name in datasets:
        problems = read_problem_ids(checkout, dataset_name)
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_reference_problem,
                    checkout,
                    task,
                    dataset_name=dataset_name,
                    iverilog=iverilog,
                    vvp=vvp,
                    verilator=verilator,
                ): task
                for task in problems
            }
            for done in concurrent.futures.as_completed(futures):
                task = futures[done]
                try:
                    results.append(compact_result(done.result()))
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                    results.append(
                        {
                            "task_id": task,
                            "dataset": dataset_name,
                            "interface_source": None,
                            "iverilog": {"status": "ERROR", "verdict": f"runner_exception: {exc}"},
                            "verilator": {"status": "ERROR", "verdict": f"runner_exception: {exc}"},
                        }
                    )
                if progress and len(results) % 25 == 0:
                    print(f"  {dataset_name}: {len(results)}/{len(problems)}", flush=True)

        results.sort(key=lambda item: item["task_id"])
        per_dataset[dataset_name] = {
            "problem_count": len(problems),
            "problems_source": str(checkout / f"dataset_{dataset_name}" / "problems.txt"),
            "iverilog_pass": sum(1 for r in results if r["iverilog"]["status"] == "PASS"),
            "verilator_pass": sum(1 for r in results if r["verilator"]["status"] == "PASS"),
            "both_pass": sum(
                1
                for r in results
                if r["iverilog"]["status"] == "PASS" and r["verilator"]["status"] == "PASS"
            ),
            "iverilog_failures": sorted(
                r["task_id"] for r in results if r["iverilog"]["status"] != "PASS"
            ),
            "verilator_failures": sorted(
                r["task_id"] for r in results if r["verilator"]["status"] != "PASS"
            ),
            "results": results,
        }
        all_results.extend(results)

    return {
        "label": "harness_validation",
        "is_model_score": False,
        "pass_at_1": None,
        "score": None,
        "explanation": (
            "Each problem's own reference implementation was compiled against its own pinned test "
            "bench and simulated. A pass means the harness path is intact for that problem; it is "
            "not a benchmark result and must never be reported as one."
        ),
        "total_problem_instances": len(all_results),
        "iverilog_pass": sum(1 for r in all_results if r["iverilog"]["status"] == "PASS"),
        "verilator_pass": sum(1 for r in all_results if r["verilator"]["status"] == "PASS"),
        "both_simulators_pass": sum(
            1
            for r in all_results
            if r["iverilog"]["status"] == "PASS" and r["verilator"]["status"] == "PASS"
        ),
        "wall_clock_seconds": round(time.time() - started, 2),
        "workers": workers,
        "datasets": per_dataset,
    }


def summarize_findings(sweep: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Derive named toolchain/dataset findings from a completed sweep.

    A reference implementation that cannot pass its own test bench is a defect in
    the pinned bundle, not a model result.  Each such problem is surfaced with
    its observed compiler output so the finding can be checked rather than
    trusted.
    """

    if sweep is None:
        return []

    findings: list[dict[str, Any]] = []
    for dataset_name, dataset in sweep["datasets"].items():
        failures = sorted(
            set(dataset["iverilog_failures"]) | set(dataset["verilator_failures"])
        )
        by_id = {row["task_id"]: row for row in dataset["results"]}
        for task_id in failures:
            row = by_id.get(task_id, {})
            both = (
                task_id in dataset["iverilog_failures"]
                and task_id in dataset["verilator_failures"]
            )
            findings.append(
                {
                    "finding": "reference_fails_its_own_test_bench",
                    "dataset": dataset_name,
                    "task_id": task_id,
                    "canonical_task_id": f"verilog_eval/{dataset_name}/{task_id}",
                    "confirmed_by_both_simulators": both,
                    "iverilog_verdict": row.get("iverilog", {}).get("verdict"),
                    "verilator_verdict": row.get("verilator", {}).get("verdict"),
                    "iverilog_stderr": row.get("iverilog", {}).get("compile_stderr"),
                    "scoreability": (
                        "unscoreable: the test bench cannot elaborate against the reference the "
                        "same directory ships, so no candidate implementation can pass this task"
                        if both
                        else "degraded: one simulator cannot run this task"
                    ),
                    "impact_on_a_real_run": (
                        "counts as a guaranteed failure for every model on this split, biasing "
                        f"pass@k down by 1/{dataset['problem_count']} "
                        f"({100.0 / dataset['problem_count']:.2f} percentage points)"
                    ),
                }
            )
    return findings


def real_run_requirements(checkout: Path) -> dict[str, Any]:
    """Spell out exactly what crossing the model-artifact boundary requires."""

    return {
        "status": "SPECIFIED_NOT_EXECUTED",
        "artifact_format": {
            "producer": str(checkout / "scripts" / "sv-generate"),
            "input": "<dataset_dir>/<Prob>_prompt.txt",
            "output": "<build>/<Prob>/<Prob>_sample<NN>.sv",
            "content": (
                "One complete SystemVerilog file defining `module TopModule (...); ... endmodule`. "
                "sv-generate extracts the code block from the model response and writes exactly "
                "that file; the Makefile hands it to Icarus unmodified alongside <Prob>_test.sv "
                "and <Prob>_ref.sv."
            ),
            "module_name_is_load_bearing": (
                "The pinned test benches instantiate TopModule by name. A sample naming the module "
                "anything else fails to elaborate."
            ),
            "interface_note": (
                "code-complete-iccad2023 prompts already carry the TopModule header, so the model "
                "completes the body. spec-to-rtl prompts do not, so the model must emit the whole "
                "module including its port list."
            ),
            "bypassing_sv_generate": (
                "sv-generate calls langchain ChatOpenAI/ChatNVIDIA and needs OPENAI_API_KEY or "
                "NVIDIA_API_KEY. Any external generator may write <Prob>/<Prob>_sample<NN>.sv "
                "directly and the official make targets then work unchanged. That file drop is the "
                "supported seam for a Tinker-served or locally served model."
            ),
        },
        "sample_layout": {
            "build_dir": "a fresh directory, never the checkout",
            "configure": (
                f"{checkout / 'configure'} --with-task=<code-complete-iccad2023|spec-to-rtl> "
                "--with-model=<model-id> --with-examples=<0|1|2|3|4> --with-samples=<N> "
                "--with-temperature=<t> --with-top-p=<p>"
            ),
            "samples_per_problem": (
                "configure --with-samples=N; files are <Prob>_sample01.sv .. <Prob>_sampleNN.sv, "
                "zero-padded with seq --format %02g"
            ),
            "per_problem_target": "gmake <Prob>-sv-iv-test VERBOSE=1",
            "full_suite_target": "gmake -j<N> sv-iv-test VERBOSE=1",
            "scoring_target": "gmake sv-iv-analyze VERBOSE=1   # writes summary.csv and summary.txt",
            "verdict_string": f"each per-sample log must contain '{PASS_MARKER} in <N> samples'",
            "also_required_per_sample": (
                "<Prob>/<Prob>_sample<NN>-sv-generate.log must exist beside the sample or "
                "sv-iv-analyze raises FileNotFoundError. It is scanned for 'prompt_tokens = <int>', "
                "'resp_tokens = <int>' and 'cost = <float>'; the Makefile normally produces it by "
                "tee-ing sv-generate. A generator that bypasses sv-generate must write this file too."
            ),
            "host_requirements": [
                "GNU Make >= 4.0 (gmake); Apple's make 3.81 silently yields no sample targets",
                "GNU coreutils seq on PATH (BSD seq has no --format)",
                "Icarus Verilog 12.0 for both iverilog and vvp (v13 cannot bind tb_mismatch)",
                "VERBOSE=1 on every make invocation; without it the non-verbose '&>>' redirect "
                "fails under Apple's bash 3.2, every log is empty, and the suite scores a silent 0%",
                "a langchain import shim (or the real package) for sv-iv-analyze, which imports "
                "langchain.schema at module scope and never uses it",
            ],
        },
        "receipt_requirements": [
            "model identity: immutable model/checkpoint id plus serving revision, not a family name",
            "task IDs: the exact pinned task-ID list and its aggregate hash from the split manifest",
            "verifier output: the raw per-sample sv-iv-test logs plus summary.csv from sv-iv-analyze",
            "W&B run identity: entity/project/run_id of the tracked run",
            "sampling parameters: temperature, top_p, max_tokens, samples per problem, shot count",
            "toolchain identity: the iverilog -V, vvp -V and verilator --version output as executed",
        ],
        "paid_call_cost_estimate": {
            "made_paid_call": False,
            "prompt_count": 312,
            "note": (
                "312 prompts = 156 problems x 2 task framings. pass@1 needs 1 sample per prompt; "
                "the published pass@5 protocol needs 5, i.e. 1560 completions."
            ),
            "assumed_tokens_per_completion": {"prompt_in": 700, "completion_out": 400},
            "estimates_usd": [
                {
                    "configuration": "312 prompts x 1 sample (pass@1), mid-tier hosted model at $3/M in + $15/M out",
                    "input_tokens": 218400,
                    "output_tokens": 124800,
                    "estimated_usd": 2.53,
                },
                {
                    "configuration": "312 prompts x 5 samples (pass@5), mid-tier hosted model at $3/M in + $15/M out",
                    "input_tokens": 1092000,
                    "output_tokens": 624000,
                    "estimated_usd": 12.64,
                },
                {
                    "configuration": "312 prompts x 5 samples, small model at $0.15/M in + $0.60/M out",
                    "input_tokens": 1092000,
                    "output_tokens": 624000,
                    "estimated_usd": 0.54,
                },
            ],
            "verification_cost": (
                "Verification is local and free: the full 312-instance reference sweep through both "
                "simulators completes in single-digit minutes on this host."
            ),
        },
    }


def build_receipt(
    checkout: Path,
    *,
    iverilog: Path,
    verilator: Path,
    vvp: Path | None = None,
    gnu_make: Path = DEFAULT_GNU_MAKE,
    gnu_coreutils_bin: Path = DEFAULT_GNU_COREUTILS_BIN,
    all_problems: bool = True,
    workers: int = 6,
    progress: bool = False,
) -> dict[str, Any]:
    observed_commit = _run(["git", "rev-parse", "HEAD"], cwd=checkout, timeout=30)
    if observed_commit["exit_code"] != 0:
        raise ValueError("cannot determine pinned NVLabs checkout commit")
    revision = observed_commit["stdout"].strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"pinned checkout revision mismatch: expected {UPSTREAM_COMMIT}, found {revision}")

    resolved_vvp, runtime_selection = resolve_icarus_runtime(iverilog, vvp)
    icarus_pair = validate_icarus_pair(
        iverilog, resolved_vvp, cwd=checkout, selection=runtime_selection
    )
    configure = _configure(checkout, iverilog)
    makefile_probe = _official_makefile_probe(
        checkout, iverilog=iverilog, gnu_make=gnu_make, gnu_coreutils_bin=gnu_coreutils_bin
    )
    smoke = run_reference_smoke(
        checkout, iverilog=iverilog, vvp=resolved_vvp, verilator=verilator
    )
    simulator_passed = (
        smoke["iverilog"]["status"] == "PASS" and smoke["verilator"]["status"] == "PASS"
    )

    sweep = (
        sweep_all_problems(
            checkout,
            iverilog=iverilog,
            vvp=resolved_vvp,
            verilator=verilator,
            workers=workers,
            progress=progress,
        )
        if all_problems
        else None
    )

    if not simulator_passed:
        status = "REFERENCE_SMOKE_PARTIAL_MODEL_BLOCKED"
    elif sweep is None:
        status = "REFERENCE_SMOKE_COMPLETE_MODEL_BLOCKED"
    elif sweep["both_simulators_pass"] == sweep["total_problem_instances"]:
        status = "HARNESS_VALIDATED_ALL_PROBLEMS_MODEL_BLOCKED"
    else:
        status = "HARNESS_VALIDATED_PARTIAL_MODEL_BLOCKED"

    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": "verilog_eval",
        "status": status,
        "evidence_class": "harness_validation",
        "is_model_score": False,
        "score": None,
        "pass_at_1": None,
        "upstream": {
            "url": "https://github.com/NVlabs/verilog-eval",
            "checkout": str(checkout),
            "revision": revision,
            "license": "MIT",
        },
        "toolchain": {
            "icarus_pair": icarus_pair,
            "verilator": _tool_version([str(verilator), "--version"], cwd=checkout),
            "gnu_make": _tool_version(
                [str(gnu_make) if Path(gnu_make).is_file() else "make", "--version"], cwd=checkout
            ),
            "compatibility_caveat": (
                "Upstream README documents Icarus Verilog v12 and states that v13 is unsupported. "
                "This receipt runs a locally built v12.0 for BOTH the compiler and the vvp runtime, "
                "verified as a matching pair, which resolves the earlier v13 caveat. Icarus v13 "
                "fails these test benches with 'Unable to bind wire/reg/memory tb_mismatch'. An "
                "earlier revision of this runner hardcoded /opt/homebrew/bin/vvp (v13) and would "
                "have paired a v12 compile with a v13 runtime; the runtime is now resolved as the "
                "sibling of the selected compiler and the pair's reported versions must match."
            ),
        },
        "official_configure": configure,
        "official_makefile_probe": makefile_probe,
        "reference_toolchain_smoke": smoke,
        "all_problems_reference_sweep": sweep,
        "dataset_findings": summarize_findings(sweep),
        "real_run_requirements": real_run_requirements(checkout),
        "model_benchmark": {
            "status": "BLOCKED",
            "score": None,
            "pass_at_1": None,
            "reason": (
                "No local model-generated HDL artifact with immutable model identity, exact task "
                "IDs, native verifier output, and W&B run identity was available; reference answers "
                "were not used as model predictions."
            ),
            "required_evidence": [
                "model-generated HDL artifact",
                "immutable model identity",
                "exact task IDs",
                "native verifier output",
                "W&B run identity",
            ],
        },
        "launch": {"paid_work_launched": False, "weight_changing_run_launched": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=_REPO_ROOT / "outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "outputs/e11_verilog_eval/e11_verilog_eval_rerun_receipt.json",
    )
    parser.add_argument("--iverilog", type=Path, default=DEFAULT_IVERILOG)
    parser.add_argument(
        "--vvp",
        type=Path,
        default=None,
        help=(
            "Optional explicit paired vvp runtime. Defaults to the sibling of --iverilog and must "
            "report the same Icarus version; this is what prevents a v12 compile from being run by "
            "the v13 runtime on PATH."
        ),
    )
    parser.add_argument("--verilator", type=Path, default=DEFAULT_VERILATOR)
    parser.add_argument("--gnu-make", type=Path, default=DEFAULT_GNU_MAKE)
    parser.add_argument("--gnu-coreutils-bin", type=Path, default=DEFAULT_GNU_COREUTILS_BIN)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--single-problem-only",
        action="store_true",
        help="Skip the all-problems reference sweep and run only the Prob001_zero smoke.",
    )
    args = parser.parse_args(argv)

    receipt = build_receipt(
        args.checkout.resolve(),
        iverilog=args.iverilog,
        vvp=args.vvp,
        verilator=args.verilator,
        gnu_make=args.gnu_make,
        gnu_coreutils_bin=args.gnu_coreutils_bin,
        all_problems=not args.single_problem_only,
        workers=args.workers,
        progress=args.progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {key: receipt[key] for key in ("status", "evidence_class", "is_model_score", "score")}
    summary["official_makefile_probe"] = receipt["official_makefile_probe"]["status"]
    sweep = receipt["all_problems_reference_sweep"]
    if sweep is not None:
        summary["sweep"] = {
            "total_problem_instances": sweep["total_problem_instances"],
            "iverilog_pass": sweep["iverilog_pass"],
            "verilator_pass": sweep["verilator_pass"],
            "both_simulators_pass": sweep["both_simulators_pass"],
            "wall_clock_seconds": sweep["wall_clock_seconds"],
        }
        summary["dataset_findings"] = [
            f"{item['finding']}: {item['canonical_task_id']}"
            for item in receipt["dataset_findings"]
        ]
    print(json.dumps(summary, indent=2))
    return 0 if receipt["reference_toolchain_smoke"]["label"] == "harness_validation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
