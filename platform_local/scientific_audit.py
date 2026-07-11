#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audit_utils import AuditIssue, run_audit


def get_issues(ctx):
    issues: list[AuditIssue] = []
    paper_tex = ctx.FINAL_DIR / "grpo_agentic_llm_paper.tex"
    paper_tex_anon = ctx.FINAL_DIR / "grpo_agentic_llm_paper_anonymous.tex"
    paper_md = ctx.FINAL_DIR / "grpo_agentic_llm_paper.md"
    report_md = ctx.FINAL_DIR / "capstone_final_report.md"
    submission_checklist = ctx.FINAL_DIR / "SUBMISSION_CHECKLIST.md"
    supplementary = ctx.FINAL_DIR / "supplementary_appendix.tex"
    eval_py = ctx.FINAL_DIR / "evaluate_gsm8k_test.py"
    result_jsons = tuple(sorted(ctx.FINAL_DIR.glob("gsm8k*.json")))

    def add(path: Path, code: str, message: str):
        issues.append(
            AuditIssue(
                code=code,
                message=message,
                location=str(path.relative_to(ctx.ROOT)),
            )
        )

    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def check_paper():
        tex = read(paper_tex)
        anon = read(paper_tex_anon)
        paper_markdown = read(paper_md)
        report = read(report_md)
        checklist = read(submission_checklist)
        supp = read(supplementary)

        abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
        if abstract_match:
            abstract = abstract_match.group(1).lower()
            if (
                "held-out" not in abstract
                and "training-set" not in abstract
                and "evaluation scope" not in abstract
            ):
                add(
                    paper_tex,
                    "ctx.paper.abstract.scope",
                    "LaTeX abstract reports GSM8K gains without explicitly saying they are training-set reward metrics, risking overclaim.",
                )

        if "held-out" not in tex.lower() and "training-set reward" not in tex.lower():
            add(
                paper_tex,
                "ctx.paper.global.scope",
                "LaTeX ctx.paper lacks an explicit held-out-vs-training-set evaluation scope warning.",
            )

        if "publishable confidence intervals" in report.lower():
            add(
                report_md,
                "report.overclaim.publishable",
                "Capstone report claims 'publishable confidence intervals' despite n=3 seeds and no held-out evaluation.",
            )

        anon_abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", anon, re.S)
        if anon_abstract_match:
            anon_abstract = anon_abstract_match.group(1).lower()
            if (
                "held-out" not in anon_abstract
                and "training-set" not in anon_abstract
                and "evaluation scope" not in anon_abstract
            ):
                add(
                    paper_tex_anon,
                    "paper_anon.abstract.scope",
                    "Anonymous LaTeX abstract still reports GSM8K gains without an explicit training-set-vs-held-out caveat.",
                )

        if (
            "held-out" not in paper_markdown.lower()
            and "training-set reward" not in paper_markdown.lower()
        ):
            add(
                paper_md,
                "paper_md.global.scope",
                "Markdown ctx.paper lacks an explicit held-out-vs-training-set scope warning.",
            )

        if re.search(
            r"\|\s*GSM8K\s*\|\s*30\.0% \± 2\.5% \(3 seeds\)\s*\|",
            checklist,
        ):
            add(
                submission_checklist,
                "ctx.checklist.gsm8k.label",
                "Submission ctx.checklist labels GSM8K as a generic result instead of explicitly marking it as training-set reward.",
            )

        for path, text in [
            (paper_tex, tex),
            (paper_tex_anon, anon),
            (paper_md, paper_markdown),
        ]:
            if "confirms grpo training stability" in text.lower():
                add(
                    path,
                    "ctx.paper.stability.overclaim",
                    "Paper claims the 3-seed GSM8K result 'confirms' training stability; this should be softened to a more accurate characterization.",
                )

        for path, text in [
            (report_md, report),
            (paper_md, paper_markdown),
            (supplementary, supp),
        ]:
            low = text.lower()
            if "99% gsm8k accuracy" in low or "99\\% gsm8k accuracy" in low:
                add(
                    path,
                    "gsm8k.peak_accuracy.overclaim",
                    "A 99% GSM8K statement appears without making clear that it refers to a peak training-step metric rather than held-out benchmark accuracy.",
                )

    def _is_name(node, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def _const_str(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def check_eval_script():
        tree = ast.parse(read(eval_py))
        source = read(eval_py)

        parser_has_seed = "--seed" in source
        parser_has_split = "--split" in source
        split_choices_locked = 'choices=["test"]' in source or "choices=['test']" in source
        checkpoint_arg_used = False
        fallback_last_number = "Fallback: extract last number" in source
        do_sample_true = False
        default_temp_nonzero = False
        has_dataset_split_metadata = '"dataset_split": args.split' in source
        has_do_sample_metadata = '"do_sample": args.do_sample' in source
        has_seed_metadata = '"seed": args.seed' in source
        has_model_source_metadata = '"model_source":' in source

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument":
                    for arg in node.args:
                        s = _const_str(arg)
                        if s == "--temperature":
                            for kw in node.keywords:
                                if (
                                    kw.arg == "default"
                                    and isinstance(kw.value, ast.Constant)
                                    and kw.value.value not in (0, 0.0)
                                ):
                                    default_temp_nonzero = True
                        if s == "--seed":
                            parser_has_seed = True
                if node.func.attr == "generate":
                    for kw in node.keywords:
                        if (
                            kw.arg == "do_sample"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            do_sample_true = True
            if isinstance(node, ast.Name) and node.id == "checkpoint_path":
                checkpoint_arg_used = True

        if default_temp_nonzero:
            add(
                eval_py,
                "eval.nondeterministic.default_temp",
                "Evaluation defaults to temperature=0.7, which makes headline accuracy nondeterministic.",
            )
        if do_sample_true:
            add(
                eval_py,
                "eval.nondeterministic_sampling",
                "HF evaluation uses do_sample=True instead of deterministic decoding, weakening rigor and reproducibility.",
            )
        if not parser_has_seed:
            add(
                eval_py,
                "eval.missing_seed",
                "Evaluation ctx.script has no seed control for stochastic generation.",
            )
        if not parser_has_split:
            add(
                eval_py,
                "eval.missing_split_arg",
                "Evaluation ctx.script does not record which dataset split it evaluates.",
            )
        if parser_has_split and not split_choices_locked:
            add(
                eval_py,
                "eval.unlocked_split",
                "Evaluation ctx.script allows non-test splits; held-out evaluation should be locked to the GSM8K test split to avoid accidental train-set reporting.",
            )
        if not checkpoint_arg_used:
            add(
                eval_py,
                "eval.unused_checkpoint_path",
                "--checkpoint_path is declared but never used, so local checkpoint evaluation is broken/misleading.",
            )
        if fallback_last_number:
            add(
                eval_py,
                "eval.lenient_answer_extraction",
                "Answer extraction falls back to the last number in the response, which can overcount correctness and invite benchmark leakage.",
            )
        if not has_dataset_split_metadata:
            add(
                eval_py,
                "eval.missing_split_metadata",
                "Saved evaluation results do not record the dataset split, weakening auditability.",
            )
        if not has_do_sample_metadata:
            add(
                eval_py,
                "eval.missing_sampling_metadata",
                "Saved evaluation results do not record whether decoding was greedy or sampled.",
            )
        if not has_seed_metadata:
            add(
                eval_py,
                "eval.missing_seed_metadata",
                "Saved evaluation results do not record the evaluation seed.",
            )
        if not has_model_source_metadata:
            add(
                eval_py,
                "eval.missing_model_source_metadata",
                "Saved evaluation results do not record the exact checkpoint/model source used for evaluation.",
            )

    def _run(cmd, cwd: Path):
        return subprocess.run(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

    def check_latex_builds():
        cleanup = [
            "grpo_agentic_llm_paper.aux",
            "grpo_agentic_llm_paper.bbl",
            "grpo_agentic_llm_paper.blg",
            "grpo_agentic_llm_paper.log",
            "grpo_agentic_llm_paper.out",
            "grpo_agentic_llm_paper.pdf",
            "grpo_agentic_llm_paper_anonymous.aux",
            "grpo_agentic_llm_paper_anonymous.log",
            "grpo_agentic_llm_paper_anonymous.out",
            "grpo_agentic_llm_paper_anonymous.pdf",
            "supplementary_appendix.aux",
            "supplementary_appendix.log",
            "supplementary_appendix.out",
            "supplementary_appendix.pdf",
        ]
        for name in cleanup:
            path = ctx.FINAL_DIR / name
            if path.exists():
                path.unlink()

        steps = [
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "grpo_agentic_llm_paper.tex",
                ],
                "latex.main.pass1",
            ),
            (["bibtex", "grpo_agentic_llm_paper"], "latex.main.bibtex"),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "grpo_agentic_llm_paper.tex",
                ],
                "latex.main.pass2",
            ),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "grpo_agentic_llm_paper.tex",
                ],
                "latex.main.pass3",
            ),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "grpo_agentic_llm_paper_anonymous.tex",
                ],
                "latex.anonymous",
            ),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "supplementary_appendix.tex",
                ],
                "latex.supplementary.pass1",
            ),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "supplementary_appendix.tex",
                ],
                "latex.supplementary.pass2",
            ),
        ]

        try:
            for cmd, code in steps:
                result = _run(cmd, ctx.FINAL_DIR)
                if result.returncode != 0:
                    add(ctx.FINAL_DIR / cmd[-1], code, f"LaTeX build step failed: {' '.join(cmd)}")
                    break
                if code == "latex.main.bibtex" and "Warning--empty journal" in result.stdout:
                    add(
                        ctx.FINAL_DIR / "references.bib",
                        "latex.bibtex.empty_journal",
                        "BibTeX emitted 'empty journal' warnings for cited references, so the bibliography metadata is incomplete.",
                    )
        finally:
            for name in cleanup:
                path = ctx.FINAL_DIR / name
                if path.exists():
                    path.unlink()

    def check_result_jsons():
        required_config = {
            "model",
            "model_source",
            "dataset",
            "dataset_config",
            "dataset_split",
            "n_samples",
            "temperature",
            "do_sample",
            "seed",
            "max_tokens",
            "test_size",
        }
        required_summary = {
            "correct",
            "incorrect",
            "errors",
            "attempted",
            "accuracy",
            "accuracy_percent",
        }

        for path in result_jsons:
            data = json.loads(read(path))
            if data.get("schema_version") != 2:
                add(
                    path,
                    "results.schema_version",
                    "Result JSON does not declare the current schema_version=2.",
                )
            if data.get("evaluation_status") not in {"completed", "failed"}:
                add(
                    path,
                    "results.status",
                    "Result JSON must declare evaluation_status as 'completed' or 'failed'.",
                )
            config = data.get("config")
            summary = data.get("summary")
            if not isinstance(config, dict) or not required_config.issubset(config):
                add(
                    path,
                    "results.config",
                    "Result JSON is missing required evaluation provenance fields in config.",
                )
                continue
            if not isinstance(summary, dict) or not required_summary.issubset(summary):
                add(path, "results.summary", "Result JSON is missing required summary fields.")
                continue
            if config.get("dataset_split") != "test":
                add(
                    path,
                    "results.non_test_split",
                    "Bundled GSM8K result JSON must be explicitly marked as test-split evaluation.",
                )
            attempted = summary.get("attempted")
            if attempted != summary.get("correct", 0) + summary.get("incorrect", 0):
                add(
                    path,
                    "results.attempted_mismatch",
                    "attempted must equal correct + incorrect in the result summary.",
                )
            if data.get("evaluation_status") == "failed" and not data.get("failure_reason"):
                add(
                    path,
                    "results.failure_reason",
                    "Failed evaluations must record a failure_reason.",
                )
            if data.get("evaluation_status") == "completed" and attempted == 0:
                add(
                    path,
                    "results.completed_zero_attempts",
                    "Completed evaluations must have at least one attempted example.",
                )

    check_paper()
    check_eval_script()
    check_latex_builds()
    check_result_jsons()

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("scientific_issues", get_issues))
