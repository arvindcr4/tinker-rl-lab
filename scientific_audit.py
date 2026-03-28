#!/usr/bin/env python3
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_TEX = ROOT / "reports/final/grpo_agentic_llm_paper.tex"
PAPER_TEX_ANON = ROOT / "reports/final/grpo_agentic_llm_paper_anonymous.tex"
PAPER_MD = ROOT / "reports/final/grpo_agentic_llm_paper.md"
REPORT_MD = ROOT / "reports/final/capstone_final_report.md"
SUBMISSION_CHECKLIST = ROOT / "reports/final/SUBMISSION_CHECKLIST.md"
EVAL_PY = ROOT / "reports/final/evaluate_gsm8k_test.py"

issues = []


def add(path: Path, code: str, message: str):
    issues.append((str(path.relative_to(ROOT)), code, message))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_paper():
    tex = read(PAPER_TEX)
    anon = read(PAPER_TEX_ANON)
    paper_md = read(PAPER_MD)
    md = read(REPORT_MD)
    checklist = read(SUBMISSION_CHECKLIST)

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if abstract_match:
        abstract = abstract_match.group(1).lower()
        if "held-out" not in abstract and "training-set" not in abstract and "evaluation scope" not in abstract:
            add(PAPER_TEX, "paper.abstract.scope", "LaTeX abstract reports GSM8K gains without explicitly saying they are training-set reward metrics, risking overclaim.")

    if "held-out" not in tex.lower() and "training-set reward" not in tex.lower():
        add(PAPER_TEX, "paper.global.scope", "LaTeX paper lacks an explicit held-out-vs-training-set evaluation scope warning.")

    if "publishable confidence intervals" in md.lower():
        add(REPORT_MD, "report.overclaim.publishable", "Capstone report claims 'publishable confidence intervals' despite n=3 seeds and no held-out evaluation.")

    anon_abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", anon, re.S)
    if anon_abstract_match:
        anon_abstract = anon_abstract_match.group(1).lower()
        if "held-out" not in anon_abstract and "training-set" not in anon_abstract and "evaluation scope" not in anon_abstract:
            add(PAPER_TEX_ANON, "paper_anon.abstract.scope", "Anonymous LaTeX abstract still reports GSM8K gains without an explicit training-set-vs-held-out caveat.")

    if "held-out" not in paper_md.lower() and "training-set reward" not in paper_md.lower():
        add(PAPER_MD, "paper_md.global.scope", "Markdown paper lacks an explicit held-out-vs-training-set scope warning.")

    if re.search(r"\|\s*GSM8K\s*\|\s*30\.0% \± 2\.5% \(3 seeds\)\s*\|", checklist):
        add(SUBMISSION_CHECKLIST, "checklist.gsm8k.label", "Submission checklist labels GSM8K as a generic result instead of explicitly marking it as training-set reward.")


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_eval_script():
    tree = ast.parse(read(EVAL_PY))
    source = read(EVAL_PY)

    parser_has_seed = "--seed" in source
    checkpoint_arg_used = False
    fallback_last_number = "Fallback: extract last number" in source
    do_sample_true = False
    default_temp_nonzero = False
    extract_uses_last_number = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                for arg in node.args:
                    s = _const_str(arg)
                    if s == "--temperature":
                        for kw in node.keywords:
                            if kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value not in (0, 0.0):
                                default_temp_nonzero = True
                    if s == "--seed":
                        parser_has_seed = True
            if node.func.attr == "generate":
                for kw in node.keywords:
                    if kw.arg == "do_sample" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        do_sample_true = True
        if isinstance(node, ast.Name) and node.id == "checkpoint_path":
            checkpoint_arg_used = True

    if default_temp_nonzero:
        add(EVAL_PY, "eval.nondeterministic.default_temp", "Evaluation defaults to temperature=0.7, which makes headline accuracy nondeterministic.")
    if do_sample_true:
        add(EVAL_PY, "eval.nondeterministic_sampling", "HF evaluation uses do_sample=True instead of deterministic decoding, weakening rigor and reproducibility.")
    if not parser_has_seed:
        add(EVAL_PY, "eval.missing_seed", "Evaluation script has no seed control for stochastic generation.")
    if not checkpoint_arg_used:
        add(EVAL_PY, "eval.unused_checkpoint_path", "--checkpoint_path is declared but never used, so local checkpoint evaluation is broken/misleading.")
    if fallback_last_number:
        add(EVAL_PY, "eval.lenient_answer_extraction", "Answer extraction falls back to the last number in the response, which can overcount correctness and invite benchmark leakage.")


def main():
    check_paper()
    check_eval_script()
    print(f"METRIC audit_issues={len(issues)}")
    for path, code, message in issues:
        print(f"ISSUE {code} {path}: {message}")


if __name__ == "__main__":
    main()
