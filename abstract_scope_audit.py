#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []

    # Validate each abstract-bearing source independently:
    #   - main paper (main_tex)
    #   - anonymous paper (anon_tex)
    #   - capstone final report (text)
    sources = {
        "main_tex": ctx.main_tex,
        "anon_tex": ctx.anon_tex,
        "capstone": ctx.text,
    }

    # ctx.* properties return lowercased text, so patterns must match lowercase.
    article_pat = (
        re.escape(r"\begin{abstract}") + "(.*?)" + re.escape(r"\end{abstract}")
    )
    chapter_pat = (
        re.escape(r"\chapter*{abstract}")
        + "(.*?)(?="
        + re.escape(r"\chapter")
        + "|"
        + re.escape(r"\section")
        + "|"
        + re.escape(r"\begin{document}")
        + ")"
    )

    for name, src in sources.items():
        # Match either \begin{abstract}...\end{abstract} (article-style) or
        # \chapter*{Abstract}...\chapter/\section (report-style capstone).
        m = re.search(article_pat + "|" + chapter_pat, src, re.S)
        if m:
            abstract = m.group(1) or m.group(2) or ""
        else:
            abstract = src[:2000]

        if "custom" not in abstract:
            issues.append(f"{name}_abstract_missing_custom_eval_caveat")
        if "50-problem subset" not in abstract:
            issues.append(f"{name}_abstract_missing_humaneval_subset_caveat")
        if "training-set reward" not in abstract:
            issues.append(f"{name}_abstract_missing_training_reward_caveat")
        if "held-out" not in abstract:
            issues.append(f"{name}_abstract_missing_heldout_caveat")

    return issues

if __name__ == '__main__':
    run_audit('abstract_issues', get_issues)