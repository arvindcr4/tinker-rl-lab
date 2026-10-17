#!/usr/bin/env python3
"""Build a double-blind anonymized export of the repository into dist/anon_export/.

Tracked files only (`git ls-files`). Default mode copies the claim-C (stack
provenance + stackdiff) reviewer set; `--all` copies the whole tree. Both modes
apply the same exclusions and identifier rewrites, then leak-scan the export
and exit 1 if any identifier survives. Nothing is uploaded.

    python3 tools/anonymize_export.py            # claim-C include set
    python3 tools/anonymize_export.py --all      # full tree (minus exclusions)
    python3 tools/anonymize_export.py --out DIR  # other destination
"""
import argparse, fnmatch, json, os, re, shutil, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BIN = 5 * 1024 * 1024

# Default include set (glob patterns over repo-relative paths).
INCLUDE = [
    "platform_*/**", "utils/**", "tests/**", "tools/**",
    "platform_hybrid/registry/**",
    "platform_hybrid/experiments/results/samestack_*.json",
    "platform_hybrid/experiments/results/tinker_gsm8k_zvf_*.json",
    "platform_hybrid/experiments/results/tinker_wandb_registry/*.jsonl",
    "REPRODUCE.md", "LICENSE*", "requirements*.txt", "pyproject.toml",
    "uv.lock", "Dockerfile", "setup.py", "setup.cfg", "conftest.py", ".env.example",
]
# Default mode keeps only result files from the explicit list above.
RESULTS_DIR = "platform_hybrid/experiments/results/"

# (reason, predicate on repo-relative path) — first match wins.
_DOC = r"\.(md|pdf|docx?|tex|txt|odt|rtf)$"
EXCLUDE = [
    ("env", lambda p: re.search(r"(^|/)\.env(?!\.example$)", p)),
    ("secret", lambda p: re.search(r"(secret|credential|\.netrc|(^|/)netrc|\.pem$|\.key$|(^|/)tokens?\.(json|txt)$)", p, re.I)),
    ("wandb_dir", lambda p: re.search(r"(^|/)wandb/", p)),
    ("bulk_dir", lambda p: re.match(r"(platform_[a-z_]+/reports|outputs|dist|archive|reports|submission|finish|deep-dive|research|\.codex[^/]*|\.claude|\.hyperresearch|autoresearch)/", p)),
    ("nested_archive", lambda p: "/archive/" in p),
    ("personal_doc", lambda p: re.search(r"(timesheet|payslip|aadhaar|pan_card|offer[_-]?letter|invoice|(^|/|_)cv[_.-])", p, re.I)
        or re.search(r"(^|/)[^/]*(resume|curriculum)[^/]*" + _DOC, p, re.I)),
    ("institution_doc", lambda p: re.search(r"(thesis|defen[cs]e|pes[_ -]|phase[0-9]?[_-]?review|viva|capstone|blind_review|university_submission|esa_phase|/submissions/|pes_review|public_documents)", p, re.I)),
    ("office_binary", lambda p: re.search(r"\.(pdf|pptx?|docx?|xlsx?|key|odp)$", p, re.I)),
    ("latex_build", lambda p: re.search(r"\.(aux|log|out|bbl|blg|fls|fdb_latexmk|synctex\.gz|toc|nav|snm)$", p)),
    ("this_tool", lambda p: p == "tools/anonymize_export.py"),
    ("pycache", lambda p: "__pycache__/" in p or p.endswith(".pyc")),
]

# Ordered rewrite rules: (rule_id, pattern, replacement, flags). Order matters.
RULES = [
    ("R01_github_url", r"(https?://)?(www\.)?github\.com/arvindcr4/tinker-rl-lab(\.git)?", "https://anonymous.4open.science/r/tinker-rl-lab", re.I),
    ("R02_github_user", r"(https?://)?(www\.)?github\.com/arvindcr4\b", "https://anonymous.4open.science/r", re.I),
    ("R03_wandb_entity", r"arvindcr4[-_]pes[-_]university", "anon-entity", re.I),
    ("R04_hf_url", r"(https?://)?huggingface\.co/(datasets/|spaces/)?arvindcr4\b", r"https://huggingface.co/\2anon-org", re.I),
    ("R05_hf_repo_id", r"\barvindcr4/", "anon-org/", re.I),
    ("R06_email", r"[A-Za-z0-9._%+\-]+@(gmail\.com|pes\.edu|pesu\.[a-z.]+|pesu\.edu)", "anon@example.org", re.I),
    ("R07_home_path", r"/(Users|home)/arvind\b", "/home/anon", 0),
    ("R07b_any_macos_home", r"/Users/[A-Za-z0-9._\-]+", "/home/anon", 0),
    ("R08_handle", r"arvindcr4", "anon-author", re.I),
    ("R09_full_name", r"\bArvind(\s+C\.?\s*R\.?(?![A-Za-z]))?", "Anonymous Author", re.I),
    ("R09b_name_embedded", r"Arvind(_?C\.?_?R)?", "AnonAuthor", re.I),
    ("R10_supervisor", r"\b(Dr\.?\s+)?Ramesh(\s+Prakash)?(\s+Guledgudd)?\b|\bGuledgudd\b", "Anonymous Advisor", re.I),
    ("R11_student_id", r"\bPES[0-9][A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}\b", "ANON-ID", re.I),
    ("R12_institution", r"PES\s+University\b|\bPESU\b|\bPES\s+Institute\s+of\s+Technology\b|\bPESIT\b|\bPES\b", "Anonymous Institution", 0),
    ("R12b_pes_token", r"(?<![A-Za-z])PESU?(?=[_\-])|(?<=[_\-])PESU?(?![A-Za-z])", "ANONINST", 0),
    ("R13_pes_lower", r"\bpes[-_ ]university\b|\bpesu\b", "anon-institution", re.I),
    ("R14_employer", r"\b(IQVIA|LanceSoft|Fieldglass)\b", "Anonymous Employer", re.I),
    ("R16_city", r"\b(Bengaluru|Bangalore)\b", "[city]", re.I),
    ("R17_course_code", r"\bUE\d{2}[A-Z]{2}\d{3}\b", "ANON-COURSE", 0),
    ("R18_degree", r"\bM\.\s?Tech\b", "graduate", 0),
    ("R15_phone", r"(?<![\d.])\+?91[\s-]?[6-9]\d{4}[\s-]?\d{5}(?!\d)", "+00 0000000000", 0),
]
COMPILED = [(rid, re.compile(p, f), r) for rid, p, r, f in RULES]

# Leak scan (case-insensitive).
LEAK = re.compile(
    r"arvindcr4|arvind|pes\s*university|\bpesu\b|@gmail|@pes\.edu|guledgudd|"
    r"(?-i:/Users/)|iqvia|lancesoft|bengaluru|bangalore|\bUE\d{2}CS\d{3}|\bPES[0-9][A-Z]{2}\d", re.I)
LEAK_BYTES = re.compile(rb"arvind|pes university|@gmail|@pes\.edu|/Users/|iqvia|lancesoft|guledgudd", re.I)


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def match_any(p, pats):
    return any(fnmatch.fnmatch(p, g) or (g.endswith("/**") and p.startswith(g[:-3] + "/")) for g in pats)


def rewrite(text, counts):
    for rid, rx, rep in COMPILED:
        text, n = rx.subn(rep, text)
        if n:
            counts[rid] += n
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="full tree instead of the claim-C include set")
    ap.add_argument("--out", default=str(ROOT / "dist" / "anon_export"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == ROOT or ROOT not in out.parents and not a.out.startswith("/"):
        sys.exit("refusing to write outside a dedicated directory")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    counts, excluded, included = Counter(), Counter(), []
    dropped_bin = []
    for p in tracked():
        src = ROOT / p
        if not src.is_file() or src.is_symlink():
            excluded["missing_or_symlink"] += 1
            continue
        if not a.all and not match_any(p, INCLUDE):
            excluded["not_in_include_set"] += 1
            continue
        if not a.all and p.startswith(RESULTS_DIR) and not match_any(p, INCLUDE[5:8]):
            excluded["result_not_selected"] += 1
            continue
        reason = next((r for r, f in EXCLUDE if f(p)), None)
        if reason:
            excluded[reason] += 1
            continue
        data = src.read_bytes()
        dst_rel = rewrite(p, Counter())
        dst = out / dst_rel
        try:
            text = data.decode("utf-8")
            if "\0" in text:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "nul")
        except UnicodeDecodeError:
            if len(data) > MAX_BIN:
                excluded["binary_over_5MB"] += 1
                continue
            if LEAK_BYTES.search(data):
                excluded["binary_with_identifier"] += 1
                dropped_bin.append(len(dropped_bin))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(rewrite(text, counts), encoding="utf-8")
        if dst_rel != p:
            counts["path_renamed"] += 1
        included.append(dst_rel)

    manifest = {"mode": "all" if a.all else "claim_c_include_set",
                "n_included": len(included), "files": sorted(included),
                "excluded_by_reason": dict(excluded.most_common()),
                "replacements_by_rule": dict(sorted(counts.items())),
                "rules": [rid for rid, *_ in RULES]}
    (out / "ANON_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    residual = []
    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out).as_posix()
        if LEAK.search(rel):
            residual.append(f"{rel}:path")
        b = f.read_bytes()
        try:
            for i, line in enumerate(b.decode("utf-8").splitlines(), 1):
                if LEAK.search(line):
                    residual.append(f"{rel}:{i}: {line.strip()[:160]}")
        except UnicodeDecodeError:
            if LEAK_BYTES.search(b):
                residual.append(f"{rel}:binary")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"export: {out}  files={len(included)}  size={size / 1e6:.1f} MB")
    print("excluded:", dict(excluded.most_common()))
    print("replacements:", dict(sorted(counts.items())))
    print(f"residual leak matches: {len(residual)}")
    for r in residual[:200]:
        print("  LEAK", r)
    return 1 if residual else 0


if __name__ == "__main__":
    sys.exit(main())
