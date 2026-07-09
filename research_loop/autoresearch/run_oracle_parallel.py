#!/usr/bin/env python3
"""Run the 40 Oracle invention queries in parallel (limited concurrency)."""
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path("/Users/arvind/Developer/tinker-rl-lab")
QUERIES_FILE = REPO_ROOT / "oracle_invention_queries.md"
OUT_DIR = Path("/tmp/oracle_invention_outputs")
OUT_DIR.mkdir(exist_ok=True)

COPY_PROFILE = os.path.expanduser("~/Library/Application Support/Google/Chrome")
TIMEOUT = "30m"
MAX_WORKERS = 3


def parse_queries(path: Path) -> list[tuple[str, str]]:
    """Return list of (query_id, full_npx_command)."""
    text = path.read_text()
    # Split by comment lines like `# 1.1 — ...`
    blocks = re.split(r"\n(?=# \d+\.\d+ — )", text)
    queries = []
    for block in blocks:
        m = re.search(r"# (\d+\.\d+) — ", block)
        if not m:
            continue
        qid = m.group(1)
        # Extract the npx command (skip the comment line)
        cmd_match = re.search(r"(npx -y @steipete/oracle.*?)(?=\n# |\n```|\Z)", block, re.DOTALL)
        if not cmd_match:
            continue
        raw = cmd_match.group(1)
        # Remove shell line-continuation backslashes so multi-line commands collapse cleanly.
        lines = raw.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.rstrip()
            if line.endswith("\\"):
                line = line[:-1].rstrip()
            cleaned_lines.append(line)
        cmd = " ".join(" ".join(cleaned_lines).split())  # normalize whitespace
        queries.append((qid, cmd))
    return queries


def build_command(qid: str, base_cmd: str) -> str:
    out_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    log_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.log"
    slug = f"zvf-invention-{qid.replace('.', '-')}"  # 3-5 words-ish
    # Add engine/browser flags if not present
    flags = f"--engine browser --model \"5.5 Pro Extended\" --browser-model-strategy current --copy-profile '{COPY_PROFILE}' --timeout {TIMEOUT} --slug {slug} --write-output {out_file}"
    return f"cd {REPO_ROOT} && {base_cmd} {flags} > {log_file} 2>&1"


def run_one(qid: str, base_cmd: str) -> tuple[str, int, str, str]:
    cmd = build_command(qid, base_cmd)
    out_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    log_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.log"
    print(f"[START] {qid}")
    result = subprocess.run(cmd, shell=True)
    print(f"[DONE] {qid} -> exit {result.returncode}")
    return qid, result.returncode, str(out_file), str(log_file)


def md_exists(qid: str) -> bool:
    md = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    return md.exists() and md.stat().st_size > 200


def main():
    queries = parse_queries(QUERIES_FILE)
    pending = [(qid, cmd) for qid, cmd in queries if not md_exists(qid)]
    skipped = [(qid, cmd) for qid, cmd in queries if md_exists(qid)]
    print(f"Total queries: {len(queries)}")
    print(f"Already completed (skipped): {len(skipped)}")
    print(f"Pending to run: {len(pending)}")
    print(f"Running with max {MAX_WORKERS} concurrent. Outputs: {OUT_DIR}\n")

    if not pending:
        print("No pending queries to run.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, qid, cmd): qid for qid, cmd in pending}
        for fut in as_completed(futures):
            qid, rc, out, log = fut.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"  {qid}: {status} | out={out} | log={log}")

    print("\nAll jobs finished. Logs and outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
