#!/usr/bin/env python3
"""Submit Oracle invention queries one after another, without waiting for each to finish."""
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/Users/arvind/Developer/tinker-rl-lab")
QUERIES_FILE = REPO_ROOT / "oracle_invention_queries.md"
OUT_DIR = Path("/tmp/oracle_invention_outputs")
OUT_DIR.mkdir(exist_ok=True)

COPY_PROFILE = os.path.expanduser("~/Library/Application Support/Google/Chrome")
TIMEOUT = "30m"
SUBMIT_DELAY_SECONDS = 30  # pause between submissions to avoid stampede


def parse_queries(path: Path) -> list[tuple[str, str]]:
    text = path.read_text()
    blocks = re.split(r"\n(?=# \d+\.\d+ — )", text)
    queries = []
    for block in blocks:
        m = re.search(r"# (\d+\.\d+) — ", block)
        if not m:
            continue
        qid = m.group(1)
        cmd_match = re.search(r"(npx -y @steipete/oracle.*?)(?=\n# |\n```|\Z)", block, re.DOTALL)
        if not cmd_match:
            continue
        raw = cmd_match.group(1)
        lines = raw.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.rstrip()
            if line.endswith("\\"):
                line = line[:-1].rstrip()
            cleaned_lines.append(line)
        cmd = " ".join(" ".join(cleaned_lines).split())
        queries.append((qid, cmd))
    return queries


def build_command(qid: str, base_cmd: str) -> str:
    out_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    log_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.log"
    slug = f"zvf-invention-{qid.replace('.', '-')}"
    flags = f"--engine browser --model \"5.5 Pro Extended\" --browser-model-strategy current --copy-profile '{COPY_PROFILE}' --timeout {TIMEOUT} --slug {slug} --write-output {out_file}"
    return f"cd {REPO_ROOT} && {base_cmd} {flags} > {log_file} 2>&1"


def main():
    queries = parse_queries(QUERIES_FILE)
    print(f"Submitting {len(queries)} queries with {SUBMIT_DELAY_SECONDS}s spacing")
    print(f"Outputs: {OUT_DIR}\n")

    procs = []
    for idx, (qid, base_cmd) in enumerate(queries, 1):
        cmd = build_command(qid, base_cmd)
        print(f"[{idx}/{len(queries)}] Submitting {qid}...")
        proc = subprocess.Popen(cmd, shell=True)
        procs.append((qid, proc))
        if idx < len(queries):
            time.sleep(SUBMIT_DELAY_SECONDS)

    print(f"\nAll {len(queries)} queries submitted.")
    print("PIDs:", ", ".join(f"{qid}={proc.pid}" for qid, proc in procs))
    print(f"Monitor with: tail -f {OUT_DIR}/*.log")


if __name__ == "__main__":
    main()
