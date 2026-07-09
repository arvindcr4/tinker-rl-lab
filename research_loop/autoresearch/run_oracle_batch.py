#!/usr/bin/env python3
"""Run Oracle invention queries in small concurrent batches, skipping ones that
already produced a non-empty .md output."""
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
TIMEOUT = "45m"  # generous; PDF-only queries can take 10-70 min
CONCURRENCY = 3
SUBMIT_DELAY_SECONDS = 10


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


def md_exists(qid: str) -> bool:
    md = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    return md.exists() and md.stat().st_size > 200


def build_command(qid: str, base_cmd: str) -> str:
    out_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.md"
    log_file = OUT_DIR / f"invention_{qid.replace('.', '_')}.log"
    slug = f"zvf-invention-{qid.replace('.', '-')}"
    flags = (
        f"--engine browser --model \"5.5 Pro Extended\" --browser-model-strategy current "
        f"--copy-profile '{COPY_PROFILE}' --timeout {TIMEOUT} "
        f"--slug {slug} --write-output {out_file}"
    )
    return f"cd {REPO_ROOT} && {base_cmd} {flags} > {log_file} 2>&1"


def wait_batch(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for qid, proc in procs:
        try:
            rc = proc.wait(timeout=45 * 60)
            print(f"  {qid} finished with exit code {rc}")
        except subprocess.TimeoutExpired:
            print(f"  {qid} timed out; killing")
            proc.kill()
            proc.wait()


def main():
    queries = parse_queries(QUERIES_FILE)
    pending = [(qid, cmd) for qid, cmd in queries if not md_exists(qid)]
    skipped = [(qid, cmd) for qid, cmd in queries if md_exists(qid)]
    print(f"Total queries: {len(queries)}")
    print(f"Already completed (skipped): {len(skipped)}")
    print(f"Pending: {len(pending)}")
    print(f"Concurrency: {CONCURRENCY}, output dir: {OUT_DIR}\n")

    idx = 0
    while idx < len(pending):
        batch = pending[idx : idx + CONCURRENCY]
        procs = []
        print(f"Batch {idx // CONCURRENCY + 1}: " + ", ".join(qid for qid, _ in batch))
        for qid, base_cmd in batch:
            cmd = build_command(qid, base_cmd)
            proc = subprocess.Popen(cmd, shell=True)
            procs.append((qid, proc))
            if qid != batch[-1][0]:
                time.sleep(SUBMIT_DELAY_SECONDS)
        wait_batch(procs)
        idx += CONCURRENCY
        if idx < len(pending):
            time.sleep(30)

    print("\nAll pending queries processed.")


if __name__ == "__main__":
    main()
