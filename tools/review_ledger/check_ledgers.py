"""Gate for the Deep Review portfolio pass.

Checks, per paper that has a dispositions file:
  * every finding id in findings.sqlite has exactly one disposition with a valid status;
  * every ledger row marked verified/recomputed cites a source artifact that exists;
  * no ledger row is left 'unsourced' (unsourced claims must be removed, rule 2).
Ingests dispositions into findings.sqlite and writes reports/deepreview/STATUS.md.

Usage: python tools/review_ledger/check_ledgers.py [--strict]
Exit code 1 under --strict if any check fails.
"""

import csv
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "reports/deepreview"
PATHLIKE = re.compile(r"[\w.*\-/{}\[\]]+")
_TRACKED_NAMES = None


def tracked_names():
    global _TRACKED_NAMES
    if _TRACKED_NAMES is None:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
        _TRACKED_NAMES = {Path(p).name for p in out.splitlines()}
    return _TRACKED_NAMES


def missing_sources(src):
    """Return the path-like words in a ledger source cell that resolve to nothing.

    Non-path words (e.g. "analytic", run ids, prose) are ignored; brace lists are
    checked by their directory prefix, globs must match, bare file names must be
    tracked somewhere in the repo.
    """
    missing = []
    for word in PATHLIKE.findall(src):
        word = word.split("#")[0].split(":")[0].split("{")[0].rstrip(".,)")
        if not ("/" in word or re.search(r"\.\w{1,6}$", word)) or re.fullmatch(r"[\d.]+", word):
            continue
        if word.startswith(("http", "W&B", "wandb", "hf", "arXiv", "doi", "www.")):
            continue
        if "*" in word or "[" in word:
            ok = any(ROOT.glob(word)) or Path(word).name in tracked_names()
        elif "/" in word:
            ok = (ROOT / word).exists()
        else:
            ok = word in tracked_names() or any(ROOT.rglob(word))
        if not ok:
            missing.append(word)
    return missing


STATUSES = {
    "fixed",
    "refuted",
    "already-fixed",
    "unsourced-removed",
    "needs-compute",
    "wontfix-scope",
}


def main():
    strict = "--strict" in sys.argv
    con = sqlite3.connect(REV / "findings.sqlite")
    ids_by_paper: dict[str, dict[int, str]] = {}
    for fid, paper, sev in con.execute("select id, paper, severity from findings"):
        ids_by_paper.setdefault(paper, {})[fid] = sev
    problems, lines = (
        [],
        [
            "# Deep Review portfolio status",
            "",
            "| Paper | Findings | Covered | " + " | ".join(sorted(STATUSES)) + " | Open CRITICAL |",
            "|---|---|---|" + "---|" * len(STATUSES) + "---|",
        ],
    )
    for paper, ids in sorted(ids_by_paper.items()):
        path = REV / "dispositions" / f"{paper}.jsonl"
        if not path.exists():
            crit = sum(1 for s in ids.values() if s == "CRITICAL")
            lines.append(
                f"| {paper} | {len(ids)} | 0 | " + " | ".join("" for _ in STATUSES) + f" | {crit} |"
            )
            continue
        seen = {}
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            fid = int(d["id"])
            if fid not in ids:
                problems.append(f"{paper}: line {n} id {fid} not in this paper")
            elif fid in seen:
                problems.append(f"{paper}: id {fid} has more than one disposition")
            if d.get("status") not in STATUSES:
                problems.append(f"{paper}: id {fid} invalid status {d.get('status')!r}")
            seen[fid] = d
            con.execute(
                "update findings set status=?, disposition=?, evidence=?, fixed_in=? where id=?",
                (
                    d.get("status"),
                    d.get("status"),
                    d.get("evidence"),
                    json.dumps(d.get("files")),
                    fid,
                ),
            )
        missing = sorted(set(ids) - set(seen))
        if missing:
            problems.append(
                f"{paper}: {len(missing)} findings without disposition (e.g. {missing[:5]})"
            )
        c = Counter(d.get("status") for d in seen.values())
        open_crit = sum(
            1
            for fid, s in ids.items()
            if s == "CRITICAL"
            and seen.get(fid, {}).get("status") in (None, "needs-compute", "wontfix-scope")
        )
        lines.append(
            f"| {paper} | {len(ids)} | {len(seen)} | "
            + " | ".join(str(c.get(s, "")) for s in sorted(STATUSES))
            + f" | {open_crit} |"
        )
        ledger = REV / "ledgers" / f"{paper}.tsv"
        if ledger.exists():
            for row in csv.reader(ledger.open(), delimiter="\t"):
                if len(row) < 6 or row[0].startswith("#") or row[0] == "location":
                    continue
                status, src = row[5].strip(), row[3].strip()
                if status == "unsourced":
                    problems.append(f"{paper}: ledger keeps an unsourced claim at {row[0]}")
                if status in ("verified", "recomputed"):
                    for p in missing_sources(src):
                        problems.append(f"{paper}: ledger source missing: {p} (at {row[0]})")
    con.commit()
    lines += ["", f"## Problems ({len(problems)})", ""] + [f"- {p}" for p in problems]
    (REV / "STATUS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(problems) - 1 :] if problems else ["gate: OK"]))
    return 1 if (strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
