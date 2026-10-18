"""Parse the combined Deep Review report into a SQLite findings database.

Usage: python tools/review_ledger/build_findings_db.py [ALL_REVIEWS.md] [batch.json]
Writes reports/deepreview/findings.sqlite and findings.jsonl.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".amp/in/artifacts/deepreview/ALL_REVIEWS.md"
)
BATCH = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".amp/in/artifacts/deepreview/batch.json"
OUT = ROOT / "reports/deepreview"
HEAD = re.compile(
    r'^## \[(CRITICAL|MAJOR|MINOR|SUGGESTION|[A-Z_ ]+)\]\s+(.*?)\s*(?:\(p\.\s*([0-9]+)\s*"(.*)"\))?\s*$'
)


def main():
    text = SRC.read_text()
    batch = json.loads(BATCH.read_text()) if BATCH.exists() else {}
    papers = re.split(r"\n# (?!HIGH LEVEL SUMMARY|Anchored comments)", text)[1:]
    rows, summaries = [], []
    for block in papers:
        name = block.split("\n", 1)[0].strip()
        summ = block.split("# Anchored comments", 1)[0]
        summaries.append(
            (name, batch.get(name, {}).get("path"), batch.get(name, {}).get("review"), summ.strip())
        )
        anchored = (
            block.split("# Anchored comments", 1)[1] if "# Anchored comments" in block else ""
        )
        cur = None
        for line in anchored.split("\n"):
            m = HEAD.match(line)
            if m:
                if cur:
                    rows.append(cur)
                cur = {
                    "paper": name,
                    "severity": m.group(1).strip(),
                    "title": m.group(2).strip(),
                    "page": int(m.group(3)) if m.group(3) else None,
                    "anchor": m.group(4),
                    "body": "",
                }
            elif cur is not None:
                cur["body"] += line + "\n"
        if cur:
            rows.append(cur)
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "findings.sqlite"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute("create table papers(name text primary key, pdf text, review text, summary text)")
    con.executemany("insert into papers values (?,?,?,?)", summaries)
    con.execute("""create table findings(id integer primary key, paper text, severity text, title text,
                   page integer, anchor text, body text,
                   status text default 'open', disposition text, evidence text, fixed_in text)""")
    con.executemany(
        "insert into findings(paper,severity,title,page,anchor,body) values (:paper,:severity,:title,:page,:anchor,:body)",
        [{**r, "body": r["body"].strip()} for r in rows],
    )
    con.commit()
    with open(OUT / "findings.jsonl", "w") as fh:
        for i, r in enumerate(rows, 1):
            fh.write(json.dumps({"id": i, **{k: v for k, v in r.items() if k != "body"}}) + "\n")
    for sev, n in con.execute(
        "select severity, count(*) from findings group by severity order by 2 desc"
    ):
        print(sev, n)
    print("total", len(rows), "papers", len(summaries))


if __name__ == "__main__":
    main()
