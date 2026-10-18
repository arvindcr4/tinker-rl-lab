"""Recount the Appendix E paper-to-taxonomy mapping of the Group 6 report.

Usage: python3 tools/review_ledger/Sem3_Group6_Original_Report_taxonomy_counts.py
Reads platform_tinker/reports/combined/base_final.tex, parses the mapping longtable
(key & title & category & name), and prints rows, unique keys, and counts per
category and per cluster (1-3, 4-6, 7-11, 12-16, 17-19).
"""
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
tex = (REPO / "platform_tinker/reports/combined/base_final.tex").read_text()
start = tex.index(r"\chapter{Paper-to-Taxonomy Mapping}")
body = tex[start:tex.index(r"\section*{Category Distribution Summary}", start)]
rows = re.findall(r"^([^&\n]+?)\s*&\s*([^&\n]+?)\s*&\s*(\d+)\s*&\s*([^\\\n]+?)\s*\\\\", body, re.M)
keys = [r[0].replace(r"\-", "").strip() for r in rows]
cats = Counter(int(r[2]) for r in rows)
clusters = {"Core Algorithms (1-3)": range(1, 4), "Learning Paradigms (4-6)": range(4, 7),
            "Advanced Methods (7-11)": range(7, 12), "Applications (12-16)": range(12, 17),
            "Infrastructure (17-19)": range(17, 20)}
print("rows", len(rows), "unique keys", len(set(keys)))
print("per category", dict(sorted(cats.items())))
print("empty categories", [c for c in range(1, 20) if cats[c] == 0])
for name, rng in clusters.items():
    print(name, sum(cats[c] for c in rng))
