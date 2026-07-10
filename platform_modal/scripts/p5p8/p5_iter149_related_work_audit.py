"""Iter 149 P5 related-work hardening: comprehensive cite-key audit for every
reference cited in paper_P5_minreport.tex + p5_*.tex sections. Extends the
iter-109 4-standard CrossRef verification to the full P5 bibliography.

Steps:
1. Extract every \cit[et|at|ept]{...} key from paper/sections/p5_*.tex and
   paper/paper_P5_minreport.tex (deduped, counts retained).
2. Parse the matching @entry from paper/references.bib (type, title, authors,
   year, journal/booktitle, doi, note/arxiv id).
3. Score each entry on a 7-field integrity checklist (relaxed for arXiv
   preprints; volume/pages tracked separately because arXiv preprints
   legitimately lack them).
4. Bucket the entries into 4 reporting-standard families:
   - BASE (the 4 verified standards; iter-109 anchor)
   - STAT-RIGOR (Henderson/Agarwal/Colas/Jordan/Miller, statistical rigor)
   - INFRA (HuggingFace TRL/vLLM/Tinker/SGLang, evaluation stack)
   - RL-ALG (PPO, GRPO, DPO + variants)
5. Compute aggregate integrity stats per family with bootstrap CIs (B=1000,
   seed=20260705) on the practical "fully_formed_arxiv_or_proceedings" rate.

The deliverable: structural hardening of the P5 bibliography. Every citation
in P5 is auditable in a single TSV; the reporting-standard lineage that P5
builds on is re-verified structurally (4 base standards), the RL algorithm
family is audited, and the gap (if any) is reported.
"""

import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
P5_TEX = list((PAPER / "sections").glob("p5_*.tex")) + [PAPER / "paper_P5_minreport.tex"]
BIB = PAPER / "references.bib"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. extract cite keys
# -----------------------------------------------------------------------------
CIT_RE = re.compile(r"\\(?:cite|citet|citep|citep|Cite|Citet|Citep|Citep)\*?\{([^}]+)\}")


def extract_cite_keys():
    """Return {key: count} across all P5 tex files."""
    counts = defaultdict(int)
    for f in P5_TEX:
        if not f.exists():
            continue
        text = f.read_text()
        for block in CIT_RE.findall(text):
            for k in block.split(","):
                k = k.strip()
                if k:
                    counts[k] += 1
    return counts


# -----------------------------------------------------------------------------
# 2. parse bib entries
# -----------------------------------------------------------------------------
def parse_bib():
    """Return {key: entry_dict}. entry_dict has fields: type, title, authors,
    year, venue, doi, arxiv, volume, pages, note. Missing fields are None."""
    text = BIB.read_text()
    out = {}
    # @type{ key, ... } block - greedy match until next @type or EOF
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2).strip()
        start = m.end()
        # find matching close brace by depth
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]
        # fields: name = {value}, or name = "value"
        entry = {"type": etype}
        for fm in re.finditer(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\")", body
        ):
            fname = fm.group(1).lower()
            fval = (fm.group(2) or fm.group(3) or "").strip()
            entry[fname] = fval
        out[key] = entry
    return out


# -----------------------------------------------------------------------------
# 3. integrity checklist
# -----------------------------------------------------------------------------
def has_year(e):
    y = e.get("year")
    if not y:
        return False, None
    try:
        yy = int(y)
        ok = 2017 <= yy <= 2026
        return ok, yy
    except ValueError:
        return False, None


def has_author(e):
    a = (e.get("author") or "").strip()
    return bool(a), len(a)


def has_title(e):
    t = (e.get("title") or "").strip()
    return bool(t), len(t)


def has_venue(e):
    v = (e.get("journal") or e.get("booktitle") or "").strip()
    return bool(v), v or None


def has_doi(e):
    d = (e.get("doi") or "").strip()
    return "10." in d, d or None


def has_arxiv(e):
    note = (e.get("note") or "") + " " + (e.get("url") or "")
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", note + " " + (e.get("journal") or ""))
    return m is not None, (m.group(1) if m else None)


def has_volume_or_pages(e):
    v = (e.get("volume") or "").strip()
    p = (e.get("pages") or "").strip()
    return bool(v) or bool(p), (v or p or None)


# -----------------------------------------------------------------------------
# 4. family classifier
# -----------------------------------------------------------------------------
BASE = {
    "mitchell2019modelcards",
    "gebru2021datasheets",
    "bender2018datastatements",
    "pushkarna2022datacards",
}
STAT_RIGOR = {
    "henderson2018deep",
    "agarwal2021deep",
    "colas2019hitchhiker",
    "jordan2024benchmarking",
    "miller2024erroreval",
    "dodge2019show",
    "pineau2020improving",
    "zhang2024benchvariance",
    "hochlehnert2025sober",
    "riddell2024contamination",
}
INFRA = {
    "biderman2024lessons",
    "krakovna2020specification",
    "kwon2023vllm",
    "zheng2024sglang",
    "thinkingmachines2024tinker",
    "hu2024openrlhf",
    "sheng2024verl",
    "vonwerra2022trl",
    "gao2023reward",
    "dao2022flashattention",
}
# RL-ALG is everything else.


def family(key):
    if key in BASE:
        return "BASE"
    if key in STAT_RIGOR:
        return "STAT"
    if key in INFRA:
        return "INFRA"
    return "RL-ALG"


# -----------------------------------------------------------------------------
# 5. fully-formed score
# -----------------------------------------------------------------------------
def fully_formed(e):
    """Relaxed integrity check appropriate for the mixed conference/arXiv
    corpus: an entry is fully_formed iff it has year + author + title + at
    least one of {venue, arxiv} + at least one of {doi, arxiv}.

    Volume/pages are tracked separately because arXiv preprints legitimately
    lack them. This matches ACM + arXiv conventions for an LLM-RL bibliography
    where ~70% of entries are technical-report (arXiv) style.
    """
    ok_y = has_year(e)[0]
    ok_a = has_author(e)[0]
    ok_t = has_title(e)[0]
    ok_v = has_venue(e)[0]
    ok_d = has_doi(e)[0]
    ok_x = has_arxiv(e)[0]
    ok_p = has_volume_or_pages(e)[0]
    legacy_fields = [ok_y, ok_a, ok_t, ok_v, ok_d, ok_x, ok_p]
    legacy_count = sum(legacy_fields)
    # practical "fully identified" — relaxed for arXiv preprints
    practical = ok_y and ok_a and ok_t and (ok_v or ok_x) and (ok_d or ok_x)
    return legacy_count, len(legacy_fields), practical


# -----------------------------------------------------------------------------
# 6. bootstrap
# -----------------------------------------------------------------------------
def bootstrap_ci(prop, n, B=1000, seed=20260705):
    rng = random.Random(seed)
    if n == 0:
        return 0.0, 0.0
    successes = int(round(prop * n))
    failures = n - successes
    samples = []
    for _ in range(B):
        cnt = sum(1 for _ in range(n) if rng.random() < successes / n)
        samples.append(cnt / n)
    samples.sort()
    lo = samples[int(0.025 * B)]
    hi = samples[int(0.975 * B)]
    return lo, hi


# -----------------------------------------------------------------------------
# 7. main
# -----------------------------------------------------------------------------
def main():
    cite_counts = extract_cite_keys()
    bib = parse_bib()

    rows = []
    for key, cnt in sorted(cite_counts.items()):
        e = bib.get(key, {})
        ok_y, yr = has_year(e)
        ok_a, alen = has_author(e)
        ok_t, tlen = has_title(e)
        ok_v, venue = has_venue(e)
        ok_d, doi = has_doi(e)
        ok_x, arxiv = has_arxiv(e)
        ok_p, vp = has_volume_or_pages(e)
        sf, sn, ff = fully_formed(e)
        rows.append(
            {
                "cite_key": key,
                "n_uses": cnt,
                "family": family(key),
                "in_bib": key in bib,
                "entry_type": e.get("type", ""),
                "year": yr or "",
                "title_len": tlen or "",
                "n_authors": alen,
                "venue": (venue or "")[:40],
                "doi": doi or "",
                "arxiv_id": arxiv or "",
                "has_year": int(ok_y),
                "has_author": int(ok_a),
                "has_title": int(ok_t),
                "has_venue": int(ok_v),
                "has_doi": int(ok_d),
                "has_arxiv": int(ok_x),
                "has_volpages": int(ok_p),
                "fields_present": sf,
                "fields_total": sn,
                "fully_formed": int(ff),
            }
        )

    # write per-cite inventory tsv
    fields = list(rows[0].keys())
    out_tsv = OUT / "p5_iter149_cite_inventory.tsv"
    with open(out_tsv, "w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in fields) + "\n")

    # aggregate per family
    fam_stats = defaultdict(lambda: {"n": 0, "ff": 0, "in_bib": 0, "uses": 0,
                                     "doi_missing": 0, "arxiv_missing": 0,
                                     "venue_missing": 0, "vp_missing": 0})
    for r in rows:
        fam_stats[r["family"]]["n"] += 1
        fam_stats[r["family"]]["ff"] += int(r["fully_formed"])
        fam_stats[r["family"]]["in_bib"] += int(r["in_bib"])
        fam_stats[r["family"]]["uses"] += int(r["n_uses"])
        fam_stats[r["family"]]["doi_missing"] += int(not r["has_doi"])
        fam_stats[r["family"]]["arxiv_missing"] += int(not r["has_arxiv"])
        fam_stats[r["family"]]["venue_missing"] += int(not r["has_venue"])
        fam_stats[r["family"]]["vp_missing"] += int(not r["has_volpages"])

    fam_rows = []
    for fam in ("BASE", "STAT", "INFRA", "RL-ALG"):
        if fam not in fam_stats:
            continue
        s = fam_stats[fam]
        n = s["n"]
        ff = s["ff"]
        if n > 0:
            p = ff / n
            lo, hi = bootstrap_ci(p, n)
        else:
            p, lo, hi = 0.0, 0.0, 0.0
        fam_rows.append(
            {
                "family": fam,
                "n_keys": n,
                "n_fully_formed": ff,
                "pct_fully_formed": round(p * 100, 2),
                "ci_lo_pct": round(lo * 100, 2),
                "ci_hi_pct": round(hi * 100, 2),
                "in_bib": s["in_bib"],
                "n_total_uses": s["uses"],
                "n_doi_missing": s["doi_missing"],
                "n_arxiv_missing": s["arxiv_missing"],
                "n_venue_missing": s["venue_missing"],
                "n_vp_missing": s["vp_missing"],
            }
        )

    out_fam = OUT / "p5_iter149_family_stats.tsv"
    with open(out_fam, "w") as f:
        f.write(
            "\t".join(
                [
                    "family",
                    "n_keys",
                    "n_fully_formed",
                    "pct_fully_formed",
                    "ci_lo_pct",
                    "ci_hi_pct",
                    "in_bib",
                    "n_total_uses",
                    "n_doi_missing",
                    "n_arxiv_missing",
                    "n_venue_missing",
                    "n_vp_missing",
                ]
            )
            + "\n"
        )
        for r in fam_rows:
            f.write(
                "\t".join(
                    str(r[k]) for k in (
                        "family",
                        "n_keys",
                        "n_fully_formed",
                        "pct_fully_formed",
                        "ci_lo_pct",
                        "ci_hi_pct",
                        "in_bib",
                        "n_total_uses",
                        "n_doi_missing",
                        "n_arxiv_missing",
                        "n_venue_missing",
                        "n_vp_missing",
                    )
                )
                + "\n"
            )

    # field-by-field gap detection: which checks fail most often?
    field_fail = defaultdict(int)
    for r in rows:
        for k_check in (
            "has_year",
            "has_author",
            "has_title",
            "has_venue",
            "has_doi",
            "has_arxiv",
            "has_volpages",
        ):
            if not r[k_check]:
                field_fail[k_check] += 1

    field_rows = []
    n_total = len(rows)
    for k_check in (
        "has_year",
        "has_author",
        "has_title",
        "has_venue",
        "has_doi",
        "has_arxiv",
        "has_volpages",
    ):
        f = field_fail[k_check]
        p = f / n_total if n_total else 0
        lo, hi = bootstrap_ci(p, n_total) if n_total else (0, 0)
        field_rows.append(
            {
                "field": k_check,
                "n_fail": f,
                "n_total": n_total,
                "pct_fail": round(p * 100, 2),
                "ci_lo_pct": round(lo * 100, 2),
                "ci_hi_pct": round(hi * 100, 2),
            }
        )

    out_fld = OUT / "p5_iter149_field_gaps.tsv"
    with open(out_fld, "w") as f:
        f.write(
            "\t".join(
                ["field", "n_fail", "n_total", "pct_fail", "ci_lo_pct", "ci_hi_pct"]
            )
            + "\n"
        )
        for r in field_rows:
            f.write(
                "\t".join(
                    str(r[k])
                    for k in ("field", "n_fail", "n_total", "pct_fail", "ci_lo_pct", "ci_hi_pct")
                )
                + "\n"
            )

    # json summary
    summary = {
        "n_cite_keys_unique": len(rows),
        "n_total_uses": sum(r["n_uses"] for r in rows),
        "n_entries_in_bib": sum(1 for r in rows if r["in_bib"]),
        "n_fully_formed": sum(int(r["fully_formed"]) for r in rows),
        "family_stats": fam_rows,
        "field_gaps": field_rows,
        "iter": 149,
        "pillar": "P5",
    }
    out_json = OUT / "p5_iter149_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # stdout digest
    print(f"P5 cite inventory: {len(rows)} unique keys, {summary['n_total_uses']} total uses")
    print(f"in_bib: {summary['n_entries_in_bib']}/{len(rows)}")
    print(f"fully_formed: {summary['n_fully_formed']}/{len(rows)}")
    print("family breakdown:")
    for r in fam_rows:
        print(
            f"  {r['family']:>8s}: n={r['n_keys']:2d} ff={r['n_fully_formed']:2d} "
            f"pct={r['pct_fully_formed']:5.1f}% CI=[{r['ci_lo_pct']:.1f},{r['ci_hi_pct']:.1f}]"
        )
    print("field gap counts (fail/N):")
    for r in field_rows:
        print(
            f"  {r['field']:>15s}: {r['n_fail']}/{r['n_total']} = {r['pct_fail']:5.1f}%"
        )


if __name__ == "__main__":
    main()
