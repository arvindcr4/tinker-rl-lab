#!/usr/bin/env python3
"""Offline structural citation audit for every LaTeX paper root in the workspace.
Checks: orphan cites/refs, missing .bib, duplicate keys/titles/DOIs, required
fields, DOI/arXiv syntax, year sanity, stale .bbl, .blg/.log corroboration.
No online DOI resolution / existence / retraction lookup is performed.
"""

import hashlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
OUT_JSON = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/citation_audit.json")
NOW_YEAR = 2026
SKIP_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "wandb",
    "node_modules",
    "__pycache__",
    ".playwright-mcp",
    ".codebase-memory",
    ".serena",
    ".squad",
    ".hyperresearch",
    ".claude",
    ".codex",
    ".gemini",
    ".omo",
    ".codex-docs",
    ".codex-pdf-open",
    ".codex-pptx",
    ".amp",
}
CITE_RE = re.compile(
    r"\\(cite|citep|citet|citealt|citealp|citeauthor|citeyear|citeyearpar|"
    r"autocite|parencite|textcite|footcite|smartcite|supercite|nocite|"
    r"citeauthor\*|fullcite|footfullcite)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}"
)
INPUT_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
BIBCMD_RE = re.compile(r"\\bibliography\{([^}]+)\}")
BIBRES_RE = re.compile(r"\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}")
BIBSTYLE_RE = re.compile(r"\\bibliographystyle\{([^}]+)\}")
BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        buf, i = [], 0
        while i < len(line):
            c = line[i]
            if c == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(c)
            i += 1
        out.append("".join(buf))
    t = "\n".join(out)
    for env in ("verbatim", "lstlisting", "minted", "Verbatim"):
        t = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", "", t, flags=re.S)
    return t


_READ_CACHE = {}


def read(p):
    k = str(p)
    if k not in _READ_CACHE:
        try:
            _READ_CACHE[k] = Path(p).read_text(errors="ignore")
        except Exception:
            _READ_CACHE[k] = ""
    return _READ_CACHE[k]


_STRIP_CACHE = {}


def strip_cached(p):
    k = str(p)
    if k not in _STRIP_CACHE:
        _STRIP_CACHE[k] = strip_comments(read(p))
    return _STRIP_CACHE[k]


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def walk_files(suffixes):
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".chart-data")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in suffixes:
                found.append(p)
    return sorted(found)


def tier_of(p):
    s = rel(p)
    if s.startswith(".agents/"):
        return "agent-scratch"
    if s.startswith("dist/"):
        return "export-copy"
    if s.startswith("autoresearch/"):
        return "snapshot-copy"
    if "/archive/" in s or s.startswith("archive/"):
        return "archive"
    if "backup" in s:
        return "backup-copy"
    if s.startswith("outputs/"):
        return "dated-export"
    if "/figures/" in s or "/tikz/" in s:
        return "figure"
    if "terminal-bench" in s or s.startswith(("tests/", "verl/")):
        return "fixture/vendor"
    return "authored"


def resolve_input(name, including_file, root_dir):
    name = name.strip()
    cands = []
    for base in (Path(including_file).parent, root_dir, ROOT):
        cands += [base / name, base / (name + ".tex")]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def collect_tex(root_tex):
    """Return list of (path, stripped_text) reachable via input/include."""
    root_dir = Path(root_tex).parent
    seen, stack, files = set(), [Path(root_tex).resolve()], []
    missing_inputs = []
    while stack:
        cur = stack.pop()
        if cur in seen or not cur.is_file():
            continue
        seen.add(cur)
        txt = strip_cached(cur)
        files.append((cur, txt))
        for m in INPUT_RE.finditer(txt):
            if m.group(1).strip().startswith("#") or "\\" in m.group(1):
                continue  # macro parameter / dynamic path, not a literal file
            tgt = resolve_input(m.group(1), cur, root_dir)
            if tgt is None:
                guarded = "IfFileExists" in txt[max(0, m.start() - 250) : m.start()]
                missing_inputs.append(
                    {
                        "from": rel(cur),
                        "line": line_of(txt, m.start()),
                        "input": m.group(1),
                        "guarded": guarded,
                    }
                )
            elif tgt not in seen:
                stack.append(tgt)
    return files, missing_inputs


def parse_bib(path):
    return parse_bib_text(read(path), rel(path))


def parse_bib_text(text, filelabel):
    """Brace-aware BibTeX parser. Returns entries list of dicts."""
    entries, i, n = [], 0, len(text)
    while i < n:
        m = re.search(r"@\s*([A-Za-z]+)\s*([{(])", text[i:])
        if not m:
            break
        start = i + m.start()
        etype = m.group(1).lower()
        opener = m.group(2)
        closer = "}" if opener == "{" else ")"
        j = i + m.end()
        if etype in ("comment", "preamble", "string"):
            depth = 1
            while j < n and depth:
                if text[j] == opener:
                    depth += 1
                elif text[j] == closer:
                    depth -= 1
                j += 1
            i = j
            continue
        depth, k = 1, j
        while k < n and depth:
            if text[k] == opener:
                depth += 1
            elif text[k] == closer:
                depth -= 1
            k += 1
        body = text[j : k - 1] if depth == 0 else text[j:]
        i = k
        comma = body.find(",")
        if comma < 0:
            continue
        key = body[:comma].strip()
        rest, fields, pos = body[comma + 1 :], {}, 0
        while pos < len(rest):
            fm = re.search(r"([A-Za-z][A-Za-z0-9_\-]*)\s*=\s*", rest[pos:])
            if not fm:
                break
            fname = fm.group(1).lower()
            vstart = pos + fm.end()
            if vstart < len(rest) and rest[vstart] == "{":
                d, q = 1, vstart + 1
                while q < len(rest) and d:
                    if rest[q] == "{":
                        d += 1
                    elif rest[q] == "}":
                        d -= 1
                    q += 1
                val, pos = rest[vstart + 1 : q - 1], q
            elif vstart < len(rest) and rest[vstart] == '"':
                q = rest.find('"', vstart + 1)
                if q < 0:
                    q = len(rest)
                val, pos = rest[vstart + 1 : q], q + 1
            else:
                q = rest.find(",", vstart)
                if q < 0:
                    q = len(rest)
                val, pos = rest[vstart:q], q
            fields[fname] = re.sub(r"\s+", " ", val).strip()
            nxt = rest.find(",", pos)
            pos = len(rest) if nxt < 0 else nxt + 1
        entries.append(
            {
                "key": key,
                "type": etype,
                "fields": fields,
                "file": filelabel,
                "line": line_of(text, start),
            }
        )
    return entries


REQUIRED = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "conference": ["author", "title", "booktitle", "year"],
    "book": ["title", "publisher", "year"],
    "incollection": ["author", "title", "booktitle", "publisher", "year"],
    "inbook": ["title", "publisher", "year"],
    "phdthesis": ["author", "title", "school", "year"],
    "mastersthesis": ["author", "title", "school", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "manual": ["title"],
    "online": ["title", "url"],
    "misc": ["title"],
    "unpublished": ["author", "title"],
    "proceedings": ["title", "year"],
}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7}(v\d+)?)$", re.I)


def entry_issues(e):
    iss, f = [], e["fields"]
    for req in REQUIRED.get(e["type"], ["title"]):
        if not f.get(req):
            iss.append(f"missing field: {req}")
    if e["type"] == "book" and not (f.get("author") or f.get("editor")):
        iss.append("missing field: author/editor")
    yr = (f.get("year") or "").strip()
    if yr:
        if not re.fullmatch(r"\d{4}[a-z]?", yr):
            iss.append(f"malformed year: {yr!r}")
        elif int(yr[:4]) > NOW_YEAR:
            iss.append(f"future year: {yr}")
        elif int(yr[:4]) < 1900:
            iss.append(f"suspicious year: {yr}")
    auth = f.get("author", "")
    if re.search(r"\bet\.?\s*al\.?\b", auth, re.I):
        iss.append("author truncated with 'et al.' in .bib (use 'and others')")
    title = f.get("title", "")
    if title and re.search(r"\b(TODO|TBD|XXX|placeholder)\b", title, re.I):
        iss.append("placeholder title")
    doi = (f.get("doi") or "").strip()
    if doi:
        if doi.lower().startswith(("http://", "https://")):
            iss.append("doi stored as URL (store bare 10.xxxx/...; style adds https://doi.org/)")
        elif doi.lower().startswith("doi:"):
            iss.append("doi has 'doi:' prefix (store bare DOI)")
        elif "dx.doi.org" in doi.lower():
            iss.append("doi uses dx.doi.org (use doi.org)")
        elif not DOI_RE.match(doi.rstrip(".")):
            iss.append(f"malformed doi syntax: {doi!r}")
        if doi.endswith(".") and DOI_RE.match(doi.rstrip(".")):
            iss.append("trailing period after DOI")
        if " " in doi:
            iss.append("doi contains whitespace")
        if re.search(r"10\.xxxx", doi, re.I):
            iss.append("placeholder DOI")
    ep = (f.get("eprint") or "").strip()
    if ep and (f.get("archiveprefix", "").lower() in ("", "arxiv")) and not ARXIV_RE.match(ep):
        iss.append(f"malformed arXiv eprint: {ep!r}")
    pages = f.get("pages", "")
    if pages and re.fullmatch(r"\d+\s*-\s*\d+", pages):
        iss.append("pages use single hyphen (BibTeX range is '--')")
    return iss


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def resolve_bib(name, root_dir, from_file):
    name = name.strip()
    if not name.endswith(".bib"):
        name += ".bib"
    for base in (Path(from_file).parent, root_dir, ROOT):
        c = base / name
        if c.is_file():
            return c.resolve()
        if c.is_symlink():
            return c  # broken symlink: report as declared path
    # search by basename under root_dir tree (depth-limited fallback)
    return None


def main():
    tex_files = walk_files({".tex"})
    bib_files = walk_files({".bib"})
    # standalone bib audit (realpath-deduped)
    bib_by_real, bib_aliases = {}, defaultdict(list)
    for b in bib_files:
        bib_aliases[str(b.resolve())].append(rel(b))
    for real, _names in bib_aliases.items():
        bib_by_real[real] = parse_bib(Path(real))
    # documentclass roots, realpath-deduped with aliases
    roots, root_aliases = {}, defaultdict(list)
    for t in tex_files:
        if r"\documentclass" in read(t):
            root_aliases[str(t.resolve())].append(rel(t))
    for real in root_aliases:
        roots[real] = Path(real)

    papers, nonpapers = [], []
    bib_use_count = Counter()
    print(f"walked tex={len(tex_files)} bib={len(bib_files)} roots={len(roots)}", flush=True)
    for idx, (real, root_tex) in enumerate(sorted(roots.items()), 1):
        if idx % 25 == 0:
            print(f"  root {idx}/{len(roots)}", flush=True)
        files, missing_inputs = collect_tex(root_tex)
        cites, nocites = {}, {}
        bib_names, styles, bibitems, embedded_bibs = [], [], {}, []
        has_cjk, biblatex_style = False, None
        for fpath, txt in files:
            has_cjk = has_cjk or bool(CJK_RE.search(txt))
            m = re.search(r"\\usepackage(?:\[([^\]]*)\])?\{biblatex\}", txt)
            if m:
                sm = re.search(r"style\s*=\s*([A-Za-z0-9\-]+)", m.group(1) or "")
                biblatex_style = ("biblatex:" + sm.group(1)) if sm else "biblatex"
            for cm in CITE_RE.finditer(txt):
                target = nocites if cm.group(1) == "nocite" else cites
                for k in cm.group(2).split(","):
                    k = k.strip()
                    if k and k not in target:
                        target[k] = {
                            "count": 0,
                            "first": f"{rel(fpath)}:{line_of(txt, cm.start())}",
                        }
                    if k:
                        target[k]["count"] += 1
            for bm in BIBCMD_RE.finditer(txt):
                bib_names += [(x, fpath, line_of(txt, bm.start())) for x in bm.group(1).split(",")]
            for bm in BIBRES_RE.finditer(txt):
                bib_names.append((bm.group(1), fpath, line_of(txt, bm.start())))
            styles += [s.strip() for s in BIBSTYLE_RE.findall(txt)]
            for bm in BIBITEM_RE.finditer(txt):
                bibitems[bm.group(1).strip()] = f"{rel(fpath)}:{line_of(txt, bm.start())}"
            for fm in re.finditer(
                r"\\begin\{filecontents\*?\}\{([^}]*\.bib)\}(.*?)\\end\{filecontents\*?\}",
                txt,
                re.S,
            ):
                embedded_bibs.append((fm.group(1).strip(), fm.group(2), fpath))
        style = biblatex_style or (styles[0] if styles else None)
        if not cites and not nocites and not bib_names and not bibitems:
            joined = "\n".join(t for _, t in files)
            nonpapers.append(
                {
                    "root": rel(root_tex),
                    "aliases": root_aliases[real],
                    "tier": tier_of(root_tex),
                    "reason": "no citations or bibliography",
                    "manuscript_like": bool("\\title" in joined and "\\section" in joined),
                }
            )
            continue
        refs, missing_bibs, used_bibs, seen_entry_locs = {}, [], [], set()
        for name, fromf, ln in bib_names:
            bp = resolve_bib(name, root_tex.parent, fromf)
            if bp is None or not Path(bp).is_file():
                missing_bibs.append(
                    {
                        "declared": name.strip(),
                        "from": f"{rel(fromf)}:{ln}",
                        "broken_symlink": bool(bp and Path(bp).is_symlink()),
                    }
                )
                continue
            used_bibs.append(rel(bp))
            bib_use_count[str(Path(bp).resolve())] += 1
            for e in bib_by_real.get(str(Path(bp).resolve()), parse_bib(bp)):
                loc = (e["key"], e["file"], e["line"])
                if loc not in seen_entry_locs:
                    seen_entry_locs.add(loc)
                    refs.setdefault(e["key"], []).append(e)
        for emb_name, emb_text, emb_from in embedded_bibs:
            used_bibs.append(f"embedded:{emb_name} in {rel(emb_from)}")
            for e in parse_bib_text(emb_text, f"embedded:{emb_name} in {rel(emb_from)}"):
                loc = (e["key"], e["file"], e["line"])
                if loc not in seen_entry_locs:
                    seen_entry_locs.add(loc)
                    refs.setdefault(e["key"], []).append(e)
        ref_keys = set(refs) | set(bibitems)
        lower_ref = {k.lower(): k for k in ref_keys}
        orphan_in, case_mismatch = [], []
        for k, info in sorted(cites.items()):
            if k == "*":
                continue
            if k not in ref_keys:
                if k.lower() in lower_ref:
                    case_mismatch.append(
                        {"cited": k, "in_bib": lower_ref[k.lower()], "first": info["first"]}
                    )
                else:
                    orphan_in.append({"key": k, "count": info["count"], "first": info["first"]})
        cited_set = set(cites) | set(nocites)
        nocite_star = "*" in nocites
        orphan_refs = (
            []
            if nocite_star
            else sorted(
                k
                for k in ref_keys
                if k not in cited_set and k.lower() not in {c.lower() for c in cited_set}
            )
        )
        orphan_bibitems = [k for k in orphan_refs if k in bibitems]
        uncited_bib_entries = [k for k in orphan_refs if k not in bibitems]
        cited_entries = [refs[k][0] for k in sorted(set(cites) & set(refs))]
        field_issues, no_identifier, old_sources, recent = [], [], 0, 0
        for e in cited_entries:
            for iss in entry_issues(e):
                field_issues.append(
                    {"key": e["key"], "issue": iss, "at": f"{e['file']}:{e['line']}"}
                )
            f = e["fields"]
            allf = " ".join(f.values())
            has_id = bool(
                f.get("doi")
                or f.get("eprint")
                or re.search(r"arxiv\.org/abs/\d{4}\.\d{4,5}", allf, re.I)
                or re.search(r"\b10\.\d{4,9}/\S+", allf)
                or re.search(r"arXiv:\s*\d{4}\.\d{4,5}", allf, re.I)
            )
            if (
                e["type"] in ("article", "inproceedings", "conference", "incollection")
                and not has_id
            ):
                no_identifier.append(e["key"])
            yr = (f.get("year") or "")[:4]
            if yr.isdigit():
                if int(yr) < NOW_YEAR - 10:
                    old_sources += 1
                if int(yr) >= NOW_YEAR - 5:
                    recent += 1
        dup_keys = [
            {"key": k, "at": [f"{e['file']}:{e['line']}" for e in v]}
            for k, v in sorted(refs.items())
            if len(v) > 1
        ]
        seen_title, dup_titles, seen_doi, dup_dois = {}, [], {}, []
        for e in cited_entries:
            nt = norm_title(e["fields"].get("title"))
            if nt and nt in seen_title and seen_title[nt] != e["key"]:
                dup_titles.append(
                    {"keys": [seen_title[nt], e["key"]], "title": e["fields"].get("title", "")[:90]}
                )
            if nt:
                seen_title.setdefault(nt, e["key"])
            d = (e["fields"].get("doi") or "").lower().strip().rstrip(".")
            d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
            if d and d in seen_doi and seen_doi[d] != e["key"]:
                dup_dois.append({"keys": [seen_doi[d], e["key"]], "doi": d})
            if d:
                seen_doi.setdefault(d, e["key"])
        # stale .bbl + build-log corroboration (sibling of root)
        stem = root_tex.with_suffix("")
        bbl_keys, blg, log_undef = None, {}, []
        if Path(str(stem) + ".bbl").is_file():
            bbl_keys = sorted(set(BIBITEM_RE.findall(read(str(stem) + ".bbl"))))
        if Path(str(stem) + ".blg").is_file():
            bt = read(str(stem) + ".blg")
            blg = {
                "warnings": re.findall(r"Warning--.*", bt)[:20],
                "missing_entries": re.findall(
                    r"I didn't find a database entry for \"([^\"]+)\"", bt
                ),
                "errors": len(re.findall(r"^I was expecting|error", bt, re.M | re.I)),
            }
        if Path(str(stem) + ".log").is_file():
            log_undef = sorted(
                set(re.findall(r"[Cc]itation [`']([^`']+)'.*undefined", read(str(stem) + ".log")))
            )
        profile = hashlib.sha256(
            json.dumps([sorted(cites), sorted(ref_keys), style], ensure_ascii=False).encode()
        ).hexdigest()[:12]
        papers.append(
            {
                "root": rel(root_tex),
                "aliases": root_aliases[real],
                "tier": tier_of(root_tex),
                "style": style,
                "n_files": len(files),
                "missing_inputs": missing_inputs,
                "cite_instances": sum(v["count"] for v in cites.values()),
                "unique_cited": len(cites),
                "ref_entries": len(ref_keys),
                "bibitems_inline": len(bibitems),
                "used_bibs": sorted(set(used_bibs)),
                "missing_bibs": missing_bibs,
                "no_bibliography": bool(cites and not bib_names and not bibitems),
                "orphan_intext": orphan_in,
                "case_mismatch": case_mismatch,
                "orphan_refs": orphan_refs,
                "orphan_bibitems": orphan_bibitems,
                "uncited_bib_entries": uncited_bib_entries,
                "nocite_star": nocite_star,
                "dup_keys": dup_keys,
                "dup_titles": dup_titles,
                "dup_dois": dup_dois,
                "field_issues": field_issues,
                "no_doi_or_arxiv": no_identifier,
                "old_sources_gt10y": old_sources,
                "recent5_pct": round(100 * recent / len(cited_entries), 1)
                if cited_entries
                else None,
                "bbl": (
                    {
                        "bbl_keys": len(bbl_keys),
                        "cited_not_in_bbl": sorted(set(cites) - set(bbl_keys or [])),
                        "in_bbl_not_cited": sorted(set(bbl_keys or []) - set(cites)),
                    }
                    if bbl_keys is not None
                    else None
                ),
                "blg": blg,
                "log_undefined_citations": log_undef,
                "has_cjk": has_cjk,
                "citation_profile": profile,
            }
        )

    shared_bibs = {r for r, c in bib_use_count.items() if c > 1}
    bib_audit = []
    for real, entries in sorted(bib_by_real.items()):
        kc = Counter(e["key"] for e in entries)
        lc = Counter(e["key"].lower() for e in entries)
        all_issues = []
        for e in entries:
            for iss in entry_issues(e):
                all_issues.append({"key": e["key"], "issue": iss, "at": f"{e['file']}:{e['line']}"})
        bib_audit.append(
            {
                "bib": rel(real),
                "aliases": bib_aliases[real],
                "entries": len(entries),
                "used_by_papers": bib_use_count.get(real, 0),
                "shared": real in shared_bibs,
                "dup_keys": sorted(k for k, v in kc.items() if v > 1),
                "case_variant_keys": sorted(
                    k for k, v in lc.items() if v > 1 and kc.get(k, 0) <= 1
                ),
                "issue_count": len(all_issues),
                "issues": all_issues,
                "types": dict(Counter(e["type"] for e in entries)),
            }
        )
    for p in papers:
        p["uses_shared_bib"] = any(str((ROOT / b).resolve()) in shared_bibs for b in p["used_bibs"])
    result = {
        "root": str(ROOT),
        "tex_files": len(tex_files),
        "documentclass_roots_raw": sum(len(v) for v in root_aliases.values()),
        "documentclass_roots_deduped": len(roots),
        "papers": papers,
        "nonpapers": nonpapers,
        "bib_audit": bib_audit,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(
        f"papers={len(papers)} nonpapers={len(nonpapers)} "
        f"bib_files_deduped={len(bib_audit)} -> {OUT_JSON}"
    )


if __name__ == "__main__":
    main()
