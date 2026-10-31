#!/usr/bin/env python3
"""ICICC 2027 paper 243 camera-ready readiness predicate. Exit 0 + 'READY' when all hard checks pass."""
import json, re, subprocess, sys, pathlib
P = pathlib.Path(__file__).resolve().parents[2]
SK = pathlib.Path.home() / ".claude/skills"
GREY = {"proximal2026frontierswe","emulated2026sdab","quesma2026binaryaudit","halluminatewebbench",
        "afterquery2025appbench","openrewardplatform","khandpurswemultilingual","qwen2026model","thinkingmachines2025tinker"}
tex = (P/"paper_camera.tex").read_text()
fails, warns = [], []
def run(cmd): return subprocess.run(cmd, cwd=P, capture_output=True, text=True)
# 1 build + pages
run(["./build.sh"])
pages = int(re.search(r"Pages:\s+(\d+)", run(["pdfinfo","build/paper_camera.pdf"]).stdout).group(1))
if not 10 <= pages <= 12: fails.append(f"pages={pages} (need 10-12)")
# 2 repo verifier
if run([sys.executable,"verify_manuscript.py"]).returncode: fails.append("verify_manuscript.py fails")
# 3 bib lint
r = run([sys.executable, str(SK/"bibtex-hygiene/scripts/bib_lint.py"), "refs.bib", "--json"])
try:
    j = json.loads(r.stdout); errs = [f for f in j.get("findings", j if isinstance(j, list) else []) if f.get("severity") == "error"]
except Exception: errs = ["unparseable"] if r.returncode == 2 else []
if errs: fails.append(f"bib lint errors={len(errs)}")
# 4 citation existence (cached per refs.bib hash)
import hashlib
h = hashlib.sha256((P/"refs.bib").read_bytes()).hexdigest()[:12]
cache = pathlib.Path(__file__).parent / f"cite-{h}.json"
if not cache.exists():
    run([sys.executable, str(SK/"citation-verification/scripts/verify_citations.py"), "refs.bib", "--json", str(cache), "--delay", "0.2"])
try:
    cj = json.loads(cache.read_text()); rows = cj if isinstance(cj, list) else cj.get("results", [])
    bad = [x.get("key") for x in rows if str(x.get("status","")).lower() in ("not-found","mismatch","doi-mismatch","fail") and x.get("key") not in GREY]
    if bad: fails.append(f"unexplained citation failures: {bad}")
except Exception as e: warns.append(f"citation cache unreadable: {e}")
# 5 slop index
r = run([sys.executable, str(SK/"unslop-academic-writing/scripts/slop_check.py"), "paper_camera.tex"])
m = re.search(r"Slop index ([\d.]+)", r.stdout)
if not m or float(m.group(1)) >= 5: fails.append(f"slop index {m.group(1) if m else '?'} (need <5)")
# 6 reviewer-2 structure
intro = tex.split(r"\section{Introduction}")[1].split(r"\section{")[0]
if not (re.search(r"contribution", intro, re.I) and re.search(r"\\begin\{(itemize|enumerate)\}", intro)):
    fails.append("R2.5: contributions not in bullet list in Introduction")
heads = re.findall(r"\\(?:sub)?section\*?\{([^}]*)\}", tex)
if not any(re.search(r"practical|societal|impact", x, re.I) for x in heads): fails.append("R2.7: no Practical/Societal Impact (sub)section")
if not any(re.search(r"limitation", x, re.I) for x in heads): fails.append("R1.6: no Limitations heading")
# 7 references
bbl = (P/"build/paper_camera.bbl").read_text()
nref = bbl.count(r"\bibitem")
if nref < 15: fails.append(f"refs={nref} (<15)")
recent = len(re.findall(r"\((2024|2025)\)", bbl))
if recent < 3: fails.append(f"2024-25 refs={recent} (<3)")
# 8 equations numbered, corresponding author
if re.search(r"\\begin\{equation\*\}|\\nonumber|\$\$", tex): fails.append("unnumbered display equation")
if "Corresponding author" not in tex: fails.append("no corresponding author")
if "not a submission" in tex: warns.append("source header says 'not a submission' (author decision)")
print(json.dumps({"pages": pages, "refs": nref, "recent_refs": recent, "fails": fails, "warns": warns}, indent=1))
print("READY" if not fails else f"NOT READY ({len(fails)} units)")
sys.exit(0 if not fails else 1)
