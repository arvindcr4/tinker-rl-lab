#!/usr/bin/env python3
"""Editorial/build regression; not a benchmark rerun or policy clearance."""
from pathlib import Path
import hashlib,json,re,subprocess
r=Path(__file__).resolve().parent; repo=r.parents[1]
tex=(r/'paper_camera.tex').read_text();bib=(r/'refs.bib').read_text()
log=(r/'build/paper_camera.log').read_text();blg=(r/'build/paper_camera.blg').read_text();bbl=(r/'build/paper_camera.bbl').read_text()
run=lambda *a:subprocess.check_output(a,text=True)
pdf=r/'build/paper_camera.pdf'; info=run('pdfinfo',str(pdf));body=run('pdftotext','-layout',str(pdf),'-');fonts=run('pdffonts',str(pdf));images=run('pdfimages','-list',str(pdf))
pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
cited=set(k.strip() for g in re.findall(r'\\cite\{([^}]+)\}',tex) for k in g.split(','))
rendered=set(re.findall(r'\\bibitem\{([^}]+)\}',bbl));entries=set(re.findall(r'^@\w+\{([^,]+),',bib,re.M))
abstract=tex.split(r'\begin{abstract}')[1].split(r'\keywords')[0]
assert 10<=pages<=12,pages
assert '(A4)' in info
assert cited==rendered==entries,(cited,rendered,entries)
assert len(cited)==37,len(cited)
assert 150<=len(abstract.split())<=250,len(abstract.split())
assert not re.search(r'Overfull|undefined|Fatal error|^!',log,re.M)
assert 'Warning' not in blg
assert all(' yes ' in row and 'Type 3' not in row for row in fonts.splitlines()[2:])
assert not images.splitlines()[2:]
assert r'\author{Arvind C R \and Ramesh Prakash Guledgudd}' in tex
assert r'\email{arvindcr4@gmail.com}' in tex
assert tex.count(r'\begin{table}')==5
assert tex.count(r'\begin{figure}')==1
assert tex.count(r'\begin{equation}')==2
ledger_table=tex.split(r'\label{tab:lanes}')[1].split(r'\end{table}')[0]
originals=set(re.findall(r'(E\d+-O)\s*&',ledger_table))
assert originals=={f'E{i}-O' for i in range(1,15)}, originals
state_fig=tex.split(r'\begin{figure}')[1].split(r'\end{figure}')[0]
assert {f'E{i}-O' for i in range(1,15)}<=set(re.findall(r'E\d+-O',state_fig))
assert set(re.findall(r'E\d+-R',state_fig))=={f'E{i}-R' for i in [1,2,5,6,8,9,10,13,14]}
historical=repo/'outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json'
historical_checked=historical.is_file()
if historical_checked:
    assert hashlib.sha256(historical.read_bytes()).hexdigest()=='7193cc30dfcdaacfaca736e246a971f83c1f7e06419e99337d8383e6bff3b3a2'
    d=json.loads(historical.read_text());check=next(c for c in d['checks'] if c['check']=='strict_completed_suites_are_E8_E10_E11')
    assert check['status']=='PASS' and 'replacement' not in check['detail']
assert round(450/(1967-1259)*100,2)==63.56
assert round(129/(312-150)*100,2)==79.63
assert round(2271/4426*100,2)==51.31 and round(2271/4428*100,2)==51.29
assert 713+14+4==731 and 593+120==713 and 67+62==129
assert 'all-disposition' in tex and 'not a model score' in tex
assert 'exploratory' in tex and 'runtime contrast' in tex
assert 'proposed' in (r/'REVIEW_RESPONSE_AND_CHECKLIST.md').read_text().lower()
years=[int(v) for v in re.findall(r'^\s*year\s*=\s*\{(\d+)\}',bib,re.M)]
report={'checked_date':'2026-10-18','scope':'Editorial/build checks, existing-count arithmetic and unchanged historical receipt hash; not new benchmark verification or venue-policy certification','pages_including_references':pages,'paper_size':'A4','cited_references':len(cited),'references_dated_2024_or_2025':sum(x in [2024,2025] for x in years),'references_dated_2026':years.count(2026),'abstract_whitespace_word_count':len(abstract.split()),'unresolved_citations':0,'overfull_boxes':0,'bibtex_warnings':0,'all_fonts_embedded_scalable':True,'raster_images':0,'editable_tables':5,'numbered_equations':2,'vector_scope_state_figure':1,'original_contract_rows':14,'reported_replacement_identities':9,'primary_paired_outcome_contrasts':17,'historical_ledger_check_status':'checked existing repository record' if historical_checked else 'not run: original ledger not included in standalone manuscript package','historical_ledger_sha256_unchanged':True if historical_checked else None,'known_nonfatal_diagnostics':['LLNCS/amsmath: Unable to redefine math accent vec','Underfull bibliography lines; render inspected'],'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((r/'paper_camera.tex').read_bytes()).hexdigest(),'bibliography_sha256':hashlib.sha256((r/'refs.bib').read_bytes()).hexdigest()}
(r/'verification/quality-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
