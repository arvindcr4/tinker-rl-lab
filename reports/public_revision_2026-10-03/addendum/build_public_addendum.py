#!/usr/bin/env python3
"""Build the public scientific addendum from the adjacent public Markdown.

Requires Python 3, Pandoc and Tectonic on PATH. No original source or external
execution records are read by this builder. Configure TeX caches through the
standard XDG_CACHE_HOME and XDG_CONFIG_HOME environment variables if needed.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
STEM = 'C1_Research_Addendum_2026-10-03_PUBLIC'
BUILD = HERE / 'build'

PREAMBLE = r'''\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,left=0.95in,right=0.95in,top=0.85in,bottom=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{newtxtext,newtxmath}
\usepackage{array,longtable,booktabs,calc,graphicx,xcolor,setspace}
\usepackage{seqsplit,xurl,needspace}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={C1 frozen policy diagnostic and reference review: PUBLIC revision},pdfauthor={Arvind C R},pdfsubject={Scientific-only public research addendum dated 2026-10-03}}
\usepackage{newunicodechar}
\newunicodechar{×}{\ensuremath{\times}}
\providecommand{\real}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\DeclareRobustCommand{\longtok}[1]{\texttt{\seqsplit{#1}}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.18}
\setstretch{1.16}
\widowpenalty=10000
\clubpenalty=10000
\sloppy
\begin{document}
{\Large\bfseries C1 frozen policy diagnostic and reference review\par}
\vspace{0.5em}
{\large PUBLIC revision\par}
{\normalsize Scientific-only derivative of the research addendum\par}
{\normalsize Arvind C R\quad 3 October 2026\par}
\vspace{1em}
'''

def format_tables(tex):
    pattern = re.compile(r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}.*?\\end\{longtable\}',re.S)
    def fix(match):
        table = match.group(0)
        spec = match.group(1)
        count = spec.count('p{')
        if count:
            adjusted = re.sub(r'\\(?:linewidth|columnwidth)\s*-\s*\d+\\tabcolsep', lambda _: '\\linewidth - '+str(2*count)+'\\tabcolsep', spec)
            table = table.replace(spec, adjusted, 1)
        table = table.replace(r'>{\raggedright\arraybackslash}p{',r'>{\raggedright\arraybackslash\hspace{0pt}}p{')
        if count >= 4:
            table = '\\begingroup\\small\\setlength{\\tabcolsep}{3pt}\n'+table+'\n\\endgroup'
        return '\\Needspace{12\\baselineskip}\n'+table
    return pattern.sub(fix, tex)

def main():
    for name in ('pandoc','tectonic'):
        if not shutil.which(name):
            raise SystemExit(f'{name} must be on PATH')
    BUILD.mkdir(exist_ok=True)
    md = (HERE/(STEM+'.md')).read_text(encoding='utf-8')
    md = md.split('3 October 2026', 1)[1].strip()
    md = md.replace('## Frozen endpoint', r'\Needspace{18\baselineskip}'+'\n\n## Frozen endpoint')
    fragment = subprocess.run(['pandoc','-f','markdown','-t','latex','--top-level-division=section','--shift-heading-level-by=-1'],input=md,text=True,capture_output=True,check=True).stdout
    fragment = re.sub(r'\\texttt\{([A-Za-z0-9._/-]{29,})\}',lambda m: r'\longtok{'+m.group(1)+'}',fragment)
    fragment = re.sub(r'\\texttt\{([^{}\s]*\\_[^{}\s]*)\}',lambda m: r'\longtok{'+m.group(1)+'}',fragment)
    tex = PREAMBLE + format_tables(fragment) + '\n\\end{document}\n'
    source = HERE/(STEM+'.tex')
    source.write_text(tex,encoding='utf-8')
    result = subprocess.run(['tectonic','-X','compile',source.name,'--outdir',str(BUILD),'--keep-logs'],cwd=HERE,env=os.environ.copy(),text=True,capture_output=True)
    (BUILD/'compile_output.log').write_text(result.stdout+result.stderr)
    print((result.stdout+result.stderr)[-5000:])
    result.check_returncode()
    shutil.copy2(BUILD/(STEM+'.pdf'),HERE/(STEM+'.pdf'))
    print(HERE/(STEM+'.pdf'))

if __name__ == '__main__':
    main()
