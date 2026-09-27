import json, re, numpy as np
import os; S=os.path.dirname(os.path.abspath(__file__))+"/"
R="/Users/arvind/Developer/agentic_repos/tinker-rl-lab/platform_hybrid/experiments/results/"
OUT="/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/PES_Phase2_Third_Review_2026-09-24/thesis/figures/fig_length_bias.tex"
orig=open(S+"fig_length_bias.orig.tex").read()
header=orig[:orig.index("\\begin{document}")]
# panel (a) traces: reuse the logged coordinates verbatim from the original source
a_plots=re.findall(r"(\\addplot\[[^\]]*mark repeat[^\]]*\] coordinates \{[^}]*\};)\n  \\addlegendentry\{([^}]*\}?[^}]*)\}", orig.split("\\begin{groupplot}")[0])
a_block=orig.split("\\begin{axis}[")[1].split("\\end{axis}")[0]
a_lines=[l for l in a_block.split("\n") if l.strip().startswith(("\\addplot","\\addlegendentry","\\addlegendimage"))]

def P(xs,ys): return " ".join(f"({x:.3f},{y:.4f})" for x,y in zip(xs,ys))

def arm(f,algo):
    runs=[r for r in json.load(open(R+f))['runs'] if r['algo']==algo]
    M=np.array([[s['mean_reward'] for s in r['step_log']] for r in runs]); n=M.shape[1]
    x=(np.arange(n)+1)/n*100
    peaks=[]; flags=0
    for row in M:
        k=int(np.argmax(row)); pos=x[k]; flag=pos<65 and row[-10:].mean()<0.9*row.max()
        flags+=flag; peaks.append((pos,row.max(),flag))
    return x,M,peaks,flags,len(runs)

def band(x,M,col):
    up=P(x,M.max(0)); lo=P(x[::-1],M.min(0)[::-1])
    return f"  \\addplot[draw=none, fill={col}, forget plot] coordinates {{{up} {lo}}} \\closedcycle;\n"

out=[header,"\\begin{document}\n\\parbox{14.6cm}{\\centering\n",
"% ---------------------------------------------------------------------------\n",
"% fig_length_bias --- peak-then-decay rule on the roster (a) and on the\n",
"% controlled sixteen-run comparison (b)-(d).  Data:\n",
"%  (a) platform_hybrid/experiments/master_results.json (reward_trace), coordinates as logged\n",
"%  (b) platform_hybrid/experiments/results/drgrpo_vs_grpo.json (5 seeds/arm)\n",
"%  (c,d) platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json (3 seeds/arm)\n",
"% Bands = min--max across seeds; line = seed mean. Flag = peak before 65% of\n",
"% logged steps AND last-10 mean reward < 0.90 x peak, computed per seed by\n",
"% figures/generators/gen_length_bias.py from the per-step logs.\n",
"% ---------------------------------------------------------------------------\n",
"\\begin{tikzpicture}\n\\begin{axis}[\n  width=14.2cm, height=5.2cm,\n  xmin=0, xmax=100, ymin=0, ymax=1.0,\n",
"  xlabel={training progress (\\% of logged steps)}, ylabel={mean reward},\n",
"  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n",
"  title style={font=\\footnotesize}, title={(a) roster GRPO runs flagged by the peak-then-decay rule (single seed each)},\n",
"  grid=major, grid style={draw=black!8}, axis line style={black!60},\n",
"  legend style={at={(0.5,-0.30)}, anchor=north, font=\\footnotesize, draw=black!25,\n",
"                inner sep=2pt, row sep=0.5pt, legend columns=2, column sep=6pt, legend cell align=left},\n]\n",
"  \\draw[black!55, densely dashed, thick] (axis cs:65,0) -- (axis cs:65,1.0);\n",
"  \\node[anchor=north west, font=\\footnotesize, text=black!70, fill=white, inner sep=1pt] at (axis cs:65.5,0.98) {65\\% of training};\n"]
for l in a_lines:
    if "65\\% of training" in l or "black!55, densely dashed, thick" in l: continue
    out.append("  "+l.strip()+"\n")
out.append("\\end{axis}\n\\end{tikzpicture}\\par\\vspace{1ex}\n")

xb,Mg,pg,fg,ng=arm('drgrpo_vs_grpo.json','grpo'); _,Md,pd,fd,nd=arm('drgrpo_vs_grpo.json','dr_grpo')
xc,Cg,cpg,cfg,cng=arm('drgrpo_gsm8k_cot_full.json','grpo'); _,Cd,cpd,cfd,cnd=arm('drgrpo_gsm8k_cot_full.json','dr_grpo')
print('flags arith',fg,fd,'gsm8k',cfg,cfd)
out.append("\\begin{tikzpicture}\n\\begin{groupplot}[\n  group style={group size=3 by 1, horizontal sep=10pt, y descriptions at=edge left},\n"
"  width=5.55cm, height=5.0cm, xmin=0, xmax=100, xtick={0,50,100},\n"
"  xlabel={\\% of logged steps}, ylabel={mean reward},\n"
"  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
"  title style={font=\\footnotesize, align=center},\n"
"  grid=major, grid style={draw=black!8}, axis line style={black!60}, ymin=0, ymax=1.05,\n]\n")
# (b)
out.append("\\nextgroupplot[title={(b) arithmetic, 0.5B\\\\5 seeds/arm; flag %d/%d, %d/%d}]\n"%(fg,ng,fd,nd))
out.append(band(xb,Mg,"pesblue!18")); out.append(band(xb,Md,"bad!14"))
out.append(f"  \\addplot[pesblue, thick] coordinates {{{P(xb,Mg.mean(0))}}};\n")
out.append(f"  \\addplot[bad, thick, densely dashed] coordinates {{{P(xb,Md.mean(0))}}};\n")
out.append("  \\draw[black!55, densely dashed] (axis cs:65,0) -- (axis cs:65,1.05);\n")
out.append("  \\node[font=\\footnotesize, align=left, anchor=south east] at (axis cs:97,0.08) {solid: GRPO\\\\dashed: Dr.\\ GRPO\\\\band: min--max};\n")
for lab,C,pk,fl,nn,col,sty in [("(c) GSM8K-CoT, GRPO",Cg,cpg,cfg,cng,"pesblue","solid"),("(d) GSM8K-CoT, Dr.\\ GRPO",Cd,cpd,cfd,cnd,"bad","densely dashed")]:
    out.append("\\nextgroupplot[title={%s\\\\3 seeds; flag %d/%d}]\n"%(lab,fl,nn))
    out.append(band(xc,C,col+"!16"))
    out.append(f"  \\addplot[{col}, thick, {sty}] coordinates {{{P(xc,C.mean(0))}}};\n")
    out.append("  \\draw[black!55, densely dashed] (axis cs:65,0) -- (axis cs:65,1.05);\n")
    pts=" ".join(f"({p:.3f},{v:.4f})" for p,v,f in pk if f); npt=" ".join(f"({p:.3f},{v:.4f})" for p,v,f in pk if not f)
    if pts: out.append(f"  \\addplot[only marks, mark=star, mark size=3pt, draw=black, fill=warn] coordinates {{{pts}}};\n")
    if npt: out.append(f"  \\addplot[only marks, mark=o, mark size=2.5pt, draw=black] coordinates {{{npt}}};\n")
    out.append("  \\addplot[black!60, densely dotted, thick] coordinates {(0,%.4f) (100,%.4f)};\n"%(C[:,-10:].mean(),C[:,-10:].mean()))
    out.append("  \\node[font=\\footnotesize, anchor=south east, fill=white, inner sep=1pt] at (axis cs:98,0.90) {last-10 mean %.2f};\n"%(C[:,-10:].mean()))
out.append("\\end{groupplot}\n\\end{tikzpicture}\n}\n\\end{document}\n")
open(OUT,"w").write("".join(out))
