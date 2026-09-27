import fitz, sys, os, statistics
FIG="/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/PES_Phase2_Third_Review_2026-09-24/thesis/figures"
import os; OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"png")
os.makedirs(OUT, exist_ok=True)
TW=0.94*(8.27-2.5)*72
args=sys.argv[1:]
render='--png' in args; verbose='-v' in args
names=[a for a in args if not a.startswith('-')] or sorted(f[:-4] for f in os.listdir(FIG) if f.endswith('.pdf') and f.startswith('fig_'))
for n in names:
    d=fitz.open(f"{FIG}/{n}.pdf"); p=d[0]; w=p.rect.width; h=p.rect.height
    sizes=[]; small=[]
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines",[]):
            for sp in l["spans"]:
                if sp["text"].strip():
                    ps=sp["size"]*TW/w
                    sizes.append(ps)
                    if ps<6: small.append((round(ps,1),sp["text"][:30]))
    longs=[x for x,t in small if len(t.strip())>3]
    print(f"[min_len>3={min([sp for sp in sizes if True]) if not longs else min(longs):.1f} n_long<6={len(longs)}]", end=' ')
    print(f"{n:28s} {w:5.0f}x{h:4.0f} scale={TW/w:.2f} min={min(sizes):.1f} med={statistics.median(sizes):.1f} n<6={len(small)}/{len(sizes)}")
    if verbose: print('   ',small[:20])
    if render:
        pix=p.get_pixmap(dpi=int(170*TW/w)); pix.save(f"{OUT}/{n}.png")
