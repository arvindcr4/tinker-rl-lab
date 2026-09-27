import json, numpy as np
from scipy.stats import spearmanr
R="/Users/arvind/Developer/agentic_repos/tinker-rl-lab/platform_hybrid/experiments/results/"
rows=[]
for f,task in [('drgrpo_vs_grpo.json','arith'),('drgrpo_gsm8k_cot_full.json','gsm8k')]:
    for r in json.load(open(R+f))['runs']:
        sl=r['step_log']; rew=np.array([s['mean_reward'] for s in sl]); ln=np.array([s['mean_comp_len'] for s in sl]); st=np.arange(len(sl))
        n=len(rew); pk=int(np.argmax(rew)); peakpos=(pk+1)/n
        t_last=rew[-1]; t_l10=rew[-10:].mean()
        f_last= peakpos<0.65 and t_last<0.9*rew.max()
        f_l10 = peakpos<0.65 and t_l10<0.9*rew.max()
        rl=spearmanr(st,ln)[0]; rr=spearmanr(st,rew)[0]
        sll=np.polyfit(st,ln,1)[0]; slr=np.polyfit(st,rew,1)[0]
        lb=((sll>0) or (rl>0)) and ((slr<=0) or (rr<=0))
        rows.append((task,r['algo'],r['seed'],n,round(rew.max(),4),pk,round(peakpos,3),round(t_last,4),round(t_l10,4),f_last,f_l10,round(rl,3),lb))
        print(rows[-1])
import collections
for key in ['last','l10']:
    c=collections.Counter((t,a) for t,a,*x in rows if (x[7] if key=='last' else x[8]))
    print(key,dict(c), sum(c.values()))
print('lengthbias flags', sum(r[-1] for r in rows))
