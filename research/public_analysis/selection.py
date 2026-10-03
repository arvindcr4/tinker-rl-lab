"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

from collections import Counter
import math

MAX_SELECTED = 64


SCREEN_N = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection_from_screen(rows):
    require(len(rows)==SCREEN_N, 'Incomplete screen: all4096 required before selection')
    require([r['original_prompt_rank'] for r in rows]==list(range(640,4736)), 'Screen order/ranks changed')
    require([r['prompt_index'] for r in rows]==list(range(SCREEN_N)), 'Screen indices changed')
    require(len({r['prompt_id'] for r in rows})==SCREEN_N, 'Duplicate screen identity')
    eligible=[];counts=Counter();per_question=[]
    for r in rows:
        cs=r['completions'];require(len(cs)==8, 'Screen must have exactly eight draws')
        reasons=[]
        if any(c['parse_status']!='ok' for c in cs):reasons.append('any_parse_failure')
        if any(c['cap_hit'] for c in cs):reasons.append('any_cap')
        if any(c['parse_status']=='ok' and c['parsed_value']==r['gold_value'] for c in cs):reasons.append('any_valid_correct')
        if not reasons:eligible.append(r['prompt_index'])
        counts.update(reasons)
        per_question.append({'prompt_id':r['prompt_id'],'screen_index':r['prompt_index'],'eligible':not reasons,'ineligibility_reasons':reasons})
    return {'eligible_screen_indices':eligible,'selected_screen_indices':eligible[:MAX_SELECTED],
            'eligible_prompt_ids':[rows[i]['prompt_id'] for i in eligible],
            'selected_prompt_ids':[rows[i]['prompt_id'] for i in eligible[:MAX_SELECTED]],
            'screened_questions':SCREEN_N,'eligible_questions':len(eligible),'selected_questions':min(len(eligible),MAX_SELECTED),
            'ineligibility_reason_counts':dict(counts),'per_screen_question':per_question}


def validate_selection(selection, rows):
    expected=selection_from_screen(rows)
    for k in ('eligible_screen_indices','selected_screen_indices','eligible_prompt_ids','selected_prompt_ids','screened_questions'):
        require(selection.get(k)==expected[k], 'Selection differs from complete screen: '+k)
    return expected


def interval(k,n):
    require(type(k)is int and type(n)is int and 0<=k<=n, 'Invalid binomial counts')
    if n==0:return {'events':k,'denominator':n,'fraction':None,'wilson_95':None,'zero_events_exact_upper_one_sided_95':None,'status':'UNDEFINED_NO_ELIGIBLE_QUESTIONS'}
    z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'events':k,'denominator':n,'fraction':p,'wilson_95':[max(0.,center-half),min(1.,center+half)],
            'zero_events_exact_upper_one_sided_95':-math.expm1(math.log(.05)/n) if k==0 else None,'status':'DEFINED'}


def primary_success(c,gold):
    return c['parse_status']=='ok' and not c['cap_hit'] and c['parsed_value']==gold


def quality(rows):
    cs=[c for r in rows for c in r['completions']];n=len(cs)
    bad=sum(c['parse_status']!='ok' for c in cs);caps=sum(c['cap_hit'] for c in cs)
    return {'questions':len(rows),'completions':n,'parse_failures':bad,'cap_hits':caps,
            'parse_failure_rate':bad/n if n else None,'cap_hit_rate':caps/n if n else None,
            'passes_smoke_quality_gate':n>0 and bad/n<=.05 and caps/n<=.05}
