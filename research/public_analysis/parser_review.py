"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import re
import secrets

def require(ok, message):
    if not ok: raise ValueError(message)


def pairs(items):
    result = {}
    for k,v in items:
        require(k not in result, 'duplicate JSON key: '+k); result[k]=v
    return result


def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def now(): return datetime.now(timezone.utc).isoformat()


def utc(text):
    x=datetime.fromisoformat(text.replace('Z','+00:00'))
    require(x.tzinfo is not None and x.utcoffset().total_seconds()==0,'timestamp must explicitly be UTC')
    return x.timestamp()


def packet_rows(records):
    rows=[];mapping={}
    for rec in records:
        if rec['v2']==rec['v3']:continue
        case_id=secrets.token_hex(16);mapping[case_id]=rec['request']['logical_request_id'];rows.append({'case_id':case_id,'raw_completion_text':rec['raw_completion_text']})
    secrets.SystemRandom().shuffle(rows);return rows,mapping


def validate_packet(packet,mapping,records):
    changed={r['request']['logical_request_id']:r for r in records if r['v2']!=r['v3']}
    require(set(mapping.values())==set(changed) and len(mapping)==len(changed),'packet mapping not all changed requests')
    seen=set()
    for row in packet:
        require(set(row)=={'case_id','raw_completion_text'},'blind packet has extra/missing fields')
        case=row['case_id'];require(case in mapping and case not in seen,'packet case missing/duplicated');seen.add(case)
        require(re.fullmatch(r'[0-9a-f]{32}',case) is not None,'packet case ID not opaque format')
        require(row['raw_completion_text']==changed[mapping[case]]['raw_completion_text'],'blind packet text/mapping differs from immutable raw receipt')
    require(seen==set(mapping),'blind packet incomplete')


def canonical_value(value):
    require(isinstance(value,str) and re.fullmatch(r'-?\d+(?:/[1-9]\d*)?',value) is not None,'review value must be exact canonical rational')
    require(str(Fraction(value))==value,'noncanonical review rational');return value


def validate_review(review,packet_sha,case_ids):
    require(isinstance(review['reviewer_id'],str) and bool(review['reviewer_id'].strip()),'reviewer identity missing')
    require(review['packet_sha256']==packet_sha,'review packet binding')
    require(review['independent_output_only_review'] is True,'independence attestation missing')
    require(utc(review['completed_at_utc'])<=utc(now()),'future review time')
    found={}
    for item in review['cases']:
        case=item['case_id'];require(case in case_ids and case not in found,'unknown/duplicate review case')
        require(item['verdict'] in ('single_numeric','ambiguous','no_single_numeric'),'invalid review verdict')
        if item['verdict']=='single_numeric':canonical_value(item['value'])
        else:require(item['value'] is None,'non-scalar review must not predict scalar')
        require(isinstance(item['rationale'],str) and bool(item['rationale'].strip()),'review rationale missing');found[case]=item
    require(set(found)==set(case_ids),'review incomplete');return found


def wrapper_labels(raw):
    # Descriptive literal evidence only, never inclusion/filtering or parser repair.
    labels=[]
    if re.search(r'(?im)^\s*####\s*\*\*(?:final\s+answer|answer)\s*:\*\*\s*$',raw):labels.append('R1_bold_answer_heading')
    if re.search(r'(?m)^\s*\$\$\s*\n\s*\\boxed[^\n]*\n\s*\$\$\s*$',raw):labels.append('R2_display_box_block')
    if re.search(r'(?m)^\s*####\s*<answer>\s*\n\s*####[^\n]*\n\s*####\s*</answer>\s*$',raw):labels.append('R3_hash_xml_block')
    return labels or ['unclassified_changed_format']


def decide(records,mapping,reviews):
    r1,r2=reviews;require(r1['reviewer_id']!=r2['reviewer_id'],'reviewers must be distinct')
    changes={r['request']['logical_request_id']:r for r in records if r['v2']!=r['v3']}
    require(set(mapping.values())==set(changes) and len(mapping)==len(changes),'changed mapping mismatch')
    annotations=[{x['case_id']:x for x in r['cases']} for r in reviews]
    disagreement=bad=0;full=[]
    for case,rid in mapping.items():
        r=changes[rid];a,b=[x[case] for x in annotations];agreed=(a['verdict'],a['value'])==(b['verdict'],b['value'])
        disagreement+=int(not agreed)
        accepted_bad=r['v3']['status']=='ok' and (not agreed or a['verdict']!='single_numeric' or a['value']!=r['v3']['value'])
        bad+=int(accepted_bad)
        full.append({'case_id':case,**r,'review1':a,'review2':b,'reviewers_agree':agreed,'accepted_ambiguous_or_wrong':accepted_bad})
    valid=[r for r in records if r['v2']['status']=='ok'];regress=[r for r in valid if r['v2']!=r['v3']]
    newly=[r for r in records if r['v2']['status']!='ok' and r['v3']['status']=='ok']
    flags={'fresh_v2_objects_preserved':not regress,'changed_extraction_review_passed':not(disagreement or bad),'positive_wrapper_observed_evidence':bool(newly)}
    decision='FAIL' if regress or disagreement or bad else 'NARROW_PASS_ON_OBSERVED_CHANGED_EXTRACTIONS' if newly else 'INCONCLUSIVE_NO_FRESH_POSITIVE_CASES'
    return {'decision':decision,**flags,'output_denominator':len(records),'prompt_denominator':len({r['request']['prompt_id'] for r in records}),'v2_status_counts':dict(Counter(r['v2']['status'] for r in records)),'v3_status_counts':dict(Counter(r['v3']['status'] for r in records)),'v2_valid_count':len(valid),'v2_valid_objects_preserved':len(valid)-len(regress),'v2_valid_objects_changed':len(regress),'all_changed_count':len(full),'newly_accepted_count':len(newly),'newly_accepted_distinct_prompts':len({r['request']['prompt_id'] for r in newly}),'reviewed_changed_count':len(full),'review1_case_count':len(r1['cases']),'review2_case_count':len(r2['cases']),'review_disagreements_unresolved':disagreement,'accepted_ambiguous_or_wrong':bad,'transition_counts':dict(Counter(r['v2']['status']+'->'+r['v3']['status'] for r in changes.values())),'value_change_count':sum(r['v2']['value']!=r['v3']['value'] for r in changes.values()),'source_change_count':sum(r['v2']['source']!=r['v3']['source'] for r in changes.values()),'reverse_status_count':sum(r['v2']['status']=='ok' and r['v3']['status']!='ok' for r in changes.values()),'newly_accepted_wrapper_counts_nonexclusive':dict(Counter(label for r in newly for label in wrapper_labels(r['raw_completion_text']))),'all_changes':full,'no_task_correctness_or_population_false_positive_claim':True}
