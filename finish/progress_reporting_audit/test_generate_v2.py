import copy
import unittest
import generate_v1 as v1
import generate_v2 as v2


def tau_fixture():
    ids = [f't{i:03}' for i in range(60)]
    snapshot = {'at':1,'started_ids_in_original_order':ids[:24],'started_unique':24,
                'graded':{k:0.0 for k in ids[:16]},'started_without_grade':ids[16:24],
                'never_started_ids':ids[24:],'full_suite_denominator':97,'subset_denominator':60,
                'prior37_refs_hash_verified':True,'no_overlap_prior37':True,'no_duplicate_attempt_intents':True,
                'full_suite_score':None,'known_failure_attempt_types':{'ContextWindowExceededError':25}}
    episodes = [{'task_id':k,'end_time':'fixture','termination_reason':'user_stop','reward_info':{'reward':0.0}} for k in ids[:16]]
    return snapshot,episodes,ids


class RefreshTests(unittest.TestCase):
    def test_native10_started_not_graded(self):
        m=v2.audit_tau_snapshot(*tau_fixture())
        self.assertEqual(m['coverage'],v1.ratio(16,97))
        self.assertEqual(m['allocation_graded_coverage'],v1.ratio(16,60))
        self.assertEqual(m['started_allocation_coverage'],v1.ratio(24,60))
        self.assertEqual(len(m['started_without_grade']),8)
        self.assertEqual(m['accuracy_or_success'],v1.ratio(0,16))

    def test_failure_events_not_unique_task_zeros(self):
        m=v2.audit_tau_snapshot(*tau_fixture())
        self.assertEqual(m['failure_event_counts_not_task_outcomes']['ContextWindowExceededError'],25)
        self.assertEqual(m['coverage']['numerator'],16)

    def test_snapshot_cannot_override_raw_reward(self):
        s,e,ids=tau_fixture();s['graded'][ids[0]]=1
        with self.assertRaises(v1.EvidenceError):v2.audit_tau_snapshot(s,e,ids)

    def test_inconsistent_partitions(self):
        for field in ('started_without_grade','never_started_ids','started_ids_in_original_order'):
            s,e,ids=tau_fixture();s[field]=[]
            with self.subTest(field=field),self.assertRaises(v1.EvidenceError):v2.audit_tau_snapshot(s,e,ids)

    def test_allocation_not_full_suite_denominator(self):
        s,e,ids=tau_fixture();s['full_suite_denominator']=60
        with self.assertRaises(v1.EvidenceError):v2.audit_tau_snapshot(s,e,ids)

    def test_zero_native_closeout_not_zero_accuracy(self):
        m=v2.infra_zero({'status':'infra','native':0,'model_calls':0},'native','infra','infra only',45)
        self.assertEqual(m['coverage']['fraction'],0)
        self.assertIsNone(m['accuracy_or_success'])
        self.assertFalse(m['complete_named_scope'])

    def test_native_execution_contradicts_infrastructure_only(self):
        with self.assertRaises(v1.EvidenceError):v2.infra_zero({'status':'infra','native':1,'model_calls':0},'native','infra','infra only',45)

    def test_canonical_label_resolution_and_mismatch(self):
        rows=[];entries=[];replacement={};registry={}
        for i,(_,original) in enumerate(v1.ORIGINAL):
            public=v1.PUBLIC_SUITES[i]
            rows.append({'lane':f'E{i+1}','original':{'suite_id':original,'name':'ambiguous'},'portfolio':{'suite_id':public,'name':'ambiguous'}})
            entries.append({'historical_lane':f'E{i+1}','historical_suite':original,'open_portfolio_suite':public,
                            'status':'RETAINED_SUITE_REQUIRES_SEPARATE_RUN_PROVENANCE' if original==public else 'NOT_YET_INTEGRATED'})
            replacement[original]={'new_suite_id':public};registry[original]={}
        resolved=v2.resolve_labels(copy.deepcopy(rows),{'lanes':entries},replacement,registry)
        for index,old,new in ((2,'SDAB','MLAgentBench'),(3,'BankerToolBench','BankerToolBench'),(6,'BinaryAudit','CyberGym'),(11,'AppBench','VisualAgentBench')):
            self.assertEqual(resolved[index]['original']['name'],old)
            self.assertEqual(resolved[index]['portfolio']['name'],new)
            self.assertEqual(resolved[index]['scope_resolution']['same_suite'],old==new)
        entries[6]['open_portfolio_suite']='binaryaudit_eval'
        with self.assertRaises(v1.EvidenceError):v2.resolve_labels(rows,{'lanes':entries},replacement,registry)


if __name__ == '__main__': unittest.main()
