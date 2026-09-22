import copy
import unittest
import generate_v3 as v3


def fixture():
    ids = [f't{i}' for i in range(60)]
    episodes = [{'task_id':x,'end_time':'fixture','termination_reason':'user_stop','reward_info':{'reward':0.0}} for x in ids[:20]]
    audit = {'status':'INDEPENDENT_NATIVE10_TERMINAL_AUDIT_PASS','full_suite_denominator':97,
             'subset_denominator':60,'full_suite_score':None,'graded_episodes':20,'passes':0,
             'native_batch_records':30,'current_started_ids':ids[:33], 'remaining_never_started':ids[33:],
             'native_failure_records':[{'task_id':x,'termination_reason':'infrastructure_error'} for x in ids[20:30]],
             'interrupted_ids':ids[30:33]}
    return audit, episodes, ids


class TerminalTests(unittest.TestCase):
    def test_terminal_counts_not_stale_snapshot(self):
        m = v3.terminal_tau(*fixture())
        self.assertEqual(m['coverage'],v3.v1.ratio(20,97))
        self.assertEqual(m['accuracy_or_success'],v3.v1.ratio(0,20))
        self.assertEqual(len(m['started_without_grade']),13)

    def test_dispositions_cannot_overlap(self):
        a,e,ids=fixture();a['interrupted_ids'][0]=ids[0]
        with self.assertRaises(ValueError):v3.terminal_tau(a,e,ids)

    def test_stale_count_or_full_score_rejected(self):
        for key,value in [('graded_episodes',16),('full_suite_score',0),('native_batch_records',33)]:
            a,e,ids=fixture();a[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):v3.terminal_tau(a,e,ids)

    def test_no_imputed_reward(self):
        a,e,ids=fixture();e[0]['reward_info']['reward']=1
        with self.assertRaises(ValueError):v3.terminal_tau(a,e,ids)

    def test_infra_attempt_is_not_accuracy(self):
        r={'status':'NATIVE_ATTEMPT_TERMINAL_INFRASTRUCTURE_FAILURE_CLEANED','native_attempts':1,
           'task_model_calls':0,'native_success':False,'native_task':'hello_world','actor_smoke_calls':2,'error':'missing poetry'}
        result={'results':[{'task_id':'hello_world','runs':[{'success':False,'agent_output':None,'error':'missing poetry'}]}]}
        m=v3.e9_terminal(r,result)
        self.assertIsNone(m['accuracy_or_success'])
        self.assertEqual(m['coverage'],v3.v1.ratio(0,34))
        self.assertEqual(m['terminal_attempt_coverage'],v3.v1.ratio(1,34))
        r['task_model_calls']=1
        with self.assertRaises(ValueError):v3.e9_terminal(r,result)


if __name__=='__main__':unittest.main()
