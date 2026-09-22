import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import generate_v1 as g

ROOT = Path(__file__).resolve().parents[2]


def audit(rows):
    return {'full_suite_denominator':300, 'outcomes':rows,'terminal_attempts':len(rows),
            'native_resolved_count':sum(r['resolved'] for r in rows)}


def outcome(task, status='NATIVE_REPORT_VERIFIED', resolved=False):
    return {'instance_id':task,'status':status,'resolved':resolved,'run_id':'synthetic'}


def episode(task, reward):
    return {'task_id':task,'end_time':'synthetic','termination_reason':'user_stop','reward_info':{'reward':reward}}


class MetricsTests(unittest.TestCase):
    def test_coverage_not_accuracy(self):
        m=g.measure('public',1967,1967,450,complete=True)
        self.assertEqual(m['coverage']['fraction'],1)
        self.assertAlmostEqual(m['accuracy_or_success']['fraction'],450/1967)
        self.assertFalse(m['full_original_suite_claim'])

    def test_unknown_is_not_zero(self):
        m=g.measure('coverage only',40,75)
        self.assertIsNone(m['accuracy_or_success'])

    def test_zero_score_is_present(self):
        self.assertEqual(g.measure('zero',2,97,0)['accuracy_or_success']['fraction'],0)

    def test_invalid_counts(self):
        for n,d in ((True,2),(-1,2),(3,2),(1,0),(1,1.0),(float('nan'),2)):
            with self.subTest(n=n,d=d),self.assertRaises(g.EvidenceError): g.ratio(n,d)

    def test_partial_cannot_be_complete(self):
        with self.assertRaises(g.EvidenceError): g.measure('partial',13,255,13,complete=True)

    def test_swe_errors_separate(self):
        m=g.aggregate_swe([audit([outcome('a',resolved=True),outcome('b','NATIVE_PATCH_APPLY_FAILED'),outcome('c','NATIVE_EMPTY_PATCH')])])
        self.assertEqual(m['coverage']['numerator'],1)
        self.assertEqual(m['terminal_attempt_coverage']['numerator'],3)
        self.assertEqual(m['accuracy_or_success']['fraction'],1)
        self.assertEqual(m['terminal_attempt_success_rate_errors_as_zero']['fraction'],1/3)

    def test_swe_duplicate_replay_rejected(self):
        a=audit([outcome('a')])
        with self.assertRaises(g.EvidenceError): g.aggregate_swe([a,a])

    def test_swe_unknown_not_native_zero(self):
        with self.assertRaises(g.EvidenceError): g.aggregate_swe([audit([outcome('a','UNKNOWN')])])

    def test_swe_denominator_drift(self):
        a=audit([]);a['full_suite_denominator']=731
        with self.assertRaises(g.EvidenceError): g.aggregate_swe([a])

    def test_tau_separate_runs(self):
        a=g.tau_run('run03',[episode('a',1),episode('b',0)])
        b=g.tau_run('run04',[episode('a',0)])
        self.assertEqual(a['accuracy_or_success']['fraction'],.5)
        self.assertEqual(b['accuracy_or_success']['fraction'],0)
        self.assertEqual(a['coverage']['denominator'],97)

    def test_tau_duplicate_within_run(self):
        with self.assertRaises(g.EvidenceError): g.tau_run('run',[episode('a',1),episode('a',0)])

    def test_tau_wrong_task(self):
        with self.assertRaises(g.EvidenceError): g.tau_run('run',[episode('a',1)],['b'])

    def test_tau_unfinished_and_nonfinite(self):
        for e in (episode('a',float('nan')),episode('a',True),dict(episode('a',1),end_time=None)):
            with self.assertRaises(g.EvidenceError): g.tau_run('run',[e])

    def test_omni_exclusions_not_accuracy_denominator(self):
        m=g.measure('parser accepted',4426,4428,2271)
        self.assertEqual(m['coverage']['denominator'],4428)
        self.assertEqual(m['accuracy_or_success']['denominator'],4426)
        self.assertFalse(m['complete_named_scope'])

    def test_e7_primary_not_inventory(self):
        self.assertEqual(g.EXPECTED_ORIGINAL[6],28)
        self.assertEqual(g.EXPECTED_PUBLIC[6],1507)

    def test_private_denominators_unknown(self):
        for index in (5,9,11,12,13): self.assertIsNone(g.EXPECTED_ORIGINAL[index])

    def test_complete_report_contract(self):
        rows = [{'lane':f'E{i+1}', 'original':{'suite_id':g.ORIGINAL[i][1],'denominator':g.EXPECTED_ORIGINAL[i]},
                 'portfolio':{'suite_id':g.PUBLIC_SUITES[i],'declared_denominator':g.EXPECTED_PUBLIC[i], 'measurements':[]}}
                for i in range(14)]
        rows[7]['portfolio']['measurements']=[g.measure('public',1967,1967,450,complete=True)]
        report={'rows':rows,'all_original_suites_complete':False,'replacement_score_lanes':['E8'],'replacement_score_lane_count':1}
        g.validate_report(report)
        altered=copy.deepcopy(report);altered['rows'][6]['original']['denominator']=46
        with self.assertRaises(g.EvidenceError): g.validate_report(altered)
        altered=copy.deepcopy(report);altered['rows'][7]['portfolio']['measurements'][0]['full_original_suite_claim']=True
        with self.assertRaises(g.EvidenceError): g.validate_report(altered)

    def test_14_lane_duplicates_and_wrong_count(self):
        with self.assertRaises(g.EvidenceError): g.validate_report({'rows':[{'lane':'E1'}]*14})
        with self.assertRaises(g.EvidenceError): g.validate_report({'rows':[]})


class FileTests(unittest.TestCase):
    def setUp(self):
        if shutil.disk_usage(ROOT).free<g.FLOOR: self.skipTest('Disk floor: no writes')
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.codex-run/finish_20260912/progress_reporting')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_hash_tamper(self):
        (self.root/'x.json').write_text('{}')
        r=g.Reader(self.root);r.load('x.json')
        (self.root/'x.json').write_text('{"changed":true}')
        with self.assertRaises(g.EvidenceError): r.stable()

    def test_new_receipt_race(self):
        r=g.Reader(self.root);r.glob('*.json')
        (self.root/'x.json').write_text('{}')
        with self.assertRaises(g.EvidenceError): r.stable()

    def test_expected_hash(self):
        (self.root/'x.json').write_text('{}')
        with self.assertRaises(g.EvidenceError): g.Reader(self.root).load('x.json','0'*64)

    def test_escape(self):
        with self.assertRaises(g.EvidenceError): g.Reader(self.root).resolve('../outside')

    def test_no_parent_output_mutation(self):
        parent=self.root/'outputs/PES_Phase2_Review_2026-09-12/finish'
        with self.assertRaises(g.EvidenceError): g.publish(self.root,parent,{})
        self.assertFalse(parent.exists())

    def test_no_overwrite(self):
        p=self.root/'finish/progress_reporting_audit/old';p.mkdir(parents=True)
        (p/'report.json').write_text('PRESERVE')
        with self.assertRaises(g.EvidenceError): g.publish(self.root,p,{})
        self.assertEqual((p/'report.json').read_text(),'PRESERVE')

    def test_disk_floor_read_and_publish(self):
        with patch.object(g.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10,9,1)):
            with self.assertRaises(g.EvidenceError): g.Reader(self.root).guard()
            with self.assertRaises(g.EvidenceError): g.publish(self.root,self.root/'finish/progress_reporting_audit/new',{})


if __name__ == '__main__': unittest.main()
