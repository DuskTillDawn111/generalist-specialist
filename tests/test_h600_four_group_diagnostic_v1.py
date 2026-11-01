"""CPU-only final-role fixtures; synthetic timings are never physical evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gsdeploy.tasks import asr_normalize

from tests import test_admission_training_v1 as training_fixtures
from tools.predictor_validation_20261027 import h600_four_group_diagnostic_v1 as m
from tools.predictor_validation_20261027.training_run_audit import quality_totals
from tools.predictor_validation_20261027.probe_validation import read


def write(path, value):
    import json
    Path(path).write_text(json.dumps(value))


class FinalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.helper=training_fixtures.AdmissionTrainingTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'final'
        with patch.object(m.native_flow,'execute',side_effect=AssertionError('No CPU GPU launch')):
            self.plan=m.prepare(self.folder)
        self.plan_path=self.folder/'plan.json'

    def fixture(self, cid='admission2_balanced'):
        job=dict(candidate_id=cid,plan=m.receipt(self.plan_path),
                 output=str(self.folder/'runs'/cid))
        self.helper.campaign['created_ns']=self.plan['created_ns']+1_000_000
        raw,path=self.helper.fixture(job)
        raw['effective_objective']={k:self.plan[k] for k in ('workload_rps','latency_reference_ms','weights',
            'memory_safe_factor','memory_budget_mib','energy_reference_watts','startup_amortization_seconds')}
        if cid.startswith('shared_'):
            raw['admission_snapshot_after']=None
            for key in ('priming_request_records','request_records'):
                for row in raw[key]:
                    for k in list(row):
                        if k.startswith('admission_'):del row[k]
        catalog=read(self.plan['catalog']['path'])
        for row in raw['request_records']:
            reference=catalog['quality']['references'][row['task']][row['pool_id']]
            row['worker_response']['prediction']=(reference['answers'] or [''])[0] if row['task']=='text' else reference['label']
            if row['task']=='asr':
                row['worker_response']['prediction']=asr_normalize(row['worker_response']['prediction'])
        write(path,raw)
        annotation=dict(**raw,native_raw=m.receipt(path),quality_evidence=dict(
            kind='measured_per_request',errors=[],totals_by_request=quality_totals(raw['request_records'],catalog),
            source_receipts=[self.plan['catalog'],m.receipt(m.native_flow.__file__),
                m.receipt(m.ROOT/'gsdeploy/tasks.py'),m.receipt(m.ROOT/'gsdeploy/quality.py')]))
        write(path.parent/'scored-raw.json',annotation)
        return raw,path

    def test_exact_final_roles_pairing_and_fixed_order_prepare_without_gpu(self):
        self.assertEqual(m.audit_plan(self.plan_path),self.plan)
        self.assertEqual(self.plan['phase'],'final_diagnostic')
        self.assertEqual(self.plan['split'],'final')
        self.assertEqual(set(self.plan['candidate_order']),set(m.IDS))
        self.assertEqual(len(self.plan['trace']),1200)
        self.assertEqual(self.plan['locked_policies'],m.LOCKS)
        self.assertFalse(self.plan['final_gain_eligible'])
        self.assertFalse((self.folder/'runs').exists())

    def test_native_admission_final_quality_semantics_and_original_six_terms(self):
        raw,path=self.fixture();before=path.read_bytes()
        with patch.object(m.training,'map_training',side_effect=AssertionError('Final cannot use training mapper')):
            result=m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual((result['requests'],result['known']), (1200,1200))
        self.assertEqual(result['split'],'final');self.assertTrue(result['not_for_training'])
        self.assertFalse(result['final_gain_eligible'])
        self.assertEqual(result['six_term_score']['terms']['Q'],0.)
        self.assertEqual(result['six_term_score']['memory']['safe_factor_application_count'],1)
        self.assertAlmostEqual(result['six_term_score']['terms']['M'],400.*1.110854/49140)
        self.assertEqual(result['original_stability']['status'],'ASSESSED')

    def test_shared_final_uses_same_strict_audit_without_fake_admission(self):
        raw,path=self.fixture('shared_spill_high')
        result=m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)
        self.assertEqual(result['mechanism']['mode'],'shared')
        self.assertTrue(result['six_term_score']['utility_complete'])

    def test_quality_annotation_cannot_rewrite_native_or_actual_quality(self):
        raw,path=self.fixture();scored=path.parent/'scored-raw.json';original=read(scored)
        bad=copy.deepcopy(original);bad['request_records'][0]['completion_time_ns']+=1
        write(scored,bad)
        with self.assertRaisesRegex(ValueError,'exact immutable'):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)
        bad=copy.deepcopy(original);bad['quality_evidence']['totals_by_request']['measurement:0']['numerator']=1.
        write(scored,bad)
        with self.assertRaisesRegex(ValueError,'recomputed'):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)

    def test_final_startup_seal_and_no_validation_relabel(self):
        raw,path=self.fixture()
        with self.assertRaisesRegex(ValueError,'follow prospective'):m.audit_actual(self.plan_path,path,seal_ns=raw['startup_begin_ns'])
        raw['request_records'][0]['split']='validation';write(path,raw)
        with self.assertRaisesRegex(ValueError,'relabelled'):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)

    def test_missing_low_frequency_request_telemetry_and_phase_falsification(self):
        raw,path=self.fixture();original=copy.deepcopy(raw)
        bad=copy.deepcopy(original);bad['request_records']=[r for r in bad['request_records'] if r['task']!='ocr']
        write(path,bad)
        with self.assertRaisesRegex(ValueError,'offered cohort'):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)
        bad=copy.deepcopy(original);bad['board_samples'][5]['owned_used_mib']+=1
        write(path,bad)
        with self.assertRaisesRegex(ValueError,'telemetry'):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)
        bad=copy.deepcopy(original);bad['request_records'][0]['execution_events'][1]['perf_ns']+=1
        write(path,bad)
        with self.assertRaises(ValueError):m.audit_actual(self.plan_path,path,seal_ns=self.plan['created_ns']+1)

    def test_report_all_eight_uses_complete_cohort_without_claiming_capacity(self):
        paths=[self.fixture(cid)[1] for cid in m.ORDER]
        result=m.report_actual(self.plan_path,paths,self.folder/'report',seal_ns=self.plan['created_ns']+1)
        self.assertEqual((result['requested'],result['known']), (9600,9600))
        self.assertEqual(result['status'],'DESCRIPTIVE_COMPLETE')
        self.assertEqual(len(result['actual_audits']),8)
        self.assertEqual(result['primary_value'],0.)
        self.assertFalse(result['final_gain_eligible'])
        self.assertFalse(result['final_seal_authenticated_by_this_helper'])
        self.assertEqual(result['descriptive_comparison']['modes']['admission2']
            ['qualified_observed_envelope']['status'],'DOMAIN_QUALIFICATION_UNKNOWN')

    def test_report_retains_bad_quality_attempt_and_full_missing_denominator(self):
        raw,path=self.fixture();scored_path=path.parent/'scored-raw.json';scored=read(scored_path)
        scored['quality_evidence']['totals_by_request']['measurement:0']['numerator']=1.
        write(scored_path,scored)
        result=m.report_actual(self.plan_path,[path],self.folder/'report',seal_ns=self.plan['created_ns']+1)
        self.assertEqual((result['requested'],result['known']), (9600,0))
        self.assertEqual(result['status'],'DESCRIPTIVE_INCOMPLETE')
        self.assertEqual(len(result['invalid_attempts']),1)
        self.assertEqual(result['invalid_attempts'][0]['recorded_completed_requests'],1200)
        self.assertTrue(result['invalid_attempts'][0]['retained_without_retry'])
        self.assertIsNone(result['primary_value'])


if __name__=='__main__':unittest.main()
