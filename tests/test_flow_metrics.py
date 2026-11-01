"""Hand-computable CPU six-term flow arithmetic, no GPU evidence."""
import copy
import unittest
from tools.predictor_validation_20261027.flow_metrics import score_flow

NS=1_000_000_000


class FlowMetricsTests(unittest.TestCase):
    def plan(self):
        return {'workload_rps':{'ocr':1,'text':1},'latency_reference_ms':{'ocr':1000,'text':1000},
                'weights':dict.fromkeys('QTLPME',1/6),'memory_safe_factor':1.1,'memory_budget_mib':1000,
                'energy_reference_watts':100,'startup_amortization_seconds':10,'resident_aliases':['s','o']}

    def raw(self):
        records=[{'request_id':'o1','task':'ocr','status':'completed','scheduled_time_ns':NS,'completion_time_ns':NS+100_000_000},
                 {'request_id':'o2','task':'ocr','status':'completed','scheduled_time_ns':NS+200_000_000,'completion_time_ns':NS+500_000_000},
                 {'request_id':'t','task':'text','status':'completed','scheduled_time_ns':NS,'completion_time_ns':2*NS+100_000_000}]
        return {'schema':'gsdeploy.native_flow_collection.v1','kind':'measured','backend':'native','phase_start_ns':NS,'deadline_ns':2*NS,
                'startup_begin_ns':0,'startup_end_ns':100_000_000,'observation_end_ns':3*NS,'request_records':records,
                'board_samples':[{'time_ns':i*100_000_000,'watts':100,'owned_used_mib':500,'owned_aliases':['s','o'],'foreign_pids':[]} for i in range(31)],
                'quality_evidence':{'totals_by_request':{'o1':{'task':'ocr','numerator':1,'denominator':1},
                    'o2':{'task':'ocr','numerator':0,'denominator':9},'t':{'task':'text','numerator':0,'denominator':1}}}}

    def test_parent_completion_and_quality_total_ratio_and_startup_to_measurement(self):
        result=score_flow(self.raw(),self.plan());terms=result['terms']
        self.assertAlmostEqual(terms['Q'],.05) # .5*(1/10), not mean per-request CER .25
        self.assertAlmostEqual(terms['T'],1/3)
        self.assertAlmostEqual(terms['L'],.65) # weighted .2s OCR and 1.1s text
        self.assertAlmostEqual(terms['M'],.55)
        self.assertAlmostEqual(terms['E'],1.1) # 100J window +100J startup /10s
        self.assertEqual(result['energy']['startup_joules'],100)
        self.assertEqual(result['completed_within_window'],2)
        self.assertIsNone(result['eligible'])

    def test_drain_energy_excluded_and_memory_included(self):
        raw=self.raw()
        for sample in raw['board_samples']:
            if sample['time_ns']>2*NS:sample.update(watts=1000,owned_used_mib=700)
        result=score_flow(raw,self.plan())
        self.assertAlmostEqual(result['terms']['E'],1.1)
        self.assertAlmostEqual(result['terms']['M'],.77)

    def test_missing_quality_stays_unknown(self):
        raw=self.raw();raw['quality_evidence']['totals_by_request'].pop('t')
        result=score_flow(raw,self.plan())
        self.assertIsNone(result['terms']['Q']);self.assertIsNone(result['diagnostic_U'])

    def test_foreign_and_missing_resident_peak_rejected(self):
        raw=self.raw();raw['board_samples'][12]['foreign_pids']=[3]
        result=score_flow(raw,self.plan());self.assertIsNone(result['terms']['M']);self.assertIsNone(result['terms']['E'])
        raw=self.raw();raw['board_samples'][12]['owned_aliases']=['s']
        self.assertIsNone(score_flow(raw,self.plan())['terms']['M'])

    def test_incomplete_drain_retains_original_completed_latency_diagnostics(self):
        raw=self.raw();raw['request_records'][-1]['status']='incomplete'
        result=score_flow(raw,self.plan())
        self.assertIsNone(result['terms']['L']);self.assertIsNone(result['terms']['Q'])
        self.assertIn('INCOMPLETE_COHORT_LATENCIES_DIAGNOSTIC_ONLY',result['unknown_or_ineligible_reasons'])

    def test_ocr_loss_above_one_allowed(self):
        raw=self.raw();raw['quality_evidence']['totals_by_request']['o1']['numerator']=25
        self.assertAlmostEqual(score_flow(raw,self.plan())['terms']['Q'],1.25)

    def test_invalid_owned_sample_cannot_silently_drop_a_peak(self):
        for value in (None,float('nan'),-1,True):
            with self.subTest(value=value):
                raw=self.raw();raw['board_samples'][12]['owned_used_mib']=value
                result=score_flow(raw,self.plan())
                self.assertIsNone(result['terms']['M'])
                self.assertIsNone(result['memory']['raw_owned_sampled_peak_mib'])
                self.assertIn('INVALID_OR_MISSING_RAW_OWNED_MEMORY_SAMPLE',result['unknown_or_ineligible_reasons'])

    def test_telemetry_order_duplicate_or_missing_time_rejected(self):
        for times in ((2,1),(1,1),(None,2)):
            with self.subTest(times=times):
                raw=self.raw()
                raw['board_samples'][0]['time_ns'],raw['board_samples'][1]['time_ns']=times
                with self.assertRaisesRegex(ValueError,'telemetry timestamps'):score_flow(raw,self.plan())

    def test_missing_or_boolean_power_stays_unknown(self):
        for value in (None,True,float('nan'),-1):
            raw=self.raw();raw['board_samples'][12]['watts']=value
            self.assertIsNone(score_flow(raw,self.plan())['terms']['E'])

if __name__=='__main__':unittest.main()
