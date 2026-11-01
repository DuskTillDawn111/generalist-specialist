"""Original rho and native backlog reconstruction, CPU fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.flow_eligibility import original_rho,native_stability,seal,report
from tools.predictor_validation_20261027.profiling import digest
from tools.predictor_validation_20261027.probe_validation import receipt

NS=1_000_000_000

class EligibilityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.catalog={'performance_mode':'independent','models':{'text':{'service_time_ms':{'text':100}}},
            'capacity_factors':{'text':{'1':1.,'2':1.5}}}
        self.candidate={'id':'base','execution_context_sha256':'cpu','routing_weights':{'text':{'s':1.}},
            'routes':{'r1':'s','r2':'s'},'resource_configuration':{'mode':'shared'}}
        self.plan={'schema':'gsdeploy.native_flow_plan.v1','aliases':{'s':{'family':'text','supported_tasks':['text']}},
            'workload_rps':{'text':2},'window_seconds':1.,'drain_seconds':1.,
            'trace':[{'request_id':'r1','offset_ns':0,'task':'text','pool_id':'p'},
                     {'request_id':'r2','offset_ns':50_000_000,'task':'text','pool_id':'p'}],
            'candidates':[self.candidate]}
        self.raw={'schema':'gsdeploy.native_flow_collection.v1','kind':'measured','backend':'native','status':'completed',
            'candidate_id':'base','execution_context_sha256':'cpu','resource_configuration':{'mode':'shared'},
            'cohort_sha256':digest(self.plan['trace']),'phase_start_ns':NS,'deadline_ns':2*NS,'observation_end_ns':2*NS,
            'collection_complete':True,'qualification_issues':[],'errors':[],
            'shutdown':{'s':{'forced':False,'exit_code':0,'bye_received':True}},'request_records':[]}
        for i,offset in enumerate([0,50_000_000]):
            self.raw['request_records'].append({'request_id':'r'+str(i+1),'task':'text','pool_id':'p','model':'s',
                'scheduled_time_ns':NS+offset,'actual_submit_time_ns':NS+offset+1_000_000,
                'completion_time_ns':NS+offset+110_000_000,'status':'completed','worker_queue_wait_ms':0})

    def write(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return p

    def test_original_planned_rate_not_finite_cohort_or_actual_busy(self):
        self.plan['workload_rps']['text']=4
        value=original_rho(self.plan,self.candidate,self.catalog)
        self.assertAlmostEqual(value['rho_by_family']['text'],.4) # 4rps * .1s, not 2 finite requests
        self.assertTrue(value['rho_threshold_satisfied']);self.assertIsNone(value['rho_admission_passed'])
        self.assertFalse(value['catalog_applicable'])

    def test_original_capacity_factor_not_replica_count_or_new_alias_limit(self):
        self.plan['aliases']['s2']={'family':'text','supported_tasks':['text']}
        self.plan['workload_rps']['text']=12
        value=original_rho(self.plan,self.candidate,self.catalog)
        self.assertAlmostEqual(value['rho_by_family']['text'],.8) # /g(2)=1.5, not /2
        self.assertTrue(value['rho_threshold_satisfied'])

    def test_missing_planned_weights_cannot_infer_from_actual_counts(self):
        self.candidate.pop('routing_weights')
        value=original_rho(self.plan,self.candidate,self.catalog)
        self.assertEqual(value['status'],'UNKNOWN');self.assertIsNone(value['rho_admission_passed'])

    def test_original_actual_submission_backlog_uses_parent_completion(self):
        value=native_stability(self.plan,self.candidate,self.raw,self.catalog)
        self.assertEqual(value['stability']['stability_state_v2'],'steady')
        self.assertTrue(value['original_stability_passed'])
        self.assertEqual(value['stability_evidence']['peak_backlog_requests'],2)
        self.assertAlmostEqual(value['stability_evidence']['peak_service_demand_backlog'],.2)

    def test_missing_submit_or_shutdown_never_default_true(self):
        self.raw['request_records'][0].pop('actual_submit_time_ns')
        value=native_stability(self.plan,self.candidate,self.raw,self.catalog)
        self.assertFalse(value['original_stability_passed'])
        self.assertIn('insufficient_submission_timestamps',value['stability']['hard_reasons'])
        self.raw['request_records'][0]['actual_submit_time_ns']=NS;self.raw.pop('shutdown')
        self.assertFalse(native_stability(self.plan,self.candidate,self.raw,self.catalog)['original_stability_passed'])

    def test_tampered_thresholds_and_source_binding_refused(self):
        from gsdeploy.stability import DEFAULT_THRESHOLDS
        thresholds={**DEFAULT_THRESHOLDS,'completion_ratio_min':0}
        with self.assertRaisesRegex(ValueError,'unchanged'):native_stability(self.plan,self.candidate,self.raw,self.catalog,thresholds)
        cat=self.write('catalog.json',self.catalog);self.plan['catalog']=receipt(cat);pp=self.write('plan.json',self.plan)
        sp=self.root/'sealed.json';seal(pp,sp);self.catalog['extra']='changed';self.write('catalog.json',self.catalog)
        with self.assertRaisesRegex(ValueError,'source receipt'):report(sp,[],self.root/'report.json')

    def test_prospective_report_unbound_rho_retains_unknown_eligibility(self):
        cat=self.write('catalog.json',self.catalog);self.plan['catalog']=receipt(cat);pp=self.write('plan.json',self.plan)
        sp=self.root/'sealed.json';sealed=seal(pp,sp);origin=sealed['created_ns']+NS
        shift=origin-NS
        for field in ('phase_start_ns','deadline_ns','observation_end_ns'):self.raw[field]+=shift
        for r in self.raw['request_records']:
            for field in ('scheduled_time_ns','actual_submit_time_ns','completion_time_ns'):r[field]+=shift
        self.raw['plan']=receipt(pp)
        result=report(sp,[self.write('raw.json',self.raw)],self.root/'report.json')
        row=result['candidates'][0]
        self.assertTrue(row['original_stability_passed']);self.assertIsNone(row['original_capacity_and_stability_passed'])
        self.assertFalse(row['final_gain_eligible'])

if __name__=='__main__':unittest.main()
