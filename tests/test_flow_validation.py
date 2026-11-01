"""Prospective CPU flow checks. Fixtures are not physical GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from tools.predictor_validation_20261027.flow_validation import seal, report, observed, validate_plan, cohort_sha
from tools.predictor_validation_20261027.probe_validation import receipt
from tools.predictor_validation_20261027.statistical_profiles import SCHEMA, full_count, phase_count
from tools.predictor_validation_20261027.empirical_resources import SCHEMA as RESOURCE_SCHEMA, resource_context
from tools.predictor_validation_20261027.profiling import digest

NS = 1_000_000_000


class FlowValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.training = self.write('training.json', {'schema':'gsdeploy.native_phase_collection.v1','kind':'measured',
            'request_records':[{'split':'training'}]})
        self.aliases = {'s':{'family':'text','supported_tasks':['text']}}
        self.plan = {'schema':'gsdeploy.native_flow_plan.v1','aliases':self.aliases,
            'trace':[{'request_id':'r1','offset_ns':0,'task':'text','pool_id':'known'},
                     {'request_id':'r2','offset_ns':50_000_000,'task':'text','pool_id':'known'}],
            'candidates':[{'id':cid,'execution_context_sha256':'exec','routes':{'r1':'s','r2':'s'},
                'predicted_startup_joules':100,'predicted_prephase_peak_mib':500} for cid in ['base','alternative']],
            'baseline_candidate_id':'base','window_seconds':1,'drain_seconds':1,
            'memory_budget_mib':1000,'memory_safe_factor':1.1,'energy_reference_watts':100,
            'startup_amortization_seconds':10,'workload_rps':{'text':2},'latency_reference_ms':{'text':1000},
            'weights':dict.fromkeys('QTLPME',1/6),'queue_sample_offsets_ns':[0,75_000_000,150_000_000,NS]}
        self.plan_path = self.write('plan.json',self.plan)
        fk,fm = full_count([{'family':'text'}]); pk,pm = phase_count([({'family':'text'},{'phase':'native'})])
        target = ['text','text','known']
        model = {'schema':SCHEMA,'execution_context_sha256':'exec','source_receipts':[receipt(self.training)],
            'profiles':[{'id':'p','family':'text','task':'text','pool_id':'known',
                'execution':{'batch_size':1,'weight_reuse':False,'backend':'statistical'},
                'segments':[{'id':'native','kind':'mixed','duration_ms':100,'demand':dict(sm=None,dram=None,l2=None)}]}],
            'handler_models':[{'target':target,'progress':{fk:1}}],
            'semantic_models':[{'target':target+['native'],'progress':{pk:1}}],
            'family_contexts':{fk:fm},'phase_contexts':{pk:pm}}
        model['model_sha256'] = digest(model)
        self.model_path = self.write('model.json',model)
        contexts = {}
        for active in ({},{'s':{}}):
            key,members = resource_context(active,self.aliases)
            contexts[key] = {'members':members,'evidence':'training',
                'scenarios':{'mean':{'watts':100,'peak_memory_mib':500},'max':{'watts':100,'peak_memory_mib':500}}}
        self.resources_path = self.write('resources.json',{'schema':RESOURCE_SCHEMA,'execution_context_sha256':'exec',
            'provenance':{'split':'training'},'source_receipts':[receipt(self.training)],'contexts':contexts})
        self.quality_path = self.write('quality.json',{'text':{'text':0}})

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def sealing(self):
        return seal(self.plan_path,self.model_path,self.resources_path,self.quality_path,self.root/'sealed')

    def raw(self,cid='base',origin=None):
        origin = origin or time.perf_counter_ns()+NS
        records=[]
        for i,offset in enumerate([0,50_000_000]):
            ws=origin+i*100_000_000
            records.append({'request_id':'r'+str(i+1),'model':'s','task':'text','pool_id':'known',
                'scheduled_time_ns':origin+offset,'status':'completed','observed_worker_start_time_ns':ws,
                'worker_response':{'start_perf_ns':ws,'end_perf_ns':ws+100_000_000},
                'completion_time_ns':ws+110_000_000})
        return {'schema':'gsdeploy.native_flow_collection.v1','kind':'measured','backend':'native','candidate_id':cid,
            'execution_context_sha256':'exec','cohort_sha256':cohort_sha(self.plan),'phase_start_ns':origin,
            'deadline_ns':origin+NS,'startup_begin_ns':origin-NS,'startup_end_ns':origin-NS//2,
            'observation_end_ns':origin+2*NS,'request_records':records,'status':'completed','errors':[],
            'rho_admission_passed':True,'stability':{'stability_state_v2':'steady'},
            'board_samples':[{'time_ns':origin-NS+i*100_000_000,'watts':100,'owned_used_mib':500,
                'owned_aliases':['s'] if i>=5 else [],'foreign_pids':[]} for i in range(31)],
            'quality_evidence':{'kind':'measured_per_request','source_receipts':[receipt(self.training)],
                'totals_by_request':{'r1':{'task':'text','numerator':0,'denominator':1},
                                    'r2':{'task':'text','numerator':0,'denominator':1}}}}

    def test_full_arrivals_parent_receipt_and_wait_service_separated(self):
        sealed=self.sealing(); prediction=sealed['candidates'][0]
        self.assertEqual(prediction['requests'][1]['wait_ms'],50)
        result=observed(self.plan,self.plan['candidates'][0],self.raw())
        self.assertEqual(result['requests'][0]['e2e_ms'],110)
        self.assertEqual(result['requests'][0]['service_ms'],100)
        self.assertEqual(result['requests'][0]['response_tail_ms'],10)
        self.assertAlmostEqual(result['metrics']['terms']['E'],1.1)
        self.assertTrue(result['metrics']['eligible_for_gain_claim'])
        self.assertTrue(result['telemetry_valid']) # residents need not all exist during load

    def test_independent_report_paired_ties_and_parent_latency_error(self):
        sealing=self.sealing();origin=sealing['created_ns']+2*NS
        paths=[self.write(cid+'.json',self.raw(cid,origin)) for cid in ['base','alternative']]
        result=report(self.root/'sealed',paths,self.root/'report')
        self.assertEqual(result['status'],'ASSESSED')
        self.assertEqual(result['paired_delta_coverage'],{'requested':1,'known':1})
        self.assertTrue(result['paired_deltas'][0]['sign_correct'])
        self.assertEqual(result['candidates'][0]['timing_errors']['e2e']['mae'],10)
        self.assertFalse(result['final_four_groups_enabled'])

    def test_missing_candidate_and_quality_stay_incomplete(self):
        sealing=self.sealing();raw=self.raw(origin=sealing['created_ns']+2*NS);raw.pop('quality_evidence')
        result=report(self.root/'sealed',[self.write('raw.json',raw)],self.root/'report')
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['candidates'][1]['status'],'NOT_RUN')
        self.assertIsNone(result['candidates'][0]['observed_metrics']['full_U'])
        self.assertEqual(result['paired_delta_coverage']['known'],0)

    def test_foreign_cohort_or_route_refused(self):
        raw=self.raw();raw['request_records'][0]['model']='foreign'
        with self.assertRaisesRegex(ValueError,'route/input'):
            observed(self.plan,self.plan['candidates'][0],raw)
        raw=self.raw();raw['cohort_sha256']='foreign'
        with self.assertRaisesRegex(ValueError,'cohort'):
            observed(self.plan,self.plan['candidates'][0],raw)

    def test_future_seal_and_mutated_training_refused(self):
        sealing=self.sealing();raw=self.raw(origin=sealing['created_ns']-1)
        with self.assertRaisesRegex(ValueError,'sealed before'):
            report(self.root/'sealed',[self.write('raw.json',raw)],self.root/'report')
        self.training.write_text('{}')
        with self.assertRaisesRegex(ValueError,'source receipt'):
            report(self.root/'sealed',[],self.root/'report')

    def test_incomplete_requests_not_dropped(self):
        raw=self.raw();raw['request_records'][-1]['status']='incomplete'
        result=observed(self.plan,self.plan['candidates'][0],raw)
        self.assertEqual(result['metrics']['cohort_size'],2)
        self.assertEqual(result['metrics']['incomplete_ids'],['r2'])
        self.assertEqual(result['queue_samples'][-1]['depth'],1)
        self.assertFalse(result['metrics']['eligible_for_gain_claim'])
        self.assertIsNone(result['metrics']['full_U'])

    def test_nonfinite_plan_and_invalid_queue_grid_refused(self):
        plan=copy.deepcopy(self.plan);plan['weights']['Q']=float('nan')
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_explicit_common_startup_comparison_keeps_absolute_unknown(self):
        for candidate in self.plan['candidates']:candidate.pop('predicted_startup_joules')
        self.plan['startup_comparison_policy']={'kind':'common_deployment_backend_omitted','group_id':'fixed',
            'candidate_ids':['base','alternative']}
        self.write('plan.json',self.plan);sealing=self.sealing()
        self.assertIsNone(sealing['candidates'][0]['metrics']['full_U'])
        self.assertIsNotNone(sealing['candidates'][0]['comparison_U'])
        paths=[self.write(cid+'.json',self.raw(cid,sealing['created_ns']+2*NS)) for cid in ['base','alternative']]
        result=report(self.root/'sealed',paths,self.root/'report')
        self.assertEqual(result['predicted_absolute_U_coverage']['known'],0)
        self.assertEqual(result['paired_delta_coverage']['known'],1)
        self.assertEqual(result['paired_deltas'][0]['actual_startup_delta_joules'],0)

    def test_missing_startup_without_declaration_cannot_claim_comparison(self):
        for candidate in self.plan['candidates']:candidate.pop('predicted_startup_joules')
        self.write('plan.json',self.plan);sealing=self.sealing()
        self.assertIsNone(sealing['candidates'][0]['comparison_U'])
        paths=[self.write(cid+'.json',self.raw(cid,sealing['created_ns']+2*NS)) for cid in ['base','alternative']]
        result=report(self.root/'sealed',paths,self.root/'report')
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['paired_delta_coverage']['known'],0)
        plan=copy.deepcopy(self.plan);plan['queue_sample_offsets_ns']=[NS+1]
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_ineligible_arithmetic_does_not_grant_gain_claim(self):
        self.sealing();paths=[]
        for cid in ['base','alternative']:
            raw=self.raw();raw['candidate_id']=cid;raw['rho_admission_passed']=False
            paths.append(self.write(cid+'.json',raw))
        result=report(self.root/'sealed',paths,self.root/'report')
        self.assertFalse(result['paired_deltas'][0]['eligible_for_gain_claim'])

if __name__=='__main__':unittest.main()
