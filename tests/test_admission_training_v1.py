"""Independent CPU native fixtures for prospective admission2 training roles.

No worker, CUDA context, or GPU collection is started. Full 1200-request traces
exercise the real campaign verifier, mechanism audit and semantic parser.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_training_v1 as m


def write(path,value):
    Path(path).write_text(json.dumps(value))


class AdmissionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'campaign'
        with patch.object(m.native_flow,'execute',side_effect=AssertionError('Prepare must never launch GPU')):
            self.campaign=m.prepare(self.folder)
        self.path=self.folder/'campaign.json';self.job=self.campaign['jobs'][0]
        self.plan=m.read(self.job['plan']['path']);self.context=m.read(self.plan['context']['path'])

    def fixture(self,job=None):
        job=job or self.job;plan=m.read(job['plan']['path']);context=m.read(plan['context']['path'])
        output=Path(job['output']);output.mkdir(parents=True,exist_ok=False)
        start=self.campaign['created_ns']+1_000_000_000
        left=start+1_000_000_000;deadline=left+600_000_000_000;right=deadline+1_000_000
        ready={a:dict(pid=100+i,worker=f,clock='time.perf_counter_ns',
            spec_sha256=context['worker_bindings'][f]['spec_sha256'],
            tasks=['text','image','asr'] if f=='generalist_2' else ['ocr' if f=='ocr' else 'text'])
            for i,(a,f) in enumerate(m.RESIDENTS.items())}
        def row(rid,alias,task,pool,scheduled,phase):
            worker_start=scheduled+20;worker_end=worker_start+1000;pid=ready[alias]['pid']
            stages=[dict(id='native_inference',kind='gpu',start_ns=worker_start+100,end_ns=worker_start+800),
                dict(id='post_sync',kind='sync',start_ns=worker_start+800,end_ns=worker_start+900)]
            forwards=[dict(id='prefill_0000',kind='gpu',start_ns=worker_start+200,end_ns=worker_start+300,forward_index=0),
                dict(id='decode_0001',kind='gpu',start_ns=worker_start+500,end_ns=worker_start+600,forward_index=1)]
            response=dict(request_id=rid,task=task,task_type=task,success=True,start_perf_ns=worker_start,
                end_perf_ns=worker_end,observed_stages=stages,
                semantic_forward_observations=dict(schema='gsdeploy.semantic_forward_observation.v1',request_id=rid,
                    clock='time.perf_counter_ns',records=forwards,instrumentation_errors=[],gpu_synchronization_added=False))
            events=[dict(request_id=rid,pid=pid,perf_ns=worker_start,received_time_ns=worker_start+1)]
            for stage in stages+forwards:
                for edge in ('start','end'):
                    timestamp=stage[edge+'_ns']
                    events.append(dict(request_id=rid,event=edge,segment_id=stage['id'],kind=stage['kind'],
                        perf_ns=timestamp,received_time_ns=timestamp+1,
                        nvtx_name=f"GS_STAGE:{rid}:{stage['kind']}:{stage['id']}"))
            events.sort(key=lambda x:x['perf_ns'])
            return dict(request_id=rid,model=alias,task=task,pool_id=pool,phase=phase,status='completed',
                scheduled_time_ns=scheduled,enqueued_time_ns=scheduled+1,
                admission_wait_start_time_ns=scheduled+2,admission_granted_time_ns=scheduled+5,
                actual_submit_time_ns=scheduled+10,submit_write_completed_time_ns=scheduled+12,
                observed_worker_start_time_ns=worker_start,worker_response=response,
                completion_time_ns=worker_end+50,execution_events=events)
        candidate=next(c for c in plan['candidates'] if c['id']==job['candidate_id'])
        records=[row(r['request_id'],candidate['routes'][r['request_id']],r['task'],r['pool_id'],left+r['offset_ns'],'measurement')
            for r in plan['trace']]
        priming=[row('priming:'+str(i),alias,task,pool,start+20_000_000+i*1_000_000,'priming')
            for i,(alias,task,pool) in enumerate(m._priming(m.read(plan['profile_plan']['path'])))]
        times=[start-1,start,*[left+i*1_000_000_000 for i in range(601)],right,right+1]
        samples=[dict(time_ns=t,watts=100.,owned_used_mib=0. if t<start+10_000_000 else 400.,
            processes_mib={} if t<start+10_000_000 else {str(v['pid']):100. for v in ready.values()},
            owned_aliases=[] if t<start+10_000_000 else list(m.RESIDENTS),foreign_pids=[])
            for t in times]
        raw=dict(schema='gsdeploy.native_flow_collection.v1',kind='measured',backend='native',status='completed',
            collection_complete=True,errors=[],runtime_errors=[],telemetry_errors=[],qualification_issues=[],
            plan=job['plan'],context=plan['context'],candidate_id=job['candidate_id'],
            execution_context_sha256=plan['execution_context_sha256'],trace_sha256=m.digest(plan['trace']),
            resource_configuration=candidate['resource_configuration'],resident_aliases=list(m.RESIDENTS),replica_to_family=m.RESIDENTS,
            ready=ready,startup_begin_ns=start,startup_end_ns=start+10_000_000,phase_start_ns=left,deadline_ns=deadline,
            arrival_loop_end_ns=deadline+10,observation_end_ns=right,request_records=records,
            priming_request_records=priming,warmup_request_records=[],board_samples=samples,
            admission_snapshot_after=dict(active={},waiting=[],closed=True,peak_active=2),
            shutdown={a:dict(forced=False,exit_code=0,bye_received=True) for a in m.RESIDENTS})
        # If two planned requests happen at the same instant, they use distinct
        # aliases under this source plan; observed peaks stay within max2.
        write(output/'invocation.json',dict(plan=job['plan'],candidate_id=job['candidate_id'],runner=plan['runner_source']))
        write(output/'raw.json',raw)
        return raw,output/'raw.json'

    def test_prepare_and_verify_exact_eight_roles_sources_routes_without_launch(self):
        value=m.verify(self.path)
        self.assertEqual(value,self.campaign);self.assertEqual(len(value['jobs']),8)
        self.assertEqual({(j['seed'],j['candidate_id']) for j in value['jobs']},{(s,c) for s in m.SEEDS for c in m.ROUTES})
        self.assertEqual({j['campaign_role'] for j in value['jobs']},{'primary_training'})
        self.assertNotIn('startup_comparison_policy',self.plan)
        self.assertNotIn('predicted_prephase_peak_mib',self.plan['candidates'][0])
        self.assertEqual(self.plan['context'],m.receipt(m.PROFILE/'context.json'))
        self.assertFalse(any(Path(j['output']).exists() for j in value['jobs']))

    def test_real_native_semantic_mapping_continuous_train_and_resource_capacity_fields(self):
        raw,path=self.fixture();before=path.read_bytes()
        with patch.object(m.flow_profiles,'map_flow_collection',side_effect=AssertionError('Shared mapper forbidden')):
            mapped=m.map_training(self.job,self.path)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(len(mapped['traces']),1);trace=mapped['traces'][0]
        self.assertEqual(len(trace['requests']),1200);self.assertEqual(trace['split'],'training')
        self.assertFalse(trace['not_for_training']);self.assertEqual(trace['controller_sha256'],self.job['controller_sha256'])
        self.assertEqual(trace['execution_context_sha256'],self.plan['execution_context_sha256'])
        self.assertNotEqual(trace['controller_sha256'],trace['execution_context_sha256'])
        evidence=mapped['resource_evidence']
        self.assertEqual(evidence['split'],'training');self.assertEqual(evidence['startup_joules'],100.)
        self.assertEqual(evidence['raw_all_sampled_owned_peak_mib'],400.)
        self.assertFalse(evidence['safe_factor_applied']);self.assertFalse(evidence['capacity_bound'])
        self.assertEqual(evidence['training_seed'],self.job['seed'])
        self.assertEqual(len(mapped['service_observations']),1200)
        self.assertAlmostEqual(mapped['service_observations'][0]['service_ms'],.001)
        self.assertEqual(mapped['service_observations'][0]['service_ms'],mapped['service_observations'][0]['worker_service_ms'])
        self.assertTrue(all(x['not_for_training'] for x in evidence['stage_history'][:18]))
        self.assertEqual(mapped['mapping_report']['original_stability']['status'],'ASSESSED')
        self.assertFalse(mapped['six_term_training_evidence_complete'])
        self.assertEqual(mapped['quality_evidence_status'],'NOT_AUDITED_BY_TRAINING_MAPPER')

    def test_rejects_validation_pilot_final_or_posthoc_job_wrappers(self):
        for role in ('predictor_validation','mechanism_validation','final_confirmation'):
            job={**self.job,'campaign_role':role}
            with self.assertRaisesRegex(ValueError,'wrappers'):m.map_training(job,self.path)
        with self.assertRaisesRegex(ValueError,'artifact path'):m.verify(self.campaign)
        raw,path=self.fixture();raw['native_raw']=m.receipt(path);write(path,raw)
        with self.assertRaisesRegex(ValueError,'Immutable native training'):m.map_training(self.job,self.path)

    def test_actual_mapped_evidence_consumed_by_frozen_resource_numeric_fit_run(self):
        from tools.predictor_validation_20261027.h600_resources_v1 import fit_run
        self.fixture();mapped=m.map_training(self.job,self.path)
        fit=fit_run(mapped['resource_evidence'],mapped['traces'],self.plan['aliases'])
        self.assertEqual(fit['execution_context_sha256'],self.plan['execution_context_sha256'])
        contexts=fit['task_specific_resource_omission_audit']['active_task_contexts']
        self.assertTrue(contexts)
        self.assertTrue(all(abs(c['mean_watts']-100.)<1e-7 for c in contexts.values()))
        self.assertTrue(all(c['max_trial_sampled_peak_mib']==400. for c in contexts.values()))
        self.assertFalse(fit['telemetry']['capacity_bound'])

    def test_rejects_startup_before_campaign_or_explicit_validation_request(self):
        raw,path=self.fixture();raw['startup_begin_ns']=self.campaign['created_ns'];write(path,raw)
        with self.assertRaisesRegex(ValueError,'after complete campaign'):m.map_training(self.job,self.path)
        raw['startup_begin_ns']=self.campaign['created_ns']+1_000_000_000
        raw['request_records'][0]['split']='validation';write(path,raw)
        with self.assertRaisesRegex(ValueError,'cannot become training'):m.map_training(self.job,self.path)

    def test_missing_foreign_or_wrong_route_request_never_drops_from_training(self):
        raw,path=self.fixture();original=copy.deepcopy(raw)
        raw['request_records'].pop();write(path,raw)
        with self.assertRaises(ValueError):m.map_training(self.job,self.path)
        raw=copy.deepcopy(original);raw['request_records'][0]['pool_id']='foreign';write(path,raw)
        with self.assertRaisesRegex(ValueError,'arrival/input/route'):m.map_training(self.job,self.path)
        raw=copy.deepcopy(original);raw['request_records'][0]['model']='ocr#0';write(path,raw)
        with self.assertRaises(ValueError):m.map_training(self.job,self.path)

    def test_real_phase_ready_telemetry_and_controller_falsification(self):
        raw,path=self.fixture();original=copy.deepcopy(raw)
        variants=[('phase',lambda x:x['request_records'][0]['execution_events'][1].update(perf_ns=x['request_records'][0]['execution_events'][1]['perf_ns']+1)),
            ('READY',lambda x:x['ready']['text#0'].update(clock='other')),
            ('telemetry',lambda x:x['board_samples'][5].update(owned_used_mib=401.)),
            ('controller',lambda x:x.update(resource_configuration=m.native_flow.resource_configuration(None)))]
        for name,mutate in variants:
            with self.subTest(name=name):
                changed=copy.deepcopy(original);mutate(changed);write(path,changed)
                with self.assertRaises(ValueError):m.map_training(self.job,self.path)

    def test_complete_priming_and_warmup0_cannot_be_fabricated_or_dropped(self):
        raw,path=self.fixture();raw['priming_request_records'].pop();write(path,raw)
        with self.assertRaisesRegex(ValueError,'largest-three priming'):m.map_training(self.job,self.path)

    def test_campaign_role_or_source_rehash_does_not_override_fixed_policy(self):
        value=copy.deepcopy(self.campaign);value['jobs'][0]['campaign_role']='validation'
        value['campaign_sha256']=m.digest({k:v for k,v in value.items() if k!='campaign_sha256'});write(self.path,value)
        with self.assertRaisesRegex(ValueError,'All eight unique'):m.verify(self.path)

    def test_execute_has_no_launch_without_main_thread_review_and_no_retry(self):
        with patch.object(m.native_flow,'execute') as execute:
            with self.assertRaises(FileNotFoundError):m.execute(self.path,self.job['job_id'],self.root/'missing-review.json')
            execute.assert_not_called()
        self.fixture()
        review=self.review()
        with patch.object(m.native_flow,'execute') as execute:
            with self.assertRaisesRegex(FileExistsError,'No retry'):m.execute(self.path,self.job['job_id'],review)
            execute.assert_not_called()

    def review(self,incomplete=False):
        report=self.root/'stage2.json';gate=self.root/'stage2-gate.json';stage3=self.root/'stage3.json'
        write(report,dict(candidate_coverage=dict(requested=4,observed=4),
            candidates=[dict(known=1200) for _ in range(4)]))
        write(gate,dict(report=m.receipt(report),status='FAIL'))
        write(stage3,dict(schema='gsdeploy.H600_admission_service_pilot.v1.report',
            status='DIAGNOSTIC_INCOMPLETE' if incomplete else 'DIAGNOSTIC_COMPLETE',
            candidates=[dict(candidate_id=c) for c in ('shared_spill_low','admission1_spill_low','admission2_spill_low')]))
        review=self.root/'review.json'
        write(review,dict(schema=m.SCHEMA+'.main_thread_review',campaign=m.receipt(self.path),
            decision='APPROVED_FOR_TRAINING_ONLY',reviewer_role='main_thread',final_gain_eligible=False,
            created_ns=self.campaign['created_ns']+100,stage2_report=m.receipt(report),stage2_gate=m.receipt(gate),stage3_report=m.receipt(stage3)))
        return review

    def test_review_preserves_negative_result_and_requires_reason_for_incomplete_stage3(self):
        review=self.review(incomplete=True)
        with self.assertRaisesRegex(ValueError,'explain training'):m._review(self.path,review)
        packet=m.read(review);packet['stage3_incomplete_training_reason']='Acquire new training evidence; old failure remains unchanged'
        write(review,packet)
        result=m._review(self.path,review);self.assertFalse(result['final_gain_eligible'])

    def test_execute_mocked_native_uses_exact_fixed_job_and_exclusive_attempt(self):
        review=self.review()
        def collect(plan,cid,output):
            self.assertEqual(plan,self.job['plan']['path']);self.assertEqual(cid,self.job['candidate_id'])
            self.assertEqual(str(output),self.job['output']);self.fixture()
        with patch.object(m.native_flow,'execute',side_effect=collect) as execute:
            result=m.execute(self.path,self.job['job_id'],review)
        execute.assert_called_once();self.assertEqual(result['request_count'],1200)
        self.assertTrue((Path(self.job['output'])/'training-complete.json').exists())
        self.assertTrue((Path(self.job['output']).parent/(self.job['candidate_id']+'.training-attempt.json')).exists())

    def test_predeclared_job_order_blocks_out_of_order_collection(self):
        review=self.review();job=self.campaign['jobs'][1]
        with patch.object(m.native_flow,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'Predeclared job order'):m.execute(self.path,job['job_id'],review)
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
