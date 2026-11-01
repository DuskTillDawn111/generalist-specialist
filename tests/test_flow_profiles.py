"""Continuous flow source/provenance/boundary guards using CPU observations."""
import copy
import json
from unittest import mock
import unittest

from tests import test_background_profiles as fixtures
from tools.predictor_validation_20261027.background_profiles import RESIDENTS,receipt
from tools.predictor_validation_20261027.flow_profiles import map_flow_collection
from tools.predictor_validation_20261027.profiling_analysis import digest
from tools.predictor_validation_20261027.background_statistical import fit_traces


class FlowProfilesTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BackgroundMapperTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.path=self.root/'flow-raw.json'
        self.raw=copy.deepcopy(self.fixture.raw)
        original_plan=json.loads((self.root/'plan.json').read_text())
        original_plan['input_assignments']=[{'task':t,'pool_id':'pool-'+t,'view':'known_input'} for t in ('image','asr')]
        original_plan.pop('plan_sha256');original_plan['plan_sha256']=digest(original_plan)
        self.profile_path=self.root/'profile-plan.json';self.write(self.profile_path,original_plan)
        for row in self.raw['ready'].values():row['clock']='time.perf_counter_ns'
        for i,row in enumerate(self.raw['request_records']):
            row.pop('split');row.pop('profile_job_id');row['phase']='measurement'
            row['scheduled_time_ns']=10_000_000+i*200_000_000
            row['actual_submit_time_ns']=row['scheduled_time_ns']
            row['completion_time_ns']=row['worker_response']['end_perf_ns']+1_000_000
            row['execution_events'].sort(key=lambda e:e['perf_ns'])
        aliases={a:{'family':f,'supported_tasks':r['tasks']} for (a,f),r in zip(RESIDENTS.items(),self.raw['ready'].values())}
        config={'schema':'gsdeploy.resource_configuration.v1','device':'0','mode':'shared','allocations':{a:{} for a in RESIDENTS}}
        trace=[{'request_id':r['request_id'],'offset_ns':i*200_000_000,'task':r['task'],'pool_id':r['pool_id']}
               for i,r in enumerate(self.raw['request_records'])]
        runner=self.root/'runner.py';runner.write_text('# CPU fixture native runner\n')
        catalog=self.root/'catalog.json';self.write(catalog,{})
        self.plan={'schema':'gsdeploy.native_flow_plan.v1','kind':'prospective_training','phase':'predictor_training','split':'training',
            'created_ns':0,'execution_context_sha256':self.fixture.identity,'aliases':aliases,'trace':trace,
            'candidates':[{'id':'candidate','execution_context_sha256':self.fixture.identity,'routes':{r['request_id']:r['model'] for r in self.raw['request_records']},
                           'resource_configuration':config}],
            'baseline_candidate_id':'candidate','window_seconds':.75,'drain_seconds':.1,
            'workload_rps':{'image':2,'asr':2},'latency_reference_ms':{'image':100,'asr':100},
            'memory_budget_mib':1000,'memory_safe_factor':1.1,'energy_reference_watts':100,'startup_amortization_seconds':10,
            'weights':dict.fromkeys('QTLPME',1/6),'context':self.raw['context'],'profile_plan':receipt(self.profile_path),
            'runner_source':receipt(runner),'catalog':receipt(catalog)}
        self.plan_path=self.root/'flow-plan.json'
        self.raw.update(schema='gsdeploy.native_flow_collection.v1',candidate_id='candidate',resident_aliases=list(RESIDENTS),
            replica_to_family=RESIDENTS,resource_configuration=config,collection_complete=True,qualification_issues=[],
            startup_begin_ns=1_000_000,startup_end_ns=5_000_000,phase_start_ns=10_000_000,deadline_ns=760_000_000,
            priming_request_records=[],warmup_request_records=[],
            shutdown={a:{'exit_code':0,'forced':False,'bye_received':True} for a in RESIDENTS})
        for key in ('executed_jobs','batches','observation_start_ns'):self.raw.pop(key)
        self.save()

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def save(self):
        self.plan.pop('plan_sha256',None);self.plan['plan_sha256']=digest(self.plan)
        self.write(self.plan_path,self.plan);self.raw['plan']=receipt(self.plan_path)
        self.raw['trace_sha256']=digest(self.plan['trace'])
        self.write(self.root/'invocation.json',{'plan':self.raw['plan'],'candidate_id':'candidate','runner':self.plan['runner_source']})
        self.write(self.path,self.raw)

    def test_continuous_cohort_trains_without_per_request_solo_trials(self):
        mapped=map_flow_collection(self.path);trace=mapped['traces'][0]
        self.assertEqual(len(mapped['traces']),1);self.assertEqual(len(trace['requests']),4)
        self.assertEqual(trace['split'],'training');self.assertFalse(trace['not_for_training'])
        self.assertEqual(trace['requests'][0]['native_request_id'],'r:0')
        self.assertNotEqual(trace['requests'][0]['request_id'],'r:0')
        self.assertEqual([s['phase'] for s in trace['requests'][0]['segments']],['prepare','prefill','decode','finalize'])
        model=fit_traces(mapped['traces'])
        self.assertEqual(model['coverage']['profiles'],2)
        self.assertEqual(model['coverage']['training_batches'],1)

    def test_validation_rejected_by_default_and_never_relabelled_by_diagnostics(self):
        self.plan.update(phase='predictor_validation',split='validation');self.save()
        with self.assertRaisesRegex(ValueError,'validation cannot enter training'):map_flow_collection(self.path)
        mapped=map_flow_collection(self.path,allow_validation_diagnostics=True)
        self.assertEqual(mapped['traces'][0]['split'],'validation')
        self.assertTrue(mapped['traces'][0]['not_for_training'])
        with self.assertRaisesRegex(ValueError,'training traces'):fit_traces(mapped['traces'])

    def test_only_prospectively_frozen_training_plan_may_derive_split(self):
        self.plan['created_ns']=self.raw['startup_begin_ns'];self.save()
        with self.assertRaisesRegex(ValueError,'before native startup'):map_flow_collection(self.path)
        self.plan['created_ns']=0;self.raw['request_records'][0]['split']='validation';self.save()
        with self.assertRaisesRegex(ValueError,'split differs'):map_flow_collection(self.path)

    def test_full_offered_cohort_and_route_identity_required(self):
        self.raw['request_records'].pop();self.save()
        with self.assertRaisesRegex(ValueError,'Full offered cohort'):map_flow_collection(self.path)

    def test_input_held_out_by_profile_plan_cannot_become_training(self):
        profile=json.loads(self.profile_path.read_text());profile['input_assignments'][0]['view']='unseen_input'
        profile.pop('plan_sha256');profile['plan_sha256']=digest(profile)
        self.write(self.profile_path,profile);self.plan['profile_plan']=receipt(self.profile_path);self.save()
        with self.assertRaisesRegex(ValueError,'frozen training domain'):map_flow_collection(self.path)

    def test_actual_START_PHASE_mismatch_or_unmatched_extra_rejected(self):
        original=copy.deepcopy(self.raw)
        self.raw['request_records'][0]['execution_events'][1]['perf_ns']+=1;self.save()
        with self.assertRaisesRegex(ValueError,'phase does not match'):map_flow_collection(self.path)
        self.raw=original;self.raw['request_records'][0]['execution_events'].append(copy.deepcopy(self.raw['request_records'][0]['execution_events'][-1]));self.save()
        with self.assertRaises(ValueError):map_flow_collection(self.path)

    def test_same_alias_overlap_rejected_without_changing_semantic_evidence(self):
        # The primitive is patched only to isolate cross-request cohort checks;
        # actual emitted-boundary validation has its separate tests above.
        from tools.predictor_validation_20261027 import flow_profiles
        original=flow_profiles._request
        def overlapping(*args,**kwargs):
            row=original(*args,**kwargs)
            if row['native_request_id']=='r:1':row['handler_start_ns']=20_000_000
            return row
        with mock.patch.object(flow_profiles,'_request',side_effect=overlapping):
            with self.assertRaisesRegex(ValueError,'FIFO and nonoverlapping'):map_flow_collection(self.path)

    def test_invalid_NVML_sample_or_owned_PID_rejected(self):
        for change in ({'owned_used_mib':float('nan')},{'foreign_pids':[999]},{'owned_aliases':['text#0']}):
            original=copy.deepcopy(self.raw)
            self.raw['board_samples'][5].update(change);self.save()
            with self.assertRaises(ValueError):map_flow_collection(self.path)
            self.raw=original

    def test_loading_resident_subset_allowed_only_before_ready(self):
        self.raw['board_samples'][0].update(owned_used_mib=0,processes_mib={},owned_aliases=[]);self.save()
        mapped=map_flow_collection(self.path)
        self.assertTrue(mapped['mapping_report']['all_resident_telemetry_after_READY_verified'])
        self.raw['board_samples'][1].update(owned_used_mib=0,processes_mib={},owned_aliases=[]);self.save()
        with self.assertRaisesRegex(ValueError,'after READY'):map_flow_collection(self.path)

    def test_snapshot_and_invocation_tampering_rejected(self):
        self.write(self.root/'invocation.json',{})
        with self.assertRaisesRegex(ValueError,'Prelaunch invocation'):map_flow_collection(self.path)
        self.save();(self.root/'worker.py').write_text('# tampered\n')
        with self.assertRaisesRegex(ValueError,'receipt'):map_flow_collection(self.path)

    def test_resource_history_preserves_full_startup_and_drain_without_safe_factor(self):
        mapped=map_flow_collection(self.path);e=mapped['resource_evidence']
        self.assertAlmostEqual(e['startup_joules'],.45)
        self.assertAlmostEqual(e['measurement_joules'],37.5)
        self.assertAlmostEqual(e['drain_joules'],2)
        self.assertEqual(e['raw_all_sampled_owned_peak_mib'],400)
        self.assertFalse(e['safe_factor_applied']);self.assertEqual(len(e['stage_history']),4)

    def test_unclean_shutdown_is_not_complete_training(self):
        self.raw['shutdown']['text#0']['forced']=True;self.save()
        with self.assertRaisesRegex(ValueError,'clean shutdown'):map_flow_collection(self.path)

    def test_native_configuration_change_cannot_borrow_shared_execution_identity(self):
        config={'schema':'gsdeploy.resource_configuration.v1','device':'0','mode':'request_admission',
            'allocations':{a:{'max_inflight':1} for a in RESIDENTS},
            'groups':{'gpu':{'aliases':sorted(RESIDENTS),'max_active':1}}}
        self.plan['candidates'][0]['resource_configuration']=config;self.raw['resource_configuration']=config;self.save()
        with self.assertRaisesRegex(ValueError,'distinct execution-context'):map_flow_collection(self.path)

    def test_public_input_plan_hash_and_telemetry_order_cannot_drift(self):
        profile=json.loads(self.profile_path.read_text());profile['extra']='changed'
        self.write(self.profile_path,profile);self.plan['profile_plan']=receipt(self.profile_path);self.save()
        with self.assertRaisesRegex(ValueError,'input plan self hash'):map_flow_collection(self.path)

    def test_full_handler_and_response_must_fit_original_bounded_drain(self):
        self.raw['request_records'][-1]['completion_time_ns']=self.raw['deadline_ns']+110_000_000
        self.raw['observation_end_ns']=900_000_000
        # Add real bracketing telemetry to isolate the completion-time rule.
        sample=copy.deepcopy(self.raw['board_samples'][-1]);sample['time_ns']=900_000_000
        self.raw['board_samples'].append(sample);self.save()
        with self.assertRaisesRegex(ValueError,'bounded drain'):map_flow_collection(self.path)


if __name__=='__main__':unittest.main()
