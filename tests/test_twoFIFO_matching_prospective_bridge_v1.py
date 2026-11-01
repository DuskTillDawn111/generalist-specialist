"""CPU synthetic law/authentication boundaries; never native fit or collection."""
import copy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_fresh181_execute_v1 as fresh
from tools.predictor_validation_20261027 import twoFIFO_fresh181_prefix_v1 as prefix
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt
from tests.test_twoFIFO_fresh181_execute_v1 import Runtime,Clock
from tests.test_twoFIFO_matching_numerics_v1 import bundle_fixture

PREP=Path('artifacts/predictor-twoFIFO-fresh181-v1-20261029-cpu01-prepared/manifest.json')


def law_fixture(contract,blocks):
    joint=m.joint.joint;vectors=[]
    slack=dict(reader_slack_ns=0,collector_dequeue_slack_ns=0,collector_processing_slack_ns=0)
    for block in blocks:
        for key in m.required_keys(contract):
            phase='ocr_inference' if key[0]=='ocr#0' else 'prefill'
            def mark(identity,at,action):
                return dict(identity=identity,reference_work_at=joint.fraction_record(F(at)),
                    protocol_slacks=slack,action=action,phase=phase if action=='phase' else None,
                    query_complete=None,event_reference_phase_id=None,clear_scope=None)
            vector=dict(public_key=list(key),training_block_id=block,
                vector_id=block+':'+':'.join(key),nodes=[dict(macro_id='CPU-synthetic',phase=phase,
                    reference_work_ns=joint.fraction_record(F(10)),marks=[mark('begin',0,'phase'),mark('RESULT',10,'result')])],
                ready_protocol_slacks=slack,transport_constraints=dict(enqueue_after_arrival_ns=0,sender_turn_extra_ns=0,
                    submit_to_write_record_ns=0,observed_submit_to_START_ns=1,previous_READY_minus_submit_ns=-1))
            vectors.append(vector)
    return dict(schema=joint.SCHEMA,features=list(m.response.FEATURES),coefficients=[0.]*12,
        blocks=blocks,heldout_block_id=None,
        vectors=vectors,measured_request_vectors=390)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name);cls.root=root
        cls.template=checked(checked(receipt(PREP))['attempts'][0]['plan']);cls.plan=copy.deepcopy(cls.template)
        cls.fit_manifest=prefix.save(root/'fit-manifest.json',dict(executor_manifest=cls.plan['matching_producer']))
        cls.fit_ref=prefix.save(root/'saved-fit.json',dict(CPU_SYNTHETIC_ONLY=True))
        collection=root/'attempt'/'collection'
        for key,name in [('prefix_path','prefix.json'),('gate_request_path','gate.json'),('seal_path','seal.json')]:
            cls.plan['forecast_barrier'][key]=str(collection/name)
        cls.plan['forecast_barrier'].update(model_sources=[cls.fit_ref],generator_sources=[receipt(m.__file__)])
        cls.plan_ref=prefix.save(root/'attempt'/'plan.json',cls.plan)
        cls.manifest_ref=prefix.save(root/'manifest.json',dict(CPU_SYNTHETIC_ONLY=True,attempts=[dict(plan=cls.plan_ref)]))
        context=checked(cls.plan['context']);clock=Clock()
        registry={owner:context['worker_bindings'][owner.split('#')[0]] for owner in fresh.OWNERS}
        runtime=Runtime(clock,registry);pool={r['pool_id']:r for r in (m.response.json.loads(l)
            for l in Path(cls.plan['prepared_pool']['path']).read_text().splitlines())}
        rows=fresh.initial_priming_rows(cls.plan,pool)
        for row in rows:
            row['scheduled_time_ns']=clock.clock();runtime.enqueue(row,{});runtime.drain([row],300)
        snapshot=fresh.history_snapshot(runtime,dict(priming_request_records=rows,request_records=[]),clock.clock)
        cls.epoch=clock.clock()+60_000_000_000
        cls.prefix_ref=prefix.make_prefix(cls.plan_ref,cls.plan,cls.manifest_ref,snapshot,cls.epoch,
            {o:[dict(CPU_TEST_ONLY=True)] for o in fresh.OWNERS},cls.plan['forecast_barrier']['prefix_path'],backend='CPU_TEST_ONLY')
        cls.gate_ref=prefix.gate_request(cls.prefix_ref,cls.plan_ref,cls.plan,cls.manifest_ref,
            cls.plan['forecast_barrier']['gate_request_path'],allow_cpu_fixture=True)
        cls.gate=checked(cls.gate_ref);cls.prefix_doc=checked(cls.prefix_ref);cls.contract=checked(cls.plan['public_contract'])
        cls.bundle=bundle_fixture(root/'bundle');cls.law=law_fixture(cls.contract,sorted({p['capture_id'] for p in cls.bundle['plans']}))
        cls.fitted=dict(features=list(m.response.FEATURES),denominator=cls.bundle['denominator'],response_law_compilation_eligible=True)
        cls.timings={}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_fifteen_real_plan_axes_context_priming_and_physical_knobs_are_explicit(self):
        result=m.physical_axes(self.bundle['plans'],self.plan,self.prefix_doc)
        self.assertEqual(len(result['all15_donor_axes']),15);self.assertFalse(result['old_physical_identity_hash_reused'])
        self.assertTrue(result['full_context_bytes_equal']);self.assertTrue(result['changed_offered_axes'])
        for mutate in [lambda p:p['runtime_config'].update(idle_residency=True),
            lambda p:p['runtime_config'].update(secret_gpu_knob=2),lambda p:p['priming'][0].update(owner='foreign'),
            lambda p:p['runtime_policy'].update(cap=2),lambda p:p.update(sources=[])]:
            plan=copy.deepcopy(self.plan);mutate(plan)
            with self.assertRaises(ValueError):m.physical_axes(self.bundle['plans'],plan,self.prefix_doc)

    def test_all_sixteen_keys_must_exist_in_each_of_three_whole_blocks(self):
        support=m.support(self.law,self.contract);self.assertEqual(len(support['checks']),48)
        self.assertTrue(support['complete_all_keys_in_every_donor_block'])
        incomplete=copy.deepcopy(self.law);incomplete['vectors'].pop()
        self.assertFalse(m.support(incomplete,self.contract)['complete_all_keys_in_every_donor_block'])
        incomplete['heldout_block_id']='training-block-4'
        with self.assertRaises(ValueError):m.support(incomplete,self.contract)

    def test_real_compiler_retains181_times12_and_rebinds_every_actual_owned_producer(self):
        begin=time.perf_counter()
        draws,selections=m.generated.compile_case_draws(self.gate['public_requests'],self.law)
        rebound=m.bind_owned_producers(draws,self.prefix_doc['seed'])
        self.timings['compile181_times12_and_bind_seconds']=time.perf_counter()-begin
        self.assertEqual(len(rebound),12);self.assertEqual(sum(d['weight'] for d in rebound),1)
        self.assertEqual(len(selections),12)
        for before,after in zip(draws,rebound):
            self.assertEqual(before['inputs']['transport_draws'],after['inputs']['transport_draws'])
            self.assertEqual(before['inputs']['rates'],after['inputs']['rates'])
            self.assertEqual(sum(map(len,after['inputs']['plans'].values())),181)
            for key,plans in after['inputs']['plans'].items():
                for plan in plans:
                    expected=self.prefix_doc['owner_producers'][key[0]]
                    self.assertEqual(plan.ready_producer_id,expected)
                    self.assertTrue(all(mark.producer_id==expected for node in plan.macros for mark in node.marks))
                    self.assertIn(plan,after['inputs']['protocol_slacks'])

    def test_real_original_engine_forecasts_future181_without_prefix_or_world_loss(self):
        draws,_=m.generated.compile_case_draws(self.gate['public_requests'],self.law)
        draws=m.bind_owned_producers(draws,self.prefix_doc['seed'])
        begin=time.perf_counter()
        result=m.decode(m.propagate(self.gate['public_requests'],draws,self.prefix_doc['seed'],self.contract,self.gate))
        self.timings['synthetic_single_macro_full181_times12_propagation_seconds']=time.perf_counter()-begin
        value=result['ensemble'];self.assertEqual(value['offered_requests'],181)
        self.assertEqual(value['world_unknown_probability_mass'],0);self.assertEqual(len(value['draws']),12)
        self.assertEqual(value['utility']['unknown_probability_mass'],1)
        for row,projection in zip(value['draws'],result['complete_draw_projection']):
            self.assertEqual(row['status'],'GENERATED_WORLD');self.assertTrue(row['world_sha256'])
            self.assertTrue(row['forecast_sha256']);self.assertTrue(row['query_fingerprint_sha256'])
            self.assertEqual(len(projection['requests']),181)
            self.assertTrue(all(r['completion_time_ns']>=p['arrival_ns'] for r,p in zip(projection['requests'],self.gate['public_requests'])))
            self.assertGreater(row['RESULT_receipts']['measurement:180'],self.epoch+80_000_000_000)
        self.assertFalse(value['original_stability_evaluated'])

    def test_unqualified_fit_never_compiles_law_and_retains_allUNKNOWN_mass(self):
        fitted=dict(self.fitted,response_law_compilation_eligible=False)
        with patch.object(m.joint,'build_joint_law') as builder,patch.object(m,'_raws') as raw_reader:
            result=m.replay_bundle(self.bundle,fitted,self.plan,self.prefix_doc,self.gate)
            builder.assert_not_called();raw_reader.assert_not_called()
        value=m.decode(result['forecast'])['ensemble']
        self.assertEqual(len(value['draws']),12);self.assertEqual(value['world_unknown_probability_mass'],1)
        self.assertFalse(result['source_supported_hypothesis_available'])
        self.assertTrue(all(len(r['all_public_request_ids_retained'])==181 for r in value['draws']))

    def test_real_engine_uses_all_four_seed_stations_and_prior_issued_counters(self):
        draws,_=m.generated.compile_case_draws(self.gate['public_requests'],self.law)
        draws=m.bind_owned_producers(draws,self.prefix_doc['seed'])
        for draw in draws[1:]:draw['inputs']=m.prospective.ensemble.UnknownForecastInputs('CPU_TEST_OTHER_WORLDS_UNAVAILABLE')
        actual=m.prospective.ensemble.forecast;seen=[];seed=self.prefix_doc['seed']
        def capture(*args,**kwargs):
            self.assertEqual(kwargs['settled_initial_state'],seed)
            result=actual(*args,**kwargs);seen.append(m.decode(result['world']));return result
        with patch.object(m.prospective.ensemble,'forecast',side_effect=capture):
            result=m.decode(m.propagate(self.gate['public_requests'],draws,seed,self.contract,self.gate))
        self.assertEqual(len(seen),1);payload=seen[0]['continuation_state']['payload'];history=payload['initial_history']
        for key in ('initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state','producer_next','owner_issued','sequence'):
            self.assertEqual(history[key],seed[key])
        self.assertEqual(payload['sequence'],seed['sequence']+181*3)
        for owner,producer in seed['owner_producers'].items():
            count=sum(r['owner']==owner for r in self.gate['public_requests'])
            self.assertEqual(payload['producer_next'][producer],seed['producer_next'][producer]+count*3)
            self.assertEqual(payload['owner_issued'][owner],seed['owner_issued'][owner]+count*3)
        self.assertEqual(result['ensemble']['world_unknown_probability_mass'],F(11,12))

    def test_active_session_authenticates_once_replays_report_and_rejects_forged_or_expired_cache(self):
        continuation=prefix.save(self.root/'continuation.json',dict(CPU_SYNTHETIC_ONLY=True))
        fitted=dict(self.fitted,response_law_compilation_eligible=False)
        intake=dict(CPU_SYNTHETIC_ONLY=True,sources=[])
        report=dict(manifest=self.fit_manifest,continuation=continuation,fit=fitted,intake_evidence=intake,
            sources=[self.fit_manifest,continuation])
        def verified(manifest,continuation,callback):return callback(self.bundle,intake)
        path=self.root/'typed-report.json'
        with patch.object(m.response,'audit_fit',return_value=dict(report=report)) as fit_audit,\
                patch.object(m.response,'with_verified_bundle',side_effect=verified) as native,\
                patch.object(fresh,'audit_plan',return_value=self.plan),\
                patch.object(m.response.original,'fitted_rows',side_effect=AssertionError('NO_FIT_ALLOWED')):
            with m.open_law_session(self.fit_ref['path']) as session:
                self.timings['CPU_MOCK_AUTH_SESSION']=session.timings
                produced=m.produce_from_session(session,self.gate_ref['path'],path,allow_cpu_fixture=True)
                saved=checked(produced);self.assertFalse(saved['source_supported_native_hypothesis'])
                self.assertEqual(m.audit_from_session(session,path),saved)
                self.assertEqual(fit_audit.call_count,1);self.assertEqual(native.call_count,1)
                self.assertTrue(saved['CPU_TEST_ONLY']);self.assertFalse(saved['replay']['source_supported_hypothesis_available'])
                with self.assertRaises(ValueError):m.audit_from_session(session,path,require_native=True)
                original=path.read_bytes();mutated=copy.deepcopy(saved);mutated['source_supported_native_hypothesis']=True
                path.write_text(m.json.dumps(mutated))
                with self.assertRaisesRegex(ValueError,'replay every'):m.audit_from_session(session,path)
                path.write_bytes(original)
                cached=session._json;session._json=cached+' '
                # Whitespace alone preserves the canonical content hash;
                # changing a cached law field cannot preserve that seal.
                changed=m.json.loads(cached);changed['unknown_reason']='foreign';session._json=m.json.dumps(changed)
                with self.assertRaisesRegex(ValueError,'cache contents changed'):session.checked_payload()
                session._json=cached
            with self.assertRaisesRegex(ValueError,'inactive or expired'):session.checked_payload()
        with self.assertRaisesRegex(ValueError,'context manager'):m.AuthenticatedLawSession(None,{},[],None,{})

    def test_missing_single_block_key_preserves_fullUNKNOWN_and_never_selects_partial_support(self):
        law=copy.deepcopy(self.law);law['vectors'].pop()
        with patch.object(m.joint,'build_joint_law',return_value=law),patch.object(m,'_raws',return_value=[]),\
                patch.object(m.generated,'compile_case_draws') as compile_draws:
            result=m.replay_bundle(self.bundle,self.fitted,self.plan,self.prefix_doc,self.gate)
            compile_draws.assert_not_called()
        self.assertEqual(m.decode(result['forecast'])['ensemble']['world_unknown_probability_mass'],1)
        self.assertFalse(result['source_supported_hypothesis_available'])

    def test_arbitrary_receipt_never_enters_prefix_or_numerical_replay(self):
        with patch.object(prefix,'audit_gate') as gate,\
                patch.object(m.response.original,'fitted_rows') as fitter,patch.object(m,'replay_bundle') as replay:
            with self.assertRaisesRegex(ValueError,'Saved new-schema complete training fit required'):
                m._replay(self.fit_ref,self.gate_ref,allow_cpu_fixture=True)
            gate.assert_not_called();fitter.assert_not_called();replay.assert_not_called()

    def test_live_session_source_change_and_restore_is_not_a_valid_cached_model(self):
        continuation=prefix.save(self.root/'guard-continuation.json',dict(CPU_SYNTHETIC_ONLY=True))
        intake=dict(CPU_SYNTHETIC_ONLY=True,sources=[])
        report=dict(manifest=self.fit_manifest,continuation=continuation,
            fit=dict(self.fitted,response_law_compilation_eligible=False),intake_evidence=intake,sources=[])
        path=Path(self.fit_ref['path']);original=path.read_bytes()
        with patch.object(m.response,'audit_fit',return_value=dict(report=report)),\
                patch.object(m.response,'with_verified_bundle',side_effect=lambda a,b,callback:callback(self.bundle,intake)):
            with self.assertRaises(ValueError):
                with m.open_law_session(path) as session:
                    path.write_bytes(original+b' ');path.write_bytes(original);session.checked_payload()
        self.assertEqual(receipt(path),self.fit_ref)


if __name__=='__main__':unittest.main()
