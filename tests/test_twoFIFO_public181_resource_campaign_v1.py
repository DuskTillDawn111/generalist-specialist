"""CPU-only preparation and adversarial public181 resource interface checks."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_public181_resource_campaign_v1 as m
from tools.predictor_validation_20261027.macro_six_term_bridge_v1 import whole_score
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode

ROOT=Path(__file__).resolve().parents[1]
FRESH=ROOT/'artifacts/predictor-twoFIFO-fresh181-v1-20261029-cpu01-prepared/manifest.json'


class Public181ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.manifest_ref=m.prepare(FRESH,'9f645b2048e341cc8875975c6e90a321',cls.root/'campaign',
                                  precohort_lead_seconds=300)
        cls.manifest=m.checked(cls.manifest_ref)
        cls.fixture=cls.root/'CPU_ONLY_NOT_NATIVE.json'
        cls.fixture.write_text(json.dumps(dict(CPU_FIXTURE_NOT_NATIVE=True)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self):
        rows=[]
        for index,attempt in enumerate(self.manifest['training_attempts']):
            plan=m.checked(attempt['plan'])
            rows.append(dict(schema=m.SCHEMA+'.training_joint_tuple',split='training',
                **{field:attempt[field] for field in ('attempt_id','capture_id','execution_id','candidate_id','plan')},
                collection_protocol_identity=self.manifest['collection_protocol_identity'],
                public_case_sha256=plan['public_case_sha256'],
                joint_tuple=dict(startup_joules=36_000.,premeasurement_owned_sampled_peak_mib=1000.+index,
                    measurement_through_drain_owned_sampled_peak_mib=2000.+index,measurement_board_joules=9000.),
                native_review=m.receipt(self.fixture),raw=m.receipt(self.fixture)))
        return rows

    def test_exact_eight_training_plus_four_disjoint_validation_preserves_every_public_axis(self):
        manifest=m.audit(self.manifest_ref['path'])
        self.assertEqual((manifest['train_attempts'],manifest['validation_attempts'],manifest['training_measured_requests']),
                         (8,4,1448))
        ids={a['execution_id'] for a in manifest['training_attempts']}
        validation={a['execution_id'] for a in manifest['independent_validation_references']}
        matching=m.checked(m.checked(manifest['fresh_validation_manifest'])['matching_producer'])
        self.assertEqual(len(ids),8)
        self.assertFalse(ids&validation)
        self.assertFalse(ids&{a['execution_id'] for a in matching['attempts']})
        for attempt in manifest['training_attempts']:
            plan=m.checked(attempt['plan']);parent=m.checked(plan['reference_validation_plan'])
            self.assertEqual(plan['public_case'],m.public_case_payload(parent))
            self.assertEqual(plan['cases'],parent['cases'])
            self.assertEqual(plan['context'],parent['context'])
            self.assertEqual(plan['runtime_config'],parent['runtime_config'])
            self.assertEqual(plan['priming_semantics'],parent['priming_semantics'])
            self.assertEqual(len(plan['priming']),18)
            self.assertEqual([{k:v for k,v in r.items() if k!='request_id'} for r in plan['priming']],
                [{k:v for k,v in r.items() if k!='request_id'} for r in parent['priming']])
            self.assertNotEqual(plan['priming'][0]['request_id'],parent['priming'][0]['request_id'])
            self.assertIsNone(plan['native_executor_entrypoint'])
            self.assertEqual((plan['measurement_ns'],plan['drain_ns']),(90_000_000_000,120_000_000_000))
            self.assertEqual(plan['collection_contract']['global_admission_cap'],None)

    def test_new_source_domain_budget_is_explicit_and_old_backend_not_promoted(self):
        doc=self.manifest;transfer=doc['source_transfer']
        self.assertEqual(doc['target_precohort_lead_seconds'],300)
        self.assertEqual(transfer['target_precohort_lead_seconds'],300)
        self.assertEqual(transfer['parent_reference_precohort_lead_seconds'],60)
        self.assertFalse(transfer['old_matching_physical_identity_reused'])
        self.assertFalse(transfer['native_source_transfer_authenticated'])
        self.assertFalse(transfer['parent_validation_native_execution_reusable'])
        self.assertTrue(doc['validation_native_backend_requires_new_preparation'])
        self.assertEqual(transfer['planned_worker_registry_sha256'],m.digest(transfer['planned_worker_registry']))
        self.assertTrue(doc['collection_contract']['startup_includes_priming_and_precohort_wait'])

    def test_rehashed_plan_cannot_change_horizon_route_context_priming_or_role(self):
        plan_path=Path(self.manifest['training_attempts'][0]['plan']['path'])
        original_plan=plan_path.read_bytes();manifest_path=Path(self.manifest_ref['path'])
        original_manifest=manifest_path.read_bytes()
        def changed_route(p):
            p['cases'][0]['arrivals'][0]['owner']='foreign-owner'
        mutations=[lambda p:p.update(measurement_ns=10_000_000_000),changed_route,
            lambda p:p['context'].update(sha256='a'*64),lambda p:p['priming'].reverse(),
            lambda p:p.update(split='validation'),lambda p:p.update(precohort_lead_seconds=60),
            lambda p:p['collection_contract'].update(global_admission_cap=2)]
        for mutate in mutations:
            try:
                plan=json.loads(original_plan);mutate(plan)
                plan_path.write_text(json.dumps(plan))
                manifest=json.loads(original_manifest)
                manifest['training_attempts'][0]['plan']=m.receipt(plan_path)
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(mutation=mutate),self.assertRaises(ValueError):m.audit(manifest_path)
            finally:
                plan_path.write_bytes(original_plan);manifest_path.write_bytes(original_manifest)

    def test_overwrite_bad_namespace_or_implicit_budget_cannot_prepare(self):
        with self.assertRaises(ValueError):
            m.prepare(FRESH,'9f645b2048e341cc8875975c6e90a321',self.root/'campaign',precohort_lead_seconds=300)
        for namespace,budget in [('not-fresh',300),('f'*32,None),('f'*32,True),('f'*32,float('inf'))]:
            with self.subTest(namespace=namespace,budget=budget),self.assertRaises(ValueError):
                m.prepare(FRESH,namespace,self.root/'bad-output',precohort_lead_seconds=budget)

    def test_resource_catalog_keeps_two_complete_joint_four_candidate_blocks(self):
        output=m.paired_resource_draws(self.rows(),self.manifest)
        self.assertEqual(len(output['draws']),2)
        self.assertEqual([r['weight'] for r in output['draws']],[dict(numerator=1,denominator=2)]*2)
        expected={a['candidate_id'] for a in self.manifest['training_attempts']}
        self.assertTrue(all(set(r['candidates'])==expected for r in output['draws']))
        self.assertEqual(sum(len(r['candidates']) for r in output['draws']),8)
        self.assertEqual(output['native_authentication'],'NOT_ESTABLISHED_BY_STRUCTURAL_HELPER')
        self.assertFalse(output['queue_resource_joint_law_qualified'])

    def test_every_donor_role_is_checked_before_first_resource_label(self):
        rows=self.rows()
        rows[0]['joint_tuple']=object()
        rows[-1]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Every donor role'):
            m.paired_resource_draws(rows,self.manifest)

    def test_missing_foreign_duplicate_or_future_tuple_cannot_fill_training(self):
        for mode in ('missing','duplicate','heldout','case','protocol','term','extra_future','nonfinite','bool'):
            rows=self.rows()
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif mode=='heldout':rows[0]['execution_id']=self.manifest['independent_validation_references'][0]['execution_id']
            elif mode=='case':rows[0]['public_case_sha256']='e'*64
            elif mode=='protocol':rows[0]['collection_protocol_identity']='e'*64
            elif mode=='term':rows[0]['joint_tuple'].pop('startup_joules')
            elif mode=='extra_future':rows[0]['heldout_future_samples']=[]
            elif mode=='nonfinite':rows[0]['joint_tuple']['startup_joules']=float('nan')
            elif mode=='bool':rows[0]['joint_tuple']['startup_joules']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.paired_resource_draws(rows,self.manifest)

    def test_bridge_uses_original_max_and_single_startup_amortization(self):
        catalog=m.paired_resource_draws(self.rows(),self.manifest)
        candidate_id=next(iter(catalog['draws'][0]['candidates']))
        owners=list(m.fresh.OWNERS)
        bridge=m.resource_bridge_payload(catalog['draws'][0],candidate_id,window_start_ns=0,resident_owners=owners)
        self.assertEqual(bridge['measurement_board_energy_joules'],9000.)
        self.assertEqual(bridge['startup_energy_joules'],36_000.)
        self.assertFalse(bridge['measurement_energy_includes_drain'])
        plan=dict(window_seconds=90,startup_amortization_seconds=3600,weights={k:1/6 for k in 'QTLPME'},
            aliases={o:dict(family='fixture',supported_tasks=['text']) for o in owners},workload_rps={'text':1.},
            latency_reference_ms={'text':1.},memory_budget_mib=10_000.,energy_reference_watts=100.,memory_safe_factor=1.1)
        public=[dict(request_id='cpu',owner=owners[0],task='text',pool_id='CPU_FIXTURE',arrival_ns=0)]
        reference=dict(original_window=[F(0),F(90_000_000_000)],original_drain_end_ns=F(210_000_000_000),
            cutoff_ns=F(210_000_000_000),RESULT_receipts={'cpu':F(1_000_000)},unfinished_request_ids=[],offered_requests=1)
        score=whole_score(encode(reference),public,plan,{'fixture':{'text':0.}},bridge)['score']
        values=catalog['draws'][0]['candidates'][candidate_id]['joint_tuple']
        self.assertAlmostEqual(score['terms']['E'],(9000/90+36_000/3600)/100)
        self.assertAlmostEqual(score['terms']['M'],1.1*max(values['premeasurement_owned_sampled_peak_mib'],
            values['measurement_through_drain_owned_sampled_peak_mib'])/10_000)

    def test_mean_of_donor_maxima_cannot_be_replaced_by_max_of_means(self):
        rows=self.rows();candidate=rows[0]['candidate_id']
        pair=[r for r in rows if r['candidate_id']==candidate]
        pair[0]['joint_tuple'].update(premeasurement_owned_sampled_peak_mib=100.,measurement_through_drain_owned_sampled_peak_mib=0.)
        pair[1]['joint_tuple'].update(premeasurement_owned_sampled_peak_mib=0.,measurement_through_drain_owned_sampled_peak_mib=100.)
        output=m.paired_resource_draws(rows,self.manifest)
        full=[max(r['candidates'][candidate]['joint_tuple']['premeasurement_owned_sampled_peak_mib'],
                  r['candidates'][candidate]['joint_tuple']['measurement_through_drain_owned_sampled_peak_mib'])
              for r in output['draws']]
        self.assertEqual(sum(full)/2,100.)
        self.assertNotEqual(sum(full)/2,max(50.,50.))

    def test_only_completed_prefix_polls_can_initialize_memory(self):
        poll=dict(poll_begin_ns=1,poll_end_ns=3,sample_ns=2,owned_mib=100.)
        self.assertEqual(m.audit_causal_prefix_polls([poll],cutoff_ns=3,forecast_origin_ns=10),100.)
        for change in (dict(poll_end_ns=4),dict(sample_ns=11),dict(poll_begin_ns=-1),dict(poll_end_ns=3.0)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.audit_causal_prefix_polls([poll|change],cutoff_ns=3,forecast_origin_ns=10)

    def test_forecast_request_rejects_new_candidate_and_has_no_target_raw_input(self):
        request=m.forecast_request(self.manifest,self.manifest['training_attempts'][0]['candidate_id'],
            queue_model_sources=[m.receipt(self.fixture)],resource_model_source=m.receipt(self.fixture))
        self.assertEqual(request['memory_origin']['basis'],'training_donor_joint_startup_tuple')
        self.assertEqual(len(request['public_case']['cases'][0]['arrivals']),181)
        self.assertNotIn('target_raw',request)
        with self.assertRaises(ValueError):m.forecast_request(self.manifest,'new-DFS-candidate',
            queue_model_sources=[],resource_model_source=m.receipt(self.fixture))


if __name__=='__main__':unittest.main()
