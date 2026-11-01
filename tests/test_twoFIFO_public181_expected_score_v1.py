"""CPU transfer and marginal counterexamples; never native authentication."""
import copy
from fractions import Fraction as F
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import json

from tools.predictor_validation_20261027 import twoFIFO_public181_expected_score_v1 as m
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=ROOT/'artifacts/predictor-public181-resource-owned-producer-v1-20261029-cpu02-prepared/manifest.json'


class ExpectedProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=m.resource_executor.audit(RESOURCE)
        donor=checked(cls.manifest['attempts'][0]['plan']);cls.donor=donor
        cls.plan=copy.deepcopy(checked(donor['reference_validation_plan']))
        cls.plan['forecast_barrier'].update(lead_seconds=300)
        cls.prefix=dict(actual_priming_semantics_sha256=donor['priming_semantics']['sha256'],
            prefix_snapshot={'snapshot_time_ns':100})
        cls.epoch=300_000_000_100
        cls.gate=dict(forecast_origin_ns=cls.epoch,measurement_end_ns=cls.epoch+90*10**9,
            fixed_drain_deadline_ns=cls.epoch+210*10**9,
            public_requests=[dict(request_id=a['request_id'],owner=a['owner'],task=a['task'],pool_id=a['pool_id'],
                arrival_ns=cls.epoch+a['arrival_ns']) for a in cls.plan['cases'][0]['arrivals']])

    def test_source_bound_different_closure_never_implicitly_proves_transfer(self):
        result=m.transfer_axes({'manifest_document':self.manifest},self.plan,self.prefix,self.gate)
        self.assertFalse(result['resource_marginal_transfer_qualified'])
        self.assertEqual(len(result['all8_axes']),8)
        self.assertIn('TRANSFER_AXIS_DIFFERS:READY_settle_budget',result['unknown_reasons'])
        self.assertIn('UNQUALIFIED_EPOCH_ANCHOR_TRANSFER_SNAPSHOT_VS_POSTSNAPSHOT_CLOCK',result['unknown_reasons'])
        self.assertIn('UNVALIDATED_STARTUP_CPU_PREDICTION_AUDIT_ACTIVITY_EFFECT_ON_BOARD_ENERGY_AND_P0',result['unknown_reasons'])
        self.assertFalse(result['protocol_hash_equality_used_as_physical_proof'])

    def test_actual_copy_anchor_excess_is_quantified_and_blocks_resource_transfer(self):
        gate=copy.deepcopy(self.gate);gate['forecast_origin_ns']+=1234567
        result=m.transfer_axes({'manifest_document':self.manifest},self.plan,self.prefix,gate)
        self.assertEqual(result['snapshot_copy_anchor_excess_ns'],1234567)
        self.assertIn('TRANSFER_AXIS_DIFFERS:actual_snapshot_lead',result['unknown_reasons'])

    def test_worker_runtime_route_or_catalog_change_cannot_hide_under_same_lead(self):
        changes=[lambda p:p['runtime_config'].update(max_pending_per_model=2),
            lambda p:p['cases'][0]['arrivals'][0].update(arrival_ns=1),
            lambda p:p.update(prepared_pool={'path':'/foreign','sha256':'0'*64})]
        for change in changes:
            plan=copy.deepcopy(self.plan);change(plan)
            result=m.transfer_axes({'manifest_document':self.manifest},plan,self.prefix,self.gate)
            self.assertFalse(result['resource_marginal_transfer_qualified'])
            self.assertTrue(any('TRANSFER_AXIS_DIFFERS' in r for r in result['unknown_reasons']))

    def rows(self):
        rows=[]
        blocks=list(dict.fromkeys(a['capture_id'] for a in self.manifest['attempts']))
        for item in self.manifest['attempts']:
            first=item['capture_id']==blocks[0]
            rows.append(dict(**{k:item[k] for k in ('attempt_id','capture_id','candidate_id','execution_id')},
                joint_tuple=dict(startup_joules=3600.,premeasurement_owned_sampled_peak_mib=100. if first else 0.,
                    measurement_through_drain_owned_sampled_peak_mib=0. if first else 100.,measurement_board_joules=9000.),
                raw={'path':'/CPU_ONLY'},native_review={'path':'/CPU_ONLY'},resource_status='CPU_FIXTURE_ONLY',
                actual_native_quality=dict(measured_Q=.1 if first else .3,
                    original_offered_rates=self.plan['runtime_config']['workload_rps'])))
        return {'rows':rows}

    def test_two_resource_blocks_keep_per_donor_max_and_original_once_startup_energy(self):
        marginals,worlds=m.resource_marginals(self.rows(),self.plan,self.gate,
            {'resource_marginal_transfer_qualified':True},{'CPU_ONLY':True},[{'CPU_ONLY':True}])
        contract=checked(self.plan['public_contract'])
        expected_M=contract['memory_safe_factor']*100/contract['memory_budget_mib']
        expected_E=(100+1)/contract['energy_reference_watts']
        self.assertEqual(len(worlds),2);self.assertEqual(sum(w['weight'] for w in worlds),F(1))
        self.assertAlmostEqual(marginals['M']['distribution']['expected_value'],expected_M)
        self.assertAlmostEqual(marginals['E']['distribution']['expected_value'],expected_E)
        self.assertAlmostEqual(marginals['Q']['distribution']['expected_value'],.2)
        self.assertNotAlmostEqual(marginals['M']['distribution']['expected_value'],expected_M/2)

    def test_unknown_block_keeps_half_mass_and_unknown_transfer_keeps_full_mass(self):
        intake=self.rows();block=intake['rows'][0]['capture_id']
        for row in intake['rows']:
            if row['capture_id']==block:row.update(joint_tuple=None,actual_native_quality=None)
        values,worlds=m.resource_marginals(intake,self.plan,self.gate,
            {'resource_marginal_transfer_qualified':True},{'CPU_ONLY':True},[{'CPU_ONLY':True}])
        for term in 'QME':
            self.assertEqual(values[term]['distribution']['unknown_probability_mass'],F(1,2))
            self.assertIsNone(values[term]['distribution']['expected_value'])
        values,_=m.resource_marginals(intake,self.plan,self.gate,
            {'resource_marginal_transfer_qualified':False},{'CPU_ONLY':True},[{'CPU_ONLY':True}])
        self.assertTrue(all(v['distribution']['unknown_probability_mass']==1 for v in values.values()))
        self.assertEqual(len(worlds),2)

    def test_missing_quality_never_changes_known_resource_terms_or_fills_fullU(self):
        intake=self.rows()
        for row in intake['rows']:row['actual_native_quality']['measured_Q']=None
        values,_=m.resource_marginals(intake,self.plan,self.gate,
            {'resource_marginal_transfer_qualified':True},{'CPU_ONLY':True},[{'CPU_ONLY':True}])
        self.assertIsNone(values['Q']['distribution']['expected_value'])
        self.assertIsNotNone(values['M']['distribution']['expected_value'])
        self.assertIsNotNone(values['E']['distribution']['expected_value'])

    def test_caller_cannot_construct_authenticated_resource_or_expected_session(self):
        with self.assertRaises(ValueError):m.ResourceSession(object(),{},None)
        with self.assertRaises(ValueError):m.ExpectedSession(object(),{},{},None,None,None)
        with self.assertRaises(ValueError):m.replay(object(),{})

    def test_full8_denominator_required_before_numeric_resource_marginals(self):
        value=self.rows();value['rows'].pop()
        with self.assertRaises(ValueError):m.resource_marginals(value,self.plan,self.gate,
            {'resource_marginal_transfer_qualified':True},{},[{'CPU_ONLY':True}])

    def test_actual_resource_session_reauthenticates_eight_unknowns_then_expires(self):
        with m.open_resource_session(RESOURCE,[]) as session:
            payload=session.checked_payload()
            self.assertEqual(len(payload['intake']['rows']),8)
            self.assertEqual(payload['intake']['unknown_resource_attempts'],8)
            self.assertIsNone(payload['intake']['paired_resource_catalog'])
        with self.assertRaisesRegex(ValueError,'expired'):session.checked_payload()

    def test_full_replay_plumbing_keeps12_queue_worlds_and8_unknown_resource_plans(self):
        # The exact typed target and original law execution are deliberately
        # mocked only for this CPU interface fixture; no native report results.
        session=object.__new__(m.ExpectedSession);session.check=lambda:None
        session.configuration={'target_plan':{'path':'/CPU_ONLY_PLAN'},'target_manifest':{'path':'/CPU_ONLY_MANIFEST'}}
        session.configuration_ref={'path':'/CPU_ONLY_PREPARATION'};session.queue=object()
        intake=self.rows()
        for row in intake['rows']:row.update(joint_tuple=None,actual_native_quality=None)
        session.resources=SimpleNamespace(checked_payload=lambda:dict(manifest_document=self.manifest,intake=intake,source_bindings=[]))
        cached={'plan_ref':session.configuration['target_plan']};prefix=dict(self.prefix,backend='CPU_TEST_ONLY')
        gate=dict(self.gate,prefix={'path':'/CPU_ONLY_PREFIX'})
        draws=[dict(draw_id='slot'+str(i),block_id='block'+str(i//4),weight=F(1,12),
            status='GENERATED_WORLD',forecast={'T':1,'L':2,'P':3}) for i in range(12)]
        distributions={t:m.ensemble.distribution(draws,lambda f,t=t:f[t]) for t in 'TLP'}
        queue=dict(sources=[self.plan['public_contract']],replay={'forecast':m.encode({'ensemble':{'draws':draws,'six_terms':distributions}})})
        with patch.object(m.typed,'_prefix_and_gate',return_value=(cached,self.plan,prefix,gate)),\
                patch.object(m.typed,'replay',return_value=queue):
            report=m.decode(m.replay(session,{'path':'/CPU_ONLY_GATE'},allow_cpu_fixture=True))
        self.assertEqual(len(report['source_queue_forecast']['replay']['forecast']['ensemble']['draws']),12)
        self.assertEqual(len(report['resource_marginal_worlds']),2)
        self.assertIsNone(report['expected_score']['expected_U_under_supplied_marginals'])
        for term in 'QME':self.assertEqual(report['expected_score']['marginals'][term]['distribution']['unknown_probability_mass'],1)
        self.assertEqual(report['expected_score']['expected_terms']['P'],3)
        self.assertFalse(report['source_evidence_authenticated']);self.assertFalse(report['queue_resource_draws_paired'])
        self.assertFalse(report['total_live_pipeline_budget_qualified']);self.assertFalse(report['native_collector_integrated'])

    def test_preparation_reconstructs_complete_freeze_and_cannot_enable_fullU_or_erase_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fit=root/'fit.json';fit.write_text(json.dumps({'CPU_TEST_ONLY':True}))
            plan=copy.deepcopy(self.plan);plan['saved_matching_fit']=receipt(fit)
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan));plan_ref=receipt(plan_path)
            manifest={'attempts':[{'plan':plan_ref}],'sources':[]}
            manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest))
            # Only preparation/audit shape is exercised. The exact target
            # source auditor is deliberately mocked for this CPU fixture.
            with patch.object(m.target,'audit',return_value=manifest),patch.object(m.target,'audit_plan',return_value=plan):
                ref=m.prepare(manifest_path,plan_path,RESOURCE,[],root/'expected')
                path=Path(ref['path']);original=path.read_bytes();value=checked(ref)
                self.assertFalse(value['total_live_pipeline_budget_qualified'])
                changes=[lambda d:d.update(sources=[]),lambda d:d.update(source_freeze=[]),
                    lambda d:d.update(total_live_pipeline_budget_qualified=True),
                    lambda d:d.update(full_U_seal_authorizes_target_enqueue=True),
                    lambda d:d.update(resource_reviews=[receipt(fit),receipt(fit)])]
                try:
                    for change in changes:
                        doc=copy.deepcopy(value);change(doc);path.write_text(json.dumps(doc))
                        with self.assertRaises(ValueError):m.audit_preparation(path)
                finally:path.write_bytes(original)


if __name__=='__main__':unittest.main()
