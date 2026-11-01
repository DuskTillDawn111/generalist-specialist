"""CPU-only Hcpu and prospective six-term counterexamples.

Native target/law fixtures are explicitly mocked; real arithmetic, source
receipt guards, fixed-mass composition and seal deadlines are exercised.
"""
import copy
from contextlib import contextmanager
from fractions import Fraction as F
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_expected_score_v2 as m
from tests import test_twoFIFO_public181_expected_score_v1 as old_fixture
RESOURCE=old_fixture.RESOURCE
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt


class ConditionalExpectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_fixture.ExpectedProducerTests.setUpClass();cls.manifest=old_fixture.ExpectedProducerTests.manifest;cls.donor=old_fixture.ExpectedProducerTests.donor
        cls.plan=copy.deepcopy(old_fixture.ExpectedProducerTests.plan);cls.prefix=copy.deepcopy(old_fixture.ExpectedProducerTests.prefix);cls.gate=copy.deepcopy(old_fixture.ExpectedProducerTests.gate)
        cls.plan['forecast_barrier']['prior_READY_settle_timeout_seconds']=10
        cls.plan['shared_dispatch_protocol']=m.shared_dispatch_protocol();cls.plan['evaluation_protocol']=copy.deepcopy(m.EVALUATION_PROTOCOL)
    rows=old_fixture.ExpectedProducerTests.rows

    def transfer(self,plan=None,gate=None):return m.transfer_axes({'manifest_document':self.manifest},plan or self.plan,self.prefix,gate or self.gate)

    def test_hcpu_permits_numeric_prediction_without_claiming_empirical_transfer(self):
        transfer=self.transfer()
        self.assertTrue(transfer['structural_transfer_supported']);self.assertTrue(transfer['conditional_resource_hypothesis_available'])
        self.assertFalse(transfer['resource_marginal_transfer_qualified']);self.assertFalse(transfer['empirical_transfer_qualified'])
        values,_=m.resource_marginals(self.rows(),self.plan,self.gate,transfer,{'CPU_ONLY':True},[self.plan['public_contract']])
        self.assertTrue(all(values[t]['distribution']['expected_value'] is not None for t in 'QME'))
        self.assertEqual(transfer['unknown_reasons'],[]);self.assertFalse(transfer['hypotheses_empirically_validated'])
        self.assertFalse(m.HCPU['assumed_full_protocol_equality'])

    def test_snapshot_anchor_or_ready_settle_mismatch_remains_mathematical_unknown(self):
        gate=copy.deepcopy(self.gate);gate['forecast_origin_ns']+=1
        transfer=self.transfer(gate=gate);self.assertFalse(transfer['conditional_resource_hypothesis_available'])
        self.assertIn('TRANSFER_AXIS_DIFFERS:actual_snapshot_lead',transfer['unknown_reasons'])
        plan=copy.deepcopy(self.plan);plan['forecast_barrier']['prior_READY_settle_timeout_seconds']=30
        self.assertFalse(self.transfer(plan=plan)['conditional_resource_hypothesis_available'])

    def test_hcpu_cannot_waive_worker_runtime_calendar_pool_or_protocol_change(self):
        changes=[lambda p:p['runtime_config'].update(max_pending_per_model=2),
            lambda p:p['cases'][0]['arrivals'][0].update(arrival_ns=1),
            lambda p:p.update(prepared_pool={'path':'/foreign','sha256':'0'*64}),
            lambda p:p['shared_dispatch_protocol'].update(function='foreign_dispatch'),
            lambda p:p['evaluation_protocol']['term_tolerances']['Q'].update(absolute_tolerance=1.)]
        for change in changes:
            plan=copy.deepcopy(self.plan);change(plan)
            transfer=self.transfer(plan=plan);self.assertFalse(transfer['conditional_resource_hypothesis_available'])
            values,_=m.resource_marginals(self.rows(),self.plan,self.gate,transfer,{},[self.plan['public_contract']])
            self.assertTrue(all(v['distribution']['unknown_probability_mass']==1 for v in values.values()))

    def test_missing_native_label_keeps_full_original_probability_mass(self):
        intake=self.rows();block=intake['rows'][0]['capture_id']
        for row in intake['rows']:
            if row['capture_id']==block:row.update(joint_tuple=None,actual_native_quality=None)
        values,worlds=m.resource_marginals(intake,self.plan,self.gate,self.transfer(),{},[self.plan['public_contract']])
        self.assertEqual(len(worlds),2)
        for term in 'QME':
            self.assertEqual(values[term]['distribution']['unknown_probability_mass'],F(1,2))
            self.assertIsNone(values[term]['distribution']['expected_value'])
        primitive=m.primitive_predictions(intake,self.plan,self.transfer())
        self.assertTrue(all(v['predicted_mean'] is None for v in primitive.values()))

    def test_unknown_quality_does_not_erase_valid_resource_observations(self):
        intake=self.rows()
        for row in intake['rows']:row['actual_native_quality']=None
        values,_=m.resource_marginals(intake,self.plan,self.gate,self.transfer(),{},[self.plan['public_contract']])
        self.assertIsNone(values['Q']['distribution']['expected_value'])
        self.assertIsNotNone(values['M']['distribution']['expected_value']);self.assertIsNotNone(values['E']['distribution']['expected_value'])

    def test_per_donor_memory_max_and_once_startup_are_not_changed_by_hcpu(self):
        values,_=m.resource_marginals(self.rows(),self.plan,self.gate,self.transfer(),{},[self.plan['public_contract']])
        contract=checked(self.plan['public_contract'])
        self.assertAlmostEqual(values['M']['distribution']['expected_value'],contract['memory_safe_factor']*100/contract['memory_budget_mib'])
        self.assertAlmostEqual(values['E']['distribution']['expected_value'],101/contract['energy_reference_watts'])
        self.assertAlmostEqual(m.primitive_predictions(self.rows(),self.plan,self.transfer())['premeasurement_owned_sampled_peak_mib']['predicted_mean'],50.)

    @contextmanager
    def report_fixture(self,root,*,missing_quality=False):
        plan=copy.deepcopy(self.plan);plan['forecast_barrier'].update(expected_report_path=str(root/'report.json'),seal_path=str(root/'seal.json'))
        def write(name,value):
            path=root/name;path.write_text(json.dumps(value));return receipt(path)
        plan_ref=write('plan.json',plan);manifest_ref=write('manifest.json',{'CPU_ONLY':True});prefix_ref=write('prefix.json',{'CPU_ONLY':True})
        gate=dict(self.gate,prefix=prefix_ref);gate_ref=write('gate.json',gate);prefix=dict(self.prefix,backend='CPU_TEST_ONLY')
        session=object.__new__(m.ExpectedSession);session.check=lambda:None;session.queue=object()
        session.configuration={'target_plan':plan_ref,'target_manifest':manifest_ref};session.configuration_ref=plan_ref
        intake=self.rows()
        if missing_quality:
            for row in intake['rows']:row['actual_native_quality']=None
        session.resources=SimpleNamespace(checked_payload=lambda:dict(manifest_document=self.manifest,intake=intake,source_bindings=[self.plan['public_contract']]))
        draws=[dict(draw_id='slot'+str(i),block_id='block'+str(i//4),weight=F(1,12),status='GENERATED_WORLD',forecast={'T':1,'L':2,'P':3}) for i in range(12)]
        distributions={t:m.ensemble.distribution(draws,lambda f,t=t:f[t]) for t in 'TLP'}
        queue=dict(sources=[self.plan['public_contract']],replay={'forecast':m.encode({'ensemble':{'draws':draws,'six_terms':distributions}})})
        with patch.object(m.typed,'_prefix_and_gate',return_value=({'plan_ref':plan_ref},plan,prefix,gate)),patch.object(m.typed,'replay',return_value=queue) as replay:
            yield session,gate_ref,plan,replay

    def test_complete_sixterm_mean_remains_explicitly_uncalibrated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.report_fixture(Path(directory)) as (session,gate,plan,replay):
                doc=m.decode(m.replay(session,gate,allow_cpu_fixture=True))
                self.assertTrue(doc['conditional_mean_available']);self.assertIsNotNone(doc['expected_score']['expected_U_under_supplied_marginals'])
                for flag in ('empirical_transfer_qualified','independent_accuracy_qualified','full_U_accuracy_qualified','final_gain_eligible','full_U_qualified'):
                    self.assertFalse(doc[flag])
                self.assertFalse(doc['queue_resource_draws_paired']);self.assertIsNone(doc['positive_delta_probability'])
                self.assertEqual(len(doc['resource_marginal_worlds']),2)

    def test_concrete_produce_and_seal_run_two_complete_passes_and_retain_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.report_fixture(root) as (session,gate,plan,replay):
                report=m.produce(session,gate['path'],root/'report.json',allow_cpu_fixture=True)
                seal=m.seal(session,gate['path'],root/'report.json',root/'seal.json',clock_ns=lambda:100,report_reference=report)
                self.assertEqual(replay.call_count,2)
                accepted=m.audit_seal_structure(session,gate,seal,now_ns=101,allow_cpu_fixture=True)
                self.assertTrue(accepted['CPU_TEST_ONLY']);self.assertFalse(accepted['full_U_accuracy_qualified'])
                with self.assertRaises(ValueError):m.audit_seal_structure(session,gate,seal,now_ns=self.gate['forecast_origin_ns'],allow_cpu_fixture=True)

    def test_missing_quality_holds_seal_without_hiding_other_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.report_fixture(root,missing_quality=True) as (session,gate,plan,replay):
                report=m.produce(session,gate['path'],root/'report.json',allow_cpu_fixture=True)
                doc=m.decode(checked(report));self.assertFalse(doc['conditional_mean_available'])
                self.assertIsNotNone(doc['expected_score']['expected_terms']['M'])
                with self.assertRaisesRegex(ValueError,'Missing mathematical support'):
                    m.seal(session,gate['path'],root/'report.json',root/'seal.json',clock_ns=lambda:100,report_reference=report)
                self.assertFalse((root/'seal.json').exists())

    def test_actual_native_resource_session_retains_eight_unknowns_and_expires(self):
        with m.open_resource_session(RESOURCE,[]) as session:
            payload=session.checked_payload();self.assertEqual(len(payload['intake']['rows']),8)
            self.assertEqual(payload['intake']['unknown_resource_attempts'],8)
        with self.assertRaisesRegex(ValueError,'expired'):session.checked_payload()

    def test_optional_preparation_cannot_rebind_resources_away_from_native4_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fit=root/'fit.json';fit.write_text(json.dumps({'CPU_ONLY':True}))
            plan=copy.deepcopy(self.plan);plan['saved_matching_fit']=receipt(fit)
            plan['expected_score_resources']=dict(manifest=receipt(RESOURCE),reviews=[])
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan));plan_ref=receipt(plan_path)
            manifest={'attempts':[{'plan':plan_ref}],'sources':[]}
            manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest))
            # Native target/source authentication is deliberately mocked only
            # to exercise complete configuration reconstruction, never native U.
            with patch.object(m.target,'audit',return_value=manifest),patch.object(m.target,'audit_plan',return_value=plan),\
                    patch.object(m.resource_executor,'audit',return_value=self.manifest):
                ref=m.prepare(manifest_path,plan_path,RESOURCE,[],root/'preparation')
                m.audit_preparation(ref['path'])
                rebound=copy.deepcopy(plan);rebound['expected_score_resources']['reviews']=[receipt(fit)]
                with patch.object(m.target,'audit_plan',return_value=rebound):
                    with self.assertRaisesRegex(ValueError,'resource binding'):
                        m.audit_preparation(ref['path'])

    def test_near_zero_sign_flip_reported_even_inside_engineering_indifference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.report_fixture(Path(directory)) as (session,gate,plan,replay):
                base=m.decode(m.replay(session,gate,allow_cpu_fixture=True))
        ids=['a','b','c','d'];predictions={};observed={}
        for i,cid in enumerate(ids):
            doc=copy.deepcopy(base);score=doc['expected_score'];terms={term:.6 for term in 'QTLPME'};terms['Q']+=i*.006
            draws=[dict(draw_id='one',block_id='block',weight=F(1),status='GENERATED_WORLD',forecast=terms)]
            marginals={term:dict(domain=score['candidate_domain'],sources=[self.plan['public_contract']],distribution=m.ensemble.distribution(draws,lambda f,t=term:f[t])) for term in terms}
            doc['expected_score']=m.marginal.combine(marginals,candidate_id=cid,candidate_domain=score['candidate_domain'],comparison_contract=score['comparison_contract'],weights=score['weights'])
            predictions[cid]=doc;actual=copy.deepcopy(terms);actual['Q']=.6-i*.006
            observed[cid]={'terms':actual,'full_U':-sum(score['weights'][t]*v for t,v in actual.items()),'weights':score['weights']}
        value=m.decode(m.evaluate_validation(predictions,observed,candidate_ids=ids))
        self.assertEqual(len(value['pairs']),6);self.assertEqual(value['known_pair_signs'],6)
        self.assertEqual(set(value['candidates'][0]['term_errors']),set('QTLPME'))
        self.assertIsNone(value['candidates'][0]['term_errors']['T']['within_engineering_tolerance'])
        self.assertTrue(all(p['exact_sign_agreement'] is False and p['predicted_engineering_indifference'] and p['observed_engineering_indifference'] for p in value['pairs']))
        observed.pop('d');value=m.decode(m.evaluate_validation(predictions,observed,candidate_ids=ids))
        self.assertEqual(value['pair_denominator'],6);self.assertEqual(len(value['pairs']),6);self.assertEqual(value['known_pair_signs'],3)
        self.assertEqual(sum(p['exact_sign_agreement'] is None for p in value['pairs']),3)
        self.assertFalse(value['full_U_accuracy_qualified'])
        observed['a']['full_U']+=.001
        with self.assertRaisesRegex(ValueError,'original scorer'):
            m.evaluate_validation(predictions,observed,candidate_ids=ids)


if __name__=='__main__':unittest.main()
