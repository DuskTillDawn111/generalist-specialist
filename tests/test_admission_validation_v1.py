"""Independent full native CPU fixtures; no GPU or outcome-based model fitting."""
import copy
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_admission_training_v1 as fixtures
from tools.predictor_validation_20261027 import admission_validation_v1 as m


class AdmissionValidationTests(unittest.TestCase):
    fixture=fixtures.AdmissionTrainingTests.fixture

    def setUp(self):
        fixtures.AdmissionTrainingTests.setUp(self)
        cp=self.root/'capacity-protocol.json';rp=self.root/'resource-protocol.json';qp=self.root/'quality-protocol.json'
        m.capacity.prepare(self.path,cp);m.resources.prepare(self.path,rp);m.quality.prepare(self.path,qp)
        root=m.training.ROOT/'artifacts/predictor-gpu-20261027'
        self.validation=self.root/'independent'
        with patch.object(m.training.native_flow,'execute',side_effect=AssertionError('No GPU prepare')):
            self.protocol=m.prepare(root/'admission-next-stage-roles-v1.json',self.path,cp,rp,qp,
                root/'background-statistical/selection-threshold-source-addendum.json',self.validation)
        self.protocol_path=self.validation/'protocol.json'
        self.plan=m.read(self.protocol['target_plan']['path'])
        self.job=dict(plan=self.protocol['target_plan'],candidate_id=m.capacity.ROUTES[0],
            output=str(self.root/'independent-native'))
        self.campaign['created_ns']=self.protocol['created_ns']
        self.sealed=dict(created_ns=self.protocol['created_ns'])

    def annotate(self):
        raw,path=self.fixture()
        raw['effective_objective']={k:self.plan[k] for k in ('workload_rps','latency_reference_ms','weights','memory_safe_factor',
            'memory_budget_mib','energy_reference_watts','startup_amortization_seconds')}
        for row in raw['request_records']:row['worker_response']['prediction']=''
        fixtures.write(path,raw)
        evidence=dict(kind='measured_per_request',errors=[],
            totals_by_request=m.training_run_audit.quality_totals(raw['request_records'],m.read(self.plan['catalog']['path'])),
            source_receipts=[self.plan['catalog'],m.receipt(m.training.native_flow.__file__),
                m.receipt(m.training_run_audit.score_result.__code__.co_filename),
                m.receipt(m.training_run_audit.sample_totals.__code__.co_filename)])
        scored={**raw,'native_raw':m.receipt(path),'quality_evidence':evidence}
        fixtures.write(path.with_name('scored-raw.json'),scored)
        return raw,path

    def test_prepare_freezes_only_public_V4_roles_sources_and_controller(self):
        self.assertEqual(m.audit_protocol(self.protocol_path),self.protocol)
        self.assertEqual(self.plan['seed'],628201);self.assertEqual(len(self.plan['trace']),1200)
        self.assertEqual(tuple(c['id'] for c in self.plan['candidates']),m.capacity.ROUTES)
        self.assertEqual(self.plan['controller_binding'],self.campaign['controller_binding'])
        self.assertTrue(self.plan['not_for_training']);self.assertEqual(self.plan['split'],'validation')
        self.assertEqual(self.plan['validation_thresholds'],m.LIMITS)
        self.assertFalse(self.protocol['training_outcomes_read']);self.assertFalse(self.protocol['independent_outcomes_read'])
        for candidate in self.plan['candidates']:
            self.assertNotIn('predicted_startup_joules',candidate)
            self.assertNotIn('predicted_prephase_peak_mib',candidate)
        self.assertNotIn('startup_comparison_policy',self.plan)

    def test_real_1200_native_mapping_and_quality_audit_never_call_training_or_shared_mapper(self):
        _,path=self.annotate();before=path.read_bytes();scored=path.with_name('scored-raw.json');sbefore=scored.read_bytes()
        with (patch.object(m.training,'map_training',side_effect=AssertionError('Independent data cannot enter training mapper')),
              patch.object(m.flow_profiles,'map_flow_collection',side_effect=AssertionError('No shared mapper'))):
            result=m.actual_evidence(path,self.protocol,self.sealed)
        self.assertEqual(path.read_bytes(),before);self.assertEqual(scored.read_bytes(),sbefore)
        self.assertEqual(result['requested'],1200);self.assertEqual(result['known'],1200)
        trace=result['semantic_validation_trace'];self.assertEqual(trace['split'],'validation')
        self.assertTrue(trace['not_for_training']);self.assertTrue(all(r['not_for_training'] for r in trace['requests']))
        self.assertEqual(len(result['service_observations']),1200)
        self.assertTrue(result['quality_audit']['independently_recomputed'])
        self.assertEqual(result['quality_audit']['count'],1200)
        self.assertEqual(result['original_stability']['status'],'ASSESSED')
        self.assertFalse(result['final_gain_eligible'])

    def test_native_training_final_or_posthoc_roles_cannot_be_relabelled(self):
        raw,path=self.annotate();original=copy.deepcopy(raw)
        for field,value in [('split','training'),('split','final'),('native_raw',m.receipt(path)),('derived_from','pilot')]:
            with self.subTest(field=field,value=value):
                changed=copy.deepcopy(original);changed[field]=value;fixtures.write(path,changed)
                with self.assertRaisesRegex(ValueError,'Original independent'):
                    m.actual_evidence(path,self.protocol,self.sealed)
        changed=copy.deepcopy(original);changed['request_records'][0]['split']='training';fixtures.write(path,changed)
        with self.assertRaisesRegex(ValueError,'cannot become independent'):
            m.actual_evidence(path,self.protocol,self.sealed)

    def test_forecast_source_seal_must_precede_startup_not_only_measurement(self):
        raw,path=self.annotate();self.sealed['created_ns']=raw['startup_begin_ns']
        with self.assertRaisesRegex(ValueError,'after complete forecast seal'):
            m.actual_evidence(path,self.protocol,self.sealed)

    def test_native_phase_ready_pid_controller_telemetry_or_priming_tamper_rejected(self):
        raw,path=self.annotate();original=copy.deepcopy(raw)
        variants=[('phase',lambda x:x['request_records'][0]['execution_events'][1].update(perf_ns=x['request_records'][0]['execution_events'][1]['perf_ns']+1)),
            ('READY',lambda x:x['ready']['text#0'].update(clock='foreign')),
            ('owned telemetry',lambda x:x['board_samples'][5].update(owned_used_mib=401.)),
            ('controller',lambda x:x.update(resource_configuration=m.training.native_flow.resource_configuration())),
            ('priming',lambda x:x['priming_request_records'].pop())]
        for name,mutate in variants:
            with self.subTest(name=name):
                changed=copy.deepcopy(original);mutate(changed);fixtures.write(path,changed)
                with self.assertRaises(ValueError):m.actual_evidence(path,self.protocol,self.sealed)

    def test_missing_request_or_wrong_route_never_disappears_from_validation_denominator(self):
        raw,path=self.annotate();original=copy.deepcopy(raw)
        raw['request_records'].pop();fixtures.write(path,raw)
        with self.assertRaisesRegex(ValueError,'Complete identical offered'):
            m.actual_evidence(path,self.protocol,self.sealed)
        raw=copy.deepcopy(original);raw['request_records'][0]['model']='ocr#0';fixtures.write(path,raw)
        with self.assertRaises(ValueError):m.actual_evidence(path,self.protocol,self.sealed)

    def test_actual_quality_is_recomputed_and_native_scoring_annotations_cannot_mutate_observations(self):
        _,path=self.annotate();scored=path.with_name('scored-raw.json');original=m.read(scored)
        changed=copy.deepcopy(original);changed['quality_evidence']['totals_by_request']['measurement:0']['numerator']+=1.
        fixtures.write(scored,changed)
        with self.assertRaisesRegex(ValueError,'recomputed totals'):
            m.actual_evidence(path,self.protocol,self.sealed)
        changed=copy.deepcopy(original);changed['request_records'][0]['completion_time_ns']+=1
        fixtures.write(scored,changed)
        with self.assertRaisesRegex(ValueError,'immutable native'):
            m.actual_evidence(path,self.protocol,self.sealed)

    def test_rehashed_public_plan_or_old_threshold_changes_do_not_override_projection(self):
        path=Path(self.protocol['target_plan']['path']);plan=m.read(path)
        plan['validation_thresholds']['service_mare_max']=.9
        plan['plan_sha256']=m.digest({k:v for k,v in plan.items() if k!='plan_sha256'});fixtures.write(path,plan)
        value=copy.deepcopy(self.protocol);value['target_plan']=m.receipt(path)
        value['source_receipts']=[m.receipt(path) if r['path']==str(path) else r for r in value['source_receipts']]
        value['protocol_sha256']=m.digest({k:v for k,v in value.items() if k!='protocol_sha256'});fixtures.write(self.protocol_path,value)
        with self.assertRaisesRegex(ValueError,'projection'):
            m.audit_protocol(self.protocol_path)

    def test_both_native_and_public_plan_rehash_cannot_change_original_objective(self):
        value=copy.deepcopy(self.protocol);target=Path(value['target_plan']['path']);native_path=Path(value['native_preparation']['path'])
        native=m.read(native_path);native['energy_reference_watts']=900.
        native['plan_sha256']=m.digest({k:v for k,v in native.items() if k!='plan_sha256'});fixtures.write(native_path,native)
        public=m._project(native,m.training.verify(self.path),self.plan['created_ns'],value['role_reservation'],value['threshold_addendum'])
        fixtures.write(target,public)
        replacements={str(target):m.receipt(target),str(native_path):m.receipt(native_path)}
        value['target_plan']=m.receipt(target);value['native_preparation']=m.receipt(native_path)
        value['source_receipts']=[replacements.get(r['path'],r) for r in value['source_receipts']]
        value['protocol_sha256']=m.digest({k:v for k,v in value.items() if k!='protocol_sha256'});fixtures.write(self.protocol_path,value)
        with self.assertRaisesRegex(ValueError,'Original fixed independent'):
            m.audit_protocol(self.protocol_path)

    def test_actual_effective_objective_must_match_the_original_plan(self):
        raw,path=self.annotate();raw['effective_objective']['energy_reference_watts']=900.;fixtures.write(path,raw)
        with self.assertRaisesRegex(ValueError,'native objective'):
            m.actual_evidence(path,self.protocol,self.sealed)

    def test_native_receipt_captured_before_audit_cannot_be_replaced_mid_parse(self):
        _,path=self.annotate();original=m.training._telemetry
        def mutate_after_read(raw):
            result=original(raw)
            path.write_text(path.read_text()+' ')
            return result
        with patch.object(m.training,'_telemetry',side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError,'Changed bound source'):
                m.actual_evidence(path,self.protocol,self.sealed)

    def test_actual_prediction_uses_checked_mode_resources_original_Q_and_complete_future_trace(self):
        from tools.predictor_validation_20261027 import statistical_profiles as statistics
        from tools.predictor_validation_20261027 import empirical_resources
        candidate=self.plan['candidates'][0];condition=m.capacity.condition(self.plan,candidate)
        generation=m.capacity._generation(self.plan,condition);binding=self.plan['controller_binding']
        contexts={};family_contexts={};phase_contexts={}
        for size in range(3):
            for aliases in itertools.combinations(self.plan['aliases'],size):
                rows=[dict(family=self.plan['aliases'][a]['family']) for a in aliases]
                fk,members=statistics.full_count(rows);family_contexts[fk]=members
                pk,members=statistics.phase_count([(r,dict(phase='native')) for r in rows]);phase_contexts[pk]=members
                if not aliases:choices=[()]
                else:choices=itertools.product(*[self.plan['aliases'][a]['supported_tasks'] for a in aliases])
                for tasks in choices:
                    active={a:dict(family=self.plan['aliases'][a]['family'],task=t) for a,t in zip(aliases,tasks)}
                    key,members=m.resources.engine._task_key(active)
                    contexts[key]=dict(members=members,mean_watts=100.+50.*size,
                        mean_trial_sampled_peak_mib=400.,max_trial_sampled_peak_mib=400.)
        source=m.receipt(self.protocol_path)
        resource=dict(schema=empirical_resources.SCHEMA,execution_context_sha256=self.plan['execution_context_sha256'],
            provenance=dict(split='training'),source_receipts=[source],contexts={},telemetry=dict(raw_idle_sampled_peak_mib=400.),
            task_specific_resource_omission_audit=dict(active_task_contexts=contexts),
            admission_training_evidence=dict(controller_binding=binding,candidate_id=candidate['id'],
                policy=m.resources.POLICY,condition=condition,generation=generation))
        resource['model_sha256']=m.digest(resource)
        losses={'text':{'text':.1},'generalist_2':dict(text=.1,image=.1,asr=.1,ocr=.1),'ocr':{'ocr':.1}}
        qdoc=dict(schema=m.quality.SCHEMA+'.model',policy=m.quality.POLICY,split='training',target_outcomes_used=False,
            candidate_id=candidate['id'],controller_binding=binding,execution_context_sha256=self.plan['execution_context_sha256'],
            catalog=self.plan['catalog'],condition=condition,generation=generation,source_receipts=[source],loss_by_family_task=losses)
        qdoc['model_sha256']=m.digest(qdoc)
        targets=sorted({(self.plan['aliases'][candidate['routes'][r['request_id']]]['family'],r['task'],r['pool_id']) for r in self.plan['trace']})
        model=dict(schema=statistics.SCHEMA,execution_context_sha256=self.plan['execution_context_sha256'],source_receipts=[source],
            profiles=[dict(id=':'.join(t),family=t[0],task=t[1],pool_id=t[2],execution=dict(batch_size=1,weight_reuse=False),
                segments=[dict(id='native',kind='mixed',duration_ms=100.,demand={})]) for t in targets],
            handler_models=[dict(target=list(t),progress={k:1. for k in family_contexts}) for t in targets],
            semantic_models=[dict(target=[*t,'native'],progress={k:1. for k in phase_contexts}) for t in targets],
            family_contexts=family_contexts,phase_contexts=phase_contexts)
        model['model_sha256']=m.digest(model)
        result=m.predict(self.plan,candidate,model,resource,qdoc,
            dict(predicted_startup_joules=100.,predicted_prephase_peak_mib=400.))
        self.assertIsNone(result['unknown_reason']);self.assertEqual(len(result['requests']),1200)
        self.assertTrue(all(r['e2e_ms'] is not None for r in result['requests']))
        self.assertTrue(result['metrics']['six_term_comparison_complete'])
        self.assertAlmostEqual(result['metrics']['terms']['Q'],.1)
        self.assertEqual(result['comparison_U'],result['metrics']['full_U'])
        self.assertTrue(result['not_for_training']);self.assertEqual(result['original_event_budget'],100000)

    def test_report_retains_missing_candidate_gate_and_recomputes_native_scoring_and_qualification(self):
        _,raw_path=self.annotate();scored=m.read(raw_path.with_name('scored-raw.json'))
        observed=m.flow.observed(self.plan,self.plan['candidates'][0],scored)
        folder=self.root/'CPU-sealed';folder.mkdir()
        fixture_source=folder/'CPU-source.json';fixtures.write(fixture_source,{'kind':'CPU_fixture'})
        source=m.receipt(fixture_source)
        predictions=[dict(candidate_id=c['id'],requests=observed['requests'],metrics=observed['metrics'],
            comparison_U=observed['metrics']['full_U'],queue_samples=observed['queue_samples'],
            comparison_scope='CPU reporter fixture only') for c in self.plan['candidates']]
        sealed=dict(created_ns=self.sealed['created_ns'],sources=dict(plan=self.protocol['target_plan'],
            model=source,resources=source,quality=source),candidates=predictions,
            training_sources=[],resource_sources=[],execution_sources=[],quality_sources=[],predictor_sources=[],
            tool_source=m.receipt(m.__file__),metric_source=m.receipt(m.flow.__file__))
        fixtures.write(folder/'predictions.json',sealed)
        capacity_path=folder/'capacity-seal.json';fixtures.write(capacity_path,{'kind':'CPU_capacity_fixture'})
        seal_audit=dict(protocol=m.receipt(self.protocol_path),capacity_seal=m.receipt(capacity_path))
        fixtures.write(folder/'seal-audit.json',seal_audit)
        groups=sorted({(self.plan['aliases'][row['model']]['family'],row['task']) for row in scored['request_records']})
        estimate=dict(status='ESTIMATED',rho_threshold_satisfied=True,
            estimates=[dict(family=f,task=t,weighted_worker_service_ms=.001) for f,t in groups])
        capacity_seal=dict(predictions=[dict(candidate_id=c['id'],prediction=estimate) for c in self.plan['candidates']])
        output=self.root/'report'
        # Only frozen model/seal verification is replaced by an explicit CPU
        # fixture; native audit, quality, old reporter/gate and capacity checks run.
        with (patch.object(m,'check_sealed',return_value=seal_audit),
              patch.object(m.capacity,'verify_seal',return_value=capacity_seal)):
            result=m.report(folder,[raw_path],output)
            self.assertEqual(result['candidate_coverage'],dict(requested=4,observed=1))
            self.assertEqual(result['status'],'INCOMPLETE')
            self.assertEqual(m.read(output/'gate.json')['status'],'INCOMPLETE')
            qualified=m.read(output/'capacity-qualification.json')['rows'][0]
            self.assertTrue(qualified['statistical_service_applicability'])
            self.assertFalse(qualified['final_gain_eligible'])
            self.assertTrue(m.verify_report(output/'report.json')['verified'])
            # Rehashing report and report-audit cannot invent a successful result.
            result['candidate_coverage']['observed']=4;fixtures.write(output/'report.json',result)
            audit=m.read(output/'validation-audit.json');audit['report']=m.receipt(output/'report.json')
            audit['audit_sha256']=m.digest({k:v for k,v in audit.items() if k!='audit_sha256'})
            fixtures.write(output/'validation-audit.json',audit)
            with self.assertRaisesRegex(ValueError,'scoring recomputation'):
                m.verify_report(output/'report.json')


class CapacityQualificationTests(unittest.TestCase):
    def evidence(self):
        return dict(candidate_id='admission2_balanced',known=1200,requested=1200,
            quality_audit=dict(count=1200,independently_recomputed=True),
            actual_metrics=dict(six_term_comparison_complete=True),
            original_stability=dict(original_stability_passed=False),
            service_observations=[dict(family='text',task='text',worker_service_ms=100.)])

    def prediction(self):
        return dict(status='ESTIMATED',rho_threshold_satisfied=True,
            estimates=[dict(family='text',task='text',weighted_worker_service_ms=100.)])

    def test_correct_means_and_rho_do_not_override_original_stability_failure(self):
        result=m.qualify_capacity(self.prediction(),self.evidence())
        self.assertTrue(result['statistical_service_applicability'])
        self.assertFalse(result['statistical_original_rho_planning_admission_passed'])
        self.assertIsNone(result['global_permit_capacity_qualified'])
        self.assertFalse(result['final_gain_eligible'])

    def test_incomplete_actual_utility_cannot_gain_capacity_qualification(self):
        evidence=self.evidence();evidence['original_stability']['original_stability_passed']=True
        evidence['actual_metrics']['six_term_comparison_complete']=False
        result=m.qualify_capacity(self.prediction(),evidence)
        self.assertIsNone(result['statistical_service_applicability'])
        self.assertIsNone(result['statistical_original_rho_planning_admission_passed'])


if __name__=='__main__':unittest.main()
