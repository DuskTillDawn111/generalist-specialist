"""CPU evidence-contract tests; none of these fixtures is physical GPU evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_full_horizon_forecast as analytical
from tests import test_flow_profiles as flow_fixtures
from tools.predictor_validation_20261027 import h600_admission_pilot_v1 as m


class AdmissionPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        m._PREREQUISITE_CACHE.clear()

    def tiny(self,gated=False,drain=1.):
        plan=dict(aliases=analytical.aliases(),window_seconds=1.,drain_seconds=drain,
            workload_rps={'text':3},queue_sample_offsets_ns=[0,50_000_000,150_000_000,1_000_000_000],
            trace=[dict(request_id='a',offset_ns=0,task='text',pool_id='x'),
                dict(request_id='b',offset_ns=25_000_000,task='text',pool_id='x'),
                dict(request_id='c',offset_ns=50_000_000,task='text',pool_id='x')])
        candidate=dict(id=m.IDS[1] if gated else m.IDS[0],execution_context_sha256='fixture',
            routes={'a':'s','b':'g','c':'s'},resource_configuration=analytical.resources(gated))
        if gated:
            candidate['resource_configuration']['groups']['gpu']=candidate['resource_configuration']['groups'].pop('all')
        return plan,candidate

    def test_full_arrivals_change_service_and_wait_without_shared_power_memory(self):
        plan,candidate=self.tiny()
        source=analytical.provider(s_ms=100.,g_ms=50.)
        with (patch.object(source,'power',side_effect=AssertionError('No power transfer')),
                patch.object(source,'memory',side_effect=AssertionError('No memory transfer'))):
            result=m.predict(plan,candidate,{},provider=source)
        rows={r['request_id']:r for r in result['requests']}
        self.assertEqual(result['known'],3)
        self.assertEqual(rows['a']['service_ms'],150.)
        self.assertEqual(rows['c']['wait_ms'],100.)
        self.assertEqual(result['execution_precheck'],'PREDICTED_COMPLETE')
        self.assertEqual(result['unsupported_predictions'],m.UNKNOWN)
        self.assertNotIn('comparison_U',result)

    def test_bounded_drain_preserves_every_request_and_never_completed_subset_p95(self):
        plan,candidate=self.tiny(gated=True,drain=.01)
        plan['window_seconds']=.1
        result=m.predict(plan,candidate,{},provider=analytical.provider(s_ms=100.,g_ms=100.))
        self.assertEqual(result['requested'],3);self.assertEqual(len(result['requests']),3)
        self.assertLess(result['known'],3)
        self.assertEqual(result['execution_precheck'],'PREDICTED_DRAIN_INFEASIBLE')
        self.assertIsNone(result['per_task']['text']['p95_ms'])
        self.assertEqual(len(result['incomplete_ids']),3-result['known'])
        self.assertFalse(result['permit_demand']['full_cohort_demand_known'])
        self.assertEqual(result['permit_demand']['available_permit_ms'],110.)

    def test_unknown_service_is_unknown_full_cohort_not_zero(self):
        plan,candidate=self.tiny()
        source=analytical.provider()
        with patch.object(source,'service',return_value={'known':False}):
            result=m.predict(plan,candidate,{},provider=source)
        self.assertEqual(result['execution_precheck'],'UNKNOWN_PREDICTION')
        self.assertEqual(result['known'],0);self.assertEqual(result['requested'],3)
        self.assertIsNone(result['per_task']['text']['p95_ms'])
        self.assertTrue(all(r['service_ms'] is None for r in result['requests']))

    def test_activation_requires_measurement_contention_and_cap_not_just_priming(self):
        candidate={'resource_configuration':m.native_flow.resource_configuration(2)}
        def row(alias,phase,wait,grant,end):
            return dict(alias=alias,phase=phase,wait_start_ns=wait,grant_ns=grant,parent_receipt_ns=end)
        audit={'requests':[row('text#0','priming',0,1,10),row('text#1','priming',0,2,9),
            row('ocr#0','priming',3,11,15),
            row('text#0','measurement',20,21,30),row('text#1','measurement',25,31,40)]}
        result=m.activation(audit,candidate)
        self.assertEqual(result['whole_run']['status'],'OBSERVED_ACTIVATION')
        self.assertEqual(result['measurement']['observed_peak'],1)
        self.assertEqual(result['measurement']['status'],'INSUFFICIENT_ACTIVATION')
        # Hitting cap2 plus another alias during clock overhead is not saturation wait.
        audit={'requests':[row('text#0','measurement',0,1,10),row('text#1','measurement',0,2,9)]}
        result=m.activation(audit,candidate)
        self.assertTrue(result['measurement']['observed_cap_reached'])
        self.assertEqual(result['measurement']['wait_under_observed_saturation_count'],0)
        self.assertEqual(result['measurement']['status'],'INSUFFICIENT_ACTIVATION')
        # Two acquire timestamp gaps without a competing alias do not establish contention.
        audit={'requests':[row('text#0','measurement',20,21,30),row('text#1','measurement',40,41,50)]}
        result=m.activation(audit,{'resource_configuration':m.native_flow.resource_configuration(1)})
        self.assertEqual(result['measurement']['positive_wait_count'],2)
        self.assertEqual(result['measurement']['wait_with_competing_alias_count'],0)
        self.assertEqual(result['measurement']['status'],'INSUFFICIENT_ACTIVATION')

    def test_prepare_fixes_seed_arrivals_routes_and_keeps_controller_identity_separate(self):
        profile=m.ROOT/'artifacts/predictor-gpu-20261027/background-training-v7'
        with patch.object(m.native_flow,'execute',side_effect=AssertionError('No GPU prepare')):
            plan=m.prepare(profile,self.root/'pilot')
        self.assertEqual(len(plan['trace']),1200);self.assertEqual(plan['seed'],627801)
        self.assertEqual([c['id'] for c in plan['candidates']],list(m.IDS))
        self.assertEqual(plan,m.audit_plan(self.root/'pilot/plan.json'))
        self.assertNotIn('startup_comparison_policy',plan)
        self.assertNotIn('prephase_memory_evidence',plan)
        for c in plan['candidates']:
            self.assertNotIn('predicted_prephase_peak_mib',c)
            self.assertEqual(c['routes'],plan['candidates'][0]['routes'])
        hashes={v['controller_sha256'] for v in plan['controller_bindings'].values()}
        self.assertEqual(len(hashes),3)
        self.assertEqual(len({c['execution_context_sha256'] for c in plan['candidates']}),1)
        plan['candidates'][1]['routes']['measurement:0']='generalist_2#0'
        plan['plan_sha256']=m.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        (self.root/'pilot/plan.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'structural projection'):m.audit_plan(self.root/'pilot/plan.json')

    def test_run_skips_infeasible_mode_without_gpu_and_preserves_three_mode_incomplete(self):
        seal_dir=self.root/'sealed';seal_dir.mkdir();(seal_dir/'predictions.json').write_text('{}')
        value=dict(prerequisite={'status':'FAIL'},sources={'plan':{'path':'unused'}},
            predictions=[dict(candidate_id=cid,requested=1200,known=1000 if i==1 else 1200,
                execution_precheck='PREDICTED_DRAIN_INFEASIBLE' if i==1 else 'PREDICTED_COMPLETE')
                for i,cid in enumerate(m.IDS)])
        with (patch.object(m,'check_sealed',return_value=value),patch.object(m.native_flow,'execute') as execute,
                patch.object(m,'actual_evidence',return_value={})):
            with self.assertRaisesRegex(ValueError,'explicitly diagnostic'):m.run(seal_dir,self.root/'denied')
            self.assertFalse((self.root/'denied').exists())
            state=m.run(seal_dir,self.root/'run',diagnostic=True)
        self.assertEqual([call.args[1] for call in execute.call_args_list],[m.IDS[0],m.IDS[2]])
        self.assertEqual(state['status'],'INCOMPLETE_DECLARED_MODES_SKIPPED')
        self.assertEqual(state['skipped'][0]['candidate_id'],m.IDS[1])
        self.assertFalse(state['final_gain_eligible'])
        self.assertTrue((self.root/'run/started.json').exists())
        self.assertTrue((self.root/'run/skipped-admission1_spill_low.json').exists())

    def prerequisite(self,status):
        """Isolate frozen-gate arithmetic while retaining exact report/audit receipts."""
        def write(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return path
        model=write('model.json',{});selection=write('selection.json',{})
        target=write('target.json',{})
        protocol={'selection':m.receipt(selection),'target_plan':m.receipt(target),'estimator_selection':{'selected':'gamma'}}
        protocol_path=write('protocol.json',protocol)
        folder=self.root/'independent-sealed';folder.mkdir()
        seal={'sources':{'model':m.receipt(model)}}
        sealpath=folder/'predictions.json';sealpath.write_text(json.dumps(seal))
        auditpath=folder/'training-audit.json';auditpath.write_text('{}')
        report={'sealed':m.receipt(sealpath),'report_source':m.receipt(m.flow_validation.__file__),
            'raw_sources':[], 'candidates':[dict(candidate_id=cid,requested=1200,known=1200)
                for cid in ('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high')]}
        report_path=write('independent-report.json',report)
        write('training-audit.json',dict(schema=m.selected.AUDIT+'.report',entry_source=m.receipt(m.selected.__file__),
            report=m.receipt(report_path),validation_protocol=m.receipt(protocol_path),
            sealed_audit=m.receipt(auditpath),strict_native_validation_evidence=[]))
        assessment=dict(status=status,checks=[])
        gate={**assessment,'plan':protocol['target_plan'],'validation_protocol':m.receipt(protocol_path),
            'report':m.receipt(report_path),'tool':m.receipt(m.selected.__file__),'final_gain_eligible':False}
        gate_path=write('gate.json',gate)
        patches=[patch.object(m.selected,'audit_protocol',return_value=protocol),
            patch.object(m.selected,'audit_model',return_value={'verified':True}),
            patch.object(m.selected,'audit_plan',return_value={}),
            patch.object(m.selected,'check_sealed',return_value={'verification':{'verified':True}}),
            patch.object(m.selected,'predictor_sources',return_value=[]),
            patch.object(m.selected,'validation_evidence',return_value=([],[])),
            patch.object(m.flow_gate,'assess',return_value=assessment)]
        for guard in patches:guard.start();self.addCleanup(guard.stop)
        return protocol_path,model,report_path,gate_path

    def test_complete_independent_fail_is_preserved_for_diagnostics(self):
        args=self.prerequisite('FAIL');result=m.audit_prerequisite(*args)
        self.assertEqual(result['status'],'FAIL');self.assertFalse(result['final_gain_eligible'])
        gate=json.loads(args[-1].read_text());gate['status']='PASS';args[-1].write_text(json.dumps(gate))
        with self.assertRaisesRegex(ValueError,'recomputed'):m.audit_prerequisite(*args)

    def test_numerical_verification_cache_still_rehashes_all_source_evidence(self):
        args=self.prerequisite('FAIL');first=m.audit_prerequisite(*args)
        with patch.object(m.selected,'check_sealed',side_effect=AssertionError('No repeated numerical fit')):
            self.assertEqual(m.audit_prerequisite(*args),first)
            (self.root/'independent-sealed/training-audit.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Changed bound source'):m.audit_prerequisite(*args)

    def test_incomplete_independent_gate_blocks_even_diagnostic_pilot(self):
        args=self.prerequisite('INCOMPLETE')
        with self.assertRaisesRegex(ValueError,'INCOMPLETE blocks'):m.audit_prerequisite(*args)

    def test_missing_actual_mode_retains_entire_declared_denominator(self):
        plan,candidate=self.tiny();source=analytical.provider()
        predictions=[]
        for cid in m.IDS:
            current=copy.deepcopy(candidate);current['id']=cid
            predictions.append(m.predict(plan,current,{},provider=source))
        seal_dir=self.root/'sealed';seal_dir.mkdir();(seal_dir/'predictions.json').write_text('{}')
        value=dict(sources={'plan':{'path':'unused'}},prerequisite={'status':'FAIL'},predictions=predictions)
        with patch.object(m,'check_sealed',return_value=value),patch.object(m,'audit_plan',return_value=plan):
            report=m.report(seal_dir,[],self.root/'missing-report')
        self.assertEqual(report['status'],'DIAGNOSTIC_INCOMPLETE')
        self.assertEqual(len(report['candidates']),3);self.assertEqual(report['paired_known'],0)
        self.assertTrue(all(c['requested']==3 and c['observed_known']==0 for c in report['candidates']))
        self.assertEqual(report['complete_three_mode_mechanism_conclusion'],'INCOMPLETE_OR_INSUFFICIENT_ACTIVATION')


class ControllerAdapterTests(unittest.TestCase):
    """Real mechanism audit and semantic parser, with only the full-plan guard isolated."""
    def setUp(self):
        self.fixture=flow_fixtures.FlowProfilesTests('test_continuous_cohort_trains_without_per_request_solo_trials')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture;self.root=self.f.root
        context=json.loads(Path(self.f.raw['context']['path']).read_text())
        for name in m.mechanism_audit.SEMANTIC_PATHS:
            path=m.ROOT/name;snapshot=self.root/name.replace('/','_');shutil.copyfile(path,snapshot)
            context['execution_sources'].append(dict(original_path=str(path),frozen=m.receipt(snapshot)))
        for family in set(m.RESIDENTS.values()):
            spec=self.root/(family+'-spec.json');spec.write_text(json.dumps({'family':family}))
            context['worker_bindings'][family].update(command=[sys.executable,'unused','--spec',str(spec)],
                spec_sha256=m.receipt(spec)['sha256'])
            context['execution_sources'].append(dict(original_path=str(spec),frozen=m.receipt(spec)))
        context['sources']=[s['frozen'] for s in context['execution_sources']]
        path=self.root/'controller-context.json';path.write_text(json.dumps(context))
        identity=m.native_flow.context_identity(context)
        self.f.raw.update(context=m.receipt(path),execution_context_sha256=identity,arrival_loop_end_ns=760_000_001)
        self.f.plan.update(context=m.receipt(path),execution_context_sha256=identity,
            phase='mechanism_validation',split='validation',runner_source=m.receipt(m.native_flow.__file__),
            trace_sha256=m.digest(self.f.plan['trace']),queue_sample_offsets_ns=[0,200_000_000],
            catalog=m.receipt(m.native_flow.BASE/'catalog.json'))
        candidate=self.f.plan['candidates'][0]
        candidate.update(id=m.IDS[2],execution_context_sha256=identity,
            resource_configuration=m.native_flow.resource_configuration(2))
        self.f.plan['baseline_candidate_id']=m.IDS[2]
        self.f.raw.update(candidate_id=m.IDS[2],resource_configuration=candidate['resource_configuration'],
            admission_snapshot_after=dict(active={},waiting=[],closed=True,peak_active=1))
        for alias,info in self.f.raw['ready'].items():
            info['spec_sha256']=context['worker_bindings'][m.RESIDENTS[alias]]['spec_sha256']
        for row in self.f.raw['request_records']:
            row.update(enqueued_time_ns=row['scheduled_time_ns'],admission_wait_start_time_ns=row['scheduled_time_ns'],
                admission_granted_time_ns=row['actual_submit_time_ns'])
            row['worker_response']['success']=True
            for event in row['execution_events']:event['received_time_ns']=event['perf_ns']
        self.write()

    def write(self):
        self.f.plan['controller_bindings']={m.IDS[2]:m._controller(self.f.plan,self.f.plan['candidates'][0])}
        self.f.save()
        invocation=dict(plan=self.f.raw['plan'],candidate_id=m.IDS[2],runner=self.f.plan['runner_source'])
        (self.root/'invocation.json').write_text(json.dumps(invocation))
        self.sealed=dict(created_ns=0,sources={'plan':m.receipt(self.f.plan_path)},
            controller_bindings=self.f.plan['controller_bindings'])

    def evidence(self):
        with (patch.object(m,'audit_plan',return_value=self.f.plan),
                patch.object(m.flow_profiles,'map_flow_collection',side_effect=AssertionError('Shared-only mapper forbidden'))):
            return m.actual_evidence(self.f.path,self.sealed)

    def test_nonshared_actual_adapter_validates_real_events_without_relabelling_shared_context(self):
        result=self.evidence()
        self.assertEqual(result['mechanism_audit']['status'],'PASS_OBSERVED_CONTRACT')
        self.assertEqual(result['known'],4)
        self.assertEqual(result['activation']['measurement']['status'],'INSUFFICIENT_ACTIVATION')
        self.assertEqual(result['requests'][0]['service_ms'],90.)
        self.assertEqual(result['requests'][0]['e2e_ms'],91.)
        self.assertTrue(result['semantic_validation_trace']['not_for_training'])
        self.assertTrue(all(r['not_for_training'] for r in result['semantic_validation_trace']['requests']))
        self.assertEqual(result['base_worker_context'],self.f.plan['context'])

    def test_nonshared_actual_mismatched_phase_telemetry_and_invocation_are_rejected(self):
        original=copy.deepcopy(self.f.raw)
        self.f.raw['request_records'][0]['execution_events'][1]['perf_ns']+=1;self.write()
        with self.assertRaises(ValueError):self.evidence()
        self.f.raw=copy.deepcopy(original);self.f.raw['board_samples'][5]['watts']=float('nan');self.write()
        with self.assertRaisesRegex(ValueError,'telemetry'):self.evidence()
        self.f.raw=copy.deepcopy(original);self.write();(self.root/'invocation.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'invocation'):self.evidence()

    def test_native_raw_and_prospective_controller_binding_cannot_be_substituted(self):
        self.sealed['created_ns']=self.f.raw['startup_begin_ns']
        with self.assertRaisesRegex(ValueError,'after exact prospective'):self.evidence()
        self.write();self.sealed['controller_bindings'][m.IDS[2]]['resource_configuration_sha256']='fake'
        with self.assertRaisesRegex(ValueError,'controller identity'):self.evidence()


if __name__=='__main__':unittest.main()
