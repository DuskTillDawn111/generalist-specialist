"""Real resource/capacity arithmetic with a mocked acquisition boundary only.

Synthetic mapped packets are CPU fixtures, never native/GPU measurements.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_capacity_v1 as capacity
from tools.predictor_validation_20261027 import admission_resources_v1 as resources
from tools.predictor_validation_20261027 import admission_training_v1 as training
from tools.predictor_validation_20261027.background_resources import _task_key
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling import digest


class AdmissionResourceCapacityTests(unittest.TestCase):
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        source=self.write('synthetic-acquisition.json',{'kind':'cpu_fixture'})
        jobs=[];self.packets={}
        for si,seed in enumerate(capacity.SEEDS):
            directory=self.root/f'seed-{seed}'
            native=training.native_flow.prepare(training.PROFILE,directory,seed=seed,duration=600.,
                drain=300.,warmup=0.,include_admission=True)
            plan=training._project(native,receipt(source),100+si)
            plan_path=self.write(f'seed-{seed}/plan.json',plan)
            for ri,cid in enumerate(capacity.ROUTES):
                candidate=next(c for c in plan['candidates'] if c['id']==cid)
                folder=directory/'runs'/cid
                raw=self.write(str(folder.relative_to(self.root)/'raw.json'),
                    {'fixture':cid,'seed':seed,'startup_begin_ns':10**18})
                job=dict(job_id=f'{seed}:{cid}',seed=seed,candidate_id=cid,plan=receipt(plan_path),
                    output=str(folder),split='training',campaign_role='primary_training')
                jobs.append(job)
                watts=100.+100.*si+10.*ri;peak=1000.+1000.*si+100.*ri
                common=dict(split='training',controller_sha256=plan['controller_binding']['controller_sha256'],
                    execution_context_sha256=plan['execution_context_sha256'],
                    resource_configuration=candidate['resource_configuration'],sources=[receipt(raw),receipt(plan_path)])
                samples=[dict(time_ns=t,watts=watts,owned_used_mib=peak if t==0 else 100.,
                    owned_aliases=[] if t==0 else list(plan['aliases']),foreign_pids=[])
                    for t in (0,1_000_000_000,1_250_000_000,1_500_000_000,1_750_000_000,2_000_000_000)]
                trace={**common,'id':f'{seed}:{cid}','observation_window_ns':dict(start=1_000_000_000,end=2_000_000_000),
                    'requests':[dict(alias='text#0',family='text',task='text',request_id=f'{seed}:{cid}',
                        handler_start_ns=1_200_000_000,handler_end_ns=1_600_000_000)]}
                evidence={**common,'measurement_window_ns':dict(start=1_000_000_000,deadline=2_000_000_000,observation_end=2_000_000_000),
                    'board_samples':samples,'startup_joules':watts,'measurement_joules':watts,'drain_joules':0.,
                    'raw_all_sampled_owned_peak_mib':peak,'raw_prephase_sampled_owned_peak_mib':peak,
                    'raw_measurement_and_drain_sampled_owned_peak_mib':100.}
                observations=[]
                for row in plan['trace']:
                    alias=candidate['routes'][row['request_id']]
                    observations.append(dict(request_id=row['request_id'],alias=alias,family=plan['aliases'][alias]['family'],
                        task=row['task'],pool_id=row['pool_id'],worker_service_ms=100.+100.*si+10.*ri,
                        split='training',controller_sha256=common['controller_sha256']))
                self.packets[job['job_id']]=dict(plan=plan,candidate=candidate,job=job,native_raw=receipt(raw),
                    traces=[trace],resource_evidence=evidence,service_observations=observations,mapping_report={})
        self.campaign=dict(jobs=jobs,controller_binding=plan['controller_binding'],source_receipts=[receipt(source)])
        self.campaign_path=self.write('campaign.json',self.campaign)
        guards=[patch.object(training,'verify',side_effect=lambda p:copy.deepcopy(self.campaign)),
            patch.object(training,'map_training',side_effect=lambda j,c:copy.deepcopy(self.packets[j['job_id']]))]
        for guard in guards:guard.start();self.addCleanup(guard.stop)
        self.capacity_protocol=self.root/'capacity-protocol.json'
        self.resource_protocol=self.root/'resource-protocol.json'
        capacity.prepare(self.campaign_path,self.capacity_protocol)
        resources.prepare(self.campaign_path,self.resource_protocol)

    def test_real_capacity_normalization_original_rho_and_whole_seed_cv(self):
        path=self.root/'capacity';capacity.fit(self.capacity_protocol,path)
        manifest=capacity.verify_manifest(path/'manifest.json')
        report=capacity.read(manifest['fit']['path'])
        self.assertEqual(len(report['whole_seed_cv']),8)
        for case in report['whole_seed_cv']:
            self.assertEqual(case['training_seeds'],[s for s in capacity.SEEDS if s!=case['seed']])
            prediction=case['prediction'];ri=capacity.ROUTES.index(case['candidate_id'])
            expected=200.+10.*ri if case['seed']==capacity.SEEDS[0] else 100.+10.*ri
            self.assertEqual(prediction['threshold'],.8);self.assertEqual(prediction['tolerance'],1e-8)
            for row in prediction['estimates']:
                if row['status']=='ESTIMATED':
                    self.assertAlmostEqual(row['weighted_worker_service_ms'],expected)
                    self.assertAlmostEqual(row['family_normalized_service_ms'],expected*row['original_g_n']/row['replicas'])
            self.assertIsNone(prediction['original_rho_admission_passed'])
            self.assertIsNone(prediction['global_permit_capacity_qualified'])
            self.assertFalse(case['qualification_passed'])
        self.assertFalse(report['final_gain_eligible'])

    def test_resources_route_isolation_full_history_peaks_and_whole_seed_exclusion(self):
        path=self.root/'resources';resources.fit(self.resource_protocol,path)
        manifest=resources.verify_manifest(path/'manifest.json')
        self.assertEqual(len(list(path.glob('*-resources.json'))),12)
        for ri,cid in enumerate(capacity.ROUTES):
            full=manifest['full']['candidate_overrides'][cid]
            self.assertEqual(full['predicted_startup_joules'],150.+10.*ri)
            self.assertEqual(full['predicted_prephase_peak_mib'],2000.+100.*ri)
            entry=manifest['folds'][0]
            self.assertEqual(entry['training_seeds'],[capacity.SEEDS[1]])
            doc=resources.read(entry['route_resources'][cid]['path'])
            run=next(p for p in self.packets.values() if p['job']['candidate_id']==cid)
            provider=resources.provider(doc,run['plan'],run['candidate'])
            active={'text#0':dict(family='text',task='text')}
            self.assertEqual(provider.power(active,run['plan']['aliases'],'mean')['watts'],200.+10.*ri)
            self.assertEqual(provider.memory(active,run['plan']['aliases'],'mean')['peak_memory_mib'],2000.+100.*ri)
            self.assertFalse(provider.memory(active,run['plan']['aliases'],'mean')['safe_factor_applied'])
            self.assertFalse(provider.power({'generalist_2#0':dict(task='asr')},run['plan']['aliases'],'mean')['known'])
        self.assertEqual(len(resources.read(manifest['resource_cv']['path'])['rows']),8)

    def test_resource_power_pooling_is_duration_weighted_within_route_only(self):
        _,runs=resources.load_runs(self.resource_protocol);cid=capacity.ROUTES[0]
        rows=[copy.deepcopy(r) for r in runs if r['candidate_id']==cid]
        for value in rows[1]['fit']['task_specific_resource_omission_audit']['active_task_contexts'].values():
            value['duration_ns']*=3
        doc=resources.merge_route(rows,cid,receipt(self.resource_protocol))
        key=_task_key({'text#0':dict(family='text',task='text')})[0]
        self.assertEqual(doc['task_specific_resource_omission_audit']['active_task_contexts'][key]['mean_watts'],175.)

    def test_shared_or_other_controller_cannot_enter_resource_or_capacity(self):
        bundle=capacity.load_training(self.capacity_protocol);run=bundle['runs'][0]
        bad=copy.deepcopy(run['plan']);candidate=copy.deepcopy(next(c for c in bad['candidates'] if c['id']==run['candidate_id']))
        candidate['resource_configuration']=training.native_flow.resource_configuration()
        with self.assertRaisesRegex(ValueError,'controller'):
            capacity.estimate(bad,candidate,bundle['runs'],capacity.read(bad['catalog']['path']))
        bad_evidence=copy.deepcopy(run['evidence']);bad_evidence['controller_sha256']='shared'
        with self.assertRaisesRegex(ValueError,'controller'):
            resources.fit_run(bad_evidence,run['traces'],bad['aliases'],bad['controller_binding'])
        for row in bundle['runs']:row['controller_sha256']='other'
        candidate=next(c for c in run['plan']['candidates'] if c['id']==run['candidate_id'])
        result=capacity.estimate(run['plan'],candidate,bundle['runs'],capacity.read(run['plan']['catalog']['path']))
        self.assertEqual(result['status'],'UNKNOWN');self.assertIsNone(result['rho_threshold_satisfied'])

    def test_missing_positive_weight_input_is_unknown_without_renormalization(self):
        bundle=capacity.load_training(self.capacity_protocol);run=bundle['runs'][0]
        candidate=next(c for c in run['plan']['candidates'] if c['id']==run['candidate_id'])
        for row in bundle['runs']:row['observations']=[o for o in row['observations'] if o['task']!='ocr']
        result=capacity.estimate(run['plan'],candidate,bundle['runs'],capacity.read(run['plan']['catalog']['path']))
        self.assertEqual(result['status'],'UNKNOWN')
        missing=next(r for r in result['estimates'] if r['task']=='ocr')
        self.assertEqual(missing['known_target_weight_mass'],0.)
        self.assertIsNone(missing['weighted_worker_service_ms']);self.assertIsNone(result['rho_threshold_satisfied'])

    def test_validation_split_or_partial_observations_are_rejected_by_loader(self):
        packet=next(iter(self.packets.values()));packet['resource_evidence']['split']='validation'
        with self.assertRaisesRegex(ValueError,'role/controller'):
            capacity.load_training(self.capacity_protocol)
        packet['resource_evidence']['split']='training';packet['service_observations'].pop()
        with self.assertRaisesRegex(ValueError,'All offered'):
            capacity.load_training(self.capacity_protocol)

    def test_missing_training_run_fails_without_creating_fit_directory(self):
        Path(self.campaign['jobs'][0]['output'],'raw.json').unlink()
        for module,protocol in ((capacity,self.capacity_protocol),(resources,self.resource_protocol)):
            with self.subTest(module=module.__name__):
                output=self.root/module.__name__
                with self.assertRaisesRegex(ValueError,'All eight'):module.fit(protocol,output)
                self.assertFalse(output.exists())

    def test_resource_rehashed_startup_tamper_is_detected_by_recompute(self):
        output=self.root/'resources';resources.fit(self.resource_protocol,output)
        path=output/'manifest.json';value=resources.read(path)
        value['full']['candidate_overrides'][capacity.ROUTES[0]]['predicted_startup_joules']+=1.
        value.pop('manifest_sha256');value['manifest_sha256']=digest(value);path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'recomputed'):resources.verify_manifest(path)

    def test_resource_provider_cannot_apply_another_route_model(self):
        _,runs=resources.load_runs(self.resource_protocol)
        doc=resources.merge_route(runs,capacity.ROUTES[0],receipt(self.resource_protocol))
        run=next(r for r in runs if r['candidate_id']==capacity.ROUTES[1])
        candidate=next(c for c in run['plan']['candidates'] if c['id']==run['candidate_id'])
        with self.assertRaisesRegex(ValueError,'matching admission'):
            resources.provider(doc,run['plan'],candidate)

    def test_capacity_mean_service_check_uses_predicted_denominator_and_missing_unknown(self):
        prediction=dict(estimates=[dict(family='text',task='text',weighted_worker_service_ms=100.),
            dict(family='ocr',task='ocr',weighted_worker_service_ms=None)])
        checks=capacity.service_checks(prediction,[dict(family='text',task='text',worker_service_ms=125.)])
        self.assertEqual(checks[0]['absolute_relative_mean_error'],.25)
        self.assertFalse(checks[0]['passed']);self.assertIsNone(checks[1]['passed'])

    def test_disjoint_public_target_can_be_sealed_without_target_outcomes(self):
        native=training.native_flow.prepare(training.PROFILE,self.root/'target-native',seed=628201,
            duration=600.,drain=300.,warmup=0.,include_admission=True)
        plan=training._project(native,receipt(self.campaign_path),300)
        plan.update(split='validation',phase='predictor_validation',kind='prospective_capacity_validation')
        plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
        target=self.write('target-plan.json',plan);sealed=self.root/'capacity-seal.json'
        result=capacity.seal(target,self.capacity_protocol,sealed)
        self.assertEqual(capacity.verify_seal(sealed),result)
        self.assertEqual(len(result['predictions']),4)
        self.assertFalse(result['target_outcomes_used'])
        self.assertIsNone(result['statistical_service_applicability'])
        self.assertIsNone(result['original_rho_admission_passed'])
        self.assertFalse(result['final_gain_eligible'])
        with self.assertRaisesRegex(ValueError,'Disjoint prospective'):
            capacity.seal(self.campaign['jobs'][0]['plan']['path'],self.capacity_protocol,self.root/'bad-seal.json')
        self.assertFalse((self.root/'bad-seal.json').exists())
        plan['seed']=628202;plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
        replacement=self.write('replacement-target.json',plan)
        with self.assertRaisesRegex(ValueError,'Disjoint prospective'):
            capacity.target_plan(replacement)

    def test_independent_mean_adapter_never_grants_qualification_without_raw_audit(self):
        prediction=dict(status='ESTIMATED',estimates=[dict(family='text',task='text',weighted_worker_service_ms=100.)])
        result=capacity.assess_independent_service(prediction,[dict(family='text',task='text',worker_service_ms=110.)])
        self.assertTrue(result['mean_service_numerical_threshold_passed'])
        self.assertIsNone(result['statistical_service_applicability'])
        self.assertIsNone(result['original_rho_admission_passed'])
        self.assertFalse(result['independent_raw_and_temporal_audit_performed'])

    def test_rehashed_capacity_prediction_tamper_rejected_by_training_recompute(self):
        output=self.root/'capacity';capacity.fit(self.capacity_protocol,output)
        manifest_path=output/'manifest.json';manifest=capacity.read(manifest_path)
        path=Path(manifest['fit']['path']);document=capacity.read(path)
        document['whole_seed_cv'][0]['prediction']['rho_by_family']['text']=0.
        path.write_text(json.dumps(document));manifest['fit']=receipt(path)
        manifest.pop('manifest_sha256');manifest['manifest_sha256']=digest(manifest)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'recomputed'):capacity.verify_manifest(manifest_path)

    def test_fit_protocols_must_precede_every_native_training_startup(self):
        for module,path in ((capacity,self.capacity_protocol),(resources,self.resource_protocol)):
            with self.subTest(module=module.__name__):
                value=capacity.read(path);value['created_ns']=10**18+1
                value.pop('protocol_sha256');value['protocol_sha256']=digest(value)
                path.write_text(json.dumps(value))
                output=self.root/(module.__name__+'-late')
                with self.assertRaisesRegex(ValueError,'before every training startup'):
                    module.fit(path,output)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
