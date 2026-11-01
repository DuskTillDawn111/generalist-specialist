import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import h600_resources_v1 as module
from tools.predictor_validation_20261027.background_resources import TaskConditionalResourceProvider, _task_key
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling import digest


class H600ResourceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.aliases={a:{'family':f} for a,f in {'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'}.items()}
        self.packets={};jobs=[]
        source=self.write('source.json',{})
        for si,seed in enumerate(module.primary.POLICY['training_seeds']):
            for ri,cid in enumerate(sorted(module.primary.weighted.ROUTES)):
                folder=self.root/f'{seed}-{cid}';folder.mkdir()
                plan=self.write(f'{seed}-{cid}/plan.json',dict(seed=seed,aliases=self.aliases))
                path=self.write(f'{seed}-{cid}/raw.json',dict(plan=receipt(plan),candidate_id=cid))
                watts=100.+si*100+ri;peak=500.+si*500+ri
                samples=[dict(time_ns=t,watts=watts,owned_used_mib=peak if t==0 else 100.,
                    owned_aliases=[] if t==0 else list(self.aliases),foreign_pids=[]) for t in (0,1_000_000_000,1_250_000_000,1_500_000_000,1_750_000_000,2_000_000_000)]
                trace=dict(split='training',id=f'{seed}:{cid}',requests=[dict(alias='text#0',family='text',task='text',request_id=f'{seed}:{cid}',
                    handler_start_ns=1_200_000_000,handler_end_ns=1_600_000_000)],
                    observation_window_ns=dict(start=1_000_000_000,end=2_000_000_000))
                evidence=dict(split='training',sources=[receipt(path)],execution_context_sha256='context',
                    measurement_window_ns=dict(start=1_000_000_000,deadline=2_000_000_000,observation_end=2_000_000_000),
                    board_samples=samples,startup_joules=watts,measurement_joules=watts,drain_joules=0.,
                    raw_all_sampled_owned_peak_mib=peak,raw_prephase_sampled_owned_peak_mib=peak,
                    raw_measurement_and_drain_sampled_owned_peak_mib=100.)
                self.packets[str(path)]=dict(resource_evidence=evidence,traces=[trace])
                jobs.append(dict(output=str(folder),plan=receipt(plan),candidate_id=cid))
        self.suite=self.write('suite.json',dict(jobs=jobs,source=receipt(source),protocol=receipt(source)))
        mocks=[patch.object(module.primary,'verify_suite',side_effect=lambda p:module.read(p)),
            patch.object(module,'map_flow_collection',side_effect=lambda p:copy.deepcopy(self.packets[str(p)]))]
        for m in mocks:m.start();self.addCleanup(m.stop)
        self.protocol=self.root/'protocol.json';module.prepare(self.suite,self.protocol)

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def fitted(self):
        output=self.root/'fit';module.fit(self.protocol,output);return output

    def test_real_engine_full_and_whole_seed_fit_recompute(self):
        output=self.fitted();manifest=module.verify_manifest(output/'manifest.json')
        cid=sorted(module.primary.weighted.ROUTES)[0]
        full=manifest['full']['candidate_overrides'][cid]
        self.assertEqual(full['predicted_startup_joules'],150.)
        self.assertEqual(full['predicted_prephase_peak_mib'],1000.)
        self.assertEqual(full['full_history_mean_peak_mib'],750.)
        fold=manifest['folds'][0]
        self.assertEqual(fold['training_seeds'],[627602])
        self.assertEqual(fold['candidate_overrides'][cid]['predicted_startup_joules'],200.)
        self.assertEqual(fold['candidate_overrides'][cid]['sources'],[manifest['full']['candidate_overrides'][cid]['sources'][1]])
        cv=module.read(output/'resource-training-cv.json')
        self.assertEqual(len(cv['rows']),8)
        self.assertEqual(cv['rows'][0]['predicted_startup_joules'],200.)

    def test_startup_partial_ownership_excluded_from_active_fit_but_peak_retained(self):
        output=self.fitted();doc=module.read(output/'heldout-seed-627602-resources.json')
        provider=TaskConditionalResourceProvider(doc,'context')
        active={'text#0':dict(family='text',task='text')}
        memory=provider.memory(active,self.aliases,'mean')
        self.assertTrue(memory['known'])
        self.assertEqual(memory['peak_memory_mib'],501.5)
        self.assertFalse(memory['capacity_bound'])
        self.assertFalse(memory['safe_factor_applied'])
        self.assertFalse(provider.power({'generalist_2#0':dict(task='asr')},self.aliases,'mean')['known'])

    def test_power_pool_is_duration_weighted_not_seed_or_memory_average(self):
        _,runs=module.load_runs(self.protocol)
        one=copy.deepcopy(runs[0]);two=copy.deepcopy(runs[4])
        for row in two['fit']['task_specific_resource_omission_audit']['active_task_contexts'].values():row['duration_ns']*=3
        doc=module.merge_runs([one,two],receipt(self.protocol))
        key=_task_key({'text#0':dict(family='text',task='text')})[0]
        self.assertEqual(doc['task_specific_resource_omission_audit']['active_task_contexts'][key]['mean_watts'],175.)

    def test_rehashed_candidate_startup_or_peak_tamper_rejected_by_recompute(self):
        output=self.fitted();path=output/'manifest.json';original=module.read(path)
        cid=sorted(module.primary.weighted.ROUTES)[0]
        for field in ('predicted_startup_joules','predicted_prephase_peak_mib'):
            with self.subTest(field=field):
                value=copy.deepcopy(original);value['folds'][0]['candidate_overrides'][cid][field]+=1.
                value.pop('manifest_sha256');value['manifest_sha256']=digest(value);path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'recomputed'):module.verify_manifest(path)

    def test_rehashed_fold_identity_or_resource_value_tamper_rejected(self):
        output=self.fitted();manifest=module.read(output/'manifest.json');path=Path(manifest['folds'][0]['resources']['path'])
        doc=module.read(path);key=next(iter(doc['task_specific_resource_omission_audit']['active_task_contexts']))
        doc['task_specific_resource_omission_audit']['active_task_contexts'][key]['mean_watts']+=1
        doc.pop('model_sha256');doc['model_sha256']=digest(doc);path.write_text(json.dumps(doc))
        manifest['folds'][0]['resources']=receipt(path);manifest.pop('manifest_sha256');manifest['manifest_sha256']=digest(manifest)
        (output/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'recomputed'):module.verify_manifest(output/'manifest.json')

    def test_missing_primary_fails_before_mapping_or_output_creation(self):
        Path(next(iter(self.packets))).unlink();output=self.root/'fit'
        with self.assertRaisesRegex(ValueError,'All eight'):module.fit(self.protocol,output)
        self.assertFalse(output.exists())

    def test_validation_mapping_cannot_enter_fit(self):
        mapped=copy.deepcopy(next(iter(self.packets.values())));mapped['resource_evidence']['split']='validation'
        with self.assertRaisesRegex(ValueError,'training-only'):
            module.fit_run(mapped['resource_evidence'],mapped['traces'],self.aliases)


if __name__=='__main__':unittest.main()
