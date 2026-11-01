"""CPU-only immutable collection plan and actual per-batch/request receipts."""
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.runtime.batching import BatchConfiguration
from tests.test_batch_owner_runtime_v2 import Clock,timed_factory
from tests.test_shared_text_batch_v2 import sample,MODEL
from tools.predictor_validation_20261027 import batch_profile_collect_v2 as collector


class BatchProfileCollectV2Tests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source/gsdeploy/backends';source.mkdir(parents=True)
        (source/'compact_native.py').write_text('CPU fixture; not executable model source\n')
        protocol=root/'protocol.json';protocol.write_text('{}')
        classes=root/'classmap.json';classes.write_text('{}')
        spec=root/'spec.json';spec.write_text(json.dumps(dict(candidate=dict(slot='text',model=MODEL),
            execution=dict(family='chatqa'),resources=dict(protocol=str(protocol),classmap=str(classes)))))
        samples=root/'samples.json';samples.write_text(json.dumps(dict(
            training=[sample('a',[1,2]),sample('b',[1,40,3])],validation=[sample('c',[1,4])])))
        schedules=root/'arrivals.json';schedules.write_text(json.dumps(dict(
            training=[dict(request_id='t1',sample_id='a',arrival_ns=0),dict(request_id='t2',sample_id='b',arrival_ns=5)],
            validation=[dict(request_id='v1',sample_id='c',arrival_ns=0)])))
        return spec,source.parents[1],samples,schedules

    def prepare(self,root):
        args=self.fixture(root)
        plan=collector.prepare(root/'plan',*args,BatchConfiguration(2,100,10,(4,12)))
        return root/'plan/plan.json',plan,args

    def test_manifest_binds_disjoint_roles_new_physical_identity_and_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,plan,args=self.prepare(root)
            self.assertEqual(collector.audit(path),plan)
            self.assertEqual(plan['execution_blockers'],['G2_AND_OCR_BACKGROUND_BINDINGS_REQUIRED'])
            self.assertFalse(plan['old_T8_V4_training_reused'])
            physical=collector.read(collector.checked(plan['deployment']))
            self.assertEqual(physical['new_text_weight_copies'],1)
            self.assertEqual(len(physical['new_physical_owners']),3)
            self.assertEqual(len(physical['comparison_only_baseline']),4)
            self.assertTrue(physical['baseline_capacity_g_not_inherited'])
            self.assertNotEqual(plan['partitions']['training']['samples'],plan['partitions']['validation']['samples'])

    def test_cpu_fake_collect_real_batch_members_and_per_request_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,plan,args=self.prepare(root)
            clock=Clock();factory,calls=timed_factory(clock)
            completed=collector.collect(path,'training',backend_factory=factory,clock=clock.now,wait=clock.wait)
            self.assertEqual(completed['status'],'CPU_TEST_ONLY')
            self.assertEqual(completed['delivered'],2)
            index=collector.read(collector.checked(completed['profile_index']))
            self.assertEqual(set(index['requests']),{'t1','t2'})
            self.assertEqual(len(index['batches']),1)
            profile=collector.read(collector.checked(index['batches'][0]))
            self.assertEqual(profile['actual_members'],['t1','t2'])
            self.assertEqual(profile['actual_batch']['backend_metrics']['generate_calls'],1)
            self.assertEqual(profile['actual_batch']['selection']['padding_tokens'],1)
            self.assertTrue(profile['batch_input_key'].startswith('batch:'))
            self.assertEqual(len(profile['request_receipts']),2)
            for ref in profile['request_receipts']:
                request=collector.read(collector.checked(ref))
                self.assertEqual(request['split'],'training')
                self.assertEqual(request['parent_receipt_ns'],request['worker_next_ready_ns'])
                self.assertEqual(request['result']['batch_execution']['batch_size'],2)
            self.assertEqual(collector.read(collector.checked(completed['telemetry']))['status'],'UNKNOWN')
            self.assertTrue(index['not_for_training'])
            self.assertFalse(profile['fit_eligible'])
            self.assertFalse((path.parent/'validation/collection').exists())
            self.assertEqual(len(calls),2)
            with self.assertRaises(FileExistsError):
                collector.collect(path,'training',backend_factory=factory,clock=clock.now,wait=clock.wait)
            self.assertEqual(len(calls),2)

    def test_overlap_rejected_before_output_and_no_default_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root)
            data=collector.read(args[2]);data['validation'][0].update(context=data['training'][0]['context'],
                question=data['training'][0]['question'])
            args[2].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'overlap'):
                collector.prepare(root/'plan',*args,BatchConfiguration(2,100,10))
            self.assertFalse((root/'plan').exists())
            with self.assertRaisesRegex(RuntimeError,'--execute'):
                collector.collect('/missing','training')

    def test_tampered_public_split_stops_before_factory_and_execute_requires_backgrounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,plan,args=self.prepare(root)
            with self.assertRaisesRegex(ValueError,'G2_AND_OCR'):
                collector.collect(path,'training',execute=True)
            self.assertFalse((path.parent/'training/collection').exists())
            (path.parent/'validation/samples.json').write_text('[]')
            clock=Clock();factory,calls=timed_factory(clock)
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                collector.collect(path,'training',backend_factory=factory,clock=clock.now,wait=clock.wait)
            self.assertFalse(calls)


if __name__=='__main__':unittest.main()
