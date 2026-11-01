"""CPU identity fixtures; verify_sources=False never claims actual verification."""
import copy
import unittest

from tools.predictor_validation_20261027.native_execution_projection_v5 import native_execution_projection


def context():
    source=dict(path='/cpu_fixture/native.py',sha256='a'*64)
    return dict(shared_native_execution_sources=[source],worker_bindings={},residents={'fixture#0':'fixture'},
        threads={'OMP_NUM_THREADS':'1'},execution={'batch_size':1},device='cpu_fixture_device',
        measurement_domain='light-kernel',profiler='cpu_fixture_profiler',affine_observation=False,
        worker_variant='fixture.py',large_weight_identity_scope='fixture_stat_only',
        large_weight_stats=dict(path='/cpu_fixture/stats.json',sha256='b'*64),
        execution_sources=[dict(original_path='/cpu_fixture/wrapper.py',frozen=dict(path='/cpu_fixture/frozen.py',sha256='c'*64))],
        sources=[],telemetry_scope='fixture_old')


class ProjectionTests(unittest.TestCase):
    def test_collector_difference_is_explicit_and_full_id_preserved(self):
        a=context();b=copy.deepcopy(a);b['telemetry_scope']='fixture_new'
        left=native_execution_projection(a,False);right=native_execution_projection(b,False)
        self.assertEqual(left['projection_sha256'],right['projection_sha256'])
        self.assertNotEqual(left['original_full_context_sha256'],right['original_full_context_sha256'])
        self.assertTrue(right['collector_invariance_is_hypothesis']);self.assertFalse(right['source_bytes_verified'])

    def test_measurement_domains_are_never_collapsed(self):
        a=context();b=copy.deepcopy(a);b['measurement_domain']='natural'
        self.assertNotEqual(native_execution_projection(a,False)['projection_sha256'],native_execution_projection(b,False)['projection_sha256'])

    def test_native_source_or_precision_change_changes_projection(self):
        a=context();b=copy.deepcopy(a);b['shared_native_execution_sources'][0]['sha256']='d'*64
        self.assertNotEqual(native_execution_projection(a,False)['projection_sha256'],native_execution_projection(b,False)['projection_sha256'])
        b=copy.deepcopy(a);b['execution']['batch_size']=2
        self.assertNotEqual(native_execution_projection(a,False)['projection_sha256'],native_execution_projection(b,False)['projection_sha256'])

    def test_outer_weight_snapshot_path_does_not_impersonate_new_weight_identity(self):
        a=context();b=copy.deepcopy(a);b['large_weight_stats']['path']='/cpu_fixture/other_snapshot.json'
        self.assertEqual(native_execution_projection(a,False)['projection_sha256'],native_execution_projection(b,False)['projection_sha256'])
        b['large_weight_stats']['sha256']='e'*64
        self.assertNotEqual(native_execution_projection(a,False)['projection_sha256'],native_execution_projection(b,False)['projection_sha256'])

    def test_missing_projection_field_is_rejected(self):
        a=context();del a['device']
        with self.assertRaises(ValueError):native_execution_projection(a,False)


if __name__=='__main__':unittest.main()
