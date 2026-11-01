import copy
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.kernel_dag_training_mapping_audit_v1 import (
    authenticated_sources, replay_catalog, require_denominator, validate_mapping,
)
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources
from tools.predictor_validation_20261027.native_collect import receipt


def fixture():
    source = dict(path='/captured.sqlite', sha256='real-source-placeholder-for-unit-test')
    k = dict(nsys_start_ns=100, nsys_end_ns=120, pid=123, stream=7,
             correlation_id=1, device=0, context_id=8, source_sqlite=source)
    row = dict(model='ocr#0', task='ocr', pool_id='input-a')
    request = dict(row=row, kernels=[k])
    node = dict(id='gpu:1', kind='gpu', kernel=copy.deepcopy(k), stream_key=[123, 0, 8, 7],
                correlation_id=1, observed_start_ns=100, observed_end_ns=120)
    profile = dict(request_id='r', alias='ocr#0', task='ocr', pool_id='input-a', split='training',
                   nodes=[node], edges=[], kernel_count=1, unresolved=[], cyclic=False,
                   source_sqlite=source)
    return request, profile


class KernelDagMappingAuditTests(unittest.TestCase):
    def test_exact_source_bound_bijection(self):
        request, profile = fixture()
        result = validate_mapping('r', request, profile, 123)
        self.assertTrue(result['canonical_bijection'])
        self.assertEqual(result['kernel_stream_keys'], [(123, 0, 8, 7)])

    def test_missing_and_duplicated_canonical_nodes_fail(self):
        for change in ('missing', 'duplicate', 'duplicate-canonical'):
            request, profile = fixture()
            if change == 'missing':
                profile['nodes'] = []
            elif change == 'duplicate':
                profile['nodes'].append(copy.deepcopy(profile['nodes'][0]))
            else:
                request['kernels'].append(copy.deepcopy(request['kernels'][0]))
            with self.assertRaisesRegex(ValueError, 'BIJECTION'):
                validate_mapping('r', request, profile, 123)

    def test_owner_request_pool_and_role_cannot_be_rebound(self):
        for field in ('request_id', 'alias', 'task', 'pool_id', 'split'):
            request, profile = fixture()
            profile[field] = 'different'
            with self.assertRaisesRegex(ValueError, 'BINDING'):
                validate_mapping('r', request, profile, 123)

    def test_ready_pid_device_context_stream_and_runtime_identity_are_checked(self):
        for field in ('pid', 'device', 'context_id', 'stream', 'correlation_id', 'nsys_start_ns'):
            request, profile = fixture()
            request['kernels'][0][field] += 1
            profile['nodes'][0]['kernel'][field] += 1
            with self.assertRaisesRegex(ValueError, 'BOUNDARY'):
                validate_mapping('r', request, profile, 123)

    def test_another_sqlite_capture_cannot_supply_identical_kernel(self):
        request, profile = fixture()
        profile['source_sqlite'] = dict(path='/another.sqlite', sha256='different-capture')
        with self.assertRaisesRegex(ValueError, 'BOUNDARY'):
            validate_mapping('r', request, profile, 123)

    def test_unresolved_and_cycles_fail(self):
        for change in ('unresolved', 'cycle'):
            request, profile = fixture()
            if change == 'unresolved':
                profile['unresolved'] = [dict(reason='NO_EXACT_EVENT_RECORD_IDENTITY')]
            else:
                profile['edges'] = [dict(before='gpu:1', after='gpu:1', kind='stream_fifo')]
            with self.assertRaisesRegex(ValueError, 'UNRESOLVED_OR_CYCLIC'):
                validate_mapping('r', request, profile, 123)

    def test_same_owner_different_stream_nodes_stay_distinct(self):
        request, profile = fixture()
        k = dict(request['kernels'][0], stream=9, correlation_id=2)
        request['kernels'].append(k)
        profile['nodes'].append(dict(profile['nodes'][0], id='gpu:2', kernel=copy.deepcopy(k),
                                     stream_key=[123, 0, 8, 9], correlation_id=2))
        profile['kernel_count'] = 2
        result = validate_mapping('r', request, profile, 123)
        self.assertEqual(result['mapped_gpu_node_count'], 2)
        self.assertEqual(len(result['kernel_stream_keys']), 2)
        self.assertEqual(profile['edges'], [])

    def test_full_offered_denominator_rejects_missing_extra_duplicate(self):
        require_denominator(['text', 'ocr'], ['ocr', 'text'])
        for actual in (['text'], ['text', 'ocr', 'extra'], ['text', 'ocr', 'ocr']):
            with self.assertRaisesRegex(ValueError, 'FULL_OFFERED'):
                require_denominator(['text', 'ocr'], actual)

    def test_replay_owner_keeps_every_supported_task(self):
        profiles = [dict(alias='g', family='generalist', task=task, pool_id='input-'+task,
                         request_id='r-'+task) for task in ('text', 'ocr', 'text')]
        profiles[-1]['pool_id'] = 'second-text'
        profiles[-1]['request_id'] = 'second-text'
        aliases, plans, requests = replay_catalog(profiles)
        self.assertEqual(aliases['g']['supported_tasks'], ['text', 'ocr'])
        self.assertEqual(len(plans), 3)
        self.assertEqual([r['arrival_ns'] for r in requests], [0, 50_000_000, 100_000_000])

    def test_replay_owner_rejects_inconsistent_family(self):
        profiles = [dict(alias='g', family=family, task='text', pool_id=str(i), request_id=str(i))
                    for i, family in enumerate(('generalist', 'specialist'))]
        with self.assertRaisesRegex(ValueError, 'OWNER_FAMILY_MISMATCH'):
            replay_catalog(profiles)

    def test_implementation_closure_includes_transitive_checkpoint_and_packages(self):
        source = Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/kernel_dag_training_mapping_audit_v1.py'
        names = {Path(s['path']).name for s in implementation_sources(source)}
        self.assertTrue({'cuda_dag_simulator_v9.py', 'dag_checkpoint_v7.py',
                         'implementation_source_closure_v27.py', '__init__.py'} <= names)

    def test_nested_receipts_are_authenticated_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.txt'
            path.write_text('before')
            ref = receipt(path)
            self.assertEqual(authenticated_sources({'raw': {'sources': [ref]}}, [ref]), [ref])
            other = dict(ref, sha256='f'*64)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                authenticated_sources(ref, {'nested': other})
            path.write_text('after')
            with self.assertRaises(ValueError):
                authenticated_sources({'nested': [ref]})


if __name__ == '__main__':
    unittest.main()
