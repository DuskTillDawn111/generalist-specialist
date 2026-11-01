import copy
import unittest

from tools.predictor_validation_20261027.request_cuda_dag_v5 import owner_requests


def fixture():
    raw = dict(ready={'g':dict(pid=123)}, request_records=[dict(request_id='r', model='g')])
    kernel = dict(nsys_start_ns=10, nsys_end_ns=20, pid=123, stream=7, correlation_id=1, device=0, context_id=8)
    trace = dict(nvtx_ranges=[dict(stage=['r','prefill'], kernels=[kernel])],
                 unattributed_kernels=[], clock_mappings={'123':dict(valid=True, offset_ns=0)})
    return raw, trace


class StartupDagIntakeTests(unittest.TestCase):
    def test_canonical_sequence_is_counted_once_with_exact_sqlite_source(self):
        raw, trace = fixture()
        source = dict(path='train.sqlite', sha256='source')
        rows = owner_requests(raw, trace, source, 'g')
        self.assertEqual(len(rows['r']['kernels']), 1)
        self.assertEqual(rows['r']['kernels'][0]['source_sqlite'], source)
        self.assertEqual(rows['r']['row'], raw['request_records'][0])

    def test_duplicate_canonical_phase_views_are_rejected(self):
        raw, trace = fixture()
        trace['unattributed_kernels'].append(copy.deepcopy(trace['nvtx_ranges'][0]['kernels'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate canonical'):
            owner_requests(raw, trace, {}, 'g')

    def test_wrong_ready_owner_and_missing_clock_are_rejected(self):
        for change in ('owner','clock','empty'):
            raw, trace = fixture()
            if change == 'owner':
                raw['ready']['g']['pid'] = 999
            elif change == 'clock':
                trace['clock_mappings']['123']['valid'] = False
            else:
                trace['nvtx_ranges'][0]['kernels'] = []
            with self.assertRaises(ValueError):
                owner_requests(raw, trace, {}, 'g')


if __name__ == '__main__':
    unittest.main()
