"""CPU checks of gap topology and actual, deduplicated native evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import stage_native_gap_evidence_review_v1 as f

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/predictor-native-gap-evidence-v1-20261028-independentreview01'


class GapEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((BASE / 'evidence-v3.json').read_text())

    def test_actual_partition_gap_quantity_and_training_only_scope(self):
        natural = self.evidence['natural']
        self.assertEqual(len(natural['gaps']), 276)
        self.assertEqual(natural['total_gap_ns'], 15129050305)
        self.assertEqual(natural['total_worker_occupancy_ns'], 46578154846)
        self.assertAlmostEqual(natural['total_gap_ns'] / natural['total_worker_occupancy_ns'], .3248099963)
        self.assertFalse(self.evidence['kernel_natural_durations_pooled'])
        self.assertFalse(self.evidence['native_CUDA_API_numerical_authentication'])
        self.assertFalse(self.evidence['GPU_started'])

    def test_exact_gap_topology_per_request_and_query_counts(self):
        groups = {}
        for gap in self.evidence['natural']['gaps']:
            groups.setdefault(gap['request_id'], []).append(gap)
        self.assertEqual(len(groups), 46)
        for rows in groups.values():
            self.assertEqual([r['kind'] for r in rows], ['entry', *['transition'] * 4, 'exit'])
            self.assertEqual([r['gap_ordinal'] for r in rows], list(range(6)))
            self.assertEqual(rows[0]['false_queries'], 0)
            self.assertEqual(rows[-1]['false_queries'], 1)
            self.assertEqual(rows[-1]['true_queries'], 0)
            for gap in rows[1:-1]:
                self.assertEqual((gap['false_queries'], gap['true_queries']), (1, 1))

    def test_native_kernel_row_identity_never_double_counted_across_gaps(self):
        for trace in self.evidence['kernel']['traces']:
            all_ids = []
            for request in trace['requests']:
                for gap in request['gaps']:
                    ids = gap['gap_launched_kernel_rowids']
                    self.assertEqual(len(ids), gap['gap_launched_unique_kernels'])
                    all_ids.extend(ids)
            self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_wrong_export_raw_binding_rejected_before_sqlite_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'export.json'
            path.write_text(json.dumps(dict(raw=dict(path='wrong', sha256='wrong'), owner='text#0')))
            with self.assertRaisesRegex(ValueError, 'raw and owner export binding'):
                f.trace_owner(path, [dict(model='text#0')], dict(path='actual', sha256='actual'), 42)

    def test_union_overlap_and_nested_intervals_are_not_summed(self):
        self.assertEqual(f.union_ns([(0, 20), (2, 10), (10, 30), (40, 50)]), 40)
        self.assertEqual(f.union_ns([(5, 5), (20, 10)]), 0)

    def test_negative_gap_rejected(self):
        row = dict(request_id='r',model='owner',pool_id='p',profile_job_id='j',execution_events=[],
            worker_response=dict(observed_stages=[dict(id='native_inference',start_ns=100,end_ns=200)],
                semantic_forward_observations=dict(records=[dict(id='prefill_0000',forward_index=0,start_ns=90,end_ns=180)])))
        before = copy.deepcopy(row)
        with self.assertRaisesRegex(ValueError, 'Nonmonotone'):
            f.gaps(row)
        self.assertEqual(row, before)


if __name__ == '__main__':
    unittest.main()
