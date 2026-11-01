"""Independent CPU-only adversarial checks of the unified intake boundary."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import unified_kernel_dag_intake_v1 as m


def kernel(left=10, right=20, correlation=1, **changes):
    return dict(nsys_start_ns=left, nsys_end_ns=right, pid=5,
                stream=7, correlation_id=correlation, device=0, context_id=1) | changes


def cohort(outside=()):
    raw = dict(ready={'ocr': {'pid': 5}}, request_records=[dict(request_id='r', model='ocr')])
    parsed = dict(clock_mappings={'5': {}}, nvtx_ranges=[dict(stage=['r'], kernels=[kernel()])],
                  unattributed_kernels=list(outside))
    return raw, parsed, dict(path='native.sqlite', sha256='a' * 64)


class IndependentUnifiedIntakeTests(unittest.TestCase):
    def test_outside_union_is_not_peer_multiplicity_and_boundaries_are_half_open(self):
        outside = [kernel(8, 14, 2), kernel(12, 16, 3), kernel(16, 18, 4),
                   kernel(0, 10, 5), kernel(20, 21, 6)]
        _, inventory = m.canonical_requests(*cohort(outside))
        self.assertEqual(inventory['outside_measured_overlap_wall_ns'], 8)
        self.assertEqual(inventory['full_session_kernels'], 6)

    def test_reversed_outside_interval_cannot_cancel_real_peer(self):
        args = cohort([kernel(10, 20, 2), kernel(20, 10, 3)])
        with self.assertRaises(ValueError):
            m.canonical_requests(*args)

    def test_native_intervals_fail_closed_for_every_canonical_role(self):
        for bounds in ((-1, 20), (10, 10), (20, 10), (10.0, 20), (True, 20)):
            for outside in (False, True):
                raw, parsed, ref = cohort()
                bad = kernel(*bounds, correlation=3)
                if outside:
                    parsed['unattributed_kernels'] = [bad]
                else:
                    parsed['nvtx_ranges'][0]['kernels'] = [bad]
                with self.subTest(bounds=bounds, outside=outside), self.assertRaises(ValueError):
                    m.canonical_requests(raw, parsed, ref)

    def test_all_native_identity_dimensions_retained_and_crossrange_duplicates_rejected(self):
        raw, parsed, ref = cohort()
        variants = [kernel(), kernel(stream=13), kernel(device=1), kernel(context_id=2)]
        parsed['nvtx_ranges'][0]['kernels'] = variants
        requests, inventory = m.canonical_requests(raw, parsed, ref)
        self.assertEqual(len(requests['r']['kernels']), 4)
        self.assertEqual(inventory['full_session_kernels'], 4)
        parsed['nvtx_ranges'].append(dict(stage=['other'], kernels=[copy.deepcopy(variants[0])]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.canonical_requests(raw, parsed, ref)

    def test_same_hardware_id_does_not_authorize_other_full_hardware_declaration(self):
        resolver = SimpleNamespace(hardware=dict(hardware_id='same', sm_count=80), index={}, source_receipts=[])
        with patch.object(m, 'AuditedOccupancyResolver', return_value=resolver):
            with self.assertRaisesRegex(ValueError, 'full hardware'):
                m.bind_static([], dict(hardware_id='same', sm_count=81), 'audit')

    def test_nontraining_plan_is_rejected_before_native_timing_parser(self):
        strict = dict(schema=m.REVIEW_SCHEMA, status='PASS', issues=[], training_only=True,
                      validation_started=False, model_fitted=False)
        plan = dict(schema='gsdeploy.unified_corun_training_plan.v8_1', training_only=True, split='validation')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            path.write_text(json.dumps(strict))
            with patch.object(m, 'load_review', return_value=(strict, plan, {})), \
                 patch.object(m, 'parse_sqlite', side_effect=AssertionError('timing label read')), \
                 patch.object(m.training, 'fit_paths', side_effect=AssertionError('fit attempted')):
                with self.assertRaisesRegex(ValueError, 'training cohort'):
                    m.intake(path, 'unused', Path(directory) / 'output')

    def test_path_rank_does_not_identify_single_observation_baselines(self):
        rows = [dict(group=('owner', index), ordered_matrix=np.vstack([np.zeros(6), np.eye(6)]))
                for index in range(13)]
        with patch.object(m.training, 'fit_paths', side_effect=AssertionError('fit attempted')):
            result = m.structural_diagnostics(dict(rejected_rows=[], features=list(range(6)), rows=rows))
        self.assertEqual(result['within_node_path_covariate_rank'], 6)
        self.assertEqual(result['singleton_baseline_groups'], 13)
        self.assertEqual(result['nuisance_adjusted_derivative_rank'], 0)
        self.assertFalse(result['fit_started'])
        self.assertFalse(result['numerical_qualification'])


if __name__ == '__main__':
    unittest.main()
