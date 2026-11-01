"""Node lifetime compiler counterexamples; CPU fixtures are never native proof."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from gsdeploy.resources import identity
from tests.test_kernel_resources import hardware, kernel
from tools.predictor_validation_20261027 import kernel_dag_response_training_v1 as m


def fixture():
    profiles = []; launches = []; bindings = []
    for owner, intervals in [('a', [('x', 0, 100), ('y', 20, 40)]),
                             ('b', [('z', 60, 80)])]:
        pid = 'profile-' + owner
        nodes = [dict(id=name, kind='gpu', kernel=dict(kernel(shared=1024), name=name))
                 for name, _, _ in intervals]
        profiles.append(dict(alias=owner, request_id=pid, task='task', pool_id='input',
            split='training', nodes=nodes, unresolved=[], cyclic=False))
        bindings.append(dict(request_id='r-'+owner, owner=owner, profile_id=pid,
                             task='task', pool_id='input'))
        launches.extend(dict(request_id='r-'+owner, owner=owner, node_id=name,
            kernel=copy.deepcopy(node['kernel']), start_ns=start*1_000_000,
            end_ns=end*1_000_000, clock_domain='shared-trace')
            for (name, start, end), node in zip(intervals, nodes))
    doc = dict(training_only=True, execution_context_sha256='context', profiles=profiles)
    cap = dict(schema=m.CAPTURE_SCHEMA, split='training', validation_labels_read=False,
        evidence_kind='cpu_fixture', execution_context_sha256='context',
        hardware_id=hardware()['hardware_id'], profiler_id='synthetic-nsys',
        capture_id='capture0', raw_sha256='a'*64,
        offered_request_ids=['r-a', 'r-b'], requests=bindings, kernels=launches)
    return doc, cap


def compile_fixture(doc=None, caps=None):
    base_doc, cap = fixture()
    return m.compile_paths(base_doc if doc is None else doc, hardware(),
                           [cap] if caps is None else caps, allow_cpu_fixture=True)


class KernelDagResponseTrainingTests(unittest.TestCase):
    def test_same_owner_multistream_peers_and_later_arrival_are_included(self):
        result = compile_fixture()
        path = next(p['path'] for p in result['observations'] if p['node_id']=='x')
        self.assertEqual([(p['left_ns']//1_000_000, p['right_ns']//1_000_000,
                           p['vector'][0]) for p in path],
                         [(0,20,0), (20,40,1), (40,60,0), (60,80,1), (80,100,0)])
        row = next(r for r in result['rows'] if r['phase']=='x')
        np.testing.assert_array_equal(row['ordered_durations'], [20]*5)
        self.assertEqual(sum(row['durations']), 100)
        self.assertEqual(row['matrix'][-1,0], 0)
        self.assertEqual(row['durations'][-1], 60)
        self.assertEqual((result['offered_kernel_count'], result['accepted_kernel_count']), (3,3))
        self.assertFalse(result['native_sources_authenticated'])

    def test_half_open_equal_clock_end_and_start_do_not_compete(self):
        doc, cap = fixture();cap['kernels'][2].update(start_ns=100_000_000, end_ns=120_000_000)
        result = compile_fixture(doc, [cap])
        z = next(p['path'] for p in result['observations'] if p['node_id']=='z')
        self.assertEqual(len(z),1);self.assertEqual(z[0]['vector'][0],0)

    def test_path_endpoint_is_left_limit_and_aggregate_endpoint_last(self):
        doc, cap = fixture();cap['kernels'][1]['end_ns']=100_000_000
        result = compile_fixture(doc, [cap]);row=next(r for r in result['rows'] if r['phase']=='x')
        self.assertEqual(row['ordered_matrix'][-1,0],1)
        self.assertEqual(row['matrix'][-1,0],1)
        self.assertEqual(sum(row['ordered_durations']),100)

    def test_clock_uncertainty_retains_entire_offered_unknown_cohort(self):
        doc, cap = fixture();cap['kernels'][2]['clock_domain']='separate-session-with-median-offset'
        result = compile_fixture(doc, [cap])
        self.assertEqual(result['accepted_kernel_count'],0)
        self.assertEqual(len(result['rejected_rows']),3)
        self.assertEqual(result['offered_requests'],2)
        self.assertEqual(m.fit_paths(result)['reason'],'INCOMPLETE_SUPPORTED_NODE_COHORT')

    def test_missing_or_duplicate_node_cannot_silently_reduce_denominator(self):
        for mode in ('missing','duplicate','foreign'):
            doc, cap = fixture()
            if mode=='missing':cap['kernels'].pop()
            elif mode=='duplicate':cap['kernels'].append(copy.deepcopy(cap['kernels'][0]))
            else:cap['kernels'][0]['node_id']='other'
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError,'canonical'):
                compile_fixture(doc,[cap])

    def test_owner_task_input_and_static_binding_are_mandatory(self):
        for change in ('owner','task','pool','static','hardware','context'):
            doc, cap = fixture()
            if change=='owner':cap['kernels'][0]['owner']='b'
            elif change=='task':cap['requests'][0]['task']='other'
            elif change=='pool':cap['requests'][0]['pool_id']='other'
            elif change=='static':cap['kernels'][0]['kernel']['grid']=[1,1,1]
            elif change=='hardware':cap['hardware_id']='other'
            else:cap['execution_context_sha256']='other'
            with self.subTest(change=change),self.assertRaises(ValueError):compile_fixture(doc,[cap])

    def test_role_gates_precede_every_capture_timing_access(self):
        class FutureLabels:
            def __iter__(self):raise AssertionError('future labels accessed')
        doc, first = fixture();second=copy.deepcopy(first)
        first['kernels']=FutureLabels();second['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):
            compile_fixture(doc,[first,second])

    def test_fixture_requires_opt_in_and_reexported_raw_cannot_be_repeat(self):
        doc, cap = fixture()
        with self.assertRaisesRegex(ValueError,'fixture opt-in'):
            m.compile_paths(doc,hardware(),[cap])
        other=copy.deepcopy(cap);other['capture_id']='different-name'
        with self.assertRaisesRegex(ValueError,'raw identities'):
            compile_fixture(doc,[cap,other])

    def test_invalid_durations_and_unresolved_profiles_fail(self):
        for mode in ('zero','backwards','bool','unresolved','duplicate_profile'):
            doc,cap=fixture()
            if mode=='zero':cap['kernels'][0]['end_ns']=0
            elif mode=='backwards':cap['kernels'][0]['end_ns']=-1
            elif mode=='bool':cap['kernels'][0]['start_ns']=False
            elif mode=='unresolved':doc['profiles'][0]['unresolved']=['missing-event']
            else:doc['profiles'].append(copy.deepcopy(doc['profiles'][0]))
            with self.subTest(mode=mode),self.assertRaises(ValueError):compile_fixture(doc,[cap])

    def test_missing_static_evidence_is_unknown_for_every_affected_target(self):
        doc,cap=fixture()
        for node in doc['profiles'][0]['nodes']:node['kernel'].pop('allocation_evidence')
        for launch in cap['kernels']:
            if launch['owner']=='a':launch['kernel'].pop('allocation_evidence')
        result=compile_fixture(doc,[cap])
        self.assertEqual(result['accepted_kernel_count'],0)
        self.assertEqual(len(result['rejected_rows']),3)

    def test_node_nuisance_rank_gate_keeps_all_six_columns(self):
        result=compile_fixture()
        with patch('tools.predictor_validation_20261027.macro_projection_contrast_training_v1.fit',
                   side_effect=AssertionError('rank deficient optimizer must not run')):
            info=m.fit_paths(result)
        self.assertEqual(info['reason'],'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        self.assertEqual(info['parameters'],3+6)
        self.assertFalse(info['columns_dropped'])
        self.assertFalse(info['numerical_qualification'])

    def test_low_memory_projected_rank_matches_explicit_nuisance_design(self):
        result=compile_fixture();rows=result['rows'];groups=sorted({r['group'] for r in rows})
        design=np.asarray([[int(r['group']==g) for g in groups]+list(v)
                           for r in rows for v in r['ordered_matrix']])
        self.assertEqual(m.fit_paths(result)['structural_design_rank'],np.linalg.matrix_rank(design))

    def test_bundle_can_only_report_unqualified_fixture_and_exact_catalog(self):
        doc,cap=fixture();bundle,prepared=m.training_bundle(doc,hardware(),[cap],allow_cpu_fixture=True)
        seal=bundle.pop('bundle_sha256')
        self.assertEqual(seal,identity(bundle));self.assertEqual(bundle['kernel_catalog_sha256'],identity(doc))
        self.assertEqual(bundle['evidence_kind'],'cpu_fixture')
        self.assertFalse(bundle['response_fit']['numerical_qualification'])
        self.assertFalse(bundle['native_sources_authenticated'])
        self.assertEqual(prepared['offered_kernel_count'],3)


if __name__=='__main__':unittest.main()
