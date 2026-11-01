"""Independent CPU checks; no capture extraction, predictor initialization or GPU."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import owner_lifecycle_mark_v21 as mark
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tests import test_owner_lifecycle_v21 as author_tests
from tools.predictor_validation_20261027.owner_lifecycle_hazard_query_v24 import LifecycleSaveHazard
from tools.predictor_validation_20261027.owner_lifecycle_hazard_v24_fit import SCHEMA as HAZARD_SCHEMA, EDGES_NS, lifecycle_features


class PrefixTests(unittest.TestCase):
    def extract(self, *, handler_end=100, kernel_start=200, kernel_end=300,
                cutoff=95, issue_end=None, valid_mapping=True):
        def ref(path):
            return dict(path=path, sha256=path)
        raw_ref, report_ref = ref('/fixture/raw.json'), ref('/fixture/report.json')
        sql_ref, parsed_ref = ref('/fixture/trace.sqlite'), ref('/fixture/parsed-measured.json')
        obs_ref = ref('/fixture/observations.json')
        row = dict(capture=raw_ref['sha256'], held_owner='text#1', next_owner='text#0',
            current_restore_start_ns=10, save_end_ns=cutoff,
            context_age_at_save_ns=cutoff-10, next_restore_start_ns=400,
            pending_frontier=[])
        data = {
            '/fixture/supervision.json': dict(training_only=True, validation_labels_used=False,
                transition_observations=[row], training_audits=[dict(raw=raw_ref,report=report_ref)]),
            raw_ref['path']: dict(request_records=[dict(request_id='r',model='text#0',
                execution_events=[dict(pid=7)],worker_response=dict(start_perf_ns=0,end_perf_ns=handler_end))]),
            report_ref['path']: dict(sources=[sql_ref,parsed_ref],kernel_ready_observations=obs_ref),
            parsed_ref['path']: dict(clock_mappings={'7':dict(valid=valid_mapping,offset_ns=0,max_boundary_residual_ns=10)}),
            obs_ref['path']: dict(rows=[dict(request_id='r',issue_end_ns=issue_end,
                observed_start_ns=kernel_start,observed_end_ns=kernel_end)]),
        }
        class Connection:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def execute(self,*args): return []
        with patch.object(mark,'read',side_effect=lambda path:copy.deepcopy(data[str(path)])), \
                patch.object(mark,'receipt',side_effect=lambda path:ref(str(path))), \
                patch.object(mark.sqlite3,'connect',return_value=Connection()):
            return mark.source_prefix_rows('/fixture/supervision.json')[0][0]

    def test_future_handler_end_cannot_change_prefix_features(self):
        # Former residual bands changed True to UNKNOWN at 90 before END=100.
        near=self.extract(handler_end=100)
        far=self.extract(handler_end=1000)
        self.assertTrue(near['states']['text#0'].handler_active)
        self.assertEqual(near['states'],far['states'])
        self.assertEqual(mark.features(near['query'],near['states'],{}),
                         mark.features(far['query'],far['states'],{}))

    def test_future_kernel_start_and_end_cannot_change_outstanding_issue_prefix(self):
        first=self.extract(kernel_start=200,kernel_end=300,issue_end=20)
        second=self.extract(kernel_start=900,kernel_end=1000,issue_end=20)
        self.assertEqual(first['states'],second['states'])
        self.assertTrue(first['states']['text#0'].issued_outstanding)
        self.assertFalse(first['states']['text#0'].gpu_ready)

    def test_observed_handler_end_changes_state_at_half_open_boundary(self):
        self.assertTrue(self.extract(handler_end=100,cutoff=99)['states']['text#0'].handler_active)
        self.assertFalse(self.extract(handler_end=100,cutoff=100)['states']['text#0'].handler_active)

    def test_invalid_clock_mapping_is_unknown(self):
        with self.assertRaisesRegex(UnknownEvidence,'CLOCK_MAPPING_REQUIRED'):
            self.extract(valid_mapping=False)


class StateMismatchTests(unittest.TestCase):
    def states(self):
        return {owner:mark.OwnerLifecycle(False,False,False) for owner in mark.RESIDENTS}

    def test_nonempty_frontier_cannot_use_all_idle_snapshot(self):
        query=dict(held_owner='text#1',context_age_ns=0,
            pending_frontier=[['text#0','text','short',['prefill']]],
            previous_held_owner=None,public_workload=None,
            pending_age_ns_by_owner=None,pending_reference_work_ns_by_owner=None)
        with self.assertRaisesRegex(ValueError,'disagree'):
            mark.features(query,self.states(),{'short':'text'})

    def test_settled_view_mismatch_is_unknown(self):
        provider=mark.LifecycleActiveServiceMark.__new__(mark.LifecycleActiveServiceMark)
        provider.current_clock_ns=None
        node=ActiveGpuNode(('text#0','g'),1.,'prefill','r','text','short')
        view=SimpleNamespace(clock_ns=1,owner_lifecycle=tuple(self.states().items()),active_gpu_nodes=(node,))
        with self.assertRaisesRegex(UnknownEvidence,'READY_FRONTIER_MISMATCH'):
            provider.observe_view(view)


class EngineBoundaryTests(unittest.TestCase):
    def test_post_boundary_source_change_is_rejected_even_on_scheduler_exception(self):
        fixture=author_tests.EngineTests('test_lifecycle_hook_preserves_all_numerical_results')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'provider.json'
            source.write_text('before')
            class Scheduler:
                def decide(self,view):
                    source.write_text('after!')
                    raise UnknownEvidence('ORIGINAL_SCHEDULER_UNKNOWN')
            engine=fixture.engine(Scheduler(),source_receipts=[mark.receipt(source)])
            with self.assertRaisesRegex(UnknownEvidence,'SOURCE_BYTES_CHANGED'):
                engine.forecast(fixture.rows(),0,30,drain_deadline_ns=1000)


class HazardBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.states={owner:mark.OwnerLifecycle(False,False,False) for owner in mark.RESIDENTS}
        vocabulary=lifecycle_features('text#0',[],self.states)
        self.hazard=LifecycleSaveHazard(dict(schema=HAZARD_SCHEMA,fixed_age_edges_ns=list(EDGES_NS),
            columns=['intercept'],coefficients=[0.],training_vocabulary=list(vocabulary),
            age_support_ns=[[0,1000000]]))

    def test_empty_no_competitor_has_positive_rate_but_no_qualified_cdf(self):
        value=self.hazard.query('text#0',[],self.states,0,100)
        self.assertEqual(value['hazard_per_ms'],1.)
        self.assertGreater(value['restricted_candidate_CDF'],0.)
        self.assertIsNone(value['conditional_CDF'])
        self.assertIn('ZERO_OBSERVED_EVENTS_DOES_NOT_CERTIFY_ZERO_HAZARD',value['OOD_reasons'])

    def test_mismatch_foreign_owner_and_untyped_state_are_rejected(self):
        with self.assertRaisesRegex(UnknownEvidence,'READY_FRONTIER_MISMATCH'):
            self.hazard.query('text#0',[['text#0','text','short',['prefill']]],self.states,0)
        with self.assertRaisesRegex(UnknownEvidence,'FOREIGN_GPU_FRONTIER_OWNER'):
            self.hazard.query('text#0',[['foreign','text','short',['prefill']]],self.states,0)
        untyped=dict(self.states);untyped['text#0']={'handler_active':False}
        with self.assertRaises(ValueError):self.hazard.query('text#0',[],untyped,0)


if __name__=='__main__':
    unittest.main()
