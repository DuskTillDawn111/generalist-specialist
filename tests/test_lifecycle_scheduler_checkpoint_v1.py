"""Real scheduler/controller state round trips; no native progress or GPU work."""
import copy
import json
import math
import unittest
from unittest.mock import patch

from tests import test_handoff_candidate_v14 as handoff_fixture
from tests import test_learned_lifecycle_scheduler_v49_independent_review as numeric_fixture
from tests import test_owner_lifecycle_v21 as engine_fixture
from tests import test_projected_state_save_hazard_v57 as state_hazard_fixture
from tests.test_context_initial_prior_v15 import fixture as prior_fixture, gap
from tools.predictor_validation_20261027 import lifecycle_scheduler_checkpoint_v1 as cp
from tools.predictor_validation_20261027.context_initial_prior_v15 import JointInitialContextPrior
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.handoff_candidate_v14 import HandoffCandidate
from tools.predictor_validation_20261027.physical_context_mark_bridge_v56 import PhysicalContextMark
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import receipt
from tools.predictor_validation_20261027.profiling import digest


class SchedulerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = numeric_fixture.IndependentReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def scheduler(self, *, physical_mark=False, seeded=False, prior=False):
        scheduler = self.fixture.scheduler(lower_rate_on_peer=True)
        # Real HandoffCandidate, with an explicitly JSON-normalized CPU model.
        model = json.loads(json.dumps(handoff_fixture.HandoffCandidateTests().fixture()))
        scheduler.oracle.handoff = HandoffCandidate(model, allow_cpu_fixture=True)
        if physical_mark:
            scheduler.oracle.next_owner = PhysicalContextMark(copy.deepcopy(scheduler.oracle.next_owner.candidate.model),
                allow_unvalidated_active_service_bridge=True)
        if seeded:
            old = scheduler.oracle.controller
            scheduler.oracle.controller = CumulativeLifecycleHazard(old._candidate,
                age_edges_ns=old.edges, sources=[receipt(self.fixture.source)], candidate_identity='a'*64,
                allow_unvalidated_approximation=True, budget_policy='SEEDED_UNIFORM', seed=29)
        if prior:
            scheduler.oracle.initial_prior = JointInitialContextPrior(prior_fixture([gap()]), allow_cpu_fixture=True)
        return scheduler

    def binding(self, clock):
        return dict(execution_context_sha256='a'*64, cohort_sha256='b'*64, graph_sha256='c'*64,
            original_now_ns=0, deadline_ns=10**9, drain_deadline_ns=2*10**9,
            clock_ns=clock, atomic_stage=cp.STAGE)

    def ready(self, scheduler):
        return {owner: 20 if i == 0 else 0 for i, owner in enumerate(sorted(scheduler.owners))}

    def export(self, scheduler, clock):
        return cp.export_scheduler_snapshot(scheduler, binding=self.binding(clock), owner_last_ready=self.ready(scheduler))

    def test_export_import_restore_call_no_state_advancing_or_budget_methods(self):
        source, target = self.scheduler(prior=True), self.scheduler(prior=True)
        source.decide(self.fixture.view(0))
        before = cp._state(source, self.ready(source))
        methods = [(cp.LearnedLifecycleScheduler, 'snapshot'), (cp.LearnedLifecycleScheduler, 'decide'),
            (cp.LearnedLifecycleOracle, 'observe_query'), (cp.LearnedLifecycleOracle, 'observe_lifecycle'),
            (cp.LearnedLifecycleOracle, 'initialize'), (CumulativeLifecycleHazard, '_budget'),
            (CumulativeLifecycleHazard, 'start_restore'), (CumulativeLifecycleHazard, 'update'),
            (cp.ObservableReferenceMark, 'observe_view')]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for cls, name in methods:
                stack.enter_context(patch.object(cls, name, side_effect=AssertionError('State-advancing method called')))
            exported = self.export(source, 0)
            detached = cp.import_scheduler_snapshot(target, exported, binding=self.binding(0))
            ready = cp.restore_scheduler_snapshot(target, exported, binding=self.binding(0))
        self.assertEqual(before, cp._state(source, self.ready(source)))
        self.assertEqual(before, detached)
        self.assertEqual(before, cp._state(target, ready))

    def test_full_budget_history_and_next_decisions_match_after_JSON_restore(self):
        for seeded in (False, True):
            with self.subTest(seeded=seeded):
                source, target = self.scheduler(seeded=seeded), self.scheduler(seeded=seeded)
                source.decide(self.fixture.view(0, handlers=('text#0',)))
                decision = source.decide(self.fixture.view(1000))
                exported = json.loads(json.dumps(self.export(source, 1000), allow_nan=False))
                ready = cp.restore_scheduler_snapshot(target, exported, binding=self.binding(1000))
                self.assertEqual(cp._state(source, ready), cp._state(target, ready))
                budget = source.oracle.hazard['hazard_budget']
                consumed = source.oracle.hazard['consumed_hazard']
                self.assertGreater(consumed, 0)
                # Old review then the later exact SAVE; no missed event jump.
                for _ in range(3):
                    clock = decision.next_transition_ns
                    view = self.fixture.view(clock)
                    decision = source.decide(view)
                    self.assertEqual(decision, target.decide(view))
                    self.assertEqual(cp._state(source, ready), cp._state(target, ready))
                    if source.pending_switch is not None:
                        break
                self.assertIsNotNone(source.pending_switch)
                self.assertEqual(source.oracle.hazard['hazard_budget'], budget)
                self.assertEqual(target.oracle.hazard['consumed_hazard'], budget)
                self.assertEqual(target.oracle.hazard['origin_restore_ns'], 0)

    def test_pending_handoff_absolute_events_and_single_observation_survive(self):
        source, target = self.scheduler(), self.scheduler()
        decision = source.decide(self.fixture.view(0))
        save = decision.next_transition_ns
        source.decide(self.fixture.view(save))
        cutoff = save + 1
        source.decide(self.fixture.view(cutoff))
        events = copy.deepcopy(source.context_events)
        exported = self.export(source, cutoff)
        ready = cp.restore_scheduler_snapshot(target, exported, binding=self.binding(cutoff))
        count = source.oracle.hazard['state']['visible_update_count']
        self.assertEqual(target.context_events, events)
        self.assertEqual(target.oracle.hazard['state']['visible_update_count'], count)
        restore = source.pending_switch.restore_start_ns
        self.assertGreater(restore, cutoff)
        self.assertEqual(source.decide(self.fixture.view(restore)), target.decide(self.fixture.view(restore)))
        self.assertEqual(cp._state(source, ready), cp._state(target, ready))
        self.assertEqual(target.restore_start_ns, restore)
        self.assertEqual(len(target.oracle.decisions), 1)

    def test_physical_v56_mark_roundtrip_keeps_all_resident_states(self):
        source, target = self.scheduler(physical_mark=True), self.scheduler(physical_mark=True)
        source.decide(self.fixture.view(0))
        ready = cp.restore_scheduler_snapshot(target, self.export(source, 0), binding=self.binding(0))
        self.assertEqual(cp._state(source, ready), cp._state(target, ready))
        kwargs = dict(held_owner='text#0', context_age_ns=0, pending_frontier=[])
        a = source.oracle.next_owner.predict_given_save_now(**kwargs)
        b = target.oracle.next_owner.predict_given_save_now(**kwargs)
        self.assertEqual(a, b)
        self.assertEqual(len(b['physical_owner_probabilities']), 4)
        self.assertFalse(b['physical_context_probabilities_filtered'])

    def test_v57_state_hazard_with_real_v24_controller_and_v56_mark_roundtrips(self):
        source, target = self.scheduler(physical_mark=True), self.scheduler(physical_mark=True)
        for scheduler in (source, target):
            scheduler.oracle.controller = CumulativeLifecycleHazard(
                cp.ProjectedStateSaveHazard(state_hazard_fixture.model()),
                age_edges_ns=scheduler.oracle.controller.edges, sources=[receipt(self.fixture.source)],
                candidate_identity='d'*64, allow_unvalidated_approximation=True)
        source.decide(self.fixture.view(0))
        source.decide(self.fixture.view(1000))
        ready = cp.restore_scheduler_snapshot(target, self.export(source, 1000), binding=self.binding(1000))
        self.assertEqual(cp._state(source, ready), cp._state(target, ready))
        self.assertEqual(source.decide(self.fixture.view(2000)), target.decide(self.fixture.view(2000)))
        self.assertEqual(cp._state(source, ready), cp._state(target, ready))

    def test_owner_last_ready_is_explicit_detached_and_can_be_in_future(self):
        source, target = self.scheduler(), self.scheduler()
        source.decide(self.fixture.view(0))
        exported = self.export(source, 0)
        ready = cp.restore_scheduler_snapshot(target, exported, binding=self.binding(0))
        self.assertEqual(ready, self.ready(source))
        ready['text#0'] = 999
        self.assertNotEqual(ready, cp.decode(exported['state'])['owner_last_ready'])
        with self.assertRaisesRegex(ValueError, 'owner_last_ready'):
            cp.export_scheduler_snapshot(source, binding=self.binding(0), owner_last_ready={})

    def test_resealed_changed_budget_rejected_before_any_target_mutation(self):
        source, target = self.scheduler(), self.scheduler()
        source.decide(self.fixture.view(0))
        exported = self.export(source, 0)
        before = copy.deepcopy((vars(target), vars(target.oracle), vars(target.oracle.next_owner)))
        state = cp.decode(exported['state'])
        hazard = state['oracle']['hazard']
        hazard['state']['hazard_budget'] += 1
        hazard['hazard_budget'] += 1
        hazard['state']['state_sha256'] = digest({key: value for key, value in hazard['state'].items() if key != 'state_sha256'})
        exported['state'] = cp.encode(state)
        exported['checkpoint_sha256'] = digest({key: value for key, value in exported.items() if key != 'checkpoint_sha256'})
        with self.assertRaisesRegex(ValueError, 'budget'):
            cp.restore_scheduler_snapshot(target, exported, binding=self.binding(0))
        # Compare only encoded dynamic state: deepcopy changes object identity.
        self.assertEqual(cp._state(target, self.ready(target)),
            dict(scheduler={k: before[0][k] for k in cp.SCHEDULER_FIELDS},
                 oracle={k: before[1][k] for k in cp.ORACLE_FIELDS},
                 mark={k: before[2][k] for k in ('states', 'clock')}, owner_last_ready=self.ready(target)))

    def test_resealed_pending_time_change_and_binding_change_are_rejected(self):
        source, target = self.scheduler(), self.scheduler()
        save = source.decide(self.fixture.view(0)).next_transition_ns
        source.decide(self.fixture.view(save))
        exported = self.export(source, save)
        state = cp.decode(exported['state'])
        state['scheduler']['context_events'][0] = (save + 9, 'RESTORE_START', 'text#1')
        exported['state'] = cp.encode(state)
        exported['checkpoint_sha256'] = digest({key: value for key, value in exported.items() if key != 'checkpoint_sha256'})
        with self.assertRaisesRegex(ValueError, 'handoff events'):
            cp.restore_scheduler_snapshot(target, exported, binding=self.binding(save))
        original = self.export(source, save)
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            cp.restore_scheduler_snapshot(target, original, binding=dict(self.binding(save), graph_sha256='f'*64))

    def test_unknown_stateful_fields_classes_and_pre_service_initial_events_reject(self):
        source = self.scheduler()
        source.decide(self.fixture.view(0))
        source.untracked_state = 1
        with self.assertRaisesRegex(ValueError, 'untracked fields'):
            self.export(source, 0)
        del source.untracked_state
        source.oracle.handoff = object()
        with self.assertRaisesRegex(ValueError, 'Unsupported static component'):
            self.export(source, 0)
        source = self.scheduler()
        source.decide(self.fixture.view(0))
        source.context_events = [(1, 'SERVICE_READY', 'text#0')]
        with self.assertRaisesRegex(ValueError, 'Initial pre-service'):
            self.export(source, 0)

    def test_source_change_rejected_and_pure_import_leaves_target_unchanged(self):
        source, target = self.scheduler(), self.scheduler()
        source.decide(self.fixture.view(0))
        exported = self.export(source, 0)
        before = cp._state(target, self.ready(target))
        decoded = cp.import_scheduler_snapshot(target, exported, binding=self.binding(0))
        decoded['scheduler']['held_owner'] = None
        self.assertEqual(cp._state(target, self.ready(target)), before)
        self.fixture.source.write_text('Changed controller evidence')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            cp.restore_scheduler_snapshot(target, exported, binding=self.binding(0))
        self.assertEqual(cp._state(target, self.ready(target)), before)

    def test_main_engine_still_rejects_component_and_codec_rejects_nonfinite(self):
        source = self.scheduler()
        source.decide(self.fixture.view(0))
        exported = self.export(source, 0)
        engine = engine_fixture.EngineTests()
        engine.setUp()
        self.addCleanup(engine.doCleanups)
        with self.assertRaisesRegex(ValueError, 'checkpoint state is not implemented'):
            engine.engine().forecast(engine.rows(), 0, 30, checkpoint=exported)
        self.assertFalse(exported['main_engine_resume_supported'])
        with self.assertRaises(ValueError):
            cp.encode(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint key'):
            cp.decode(['dict', [[['scalar', 'x'], ['scalar', 1]], [['scalar', 'x'], ['scalar', 2]]]])


if __name__ == '__main__':
    unittest.main()
