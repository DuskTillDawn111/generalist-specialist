"""Small CPU fixtures for v50 wiring, dedicated forks, and lossless outputs."""
import ast
import contextlib
import copy
import gzip
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_forecast_v44 as old_factory
from tools.predictor_validation_20261027 import projected_handler_forecast_v50 as factory
from tools.predictor_validation_20261027 import projected_handler_preflight_v50 as runner
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v49 import LearnedLifecycleScheduler
from tools.predictor_validation_20261027.native_collect import receipt


class PrivateNumericFixture:
    """No learned providers: mutation detects any reused process/model history."""
    instances = []
    fail_candidate = None

    def __init__(self, configuration, **kwargs):
        self._lifecycle_sources = [receipt(configuration['source'])]
        self.bundle = dict(startup=dict(training_model={}))
        self.history = []
        self.numbers = np.array([2., 3.])
        self.instances.append(self)

    def execute(self, plan_ref, cid, public, routes, startup_ref, **kwargs):
        if cid == self.fail_candidate:
            raise RuntimeError('independent injected CPU fixture failure')
        prior_history, prior_numbers = copy.deepcopy(self.history), self.numbers.tolist()
        self.history.append(cid)
        self.numbers *= runner.CANDIDATES.index(cid)+2
        rows = [dict(request_id=str(i), alias='text#0', task='text', pool_id=str(i), arrival_ns=0)
            for i in range(57)]
        return dict(status='UNKNOWN', reason='INDEPENDENT_CPU_FIXTURE_ONLY', full_public_cohort=rows,
            fixture=dict(prior_history=prior_history, prior_numbers=prior_numbers,
                history=self.history, numbers=self.numbers.tolist(), process_id=os.getpid()))


class IndependentV50Tests(unittest.TestCase):
    def test_factory_changes_only_version_scheduler_and_republication_metadata(self):
        old = Path(old_factory.__file__).read_text()
        expected = old.replace('learned_lifecycle_scheduler_v27', 'learned_lifecycle_scheduler_v49')
        expected = expected.replace('v44', 'v50').replace('V44', 'V50').replace('version=44', 'version=50')
        expected = expected.replace('projected_handler_bridge_v50', 'projected_handler_bridge_v44')
        expected = expected.replace('lifecycle_only_reviews_reach_engine=True,',
            'lifecycle_only_reviews_reach_engine=True, expired_empty_frontier_reviews_republished=True,')
        trees = [ast.parse(source) for source in (expected, Path(factory.__file__).read_text())]
        for tree in trees:
            tree.body = tree.body[1:]  # Module documentation is descriptive.
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        names = {Path(r['path']).name for r in implementation_sources(factory.__file__)}
        self.assertTrue({'learned_lifecycle_scheduler_v49.py','learned_lifecycle_scheduler_v27.py',
            'projected_handler_bridge_v44.py','cumulative_lifecycle_hazard_v24.py'} <= names)

    def test_engine_builds_new_mark_oracle_scheduler_and_preserves_compiled_inputs(self):
        model = object.__new__(factory.ObservableContextForecast)
        model._execution_binding = {}, []
        model._mark_model = candidate().model
        model.initial_prior = model.handoff = model.save_controller = object()
        model.lifecycle_identity = 'a'*64
        model.aliases = {'text#0':{}}
        graph = SimpleNamespace(aliases=model.aliases, plans={})
        model.compiled = SimpleNamespace(graph=graph, duration=object(), release=object(),
            phase=object(), assert_sources_unchanged=mock.Mock())
        model.lifecycle = model.resources = object()
        model._lifecycle_sources = ()

        def build(aliases, plans, **kwargs):
            return SimpleNamespace(aliases=aliases, plans=plans,
                duration_provider=kwargs['duration_provider'], release_provider=kwargs['release_provider'],
                phase_provider=kwargs['phase_provider'], queue_diagnostics=None,
                scheduler=kwargs['scheduler'], construction=kwargs)

        with mock.patch.object(factory, 'LifecycleDagPhaseModel', side_effect=build):
            first = model.engine(max_events=123)
            first_mark = model.mark
            first.scheduler.oracle.hazard = {'prior_candidate':True}
            first_mark.clock, first_mark.states = 999, {'prior_candidate':True}
            second = model.engine(max_events=123)
        self.assertIsInstance(second.scheduler, LearnedLifecycleScheduler)
        self.assertIsNot(first.scheduler, second.scheduler)
        self.assertIsNot(first.scheduler.oracle, second.scheduler.oracle)
        self.assertIsNot(first_mark, model.mark)
        self.assertIsNone(model.mark.clock)
        self.assertIsNone(second.scheduler.oracle.hazard)
        self.assertIs(first.scheduler.oracle.controller, second.scheduler.oracle.controller)
        self.assertIs(second._engine._engine.construction['frozen_model'], graph)
        self.assertEqual(second._engine._engine.construction['max_events'], 123)
        self.assertEqual(model.compiled.assert_sources_unchanged.call_count, 2)

    def run_fixture(self, directory, fail_candidate=None):
        root = Path(directory)
        prepared = root/'prepared'
        prepared.mkdir()
        config = root/'config.json'
        source = root/'numeric-source'
        source.write_text('Small CPU process isolation fixture')
        config.write_text(json.dumps(dict(source=str(source))))
        plan = dict(candidates=[dict(id=cid, execution_context_sha256='fixture') for cid in runner.CANDIDATES],
            run_id='CPU-fixture', queue_sample_offsets_ns=[0])
        (prepared/'plan.json').write_text(json.dumps(plan))
        for cid in runner.CANDIDATES:
            (prepared/('public-'+cid+'.json')).write_text(json.dumps(dict(forecast_inputs=dict(phase_start_ns=0))))
            (prepared/('routes-'+cid+'.json')).write_text('{}')
        PrivateNumericFixture.instances = []
        with mock.patch.object(runner, 'ObservedForecast', PrivateNumericFixture), \
                mock.patch.object(PrivateNumericFixture, 'fail_candidate', fail_candidate), \
                mock.patch.object(runner, 'implementation_sources', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            if fail_candidate is None:
                report = runner.run(config, prepared, root/'output')
            else:
                with self.assertRaisesRegex(RuntimeError, 'CPU candidate processes failed'):
                    runner.run(config, prepared, root/'output')
                report = json.loads((root/'output/report.json').read_text())
        self.assertEqual(PrivateNumericFixture.instances[0].history, [])
        np.testing.assert_array_equal(PrivateNumericFixture.instances[0].numbers, [2., 3.])
        self.assertIsNone(runner._PARALLEL_MODEL)
        self.assertTrue(report['source_bytes_unchanged'])
        return report, root/'output'

    def test_dedicated_forks_keep_all_four_numeric_and_history_states_private(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as directory:
            report, output = self.run_fixture(directory)
            self.assertEqual(report['status'], 'FOUR_FULL_COHORT_CPU_PREFLIGHTS_EXECUTED')
            self.assertEqual(len({cell['independent_process_id'] for cell in report['cells']}), 4)
            for index, cell in enumerate(report['cells']):
                self.assertEqual(cell['cohort_size'], 57)
                with gzip.open(cell['result']['artifact']['path'], 'rt') as stream:
                    result = json.load(stream)
                fixture = result['fixture']
                self.assertEqual(fixture['prior_history'], [])
                self.assertEqual(fixture['prior_numbers'], [2., 3.])
                self.assertEqual(fixture['history'], [cell['candidate_id']])
                self.assertEqual(fixture['numbers'], [2.*(index+2), 3.*(index+2)])
                self.assertNotEqual(fixture['process_id'], os.getpid())
                self.assertTrue(cell['result']['complete_lossless_roundtrip_verified'])
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_one_child_failure_retains_other_results_and_reaps_all_children(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as directory:
            report, output = self.run_fixture(directory, runner.CANDIDATES[1])
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(len(report['cells']), 3)
            self.assertEqual(report['candidate_failures'][0]['candidate_id'], runner.CANDIDATES[1])
            self.assertEqual(report['candidate_failures'][0]['failure']['type'], 'RuntimeError')
            self.assertFalse((output/(runner.CANDIDATES[1]+'-forecast.json.gz')).exists())
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)


if __name__ == '__main__':
    unittest.main()
