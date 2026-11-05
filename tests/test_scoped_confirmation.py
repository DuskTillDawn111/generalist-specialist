import unittest
from unittest.mock import patch

from experiments.prepare_scoped_confirmation import select_panel, realized_key, TASKS
from experiments.score_scoped_confirmation import assess_panel, score
from experiments.collect_activation_training import digest
from tests import test_scoped_routing_objective as objective_fixture


def route(share):
    return {t: {t: 1-share, 'generalist_2': share} for t in TASKS}


def run(method, share, utility=-1.):
    w = (dict(routing=route(share), score=utility) if method == 'Joint-K2' else
         dict(routing_qps=route(share), best_feasible_utility=utility))
    return method, 11, dict(winner=w, events=[])


class ConfirmationPanelTests(unittest.TestCase):
    def test_duplicate_winners_do_not_consume_two_gpu_slots(self):
        panel = select_panel([run('Joint-K2', 0.), run('Activation-BO-MADS', 0.)], dict.fromkeys(TASKS, 1.))
        self.assertEqual(len(panel), 3)
        self.assertEqual(panel[0]['represents_methods'], ['Joint-K2', 'Activation-BO-MADS'])
        self.assertEqual(panel[1]['category'], 'nongrid')
        self.assertEqual(panel[1]['source'], 'fixed_control_not_search_winner')
        self.assertEqual(panel[2]['category'], 'grid')

    def test_internal_grid_and_nongrid_winners_get_one_shared_endpoint_control(self):
        panel = select_panel([run('Joint-K2', .5), run('Activation-BO-MADS', .3)], dict.fromkeys(TASKS, 1.))
        self.assertEqual(len(panel), 3)
        self.assertEqual({r['category'] for r in panel}, {'endpoint', 'grid', 'nongrid'})
        self.assertEqual(len({realized_key(r['routing']) for r in panel}), 3)

    def test_budget_is_not_expanded_to_fit_conflicting_categories(self):
        joint = run('Joint-K2', .5)
        other = run('Activation-BO-MADS', .5)
        other[2]['winner']['routing_qps']['asr'] = {'asr': 0., 'generalist_2': 1.}
        with self.assertRaisesRegex(ValueError, 'Three-candidate budget'):
            select_panel([joint, other], dict.fromkeys(TASKS, 1.))

    def test_missing_method_cannot_be_reported_as_comparison(self):
        with self.assertRaises(ValueError):
            select_panel([run('Joint-K2', 0.)], dict.fromkeys(TASKS, 1.))


class ConfirmationRegretTests(unittest.TestCase):
    def reports(self, a_values, b_values, a_pred=-1., b_pred=-2.):
        values = []
        for candidate, actual, predicted in [('a', a_values, a_pred), ('b', b_values, b_pred)]:
            for i, value in enumerate(actual, 1):
                values.append(dict(candidate_id=candidate, round=i, U=value, predicted_U=predicted,
                    constraints_eligible=True, **{t+'_contribution': value/6 for t in 'QT LPME'.replace(' ', '')}))
        return values

    def test_near_tie_not_forced_into_strict_ranking(self):
        result = assess_panel(self.reports([-1., -1.], [-.99, -.99]))
        self.assertTrue(result['descriptive_panel_selection_accepted'])
        self.assertTrue(result['paired_comparisons'][0]['near_tie'])
        self.assertAlmostEqual(result['mean_selection_regret'], .01)
        self.assertFalse(result['population_confidence_claim'])

    def test_one_bad_round_not_hidden_by_mean(self):
        result = assess_panel(self.reports([-1., -.90], [-.95, -.95]))
        self.assertAlmostEqual(result['mean_selection_regret'], 0.)
        self.assertFalse(result['descriptive_panel_selection_accepted'])
        self.assertAlmostEqual(result['per_round'][0]['selection_regret'], .05)

    def test_measured_infeasible_selected_candidate_cannot_pass_on_high_utility(self):
        reports = self.reports([-.5, -.5], [-1., -1.])
        reports[0]['constraints_eligible'] = False
        result = assess_panel(reports)
        self.assertFalse(result['descriptive_panel_selection_accepted'])
        self.assertIsNone(result['mean_selection_regret'])

    def test_missing_or_duplicate_run_never_creates_a_paired_result(self):
        reports = self.reports([-1., -1.], [-1., -1.])
        for bad in (reports[:-1], [reports[0], reports[0], *reports[2:]]):
            with self.assertRaises(ValueError):
                assess_panel(bad)


class ConfirmationScoringTests(unittest.TestCase):
    def setUp(self):
        fixture = objective_fixture.ScopedObjectiveTests()
        fixture.setUp()
        self.system, self.evidence, plan = fixture.system, fixture.evidence, fixture.plan
        self.evidence['capacity_contract'] = 'synthetic unit-test rho contract'
        context = dict(resident_aliases=plan['aliases'], runtime_policy={'scope': 'g2_strict'})
        self.evidence['execution_context_sha256'] = digest(context)
        plan.update(execution_context_sha256=digest(context), mode='overlap', split='confirmation',
                    not_for_training=True, evaluation_role='locked_candidate_confirmation',
                    run_id='test-confirmation', candidate_id='a', round=1)
        plan['jobs'][0]['design']['routing_rps'] = {t: {t: 2.} for t in TASKS}
        ready, shutdown = {}, {}
        for i, (alias, family) in enumerate(plan['aliases'].items(), 1):
            strict = family == 'generalist_2'
            ready[alias] = dict(pid=i, clock='time.perf_counter_ns', runtime_scope=dict(
                scope='g2_strict', slot=family, strict_requested=strict, deterministic_algorithms=strict,
                cublas_workspace_config=':4096:8' if strict else None,
                deterministic_warn_only=False, cudnn_benchmark=False, cudnn_deterministic=True))
            shutdown[alias] = dict(forced=False, exit_code=0, bye_received=True)
        start = 2_000_000_000
        rows, predictions = [], []
        for i, request in enumerate(sorted(plan['jobs'][0]['requests'], key=lambda r: r['offset_ms'])):
            at = start+round(request['offset_ms']*1e6)
            rid = f'training:{i}'
            quality = [int(request['pool_id']), 1 if request['pool_id'] == '0' else 9]
            rows.append(dict(request_id=rid, profile_job_id='single', phase='training', status='completed',
                model=request['alias'], task=request['task'], pool_id=request['pool_id'],
                scheduled_time_ns=at, completion_time_ns=at+50_000_000,
                previous_handler_next_request_ready={'pid': ready[request['alias']]['pid']},
                worker_response=dict(success=True, quality=quality, output_sha256='same')))
            predictions.append(dict(request_id=rid, **request, predicted_e2e_ms=50.,
                                    predicted_finish_ns=round(request['offset_ms']*1e6)+50_000_000))
        for row in self.evidence['quality']:
            row['output_sha256'] = 'same'
        raw = dict(schema='gsdeploy.native_activity_capture.v1', status='completed', kind='cpu_fixture',
            split='confirmation', run_id=plan['run_id'], admission_enabled=False,
            execution_context_sha256=digest(context), resident_aliases=plan['aliases'],
            ready=ready, shutdown=shutdown, request_records=rows, priming_request_records=[],
            startup_begin_time_ns=0, observation_start_ns=start-1, observation_end_ns=start+1_000_000_000,
            batches=[dict(id='single', start_ns=start, arrival_end_ns=start+1_000_000_000,
                          request_ids=[r['request_id'] for r in rows])],
            board_samples=[dict(time_ns=-1, owned_used_mib=1000.), dict(time_ns=start, owned_used_mib=50.),
                           dict(time_ns=start+1_000_000_001, owned_used_mib=1000.)])
        self.bundle = dict(raw=raw, plan=plan, context=context, source='synthetic_unit_fixture')
        self.prediction = dict(U=-1., raw_peak_estimate_mib=50.,
            episode_energy_evidence={'episode_joules': 100.}, prediction=dict(jobs=[dict(
                predicted_arrival_end_ns=1_000_000_000, requests=predictions)]))

    def measured(self):
        with patch('experiments.score_scoped_confirmation.describe_energy', return_value=dict(
                coverage_valid=True, totals=[dict(window='startup_through_observation_end', joules=100.)])), \
             patch('experiments.score_scoped_confirmation.score_output', side_effect=lambda response, *args: response['quality']), \
             patch('experiments.score_scoped_confirmation.output_identity', side_effect=lambda response: response['output_sha256']):
            return score(self.bundle, self.prediction, self.system, self.evidence)

    def test_measured_six_terms_and_memory_episode_boundary(self):
        result = self.measured()
        self.assertEqual(result['owned_peak_mib'], 50.)
        self.assertAlmostEqual(result['Q_norm'], .1)
        self.assertAlmostEqual(result['U'], -(.1+0+.5+.5+.5+.1)/6)
        self.assertTrue(result['constraints_eligible'])
        self.assertEqual(result['throughput']['relative_error'], 0.)

    def test_changed_output_scored_instead_of_silently_reusing_training_quality(self):
        row = self.bundle['raw']['request_records'][0]
        row['worker_response'].update(quality=[2, 1], output_sha256='changed')
        result = self.measured()
        self.assertAlmostEqual(result['Q_norm'], .15)
        self.assertEqual(result['changed_output_ids'], [row['request_id']])

    def test_failed_output_or_changed_assignment_cannot_enter_ranking(self):
        self.bundle['raw']['request_records'][0]['worker_response']['success'] = False
        with self.assertRaises(ValueError):
            self.measured()
        self.bundle['raw']['request_records'][0]['worker_response']['success'] = True
        self.prediction['prediction']['jobs'][0]['requests'][0]['pool_id'] = 'different'
        with self.assertRaises(ValueError):
            self.measured()


if __name__ == '__main__':
    unittest.main()
