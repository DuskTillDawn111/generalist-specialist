import copy
import math
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import idle_physical_save_hazard_v53 as v
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle


def stats(t=1_000_000_000, n=0):
    return dict(rows=1 if t else 0, exposure_ns=t, events=n, unresolved_rows=0,
                unresolved_exposure_ns=0, unresolved_events=0,
                age_min_ns=0 if t else None, age_max_ns=t if t else None)


class Fallback:
    def __init__(self): self.calls = []; self.result = dict(hazard_per_ms=1., OOD_reasons=['ORIGINAL'])
    def query(self, *args):
        self.calls.append(args)
        return self.result


def candidate():
    model = dict(schema=v.SCHEMA, parameters=v.PARAMETERS, training_only=True, validation_labels_used=False,
                 sources=[], fitted={o:v.posterior(stats(0 if o == 'ocr#0' else 10**9)) for o in v.OWNERS})
    model['model_sha256'] = v.digest(model)
    return v.IdlePhysicalSaveHazard(model, Fallback())


class IdleHazardTests(unittest.TestCase):
    def test_counting_process_split_exposure_event_once_and_no_pooling(self):
        def row(left, right, event, qualified=True, active=False):
            return dict(capture='a', held_owner='text#0', driver_context=[1,2,3,4],
                restore_start_ns=0, start_ns=left, end_ns=right, age_entry_ns=left, age_exit_ns=right,
                exposure_qualified=qualified, physical_SAVE_END_observed=event,
                shared_left_features=dict(held_owner='text#0', previous_held_owner=None,
                    context_age_ns=left, handler_active={o:active for o in v.OWNERS}))
        rows = [row(0, 10, False), row(10, 20, True), row(20, 30, False, qualified=False),
                row(30, 40, True, active=True)]
        counts = v.aggregate(rows, ['a', 'b'])
        stats0 = v.pool(counts, ['a'], 'text#0')
        self.assertEqual((stats0['rows'], stats0['exposure_ns'], stats0['events']), (2, 20, 1))
        self.assertEqual(stats0['unresolved_exposure_ns'], 10)
        self.assertEqual(v.posterior(v.pool(counts, ['a'], 'text#1'))['status'], 'UNKNOWN_NO_EXPOSURE')
        self.assertEqual(v.posterior(v.pool(counts, ['b'], 'text#0'))['status'], 'UNKNOWN_NO_EXPOSURE')
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            v.aggregate([rows[0], rows[0]], ['a'])

    def test_zero_event_units_intervals_and_no_exposure(self):
        p = v.posterior(stats(t=2*10**9))
        self.assertEqual(p['shape'], .5)
        self.assertEqual(p['hazard_per_second'], .25)
        self.assertEqual(p['hazard_per_ms'], .00025)
        self.assertGreater(p['credible95_per_second'][0], 0)
        self.assertGreater(p['zero_event_frequentist_one_sided95_upper_per_second'], .25)
        self.assertEqual(v.posterior(stats(0))['status'], 'UNKNOWN_NO_EXPOSURE')
        with self.assertRaises(ValueError): v.posterior(stats(0, 1))
        self.assertEqual(v.posterior(stats(2*10**9, 3))['hazard_per_second'], 1.75)

    def test_gamma_poisson_predictive_exact_zero_and_unexposed(self):
        p = v.posterior(stats())
        result = v.predictive(p, stats())
        self.assertAlmostEqual(result['predictive_zero_probability'], math.sqrt(.5))
        self.assertAlmostEqual(result['expected_SAVE'], .5)
        self.assertTrue(result['observed_within_predictive95'])
        self.assertEqual(v.predictive(v.posterior(stats(0)), stats())['status'], 'UNKNOWN_NO_TRAINING_EXPOSURE')
        self.assertEqual(v.predictive(p, stats(0))['status'], 'NO_HELDOUT_EXPOSURE')

    def test_exact_twelve_false_and_empty_frontier_only(self):
        query = candidate(); idle = {o:OwnerLifecycle(False, False, False) for o in v.OWNERS}
        r = query.query('text#0', [], idle, 0, 10**9)
        self.assertEqual(r['hazard_per_ms'], .0005)
        self.assertAlmostEqual(r['restricted_candidate_CDF'], -math.expm1(-.5))
        self.assertAlmostEqual(r['posterior_predictive_CDF'], 1-math.sqrt(.5))
        self.assertIsNone(r['conditional_CDF'])
        self.assertIn('UNVALIDATED_HANDLER_ONLY_TO_COMPLETE_IDLE_BRIDGE', r['OOD_reasons'])
        for field in v.FIELDS:
            for value in (True, None):
                fields = {f:False for f in v.FIELDS}; fields[field] = value
                if field == 'gpu_ready' and value is True: fields['issued_outstanding'] = True
                changed = dict(idle, **{'text#0':OwnerLifecycle(**fields)})
                self.assertIs(query.query('text#0', [], changed, 0), query.fallback.result)
        self.assertIs(query.query('text#0', [('text#0','text','p',('gpu',))], idle, 0), query.fallback.result)
        with self.assertRaisesRegex(UnknownEvidence, 'NO_HELD_OWNER_EXPOSURE'):
            query.query('ocr#0', [], idle, 0)

    def test_exact_posterior_cannot_be_tampered_even_with_rehash(self):
        query = candidate(); model = copy.deepcopy(query.model)
        model['fitted']['text#0']['hazard_per_ms'] = 0
        model['model_sha256'] = v.digest({k:x for k,x in model.items() if k != 'model_sha256'})
        with self.assertRaisesRegex(ValueError, 'exact unpooled'):
            v.IdlePhysicalSaveHazard(model, Fallback())

    def test_cumulative_controller_integrates_old_rate_no_redraw_no_rewind(self):
        query = candidate(); idle = {o:OwnerLifecycle(False, False, False) for o in v.OWNERS}
        active = dict(idle, **{'text#0':OwnerLifecycle(True, True, True)})
        controller = CumulativeLifecycleHazard(query, age_edges_ns=[0],
            sources=[v.receipt(Path(v.__file__))], candidate_identity=query.model['model_sha256'],
            allow_unvalidated_approximation=True)
        started = controller.start_restore('text#0', 0, [], 'restore', idle)
        self.assertEqual(started['hazard_budget'], math.log(2))
        self.assertEqual(started['consumed_hazard'], 0)
        self.assertAlmostEqual(started['absolute_save_candidate_ns']/1e9, math.log(2)/.5, places=8)
        updated = controller.update(started['state'], 500_000_000, [], active)
        self.assertAlmostEqual(updated['consumed_hazard'], .25)
        self.assertEqual(updated['hazard_budget'], started['hazard_budget'])
        self.assertFalse(updated['budget_redrawn'])
        self.assertEqual(updated['state']['restore_key'], 'restore')
        with self.assertRaisesRegex(ValueError, 'Monotone'):
            controller.update(updated['state'], 499_999_999, [], idle)


if __name__ == '__main__': unittest.main()
