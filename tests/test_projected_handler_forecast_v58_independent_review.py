"""Independent actual fitted-model wiring tests; no base-provider compilation/GPU."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import projected_handler_forecast_v58 as v
from tools.predictor_validation_20261027 import projected_handler_preflight_v58 as p
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
from tools.predictor_validation_20261027.physical_context_mark_bridge_v56 import PhysicalContextMark
from tools.predictor_validation_20261027.projected_handler_bridge_v44 import feature_projection
from tools.predictor_validation_20261027.whole57_shared_observable_risk_v31 import OWNERS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'artifacts/predictor-projected-handler-v58-20261028-configuration01/configuration.json'


def config():
    return json.loads(CONFIG.read_text())


def construct(value):
    def base(model, *_args, **_kwargs):
        model.compiled = SimpleNamespace(sources=[])
    with patch.object(v.V16Forecast, '__init__', base):
        return v.ObservableContextForecast(value, allow_unvalidated_training_hypothesis=True)


def states(bits):
    return {o:OwnerLifecycle(bool(bits & (1 << (3-i))), False, False) for i,o in enumerate(OWNERS)}


class IndependentV58Tests(unittest.TestCase):
    def test_actual_two_model_identity_and_complete_source_receipts(self):
        value = config(); model = construct(value)
        expected = v.digest(dict(projected_handler=value['observable_context_model'],
                                 state_conditioned_SAVE=value['state_SAVE_model']))
        provenance = model.save_controller.provenance()['configuration']
        self.assertEqual(provenance['candidate_identity'], expected)
        self.assertEqual(provenance['budget_policy'], 'MEDIAN_LOG2')
        self.assertFalse(provenance['physical_execution_gate'])
        self.assertEqual(model.lifecycle_identity, v.digest(value))
        physical, state = v.checked(value['observable_context_model']), v.checked(value['state_SAVE_model'])
        for ref in v.strict_references([value, physical, state]):
            self.assertIn(ref, model._lifecycle_sources)
        names = {Path(ref['path']).name for ref in model._lifecycle_sources}
        self.assertIn('physical_context_mark_bridge_v56.py', names)
        self.assertIn('projected_state_save_hazard_v57.py', names)
        self.assertNotIn('idle_physical_save_hazard_v53.py', names - {Path(r['path']).name for r in state['sources']})
        v.verify_sources(model._lifecycle_sources)

    def test_actual_physical_raw_probabilities_all64_states_survive_request_domain(self):
        model = construct(config()); mark = model.mark
        self.assertIs(type(mark), PhysicalContextMark)
        count = 0
        for bits in range(16):
            lifecycle = states(bits)
            mark.observe_view(SimpleNamespace(clock_ns=bits, owner_lifecycle=tuple(lifecycle.items())))
            for holder in OWNERS:
                args = dict(held_owner=holder, context_age_ns=10_000, pending_frontier=[])
                predicted = mark.predict_given_save_now(**args)
                original = mark.candidate.next_owner(feature_projection(holder, None, 10_000, lifecycle))
                self.assertEqual(predicted['next_owner_probabilities'], original['physical_owner_probabilities'])
                self.assertEqual(predicted['most_likely_next_owner'], original['most_likely_physical_owner'])
                self.assertFalse(predicted['reference_service_progress_assigned'])
                self.assertFalse(predicted['physical_context_probabilities_filtered'])
                self.assertEqual(predicted, mark.predict_given_save_now(**dict(args,
                    pending_frontier=[['text#0','text','pool',['decode']]],
                    public_workload=[['text#1','future',10**12]],
                    pending_age_ns_by_owner={'text#1':10**12},
                    pending_reference_work_ns_by_owner={'text#1':10**12})))
                count += 1
        self.assertEqual(count, 64)

    def test_fresh_mark_cannot_inherit_previous_forecast_state(self):
        model = construct(config()); fresh = model._new_mark()
        model.mark.observe_view(SimpleNamespace(clock_ns=123, owner_lifecycle=tuple(states(1).items())))
        self.assertIsNone(fresh.clock); self.assertIsNone(fresh.states)
        self.assertEqual(fresh.candidate.model, model.mark.candidate.model)
        self.assertIsNone(model._new_mark().states)

    def test_real_state_change_integrates_old_rate_preserves_attained_age_and_budget(self):
        model = construct(config()); controller = model.save_controller
        fitted = v.checked(config()['state_SAVE_model'])['fitted']
        keys = [k for k,x in fitted.items() if k.startswith('text#0:') and x['exposure_ns']]
        a,b = keys[:2]
        old, new = states(int(a.split(':')[1],2)), states(int(b.split(':')[1],2))
        restore, observation, now = 700, 1700, 2700
        first = controller.start_restore('text#0', restore, [], 'one-observed-restore', old, observation_ns=observation)
        original_state = copy.deepcopy(first['state'])
        rate = fitted[a]['hazard_per_ms']
        self.assertGreater(first['absolute_save_candidate_ns'], now)
        changed = controller.update(first['state'], now, [], new)
        self.assertEqual(first['state'], original_state)
        self.assertEqual(changed['origin_restore_ns'], restore)
        self.assertEqual(changed['restore_age_ns'], now-restore)
        self.assertEqual(changed['observation_ns'], observation)
        self.assertFalse(changed['preentry_hazard_integrated'])
        self.assertAlmostEqual(changed['consumed_hazard'], rate*(now-observation)/1e6)
        self.assertEqual(changed['hazard_budget'], math.log(2))
        self.assertFalse(changed['budget_redrawn'])
        self.assertEqual(changed['state']['lifecycle']['text#0']['handler_active'], new['text#0'].handler_active)

    def test_mismatched_capture_guard_precedes_state_candidate_and_rejects_cap_option(self):
        value = config(); actual = v.checked
        def checked(ref):
            data = actual(ref)
            if ref == value['state_SAVE_model']:
                data = copy.deepcopy(data)
                data['captures'][next(iter(data['captures']))]['route'] = 'different-route'
            return data
        with patch.object(v, 'checked', side_effect=checked), patch.object(v, 'ProjectedStateSaveHazard') as candidate:
            with self.assertRaisesRegex(ValueError, 'same four'):
                construct(value)
            candidate.assert_not_called()
        for cap in (1,2):
            changed = copy.deepcopy(value); changed['options']['global_request_cap'] = cap
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                construct(changed)

    def test_original_four_cohort_preflight_is_exact_copy_except_entry_and_schema(self):
        old = Path(p.__file__).with_name('projected_handler_preflight_v50.py').read_text()
        self.assertEqual(Path(p.__file__).read_text(), old.replace('projected_handler_forecast_v50',
            'projected_handler_forecast_v58').replace('CPU_preflight.v50','CPU_preflight.v58'))
        self.assertEqual(p.CANDIDATES, ('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high'))


if __name__ == '__main__': unittest.main()
