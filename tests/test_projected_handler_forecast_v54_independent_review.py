"""Independent actual-fit audit of v54 wiring; CPU provider shell only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import projected_handler_forecast_v54 as v
from tools.predictor_validation_20261027 import projected_handler_preflight_v54 as p
from tools.predictor_validation_20261027.implementation_source_closure_v27 import strict_references
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
from tools.predictor_validation_20261027.projected_handler_bridge_v44 import ObservableReferenceMark
from tools.predictor_validation_20261027.whole57_shared_observable_risk_v31 import OWNERS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'artifacts/predictor-projected-handler-v50-20261028-configuration01/configuration.json'
IDLE = ROOT/'artifacts/predictor-idle-physical-save-v53-20261028-training01/model.json'


def config():
    value = json.loads(CONFIG.read_text())
    value.update(schema=v.SCHEMA, options=copy.deepcopy(v.OPTIONS_V54), idle_SAVE_model=v.receipt(IDLE))
    return value


def construct(value):
    def base(model, *_args, **_kwargs):
        model.compiled = SimpleNamespace(sources=[])
    with patch.object(v.V16Forecast, '__init__', base):
        return v.ObservableContextForecast(value, allow_unvalidated_training_hypothesis=True)


class IndependentV54Tests(unittest.TestCase):
    def test_both_exact_model_refs_control_budget_identity_and_full_source_closure(self):
        value = config(); model = construct(value)
        expected = v.digest(dict(projected_handler=value['observable_context_model'], all_idle_SAVE=value['idle_SAVE_model']))
        provenance = model.save_controller.provenance()['configuration']
        self.assertEqual(provenance['candidate_identity'], expected)
        self.assertEqual(provenance['budget_policy'], 'MEDIAN_LOG2')
        self.assertEqual(model.lifecycle_identity, v.digest(value))
        idle = v.checked(value['idle_SAVE_model']); physical = v.checked(value['observable_context_model'])
        for reference in strict_references([value, idle, physical]):
            self.assertIn(reference, model._lifecycle_sources)
        names = {Path(r['path']).name for r in model._lifecycle_sources}
        self.assertIn('idle_physical_save_hazard_v53.py', names)
        self.assertNotIn('idle_physical_context_bridge_v51.py', names)
        self.assertFalse(any(name.endswith('_v51.py') for name in names))
        v.verify_sources(model._lifecycle_sources)

    def test_idle_event_still_cannot_invent_active_mark_and_new_mark_has_no_history(self):
        model = construct(config())
        self.assertIs(type(model.mark), ObservableReferenceMark)
        idle = {o:OwnerLifecycle(False, False, False) for o in OWNERS}
        result = model.save_controller.start_restore('text#0', 0, [], 'one-restore', idle)
        crossing = result['absolute_save_candidate_ns']
        terminal = model.save_controller.update(result['state'], crossing, [], idle)
        self.assertEqual(terminal['status'], 'SAVE_BUDGET_REACHED')
        self.assertFalse(terminal['budget_redrawn'])
        model.mark.observe_view(SimpleNamespace(clock_ns=crossing, owner_lifecycle=tuple(idle.items())))
        with self.assertRaisesRegex(UnknownEvidence, 'NO_VISIBLE_ACTIVE_REFERENCE_TARGET'):
            model.mark.predict_given_save_now(held_owner='text#0', context_age_ns=crossing, pending_frontier=[])
        fresh = model._new_mark()
        self.assertIsNone(fresh.clock)
        self.assertIsNone(fresh.states)
        self.assertEqual(fresh.candidate.model, model.mark.candidate.model)

    def test_active_physical_probabilities_exactly_original_v42_mark(self):
        value = config(); model = construct(value)
        original = ObservableReferenceMark(v.checked(value['observable_context_model']),
                                            allow_unvalidated_active_service_bridge=True)
        states = {o:OwnerLifecycle(o == 'text#0', None if o == 'text#0' else False, False) for o in OWNERS}
        view = SimpleNamespace(clock_ns=1_000_000, owner_lifecycle=tuple(states.items()))
        model.mark.observe_view(view); original.observe_view(view)
        kwargs = dict(held_owner='text#1', previous_held_owner='text#0', context_age_ns=1_000_000, pending_frontier=[])
        self.assertEqual(model.mark.predict_given_save_now(**kwargs), original.predict_given_save_now(**kwargs))

    def test_capture_guard_precedes_candidate_and_unknown_options_rejected(self):
        value = config(); actual = v.checked
        def read(ref):
            result = actual(ref)
            if ref == value['idle_SAVE_model']:
                result = copy.deepcopy(result)
                member = next(iter(result['captures']))
                result['captures'][member]['route'] = 'another-route'
            return result
        with patch.object(v, 'checked', side_effect=read), patch.object(v, 'IdlePhysicalSaveHazard') as candidate:
            with self.assertRaisesRegex(ValueError, 'same four'):
                construct(value)
            candidate.assert_not_called()
        value['options']['same_holder_reacquisition'] = 'IGNORE'
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            construct(value)

    def test_preflight_remains_original_four_isolated_cpu_cohorts(self):
        # This is a source-identity audit of an intentionally exact copy.
        old = Path(p.__file__).with_name('projected_handler_preflight_v50.py').read_text()
        new = Path(p.__file__).read_text()
        self.assertEqual(new, old.replace('projected_handler_forecast_v50', 'projected_handler_forecast_v54')
                         .replace('CPU_preflight.v50', 'CPU_preflight.v54'))
        self.assertEqual(p.CANDIDATES, ('shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high'))


if __name__ == '__main__': unittest.main()
