"""Independent v3 timing/cohort negatives; all execution dependencies mocked."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_owned_training_review_v3 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v3 as collector
from tools.predictor_validation_20261027 import twoFIFO_owned_training_review_v3 as review


class CollectorContractIndependentTests(unittest.TestCase):
    def test_existing_attempt_cannot_be_reexecuted_and_original_budget_not_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=Path(directory)/'plan.json';plan.write_text('{}');collection=Path(directory)/'collection';collection.mkdir()
            with patch.object(collector,'audit_plan',return_value={}),patch.object(collector.supervisor,'public_environment_gate'),patch.object(collector.supervisor,'supervise') as launch:
                with self.assertRaisesRegex(ValueError,'no retries'):collector.execute(plan,collection)
            launch.assert_not_called()

    def test_nested_complete_but_duplicate_preparation_identity_is_rejected(self):
        plan,raw,ledger=fixtures.fixture()
        raw['case_preparations'][0]['prior_history_index']=1
        with self.assertRaisesRegex(ValueError,'arrival preparation'):review.inspect_checkpoint_activity(plan,raw,ledger)

    def test_mutant_known_drain_or_public_arrival_digest_does_not_pass_timing_only(self):
        for kind in ('drain','arrival'):
            plan,raw,ledger=fixtures.fixture()
            if kind=='drain':raw['cases'][0]['fixed_drain_deadline_ns']+=1
            else:plan['cases'][0]['arrivals'][0]['arrival_ns']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):review.inspect_cohort(plan,raw)

    def test_preserved_cross_owner_parallelism_and_no_global_one_or_two_limit(self):
        plan,raw,ledger=fixtures.fixture()
        result=review.inspect_cohort(plan,raw)
        self.assertEqual(result['issues'],[]);self.assertFalse(result['global_admission_cap_added'])
        self.assertEqual(raw['request_records'][0]['observed_worker_start_time_ns'],raw['request_records'][2]['observed_worker_start_time_ns'])

    def test_unknown_extra_activity_even_outside_protected_window_is_rejected(self):
        plan,raw,ledger=fixtures.fixture();extra=copy.deepcopy(ledger['checkpoint_spans'][-1])
        extra.update(span_id=len(ledger['checkpoint_spans']),kind='unreviewed_full_copy',start_ns=extra['end_ns']+1,end_ns=extra['end_ns']+2)
        ledger['checkpoint_spans'].append(extra)
        with self.assertRaises(ValueError):review.inspect_checkpoint_activity(plan,raw,ledger)


if __name__=='__main__':unittest.main()
