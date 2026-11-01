import copy
import math
import unittest

from tools.predictor_validation_20261027 import reader_completion_hazard_v46 as v


def state(age=0, peers=()):
    return dict(alias='text#0', task='text', pool_id='p', attained_reader_handler_age_ns=age,
        active_handler_by_owner={o:o in ('text#0', *peers) for o in v.OWNERS})


def row(capture, begin, end, done, peers=()):
    return dict(capture=capture, request_id=capture+':0', cohort='measured',origin_reader_START_ns=1000,
        start_ns=begin+1000, end_ns=end+1000, age_entry_ns=begin, age_exit_ns=end,
        completion_observed=done, features=state(begin, peers))


class ReaderCompletionTests(unittest.TestCase):
    def test_fixed_rate_survival_is_conditioned_on_attained_age_without_double_wait(self):
        fs=v.tokens(state(),True)
        fit=dict(age_edges_ns=list(v.AGE_EDGES_NS),columns=['intercept'],coefficients=[math.log(2)],
            include_peers=True,vocabulary=list(fs),state_age_support_ns={v.digest(fs):[[0,10**10]]})
        model=v.ReaderCompletionCandidate(fit)
        self.assertAlmostEqual(model.predict(state(500_000_000),1_000_000_000)['candidate_completion_CDF'],1-math.exp(-2))
        self.assertAlmostEqual(model.quantile(state(500_000_000),.5)['candidate_residual_ns'],math.log(2)/2*1e9,delta=1)
        self.assertFalse(model.rate(state())['reader_RESULT_equals_worker_READY'])

    def test_peer_change_preserves_age_exposure_and_single_terminal_event(self):
        rows=[row('a',0,100_000_000,False),row('a',100_000_000,500_000_000,True,('text#1',)),
              row('b',0,100_000_000,False),row('b',100_000_000,500_000_000,True,('text#1',))]
        data=v.aggregate(rows,True)
        self.assertEqual(data['counts']['exposure_ns'],1_000_000_000)
        self.assertEqual(data['counts']['completion_events'],2)
        fitted=v.fit(rows,include_peers=True)
        self.assertTrue(fitted['optimizer']['success'])
        self.assertGreater(v.ReaderCompletionCandidate(fitted).rate(state())['hazard_per_second'],0)
        self.assertEqual(len(v.episodes(rows)),2)
        self.assertEqual(sum(x['completion_observed'] for x in rows),2)

    def test_future_endpoint_and_worker_state_are_not_reader_features(self):
        invalid=dict(state(),future_END_ns=1000)
        with self.assertRaises(ValueError):v.features(invalid)
        invalid=copy.deepcopy(state());invalid['active_handler_by_owner']['ocr#0']=None
        with self.assertRaises(ValueError):v.features(invalid)
        invalid=copy.deepcopy(state());invalid['active_handler_by_owner']['text#0']=False
        with self.assertRaises(ValueError):v.features(invalid)

    def test_unknown_states_are_reported_and_landmarks_require_survival(self):
        rows=[row('a',0,400_000_000,True)]
        points=v.landmarks(rows)
        self.assertEqual([x['features']['attained_reader_handler_age_ns'] for x in points],[0,100_000_000,250_000_000])
        fitted=v.fit(rows,include_peers=True)
        model=v.ReaderCompletionCandidate(fitted)
        p=model.predict(state(300_000_000),1_000_000_000)
        self.assertIn('FUTURE_AGE_OUTSIDE_CURRENT_READER_STATE_EXPOSURE',p['OOD_reasons'])
        self.assertGreater(p['candidate_completion_CDF'],0)
        self.assertIn('UNSEEN_READER_STATE_OR_INPUT',model.rate(state(peers=('ocr#0',)))['OOD_reasons'])


if __name__=='__main__':unittest.main()
