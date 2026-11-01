"""Independent analytic survival and exposure checks; no GPU or raw fits."""
import copy
import math
import unittest

from tools.predictor_validation_20261027 import reader_completion_hazard_v46 as v


def state(age=0,peer=False):
    return dict(alias='text#0',task='text',pool_id='fixture',attained_reader_handler_age_ns=age,
        active_handler_by_owner={o:o=='text#0' or (peer and o=='text#1') for o in v.OWNERS})


def interval(lo,hi,done,peer=False):
    return dict(capture='tiny',request_id='r',cohort='measured',origin_reader_START_ns=100,start_ns=100+lo,end_ns=100+hi,
        age_entry_ns=lo,age_exit_ns=hi,completion_observed=done,features=state(lo,peer))


def model(include_peers=True):
    fs=v.tokens(state(),include_peers)
    return dict(columns=['intercept','age_bin:0','age_bin:1','age_bin:2'],
        coefficients=[0.,math.log(2),math.log(4),math.log(8)],
        age_edges_ns=list(v.AGE_EDGES_NS),include_peers=include_peers,vocabulary=list(fs),
        state_age_support_ns={v.digest(fs):[[0,20_000_000_000]]})


class IndependentCompletionChecks(unittest.TestCase):
    def test_attained_age_crosses_two_bins_with_analytic_conditional_survival(self):
        m=v.ReaderCompletionCandidate(model())
        # From age100ms:25ms at2/s,125ms at4/s,50ms at8/s.
        expected_hazard=.025*2+.125*4+.05*8
        self.assertAlmostEqual(m.predict(state(100_000_000),200_000_000)['candidate_completion_CDF'],
            1-math.exp(-expected_hazard),places=13)
        target_q=1-math.exp(-.75)
        # H=.05+.5+.2 reaches age275ms:175ms remaining, not275ms.
        self.assertAlmostEqual(m.quantile(state(100_000_000),target_q)['candidate_residual_ns'],175_000_000,delta=1)

    def test_event_on_bin_edge_counts_once_and_integer_exposure_is_conserved(self):
        rows=[interval(0,125_000_000,False),interval(125_000_000,250_000_000,True,True)]
        data=v.aggregate(rows,True)
        self.assertEqual(data['counts']['exposure_ns'],250_000_000)
        self.assertEqual(data['counts']['completion_events'],1)
        cells=data['rows'];self.assertEqual(sum(r['exposure_ns'] for r in cells),250_000_000)
        self.assertEqual([(r['age_bin'],r['events']) for r in cells],[(0,0),(1,1)])
        self.assertEqual([x['features']['attained_reader_handler_age_ns'] for x in v.landmarks(rows)],
            [0,100_000_000])

    def test_unseen_peer_state_keeps_full_denominator_but_is_excluded_from_common_support(self):
        rows=[interval(0,200_000_000,True,True)]
        result=v.evaluate(model(True),model(False),rows)
        self.assertEqual(result['landmarks'],2)
        for h in v.HORIZONS_NS:
            values=result['per_horizon'][str(h)]
            self.assertEqual(values['all_landmarks_including_flagged']['n'],2)
            self.assertEqual(values['both_models_observed_state_age_support']['n'],0)
        self.assertTrue(result['full_population_quantile_diagnostic']['includes_flagged_extrapolation'])
        self.assertFalse(result['six_term_utility_produced'])

    def test_episode_duplicate_gap_and_missing_terminal_are_rejected(self):
        valid=[interval(0,100_000_000,False),interval(100_000_000,200_000_000,True)]
        self.assertEqual(len(v.episodes(valid)),1)
        duplicate=[*copy.deepcopy(valid),copy.deepcopy(valid[-1])]
        gap=[valid[0],interval(110_000_000,200_000_000,True)]
        missing=[interval(0,200_000_000,False)]
        for invalid in (duplicate,gap,missing):
            with self.assertRaises(ValueError):v.episodes(invalid)
            with self.assertRaises(ValueError):v.fit(invalid,include_peers=True)

    def test_input_identity_drift_and_fabricated_origin_cannot_enter_fit(self):
        rows=[interval(0,100_000_000,False),interval(100_000_000,200_000_000,True)]
        for field in ('alias','task','pool_id'):
            changed=copy.deepcopy(rows)
            changed[-1]['features'][field]='ocr#0' if field=='alias' else 'changed'
            if field=='alias':changed[-1]['features']['active_handler_by_owner']['ocr#0']=True
            with self.assertRaises(ValueError):v.fit(changed,include_peers=True)
        changed=copy.deepcopy(rows);changed[-1]['origin_reader_START_ns']+=1
        with self.assertRaises(ValueError):v.fit(changed,include_peers=True)


if __name__=='__main__':unittest.main()
