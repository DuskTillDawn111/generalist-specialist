import copy
import unittest
from tools.predictor_validation_20261027.known_input_campaign_seal_v5 import check_complete,public_inputs
from tools.predictor_validation_20261027.forecast_gate_v2 import _public_cohort
from gsdeploy.forecasting.workload import FutureWorkload


class CompleteSealPreflightTests(unittest.TestCase):
    def test_future_schedule_uses_offsets_and_full_cohort_parser_and_workload(self):
        plan=dict(aliases={'a':{}},window_seconds=1.,drain_seconds=1.,workload_rps={'text':2.},
            trace=[dict(request_id='initial',pool_id='input',task='text',offset_ns=0),
                   dict(request_id='future',pool_id='input',task='text',offset_ns=500_000_000)])
        document,routes=public_inputs(plan,dict(routes={'initial':'a','future':'a'}))
        public,cohort=_public_cohort(document)
        self.assertEqual(cohort['future']['arrival_ns'],500_000_000)
        schedule=public['future_workload']['declared_arrivals']
        self.assertEqual(set(schedule[0]),{'request_id','task','pool_id','offset_ns'})
        arrivals=FutureWorkload(**public['future_workload']).suffix(0,0,1_000_000_000)['arrivals']
        self.assertEqual(arrivals[0]['arrival_ns'],500_000_000)
        poisoned=copy.deepcopy(document);row=poisoned['forecast_inputs']['future_workload']['declared_arrivals'][0]
        row['arrival_ns']=row.pop('offset_ns')
        with self.assertRaisesRegex(ValueError,'identity/time'):_public_cohort(poisoned)

    def test_null_terms_timings_grid_or_missing_candidates_prohibit_collection(self):
        protocol=dict(validation_domain={'run':[dict(request_id='r')]},candidates=[dict(id='a'),dict(id='b')],
            queue_sample_offsets_ns=[0,10])
        predictions=dict(cells=[dict(run_id='run',candidate_id=cid,
            requests=[dict(request_id='r',service_ms=1,queue_ms=0,e2e_ms=2)],terms=dict.fromkeys('QTLPME',1.),
            queue_depth_samples=[dict(offset_ns=n,depth=0) for n in (0,10)]) for cid in ('a','b')])
        self.assertEqual(check_complete(protocol,predictions)['request_rows'],2)
        for change in (lambda p:p['cells'].pop(),lambda p:p['cells'][0]['terms'].update(E=None),
                       lambda p:p['cells'][0]['requests'][0].update(service_ms=None),
                       lambda p:p['cells'][0]['queue_depth_samples'].pop()):
            changed=copy.deepcopy(predictions);change(changed)
            with self.assertRaises(ValueError):check_complete(protocol,changed)


if __name__=='__main__':unittest.main()
