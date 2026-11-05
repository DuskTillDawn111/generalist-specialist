"""Causal request ordinals are public state, never future outputs or hidden history."""
import copy
from dataclasses import replace
import math
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import NativeContextProvider
from gsdeploy.forecasting.native_order import SCHEMA,identity,request_orders
from tests.test_native_paper_integration import activity_config
from tests.test_paper_integration import evaluate


def ordered_config():
    c=activity_config();data=copy.deepcopy(c.catalog)
    for entry in data['paper_profiled_fifo_v1']['deployments']:
        if 'g' not in entry['deployment']:continue
        order=dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
            execution_context_sha256=entry['profiles']['execution_context_sha256'],
            descriptor='post_prime_owner_request_ordinal',physical_cause_identified=False,
            groups=[dict(family='g',task='text',segment_id='gpu',accepted=True,
                independent_run_ids=['analytic-one','analytic-two'],tau_requests=8.,
                saturation_ordinal=32,coefficient=math.log(2.))])
        entry['profiles']['native_order_law']=order
        entry['profiles']['native_activity_law']['native_order_law_sha256']=identity(order)
        entry['episode_protocol']=dict(warmup_seconds=0.,priming_policy='largest_three_per_task',
            drain_seconds=2.,arrival_process='deterministic_periodic')
    return replace(c,catalog=data)


class NativeOrderTests(unittest.TestCase):
    def test_static_route_assigns_fifo_owner_counts_and_reprices_baseline(self):
        c=ordered_config();result=evaluate(c,{'g':1},{'text':{'g':2.}},trace=((0.,'text'),(0.,'text')))
        self.assertTrue(result['eligible'],result['reason'])
        self.assertEqual([r['native_order'] for r in result['request_mapping']],[0,1])
        finishes=result['forecast']['finishes']
        self.assertAlmostEqual(finishes['0']['handler_end_ns'],160_000_000,delta=2)
        self.assertAlmostEqual(finishes['1']['start_ns'],180_000_000,delta=2)
        expected=180_000_000+80e6*math.exp(math.log(2.)*math.exp(-1/8))+5e6
        self.assertAlmostEqual(finishes['1']['finish_ns'],expected,delta=2)
        self.assertFalse(result['native_eligible'])

    def test_ordinal_is_per_owner_and_keeps_precise_zero_route_boundary(self):
        c=ordered_config();result=evaluate(c,{'g':1,'s':1},{'text':{'g':1.,'s':1.}},
            trace=((0.,'text'),(0.,'text'),(.4,'text'),(.4,'text')))
        self.assertEqual([r['native_order'] for r in result['request_mapping']],[0,0,1,1])
        result=evaluate(c,{'g':1,'s':1},{'text':{'g':0.,'s':2.}},trace=((0.,'text'),(0.,'text')))
        self.assertTrue(all(r['alias']=='s#0' for r in result['request_mapping']))

    def test_unknown_history_and_changed_order_law_cannot_silently_use_steady(self):
        c=ordered_config();entry=next(e for e in c.catalog['paper_profiled_fifo_v1']['deployments'] if e['deployment']=={'g':1})
        provider=NativeContextProvider(entry['profiles'],entry['profiles']['execution_context_sha256'],allow_cpu_fixture=True)
        with self.assertRaisesRegex(UnknownEvidence,'ORDINAL'):
            provider.plan(dict(alias='g#0',task='text',pool_id='x'),{'g#0':{'family':'g'}},'mean')
        changed=copy.deepcopy(entry['profiles']);changed['native_order_law']['groups'][0]['coefficient']=0.
        with self.assertRaisesRegex(ValueError,'bound request-order'):
            NativeContextProvider(changed,changed['execution_context_sha256'],allow_cpu_fixture=True)

    def test_request_order_uses_actual_fifo_owner_sequence(self):
        raw=dict(request_records=[dict(request_id='one',model='a'),dict(request_id='two',model='b'),dict(request_id='three',model='a')])
        self.assertEqual(request_orders(raw),{'one':0,'two':0,'three':1})

    def test_table_extrema_are_not_transferred_as_empirical_law_bounds(self):
        from gsdeploy.continuous import ImmutableContinuousProblem,SessionPredictorOracle
        from gsdeploy.continuous.paper_bounds import PaperDeploymentBounds
        problem=ImmutableContinuousProblem(ordered_config())
        with SessionPredictorOracle(problem) as oracle:
            primitives=PaperDeploymentBounds(problem,oracle)._primitives({'g':1})
            self.assertTrue(all(value==0. for value in primitives['handlers'].values()))
            self.assertEqual(primitives['memory'],0.)
            self.assertEqual(primitives['power'],0.)


if __name__=='__main__':unittest.main()
