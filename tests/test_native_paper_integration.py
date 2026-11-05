"""Empirical activity providers share original FIFO/quality/rho/utility code."""
import copy
from dataclasses import replace
import math
import unittest

from gsdeploy.forecasting.native_context import fit_native_progress
from gsdeploy.forecasting.native_order import SCHEMA as ORDER_SCHEMA, identity as order_identity
from tests.test_paper_integration import fixture_config,evaluate


def activity_config():
    c=fixture_config();data=copy.deepcopy(c.catalog)
    for entry in data['paper_profiled_fifo_v1']['deployments']:
        template=entry['profiles'];observations=[]
        for run in ('one','two'):
            for profile in template['profiles']:
                alias=profile['family']+'#0'
                for key,context in template['contexts'].items():
                    if alias not in context['members']:continue
                    speed=context['scenarios']['mean']['progress'][alias]
                    observations.append(dict(id=f'{run}:{alias}:{key}',split='training',run_id=run,
                        execution_context_sha256=template['execution_context_sha256'],target_alias=alias,
                        profile_id=profile['id'],segment_id='gpu',intervals=[dict(
                            duration_ms=profile['segments'][0]['duration_ms']['mean']/speed,members=context['members'])]))
        law=fit_native_progress(template,observations)
        names=[tuple(n) for n in law['categories']]
        # Analytic CPU fixture energy. No physical telemetry or bound is claimed.
        law['resources']=dict(terms=[[]]+[[i] for i in range(len(names))],coefficients=[20.]+[40.]*len(names),
            support_vertices=[[int(a in context['members']) for a in sorted(f+'#0' for f in entry['deployment'])]
                              for context in template['contexts'].values()],resident_observed_peak_mib=20.*len(entry['deployment']),
            independent_run_ids=['analytic-one','analytic-two'])
        template['contexts']={};template['native_activity_law']=law
    return replace(c,catalog=data)


class NativePaperIntegrationTests(unittest.TestCase):
    def order_config(self):
        c=activity_config()
        for entry in c.catalog['paper_profiled_fifo_v1']['deployments']:
            template=entry['profiles'];context=template['execution_context_sha256']
            law=dict(schema=ORDER_SCHEMA,status='FITTED_UNVALIDATED',execution_context_sha256=context,
                descriptor='post_prime_owner_request_ordinal',physical_cause_identified=False,
                groups=[dict(family=p['family'],task=p['task'],segment_id='gpu',accepted=True,
                    independent_run_ids=['one','two'],tau_requests=1.,saturation_ordinal=32,
                    coefficient=math.log(2.)) for p in template['profiles']])
            template['native_order_law']=law
            template['native_activity_law']['native_order_law_sha256']=order_identity(law)
            entry['episode_protocol']=dict(warmup_seconds=1.,priming_policy='largest_three_per_task',
                                           drain_seconds=2.,arrival_process='deterministic_periodic')
        return c

    def test_warmup_counts_advance_order_before_formal_requests(self):
        c=self.order_config()
        result=evaluate(c,{'s':1},{'text':{'s':2.}},trace=((0.,'text'),))
        self.assertTrue(result['eligible'],result.get('reason'))
        self.assertEqual(result['post_prime_owner_ordinals_at_measurement'],{'s#0':2})
        self.assertEqual(result['request_mapping'][0]['native_order'],2)
        expected_ms=100.*math.exp(math.log(2.)*math.exp(-2.))
        self.assertAlmostEqual(result['forecast']['finishes']['0']['handler_end_ns'],expected_ms*1e6,delta=2)
        self.assertFalse(result['independent_accuracy_qualified'])

    def test_warmup_has_its_own_router_and_cannot_default_to_zero_history(self):
        c=self.order_config();route={'text':{'g':1.,'s':1.}}
        result=evaluate(c,{'g':1,'s':1},route,trace=((0.,'text'),(0.,'text')))
        self.assertEqual(result['post_prime_owner_ordinals_at_measurement'],{'g#0':1,'s#0':1})
        self.assertEqual([r['native_order'] for r in result['request_mapping']],[1,1])
        del c.catalog['paper_profiled_fifo_v1']['deployments'][-1]['episode_protocol']
        rejected=evaluate(c,{'g':1,'s':1},route,trace=((0.,'text'),(0.,'text')))
        self.assertIsNone(rejected['U'])
        self.assertIn('MATCHED_POST_PRIME_WARMUP_PROTOCOL',rejected['reason'])

    def test_activity_law_uses_same_fifo_receipt_and_utility_arithmetic(self):
        c=activity_config();old=fixture_config();route={'text':{'g':1.,'s':1.}};trace=((0.,'text'),(0.,'text'))
        actual=evaluate(c,{'g':1,'s':1},route,trace=trace)
        expected=evaluate(old,{'g':1,'s':1},route,trace=trace)
        self.assertTrue(actual['eligible'],actual['reason'])
        self.assertAlmostEqual(actual['U'],expected['U'],places=6)
        for rid in ('0','1'):
            self.assertAlmostEqual(actual['forecast']['finishes'][rid]['finish_ns'],
                                   expected['forecast']['finishes'][rid]['finish_ns'],delta=2)
        self.assertEqual(actual['rho'],expected['rho'])
        self.assertFalse(actual['native_eligible']);self.assertFalse(actual['independent_accuracy_qualified'])

    def test_unqualified_empirical_provider_does_not_gain_physical_admission(self):
        c=activity_config();route={'text':{'s':2.,'g':0.}}
        result=evaluate(c,{'g':1,'s':1},route,trace=((0.,'text'),(0.,'text')))
        self.assertTrue(result['eligible'])
        self.assertFalse(result['final_gain_eligible'])
        self.assertAlmostEqual(result['forecast']['finishes']['1']['start_ns'],120_000_000,delta=2)

    def test_private_optimizer_session_loads_native_provider_without_thawing_inputs(self):
        from gsdeploy.continuous import Budget,ImmutableContinuousProblem,SessionPredictorOracle
        problem=ImmutableContinuousProblem(activity_config())
        with SessionPredictorOracle(problem) as oracle:
            result=oracle.predict(problem,{'g':1,'s':1},{'text':{'g':1.,'s':1.}},
                                  problem.train_traces[0],Budget(5.,4,4))
            self.assertTrue(result['eligible'],result.get('reason'))
            self.assertFalse(result['native_eligible'])


if __name__=='__main__':unittest.main()
