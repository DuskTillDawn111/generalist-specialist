"""Temporal descriptors use causal starts and explicit numerical events only."""
import copy
import math
import unittest
from dataclasses import replace

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import NativeContextProvider,fit_native_progress
from gsdeploy.forecasting.native_temporal import (SCHEMA,AGE_EDGES,LEGACY_AGE_EDGES,age_bin,fit_temporal_progress,
    BOUNDED_SCHEMA,BOUNDED_AGE_EDGES,BOUNDED_POLICY)
from gsdeploy.forecasting.native_training import fit_native_captures
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA,validate_native_fit
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_native_activity_context import fixture
from tests.test_native_training import capture,POOL
from tests.test_native_validation import validation


def temporal_template():
    template,observations=fixture();law=fit_native_progress(template,observations);groups=[]
    for family,task in [('a','text'),('b','image')]:
        groups.append(dict(family=family,task=task,segment_id='native',terms=[['count',1],['age',1]] if family=='a' else [['one']],
            coefficients=[math.log(.5),math.log(2)/math.log(1.5)] if family=='a' else [0.],
            profile_ids=['a1','a2'] if family=='a' else ['b1'],independent_run_ids=['analytic-one','analytic-two'],
            support_vertices=[[0.,0.],[0.,1.]] if family=='a' else [[0.,0.],[1.,0.]],
            feature_envelope=dict(size=[[-2.,2.],[-2.,2.]],age=[[0.,math.log(8.)],[0.,math.log(8.)]]),
            own_age_envelope=[0.,math.log(8.)]))
    law['progress_groups']=[];law['temporal_progress']=dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
        execution_context_sha256='fixture',admission_enabled=False,physical_bound=False,
        age_edges=AGE_EDGES,categories=[['a','text'],['b','image']],medians_ms=[15.,5.],groups=groups)
    template['native_activity_law']=law
    return template


class NativeTemporalTests(unittest.TestCase):
    def test_declared_bounded_basis_continues_tail_with_original_unknown_guards(self):
        template=temporal_template();spec=template['native_activity_law']['temporal_progress']
        spec.update(schema=BOUNDED_SCHEMA,age_edges=BOUNDED_AGE_EDGES,age_descriptor_policy=BOUNDED_POLICY)
        for g in spec['groups']:
            g.update(terms=[['one']],coefficients=[math.log(.1)])
        provider=NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        aliases={'a#0':dict(family='a',supported_tasks=['text']),'b#0':dict(family='b',supported_tasks=['image'])}
        queues={'a#0':[dict(request_id='a',alias='a#0',task='text',pool_id='a1',arrival_ns=0)],
                'b#0':[dict(request_id='b',alias='b#0',task='image',pool_id='b1',arrival_ns=0)]}
        result=FullHorizonModel(aliases,provider).forecast(queues,0,1,drain_deadline_ns=1_000_000_000,
            require_energy=False,require_memory=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],101_000_000)
        self.assertEqual(result['finishes']['b']['finish_ns'],51_000_000)
        self.assertEqual(age_bin(1.,BOUNDED_AGE_EDGES),age_bin(15.9,BOUNDED_AGE_EDGES))
        active={'a#0':dict(profile_id='a1',segment_id='native',native_age_ms=160.,native_baseline_ms=10.)}
        with self.assertRaisesRegex(UnknownEvidence,'DESCRIPTOR_DOMAIN'):
            provider.native_law.progress('a#0',active['a#0'],active)
        active['a#0']['native_age_ms']=0.
        active['b#0']=dict(profile_id='b1',segment_id='native',native_age_ms=0.,native_baseline_ms=5.)
        active['b#1']=dict(active['b#0'])
        with self.assertRaisesRegex(UnknownEvidence,'COUNTS_OUTSIDE'):
            provider.native_law.progress('a#0',active['a#0'],active)

    def test_bounded_age_basis_requires_distinct_schema_and_explicit_policy(self):
        template=temporal_template();spec=template['native_activity_law']['temporal_progress']
        spec.update(age_edges=BOUNDED_AGE_EDGES)
        with self.assertRaises(ValueError):NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        spec.update(schema=BOUNDED_SCHEMA)
        with self.assertRaises(ValueError):NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        spec.update(age_descriptor_policy=BOUNDED_POLICY)
        NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        spec.update(schema=SCHEMA,age_edges=AGE_EDGES)
        with self.assertRaises(ValueError):NativeContextProvider(template,'fixture',allow_cpu_fixture=True)

    def test_age_events_reprice_before_a_competitor_finishes(self):
        template=temporal_template();provider=NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        aliases={'a#0':dict(family='a',supported_tasks=['text']),'b#0':dict(family='b',supported_tasks=['image'])}
        model=FullHorizonModel(aliases,provider)
        queues={'a#0':[dict(request_id='a',alias='a#0',task='text',pool_id='a1',arrival_ns=0)],
                'b#0':[dict(request_id='b',alias='b#0',task='image',pool_id='b1',arrival_ns=0)]}
        result=model.forecast(queues,0,1,drain_deadline_ns=1_000_000_000,require_energy=False,require_memory=False)
        gamma=math.log(2)/math.log(1.5)
        work=sum(.5*.5*(1+i*.1)**gamma for i in range(10))
        expected_ms=5.+10.-work+1.  # residual a work, then its measured 1 ms host tail
        self.assertAlmostEqual(result['finishes']['a']['finish_ns']/1e6,expected_ms,places=5)
        self.assertGreater(len(result['transitions']),10)
        self.assertEqual(result['finishes']['b']['finish_ns'],6_000_000)

    def test_missing_causal_age_and_unsupported_context_are_unknown(self):
        template=temporal_template();provider=NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        progress=provider.native_law.temporal_progress
        active={'a#0':dict(profile_id='a1',segment_id='native')}
        with self.assertRaisesRegex(UnknownEvidence,'START_OR_BASELINE'):progress.progress('a#0',active['a#0'],active)
        active['a#0'].update(native_age_ms=0.,native_baseline_ms=10.)
        active['b#0']=dict(profile_id='b1',segment_id='native',native_age_ms=0.,native_baseline_ms=5.)
        active['b#1']=dict(active['b#0'])
        with self.assertRaisesRegex(UnknownEvidence,'COUNTS_OUTSIDE'):progress.progress('a#0',active['a#0'],active)
        with self.assertRaisesRegex(UnknownEvidence,'CONTINUATION'):
            provider.plan(dict(alias='a#0',task='text',pool_id='a1',head_age_ms=1.),{'a#0':{'family':'a'}},'mean')

    def test_rounding_does_not_repeat_age_boundary_or_extrapolate_age(self):
        self.assertEqual(age_bin(.3-1e-10),3)
        with self.assertRaises(UnknownEvidence):age_bin(8.,LEGACY_AGE_EDGES)
        self.assertGreater(age_bin(8.75),0)
        with self.assertRaises(UnknownEvidence):age_bin(16.)

    def test_fitted_fixture_replays_heldout_without_changing_utility_admission(self):
        fit=fit_native_captures([capture('s1','solo'),capture('s2','solo'),capture('o1','overlap'),capture('o2','overlap')],POOL,block_seconds=.5)
        original=copy.deepcopy(fit);spec=fit_temporal_progress(fit)
        # Two identical alignments cannot identify an age effect; more columns
        # in the interval matrix do not create independent request equations.
        self.assertTrue(all(not g['accepted'] for g in spec['fit_diagnostics']))
        self.assertEqual(spec['groups'],[])
        self.assertEqual(original,fit)
        law=fit['profiles']['native_activity_law'];groups=[]
        for g in law['progress_groups']:
            groups.append(dict(g,terms=[['one'] if not t else ['count',t[0]] for t in g['terms']],
                profile_ids=[p['id'] for p in fit['profiles']['profiles'] if p['family']==g['family'] and p['task']==g['task']],
                feature_envelope=dict(size=[[-2.,2.]]*3,age=[[0.,math.log(8.)]]*3),own_age_envelope=[0.,math.log(8.)]))
        spec.update(groups=groups,medians_ms=[10.,10.,10.]);law['progress_groups']=[];law['temporal_progress']=spec
        report=validate_native_fit(fit,[validation('v1'),validation('v2')],DEFAULT_CRITERIA)
        self.assertTrue(report['development_accuracy_passed'],report['reasons'])
        self.assertFalse(report['admission_enabled'])
        self.assertFalse(report['formal_utility_qualified'])

    def test_optimizer_session_preserves_timing_under_an_immutable_temporal_catalog(self):
        from gsdeploy.continuous import Budget,ImmutableContinuousProblem,SessionPredictorOracle
        from tests.test_native_paper_integration import activity_config
        c=activity_config();catalog=copy.deepcopy(c.catalog)
        for entry in catalog['paper_profiled_fifo_v1']['deployments']:
            law=entry['profiles']['native_activity_law'];names=law['categories'];groups=[]
            for g in law['progress_groups']:
                terms=[['one'] if not term else ['count',term[0]] if len(term)==1 else ['pair',*term] for term in g['terms']]
                groups.append(dict(g,terms=terms,
                    profile_ids=[p['id'] for p in entry['profiles']['profiles'] if (p['family'],p['task'])==(g['family'],g['task'])],
                    feature_envelope=dict(size=[[-3.,3.]]*len(names),age=[[0.,math.log(8.)]]*len(names)),
                    own_age_envelope=[0.,math.log(8.)]))
            law['progress_groups']=[];law['temporal_progress']=dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
                execution_context_sha256=entry['profiles']['execution_context_sha256'],admission_enabled=False,
                physical_bound=False,categories=names,medians_ms=[80.]*len(names),age_edges=AGE_EDGES,groups=groups)
        problem=ImmutableContinuousProblem(replace(c,catalog=catalog));old=ImmutableContinuousProblem(c)
        route={'text':{'g':1.,'s':1.}};deployment={'g':1,'s':1}
        with SessionPredictorOracle(problem) as oracle,SessionPredictorOracle(old) as original:
            actual=oracle.predict(problem,deployment,route,problem.train_traces[0],Budget(5.,4,4))
            expected=original.predict(old,deployment,route,old.train_traces[0],Budget(5.,4,4))
        self.assertTrue(actual['eligible'],actual.get('reason'))
        self.assertAlmostEqual(actual['U'],expected['U'],places=6)
        self.assertFalse(actual['native_eligible'])


if __name__=='__main__':unittest.main()
