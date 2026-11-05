"""Analytic CPU fixtures; native accuracy is intentionally not asserted."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import (NativeContextLaw,NativeContextProvider,
    fit_native_progress,fit_native_resources,within_support)
from gsdeploy.forecasting.profiles import PROFILE_SCHEMA,context_key


def fixture():
    profiles=[]
    for ident,family,task,solo in [('a1','a','text',10.),('a2','a','text',20.),('b1','b','image',5.)]:
        profiles.append(dict(id=ident,family=family,task=task,pool_id=ident,
            execution=dict(batch_size=1,backend='analytic_fixture',weight_reuse=False),segments=[
                dict(id='native',kind='gpu',duration_ms={'mean':solo},demand=dict(sm=None,dram=None,l2=None)),
                dict(id='tail',kind='host',duration_ms={'mean':1.},demand=dict(sm=0,dram=0,l2=0))]))
    template=dict(schema=PROFILE_SCHEMA,execution_context_sha256='fixture',
        provenance=dict(kind='cpu_fixture',sources=['analytic work conservation'],split='training'),
        profiles=profiles,contexts={})
    observations=[]
    for run in ('one','two'):
        for p in profiles:
            alias=p['family']+'#0';other='b#0' if alias=='a#0' else 'a#0'
            target={'profile_id':p['id'],'segment_id':'native'}
            other_target=dict(profile_id='b1' if other=='b#0' else 'a1',segment_id='native')
            for concurrent in (False,True):
                members={alias:target}
                if concurrent:members[other]=other_target
                speed=(.5 if alias=='a#0' else .8) if concurrent else 1.
                observations.append(dict(id=f'{run}:{p["id"]}:{concurrent}',split='training',run_id=run,
                    execution_context_sha256='fixture',target_alias=alias,profile_id=p['id'],segment_id='native',
                    intervals=[dict(duration_ms=p['segments'][0]['duration_ms']['mean']/speed,members=members)]))
    return template,observations


class NativeActivityTests(unittest.TestCase):
    def test_pooled_fit_uses_pinned_input_baselines_and_real_activity(self):
        template,observations=fixture();spec=fit_native_progress(template,observations)
        self.assertEqual(len(spec['progress_groups']),2)
        profiles={p['id']:p for p in template['profiles']}
        law=NativeContextLaw(spec,profiles,'fixture')
        active={'a#0':dict(profile_id='a2',segment_id='native'),
                'b#0':dict(profile_id='b1',segment_id='native')}
        self.assertAlmostEqual(law.progress('a#0',active['a#0'],active),.5,places=6)
        self.assertAlmostEqual(law.progress('b#0',active['b#0'],active),.8,places=6)
        active.pop('b#0')
        self.assertAlmostEqual(law.progress('a#0',active['a#0'],active),1.,places=6)
        self.assertFalse(spec['physical_bound']);self.assertEqual(spec['status'],'FITTED_UNVALIDATED')
        self.assertTrue(all(s['demand']['sm'] is None for p in profiles.values() for s in p['segments'] if s['kind']=='gpu'))

    def test_unobserved_replica_activity_cannot_extrapolate(self):
        template,observations=fixture();spec=fit_native_progress(template,observations)
        law=NativeContextLaw(spec,{p['id']:p for p in template['profiles']},'fixture')
        active={'a#0':dict(profile_id='a1',segment_id='native'),
            'b#0':dict(profile_id='b1',segment_id='native'),
            'b#1':dict(profile_id='b1',segment_id='native')}
        with self.assertRaisesRegex(UnknownEvidence,'OUTSIDE_TRAINING_HULL'):
            law.progress('a#0',active['a#0'],active)

    def test_same_run_repeated_requests_do_not_replace_independent_runs(self):
        template,observations=fixture()
        for row in observations:row['run_id']='one'
        fitted=fit_native_progress(template,observations)
        self.assertEqual(fitted['progress_groups'],[])
        self.assertTrue(all(not d['accepted'] for d in fitted['fit_diagnostics']))
        observations[0]['split']='confirmation'
        with self.assertRaisesRegex(ValueError,'training observations'):
            fit_native_progress(template,observations)

    def test_unidentifiable_progress_is_reported_unknown(self):
        template,observations=fixture()
        # Only concurrent cases confound intercept and the interference term.
        observations=[r for r in observations if len(r['intervals'][0]['members'])==2]
        fit=fit_native_progress(template,observations)
        self.assertEqual(fit['progress_groups'],[])
        self.assertTrue(any(d.get('rank',0)<d['parameters'] for d in fit['fit_diagnostics']))

    def test_provider_preserves_identity_and_incomplete_exact_context_checks(self):
        template,observations=fixture();template['native_activity_law']=fit_native_progress(template,observations)
        provider=NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        active={'a#0':dict(profile_id='a1',segment_id='native',task='text',pool_id='a1'),
            'b#0':dict(profile_id='b1',segment_id='native',task='image',pool_id='b1')}
        aliases={'a#0':{'family':'a'},'b#0':{'family':'b'}}
        row=dict(alias='a#0',segment_duration_ms=10.)
        self.assertAlmostEqual(provider.service(row,active,aliases,'mean')['service_ms'],20.,places=6)
        self.assertFalse(provider.power(active,aliases,'mean')['known'])
        invalid=copy.deepcopy(active);invalid['a#0']['pool_id']='foreign'
        with self.assertRaisesRegex(ValueError,'identity'):
            provider.service(row,invalid,aliases,'mean')
        base=copy.deepcopy(template)
        base['contexts'][context_key(active)]=dict(members={a:{k:r[k] for k in ('profile_id','segment_id')} for a,r in active.items()},
            scenarios={'other':dict(progress={'a#0':1.,'b#0':1.},board_watts=None,peak_memory_mib=None)},evidence='fixture')
        provider=NativeContextProvider(base,'fixture',allow_cpu_fixture=True)
        with self.assertRaises(UnknownEvidence):provider.service(row,active,aliases,'mean')

    def test_support_hull_never_fills_unobserved_joint_activity(self):
        self.assertTrue(within_support([1,0],[[0,0],[2,0]]))
        self.assertFalse(within_support([1,1],[[0,0],[1,0],[0,1]]))

    def test_board_energy_fit_integrates_activity_and_preserves_negative_coefficients(self):
        template,_=fixture()
        def run(ident):
            return dict(run_id=ident,split='training',execution_context_sha256='fixture',
                observation_window_ns=[0,10_000_000_000],telemetry_errors=[],
                spans=[dict(alias='a#0',profile_id='a1',segment_id='native',
                            start_ns=1_000_000_000,end_ns=4_000_000_000)],
                board_samples=[dict(time_ns=i*250_000_000,watts=30. if 4<=i<16 else 40.,
                    owned_used_mib=200.,foreign_pids=[]) for i in range(41)])
        resource,diagnostic=fit_native_resources(template,[run('one'),run('two')],max_gap_seconds=.3)
        self.assertTrue(diagnostic['accepted'])
        self.assertAlmostEqual(resource['coefficients'][0],40.,places=6)
        self.assertAlmostEqual(resource['coefficients'][1],-10.,places=6)
        self.assertEqual(resource['resident_observed_peak_mib'],200.)
        self.assertEqual(resource['independent_run_ids'],['one','two'])
        resource,diagnostic=fit_native_resources(template,[run('one')],max_gap_seconds=.3)
        self.assertIsNone(resource);self.assertFalse(diagnostic['accepted'])
        bad=run('two');bad['board_samples'][0]['foreign_pids']=[123]
        with self.assertRaisesRegex(ValueError,'telemetry'):
            fit_native_resources(template,[run('one'),bad],max_gap_seconds=.3)


if __name__=='__main__':unittest.main()
