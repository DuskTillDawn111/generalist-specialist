"""Analytic host-band accounting; no physical accuracy is asserted."""
import copy
import unittest

from experiments.activation_semantic_bands import SCHEMA, BandContextProvider, band_layout, virtual_profiles
from experiments.fit_activation_semantic_bands import fit_band_captures
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import fit_native_progress
from gsdeploy.forecasting.native_energy import SCHEMA as ENERGY_SCHEMA, IDLE_EDGES_SECONDS
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_native_activity_context import fixture
from tests.test_native_training import capture, POOL


def band_template():
    template,_=fixture()
    for profile in template['profiles']:
        native=profile['segments'][0]
        duration=native['duration_ms']['mean']
        if profile['family']=='a':
            bands=[dict(native,id='native.first',native_root_id='native',native_band_role='first',
                        duration_ms={'mean':duration*.4}),
                   dict(native,id='native.continuation',native_root_id='native',native_band_role='continuation',
                        duration_ms={'mean':duration*.6})]
        else:
            bands=[dict(native,native_root_id='native',native_band_role='image_native',duration_ms={'mean':10.})]
        profile['segments']=bands+profile['segments'][1:]
    virtual,mapping=virtual_profiles(template)
    lookup={profile['id']:profile for profile in virtual}
    observations=[]
    for run in ('one','two'):
        for profile in virtual:
            target='b#0' if profile['original_family']=='b' else 'a#0'
            peers=[None]
            if target=='a#0':
                peers.append(mapping['b1','native'])
            else:
                peers.extend([mapping['a1','native.first'],mapping['a1','native.continuation']])
            for peer in peers:
                members={target:dict(profile_id=profile['id'],segment_id='native_band')}
                if peer:
                    members['a#0' if target=='b#0' else 'b#0']=dict(profile_id=peer,segment_id='native_band')
                speed=.5 if peer and profile['family']=='a::first' else 1.
                duration=profile['segments'][0]['duration_ms']['mean']/speed
                observations.append(dict(id=f'{run}:{profile["id"]}:{peer}',split='training',run_id=run,
                    execution_context_sha256='fixture',target_alias=target,profile_id=profile['id'],segment_id='native_band',
                    intervals=[dict(duration_ms=duration,members=members)]))
    law=fit_native_progress(dict(profiles=virtual,execution_context_sha256='fixture'),observations)
    assert all(group['accepted'] for group in law['fit_diagnostics'])
    template['derived_host_call_band_model']=dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
        execution_context_sha256='fixture',physical_bound=False,admission_enabled=False,
        decode_stopping_policy='training_mean_band_work_no_future_tokens',progress_law=law,
        observed_peak_memory_mib=123.)
    template['native_idle_age_energy']=dict(schema=ENERGY_SCHEMA,status='FITTED_UNVALIDATED',
        admission_enabled=False,execution_context_sha256='fixture',categories=[['a','text'],['b','image']],
        idle_edges_seconds=IDLE_EDGES_SECONDS,independent_run_ids=['one','two'],active_terms=[[]],
        coefficients=[150.,90.,90.,90.,30.],maximum_observed_idle_age_seconds=30.,
        support_vertices=[[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    return template


class SemanticBandTests(unittest.TestCase):
    def test_phase_changes_reprice_same_fifo_and_preserve_energy_union(self):
        template=band_template()
        provider=BandContextProvider(template,'fixture',allow_cpu_fixture=True)
        aliases={'a#0':dict(family='a',supported_tasks=['text']),'b#0':dict(family='b',supported_tasks=['image'])}
        queues={'a#0':[dict(request_id='a',alias='a#0',task='text',pool_id='a1',arrival_ns=0)],
                'b#0':[dict(request_id='b',alias='b#0',task='image',pool_id='b1',arrival_ns=0)]}
        result=FullHorizonModel(aliases,provider).forecast(queues,0,1,drain_deadline_ns=1_000_000_000,
            energy_state=0,require_episode_energy=True,require_memory=False,require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'],15_000_000)
        self.assertEqual(result['finishes']['b']['finish_ns'],11_000_000)
        self.assertEqual(result['final_energy_state'],14_000_000)
        self.assertAlmostEqual(result['through_drain_board_energy_joules'],2.19,places=6)
        # More unobserved peer replicas still cannot be extrapolated.
        active={'a#0':dict(profile_id='a1',segment_id='native.first',task='text',pool_id='a1'),
                'b#0':dict(profile_id='b1',segment_id='native',task='image',pool_id='b1')}
        active['b#1']=dict(active['b#0'])
        aliases['b#1']=dict(aliases['b#0'])
        with self.assertRaisesRegex(UnknownEvidence,'OUTSIDE_TRAINING_HULL'):
            provider._context(active,aliases,'mean')

    def test_training_boundaries_do_not_use_eventual_token_count_or_decode_list(self):
        root=dict(id='native',kind='gpu',start_ns=100,end_ns=200)
        def event(start,end):
            return dict(request_id='r',component='language',role='language_decode',host_start_ns=start,
                host_end_ns=end,device_event_complete=True,device_elapsed_ms=1.,stream_id=7,end_current_stream_id=7)
        row=dict(request_id='r',execution_events=[dict(event='start',segment_id='native',kind='gpu',perf_ns=100),
            dict(event='end',segment_id='native',kind='gpu',perf_ns=200)],worker_response=dict(
                start_perf_ns=100,end_perf_ns=200,observed_stages=[root],
                semantic_phase_schema='gsdeploy.semantic_native_observation.v1',
                semantic_phase_events=[event(110,140),event(150,180)],generated_tokens=999))
        observed=band_layout(row,'generalist_2')
        changed=copy.deepcopy(row)
        changed['worker_response']['generated_tokens']=1
        changed['worker_response']['semantic_phase_events'].pop()
        self.assertEqual(observed,band_layout(changed,'generalist_2'))
        self.assertEqual([(span['start_ns'],span['end_ns']) for span in observed],[(100,140),(140,200)])
        self.assertTrue(all(span['mixed_host_transfers_CUDA'] for span in observed))

    def test_energy_context_and_exact_table_cannot_be_silently_mixed(self):
        template=band_template()
        template['native_idle_age_energy']['execution_context_sha256']='foreign'
        with self.assertRaisesRegex(ValueError,'contexts differ'):
            BandContextProvider(template,'fixture',allow_cpu_fixture=True)

    def test_multiple_detection_calls_end_before_recognition(self):
        def event(component,start,end):
            return dict(request_id='r',component=component,role=component,host_start_ns=start,
                host_end_ns=end,device_event_complete=True,device_elapsed_ms=1.,stream_id=7,end_current_stream_id=7)
        row=dict(request_id='r',execution_events=[dict(event='start',segment_id='native',kind='gpu',perf_ns=100),
            dict(event='end',segment_id='native',kind='gpu',perf_ns=200)],worker_response=dict(
                start_perf_ns=100,end_perf_ns=200,observed_stages=[dict(id='native',kind='gpu',start_ns=100,end_ns=200)],
                semantic_phase_schema='gsdeploy.semantic_native_observation.v1',semantic_phase_events=[
                    event('ocr_detection',110,120),event('ocr_detection',125,140),event('ocr_recognition',150,180)]))
        bands=band_layout(row,'ocr')
        self.assertEqual(bands[0]['end_ns'],140)
        row['worker_response']['semantic_phase_events'].append(event('ocr_detection',185,190))
        with self.assertRaisesRegex(UnknownEvidence,'ORDERED_DETECTION_RECOGNITION'):
            band_layout(row,'ocr')

    def test_training_derives_separate_observations_and_retains_raw_phases(self):
        bundles=[]
        for run,mode in [('s1','solo'),('s2','solo'),('o1','overlap'),('o2','overlap')]:
            bundle=capture(run,mode)
            for row in bundle['raw']['request_records']:
                span=row['worker_response']['observed_stages'][1]
                component='language' if row['model'].startswith('generalist') else 'image_features'
                row['worker_response'].update(semantic_phase_schema='gsdeploy.semantic_native_observation.v1',
                    semantic_phase_events=[dict(request_id=row['request_id'],component=component,role=component,
                        host_start_ns=span['start_ns']+1,host_end_ns=(span['start_ns']+span['end_ns'])//2,
                        device_event_complete=True,device_elapsed_ms=1.,stream_id=7,end_current_stream_id=7)])
            prime=copy.deepcopy(bundle['raw']['request_records'][0])
            offset=prime['worker_response']['end_perf_ns']+1_000_000
            for event in prime['execution_events']:
                event['perf_ns']-=offset
            for span in prime['worker_response']['observed_stages']:
                span['start_ns']-=offset;span['end_ns']-=offset
            prime['worker_response']['start_perf_ns']-=offset
            prime['worker_response']['end_perf_ns']-=offset
            bundle['raw']['priming_request_records']=[prime]
            bundles.append(bundle)
        before=copy.deepcopy(bundles)
        result=fit_band_captures(bundles,POOL)
        self.assertEqual(bundles,before)
        diagnostics=result['profiles']['derived_host_call_band_model']['progress_law']['fit_diagnostics']
        self.assertTrue(any(row['accepted'] for row in diagnostics))
        # Every image peer has the same first/continuation exposure in this
        # fixture. Their separate coefficients cannot be identified.
        self.assertTrue(any(row['family']=='image::image_native' and not row['accepted']
                            and row['reason']=='UNIDENTIFIABLE_OR_EXCESSIVE_TRAINING_WORK_ERROR' for row in diagnostics))
        self.assertFalse(result['independent_accuracy_qualified'])
        self.assertFalse(result['training_fit_gates_passed']) # insufficient idle-energy design in this fixture
        self.assertTrue(all(doc['schema']=='gsdeploy.derived_host_call_band_observations.v1' for doc in result['normalized_runs']))
        heldout=copy.deepcopy(bundles)
        heldout[-1]['raw']['split']=heldout[-1]['plan']['split']='development_validation'
        with self.assertRaisesRegex(ValueError,'training-only'):
            fit_band_captures(heldout,POOL)


if __name__=='__main__':
    unittest.main()
