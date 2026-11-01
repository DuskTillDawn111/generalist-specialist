"""Resource denominator, bracket and capture-leaveout counterexamples."""
import copy
import hashlib
import json
import unittest

from tools.predictor_validation_20261027 import natural_macro_resources_candidate_v1 as r


def fixtures():
    roster=('a','idle')
    def sample(at,watts=100,memory=400,duration=2):
        return dict(time_ns=at,query_duration_ns=duration,watts=watts,owned_used_mib=memory,
            owned_aliases=list(roster),phase='measurement',foreign_pids=[],unresolved_nvml_pids=[],unresolved_ownership_reasons=[])
    raw=dict(observation_start_ns=100,observation_end_ns=200,
        lifecycle_boundaries=[dict(boundary='startup_begin',time_ns=10),dict(boundary='measurement_and_drain_end',time_ns=200),
            dict(boundary='monitoring_end',time_ns=220),dict(boundary='shutdown_begin',time_ns=230)],
        shutdown_memory_sampled=False,
        board_samples=[sample(t) for t in (1,20,50,90,101,110,120,130,140,150,160,170,180,190,201,210)],
        priming_request_records=[{} for _ in range(18)],
        request_records=[dict(request_id='one',model='a',task='text',pool_id='p',observed_worker_start_time_ns=120,
            previous_handler_next_request_ready=dict(perf_ns=180))])
    macros=dict(requests=[dict(request_id='one',macros=[dict(left_ns=120,right_ns=180,semantic_phase='prefill')])])
    return raw,macros,roster


def table(capture='one',watts=100):
    raw,macros,roster=fixtures()
    for sample in raw['board_samples']:sample['watts']=watts
    result=r.observed_table(raw,macros,roster)
    result.update(capture_sha256=capture,execution_context_sha256='new-natural',priming_content_identity='18-exact-inputs')
    return result


def bind(timeline):
    context=dict(cpu_fixture='explicit-only',execution_sources=[])
    execution=r.context_identity(context)
    binding=dict(basis='caller_declared_explicit_reference_execution_binding',execution_context_sha256=execution,
        resource_timeline_sha256=hashlib.sha256(json.dumps(timeline,sort_keys=True).encode()).hexdigest())
    return context,binding


class NewNaturalResourceTests(unittest.TestCase):
    def test_global_timeline_adapter_integrates_measurement_once_and_memory_through_drain(self):
        from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode
        from fractions import Fraction as F
        class Provider:
            def power(self,active,*,resident_owners):return dict(known=True,watts=100)
            def memory(self,active,*,resident_owners):return dict(known=True,peak_memory_mib=500 if active else 400)
            def startup(self):return dict(joules=3600,premeasurement_owned_sampled_peak_mib=450)
        slices=[dict(duration_ns=F(5),measurement_overlap_ns=F(5),resident_owners=['a','idle'],actual_START_to_READY_handlers=[]),
            dict(duration_ns=F(5),measurement_overlap_ns=F(3),resident_owners=['a','idle'],actual_START_to_READY_handlers=[dict(owner='a')])]
        timeline=encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',slices=slices,
            total_wall_ns=F(10),total_measurement_overlap_ns=F(8)))
        context,binding=bind(timeline);provider=Provider();provider.execution_context_sha256=r.context_identity(context)
        result=r.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
        self.assertEqual(result['board_energy_joules'],100*8/1e9)
        self.assertEqual(result['peak_memory_mib'],500)
        self.assertEqual(result['startup_energy_joules'],3600)
        self.assertFalse(result['startup_energy_already_amortized'])

    def test_timeline_adapter_unknown_drain_power_does_not_poison_measurement_E(self):
        from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode
        from fractions import Fraction as F
        class Provider:
            def power(self,active,*,resident_owners):return dict(known=not active,watts=None if active else 100)
            def memory(self,active,*,resident_owners):return dict(known=not active,peak_memory_mib=None if active else 400)
            def startup(self):return dict(joules=3600,premeasurement_owned_sampled_peak_mib=450)
        slices=[dict(duration_ns=F(8),measurement_overlap_ns=F(8),resident_owners=['a'],actual_START_to_READY_handlers=[]),
            dict(duration_ns=F(2),measurement_overlap_ns=F(0),resident_owners=['a'],actual_START_to_READY_handlers=[dict(owner='a')])]
        timeline=encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',slices=slices,
            total_wall_ns=F(10),total_measurement_overlap_ns=F(8)))
        context,binding=bind(timeline);provider=Provider();provider.execution_context_sha256=r.context_identity(context)
        result=r.predict_timeline(timeline,provider,target_context=context,execution_binding=binding)
        self.assertEqual(result['board_energy_joules'],100*8/1e9)
        self.assertIsNone(result['peak_memory_mib'])

    def test_timeline_adapter_rejects_changed_target_or_missing_timeline_binding(self):
        class Provider:execution_context_sha256='different'
        timeline=dict(schema='gsdeploy.macro_global_resource_timeline.v1')
        context,binding=bind(timeline)
        with self.assertRaises(ValueError):
            r.predict_timeline(timeline,Provider(),target_context=context,execution_binding=binding)
        Provider.execution_context_sha256=r.context_identity(context)
        with self.assertRaises(ValueError):
            r.predict_timeline(timeline,Provider(),target_context=context,execution_binding={})

    def test_shutdown_poll_cannot_enter_owned_memory_training(self):
        raw,macro,roster=fixtures();raw['board_samples'][-1]['time_ns']=240
        with self.assertRaises(ValueError):r.observed_table(raw,macro,roster)

    def test_global_idle_and_activity_partition_and_full_sample_denominator(self):
        x=table()
        self.assertEqual([s['duration_ns'] for s in x['slices']],[20,60,20])
        self.assertEqual(sum(s['duration_ns'] for s in x['slices']),100)
        self.assertEqual(x['raw_monitor_sample_count'],len(x['polls']))
        self.assertEqual(x['observation_joules'],100*100/1e9)
        self.assertIsNone(x['scoring_measurement_energy_joules'])
        self.assertEqual(x['priming_requests'],18)

    def test_poll_crossing_phase_boundary_retained_without_state_label(self):
        raw,macro,roster=fixtures();raw['board_samples'][6]['query_duration_ns']=10
        x=r.observed_table(raw,macro,roster)
        self.assertIsNone(next(p for p in x['polls'] if p['midpoint_ns']==120)['state_key'])
        self.assertGreater(x['observation_power_training_uncovered_duration_ns'],0)
        self.assertEqual(len(x['polls']),len(raw['board_samples']))

    def test_poll_midpoint_rounding_reconstructs_query_bracket(self):
        self.assertEqual(r.poll_bracket(dict(time_ns=101,query_duration_ns=3)),(100,103))
        with self.assertRaises(ValueError):r.poll_bracket(dict(time_ns=101))

    def test_inactive_resident_memory_retained_not_subtracted_or_summed(self):
        a=table();b=table('two');model=r.fit_tables([a,b])
        idle=next(s for s in model['states'].values() if not s['members'])
        self.assertEqual(idle['mean']['peak_memory_mib'],400)
        self.assertFalse(model['memory_safe_factor_applied'])
        self.assertFalse(model['capacity_bound'])

    def test_missing_resident_during_observation_rejects_memory_table(self):
        raw,macro,roster=fixtures();raw['board_samples'][8]['owned_aliases']=['a']
        with self.assertRaises(ValueError):r.observed_table(raw,macro,roster)

    def test_adjacent_power_pairs_never_sum_parallel_owner_durations(self):
        raw,macro,roster=fixtures()
        raw['request_records'].append(dict(raw['request_records'][0],request_id='two',model='idle'))
        macro['requests'].append(dict(macro['requests'][0],request_id='two'))
        x=r.observed_table(raw,macro,roster)
        self.assertEqual(sum(s['duration_ns'] for s in x['slices']),100)
        self.assertEqual(x['observation_joules'],100*100/1e9)
        self.assertEqual(len(x['slices'][1]['members']),2)
        self.assertLessEqual(x['state_power_training_duration_ns'],100)

    def test_one_poll_has_memory_but_not_power_pair_support(self):
        a=table();b=table('two')
        for x in (a,b):
            for row in x['slices']:row.update(power_training_joules=0.,power_training_duration_ns=0)
        model=r.fit_tables([a,b])
        self.assertTrue(all(s['memory_known'] for s in model['states'].values()))
        self.assertTrue(all(not s['power_known'] for s in model['states'].values()))
        d=r.replay_diagnostic(table('held'),model)
        self.assertIsNone(d['observation_energy'])
        self.assertEqual(d['power_unknown_duration_ns'],100)

    def test_missing_state_retains_whole_leaveout_denominator(self):
        a=table();b=table('two');held=table('held')
        model=r.fit_tables([a,b]);state=held['slices'][1]['state_key'];del model['states'][state]
        d=r.replay_diagnostic(held,model)
        self.assertEqual(d['power_unknown_duration_ns'],60)
        self.assertEqual(d['power_known_duration_ns'],40)
        self.assertEqual(d['observation_duration_ns'],100)
        self.assertIsNone(d['observation_energy']);self.assertIsNone(d['raw_monitor_owned_sampled_peak'])

    def test_capture_mean_and_startup_amortization_remain_unapplied(self):
        a=table(watts=100);b=table('two',watts=300)
        model=r.fit_tables([a,b]);self.assertEqual(model['startup']['joules'],200*90/1e9)
        self.assertEqual(model['startup']['startup_amortization_seconds'],3600)
        self.assertTrue(all(x['mean']['watts']==200 for x in model['states'].values()))

    def test_repeat_capture_wrong_priming_and_changed_execution_do_not_pool(self):
        for mutation in ('repeat','priming','execution'):
            a=table();b=table('two')
            if mutation=='repeat':b['capture_sha256']='one'
            elif mutation=='priming':b['priming_requests']=12
            else:b['execution_context_sha256']='old-worker'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.fit_tables([a,b])

    def test_peak_uses_full_raw_monitor_including_startup_and_final_drain_sample(self):
        raw,macro,roster=fixtures();raw['board_samples'][-1]['owned_used_mib']=999
        x=r.observed_table(raw,macro,roster)
        self.assertEqual(x['raw_monitor_owned_sampled_peak_mib'],999)
        self.assertEqual(x['premeasurement_owned_sampled_peak_mib'],400)
        self.assertTrue(all(s['sampled_owned_peak_mib']==400 for s in x['slices']))

    def test_inputs_not_mutated_and_unseen_public_input_does_not_share_state(self):
        raw,macro,roster=fixtures();before=copy.deepcopy((raw,macro,roster))
        r.observed_table(raw,macro,roster);self.assertEqual((raw,macro,roster),before)
        first=dict(owner='a',task='text',pool_id='p',response_phase='prefill')
        second=dict(first,pool_id='q')
        self.assertNotEqual(r.key([first],roster)[0],r.key([second],roster)[0])


if __name__=='__main__':unittest.main()
