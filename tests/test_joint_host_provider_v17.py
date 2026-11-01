"""Counterexamples for source, clock, cohort, and joint-preparation contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.joint_host_provider_v17 import (
    BANK_SCHEMA, JointHostScenarioProvider, digest, receipt)


class JointHostProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root/'training-source.json'
        source.write_text('{"split":"training","CPU_fixture":true}')
        self.ref = receipt(source)
        workload = [dict(alias=a, task='text', pool_id=p, arrival_ns=0)
                    for a, p in [('a', 'x'), ('b', 'y')]]
        # A has larger post-handler preparation, but earlier semantic start:
        # H_a=10,P_a=30; H_b=35,P_b=10. P alone reverses the order.
        rows = [dict(alias=a, request_id='r'+a, scheduled_offset_ns=0,
            submit_offset_ns=1, handler_start_offset_ns=h,
            pre_sync_start_offset_ns=h+1, pre_sync_end_offset_ns=h+3,
            first_semantic_start_offset_ns=s, synchronization_wait_removed=False,
            first_semantic_is_first_CUDA_release=False)
            for a, h, s in [('a', 10, 40), ('b', 35, 45)]]
        self.scenario = dict(domain='natural', source_raw=self.ref,
            source_review=self.ref, execution_context_sha256='c'*64,
            job_id='job', job_name='pair', public_workload=workload,
            workload_sha256=digest(workload), prefixes=rows,
            joint_event_field='first_semantic_start_offset_ns',
            joint_event_order=['a', 'b'], joint_event_gap_from_first_ns={'a': 0, 'b': 5},
            prefix_clock_origin=dict(domain='host_perf_counter', ns=1000),
            public_arrival_and_light_offsets_share_origin=True,
            empirical_scenario_probability=None, counterfactual_host_supply_identified=False)
        self.bank = dict(schema=BANK_SCHEMA, sources=[self.ref], scenarios=[self.scenario],
            training_only=True, whole_job_correlation_preserved=True,
            per_owner_mean_composition_used=False, cross_domain_pairing_identified=False,
            cross_domain_joint_probabilities_assigned=False,
            synchronization_wait_is_exogenous_supply=False, complete_host_supply_provider=False)

    def provider(self):
        path = self.root/'observations.json'
        path.write_text(json.dumps(self.bank))
        return JointHostScenarioProvider(receipt(path))

    def selection(self, provider, **changes):
        query = dict(domain='natural', workload=self.scenario['public_workload'],
                     execution_context_sha256='c'*64)
        query.update(changes)
        return provider.select(**query)

    def test_joint_vector_separates_handler_from_preparation_and_keeps_wait(self):
        provider = self.provider()
        selected = self.selection(provider)
        sid = selected['scenarios'][0]['scenario_id']
        schedule = provider.schedule(sid, checkpoint='first_semantic_start_offset_ns')
        vector = provider.preparation_vector(sid)
        self.assertEqual([e['alias'] for e in schedule['events']], ['a', 'b'])
        self.assertEqual([r['handler_to_first_semantic_ns'] for r in vector['vectors']], [30, 10])
        self.assertEqual([r['scheduled_to_handler_ns'] for r in vector['vectors']], [10, 35])
        self.assertEqual(vector['vectors'][0]['pre_sync_end_from_handler_ns'], 3)
        self.assertFalse(vector['actual_first_ready_order_preserved_by_P_alone'])
        self.assertIsNone(selected['scenario_probabilities'])
        self.assertFalse(schedule['causal_release'])

    def test_changed_input_route_arrival_context_and_subset_are_unknown(self):
        provider = self.provider()
        changed = []
        for key, value in [('pool_id', 'unseen'), ('alias', 'new'), ('arrival_ns', 1)]:
            workload = copy.deepcopy(self.scenario['public_workload'])
            workload[0][key] = value
            changed.append(dict(workload=workload))
        changed.extend([dict(workload=self.scenario['public_workload'][:1]),
                        dict(execution_context_sha256='d'*64),
                        dict(source_capture_sha256='e'*64)])
        for query in changed:
            with self.subTest(query=query), self.assertRaises(UnknownEvidence):
                self.selection(provider, **query)

    def test_counterfactual_release_requires_unavailable_evidence(self):
        provider = self.provider()
        sid = self.selection(provider)['scenarios'][0]['scenario_id']
        with self.assertRaises(UnknownEvidence):
            self.selection(provider, purpose='counterfactual_release')
        with self.assertRaises(UnknownEvidence):
            provider.schedule(sid, checkpoint='first_semantic_start_offset_ns',
                              purpose='counterfactual_release')

    def test_missing_member_stays_in_denominator_and_complete_schedule_rejects(self):
        self.scenario['prefixes'][1]['first_semantic_start_offset_ns'] = None
        self.scenario['joint_event_order'] = ['a']
        self.scenario['joint_event_gap_from_first_ns'] = {'a': 0}
        provider = self.provider()
        sid = self.selection(provider)['scenarios'][0]['scenario_id']
        with self.assertRaises(UnknownEvidence):
            provider.schedule(sid, checkpoint='first_semantic_start_offset_ns')
        result = provider.schedule(sid, checkpoint='first_semantic_start_offset_ns',
                                   require_complete=False)
        self.assertEqual(result['cohort_size'], 2)
        self.assertEqual(result['missing_request_ids'], ['rb'])
        self.assertIsNone(result['events'][1]['offset_ns'])
        with self.assertRaises(UnknownEvidence): provider.preparation_vector(sid)
        self.assertEqual(len(provider.preparation_vector(sid, require_complete=False)['vectors']), 2)

    def test_light_clock_not_transferred_or_paired_to_natural(self):
        light = copy.deepcopy(self.scenario)
        light.update(domain='unified-light-v8_1', job_id='light-job',
                     public_arrival_and_light_offsets_share_origin=False,
                     prefix_clock_origin=dict(domain='native_Nsight_session', ns=9000),
                     joint_event_field='first_kernel_launch_API_entry_ns')
        light['prefixes'] = [dict(alias=a, request_id='l'+a,
            first_kernel_launch_API_entry_ns=t,
            API_wall_contains_possible_GPU_wait=True,
            observed_prefix_is_counterfactual_release=False) for a,t in [('a', 100), ('b', 105)]]
        self.bank['scenarios'].append(light)
        provider = self.provider()
        natural = self.selection(provider)
        selected = self.selection(provider, domain='unified-light-v8_1')
        sid = selected['scenarios'][0]['scenario_id']
        self.assertEqual(len(natural['scenarios']), 1)
        self.assertIsNone(selected['cross_domain_pairing'])
        with self.assertRaises(UnknownEvidence):
            provider.schedule(sid, checkpoint='first_kernel_launch_API_entry_ns', clock_origin='public_arrival')
        with self.assertRaises(UnknownEvidence): provider.preparation_vector(sid)

    def test_receipt_and_post_construction_source_tampering_rejected(self):
        provider = self.provider()
        Path(self.ref['path']).write_text('changed')
        with self.assertRaises(ValueError): provider.verify_sources()
        with self.assertRaises(ValueError): self.provider()

    def test_duplicate_and_invented_probability_cannot_change_scenario_mass(self):
        self.bank['scenarios'].append(copy.deepcopy(self.scenario))
        with self.assertRaises(ValueError): self.provider()
        self.bank['scenarios'].pop()
        self.scenario['empirical_scenario_probability'] = .5
        with self.assertRaises(ValueError): self.provider()

    def test_detached_results_cannot_mutate_frozen_vectors(self):
        provider = self.provider()
        result = self.selection(provider)
        result['scenarios'][0]['prefixes'][0]['first_semantic_start_offset_ns'] = 0
        self.assertEqual(self.selection(provider)['scenarios'][0]['prefixes'][0]
                         ['first_semantic_start_offset_ns'], 40)

    def test_repeated_owner_full_cohort_not_silently_split_into_jobs(self):
        provider = self.provider()
        workload = copy.deepcopy(self.scenario['public_workload'])
        workload.append(dict(workload[0], arrival_ns=100))
        with self.assertRaisesRegex(UnknownEvidence, 'REPEATED_OWNER'):
            self.selection(provider, workload=workload)


if __name__ == '__main__': unittest.main()
