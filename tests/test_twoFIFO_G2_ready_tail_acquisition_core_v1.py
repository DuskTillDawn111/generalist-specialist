"""Synthetic CPU counterexamples; no native qualification or launch occurs."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.predictor_validation_20261027 import twoFIFO_G2_ready_tail_acquisition_core_v1 as m
from tools.predictor_validation_20261027.stage_response_fit_v1 import profile


def macro(rid, *, positive_ns=0, total_ns=1000, origin=2**60, pool=m.TARGET_POOL):
    pieces = []
    cursor = origin
    for duration, count in [(total_ns-positive_ns, 0), (positive_ns, 1)]:
        if not duration:
            continue
        peers = [] if not count else [dict(owner='text#0', request_id='peer', phase_id='decode-1',
            phase='decode', available_ns=cursor-1)]
        state = {m.phase.COUNT:count, m.phase.SAME:0, 'peer_macros':peers,
                 'cutoff_processing_ns':cursor}
        pieces.append(dict(left_ns=cursor, right_ns=cursor+duration, duration_ns=duration,
            state=state, each_state_uses_only_its_processed_prefix=True))
        cursor += duration
    return dict(owner='generalist_2#0', pool_id=pool, task='text',
        macro_id='native_end_to_worker_ready', semantic_phase='ready_tail', kind='host_ready_response',
        request_id=rid, forward_shape=None, left_ns=origin, right_ns=origin+total_ns,
        response_wall_ns=total_ns, causal_covariate_path=pieces)


def document(block, shard=1, *, positive_ns=1, matching=True):
    count = 26 if matching else 11
    requests = []
    for index in range(count):
        rid = f'b{block}-s{shard}-r{index}'
        value = macro(rid, positive_ns=positive_ns if index % 2 else 0)
        requests.append(dict(request_id=rid, macros=[value], worker_start_to_ready_ns=1000))
    return dict(capture_id='block-'+str(block), attempt_id=f'b{block}-s{shard}',
        requests=requests, measured_requests=count, priming_rows_excluded_as_labels=18,
        validation_labels_read=False, fitted_model_qualified=False,
        whole_attempt_history_used_for_visible_prefix=True, case_history_reset=False)


def documents(*, positive_blocks=(1, 2, 3), matching=True):
    return [document(block, shard, positive_ns=int(block in positive_blocks), matching=matching)
        for block in (1, 2, 3) for shard in range(1, 6 if matching else 2)]


def statistics(docs, **kwargs):
    return m.full_and_leaveblock_statistics(docs,
        expected_attempt_ids=[d['attempt_id'] for d in docs], **kwargs)


class CalendarAndOutcomes(unittest.TestCase):
    def test_original_calendar_exact33_plus54_fixed_offsets_and_no_admission_cap(self):
        contract = m.calendar_contract()
        self.assertEqual((contract['measured_offered'], contract['priming_history_offered']), (33, 54))
        self.assertFalse(contract['executable'])
        for block in contract['blocks']:
            self.assertIsNone(block['runtime_policy']['admission'])
            self.assertEqual((block['measurement_ns'], block['drain_ns']), (10_000_000_000, 60_000_000_000))

    def test_changed_calendar_cannot_be_rebound_by_rehashing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'report.json').write_bytes((m.PROPOSAL/'report.json').read_bytes())
            content = (m.PROPOSAL/'cases-by-block.json').read_text().replace('140710722', '140710723')
            (path/'cases-by-block.json').write_text(content)
            with self.assertRaisesRegex(ValueError, 'unchanged predeclared'):
                m.calendar_contract(path)

    def test_no_capture_still_retains_all87_explicit_unknown_outcomes(self):
        ledger = m.outcome_ledger(m.calendar_contract()['blocks'])
        self.assertEqual((ledger['offered'], ledger['measured_offered'], ledger['priming_offered']), (87, 33, 54))
        self.assertEqual(ledger['incomplete_or_unobserved'], 87)
        self.assertTrue(all(r['exposure_support'].startswith('UNKNOWN') for r in ledger['rows']))
        self.assertFalse(ledger['native_evidence_authenticated'])

    def raw(self, block):
        return dict(capture_id=block['capture_id'], attempt_id=block['attempt_id'],
            request_records=[dict(request_id=r['request_id'], status='not_started')
                for c in block['cases'] for r in c['arrivals']],
            priming_request_records=[dict(request_id=r['request_id'], status='not_started') for r in block['priming']])

    def test_failed_and_unstarted_rows_preserved_verbatim_no_selected_labels(self):
        blocks = m.calendar_contract()['blocks']; raw = self.raw(blocks[0])
        raw['request_records'][0].update(status='error', worker_response={'success':False, 'error':'failure'})
        raw['request_records'][1].update(status='completed', worker_response={'success':True})
        ledger = m.outcome_ledger(blocks, {blocks[0]['attempt_id']:raw})
        self.assertEqual(ledger['rows'][0]['observed_row'], raw['request_records'][0])
        self.assertEqual(ledger['rows'][1]['status'], 'completed')
        self.assertTrue(all(r['prediction_label_eligible'] is False for r in ledger['rows']))
        self.assertEqual(ledger['offered'], 87)

    def test_omitted_failed_or_duplicate_offered_row_is_rejected(self):
        blocks = m.calendar_contract()['blocks']
        for kind in ('missing', 'duplicate', 'foreign'):
            raw = self.raw(blocks[0])
            if kind == 'missing':raw['request_records'].pop()
            elif kind == 'duplicate':raw['request_records'][1] = raw['request_records'][0]
            else:raw['attempt_id'] = 'other'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.outcome_ledger(blocks, {blocks[0]['attempt_id']:raw})


class ExposureAndRank(unittest.TestCase):
    def test_one_ns_at_large_absolute_clock_stays_positive_with_exact_integer_mass(self):
        doc = document(1)
        stats = m.exposure_statistics([doc])
        self.assertEqual(stats['positive_peer_exposure_ns'], 13)
        self.assertEqual(stats['total_ns'], 26000)
        segment = stats['records'][1]['pieces'][-1]
        self.assertEqual(segment['right_ns']-segment['left_ns'], 1)
        self.assertEqual(segment['duration_ns'], 1)
        self.assertTrue(stats['same_group_n0_and_positive_observed'])
        self.assertFalse(stats['semantic_positivity_is_numerical_qualification'])

    def test_public_overlap_intent_without_processed_peer_is_zero(self):
        doc = document(1, positive_ns=0)
        doc['public_arrival_intent'] = 'queued peer before G2'
        stats = m.exposure_statistics([doc])
        self.assertFalse(stats['any_G2_tail_positive_peer_exposure'])
        self.assertEqual(stats['positive_peer_exposure_ns'], 0)

    def test_future_peer_or_duration_inflation_rejected(self):
        for kind in ('future', 'duration', 'float_clock', 'missing_path', 'history_reset'):
            doc = document(1)
            item = doc['requests'][1]['macros'][0]
            piece = item['causal_covariate_path'][-1]
            if kind == 'future':piece['state']['peer_macros'][0]['available_ns'] = piece['left_ns']+1
            elif kind == 'duration':piece['duration_ns'] += 1
            elif kind == 'float_clock':piece['left_ns'] = float(piece['left_ns'])
            elif kind == 'missing_path':item['causal_covariate_path'] = []
            else:doc['case_history_reset'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.exposure_statistics([doc])

    def test_positive_and_zero_in_different_nuisance_groups_are_not_same_group_support(self):
        doc = document(1)
        for request in doc['requests']:
            value = request['macros'][0]
            if value['causal_covariate_path'][-1]['state'][m.phase.COUNT]:
                # Give these requests entirely positive paths and a different input.
                request['macros'] = [macro(request['request_id'], positive_ns=1000, pool='other-input')]
        stats = m.exposure_statistics([doc])
        self.assertTrue(stats['any_G2_tail_positive_peer_exposure'])
        self.assertFalse(stats['same_group_n0_and_positive_observed'])

    def test_full_and_leaveout_keep_all15_390_plus270_and_whole_blocks(self):
        result = statistics(documents(positive_blocks=(1,)))
        self.assertEqual((result['attempt_count'], result['measured_requests'], result['priming_requests']), (15, 390, 270))
        self.assertEqual([p['exposure']['measured_requests'] for p in result['populations']], [390, 260, 260, 260])
        fold = next(p for p in result['populations'] if p['heldout_block_id'] == 'block-1')
        self.assertFalse(fold['exposure']['any_G2_tail_positive_peer_exposure'])
        self.assertTrue(result['populations'][0]['exposure']['any_G2_tail_positive_peer_exposure'])

    def test_three_targeted_blocks_33_plus54_are_separate_domain(self):
        result = statistics(documents(matching=False))
        self.assertEqual((result['measured_requests'], result['priming_requests']), (33, 54))
        with self.assertRaisesRegex(ValueError, 'original matching15'):
            m.conditional_design_status(matching_complete_and_independently_rederived=True, statistics=result)

    def test_partial_or_survivor_population_cannot_become_complete_statistics(self):
        original = documents()
        for kind in ('one_shard', '14_attempts', 'dropped_request', 'missing_prime', 'duplicate'):
            changed = copy.deepcopy(original)
            if kind == 'one_shard':changed = changed[:1]
            elif kind == '14_attempts':changed.pop()
            elif kind == 'dropped_request':
                changed[0]['requests'].pop(); changed[0]['measured_requests'] -= 1
            elif kind == 'missing_prime':changed[0]['priming_rows_excluded_as_labels'] = 17
            else:changed[-1] = changed[0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.full_and_leaveblock_statistics(changed, expected_attempt_ids=[d['attempt_id'] for d in original])

    def rows(self):
        rows = []
        for block in (1, 2, 3):
            for index in range(2):
                vector = np.zeros(12);vector[7] = index
                rows.append(dict(group=('same-input',), capture='block-'+str(block),
                    durations=np.array([.001+block*.0001]), matrix=vector.reshape(1, 12)))
        return rows

    def test_derivative_matches_original_profile_at_finite_point_and_nuisance_centering(self):
        rows = self.rows();beta = np.asarray(m.FINITE_PROBE_BETA)
        _, _, _, derivative, _ = profile(beta, rows)
        centered = derivative-derivative.mean(axis=0)
        result = m.derivative_diagnostic(rows, beta)
        self.assertAlmostEqual(result['G2_tail_nuisance_adjusted_derivative_l2'], np.linalg.norm(centered[:, 7]))
        self.assertEqual(result['nuisance_adjusted_derivative_rank'], 1)
        self.assertEqual(len(result['features']), 12)
        self.assertFalse(result['joint_full_column_rank'])
        self.assertFalse(result['optimizer_run'])

    def test_constant_peer_fraction_is_nuisance_confounded_even_with_positive_exposure(self):
        rows = self.rows()
        for row in rows:row['matrix'][0, 7] = 1
        result = m.derivative_diagnostic(rows, np.zeros(12))
        self.assertEqual(result['G2_tail_nuisance_adjusted_derivative_l2'], 0)
        self.assertEqual(result['nuisance_adjusted_derivative_rank'], 0)

    def test_finite_derivative_preserves_tiny_mixed_path_and_matches_original_weights(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            zero = np.zeros(12); peer = np.zeros(12);peer[7] = 1
            row.update(durations=np.array([.001+index*.0001, .000001]),
                       matrix=np.stack((zero, peer)))
        beta = np.asarray(m.FINITE_PROBE_BETA)
        _, _, _, derivative, _ = profile(beta, rows)
        expected = np.linalg.norm((derivative-derivative.mean(axis=0))[:, 7])
        result = m.derivative_diagnostic(rows, beta)
        self.assertGreater(result['G2_tail_nuisance_adjusted_derivative_l2'], 0)
        self.assertAlmostEqual(result['G2_tail_nuisance_adjusted_derivative_l2'], expected, places=15)

    def test_full_rank_needs_all_original12_features_no_column_deletion(self):
        rows = [dict(group=('same',), capture='block-1', durations=np.ones(1), matrix=np.zeros((1, 12)))]
        rows += [dict(group=('same',), capture='block-1', durations=np.ones(1), matrix=v.reshape(1, 12))
                 for v in np.eye(12)]
        result = m.derivative_diagnostic(rows, np.zeros(12))
        self.assertTrue(result['joint_full_column_rank'])
        self.assertEqual(result['nuisance_adjusted_derivative_rank'], 12)
        self.assertFalse(result['fitted_root_numerical_qualification'])
        with self.assertRaises(ValueError):
            m.derivative_diagnostic([dict(rows[0], matrix=np.zeros((1, 11)))], np.zeros(12))


class ConditionalHolds(unittest.TestCase):
    def test_partial_matching_never_uses_a_zero_exposure_snapshot_to_activate(self):
        result = m.conditional_design_status(matching_complete_and_independently_rederived=False, statistics=None)
        self.assertIn('MATCHING15_NOT_COMPLETE', result['status'])
        self.assertFalse(result['executable'])

    def test_positive_exposure_alone_never_establishes_no_need_when_rank_is_unassessed(self):
        stats = statistics(documents())
        result = m.conditional_design_status(matching_complete_and_independently_rederived=True,
            statistics=stats, resource_chain_complete_and_rederived=True)
        self.assertIn('NUMERICAL_SUPPORT_UNASSESSED', result['status'])
        self.assertIsNone(result['targeted_acquisition_candidate'])

    def test_generic_rank_gap_with_positive_exposure_requires_attribution(self):
        stats = statistics(documents())
        for population in stats['populations']:
            population['zero_point'] = {'joint_full_column_rank':False}
            population['finite_predeclared_probe'] = {'joint_full_column_rank':False}
        result = m.conditional_design_status(matching_complete_and_independently_rederived=True,
            statistics=stats, resource_chain_complete_and_rederived=True)
        self.assertIn('REQUIRES_ATTRIBUTION', result['status'])
        self.assertIsNone(result['targeted_acquisition_candidate'])

    def test_no_same_group_n0_positive_is_reported_even_with_positive_ns(self):
        stats = statistics(documents())
        stats['populations'][1]['exposure']['same_group_n0_and_positive_observed'] = False
        result = m.conditional_design_status(matching_complete_and_independently_rederived=True,
            statistics=stats, resource_chain_complete_and_rederived=True)
        self.assertIn('SAME_GROUP_SUPPORT_GAP', result['status'])
        self.assertTrue(result['targeted_acquisition_candidate'])
        self.assertFalse(result['native_ready'])

    def test_full_and_every_fold_rank_pass_may_describe_current_diagnostic_no_need(self):
        stats = statistics(documents())
        for population in stats['populations']:
            population['zero_point'] = {'joint_full_column_rank':True}
            population['finite_predeclared_probe'] = {'joint_full_column_rank':True}
        result = m.conditional_design_status(matching_complete_and_independently_rederived=True,
            statistics=stats, resource_chain_complete_and_rederived=True)
        self.assertIn('NOT_NEEDED_AT_REPORTED', result['status'])
        self.assertFalse(result['targeted_acquisition_candidate'])
        self.assertFalse(result['activation_authenticated'])

    def test_missing_fold_exposure_requires_resource_chain_then_new_reviewed_executor(self):
        stats = statistics(documents(positive_blocks=(1,)))
        pending = m.conditional_design_status(matching_complete_and_independently_rederived=True, statistics=stats)
        self.assertIn('KERNEL_RESOURCE_CHAIN_NOT_COMPLETE', pending['status'])
        conditional = m.conditional_design_status(matching_complete_and_independently_rederived=True,
            statistics=stats, resource_chain_complete_and_rederived=True)
        self.assertIn('REQUIRES_NEW_NATIVE_PRODUCER_AND_INDEPENDENT_REVIEW', conditional['status'])
        self.assertFalse(conditional['native_ready'])
        self.assertFalse(conditional['activation_authenticated'])

    def test_dependency_probe_is_metadata_only_and_contains_both_canonical_finals(self):
        result = m.dependency_probe()
        self.assertTrue(result['matching_final_path'].endswith('cpu02-prepared/run-v1/report.json'))
        self.assertTrue(result['resource_final_path'].endswith('cpu01-prepared/run-v1/report.json'))
        self.assertFalse(result['partial_matching_labels_read'])
        self.assertFalse(result['executable'])


if __name__ == '__main__':
    unittest.main()
