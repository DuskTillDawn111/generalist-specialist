"""Independent CPU review against the actual frozen training-only model.

No fitting, validation labels, whole-DAG extraction, GPU, or production seal.
Digest reconstruction deliberately does not import the producer's digest helper.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from tools.predictor_validation_20261027.cumulative_save_hazard_v14 import CumulativeSaveHazardController as Old
from tools.predictor_validation_20261027.cumulative_save_hazard_v15 import CumulativeSaveHazardController as New
from tools.predictor_validation_20261027.native_collect import receipt

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'artifacts/predictor-save-hazard-v13-20261028-training01/age_additive-model.json'
A = [['text#0', 'text', 'tydiqa-new-native-e0ff7b578f9c742a6697469a', ['prefill']]]
B = [['text#1', 'text', 'tydiqa-new-native-02ea54b33b0492f20de11e0f', ['decode']]]
D = [['ocr#0', 'ocr', 'X51005442341', ['ocr_inference']]]
METRICS = {}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class IndependentCompactReview(unittest.TestCase):
    def controller(self, cls=New, **kw):
        return cls(receipt(MODEL), allow_unvalidated_approximation=True, **kw)

    def same_original_values(self, old, new):
        # Check complete original key sets explicitly, rather than recursively
        # removing arbitrary metadata-looking keys from numerical subobjects.
        top_meta = {'schema', 'controller_identity', 'state'}
        extras = {'empty_audit_summary', 'audit_scope_id', 'complete_empty_audit_external'}
        self.assertEqual(set(new), set(old) | extras)
        for key in set(old) - top_meta:
            self.assertEqual(json.dumps(old[key], sort_keys=True), json.dumps(new[key], sort_keys=True), key)
        state_meta = {'schema', 'controller_identity', 'state_sha256', 'empty_intervals'}
        self.assertEqual(set(new['state']), (set(old['state']) - {'empty_intervals'}) | {'audit_scope_id', 'empty_audit_summary'})
        for key in set(old['state']) - state_meta:
            self.assertEqual(json.dumps(old['state'][key], sort_keys=True), json.dumps(new['state'][key], sort_keys=True), 'state.' + key)
        self.assertEqual(new['state']['state_sha256'], sha({k:v for k,v in new['state'].items() if k != 'state_sha256'}))

    def test_actual_model_stepwise_values_and_delayed_entry(self):
        comparisons = profiles = 0
        terminal_profiles = unknown_profiles = empty_events = 0
        seeds = [None, -7, 0, 42]
        for policy in ('NO_QUERY_PAUSE', 'UNKNOWN'):
            for seed in seeds:
                kw = dict(empty_frontier_policy=policy, budget_policy='MEDIAN_LOG2' if seed is None else 'SEEDED_UNIFORM')
                if seed is not None:
                    kw['seed'] = seed
                old, new = self.controller(Old, **kw), self.controller(**kw)
                for holder in ('generalist_2#0', 'ocr#0', 'text#0', 'text#1'):
                    for entry in (0, 2000, 50_000_000, 88_000_000):
                        profiles += 1
                        cursor = new.begin_audit_scope('reused-candidate-scope-name')
                        left = old.start_restore(holder, 0, A, 'same-key', observation_ns=entry)
                        right = new.start_restore(holder, 0, A, 'same-key', observation_ns=entry)
                        self.same_original_values(left, right); comparisons += 1
                        expected_budget = math.log(2.) if seed is None else -math.log((int(sha(dict(seed=seed, restore_key='same-key', controller_identity=old.identity))[:13], 16) + .5) / 2**52)
                        self.assertEqual(right['hazard_budget'], expected_budget)
                        self.assertEqual(right['consumed_hazard'], 0.)
                        self.assertEqual(right['conditional_entry_age_ns'], entry)
                        # Same-clock calls, visible changes, explicit empty intervals,
                        # age-bin boundaries, and a late terminal probe.
                        schedule = [(0, A), (1, []), (2, []), (3, B), (4, A+B),
                                    (5, D), (10, A), (1000, B), (2000, A),
                                    (5000, D), (10000, A), (20000, B),
                                    (50000, A), (100000, B), (200000, A),
                                    (500000, D), (1000000, A), (100000000, A)]
                        for offset, frontier in schedule:
                            left = old.update(left['state'], entry+offset, frontier)
                            right = new.update(right['state'], entry+offset, frontier)
                            self.same_original_values(left, right); comparisons += 1
                        terminal_profiles += left['state']['terminal_save_candidate_ns'] is not None
                        unknown_profiles += not left['state']['cumulative_hazard_history_known']
                        exported = new.export_audit_since(cursor)
                        intervals = [{k:r[k] for k in ('start_ns','end_ns','policy')} for r in exported['events']]
                        self.assertEqual(intervals, left['state']['empty_intervals'])
                        empty_events += len(intervals)
        self.assertGreater(terminal_profiles, 0)
        self.assertGreater(unknown_profiles, 0)
        METRICS['actual_model_replay'] = dict(profiles=profiles, exact_original_field_comparisons=comparisons,
            terminal_profiles=terminal_profiles, unknown_history_profiles=unknown_profiles,
            exported_intervals_compared_to_v14=empty_events, all_original_values_exact=True,
            budget_policies=['MEDIAN_LOG2', 'SEEDED_UNIFORM'], seeds=seeds,
            delayed_entry_ages_ns=[0,2000,50_000_000,88_000_000])

    def test_four_candidates_complete_independent_ledger_reconstruction(self):
        ctl = self.controller(empty_frontier_policy='NO_QUERY_PAUSE', budget_policy='SEEDED_UNIFORM', seed=42)
        cursors, states, saved_exports = [], {}, []
        for candidate in range(4):
            cursor = ctl.begin_audit_scope('candidate-%d' % candidate); cursors.append(cursor)
            body = {k:v for k,v in cursor.items() if k != 'cursor_sha256'}
            self.assertEqual(cursor['cursor_sha256'], sha(body))
            for key, origin, observation, schedule in [
                ('same-key', 0, 10, [(10,[]),(13,A),(14,[]),(19,[]),(19,[]),(21,A)]),
                ('another-key', 30, 40, [(44,[]),(45,A)])]:
                result = ctl.start_restore('text#0', origin, [], key, observation_ns=observation)
                for clock, frontier in schedule:
                    result = ctl.update(result['state'], clock, frontier)
                states[(cursor['scope_id'], key, origin, observation)] = result['state']
            saved_exports.append(ctl.export_audit_since(cursor))
        last_ledger = sha(dict(schema='gsdeploy.scoped_empty_hazard_audit.v15', controller_identity=ctl.identity))
        offset = 0
        summaries = {}
        all_events = []
        for index, cursor in enumerate(cursors):
            export = ctl.export_audit_since(cursor)
            self.assertEqual(export, saved_exports[index])  # Closed scopes stay immutable.
            self.assertEqual(export['ledger_hash_before'], last_ledger)
            self.assertEqual(export['start_offset'], index*5)
            self.assertEqual(export['end_offset'], (index+1)*5)
            self.assertEqual(export['empty_interval_count'], 5)
            self.assertEqual(export['total_empty_interval_ns'], 15)
            for row in export['events']:
                self.assertEqual(row['scope_id'], cursor['scope_id'])
                self.assertEqual(row['scope_key'], 'candidate-%d' % index)
                self.assertEqual(row['ledger_offset'], offset); offset += 1
                self.assertEqual(row['ledger_previous_hash'], last_ledger)
                last_ledger = sha({k:v for k,v in row.items() if k != 'ledger_entry_hash'})
                self.assertEqual(row['ledger_entry_hash'], last_ledger)
                key = (row['scope_id'], row['restore_key'], row['origin_restore_ns'], row['observation_ns'])
                if key not in summaries:
                    summaries[key] = dict(count=0, total_ns=0, hashchain=sha(dict(
                        controller_identity=ctl.identity, scope_id=key[0], restore_key=key[1],
                        origin_restore_ns=key[2], observation_ns=key[3])), last_interval=None)
                summary = summaries[key]
                interval = {k:row[k] for k in ('start_ns','end_ns','policy')}
                self.assertEqual(row['state_previous_hash'], summary['hashchain'])
                summary['hashchain'] = sha(dict(previous_hash=summary['hashchain'], interval=interval))
                summary['count'] += 1; summary['total_ns'] += row['end_ns']-row['start_ns']
                summary['last_interval'] = interval
                self.assertEqual(row['state_interval_hash'], summary['hashchain'])
                self.assertEqual(row['state_interval_count'], summary['count'])
                all_events.append(row)
            self.assertEqual(export['ledger_hash_after'], last_ledger)
            json.loads(json.dumps(export, allow_nan=False))
        self.assertEqual(len(summaries), 8)
        for key, expected in summaries.items():
            self.assertEqual(states[key]['empty_audit_summary'], expected)
        altered = copy.deepcopy(saved_exports[0]); altered['events'][0]['end_ns'] = -1
        self.assertEqual(ctl.export_audit_since(cursors[0]), saved_exports[0])
        with self.assertRaises(ValueError):
            ctl.update(states[next(iter(states))], 50, [])
        METRICS['four_candidate_audit'] = dict(candidates=4, restore_histories=8, events=len(all_events),
            total_empty_ns=sum(r['end_ns']-r['start_ns'] for r in all_events),
            repeated_restore_keys_isolated=True, full_ledger_and_per_restore_chains_rebuilt=True,
            closed_exports_stable=True, returned_json_deepcopy_isolated=True)

    def test_audit_scope_and_export_frequency_are_numerically_irrelevant(self):
        left = self.controller(empty_frontier_policy='NO_QUERY_PAUSE', budget_policy='SEEDED_UNIFORM', seed=-7)
        right = self.controller(empty_frontier_policy='NO_QUERY_PAUSE', budget_policy='SEEDED_UNIFORM', seed=-7)
        cursor = left.begin_audit_scope('frequent-exports')
        # Include unused prior scopes: ordinal must not affect seeded draws.
        right.begin_audit_scope('unused'); right.begin_audit_scope('different-name')
        a = left.start_restore('text#0', 0, A, 'same-key', observation_ns=88_000_000)
        b = right.start_restore('text#0', 0, A, 'same-key', observation_ns=88_000_000)
        def comparable(r):
            return ({k:v for k,v in r.items() if k not in {'schema','controller_identity','state','empty_audit_summary','audit_scope_id','complete_empty_audit_external'}},
                    {k:v for k,v in r['state'].items() if k not in {'schema','controller_identity','state_sha256','audit_scope_id','empty_audit_summary'}})
        for i in range(101):
            self.assertEqual(comparable(a), comparable(b))
            left.export_audit_since(cursor)
            frontier = [] if i % 2 else B
            a = left.update(a['state'], 88_000_001+i, frontier)
            b = right.update(b['state'], 88_000_001+i, frontier)
        self.assertEqual(comparable(a), comparable(b))
        METRICS['audit_numeric_irrelevance'] = dict(exact_step_comparisons=102, scope_name_ordinal_and_export_frequency_invariant=True)

    def test_fixed_history_structure_for_two_thousand_empty_intervals(self):
        ctl = self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        cursor = ctl.begin_audit_scope('fixed-history')
        result = ctl.start_restore('text#0', 0, [], 'same-key')
        shapes, sizes = set(), []
        for clock in range(1,2001):
            result = ctl.update(result['state'], clock, [])
            state = result['state']; summary = state['empty_audit_summary']
            self.assertNotIn('empty_intervals', state)
            shapes.add((tuple(sorted(state)), tuple(sorted(summary)), tuple(sorted(summary['last_interval']))))
            sizes.append(len(json.dumps(state, sort_keys=True, separators=(',', ':'))))
        self.assertEqual(len(shapes), 1)
        self.assertLess(max(sizes)-min(sizes), 80)  # Only integer digit growth.
        export = ctl.export_audit_since(cursor)
        self.assertEqual(export['empty_interval_count'], 2000)
        self.assertEqual(export['total_empty_interval_ns'], 2000)
        self.assertEqual(result['empty_audit_summary']['count'], 2000)
        METRICS['fixed_history_structure'] = dict(empty_updates=2000, distinct_field_shapes=len(shapes),
            min_state_json_bytes=min(sizes), max_state_json_bytes=max(sizes),
            full_external_events=2000, sealed_state_history_list_absent=True,
            qualification='Fixed history structure; integer digit width and visible frontier content can change byte size. External audit storage grows linearly.')


if __name__ == '__main__':
    unittest.main()
