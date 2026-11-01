"""Independent CPU checks of the real cpu02 matching preparation.

No raw GPU outcome, fit, validation label, or mocked upstream authentication.
The calendar proof deliberately makes no physical overlap/rank claim.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_matching_supplement_v1 as m


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts/predictor-twoFIFO-matching-macro-supplement-v1-20261029-cpu02-prepared/manifest.json'
OWNERS = ('text#0', 'text#1', 'generalist_2#0', 'ocr#0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(ref):
    assert sha(ref['path']) == ref['sha256'], ref
    return json.loads(Path(ref['path']).read_text())


def pool(ref):
    assert sha(ref['path']) == ref['sha256'], ref
    rows = [json.loads(line) for line in Path(ref['path']).read_text().splitlines()]
    assert len(rows) == len({row['pool_id'] for row in rows})
    return {row['pool_id']: row for row in rows}


def key(row):
    return row['owner'], row['task'], row['pool_id']


def directed_modes(cases):
    result = set()
    for case in cases:
        active = frozenset(row['owner'] for row in case['arrivals'])
        for row in case['arrivals']:
            result.add((key(row), active))
    return result


class MatchingSupplementIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run the actual full authentication path, then independently recompute
        # properties directly from the source contracts and saved arrivals.
        cls.manifest = m.audit(MANIFEST)
        cls.defs = read(cls.manifest['artifacts']['definitions'])['blocks']
        cls.plans = [[read(ref) for ref in block['attempts']]
                     for block in cls.manifest['blocks']]
        cls.old = [read(ref) for ref in cls.manifest['old_training_plans']]
        prep = read(cls.manifest['public_preparation'])
        cls.contract = read(prep['artifacts']['public_contract'])
        cls.required = {(candidate['routes'][row['request_id']], row['task'], row['pool_id'])
                        for candidate in cls.contract['candidates']
                        for row in cls.contract['trace']}

    def test_real_audit_and_historical_source_snapshot_receipts(self):
        self.assertEqual(len(self.required), 16)
        for ref in self.manifest['source_closure']:
            self.assertEqual(sha(ref['path']), ref['sha256'])
        snapshot = json.loads((ROOT / 'artifacts/predictor-twoFIFO-matching-macro-supplement-v1-20261029-cpu01-author/source-snapshot-receipt.json').read_text())
        old_author = read(snapshot['historical_author_report'])
        self.assertEqual(snapshot['original_source']['sha256'],
                         snapshot['preserved_source']['sha256'])
        self.assertEqual(sha(snapshot['preserved_source']['path']), old_author['source']['sha256'])
        self.assertNotEqual(sha(m.__file__), old_author['source']['sha256'])

    def test_union22_preserves_old16_and_adds_exact_six_public_payloads(self):
        old = pool(self.manifest['original_pool'])
        public = pool(self.manifest['public_catalog'])
        union = pool(self.manifest['artifacts']['pool'])
        wanted = {row['pool_id'] for row in self.contract['trace']}
        self.assertEqual(len(old), 16)
        self.assertEqual(len(union), 22)
        self.assertEqual(set(union), set(old) | wanted)
        self.assertEqual(len(set(union) - set(old)), 6)
        self.assertEqual(sorted(set(union) - set(old)), self.manifest['summary']['added_input_ids'])
        for name, row in old.items():
            self.assertEqual(union[name], row)
        for name in wanted:
            self.assertEqual(union[name], public[name])
        for row in union.values():
            if row['task_type'] != 'text':
                sample = row['sample']
                self.assertEqual(sha(sample['input']), sample['input_sha256'])

    def test_all16_keys_have_five_directed_modes_in_every_saved_block(self):
        expected = set()
        for target in self.required:
            owner = target[0]
            expected.add((target, frozenset([owner])))
            expected.add((target, frozenset(OWNERS)))
            expected.update((target, frozenset([owner, peer]))
                            for peer in OWNERS if peer != owner)
        self.assertEqual(len(expected), 80)
        self.assertEqual(sum(len(active) == 2 for _, active in expected), 48)
        for plans in self.plans:
            cases = [case for plan in plans for case in plan['cases']]
            self.assertEqual(directed_modes(cases), expected)

    def test_shards_merge57_cases_without_losing_or_duplicating_any_arrival(self):
        for index, plans in enumerate(self.plans):
            saved = [case for plan in plans for case in plan['cases']]
            original = self.defs[index]['cases']
            self.assertEqual([len(plan['cases']) for plan in plans], [13, 13, 13, 13, 5])
            self.assertEqual(Counter(case['case_id'] for case in saved),
                             Counter(case['case_id'] for case in original))
            self.assertEqual(len(saved), 57)
            self.assertEqual(sum(len(case['arrivals']) for case in saved), 130)
            by_id = {case['case_id']: case for case in original}
            for case in saved:
                baseline = by_id[case['case_id']]
                self.assertEqual(case['mode'], baseline['mode'])
                self.assertEqual(case['offered'], len(baseline['arrivals']))
                stripped = [{k: v for k, v in row.items()
                             if k not in ('request_id', 'arrival_index')}
                            for row in case['arrivals']]
                self.assertEqual(stripped, baseline['arrivals'])
            shift = index * 17 % 57
            self.assertEqual([case['case_id'] for case in saved],
                             [case['case_id'] for case in original[shift:] + original[:shift]])
        self.assertEqual(sum(len(p['priming']) for plans in self.plans for p in plans), 270)

    def test_multiscale_offsets_order_reversal_and_same_key_combinations(self):
        for index, block in enumerate(self.defs):
            all4orders = []
            for case in block['cases']:
                times = [row['arrival_ns'] for row in case['arrivals']]
                self.assertEqual(times, sorted(times))
                self.assertLess(max(times), 10_000_000_000)
                if case['mode'] == 'solo_same_owner_burst':
                    self.assertEqual(times, [0, 1_000_000])
                elif case['mode'] == 'pair_activity':
                    self.assertEqual(times, [0, (1_000_000, 20_000_000, 300_000_000)[index]])
                else:
                    step = (1_000_000, 20_000_000, 50_000_000)[index]
                    self.assertEqual(times, [0, step, 2 * step, 3 * step])
                    all4orders.append(tuple(row['owner'] for row in case['arrivals']))
            self.assertGreater(len(set(all4orders)), 1)
        for aligned in zip(*(block['cases'] for block in self.defs)):
            self.assertTrue(all(Counter(key(r) for r in c['arrivals']) ==
                                Counter(key(r) for r in aligned[0]['arrivals']) for c in aligned))
            if aligned[0]['mode'] == 'pair_activity':
                first, second = [[row['owner'] for row in c['arrivals']] for c in aligned[:2]]
                self.assertEqual(first, second[::-1])

    def test_training_identities_disjoint_from_original_training_and_public_trace(self):
        plans = [plan for block in self.plans for plan in block]
        ids = [row['request_id'] for plan in plans for case in plan['cases'] for row in case['arrivals']]
        old_ids = {row['request_id'] for plan in self.old for case in plan['cases'] for row in case['arrivals']}
        old_ids.update(row['request_id'] for row in self.contract['trace'])
        self.assertEqual(len(ids), 390)
        self.assertEqual(len(set(ids)), 390)
        self.assertFalse(set(ids) & old_ids)
        for field in ('attempt_id', 'execution_id'):
            actual = [plan[field] for plan in plans]
            self.assertEqual(len(set(actual)), 15)
            self.assertFalse(set(actual) & {p.get(field) for p in self.old})
        self.assertEqual(len({p['capture_id'] for p in plans}), 3)
        self.assertFalse({p['capture_id'] for p in plans} & {p['capture_id'] for p in self.old})
        for plan in plans:
            self.assertEqual(plan['priming'], self.old[0]['priming'])
            self.assertEqual(len(plan['priming']), 18)
            self.assertEqual(plan['split'], 'training')
            self.assertTrue(plan['no_retries'])

    def test_exact_parent_execution_preserved_with_unqualified_context_extension(self):
        old = read(self.manifest['original_context'])
        new = read(self.manifest['artifacts']['candidate_context'])
        for field in old:
            if field not in ('declared_inputs', 'source_rebound_context_sha256'):
                self.assertEqual(new[field], old[field])
        self.assertEqual(new['declared_inputs'][:16], old['declared_inputs'])
        self.assertEqual(len(new['declared_inputs']), 22)
        self.assertNotIn('source_rebound_context_sha256', new)
        qualification = new['matching_macro_supplement']
        self.assertFalse(qualification['statistical_context_transfer_qualified'])
        self.assertFalse(qualification['physical_context_transfer_qualified'])
        for plans in self.plans:
            for plan in plans:
                self.assertEqual(plan['runtime_policy'], self.old[0]['runtime_policy'])
                self.assertIsNone(plan['runtime_policy']['admission'])
                self.assertTrue(plan['runtime_policy']['enqueue_all_without_waiting_for_RESULT_or_READY'])
                self.assertFalse(plan['runtime_policy']['batching'])
                self.assertFalse(plan['runtime_policy']['weight_reuse'])
                for field in ('runtime_config', 'collector_timing_policy', 'supervisor_budget', 'preflight_free_bytes_required'):
                    self.assertEqual(plan[field], self.old[0][field])

    def test_calendar_not_actual_overlap_rank_resource_fit_or_validation(self):
        for field in ('GPU_started', 'numerical_fit_run', 'validation_labels_read',
                      'qualified_fitted_model', 'resource_fit_eligible', 'kernel_resource_fit',
                      'main_queue_provider_eligible', 'final_gain_eligible'):
            self.assertIs(self.manifest[field], False)
        self.assertIsNone(self.manifest['execute_entrypoint'])
        for plans in self.plans:
            for plan in plans:
                self.assertIsNone(plan['execute_entrypoint'])
                self.assertEqual(plan['training_role'], 'complete_native_request_macro_template_support_not_kernel_or_resource_fit')
                for case in plan['cases']:
                    self.assertIs(case['actual_peer_overlap_known'], False)
                    self.assertIs(case['actual_phase_coverage_known'], False)
        self.assertFalse(read(self.manifest['artifacts']['coverage'])['measured_overlap_inferred_from_calendar'])
        for field in ('GPU_started', 'kernel_resource_fit', 'main_queue_provider_eligible'):
            changed = copy.deepcopy(self.manifest)
            changed[field] = True
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / 'manifest.json'
                target.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'CPU-only'):
                    m.audit(target)


if __name__ == '__main__':
    unittest.main()
