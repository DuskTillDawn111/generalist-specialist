import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027 import duration_crossfit_report as module
from tools.predictor_validation_20261027.native_flow import receipt


class DurationCrossfitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.verifier_source = self.write('verifier.py', {})
        self.predictor_source = self.write('predictor.py', {})
        self.resources = self.write('resources.json', {})
        self.quality = self.write('quality.json', {})
        jobs, plans, folds = [], {}, []
        self.verifications = {}
        for seed in (1, 2):
            plan = {'seed': seed, 'trace': [{'request_id': 'r'}], 'split': 'training',
                    'phase': 'predictor_training', 'utility_tie_tolerance': 0.1,
                    'candidates': [{'id': cid, 'execution_context_sha256': 'ctx'} for cid in sorted(module.ROUTES)],
                    'baseline_candidate_id': 'shared_skewed'}
            plans[seed] = self.write(f'plan-{seed}.json', plan)
            for cid in sorted(module.ROUTES):
                raw = self.write(f'{seed}-{cid}-raw.json', {'plan': receipt(plans[seed]), 'candidate_id': cid})
                jobs.append({'flow_id': f'seed-{seed}__{cid}', 'seed': seed,
                             'raw_path': str(raw), 'plan': receipt(plans[seed]), 'candidate_id': cid})
        self.protocol = self.write('protocol.json', {'expected_flow_count': 8, 'flow_jobs': jobs})
        selection = self.write('selection.json', {})
        addendum = self.write('addendum.json', {})
        duration_protocol = self.write('duration-protocol.json', {
            'selection_protocol': receipt(selection), 'threshold_addendum': receipt(addendum)})
        for seed in (1, 2):
            model = self.write(f'model-{seed}.json', {'training_evidence_tree': {'protocol': receipt(self.protocol)}})
            training = [job for job in jobs if job['seed'] != seed]
            heldout = [job for job in jobs if job['seed'] == seed]
            self.verifications[str(model)] = {'verified': True, 'duration_model_verified': True,
                'omitted_seed': seed, 'omitted_flow_ids': [j['flow_id'] for j in heldout],
                'training_flow_count': 4, 'training_flow_ids': [j['flow_id'] for j in training],
                'training_raw_receipts': [receipt(j['raw_path']) for j in training],
                'training_trace_ids': ['flow:' + receipt(j['raw_path'])['sha256'] for j in training],
                'execution_context_sha256': 'ctx'}
            flows = []
            for j in heldout:
                raw_ref = receipt(j['raw_path'])
                flows.append({'flow_id': j['flow_id'], 'heldout_raw': raw_ref, 'heldout_plan': j['plan'],
                              'candidate_id': j['candidate_id'], 'heldout_trace_id': 'flow:' + raw_ref['sha256']})
            folds.append({'unit': 'seed', 'fold_id': f'seed-{seed}', 'heldout_seed': seed,
                          'heldout_flows': flows, 'training_flow_ids': [j['flow_id'] for j in training],
                          'model': receipt(model)})
        self.manifest = self.write('manifest.json', {'schema': module.MANIFEST_SCHEMA,
            'estimator': module.ESTIMATOR, 'training_verifier': {'name': 'verify_duration_model',
            'source': receipt(self.verifier_source)}, 'protocol': receipt(self.protocol),
            'duration_protocol': receipt(duration_protocol),
            'folds': folds, 'retrospective_training_cross_validation': True, 'no_independent_validation': True})
        mock = patch.object(module, 'verify_duration_model', side_effect=lambda path: self.verifications[path])
        mock.start()
        self.addCleanup(mock.stop)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def mutate(self, callback):
        value = json.loads(self.manifest.read_text())
        callback(value)
        self.write('manifest.json', value)

    def test_complete_seed_identity_contract(self):
        value = module.audit_manifest(self.manifest)
        self.assertEqual(len(value['cases']), 8)
        self.assertEqual(value['contracts'][1]['baseline'], 'shared_skewed')
        self.assertEqual(value['contracts'][1]['tie_tolerance'], 0.1)

    def test_unknown_schema_or_wrong_verifier_does_not_pass(self):
        self.mutate(lambda d: d.update(estimator='old-loss'))
        with self.assertRaisesRegex(ValueError, 'duration'):
            module.audit_manifest(self.manifest)

    def test_missing_seed_or_route_rejected(self):
        self.mutate(lambda d: d['folds'][0]['heldout_flows'].pop())
        with self.assertRaisesRegex(ValueError, 'routes'):
            module.audit_manifest(self.manifest)

    def test_duplicate_seed_rejected(self):
        self.mutate(lambda d: d['folds'].__setitem__(1, copy.deepcopy(d['folds'][0])))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.audit_manifest(self.manifest)

    def test_invented_trace_cannot_mask_native_raw(self):
        self.mutate(lambda d: d['folds'][0]['heldout_flows'][0].update(heldout_trace_id='fake'))
        with self.assertRaisesRegex(ValueError, 'identity'):
            module.audit_manifest(self.manifest)

    def test_heldout_raw_or_hash_in_training_rejected(self):
        document = json.loads(self.manifest.read_text())
        fold = document['folds'][0]
        heldout = fold['heldout_flows'][0]['heldout_raw']
        self.verifications[fold['model']['path']]['training_raw_receipts'].append(
            {'path': 'renamed-source', 'sha256': heldout['sha256']})
        with self.assertRaisesRegex(ValueError, 'identity'):
            module.audit_manifest(self.manifest)

    def test_model_must_exclude_whole_seed_and_train_exact_complement(self):
        document = json.loads(self.manifest.read_text())
        verification = self.verifications[document['folds'][0]['model']['path']]
        verification['omitted_flow_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'exclude'):
            module.audit_manifest(self.manifest)

    def test_validation_sources_cannot_be_relabeled_by_manifest(self):
        document = json.loads(self.manifest.read_text())
        ref = document['folds'][0]['heldout_flows'][0]['heldout_plan']
        plan = json.loads(Path(ref['path']).read_text())
        plan['split'] = 'validation'
        Path(ref['path']).write_text(json.dumps(plan))
        # Immutable receipt mismatch rejects even before role checking.
        with self.assertRaises((ValueError, RuntimeError)):
            module.audit_manifest(self.manifest)

    def test_fresh_run_uses_plan_baseline_ties_and_retains_unknown_denominator(self):
        utilities = {'shared_balanced': 1., 'shared_skewed': 2.,
                     'shared_spill_low': 2.04, 'shared_spill_high': 3.}
        def compare(plan, candidate, prediction, raw):
            return {'requested': 2, 'known': 1, 'candidate_id': candidate['id'],
                    'predicted_comparison_U': utilities[candidate['id']],
                    'observed_U': utilities[candidate['id']], 'eligible_for_gain_claim': False}
        with patch.object(module, 'predictor_sources', return_value=[receipt(self.predictor_source)]), \
             patch.object(module, 'verifier_source', return_value=receipt(self.verifier_source)), \
             patch.object(module.core, 'scored_heldout', side_effect=lambda ref: ({}, ref)), \
             patch.object(module.core, 'predict', return_value={}) as predict, \
             patch.object(module.core, 'compare', side_effect=compare):
            result = module.run(self.manifest, self.resources, self.quality, self.root / 'report')
        self.assertEqual(predict.call_count, 8)
        self.assertEqual((result['known'], result['requested']), (8, 16))
        self.assertFalse(result['complete_prediction_coverage'])
        self.assertFalse(result['independent_validation'])
        self.assertFalse(result['final_gain_eligible'])
        self.assertEqual(result['predictor_source_receipts'], [receipt(self.predictor_source)])
        deltas = result['by_seed']['1']['paired_deltas']
        self.assertEqual({r['candidate_id'] for r in deltas}, module.ROUTES - {'shared_skewed'})
        self.assertTrue((self.root / 'report/report.json').exists())

    def test_changed_predictor_source_during_replay_prevents_report(self):
        def change(*args):
            self.predictor_source.write_text('changed')
            return {}
        with patch.object(module, 'predictor_sources', return_value=[receipt(self.predictor_source)]), \
             patch.object(module, 'verifier_source', return_value=receipt(self.verifier_source)), \
             patch.object(module.core, 'scored_heldout', side_effect=lambda ref: ({}, ref)), \
             patch.object(module.core, 'predict', side_effect=change), \
             patch.object(module.core, 'compare', side_effect=lambda *args: {'requested': 1, 'known': 1,
                 'predicted_comparison_U': 1., 'observed_U': 1.}):
            with self.assertRaisesRegex(ValueError, 'Changed bound source'):
                module.run(self.manifest, self.resources, self.quality, self.root / 'report')
        self.assertFalse((self.root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
