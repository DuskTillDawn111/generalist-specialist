"""Independent CPU contracts. Native extraction is synthetic; no solver runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_source_rebound_resource_v1 import inputs, prepared
from tests.test_twoFIFO_source_rebound_phase12_fit_v1 import save, complete_final
from tests.test_twoFIFO_fold_resource_rescore_v1 import setup_world
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_resource_v1 as m
from tools.predictor_validation_20261027.actual_worker_world_score_v3 import score_world


def model_at(root):
    design, manifest, final = prepared(root)
    bundle, intake, context = inputs(root/'data')
    save(final, complete_final(manifest['executor_manifest']))
    def ingest(*args, output):
        return save(Path(output)/'report.json', intake)
    return design, manifest, bundle, intake, context, ingest


class IndependentReboundResourceTests(unittest.TestCase):
    def test_legacy_macro_schema_fails_before_pure_resource_tables(self):
        with tempfile.TemporaryDirectory() as work:
            bundle, intake, _ = inputs(Path(work))
            bundle['documents'][0]['schema'] = 'gsdeploy.observed_handler_macros.v2'
            with patch.object(m.resource, 'table_from_verified') as table:
                with self.assertRaisesRegex(ValueError, 'macro identity'):
                    m.build_from_bundle(bundle, intake)
            table.assert_not_called()

    def test_old_raw_schema_with_fresh_receipts_is_not_relabelled(self):
        with tempfile.TemporaryDirectory() as work:
            bundle, intake, _ = inputs(Path(work))
            review_ref = bundle['review_receipts'][0]
            review = m.checked(review_ref)
            old_raw = review['raw']
            raw = m.checked(old_raw)
            raw['schema'] = 'gsdeploy.actual_worker_native_old.v3'
            Path(old_raw['path']).write_text(json.dumps(raw))
            new_raw = m.receipt(old_raw['path'])
            review['raw'] = new_raw
            review['sources'] = [new_raw if r == old_raw else r for r in review['sources']]
            Path(review_ref['path']).write_text(json.dumps(review))
            new_review = m.receipt(review_ref['path'])
            bundle['review_receipts'][0] = new_review
            bundle['documents'][0]['sources'] = [new_raw]
            intake['sources'] = m.strict_references([
                new_raw if r == old_raw else new_review if r == review_ref else r
                for r in intake['sources']])
            with patch.object(m.resource, 'table_from_verified') as table:
                with self.assertRaisesRegex(ValueError, 'raw/context identity'):
                    m.build_from_bundle(bundle, intake)
            table.assert_not_called()

    def test_last_table_source_mutation_is_rejected_after_actual_aggregation(self):
        with tempfile.TemporaryDirectory() as work:
            bundle, intake, _ = inputs(Path(work))
            raw_path = Path(m.checked(bundle['review_receipts'][0])['raw']['path'])
            original = m.resource.table_from_verified
            calls = []
            def table(*args):
                result = original(*args)
                calls.append(args[2]['attempt_id'])
                if len(calls) == 12:
                    raw_path.write_text('{}')
                return result
            with patch.object(m.resource, 'table_from_verified', side_effect=table):
                with self.assertRaisesRegex(ValueError, 'Changed'):
                    m.build_from_bundle(bundle, intake)
            self.assertEqual(len(calls), 12)

    def test_new_implementation_closure_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            bundle, intake, _ = inputs(root)
            original = m.implementation_sources(m.__file__)
            extra = save(root/'new-import.json', {'CPU_FIXTURE': True})
            with patch.object(m, 'implementation_sources', side_effect=[original,
                    m.strict_references([original, extra])]):
                with self.assertRaisesRegex(ValueError, 'implementation closure changed'):
                    m.build_from_bundle(bundle, intake)

    def test_saved_model_rederives_every_fold_not_just_selected_fold(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            design, manifest, bundle, intake, context, ingest = model_at(root)
            target = next(p for p in bundle['plans'] if p['capture_id'] == 'training-block-3')
            with patch.object(m.response, 'audit', return_value=manifest), \
                 patch.object(m.response.intake, 'intake', side_effect=ingest), \
                 patch.object(m.response.adapter, 'build_training_rows', return_value=bundle), \
                 patch.object(m.response.original, 'fitted_rows', side_effect=AssertionError('no solver')):
                model = m.run(design['path'], root/'model', fit_enabled=True)
                provider = m.FoldResourceTrainingProvider(model['path'], target, context,
                    heldout_block_id='training-block-3', allow_training_resource_hypothesis=True)
                self.assertEqual(len(provider.document['training_attempt_ids']), 8)
                self.assertEqual(len(provider.fold_binding['heldout_attempt_ids']), 4)
                self.assertFalse(set(provider.document['training_attempt_ids']) &
                    set(provider.fold_binding['heldout_attempt_ids']))
                for field, value in (('capture_id', 'training-block-1'),
                                     ('attempt_id', 'unseen-attempt')):
                    foreign = copy.deepcopy(target)
                    foreign[field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        m.FoldResourceTrainingProvider(model['path'], foreign, context,
                            heldout_block_id='training-block-3', allow_training_resource_hypothesis=True)
                bad = m.checked(model)
                bad['whole_block_leaveout'][0]['training_model']['startup']['joules'] += 999
                corrupt = save(root/'corrupt.json', bad)
                with self.assertRaisesRegex(ValueError, 'must rederive'):
                    m.FoldResourceTrainingProvider(corrupt['path'], target, context,
                        heldout_block_id='training-block-3', allow_training_resource_hypothesis=True)

    def test_real_original_score_missing_history_and_drain_peak_scopes(self):
        plan, context, tables, requests, args, document, history = setup_world()
        full, folds = m._fits(tables)
        selected = next(f['training_model'] for f in folds if f['heldout_block_id'] == 'training-block-3')
        provider = object.__new__(m.FoldResourceTrainingProvider)
        provider.document = selected
        provider.scenario = 'mean'
        provider.target_plan = plan
        provider.target_context = context
        provider.physical_execution_identity = m.resource.physical_execution_identity(plan, context)
        def score(initial):
            return score_world(document['draws'][0]['forecast']['world'], requests,
                args['scoring_plan'], args['quality'], window_start_ns=args['window_start_ns'],
                window_end_ns=args['window_end_ns'], drain_end_ns=args['drain_end_ns'],
                resource_provider=provider, target_context=context,
                resource_source_evidence={'CPU_FIXTURE_NOT_NATIVE': True},
                resource_memory_initial_state=initial)
        missing = score(None)
        self.assertIsNone(missing['original_score']['terms']['M'])
        self.assertIsNone(missing['original_score']['full_U'])
        self.assertAlmostEqual(missing['original_score']['terms']['E'], (3+15/3600)/10)
        explicit = copy.deepcopy(history)
        explicit['prior_owned_sampled_peak_mib'] = 8000
        known = score(explicit)
        self.assertEqual(known['original_score']['terms']['M'], 8000*1.2/1000)
        self.assertIsNotNone(known['original_score']['full_U'])
        board = document['draws'][0]['forecast']['scoring']['resource_timeline']['slices']
        changed = copy.deepcopy(tables)
        from tools.predictor_validation_20261027.macro_resource_timeline_v1 import clock
        for table in changed:
            for row, interval in zip(table['slices'], board):
                if clock(interval['measurement_overlap_ns']) == 0:
                    row['sampled_owned_peak_mib'] = 99999
        provider.document = next(f['training_model'] for f in m._fits(changed)[1]
            if f['heldout_block_id'] == 'training-block-3')
        drain = score(explicit)
        self.assertEqual(drain['original_score']['terms']['E'], known['original_score']['terms']['E'])
        self.assertEqual(drain['original_score']['terms']['M'], 99999*1.2/1000)


if __name__ == '__main__':
    unittest.main()
