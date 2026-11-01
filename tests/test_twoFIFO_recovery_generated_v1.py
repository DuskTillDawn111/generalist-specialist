"""CPU checks of recovery identity and saved-fit reuse; no GPU or fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_recovery_generated_v1 as m


class RecoveryGeneratedTests(unittest.TestCase):
    def test_predeclare_keeps_original_calendar_draws_and_six_scores(self):
        prior = json.loads(Path('artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-prepared/design.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text('{}')
            recovery = dict(path=str(manifest), sha256=m.receipt(manifest)['sha256'])
            with patch.object(m.response, 'audit', return_value=dict(sources=[], recovery_manifest=recovery)):
                ref = m.prepare(manifest, prior['scoring_plan']['path'], prior['quality']['path'], root / 'prepared')
                doc = m.audit(ref['path'])
                self.assertEqual(doc['scheme'], prior['scheme'])
                self.assertEqual(doc['six_term_weights'], prior['six_term_weights'])
                self.assertEqual(doc['expected_measured_requests'], 546)
                for field, bad in [('recovery_manifest', {'path': 'foreign', 'sha256': 'f'*64}),
                                   ('scheme_sha256', 'changed'), ('expected_cases', 149),
                                   ('queue_bounds_used_for_utility', True)]:
                    modified = copy.deepcopy(doc)
                    modified[field] = bad
                    path = root / (field + '.json')
                    path.write_text(json.dumps(modified))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        m.audit(path)
                (root / 'phase12-fit-claim.json').write_text('{}')
                with self.assertRaises(ValueError):
                    m.prepare(manifest, prior['scoring_plan']['path'], prior['quality']['path'], root / 'late')
                self.assertFalse((root / 'late').exists())

    def test_saved_fit_reuse_keeps_recovery_provenance_and_detects_source_drift(self):
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def put(name, value):
                    path = root / name
                    path.write_text(json.dumps(value))
                    return m.receipt(path)
                design_ref = put('design.json', {})
                manifest_ref, fit_ref = put('manifest.json', {}), put('fit.json', {})
                final_ref, score_ref = put('final.json', {}), put('score.json', {})
                quality_ref, recovery_ref = put('quality.json', {}), put('recovery.json', {})
                intake = dict(schema=m.response.intake.SCHEMA, sources=[])
                design = dict(fit_manifest=manifest_ref, scoring_plan=score_ref, quality=quality_ref,
                              recovery_manifest=recovery_ref, sources=[])
                fit = dict(manifest=manifest_ref, continuation=final_ref, intake_evidence=intake,
                           fit={'fixture': True}, sources=[])
                calls = []
                def verified(manifest, final, callback):
                    calls.append('verified')
                    self.assertEqual((manifest, final), (manifest_ref['path'], final_ref['path']))
                    return callback({'verified_bundle': True}, intake)
                def numerical(bundle, fitted, score, quality, output):
                    calls.append('numerical')
                    Path(output).mkdir()
                    if drift:
                        Path(fit_ref['path']).write_text('changed')
                    return dict(schema=m.numerical.SCHEMA, numerical_fit_run=False,
                                all_offered_requests_and_original_probability_mass_retained=True)
                with patch.object(m, 'audit', return_value=design), \
                     patch.object(m.response, 'audit', return_value={}), \
                     patch.object(m.response, 'check_intake') as check, \
                     patch.object(m.response, 'audit_fit', return_value=dict(report=fit, receipt=fit_ref)), \
                     patch.object(m.response, 'with_verified_bundle', side_effect=verified), \
                     patch.object(m.numerical, 'evaluate_bundle', side_effect=numerical), \
                     patch.object(m.response, 'fit_bundle', side_effect=AssertionError('refitting forbidden')):
                    if drift:
                        with self.assertRaises(ValueError):
                            m.run(design_ref['path'], fit_ref['path'], root / 'out')
                        self.assertFalse((root / 'out' / 'report.json').exists())
                    else:
                        doc = m.checked(m.run(design_ref['path'], fit_ref['path'], root / 'out'))
                        self.assertEqual(doc['schema'], m.SCHEMA)
                        self.assertEqual(doc['upstream_intake_schema'], m.response.intake.SCHEMA)
                        self.assertEqual(doc['recovery_manifest'], recovery_ref)
                        self.assertEqual(doc['numerical_interface_schema'], m.numerical.SCHEMA)
                        self.assertFalse(doc['numerical_fit_run'])
                    check.assert_called_once()
                self.assertEqual(calls, ['verified', 'numerical'])

    def test_old_producer_fit_is_rejected_before_any_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / 'design.json'
            design.write_text('{}')
            fit = root / 'old-fit.json'
            fit.write_text(json.dumps(dict(schema=m.numerical.response.SCHEMA,
                status='COMPLETE_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS', numerical_fit_run=True)))
            with patch.object(m, 'audit', return_value={}), patch.object(m.numerical, 'evaluate_bundle') as evaluate:
                with self.assertRaises(ValueError):
                    m.run(design, fit, root / 'out')
                evaluate.assert_not_called()
                self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
