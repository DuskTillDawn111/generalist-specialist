"""CPU projection fixtures only; no synthetic values are campaign evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import campaign_final_summary_v2 as m
from tests import test_h600_four_group_statistics_v1 as statistics_fixture
from tools.predictor_validation_20261027 import h600_four_group_statistics_v1 as stats


class CampaignSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)/'sources'
        self.base.mkdir()

    def put(self, path, value):
        path = Path(path)
        if not path.is_absolute():
            path = self.base/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def summary(self):
        return m.build(m.Snapshot(self.base))

    def test_missing_reports_keep_pending_and_declared_cells_without_guessing_counts(self):
        summary = self.summary()
        self.assertEqual(summary['status'], 'PENDING_CAMPAIGN')
        self.assertEqual(len(summary['stages']['final_four_groups']['cells']), 8)
        self.assertTrue(all(row['known'] is None and row['terms'] is None
            for row in summary['stages']['final_four_groups']['cells']))
        self.assertTrue(all(row['status'] == 'PENDING' for row in
            summary['stages']['admission_training']['fitted_manifests'].values()))
        self.assertIn('admission2/spill_high', m.markdown(summary))

    def test_prediction_gap_distinct_from_native_quality_completion_and_gate(self):
        self.put(m.SOURCES['validation_report'], dict(status='ASSESSED', candidates=[dict(
            candidate_id='admission2_balanced', requested=1200, known=0,
            observed_metrics=dict(full_U=-5, six_term_comparison_complete=True))],
            delta_sign_accuracy=None, rank_accuracy=None))
        self.put(m.SOURCES['validation_gate'], dict(status='INCOMPLETE',
            failed_checks=['queue'], unknown_checks=['sign']))
        self.put(m.SOURCES['validation_audit'], dict(evidence=[dict(
            candidate_id='admission2_balanced', requested=1200, known=1200,
            quality_audit=dict(count=1200, independently_recomputed=True),
            actual_metrics=dict(six_term_comparison_complete=True))]))
        summary = self.summary()
        actual = summary['stages']['admission_validation']
        self.assertEqual(actual['candidates'][0]['known'], 0)
        self.assertEqual(actual['native_evidence'][0]['quality_count'], 1200)
        self.assertEqual(actual['gate']['status'], 'INCOMPLETE')
        self.assertEqual(actual['gate']['unknown_checks'], ['sign'])
        text = m.markdown(summary)
        self.assertIn('0/1200|1200/1200|True', text)
        self.assertIn('V4失败项：queue', text)
        self.assertIn('V4未知项：sign', text)

    def test_real_statistics_fixture_requires_direct_actual_audits_for_six_terms(self):
        # Reuse the frozen pure-statistics test's real row schema. Its block
        # output intentionally has no rows/terms, only utility maps.
        fixture = statistics_fixture.FourGroupStatisticsTests()
        fixture.setUp()
        rows = fixture.rows
        for row in rows:
            row['original_qualification'] = 'UNKNOWN'
        rows[0]['original_qualification'] = 'FAIL'
        comparison = stats.block(fixture.locked, rows)
        self.assertNotIn('rows', comparison)
        audits = []
        for row in rows:
            cid = row['mode']+'_'+row['route']
            audits.append(self.put('final-audits/'+cid+'.json', dict(
                candidate_id=cid, requests=1200, known=1200, quality=dict(count=1200),
                six_term_score=dict(terms=row['terms'], diagnostic_U=-row['terms']['Q'], utility_complete=True),
                original_stability=dict(original_stability_passed=cid != 'shared_balanced',
                    stability=dict(stability_state_v2='unstable' if cid=='shared_balanced' else 'stable')))))
        self.put(m.SOURCES['final_report'], dict(status='DESCRIPTIVE_COMPLETE', known=9600,
            requested=9600, descriptive_comparison=comparison, actual_audits=audits,
            primary_value=comparison['modes']['admission2']['locked_allow_minus_strong_no_spill']))
        summary = self.summary()
        final = summary['stages']['final_four_groups']
        self.assertEqual(final['primary_value'], -2.)
        self.assertEqual(len(final['cells']), 8)
        self.assertEqual(final['cells'][0]['terms'], rows[0]['terms'])
        self.assertEqual(final['cells'][0]['original_qualification'], 'FAIL')
        self.assertFalse(final['cells'][0]['original_stability_passed'])
        self.assertIsNone(final['execution_completion'])
        self.assertFalse(summary['final_gain_eligible'])
        self.assertIn('shared/balanced|ACTUAL_AUDIT_PRESENT|1200/1200|-2.000000', m.markdown(summary))
        self.assertEqual(len([k for k in summary['source_receipts'] if k.startswith('final_actual_audit_')]),8)
        # Utility maps alone cannot manufacture the missing actual six terms.
        Path(audits[0]['path']).unlink()
        missing = self.summary()['stages']['final_four_groups']['cells'][0]
        self.assertEqual(missing['status'], 'PENDING')
        self.assertIsNone(missing['terms'])
        self.assertEqual(missing['U'], -2.)

    def test_manifest_cv_and_actual_full_source_counts_projected_without_fit(self):
        raw = dict(path='synthetic/raw.json', sha256='fixture')
        capacity_fit = self.put('capacity-fit.json', dict(
            full_training_references=[dict(seed=17, candidate_id='admission2_balanced', heldout_raw=raw)],
            whole_seed_cv=[dict(seed=18,candidate_id='admission2_balanced', training_seeds=[17],
                service_applicability_diagnostic=False, service_checks=[])],
            independent_validation_sources_used=[]))
        self.put(m.SOURCES['capacity_manifest'], dict(schema='capacity.manifest', fit=capacity_fit,
            source_receipts=[raw], independent_validation_sources_used=[]))
        for kind in ('resource', 'quality'):
            cv = self.put(kind+'-cv.json', dict(rows=[dict(seed=18, candidate_id='admission2_balanced',
                training_seeds=[17], unknown_contexts=[])]))
            evidence = dict(training_seeds=[17], runs=[dict(seed=17,raw=raw,quality_audit=dict(count=37))])
            model = dict(admission_training_evidence=evidence) if kind == 'resource' else dict(
                **evidence, request_count=37)
            full = self.put(kind+'-model.json', model)
            self.put(m.SOURCES[kind+'_manifest'], dict(schema=kind+'.manifest',
                **{kind+'_cv':cv}, full=dict(training_seeds=[17],
                    **{'route_resources' if kind=='resource' else 'route_quality':{'admission2_balanced':full}}),
                folds=[dict(omitted_seed=18,training_seeds=[17])], source_receipts=[raw],
                independent_validation_sources_used=[], no_validation_data=True))
        summary = self.summary()
        training = summary['stages']['admission_training']
        self.assertEqual(training['status'], 'PENDING')  # no completion fabricated
        manifests = training['fitted_manifests']
        self.assertEqual(manifests['capacity']['cv']['row_count'],1)
        self.assertFalse(manifests['capacity']['cv']['rows'][0]['service_applicability_diagnostic'])
        self.assertEqual(manifests['resource']['full_source_counts']['unique_native_raw_receipts'],1)
        self.assertEqual(manifests['quality']['full_source_counts']['total_quality_requests'],37)
        self.assertNotEqual(manifests['quality']['full_source_counts']['total_quality_requests'],9600)
        self.assertIn('|quality|MANIFEST_PRESENT|1|1|37|',m.markdown(summary))

    def test_direct_reference_sha_and_input_mutation_checks_and_no_overwrite(self):
        ref=self.put('value.json',dict(status='FAIL'))
        snapshot=m.Snapshot(self.base)
        snapshot.read_ref('value',ref)
        self.put('value.json',dict(status='PASS'))
        with self.assertRaisesRegex(ValueError,'Input changed'):snapshot.recheck()
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.Snapshot(self.base).read_ref('bad',ref)
        out=Path(self.temp.name)/'out'
        m.run(self.base,out)
        before=(out/'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):m.run(self.base,out)
        self.assertEqual((out/'summary.json').read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
