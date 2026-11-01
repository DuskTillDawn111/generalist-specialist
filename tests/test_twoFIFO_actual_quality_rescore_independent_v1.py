"""CPU sidecar integration; upstream generated.audit is explicitly mocked.

Native quality arithmetic, file receipts, cohort checks and output writes are
real. These fixtures do not establish actual native training authentication.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_actual_quality_rescore_v1 import fixture
from tools.predictor_validation_20261027 import twoFIFO_actual_quality_rescore_v1 as m


def write(path, value):
    path.write_text(json.dumps(value))
    return m.receipt(path)


def full_fixture(root, *, unknown_first=False):
    raw_template, plan_template, _, score, catalog, original, prediction = fixture(root)
    case_template = copy.deepcopy(plan_template['cases'][0])
    records_template = copy.deepcopy(raw_template['request_records'])
    context_ref = write(root/'context.json', dict(residents={'ocr#0': 'ocr'}))
    plan_refs, plans, raws, raw_refs, by_case, output_rows = [], [], [], [], [], []
    index = 0
    for block in range(3):
        for shard in range(4):
            attempt = 'block-' + str(block) + '-shard-' + str(shard)
            plan = copy.deepcopy(plan_template)
            plan.update(capture_id='block-' + str(block), attempt_id=attempt,
                        context=context_ref, cases=[])
            raw = copy.deepcopy(raw_template)
            raw.update(capture_id=plan['capture_id'], attempt_id=attempt, cases=[], request_records=[], context=context_ref)
            for _ in range(13 if shard < 2 else 12):
                count = 4 if index < 96 else 3
                case = dict(case_id='case-' + str(index), arrivals=[])
                for ordinal in range(count):
                    source = ordinal % 2
                    request_id = 'case-' + str(index) + '-r' + str(ordinal)
                    arrival = copy.deepcopy(case_template['arrivals'][source])
                    arrival.update(request_id=request_id, arrival_ns=ordinal)
                    record = copy.deepcopy(records_template[source])
                    record.update(request_id=request_id, scheduled_time_ns=100+ordinal, completion_time_ns=200+ordinal)
                    record['worker_response']['request_id'] = request_id
                    if unknown_first and index == 0 and ordinal == 0: record['status'] = 'pending'
                    case['arrivals'].append(arrival); raw['request_records'].append(record)
                plan['cases'].append(case)
                raw['cases'].append(dict(case_id=case['case_id'], observation_start_ns=100))
                by_case.append((len(plans), case)); index += 1
            ref = write(root/(attempt+'-plan.json'),plan)
            raw['plan'] = ref
            plan_refs.append(ref); plans.append(plan); raws.append(raw)
    executor_ref = write(root/'executor.json', dict(attempts=[dict(plan=r) for r in plan_refs]))
    review_refs = []
    for i,raw in enumerate(raws):
        ref = write(root/('raw-'+str(i)+'.json'),raw); raw_refs.append(ref)
        review_refs.append(write(root/('review-'+str(i)+'.json'), dict(raw=ref, issues=[],
            schema=m.generated.response.intake.native_review.SCHEMA,
            source_rebound_manifest=executor_ref, inputs={'plan.json':plan_refs[i]},
            training_feature_extraction_eligible=True)))
    score_ref=write(root/'score.json', score)
    manifest_ref=write(root/'fit-manifest.json',dict(original_executor_manifest=executor_ref))
    design=dict(scoring_plan=score_ref,fit_manifest=manifest_ref)
    design_ref=write(root/'design.json',design)
    intake=dict(attempts=[dict(attempt_id=p['attempt_id'],training_review=r) for p,r in zip(plans,review_refs)])
    fitted=dict(manifest=manifest_ref,intake_evidence=intake,continuation={'path':'mock-continuation'},sources=[])
    fit_ref=write(root/'fit.json',fitted)
    resource_design=write(root/'resource-design.json',dict(fit_manifest=manifest_ref))
    model_ref=write(root/'model.json',dict(schema=m.resource.SCHEMA,status=m.resource.COMPLETE_STATUS,
        design=resource_design,intake_evidence=intake,continuation=fitted['continuation'],sources=[]))
    for index,(i,case) in enumerate(by_case):
        pred=copy.deepcopy(prediction); pred.update(target_plan=plan_refs[i],resource_model=model_ref,
            resource_fold_binding=dict(heldout_block_id=plans[i]['capture_id']))
        archived=write(root/('archived-'+str(index)+'.json'),dict(case_id=case['case_id']))
        pred['archived_prediction']=archived
        pred_ref=write(root/('prediction-'+str(index)+'.json'),pred)
        observed=copy.deepcopy(original)
        observed.update(attempt_id=plans[i]['attempt_id'],case_id=case['case_id'],offered_requests=len(case['arrivals']),
            reader_labels=dict(requests=[dict(request_id=r['request_id']) for r in case['arrivals']]))
        observed['measured_resources'].update(original_window_ns=[100,10_000_000_100],
            source_evidence=dict(raw=raw_refs[i],training_review=review_refs[i],persisted_resource_prediction=pred_ref))
        label_ref=write(root/('observed-'+str(index)+'.json'),observed)
        output_rows.append(dict(block_id=plans[i]['capture_id'],attempt_id=plans[i]['attempt_id'],
            case_id=case['case_id'],offered_requests=len(case['arrivals']),prediction=pred_ref,
            measured_label=label_ref))
    folds=[]
    for block in range(3):
        block_id='block-'+str(block)
        folds.append(dict(heldout_block_id=block_id,donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=block_id],
            heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==block_id],
            training_block_ids=['block-'+str(b) for b in range(3) if b!=block],
            cases=[dict(attempt_id=r['attempt_id'],case_id=r['case_id'],
                        prediction=m.checked(r['prediction'])['archived_prediction']) for r in output_rows if r['block_id']==block_id]))
    parent_ref=write(root/'parent.json',dict(schema=m.generated.SCHEMA,response_fit_reused=True,
        design=design_ref,response_fit=fit_ref,whole_block_leaveout=folds,sources=[]))
    report=dict(schema=m.previous.SCHEMA,status='COMPLETE_REBOUND_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS',
        generated_producer_schema=m.generated.SCHEMA,labels_generated=True,validation_labels_read=False,
        cases=150,offered_requests=546,completed_generated_report=parent_ref,resource_model=model_ref,
        sources=[parent_ref,design_ref,score_ref],output_cases=output_rows,
        fold_provider_cache={'block-'+str(b):dict(heldout_block_id='block-'+str(b)) for b in range(3)})
    design['_mock_fitted']=fitted
    design['_mock_manifest']=m.checked(manifest_ref)
    return report,catalog,design


class IndependentActualQualityRescoreTests(unittest.TestCase):
    def test_duplicate_reader_registry_cannot_pass_set_only_cohort_check(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory))
            args[5]['reader_labels']['requests'].append(dict(request_id='r0'))
            with self.assertRaisesRegex(ValueError,'identical offered cohort'):
                m.evaluate_case(*args[:3],100,*args[3:])

    def test_unknown_prediction_expectations_and_exact_distribution_survive_actual_Q(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory))
            expected=dict(expected_value=None,known_probability_mass={'numerator':1,'denominator':3},
                unknown_probability_mass={'numerator':2,'denominator':3},conditional_on_known_mean=.99)
            args[6]['ensemble']['six_terms']['Q']=expected
            args[6]['ensemble']['utility']['expected_value']=None
            measured,score,errors=m.evaluate_case(*args[:3],100,*args[3:])
            self.assertEqual(measured['measured_Q'],.1)
            self.assertIsNotNone(score['full_U'])
            self.assertIsNone(errors['terms']['Q']['error'])
            self.assertIsNone(errors['U_error'])
            self.assertEqual(errors['terms']['Q']['original_distribution'],expected)
            self.assertFalse(errors['UNKNOWN_mass_dropped'])
            self.assertFalse(errors['conditional_U_used_as_complete_U'])

    def run_fixture(self, root, report, catalog, design):
        report_ref = write(root/'rescore.json', report)
        with patch.object(m.generated, 'audit', return_value={k:v for k,v in design.items() if not k.startswith('_mock_')}), \
             patch.object(m.generated.response, 'audit_fit', return_value={'report':design['_mock_fitted']}), \
             patch.object(m.generated.response, 'audit', return_value=design['_mock_manifest']), \
             patch.object(m.resource, 'audit', return_value={'fit_manifest':design['fit_manifest']}):
            return m.checked(m.run(report_ref['path'], catalog['path'], root/'sidecar'))

    def test_150case_546request_sidecar_keeps_unknown_and_original_prediction_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, catalog, design = full_fixture(root, unknown_first=True)
            originals = {ref['path']: ref for row in report['output_cases'] for ref in (row['prediction'], row['measured_label'])}
            out = self.run_fixture(root, report, catalog, design)
            self.assertEqual((out['cases'], out['offered_requests']), (150, 546))
            self.assertEqual(out['known_actual_quality_cases'], 149)
            self.assertEqual(out['full_six_term_label_cases'], 149)
            self.assertIsNone(out['output_cases'][0]['score_errors']['terms']['Q']['error'])
            self.assertFalse(out['output_cases'][0]['score_errors']['frozen_quality_used_on_both_sides'])
            self.assertTrue(out['output_cases'][0]['score_errors']['actual_native_outputs_used_for_measured_quality'])
            self.assertEqual({path: m.receipt(path) for path in originals}, originals)

    def test_numerical_upstream_values_are_explicitly_trusted_not_rederived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, catalog, design = full_fixture(root)
            row = report['output_cases'][0]
            prediction = m.checked(row['prediction'])
            prediction['ensemble']['six_terms']['Q']['expected_value'] = .99
            foreign = write(root/'foreign-prediction.json', prediction)
            measured = m.checked(row['measured_label'])
            measured['measured_resources']['source_evidence']['persisted_resource_prediction'] = foreign
            row['prediction'] = foreign
            row['measured_label'] = write(root/'foreign-measured.json', measured)
            out = self.run_fixture(root, report, catalog, design)
            self.assertEqual(out['output_cases'][0]['original_prediction'], foreign)
            self.assertEqual(out['output_cases'][0]['score_errors']['terms']['Q']['predicted_unconditional_expected_value'], .99)
            self.assertFalse(out['upstream_predictions_numerically_recomputed'])
            self.assertFalse(out['upstream_timing_and_resource_labels_numerically_recomputed'])

    def test_foreign_prediction_archive_resource_or_fold_is_rejected(self):
        for kind in ('archive','resource','fold'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);report,catalog,design=full_fixture(root)
                row=report['output_cases'][0];prediction=m.checked(row['prediction'])
                if kind=='archive':prediction['archived_prediction']=report['output_cases'][1]['prediction']
                elif kind=='resource':prediction['resource_model']={'path':'foreign-model','sha256':'f'*64}
                else:prediction['resource_fold_binding']={'heldout_block_id':'wrong-block'}
                row['prediction']=write(root/'foreign-prediction.json',prediction)
                with self.assertRaisesRegex(ValueError,'archive/resource/fold lineage'):
                    self.run_fixture(root,report,catalog,design)

    def test_changed_design_receipt_and_foreign_plan_are_rejected(self):
        for kind in ('design','plan'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);report,catalog,design=full_fixture(root)
                if kind=='design':
                    Path(root/'design.json').write_text('{"changed":true}')
                    # Remove the redundant source reference to exercise direct parent design binding.
                    report['sources']=[r for r in report['sources'] if r['path']!=str(root/'design.json')]
                else:
                    row=report['output_cases'][0];prediction=m.checked(row['prediction'])
                    prediction['target_plan']=report['output_cases'][-1]['prediction']
                    row['prediction']=write(root/'foreign-plan-prediction.json',prediction)
                with self.assertRaises(ValueError):self.run_fixture(root,report,catalog,design)

    def test_foreign_canonical_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, catalog, design = full_fixture(root)
            row = report['output_cases'][0]
            measured = m.checked(row['measured_label'])
            evidence = measured['measured_resources']['source_evidence']
            review = m.checked(evidence['training_review'])
            review['inputs']['plan.json'] = write(root/'unrelated-plan.json', dict(attempt_id='unrelated'))
            evidence['training_review'] = write(root/'foreign-review.json', review)
            row['measured_label'] = write(root/'foreign-measured.json', measured)
            with self.assertRaisesRegex(ValueError, 'exact persisted prediction'):
                self.run_fixture(root, report, catalog, design)

    def test_duplicate_case_and_incomplete_counts_are_rejected(self):
        for kind in ('duplicate', 'incomplete'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report, catalog, design = full_fixture(root)
                if kind == 'duplicate':
                    report['output_cases'][-1] = copy.deepcopy(report['output_cases'][0])
                else:
                    report['output_cases'].pop()
                with self.assertRaises(ValueError):
                    self.run_fixture(root, report, catalog, design)


if __name__ == '__main__':
    unittest.main()
