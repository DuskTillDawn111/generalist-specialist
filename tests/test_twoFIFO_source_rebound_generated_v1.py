import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_training_intake_v4 as fixture_intake
from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_generated_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_joint_law_v1 as law
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_phase12_adapter_v1 as adapter


def new_documents(plans):
    docs = fixture_intake.documents(plans)
    for doc in docs:
        doc.update(schema=adapter.MACRO_SCHEMA, fitted_model_qualified=False,
                   source_rebound_manifest={'path': 'cpu-fixture-manifest', 'sha256': 'fixture'})
        for request in doc['requests']:
            request['pool_id'] = request['macros'][0]['pool_id']
    return docs


class ReboundGeneratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_intake.IntakeTests.setUpClass()
        cls.plans = fixture_intake.IntakeTests.plans

    def test_predeclared_score_and_draw_contract_and_closure(self):
        prior = json.loads(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit_manifest = root / 'fit-manifest.json'
            fit_manifest.write_text('{}')
            with patch.object(m.response, 'audit', return_value={'sources': []}):
                ref = m.prepare(fit_manifest, prior['scoring_plan']['path'], prior['quality']['path'], root / 'prepared')
                doc = m.audit(ref['path'])
                self.assertEqual(doc['scheme'], m.generated.SCHEME)
                self.assertFalse(doc['queue_bounds_used_for_utility'])
                for field, value in (('expected_measured_requests', 545), ('queue_bounds_used_for_utility', True),
                                     ('scheme_sha256', 'changed'), ('six_term_weights', 'partial')):
                    bad = copy.deepcopy(doc)
                    bad[field] = value
                    path = root / (field + '.json')
                    path.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        m.audit(path)
                (root / 'phase12-fit-claim.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'before the one permitted fit'):
                    m.prepare(fit_manifest, prior['scoring_plan']['path'], prior['quality']['path'], root / 'late')
                self.assertFalse((root / 'late').exists())

    def test_new_source_schema_and_dynamic_macro_denominator(self):
        docs = new_documents(self.plans)
        # Two macros per offered request demonstrate that 546 is not a row cap.
        for doc in docs:
            for request in doc['requests']:
                first = request['macros'][0]
                second = copy.deepcopy(first)
                duration = first['response_wall_ns']
                second.update(macro_id='tail', left_ns=first['right_ns'], right_ns=first['right_ns'] + duration)
                second['causal_covariate_path'][0].update(left_ns=second['left_ns'], right_ns=second['right_ns'])
                second['causal_covariate_path'][0]['state']['cutoff_processing_ns'] = second['left_ns']
                request['macros'].append(second)
                request['worker_start_to_ready_ns'] += duration
            doc['macros'] *= 2
        raws = [dict(capture_id=p['capture_id'], attempt_id=p['attempt_id'], execution_id=p['execution_id'],
            backend='native', status='completed', priming_request_records=[{} for _ in range(18)],
            request_records=[{'request_id': r['request_id']} for r in d['requests']])
            for p, d in zip(self.plans, docs)]
        denominator = law.complete_documents(docs, self.plans, raws)
        self.assertEqual(denominator['measured_requests'], 546)
        self.assertEqual(sum(d['macros'] for d in docs), 1092)
        bad = copy.deepcopy(docs)
        bad[0]['schema'] = adapter.frozen.MACRO_SCHEMA
        with self.assertRaises(ValueError):
            law.complete_documents(bad, self.plans, raws)
        bad = copy.deepcopy(docs)
        bad[0]['requests'][0]['macros'][1]['left_ns'] += 1
        with self.assertRaises(ValueError):
            law.complete_documents(bad, self.plans, raws)

    def test_unqualified_fit_evaluates_all_150_cases_and_546_requests(self):
        docs = new_documents(self.plans)
        rows, denominator = adapter._rows(docs, self.plans)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = []
            for index, (plan, doc) in enumerate(zip(self.plans, docs)):
                plan_path = root / ('plan-' + str(index) + '.json')
                plan_path.write_text(json.dumps(plan))
                cases, requests = [], []
                for ordinal, case in enumerate(plan['cases']):
                    epoch = 10**12 + ordinal * 100_000_000_000
                    cases.append(dict(case_id=case['case_id'], observation_start_ns=epoch))
                    requests.extend(dict(request_id=a['request_id'], status='completed',
                        completion_time_ns=epoch+a['arrival_ns']+1000) for a in case['arrivals'])
                raw_path = root / ('raw-' + str(index) + '.json')
                raw_path.write_text(json.dumps(dict(cases=cases, case_histories=[], request_records=requests)))
                review_path = root / ('review-' + str(index) + '.json')
                review_path.write_text(json.dumps(dict(inputs={'plan.json': m.receipt(plan_path)}, raw=m.receipt(raw_path))))
                reviews.append(m.receipt(review_path))
            bundle = dict(schema=adapter.SCHEMA, design=adapter.DESIGN, features=list(adapter.FEATURES),
                denominator=denominator, plans=self.plans, documents=docs, full={'rows': rows},
                review_receipts=reviews, numerical_fit_run=False, validation_labels_read=False, GPU_started=False)
            fitted = dict(denominator=denominator, features=list(adapter.FEATURES),
                          response_law_compilation_eligible=False)
            output = root / 'evaluation'
            original_labels = m.generated.native_labels
            calls = []

            def labels_after_save(raw, case, epoch, score, quality):
                # Real case entry must persist prediction AND bounds first.
                predictions = list(output.glob('*/case-*/prediction.json'))
                bounds = list(output.glob('*/case-*/queue-envelope.json'))
                self.assertEqual(len(predictions), len(calls)+1)
                self.assertEqual(len(bounds), len(calls)+1)
                calls.append(case['case_id'])
                return original_labels(raw, case, epoch, score, quality)

            score_path = Path('artifacts/predictor-twoFIFO-generated-score-design-v1-20261028-cpu01/scoring-plan.json')
            design = json.loads(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json').read_text())
            score, quality = json.loads(score_path.read_text()), m.checked(design['quality'])
            with patch.object(m.generated, 'native_labels', side_effect=labels_after_save), \
                 patch.object(m.laws, 'build_joint_law', side_effect=AssertionError('unqualified law must not compile')):
                result = m.evaluate_bundle(bundle, fitted, score, quality, output)
            self.assertEqual(result['forecasts_generated'], 150)
            self.assertEqual(result['summary']['offered_requests'], 546)
            self.assertEqual(result['ensemble_worlds_unknown'], 1200)
            self.assertEqual(result['ensemble_worlds_generated'], 0)
            self.assertEqual(result['summary']['point_supported_requests'], 0)
            self.assertEqual(len(calls), 150)
            self.assertFalse(result['numerical_fit_run'])
            self.assertFalse(result['independent_prediction_accuracy_validated'])
            for fold in result['whole_block_leaveout']:
                self.assertEqual(len(fold['donor_attempt_ids']), 8)
                self.assertEqual(len(fold['heldout_attempt_ids']), 4)
                self.assertNotIn(fold['heldout_block_id'], fold['training_block_ids'])

    def test_compiler_preserves_protocol_and_excludes_heldout_donors(self):
        plan, raw, requests, fitted, _ = source_fixture()
        plans, raws, docs = [], [], []
        for block in ('training-block-1', 'training-block-2', 'training-block-3'):
            for shard in range(4):
                identity = dict(capture_id=block, attempt_id=block+'-shard-'+str(shard), execution_id=block+str(shard))
                p, r = copy.deepcopy(plan), copy.deepcopy(raw)
                p.update(identity)
                r.update(identity)
                r['priming_request_records'][0]['model'] = 'text#1'
                docs.append(dict(requests=copy.deepcopy(requests), source_rebound_manifest={'path': 'fixture', 'sha256': 'fixture'}, **identity))
                plans.append(p)
                raws.append(r)
        blocks = sorted({p['capture_id'] for p in plans})
        denominator = dict(measured_requests=546)
        root = dict(fitted, numerical_qualification=True, whole_blocks=blocks)
        qualified = dict(response_law_compilation_eligible=True, features=list(adapter.FEATURES),
            fitted_root_numerical_qualification=True, denominator=denominator, full_fit=root,
            whole_block_leaveout=[dict(heldout_block_id=b,
                training_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id'] != b],
                heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id'] == b],
                training_fit=dict(root, whole_blocks=[x for x in blocks if x != b]),
                evaluation=dict(summary=dict(unknown_rows=0))) for b in blocks])
        # Exact12 gate itself is tested above on all 546 inputs. This fixture
        # exercises the real vector/protocol compiler with complete source marks.
        with patch.object(law, 'complete_documents', return_value=denominator):
            result = law.build_joint_law(docs, plans, raws, qualified, heldout_block_id=blocks[2])
            self.assertEqual(len(result['vectors']), 24)
            self.assertEqual(result['donor_startup_attempts'], 8)
            self.assertTrue(all(v['training_block_id'] != blocks[2] for v in result['vectors']))
            vector = result['vectors'][0]
            compiled = law.joint.compile_draw_v4(vector, 0)
            self.assertEqual(len(compiled['plan'].macros), len(vector['nodes']))
            bad = copy.deepcopy(qualified)
            bad['whole_block_leaveout'][0]['training_attempt_ids'][0] = 'foreign-attempt'
            with self.assertRaises(ValueError):
                law.build_joint_law(docs, plans, raws, bad)


if __name__ == '__main__':
    unittest.main()
