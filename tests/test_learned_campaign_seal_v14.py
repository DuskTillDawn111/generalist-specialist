"""Small CPU fakes exercise actual v5 freeze/seal/precollection contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import learned_campaign_seal_v14 as seal
from tools.predictor_validation_20261027.complete_training_index_v12 import records_for_capture
from tools.predictor_validation_20261027.known_input_campaign_seal_v5 import public_inputs
from tools.predictor_validation_20261027.known_input_gate_v5 import DATA_SCHEMA
from tools.predictor_validation_20261027.known_input_observed_v5 import verify_before_collection
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.profiling import digest


class LearnedCampaignSealTests(unittest.TestCase):
    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return path

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.out = self.root/'sealed'; self.calls = []
        self.source = self.root/'source.py'; self.source.write_text('# unchanged fake source\n')
        catalog = self.root/'runtime-inputs.jsonl'
        catalog.write_text(json.dumps(dict(pool_id='input', task_type='ocr', sample=dict(input_sha256='content')))+'\n')
        context = self.write('context.json', dict(execution_sources=[dict(original_path=str(catalog), frozen=receipt(catalog))]))
        aliases = {'ocr#0':dict(family='ocr', supported_tasks=['ocr'])}
        defaults = dict(aliases=aliases, latency_reference_ms={'ocr':1000.}, memory_budget_mib=1000.,
                        energy_reference_watts=100., memory_safe_factor=1.1)
        trace = [dict(request_id='fresh:'+str(i), pool_id='input', task='ocr', offset_ns=i*10) for i in range(57)]
        candidates = [dict(id=cid, execution_context_sha256='ctx', routes={r['request_id']:'ocr#0' for r in trace})
                      for cid in sorted(seal.CANDIDATES)]
        original = dict(schema='gsdeploy.native_known_input_flow_plan.v5',
            validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
            **defaults, trace=trace, candidates=candidates, baseline_candidate_id='shared_balanced',
            window_seconds=1., drain_seconds=1., warmup_seconds=0., startup_amortization_seconds=3600.,
            workload_rps={'ocr':1.}, weights=dict.fromkeys('QTLPME', 1/6),
            queue_sample_offsets_ns=[0, 500, 1000], queue_depth_boundary='waiting_until_handler_start',
            validation_thresholds=dict(service_mare_max=.2, e2e_mare_max=.2, p95_mare_max=.2,
                queue_depth_mae_max=1., delta_sign_accuracy_min=1., pair_rank_accuracy_min=1.))
        original['plan_sha256'] = digest(original)
        original_path = self.write('original-plan.json', original)
        self.plan = copy.deepcopy(original); self.plan.pop('plan_sha256')
        self.plan.update(run_id='fresh-run', split='validation', context=receipt(context),
                         public_contract_source=receipt(original_path))
        self.plan['plan_sha256'] = digest(self.plan); self.plan_path = self.write('prepared/plan.json', self.plan)
        for candidate in candidates:
            public, routes = public_inputs(self.plan, candidate)
            self.write('prepared/public-'+candidate['id']+'.json', public)
            self.write('prepared/routes-'+candidate['id']+'.json', routes)
        startup = self.write('startup-model.json', dict(training_only=True))
        bundle = self.write('component-bundle.json', dict(forecast_defaults=defaults, startup=dict(training_model=receipt(startup))))
        self.config = dict(schema=seal.MODEL_SCHEMA, base_model_configuration=dict(component_bundle=receipt(bundle)))
        self.config_path = self.write('configuration.json', self.config)
        records, captures = [], []
        for i in range(2):
            training_context = self.write('training/capture'+str(i)+'/context.json', dict(training_only=True))
            training_plan = self.write('training/capture'+str(i)+'/plan.json', dict(split='training', training_only=True, execution_context_sha256='training'))
            raw_doc = dict(kind='measured', status='completed', execution_context_sha256='training',
                request_records=[dict(request_id='training:'+str(i), pool_id='input', task='ocr', split='training', status='completed')],
                priming_request_records=[dict(request_id='priming:'+str(i), pool_id='input', task='ocr', phase='priming', status='completed')])
            raw = self.write('training/capture'+str(i)+'/raw.json', raw_doc)
            review = self.write('training/capture'+str(i)+'/review.json', dict(schema='gsdeploy.training_collection_strict_review.v3',
                training_only=True, status='PASS', issues=[], evidence=dict(raw=receipt(raw), plan=receipt(training_plan), context=receipt(training_context))))
            r, measured, priming = records_for_capture(raw_doc, json.loads(training_plan.read_text()), receipt(raw)); records.extend(r)
            captures.append(dict(run_id=r[0]['run_id'], raw=receipt(raw), plan=receipt(training_plan), context=receipt(training_context),
                                 reviews=[receipt(review)], measured_requests=measured, priming_requests=priming))
        self.raw_refs = [c['raw'] for c in captures]
        self.index_path = self.write('index/training-index.json', dict(schema=DATA_SCHEMA, split='training', records=records))
        manifest = self.write('index/source-manifest.json', dict(captures=captures))
        lineage = self.write('index/lineage-audit.json', dict(unresolved_raw_references=[]))
        disjoint = self.write('index/public-disjointness.json', dict(validation_raw_or_labels_read=False))
        self.write('index/report.json', dict(schema='gsdeploy.complete_training_index_audit.v12',
            status='INDEX_BUILT_PUBLIC_DISJOINT_WITH_EXPLICIT_LINEAGE_BOUNDARIES', unresolved_raw_lineage_count=0,
            validation_raw_or_labels_read=False, records=len(records), independent_original_raw_runs=2,
            artifacts={p.name:receipt(p) for p in [self.index_path, manifest, lineage, disjoint]}))
        summaries = []
        for c in candidates:
            result = self.result(c['id'], shift=10, term=.75)
            path = self.write('preflight/forecast-'+c['id']+'.json', result)
            summaries.append(dict(candidate_id=c['id'], status=seal.COMPLETE, full_declared_requests=57, result=receipt(path)))
        before = [receipt(self.plan_path), receipt(self.config_path), receipt(self.source)]
        self.write('preflight/source-before.json', before); self.write('preflight/source-after.json', before)
        self.preflight_path = self.write('preflight/report.json', dict(status='FULL_PUBLIC_PREFLIGHT_EXECUTED',
            plan=receipt(self.plan_path), configuration=receipt(self.config_path), core_and_public_sources_unchanged=True,
            training_sources_guarded=True, full_four_candidate_forecast_executed=True, validation_outcomes_read=False,
            GPU_started=False, completeness=dict(status='COMPLETE_PRESEALED_PREDICTIONS'), candidates=summaries))
        self.incomplete = False; self.mutate = False

    def result(self, cid, *, shift, term):
        candidate = next(c for c in self.plan['candidates'] if c['id'] == cid)
        rows = [dict(request_id=r['request_id'], task=r['task'], pool_id=r['pool_id'],
                     arrival_ns=r['offset_ns'], alias=candidate['routes'][r['request_id']]) for r in self.plan['trace']]
        return dict(status=seal.COMPLETE, candidate_id=cid, plan=receipt(self.plan_path), configuration=self.config,
            full_public_cohort=rows, forecast=dict(cohort={r['request_id']:r for r in rows},
                finishes={r['request_id']:dict(r, start_ns=r['arrival_ns']+shift,
                    handler_end_ns=r['arrival_ns']+shift+20, finish_ns=r['arrival_ns']+shift+30) for r in rows}),
            metrics=dict(startup_accounting='included_exactly_once', terms=dict.fromkeys('QTLPME', term)))

    def run_sealer(self):
        case = self
        class FakeProvider:
            def __init__(self, configuration, **kwargs):
                case.assertTrue((case.out/'protocol.json').exists())
                case.calls.append('construct_after_freeze')
                self.availability_training_audit = [dict(raw=r) for r in case.raw_refs]
                self.bundle = json.loads(Path(configuration['base_model_configuration']['component_bundle']['path']).read_text())
            def execute(self, plan_ref, cid, public, routes, startup, *, max_events):
                case.assertEqual(max_events, 5_000_000); case.calls.append(cid)
                result = case.result(cid, shift=50, term=.25)
                if case.incomplete:
                    result['status'] = 'UNKNOWN'; result['forecast']['finishes'].pop(next(iter(result['forecast']['finishes'])))
                    result['metrics']['terms']['E'] = None
                if case.mutate:
                    case.source.write_text('# changed during prediction\n')
                return result
        with patch.object(seal, 'LearnedAvailabilityForecast', FakeProvider), patch.object(seal, 'prediction_source_paths', return_value=[self.source]):
            return seal.prepare_and_seal(self.plan_path, self.index_path, self.config_path,
                                         self.preflight_path, self.out)

    def test_freeze_then_fresh_all_candidates_seal_and_v11_precollection_compatibility(self):
        report = self.run_sealer()
        self.assertEqual(self.calls, ['construct_after_freeze']+[c['id'] for c in self.plan['candidates']])
        protocol = json.loads(Path(report['protocol']['path']).read_text())
        prediction = json.loads(Path(report['predictions']['path']).read_text())
        self.assertEqual(protocol['thresholds'], self.plan['validation_thresholds'])
        self.assertEqual(protocol['weights'], self.plan['weights'])
        self.assertEqual(list(protocol['validation_domain']), ['fresh-run'])
        self.assertEqual([len(c['requests']) for c in prediction['cells']], [57]*4)
        self.assertTrue(all(c['terms']['E'] == .25 and c['requests'][0]['queue_ms'] == 50/1e6 for c in prediction['cells']))
        for source in seal.collection_source_paths():
            self.assertIn(receipt(source), protocol['sources'])
        check = verify_before_collection(report['protocol']['path'], report['predictions']['path'], self.plan_path,
            run_id='fresh-run', candidate_id='shared_balanced', output_root=self.root/'future-collection')
        self.assertFalse(check['GPU_started'])
        self.assertFalse((self.root/'future-collection').exists())
        with self.assertRaises(FileExistsError):
            self.run_sealer()

    def test_incomplete_numeric_preflight_prevents_even_protocol_freeze(self):
        data = json.loads(self.preflight_path.read_text()); data['completeness']['status'] = 'INCOMPLETE'
        self.preflight_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'numeric'):
            self.run_sealer()
        self.assertFalse(self.out.exists()); self.assertEqual(self.calls, [])

    def test_unknown_fresh_predictions_cannot_borrow_preflight_cells(self):
        self.incomplete = True
        with self.assertRaisesRegex(ValueError, 'Incomplete predictions'):
            self.run_sealer()
        self.assertEqual(len(self.calls), 5)
        self.assertFalse((self.out/'predictions.json').exists())
        self.assertTrue((self.out/'failed-production.json').exists())

    def test_source_mutation_during_fresh_prediction_prohibits_seal(self):
        self.mutate = True
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_sealer()
        self.assertFalse((self.out/'predictions.json').exists())
        self.assertTrue((self.out/'source-after.json').exists())

    def test_missing_priming_record_in_claimed_complete_index_is_rejected(self):
        index = json.loads(self.index_path.read_text()); index['records'].pop(); self.index_path.write_text(json.dumps(index))
        report_path = self.index_path.with_name('report.json'); report = json.loads(report_path.read_text())
        report['artifacts']['training-index.json'] = receipt(self.index_path); report['records'] -= 1
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'measured and priming'):
            self.run_sealer()
        self.assertFalse(self.out.exists())

    def test_original_threshold_change_cannot_be_hidden_by_rehashing_plan(self):
        plan = copy.deepcopy(self.plan); plan['validation_thresholds']['service_mare_max'] = .9
        plan['plan_sha256'] = digest({k:v for k,v in plan.items() if k != 'plan_sha256'})
        self.plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'contract changed'):
            self.run_sealer()
        self.assertFalse(self.out.exists())

    def test_training_audit_historical_gate_metadata_is_bound_without_becoming_model_input(self):
        manifest_path = self.index_path.with_name('source-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        manifest['configuration'] = dict(old_protocol=dict(path=str(self.root/'unused-old-gate.json'), sha256='old'))
        manifest_path.write_text(json.dumps(manifest))
        report_path = self.index_path.with_name('report.json'); report = json.loads(report_path.read_text())
        report['artifacts']['source-manifest.json'] = receipt(manifest_path); report_path.write_text(json.dumps(report))
        result = self.run_sealer()
        protocol = json.loads(Path(result['protocol']['path']).read_text())
        self.assertIn(receipt(manifest_path), protocol['sources'])
        nested_startup = json.loads((self.root/'component-bundle.json').read_text())['startup']['training_model']
        self.assertIn(nested_startup, protocol['sources'])


if __name__ == '__main__':
    unittest.main()
