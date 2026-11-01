"""Independent CPU native-flow adapter cases; none is physical GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.known_input_gate_v5 import DATA_SCHEMA, freeze_gate_protocol, seal_predictions, evaluate_prediction_gate
from tools.predictor_validation_20261027.known_input_observed_v5 import (
    build_observed_run, verify_before_collection, receipt,
)

NS = 1_000_000_000


class KnownInputObservedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(schema='gsdeploy.native_known_input_flow_plan.v5',validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS', aliases={'o': {'family': 'ocr', 'supported_tasks': ['ocr']}},
            trace=[dict(request_id='r1', pool_id='heldout-1', task='ocr', offset_ns=0),
                   dict(request_id='r2', pool_id='heldout-2', task='ocr', offset_ns=50_000_000)],
            candidates=[dict(id=cid, execution_context_sha256='exec', routes={'r1': 'o', 'r2': 'o'}) for cid in ('base', 'alt')],
            baseline_candidate_id='base', window_seconds=1., drain_seconds=1., memory_budget_mib=1000.,
            memory_safe_factor=1.1, energy_reference_watts=100., startup_amortization_seconds=10.,
            workload_rps={'ocr': 2.}, latency_reference_ms={'ocr': 1000.}, weights=dict.fromkeys('QTLPME', 1/6),
            queue_sample_offsets_ns=[0, 75_000_000, 150_000_000, NS], queue_depth_boundary='waiting_until_handler_start')
        from gsdeploy.resources import identity
        self.plan['plan_sha256']=identity(self.plan)
        self.plan_path = self.write('plan.json', self.plan)
        self.training = self.write('training-index.json', dict(schema=DATA_SCHEMA, split='training', records=[
            dict(run_id='train', request_id='t', pool_id='heldout-1', task='ocr'),dict(run_id='train', request_id='t2', pool_id='heldout-2', task='ocr')]))
        self.catalog=self.root/'public.jsonl'
        self.catalog.write_text('\n'.join(json.dumps(dict(pool_id=pool,task_type='ocr',sample=dict(input_sha256=pool))) for pool in ('heldout-1','heldout-2')))
        self.model = self.write('model.json', {'model': 'CPU fixture'})
        self.quality_source = self.write('quality-reference.json', {'reference': 'CPU fixture'})
        domain = [dict(run_id='validation', requests=[{k: r[k] for k in ('request_id', 'pool_id', 'task')} for r in self.plan['trace']])]
        self.protocol = freeze_gate_protocol(model_paths={'model': self.model}, source_paths=[self.plan_path,self.catalog],public_input_catalog_path=self.catalog,
            training_data_paths=[self.training], candidates=[dict(id=cid, model_id='model', execution_context_sha256='exec') for cid in ('base', 'alt')],
            baseline_candidate_id='base', validation_domain=domain, thresholds={'queue_depth_mae_max': 0., 'delta_sign_accuracy_min': 1.},
            created_unix_ns=100, weights=self.plan['weights'], queue_sample_offsets_ns=self.plan['queue_sample_offsets_ns'])
        self.protocol_path = self.write('protocol.json', self.protocol)
        cells = [dict(run_id='validation', candidate_id=cid, execution_context_sha256='exec',
            requests=[dict(request_id='r1', pool_id='heldout-1', task='ocr', service_ms=100., queue_ms=0., e2e_ms=110.),
                      dict(request_id='r2', pool_id='heldout-2', task='ocr', service_ms=100., queue_ms=50., e2e_ms=160.)],
            terms=dict(Q=.1, T=0., L=.135, P=.1575, M=.88, E=1.1),
            queue_depth_samples=[dict(offset_ns=o, depth=d) for o, d in zip(self.plan['queue_sample_offsets_ns'], [0, 1, 0, 0])]) for cid in ('base', 'alt')]
        self.predictions = seal_predictions(self.protocol, cells, created_unix_ns=200)
        self.predictions_path = self.write('predictions.json', self.predictions)

    def write(self, name, document):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        return path

    def raw(self, cid):
        output = self.root/('collection-'+cid)
        pre = verify_before_collection(self.protocol_path, self.predictions_path, self.plan_path,
            run_id='validation', candidate_id=cid, output_root=output)
        records = []
        for index, offered in enumerate(self.plan['trace']):
            start = 2*NS+index*100_000_000
            end, received = start+100_000_000, start+110_000_000
            records.append(dict(request_id=offered['request_id'], pool_id=offered['pool_id'], task='ocr', model='o',
                scheduled_time_ns=2*NS+offered['offset_ns'], status='completed', observed_worker_start_time_ns=start,
                execution_events=[dict(request_id=offered['request_id'], pid=123, perf_ns=start, received_time_ns=start+1)],
                worker_response=dict(start_perf_ns=start, end_perf_ns=end), completion_time_ns=received,
                response_reader_event_enqueue_time_ns=received))
        samples = []
        for index in range(13):
            stamp = NS+index*250_000_000
            memory = 600 if index == 0 else 800 if index == 10 else 500
            samples.append(dict(time_ns=stamp, watts=100, processes_mib={'123': memory}, owned_used_mib=memory,
                owned_aliases=['o'], foreign_pids=[]))
        raw = dict(schema='gsdeploy.native_flow_collection.v1',validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS', split='validation', kind='cpu_fixture', backend='cpu_fixture',
            candidate_id=cid, execution_context_sha256='exec', plan=receipt(self.plan_path),
            precollection=receipt(pre['receipt_path']), collection_started_unix_ns=pre['checked_unix_ns']+1,
            phase_start_ns=2*NS, deadline_ns=3*NS, observation_end_ns=4*NS, startup_begin_ns=NS, startup_end_ns=NS+500_000_000,
            effective_objective={k: self.plan[k] for k in ('workload_rps', 'latency_reference_ms', 'weights', 'memory_safe_factor',
                'memory_budget_mib', 'energy_reference_watts', 'startup_amortization_seconds')},
            resident_aliases=['o'], ready={'o': {'pid': 123}}, request_records=records, board_samples=samples,
            status='completed', collection_complete=True, errors=[], runtime_errors=[], telemetry_errors=[], qualification_issues=[],
            rho_admission_passed=True, stability={'stability_state_v2': 'steady'},
            quality_evidence=dict(kind='measured_per_request', source_receipts=[receipt(self.quality_source)], errors=[],
                totals_by_request={'r1': dict(task='ocr', numerator=1, denominator=1), 'r2': dict(task='ocr', numerator=0, denominator=9)}))
        path = self.write(str(output.relative_to(self.root)/'raw.json'), raw)
        return dict(candidate_id=cid, raw_path=path, precollection_path=pre['receipt_path']), raw

    def adapt(self, collections, name='adapter'):
        return build_observed_run(self.protocol_path, self.predictions_path, self.plan_path, run_id='validation',
            collections=collections, output_dir=self.root/name, allow_cpu_fixture=True)

    def replace_raw(self, collection, raw):
        Path(collection['raw_path']).write_text(json.dumps(raw))

    def observed(self, report):
        return json.loads(Path(report['observed']['path']).read_text())

    def test_measured_arithmetic_uses_parent_receipt_ratio_totals_and_startup_drain_peak_once(self):
        collections = [self.raw(cid)[0] for cid in ('base', 'alt')]
        report = self.adapt(collections)
        self.assertEqual(report['status'], 'COMPLETE')
        cell = self.observed(report)['cells'][0]
        self.assertEqual(cell['requests'][0]['service_ms'], 100.)
        self.assertEqual(cell['requests'][0]['e2e_ms'], 110.)
        self.assertEqual(cell['requests'][1]['queue_ms'], 50.)
        self.assertAlmostEqual(cell['terms']['Q'], .1)  # not mean CER=.5
        self.assertAlmostEqual(cell['terms']['M'], .88)  # drain 800MiB, factor once
        self.assertAlmostEqual(cell['terms']['E'], 1.1)  # no drain energy
        gate = evaluate_prediction_gate(self.protocol, self.predictions, [report['observed']['path']])
        self.assertEqual(gate['status'], 'PASS')
        self.assertEqual(gate['evidence_kind'], 'cpu_fixture')
        self.assertFalse(gate['physical_gain_certified'])

    def test_missing_start_is_unknown_not_receipt_or_inferred_service(self):
        collection, raw = self.raw('base')
        raw['request_records'][0]['execution_events'] = []
        self.replace_raw(collection, raw)
        report = self.adapt([collection])
        row = self.observed(report)['cells'][0]['requests'][0]
        self.assertIsNone(row['service_ms'])
        self.assertIsNone(row['queue_ms'])
        self.assertEqual(row['e2e_ms'], 110.)
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_missing_receipt_retains_cohort_and_makes_full_latency_unknown(self):
        collection, raw = self.raw('base')
        raw['request_records'][1].pop('response_reader_event_enqueue_time_ns')
        self.replace_raw(collection, raw)
        report = self.adapt([collection])
        cell = self.observed(report)['cells'][0]
        self.assertEqual(len(cell['requests']), 2)
        self.assertIsNone(cell['requests'][1]['e2e_ms'])
        self.assertTrue(all(cell['terms'][k] is None for k in ('T', 'L', 'P')))
        self.assertEqual(cell['queue_depth_samples'][-1]['depth'], 0)  # waiting ends at observed START, not receipt

    def test_missing_chronology_emits_no_gate_file(self):
        collection, raw = self.raw('base')
        raw.pop('collection_started_unix_ns')
        self.replace_raw(collection, raw)
        report = self.adapt([collection])
        self.assertFalse(report['observed_emitted'])
        self.assertNotIn('observed', report)
        self.assertFalse((self.root/'adapter/observed-run.json').exists())

    def test_smoke_training_and_profiler_flow_refused(self):
        collection, raw = self.raw('base')
        for index, (key, value) in enumerate([('split', 'smoke'), ('split', 'training'),
                ('schema', 'gsdeploy.native_phase_collection.v1'), ('performance_scope', 'profiler_perturbed')]):
            altered = {**raw, key: value}
            self.replace_raw(collection, altered)
            with self.assertRaises(ValueError):
                self.adapt([collection], 'attempt'+str(index))

    def test_changed_pid_endpoints_routes_and_objective_refused(self):
        collection, raw = self.raw('base')
        variants = []
        changed = copy.deepcopy(raw); changed['request_records'][0]['execution_events'][0]['pid'] = 999; variants.append(changed)
        changed = copy.deepcopy(raw); changed['request_records'][0]['completion_time_ns'] += 1; variants.append(changed)
        changed = copy.deepcopy(raw); changed['request_records'][0]['worker_response']['end_perf_ns'] = 4*NS; variants.append(changed)
        changed = copy.deepcopy(raw); changed['request_records'][0]['model'] = 'foreign'; variants.append(changed)
        changed = copy.deepcopy(raw); changed['effective_objective']['memory_safe_factor'] = 2; variants.append(changed)
        for index, changed in enumerate(variants):
            self.replace_raw(collection, changed)
            with self.assertRaises(ValueError):
                self.adapt([collection], 'invalid'+str(index))

    def test_missing_memory_or_power_does_not_invent_other_resource(self):
        collection, raw = self.raw('base')
        changed = copy.deepcopy(raw)
        for sample in changed['board_samples']:
            sample.pop('watts')
        self.replace_raw(collection, changed)
        cell = self.observed(self.adapt([collection], 'power'))['cells'][0]
        self.assertIsNone(cell['terms']['E'])
        self.assertAlmostEqual(cell['terms']['M'], .88)
        changed = copy.deepcopy(raw)
        for sample in changed['board_samples']:
            sample.pop('processes_mib')
        self.replace_raw(collection, changed)
        cell = self.observed(self.adapt([collection], 'memory'))['cells'][0]
        self.assertIsNone(cell['terms']['M'])
        self.assertAlmostEqual(cell['terms']['E'], 1.1)

    def test_missing_quality_and_mutated_source_are_distinct(self):
        collection, raw = self.raw('base')
        raw.pop('quality_evidence')
        self.replace_raw(collection, raw)
        self.assertIsNone(self.observed(self.adapt([collection]))['cells'][0]['terms']['Q'])
        self.quality_source.write_text('{}')
        collection2, raw2 = self.raw('alt')
        raw2['quality_evidence']['source_receipts'][0]['sha256'] = 'bad'
        self.replace_raw(collection2, raw2)
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.adapt([collection2], 'changed-quality')

    def test_precheck_refuses_existing_collection_or_mutated_model(self):
        collection, raw = self.raw('base')
        with self.assertRaises(FileExistsError):
            verify_before_collection(self.protocol_path, self.predictions_path, self.plan_path,
                run_id='validation', candidate_id='base', output_root=Path(collection['raw_path']).parent)
        self.model.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity'):
            verify_before_collection(self.protocol_path, self.predictions_path, self.plan_path,
                run_id='validation', candidate_id='alt', output_root=self.root/'fresh')

    def test_backdated_actual_collection_and_runtime_failure_cannot_pass_gate(self):
        collection, raw = self.raw('base')
        raw['collection_started_unix_ns'] = 0
        self.replace_raw(collection, raw)
        with self.assertRaisesRegex(ValueError, 'before'):
            self.adapt([collection])
        raw['collection_started_unix_ns'] = json.loads(Path(collection['precollection_path']).read_text())['checked_unix_ns']+1
        raw['runtime_errors'] = ['collector failure']
        self.replace_raw(collection, raw)
        cell = self.observed(self.adapt([collection]))['cells'][0]
        self.assertTrue(all(v is None for v in cell['terms'].values()))

    def test_waiting_grid_does_not_count_active_service_or_response_tail(self):
        collection, raw = self.raw('base')
        cell = self.observed(self.adapt([collection]))['cells'][0]
        self.assertEqual([p['depth'] for p in cell['queue_depth_samples']], [0, 1, 0, 0])
        raw['request_records'][0]['execution_events'] = []
        self.replace_raw(collection, raw)
        unknown = self.observed(self.adapt([collection], 'missing-start-grid'))['cells'][0]
        self.assertTrue(all(p['depth'] is None for p in unknown['queue_depth_samples']))

    def test_telemetry_boundary_gaps_stay_unknown_without_extrapolation(self):
        collection, raw = self.raw('base')
        raw['board_samples'] = raw['board_samples'][1:-1]
        self.replace_raw(collection, raw)
        cell = self.observed(self.adapt([collection]))['cells'][0]
        self.assertIsNone(cell['terms']['M'])
        self.assertIsNone(cell['terms']['E'])

    def test_nvml_foreign_process_invalid_sum_and_missing_cohort(self):
        collection, raw = self.raw('base')
        changed = copy.deepcopy(raw)
        changed['board_samples'][3]['processes_mib']['999'] = 10
        changed['board_samples'][3]['foreign_pids'] = [999]
        self.replace_raw(collection, changed)
        cell = self.observed(self.adapt([collection]))['cells'][0]
        self.assertIsNone(cell['terms']['M'])
        self.assertIsNone(cell['terms']['E'])
        changed = copy.deepcopy(raw)
        changed['board_samples'][3]['owned_used_mib'] += 10
        self.replace_raw(collection, changed)
        with self.assertRaisesRegex(ValueError, 'Owned memory sum'):
            self.adapt([collection], 'invalid-sum')
        raw['request_records'].pop()
        self.replace_raw(collection, raw)
        with self.assertRaisesRegex(ValueError, 'offered request'):
            self.adapt([collection], 'missing-cohort')

    def test_cpu_opt_in_and_annotated_native_raw_preservation(self):
        collection, raw = self.raw('base')
        with self.assertRaisesRegex(ValueError, 'CPU fixtures'):
            build_observed_run(self.protocol_path, self.predictions_path, self.plan_path, run_id='validation',
                collections=[collection], output_dir=self.root/'no-fixture-permission')
        native = {k: v for k, v in raw.items() if k != 'quality_evidence'}
        native_path = Path(collection['raw_path'])
        native_path.write_text(json.dumps(native))
        scored = {**raw, 'native_raw': receipt(native_path)}
        scored_path = self.write('collection-base/scored-raw.json', scored)
        collection['raw_path'] = scored_path
        self.assertTrue(self.adapt([collection])['observed_emitted'])
        scored['request_records'][0]['completion_time_ns'] += 1
        scored_path.write_text(json.dumps(scored))
        with self.assertRaisesRegex(ValueError, 'Annotated'):
            self.adapt([collection], 'rewritten-native')


if __name__ == '__main__':
    unittest.main()
