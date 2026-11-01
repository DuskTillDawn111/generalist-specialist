"""CPU synthetic evidence mutations for strict queue-training review."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_native_queue_training_v8_independent_review as independent
from tools.predictor_validation_20261027 import native_queue_training_review_v8 as review
from tools.predictor_validation_20261027 import native_queue_training_v8 as queue


class Adapter:
    """A CPU fixture protocol; real review loads each frozen compact adapter."""
    def build(self, candidate, sample):
        return dict(generation=dict(max_new_tokens=2), sample_id=sample['id'])

    def finish(self, request, sample, raw_output, evaluate_quality=False, **observation):
        return dict(request_sha256='fixture-' + sample['id'], protocol_sha256='fixture-protocol',
                    prediction=raw_output, success=True, finish_reason='stop', generated_tokens=2)


class QueueReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = independent.QueueIndependentReview(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        context_path = fixture.profile / 'context.json'
        context = json.loads(context_path.read_text())
        extra_sources = [fixture.base / 'runtime-inputs.jsonl']
        for family, binding in context['worker_bindings'].items():
            spec = fixture.profile / (family + '-spec.json')
            queue.save(spec, dict(candidate=dict(slot=family)))
            binding['spec'] = str(spec)
            extra_sources.append(spec)
        context['execution_sources'] += [dict(original_path=str(p), frozen=queue.receipt(p)) for p in extra_sources]
        context['sources'] = [s['frozen'] for s in context['execution_sources']]
        context_path.write_text(json.dumps(context))
        profile_path = fixture.profile / 'plan.json'
        profile = json.loads(profile_path.read_text())
        profile['execution_context_sha256'] = queue.context_identity(context)
        fixture.reseal(profile)
        profile_path.write_text(json.dumps(profile))
        references = {task: {} for task in ('text', 'image', 'asr', 'ocr')}
        for row in fixture.inputs:
            references[row['task']][row['pool_id']] = dict(task=row['task'], label='synthetic answer', answers=['synthetic answer'])
        (fixture.base / 'catalog.json').write_text(json.dumps(dict(safe_factor=1.1, quality=dict(references=references))))
        self.plan = fixture.prepare()
        self.plan_path = self.root / 'prepared/plan.json'
        self.run = self.root / 'run'
        self.collection = self.run / 'collection'
        self.collection.mkdir(parents=True)
        self.pids = {alias: 1000 + i for i, alias in enumerate(queue.RESIDENTS)}
        context = json.loads((self.root / 'prepared/context.json').read_text())
        self.raw = dict(schema='gsdeploy.native_flow_collection.v1', queue_training_schema=queue.SCHEMA,
            kind='measured', backend='native', status='completed', collection_complete=True,
            split='training', training_only=True, errors=[], runtime_errors=[], telemetry_errors=[], qualification_issues=[],
            execution_context_sha256=self.plan['execution_context_sha256'], run_id=self.plan['run_id'],
            group_id=self.plan['group_id'], trace_sha256=self.plan['trace_sha256'], candidate_id='shared_declared_training',
            resource_configuration=self.plan['candidates'][0]['resource_configuration'], replica_to_family=queue.RESIDENTS,
            resident_aliases=list(queue.RESIDENTS), plan=queue.receipt(self.plan_path), context=self.plan['context'],
            effective_objective={k: self.plan[k] for k in ('workload_rps', 'latency_reference_ms', 'weights',
                'memory_safe_factor', 'memory_budget_mib', 'energy_reference_watts', 'startup_amortization_seconds')},
            startup_begin_ns=10_000_000_000, startup_end_ns=15_000_000_000,
            phase_start_ns=100_000_000_000, deadline_ns=116_000_000_000, observation_end_ns=117_000_000_000,
            collection_started_unix_ns=20, warmup_request_records=[],
            ready={alias: dict(pid=self.pids[alias], clock='time.perf_counter_ns', worker=family,
                spec_sha256=queue.receipt(context['worker_bindings'][family]['spec'])['sha256'],
                tasks=self.plan['aliases'][alias]['supported_tasks']) for alias, family in queue.RESIDENTS.items()},
            shutdown={alias: dict(exit_code=0, forced=False, bye_received=True) for alias in queue.RESIDENTS})
        self.raw['priming_request_records'] = [self.row('priming:' + str(i), alias, item['task'], item['pool_id'],
            20_000_000_000 + i * 1_000_000_000, 'priming') for i, (alias, item) in enumerate(review.priming_items(profile))]
        self.raw['request_records'] = [self.row(r['request_id'], r['alias'], r['task'], r['pool_id'],
            self.raw['phase_start_ns'] + r['offset_ns'], 'measurement') for r in self.plan['trace']]
        samples = []
        for at in range(9_000_000_000, 119_000_000_000, 1_000_000_000):
            owned = {} if at < self.raw['startup_end_ns'] else {pid: alias for alias, pid in self.pids.items()}
            samples.append(dict(time_ns=at, query_duration_ns=100, watts=100., board_total_mib=49140.,
                board_used_mib=100. * len(owned), board_free_mib=49140. - 100. * len(owned),
                processes_mib={pid: 100. for pid in owned}, owned_pid_aliases=owned,
                owned_aliases=list(owned.values()), owned_used_mib=100. * len(owned), foreign_pids=[]))
        self.raw['board_samples'] = samples
        self.write_raw()
        budget = dict(free_disk_reserve_bytes=12 * 2**30, disk_abort_margin_bytes=2**30,
            incremental_storage_scenario_bytes=6 * 2**30, total_timeout_seconds=1200.,
            watchdog_interval_seconds=.2, cleanup_grace_seconds=8.)
        command = [review.sys.executable, '-m', 'tools.predictor_validation_20261027.native_queue_training_v8',
                   'cohort', '--plan', str(self.plan_path), '--output', str(self.collection)]
        self.intent = dict(schema='gsdeploy.native_queue_training_execution.v8', plan=queue.receipt(self.plan_path),
            runner=queue.receipt(queue.__file__), command=command, budget=budget, split='training', training_only=True,
            started_unix_ns=10, status='FAILED_NO_RETRY', GPU_started=False)
        self.execution = dict(self.intent, status='COLLECTED_TRAINING_ONLY', GPU_started=True, ended_unix_ns=30,
            final_source_audit='PASS',
            raw=queue.receipt(self.collection / 'raw.json'), watchdog=dict(status='EXITED', return_code=0,
                samples=[dict(elapsed_seconds=0., free_bytes=20 * 2**30), dict(elapsed_seconds=100., free_bytes=20 * 2**30)],
                cleanup=dict(events=[], live_survivors=[])))
        queue.save(self.run / 'execution-start.json', self.intent)
        self.write_execution()
        queue.save(self.collection / 'invocation.json', dict(plan=queue.receipt(self.plan_path),
            candidate_id='shared_declared_training', runner=queue.receipt(queue.__file__)))
        queue.save(self.collection / 'execution-start.json', dict(plan=queue.receipt(self.plan_path),
            runner=queue.receipt(queue.__file__), started_unix_ns=20, split='training', training_only=True,
            full_trace_predeclared=True, original_measurement_window_seconds=16.))

    def row(self, rid, alias, task, pool_id, at, phase):
        begin, end = at + 1_000_000, at + 3_000_000
        pid = self.pids[alias]
        def event(stamp, **values):
            return dict(request_id=rid, pid=pid, perf_ns=stamp, received_time_ns=stamp + 100,
                        collector_dequeued_time_ns=stamp + 200, collector_processing_time_ns=stamp + 300, **values)
        row = dict(request_id=rid, model=alias, task=task, pool_id=pool_id, phase=phase,
            scheduled_time_ns=at, actual_submit_time_ns=at + 100, status='completed',
            observed_worker_start_time_ns=begin, completion_time_ns=end + 100,
            response_reader_event_enqueue_time_ns=end + 100, response_collector_dequeued_time_ns=end + 200,
            response_collector_processing_time_ns=end + 300,
            execution_events=[event(begin), event(begin + 1000, event='start', kind='prefill', segment_id=0),
                event(end - 1000, event='end', kind='prefill', segment_id=0)],
            previous_handler_next_request_ready=dict(event(end + 200), clock='host_perf_counter_ns',
                boundary='previous_handler_next_request_ready'),
            worker_response=dict(request_id=rid, task_type=task, task=task, sample_id=pool_id,
                start_perf_ns=begin, end_perf_ns=end, success=True, service_ms=2., worker_latency_ms=2.,
                raw_output='synthetic answer', prediction='synthetic answer', input_tokens=5,
                finish_reason='ctc_complete' if task == 'ocr' else 'stop', generated_tokens=2,
                request_sha256='fixture-' + pool_id, protocol_sha256='fixture-protocol',
                effective_generation=dict(max_new_tokens=2),
                semantic_forward_observations=dict(records=[dict(synthetic=True)], instrumentation_errors=[])))
        if phase == 'measurement':
            row.update(split='training', validation_view='TRAINING_ONLY_NOT_VALIDATION')
        return row

    def write_raw(self):
        (self.collection / 'raw.json').write_text(json.dumps(self.raw))
        catalog = json.loads((self.fixture.base / 'catalog.json').read_text())
        quality = dict(kind='measured_per_request', totals_by_request=review.recompute_quality(self.raw, catalog), errors=[],
            source_receipts=[self.plan['catalog'], queue.receipt(queue.__file__),
                queue.receipt(review.ROOT / 'gsdeploy/tasks.py'), queue.receipt(review.ROOT / 'gsdeploy/quality.py')])
        self.scored = dict(self.raw, quality_evidence=quality, native_raw=queue.receipt(self.collection / 'raw.json'))
        (self.collection / 'scored-raw.json').write_text(json.dumps(self.scored))
        if hasattr(self, 'execution'):
            self.execution['raw'] = queue.receipt(self.collection / 'raw.json')
            self.write_execution()

    def write_execution(self):
        (self.run / 'execution.json').write_text(json.dumps(self.execution))

    def evaluate(self):
        with patch.object(review, 'load_cpu_adapter', return_value=Adapter()):
            return review.review(self.plan_path, self.run, self.root / 'review')

    def test_complete_synthetic_capture_passes_only_as_training(self):
        result = self.evaluate()
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['request_timings']), 24)
        self.assertTrue(result['six_term_diagnostics']['utility_complete'])
        self.assertFalse(result['original_stability_evaluated'])
        self.assertEqual(result['independent_prediction_accuracy'], 'UNKNOWN')

    def test_route_mutation_rejected_with_resealed_output_receipts(self):
        self.raw['request_records'][0]['model'] = 'text#1'
        self.write_raw()
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('route/input/scheduled' in s for s in result['issues']))

    def test_missing_phase_anchor_rejected(self):
        self.raw['request_records'][0]['execution_events'].pop()
        self.write_raw()
        result = self.evaluate()
        self.assertTrue(any('INCOMPLETE_PHASE_ANCHORS' in s for s in result['issues']))

    def test_forged_prediction_rejected_by_cpu_reparse(self):
        self.raw['request_records'][0]['worker_response']['prediction'] = 'forged'
        self.write_raw()
        result = self.evaluate()
        self.assertTrue(any('CPU_REPARSED_OUTPUT_MISMATCH' in s for s in result['issues']))

    def test_saved_quality_totals_recomputed(self):
        self.scored['quality_evidence']['totals_by_request']['measurement:0']['numerator'] = 1.
        (self.collection / 'scored-raw.json').write_text(json.dumps(self.scored))
        result = self.evaluate()
        self.assertIn('QUALITY_EVIDENCE_DIFFERS_FROM_RECOMPUTED_TRAINING_OUTPUTS', result['issues'])

    def test_reversed_raw_telemetry_never_sorted_into_pass(self):
        self.raw['board_samples'][2:4] = list(reversed(self.raw['board_samples'][2:4]))
        self.write_raw()
        result = self.evaluate()
        self.assertTrue(any('strictly increase' in s for s in result['issues']))

    def test_one_missing_primer_rejected(self):
        self.raw['priming_request_records'].pop()
        self.write_raw()
        result = self.evaluate()
        self.assertIn('EXACT_ORIGINAL_TWELVE_PRIMING_REQUESTS_REQUIRED', result['issues'])

    def test_watchdog_survivor_rejected(self):
        self.execution['watchdog']['cleanup']['live_survivors'] = [1234]
        self.write_execution()
        result = self.evaluate()
        self.assertIn('WATCHDOG_OR_OWNED_CLEANUP_FAILED', result['issues'])

    def test_primer_ready_reader_must_finish_before_measurement(self):
        self.raw['priming_request_records'][-1]['previous_handler_next_request_ready']['collector_processing_time_ns'] = self.raw['phase_start_ns'] + 1
        self.write_raw()
        result = self.evaluate()
        self.assertIn('PRIMING_READER_BOUNDARIES_NOT_COMPLETE_BEFORE_MEASUREMENT', result['issues'])

    def test_controller_final_source_audit_required(self):
        del self.execution['final_source_audit']
        self.write_execution()
        result = self.evaluate()
        self.assertIn('SUPERVISED_COLLECTION_NOT_QUALIFIED', result['issues'])


if __name__ == '__main__':
    unittest.main()
