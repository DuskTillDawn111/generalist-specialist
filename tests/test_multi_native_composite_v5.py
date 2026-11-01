"""Small CPU evidence fixtures exercise original providers, dispatch and DAG v6."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from gsdeploy.forecasting.lifecycle import export_training_observations
from tools.predictor_validation_20261027 import multi_native_composite_v5 as tool
from tools.predictor_validation_20261027 import forecast_gate_multi_native_v5 as gate
from tools.predictor_validation_20261027.native_collect import receipt, context_identity
from tools.predictor_validation_20261027.pooled_lifecycle_v4 import fit as fit_pooled


def fixtures(root):
    def save(name, doc):
        path = root / name
        path.write_text(json.dumps(doc))
        return receipt(path)
    source = root / 'native-source.txt'; source.write_text('CPU fixture, no model code')
    source_ref = receipt(source)
    stats = save('weight-stats.json', [])
    residents = {'text#0': 'text', 'text#1': 'text', 'generalist_2#0': 'generalist_2', 'ocr#0': 'ocr'}
    aliases = {a: dict(family=f, supported_tasks=['image', 'asr'] if f == 'generalist_2' else [f])
               for a, f in residents.items()}
    def ctx(name, domain, collector):
        doc = dict(shared_native_execution_sources=[source_ref], worker_bindings={}, residents=residents,
            threads={'OMP_NUM_THREADS': '1'}, execution={'batch_size': 1}, device='CPU-fixture',
            measurement_domain=domain, profiler='fixture' if domain == 'light-kernel' else None,
            affine_observation=False, worker_variant='fixture', large_weight_identity_scope='stat_only',
            large_weight_stats=stats, execution_sources=[dict(original_path=str(source), frozen=source_ref)],
            sources=[], collector=collector)
        return save(name, doc), context_identity(doc)
    specs, reviews = [], []
    for cid, tasks in [('original', ['text', 'ocr']), ('modality', ['image', 'asr'])]:
        light_ref, light_id = ctx(cid + '-light.json', 'light-kernel', cid)
        natural_ref, natural_id = ctx(cid + '-natural.json', 'natural', cid)
        profiles = []
        for task in tasks:
            alias = {'text': 'text#0', 'ocr': 'ocr#0'}.get(task, 'generalist_2#0')
            for i in range(2):
                profiles.append(dict(alias=alias, family=residents[alias], task=task, pool_id=task + str(i),
                    request_id=task + str(i), cyclic=False, unresolved=[], source_sqlite=source_ref,
                    nodes=[dict(id='gpu', kind='gpu', duration_ns=10, kernel=dict(task=task),
                                issue_start_ns=0, issue_end_ns=1, issue_thread=1),
                           dict(id='copy:1', kind='copy', duration_ns=3,
                                issue_start_ns=2, issue_end_ns=3, issue_thread=1),
                           dict(id='end', kind='handler_end', duration_ns=0)],
                    edges=[dict(before='gpu', after='copy:1'), dict(before='copy:1', after='end')]))
        dag_ref = save(cid + '-dag.json', dict(schema='gsdeploy.request_cuda_dag_profiles.v2',
            training_only=True, execution_context_sha256=light_id, profiles=profiles))
        dag_review = save(cid + '-dag-review.json', dict(schema='gsdeploy.multitask_dag_independent_cpu_review.v5',
            profiles=[dict(alias=p['alias'], task=p['task'], pool_id=p['pool_id'], source_dag=dag_ref,
                original_full_context_sha256=light_id, complete_canonical_kernel_identity_match=True) for p in profiles]))
        natural_reviews = []
        for capture in range(2):
            raw_ref = save(f'{cid}-raw-{capture}.json', dict(kind='measured', status='completed',
                execution_context_sha256=natural_id, independent_capture_fixture=capture,
                request_records=[dict(request_id=p['request_id'], model=p['alias'], task=p['task'],
                    pool_id=p['pool_id'], split='training', status='completed', worker_response=dict(
                    task=p['task'], success=True, prediction=1 if p['task'] == 'image' else 'word',
                    start_perf_ns=1000, end_perf_ns=1107,
                    observed_stages=[dict(id='final_sync', end_ns=1100)]),
                    completion_time_ns=1112, response_reader_event_enqueue_time_ns=1112,
                    previous_handler_next_request_ready=dict(request_id=p['request_id'], pid=123,
                        perf_ns=1110, received_time_ns=1113, clock='host_perf_counter_ns',
                        boundary='previous_handler_next_request_ready', scope='after_response_flush_before_next_read')) for p in profiles]))
            plan_ref = save(f'{cid}-plan-{capture}.json', dict(domain='natural', split='training', execution_context_sha256=natural_id))
            natural_reviews.append(save(f'{cid}-review-{capture}.json', dict(
                schema='gsdeploy.startup_training_collection_strict_review.v4', status='PASS', issues=[],
                training_only=True, raw=raw_ref, plan=plan_ref, context=natural_ref,
                evaluator_sources=[source_ref], owners={})))
        host_dir = root / (cid + '-host')
        tool.fit_host_component(dag_ref['path'], light_ref['path'], natural_ref['path'],
                                [r['path'] for r in natural_reviews], host_dir)
        observations, totals, captures = [], [], []
        clock_contract = dict(handler='host_perf_counter_ns', runtime='host_perf_counter_ns',
            worker_next_ready='host_perf_counter_ns', same_host_verified=True)
        clock_ref = save(cid + '-clock.json', dict(schema='gsdeploy.natural_training_host_clock_evidence.v2',
            clock_contract=clock_contract, sources=[tool.checked(r)['raw'] for r in natural_reviews]))
        clock = dict(clock_contract, source='sha256:' + clock_ref['sha256'])
        references = {task: {p['pool_id']: dict(task=task, answers=['word']) if task == 'text'
            else dict(task=task, label=1 if task == 'image' else 'word') for p in profiles if p['task'] == task} for task in tasks}
        catalog = save(cid + '-catalog.json', dict(quality=dict(references=references)))
        for ref in natural_reviews:
            review = tool.checked(ref); raw = tool.checked(review['raw']); run = 'raw-sha256:' + review['raw']['sha256']
            observations.extend(export_training_observations(dict(raw, split='training', admission_enabled=False),
                aliases, execution_context_sha256=natural_id, source=run, clock_contract=clock))
            captures.append(dict(review=ref, raw=review['raw'], run_id=run))
            for row in raw['request_records']:
                task, pool = row['task'], row['pool_id']
                numerator, denominator = tool.sample_totals(task, tool.score_result(references[task][pool], row['worker_response']))
                totals.append(dict(run_id=run, request_id=row['request_id'], family=residents[row['model']], task=task,
                    pool_id=pool, numerator=numerator, denominator=denominator))
        observations_ref = save(cid + '-observations.json', dict(observations=observations))
        totals_ref = save(cid + '-totals.json', dict(records=totals)); evidence = 'sha256:' + totals_ref['sha256']
        index = save(cid + '-training-index.json', dict(split='training', records=[
            {k:r[k] for k in ('run_id', 'request_id', 'pool_id', 'task')} for r in totals]))
        quality = {}
        for task in tasks:
            family = residents[next(p['alias'] for p in profiles if p['task'] == task)]
            rows = [r for r in totals if r['task'] == task]
            quality.setdefault(family, {})[task] = dict(by_input={p['pool_id']:tool.expected_totals(
                [r for r in rows if r['pool_id'] == p['pool_id']], evidence) for p in profiles if p['task'] == task},
                fallback=tool.expected_totals(rows, evidence))
        quality_ref = save(cid + '-quality.json', quality)
        quality_report = save(cid + '-quality-report.json', dict(schema='gsdeploy.natural_training_handoff_quality_fit.v4',
            status='FITTED', captures=captures, clock_evidence=clock_ref, catalog=catalog,
            execution_context_sha256=natural_id, quality_loss=quality_ref, observations=observations_ref,
            training_totals=totals_ref, training_index=index, sources=[source_ref]))
        lifecycle = fit_pooled(observations, natural_id)
        lifecycle['sources'] = [observations_ref, quality_report, source_ref]
        lifecycle_ref = save(cid + '-lifecycle.json', lifecycle)
        specs.append(dict(id=cid, tasks=tasks, dag=dag_ref, dag_review=dag_review, light_context=light_ref,
            natural_context=natural_ref, host_supply=receipt(host_dir / 'model.json'), lifecycle=lifecycle_ref,
            quality=quality_ref, quality_report=quality_report, historical_resolution=None))
        reviews.extend(natural_reviews)
    hypotheses = {key: True for key in tool.OPT_INS}
    contract = tool.compose(specs, specs[0]['natural_context']['path'], hypotheses=hypotheses)
    contract_ref = save('composite.json', contract)
    return dict(save=save, specs=specs, reviews=reviews, hypotheses=hypotheses,
                contract=contract, contract_ref=contract_ref, aliases=aliases, source=source)


class MultiNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = fixtures(self.root)

    def provider(self):
        return tool.MultiNativeProvider(self.f['contract'], self.f['aliases'], host_scenario='front_loaded_supply')

    def test_original_contexts_and_host_lifecycle_dispatch_are_preserved(self):
        provider = self.provider()
        original_ids = set()
        for alias, task, pool in provider.plans:
            row = dict(alias=alias, task=task, pool_id=pool, request_id=pool, arrival_ns=0)
            result = provider.delays(row, self.f['aliases'], 'profiled')
            original_ids.add(result['original_execution_context_sha256'])
            self.assertEqual(result['parent_receipt_ms'], .000005)
            self.assertEqual(result['worker_next_ready_ms'], .000003)
            release = provider.release(provider.plans[alias, task, pool]['nodes'][-1], row)
            self.assertEqual(release, dict(not_before_offset_ns=0, after_predecessors_ns=7))
            self.assertEqual(provider.provenance(row)['composite_execution_sha256'], provider.execution_context_sha256)
        self.assertEqual(len(original_ids), 2)
        self.assertNotIn(provider.execution_context_sha256, original_ids)
        self.assertFalse(provider.physical_evidence)

    def test_fit_is_real_model_plus_exact_index_not_feasibility_wrapper(self):
        model = tool.checked(self.f['specs'][1]['host_supply'])
        self.assertEqual(model['schema'], 'gsdeploy.dag_host_supply_training_hypothesis.v5')
        self.assertEqual(len(model['observations']), 8)
        self.assertEqual(model['supported_group_count'], 2)
        self.assertEqual(len(tool.checked(model['training_index'])['records']), 8)
        self.assertFalse(model['identification_complete'])
        self.assertFalse(model['natural_timing_accuracy_validated'])

    def test_duplicate_route_or_missing_optin_is_rejected(self):
        specs = copy.deepcopy(self.f['specs']); duplicate = copy.deepcopy(specs[0]); duplicate['id'] = 'duplicate'
        with self.assertRaisesRegex(ValueError, 'Duplicate multi-native'):
            tool.compose(specs + [duplicate], specs[0]['natural_context']['path'], hypotheses=self.f['hypotheses'])
        with self.assertRaisesRegex(ValueError, 'explicit opt-in'):
            tool.compose(specs, specs[0]['natural_context']['path'], hypotheses={})

    def test_tampered_host_mean_or_training_index_cannot_pass_with_new_receipt(self):
        spec = copy.deepcopy(self.f['specs'][1]); model = tool.checked(spec['host_supply'])
        model['groups'][0]['joint_mean']['terminal_tail_ns'] += 1
        spec['host_supply'] = self.f['save']('bad-host.json', model)
        with self.assertRaisesRegex(ValueError, 'recomputed original training evidence'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])
        model = tool.checked(self.f['specs'][1]['host_supply'])
        index = tool.checked(model['training_index']); index['records'].pop()
        model['training_index'] = self.f['save']('incomplete-index.json', index)
        spec['host_supply'] = self.f['save']('bad-index-host.json', model)
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])

    def test_one_capture_or_validation_row_is_not_training(self):
        spec = self.f['specs'][1]; reviews = tool.checked(spec['host_supply'])['captures']
        with self.assertRaisesRegex(UnknownEvidence, 'TWO_CAPTURE'):
            tool.fit_host_component(spec['dag']['path'], spec['light_context']['path'], spec['natural_context']['path'],
                                    [reviews[0]['review']['path']], self.root / 'one-capture')
        review = tool.checked(reviews[0]['review']); raw = tool.checked(review['raw'])
        raw['request_records'][0]['split'] = 'validation'
        review['raw'] = self.f['save']('validation-raw.json', raw)
        ref = self.f['save']('validation-review.json', review)
        with self.assertRaisesRegex(ValueError, 'completed training rows'):
            tool.fit_host_component(spec['dag']['path'], spec['light_context']['path'], spec['natural_context']['path'],
                                    [ref['path'], reviews[1]['review']['path']], self.root / 'validation-fit')

    def test_changed_target_source_and_unknown_input_are_rejected(self):
        provider = self.provider()
        with self.assertRaisesRegex(UnknownEvidence, 'NO_EXACT_MULTI_NATIVE'):
            provider.component(dict(alias='generalist_2#0', task='asr', pool_id='unseen'))
        self.f['source'].write_text('changed')
        with self.assertRaises(ValueError):
            self.provider()

    def test_changed_domain_projection_or_incomplete_canonical_review_rejected(self):
        spec = copy.deepcopy(self.f['specs'][1])
        doc = tool.checked(spec['light_context']); doc['execution']['batch_size'] = 2
        spec['light_context'] = self.f['save']('wrong-light.json', doc)
        with self.assertRaisesRegex(ValueError, 'native projection'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])

    def test_nested_quality_dependency_and_refreshed_lifecycle_mean_are_rejected(self):
        spec = copy.deepcopy(self.f['specs'][1]); report = tool.checked(spec['quality_report'])
        dependency = self.root / 'quality-dependency.txt'; dependency.write_text('original')
        report['sources'].append(receipt(dependency))
        spec['quality_report'] = self.f['save']('quality-nested.json', report)
        contract = tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'],
                                hypotheses=self.f['hypotheses'])
        dependency.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'Changed receipt'):
            tool.MultiNativeProvider(contract, self.f['aliases'], host_scenario='front_loaded_supply')
        spec = copy.deepcopy(self.f['specs'][1]); model = tool.checked(spec['lifecycle'])
        model['lifecycle']['profiles'][0]['scenarios']['mean']['parent_receipt_ms'] += 123
        spec['lifecycle'] = self.f['save']('changed-lifecycle.json', model)
        with self.assertRaisesRegex(ValueError, 'recomputed original training observations'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])

    def test_quality_totals_recomputed_from_actual_training_responses(self):
        spec = copy.deepcopy(self.f['specs'][1]); report = tool.checked(spec['quality_report'])
        totals = tool.checked(report['training_totals']); totals['records'][0]['numerator'] += 1
        report['training_totals'] = self.f['save']('changed-totals.json', totals)
        spec['quality_report'] = self.f['save']('changed-quality-report.json', report)
        with self.assertRaisesRegex(ValueError, 'actual completed training responses'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])
        spec = copy.deepcopy(self.f['specs'][1]); review = tool.checked(spec['dag_review']); review['profiles'].pop()
        spec['dag_review'] = self.f['save']('incomplete-dag-review.json', review)
        with self.assertRaisesRegex(ValueError, 'canonical DAG denominator'):
            tool.compose([self.f['specs'][0], spec], self.f['specs'][0]['natural_context']['path'], hypotheses=self.f['hypotheses'])

    def gate_inputs(self):
        provider = self.provider(); queues = {a: [] for a in self.f['aliases']}
        for i, (alias, task, pool) in enumerate(provider.plans):
            queues[alias].append(dict(request_id=str(i), alias=alias, task=task, pool_id=pool, arrival_ns=0))
        tasks = ['text', 'ocr', 'image', 'asr']
        public = dict(phase_start_ns=0, now_ns=0, deadline_ns=1000, drain_deadline_ns=10000, queues=queues,
            future_workload=dict(rates={t: 1 for t in tasks}, process='declared_schedule', declared_arrivals=[],
                input_pools={t: [k[2] for k in provider.plans if k[1] == t] for t in tasks}))
        fit_ref = self.f['save']('numeric-fit.json', {})
        copy_ref = self.f['save']('copy-fit.json', {})
        bundle = dict(schema=gate.BUNDLE_SCHEMA, kind='trained_hypothesis', composite=self.f['contract_ref'],
            semantic_fit=fit_ref, copy_memset_fit=copy_ref, allow_numeric_semantic_hypothesis=True,
            allow_copy_memset_numeric_hypothesis=True, host_supply_scenario='front_loaded_supply',
            training_template_diagnostic=True, weights={k: 1 / 6 for k in 'QTLPME'},
            dag_hypothesis_options=dict(allow_independent_copy_hypothesis=True, allow_context_time_slice_hypothesis=True),
            forecast_defaults=dict(aliases=self.f['aliases'], resources=None, execution_contract='strict_lifecycle_v2',
                route_initial_with_continuation_policy=False, latency_reference_ms={t: 1000 for t in tasks},
                continuation_routing_rps={t: {next(a for a, row in self.f['aliases'].items() if t in row['supported_tasks']): 1} for t in tasks},
                memory_budget_mib=1000, energy_reference_watts=100, memory_safe_factor=1))
        return bundle, public

    def resolvers(self, bundle, *, missing=False):
        result = dict(hypothesis_only=True, deployment_qualified=False, strict_known=False,
            numeric_hypothesis_available=not missing, strict_failure_reasons=['ORIGINAL_STRICT_FAILURE'], duration_ns=5.5)
        class Resolver:
            def __init__(self, ref): self.fit_source = ref
            def __call__(self, *args): return copy.deepcopy(result)
        return patch.object(gate, 'ExactSourceSemanticHypothesisResolver', return_value=Resolver(bundle['semantic_fit'])), \
            patch.object(gate, 'ExactSourceCopyMemsetHypothesisResolver', return_value=Resolver(bundle['copy_memset_fit']))

    def test_actual_v6_engine_four_task_dispatch_and_unchanged_strict_failures(self):
        bundle, public = self.gate_inputs(); baseline, copy_provider = self.resolvers(bundle)
        with baseline, copy_provider:
            result = gate.execute_bundle(bundle, public, allow_trained_hypothesis=True, allow_training_diagnostic=True)
        self.assertEqual(len(result['forecast']['finishes']), 8)
        self.assertEqual({r['task'] for r in result['forecast']['finishes'].values()}, {'text', 'ocr', 'image', 'asr'})
        self.assertEqual(result['baseline_coverage']['gpu']['numeric_hypothesis_nodes'], 8)
        self.assertEqual(result['baseline_coverage']['gpu']['strict_qualified_nodes'], 0)
        self.assertEqual(result['baseline_coverage']['copy_memset']['strict_failure_reason_counts'], {'ORIGINAL_STRICT_FAILURE': 8})
        self.assertIsNone(result['deployment_utility'])
        self.assertFalse(result['startup_prediction_verified'])
        self.assertFalse(result['prospective_prediction_seal_created'])
        self.assertFalse(result['GPU_started'])
        for row in result['forecast']['finishes'].values():
            self.assertEqual(row['finish_ns'] - row['handler_end_ns'], 5)
            self.assertEqual(row['worker_next_ready_ns'] - row['handler_end_ns'], 3)

    def test_missing_numeric_never_falls_back_to_node_duration(self):
        bundle, public = self.gate_inputs(); baseline, copy_provider = self.resolvers(bundle, missing=True)
        with baseline, copy_provider, self.assertRaisesRegex(UnknownEvidence, 'NUMERIC_NODE_DURATION_UNAVAILABLE'):
            gate.execute_bundle(bundle, public, allow_trained_hypothesis=True, allow_training_diagnostic=True)

    def test_queue_diagnostics_share_unique_template_durations_for_repeated_requests(self):
        bundle, public = self.gate_inputs(); bundle['training_template_diagnostic'] = False
        bundle.update(generation=dict(generation_work=bundle['semantic_fit'], generation_dag_support=bundle['semantic_fit']),
                      allow_exact_content_generation_hypothesis=True)
        public['queues']['text#0'].append(dict(public['queues']['text#0'][0], request_id='repeat'))
        baseline, copy_provider = self.resolvers(bundle)
        with baseline, copy_provider, patch('tools.predictor_validation_20261027.known_input_generation_v6.check', return_value={}):
            result = gate.execute_bundle(bundle, public, allow_trained_hypothesis=True)
        self.assertEqual(len(result['forecast']['finishes']), 9)
        self.assertEqual(result['baseline_coverage']['gpu']['numeric_hypothesis_nodes'], 8)
        self.assertEqual(result['baseline_coverage']['gpu']['coverage_denominator'],
                         'unique_used_template_nodes_cached_for_all_requests')
        diagnostics = result['forecast']['queue_diagnostics']
        self.assertTrue(diagnostics['offered_work_envelopes'])
        self.assertFalse(diagnostics['visible_state_correction_applied'])
        self.assertFalse(diagnostics['six_term_utility_changed'])

    def test_prospective_scope_requires_generation_and_diagnostic_requires_full_cohort(self):
        bundle, public = self.gate_inputs(); bundle['training_template_diagnostic'] = False
        with self.assertRaisesRegex(UnknownEvidence, 'GENERATION_SUPPORT'):
            gate.execute_bundle(bundle, public, allow_trained_hypothesis=True)
        bundle['training_template_diagnostic'] = True
        public['queues']['text#0'].pop()
        with self.assertRaisesRegex(ValueError, 'every exact training template'):
            gate.execute_bundle(bundle, public, allow_trained_hypothesis=True, allow_training_diagnostic=True)


if __name__ == '__main__':
    unittest.main()
