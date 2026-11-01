"""Strict CPU review of predeclared native queue training, never validation.

Checks the original complete measurement window, bounded parent receipts,
four resident owners and twelve primers, explicit native event chains, CPU
output reparsing, source/weight identities and the original six-term arithmetic.
A PASS qualifies a training capture only; it grants no stability or accuracy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from . import native_queue_training_v8 as collector
from .background_collect import ROOT, receipt, save
from .flow_metrics import score_flow
from .known_input_observed_v5 import _timings, _resources
from .profiling import RESIDENTS
from .smoke_verifier_v3 import (
    check_receipt, finite, stamp, load_cpu_adapter, validate_output, validate_request,
)

SCHEMA = 'gsdeploy.native_queue_training_strict_review.v8'


def priming_items(profile):
    rows = []
    for alias, family in RESIDENTS.items():
        tasks = ['text', 'image', 'asr'] if family == 'generalist_2' else ['ocr' if family == 'ocr' else 'text']
        for task in tasks:
            rows.extend((alias, item) for item in sorted(
                (r for r in profile['inputs'] if r['task'] == task),
                key=lambda r: r['input_size'], reverse=True)[:3])
    return rows


def collection_issues(plan, profile, raw):
    issues = []
    if (raw.get('schema') != 'gsdeploy.native_flow_collection.v1'
            or raw.get('queue_training_schema') != collector.SCHEMA
            or raw.get('kind') != 'measured' or raw.get('backend') != 'native'
            or raw.get('status') != 'completed' or raw.get('collection_complete') is not True
            or raw.get('split') != 'training' or raw.get('training_only') is not True
            or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors')
            or raw.get('qualification_issues')):
        issues.append('COLLECTION_NOT_COMPLETE_NATIVE_TRAINING')
    candidate = plan['candidates'][0]
    for key in ('execution_context_sha256', 'run_id', 'group_id', 'trace_sha256'):
        if raw.get(key) != plan[key]:
            issues.append('RAW_PLAN_IDENTITY_MISMATCH:' + key)
    if (raw.get('candidate_id') != candidate['id']
            or raw.get('resource_configuration') != candidate['resource_configuration']
            or raw.get('replica_to_family') != RESIDENTS
            or raw.get('resident_aliases') != list(RESIDENTS)):
        issues.append('RAW_ORIGINAL_OWNERS_OR_RESOURCE_MISMATCH')
    objective = {k: plan[k] for k in ('workload_rps', 'latency_reference_ms', 'weights',
        'memory_safe_factor', 'memory_budget_mib', 'energy_reference_watts', 'startup_amortization_seconds')}
    if raw.get('effective_objective') != objective:
        issues.append('ORIGINAL_SIX_TERM_OBJECTIVE_CHANGED')
    rows = raw.get('request_records', [])
    if ([r.get('request_id') for r in rows] != [r['request_id'] for r in plan['trace']]
            or any(r.get('phase') != 'measurement' or r.get('split') != 'training'
                   or r.get('validation_view') != 'TRAINING_ONLY_NOT_VALIDATION' for r in rows)
            or raw.get('warmup_request_records') != []):
        issues.append('EXACT_FULL_MEASUREMENT_COHORT_REQUIRED')
    expected = [(a, i['task'], i['pool_id']) for a, i in priming_items(profile)]
    primers = raw.get('priming_request_records', [])
    if (len(expected) != 12 or [(r.get('model'), r.get('task'), r.get('pool_id')) for r in primers] != expected
            or any(r.get('request_id') != 'priming:' + str(i) or r.get('phase') != 'priming'
                   or r.get('status') != 'completed' for i, r in enumerate(primers))):
        issues.append('EXACT_ORIGINAL_TWELVE_PRIMING_REQUESTS_REQUIRED')
    ready = raw.get('ready', {})
    if (set(ready) != set(RESIDENTS) or any(type(r.get('pid')) is not int or r['pid'] <= 0 for r in ready.values())
            or len({r.get('pid') for r in ready.values()}) != 4):
        issues.append('FOUR_DISTINCT_READY_OWNERS_REQUIRED')
    shutdown = raw.get('shutdown', {})
    if (set(shutdown) != set(RESIDENTS) or any(r.get('exit_code') != 0 or r.get('forced') is not False
            or r.get('bye_received') is not True for r in shutdown.values())):
        issues.append('CLEAN_FOUR_OWNER_SHUTDOWN_REQUIRED')
    if raw.get('precollection') or 'predictions' in raw:
        issues.append('TRAINING_MUST_NOT_CLAIM_VALIDATION')
    return issues


def controller_issues(plan_path, run, execution, intent, raw):
    issues = []
    expected_budget = dict(free_disk_reserve_bytes=12 * 2**30, disk_abort_margin_bytes=2**30,
        incremental_storage_scenario_bytes=6 * 2**30, total_timeout_seconds=1200.,
        watchdog_interval_seconds=.2, cleanup_grace_seconds=8.)
    command = [sys.executable, '-m', 'tools.predictor_validation_20261027.native_queue_training_v8',
               'cohort', '--plan', str(plan_path), '--output', str(run / 'collection')]
    if (execution.get('schema') != 'gsdeploy.native_queue_training_execution.v8'
            or execution.get('status') != 'COLLECTED_TRAINING_ONLY'
            or execution.get('split') != 'training' or execution.get('training_only') is not True
            or execution.get('GPU_started') is not True or execution.get('error')
            or execution.get('final_source_audit') != 'PASS'
            or execution.get('cleanup_error') or execution.get('source_error')
            or execution.get('exception_cleanup') or execution.get('budget') != expected_budget
            or execution.get('command') != command):
        issues.append('SUPERVISED_COLLECTION_NOT_QUALIFIED')
    for key in ('schema', 'plan', 'runner', 'command', 'budget', 'split', 'training_only', 'started_unix_ns'):
        if execution.get(key) != intent.get(key):
            issues.append('CONTROLLER_INTENT_CHANGED:' + key)
    for key, expected in (('plan', plan_path), ('runner', Path(collector.__file__)),
                          ('raw', run / 'collection/raw.json')):
        if not check_receipt(execution.get(key), expected):
            issues.append('CONTROLLER_RECEIPT_FAILED:' + key)
    started, child, ended = execution.get('started_unix_ns'), raw.get('collection_started_unix_ns'), execution.get('ended_unix_ns')
    if not all(stamp(t) for t in (started, child, ended)) or not started <= child <= ended:
        issues.append('CONTROLLER_CHILD_UNIX_CHRONOLOGY_INVALID')
    if intent.get('GPU_started') is not False or intent.get('status') != 'FAILED_NO_RETRY':
        issues.append('PREEXECUTION_INTENT_REQUIRED')
    watchdog = execution.get('watchdog', {})
    samples = watchdog.get('samples', [])
    if (watchdog.get('status') != 'EXITED' or watchdog.get('return_code') != 0
            or watchdog.get('cleanup', {}).get('live_survivors') != [] or not samples):
        issues.append('WATCHDOG_OR_OWNED_CLEANUP_FAILED')
    elapsed = [s.get('elapsed_seconds') for s in samples]
    if (any(not finite(t) for t in elapsed) or any(a > b for a, b in zip(elapsed, elapsed[1:]))
            or any(not finite(s.get('free_bytes')) or s['free_bytes'] < 13 * 2**30 for s in samples)
            or (elapsed and elapsed[-1] > expected_budget['total_timeout_seconds'])
            or (samples and any(samples[0]['free_bytes'] - s['free_bytes'] > 6 * 2**30 for s in samples))):
        issues.append('DISK_TIME_OR_STORAGE_SCENARIO_BUDGET_FAILED')
    return issues


def board_issues(raw):
    """Check board fields absent from the inherited owned-PID resource helper."""
    issues = []
    for sample in raw.get('board_samples', []):
        if (not stamp(sample.get('query_duration_ns'))
                or any(not finite(sample.get(k)) for k in ('board_used_mib', 'board_free_mib', 'board_total_mib'))
                or not finite(sample.get('board_total_mib'), positive=True)
                or sample['board_used_mib'] + sample['board_free_mib'] > sample['board_total_mib'] + 1e-3):
            issues.append('INVALID_BOARD_MEMORY_OR_QUERY_SAMPLE')
            break
        values = {int(pid): value for pid, value in sample.get('processes_mib', {}).items()}
        owners = {int(pid): alias for pid, alias in sample.get('owned_pid_aliases', {}).items()}
        if (not set(owners) <= set(values) or set(owners.values()) != set(sample.get('owned_aliases', []))
                or len(set(owners.values())) != len(owners)
                or any(alias not in RESIDENTS for alias in owners.values())
                or abs(sum(values[p] for p in owners) - sample.get('owned_used_mib', -1)) > 1e-6):
            issues.append('OWNED_PID_ALIAS_MAPPING_INVALID')
            break
        ready = raw.get('ready', {})
        if any(raw['startup_end_ns'] <= sample['time_ns'] and ready.get(alias, {}).get('pid') != pid
               for pid, alias in owners.items()):
            issues.append('NVML_READY_PID_MAPPING_CHANGED')
            break
    return issues


def recompute_quality(raw, catalog):
    totals = {}
    for row in raw['request_records']:
        result = copy.deepcopy(row['worker_response'])
        if row['task'] == 'ocr':
            result['prediction'] = page_ocr_text(result['prediction'])
        reference = catalog['quality']['references'][row['task']][row['pool_id']]
        numerator, denominator = sample_totals(row['task'], score_result(reference, result))
        totals[row['request_id']] = dict(task=row['task'], numerator=numerator, denominator=denominator)
    return totals


def review(plan_path, run, output):
    plan_path, run, output = map(lambda p: Path(p).resolve(), (plan_path, run, output))
    output.mkdir(parents=True, exist_ok=False)
    issues, owners, timings, diagnostics = [], {}, [], None
    evidence, raw = {}, None
    try:
        plan = json.loads(plan_path.read_text())
        collector.verify_sources(plan)
        context = json.loads(Path(plan['context']['path']).read_text())
        profile = json.loads(Path(plan['profile_plan']['path']).read_text())
        parent_context = json.loads(collector.checked(context['parent_training_context']).read_text())
        if collector.context_identity(parent_context) != profile['execution_context_sha256']:
            issues.append('ORIGINAL_PROFILE_CONTEXT_BINDING_CHANGED')
        paths = dict(plan=plan_path, context=Path(plan['context']['path']),
            raw=run / 'collection/raw.json', scored_raw=run / 'collection/scored-raw.json',
            execution=run / 'execution.json', intent=run / 'execution-start.json',
            invocation=run / 'collection/invocation.json', start=run / 'collection/execution-start.json')
        evidence = {key: receipt(path) for key, path in paths.items()}
        raw, scored, execution, intent, invocation, start = [json.loads(paths[k].read_text()) for k in
            ('raw', 'scored_raw', 'execution', 'intent', 'invocation', 'start')]
        for key, ref in (('plan', receipt(plan_path)), ('context', plan['context'])):
            if raw.get(key) != ref:
                issues.append('RAW_INPUT_RECEIPT_FAILED:' + key)
        if (scored.get('native_raw') != evidence['raw']
                or {k: v for k, v in scored.items() if k not in ('native_raw', 'quality_evidence')} != raw):
            issues.append('SCORED_RAW_CHANGED_NATIVE_COLLECTION')
        candidate = plan['candidates'][0]
        if invocation != dict(plan=receipt(plan_path), candidate_id=candidate['id'], runner=receipt(collector.__file__)):
            issues.append('COLLECTION_INVOCATION_CHANGED')
        if (start.get('plan') != receipt(plan_path) or start.get('runner') != receipt(collector.__file__)
                or start.get('split') != 'training' or start.get('training_only') is not True
                or start.get('full_trace_predeclared') is not True
                or start.get('original_measurement_window_seconds') != plan['window_seconds']
                or start.get('started_unix_ns') != raw.get('collection_started_unix_ns')):
            issues.append('CHILD_PREEXECUTION_DECLARATION_INVALID')
        issues += controller_issues(plan_path, run, execution, intent, raw)
        issues += collection_issues(plan, profile, raw)
        timings, reasons, _, _, _ = _timings(raw, plan, candidate)
        issues += reasons
        memory_ok, power_ok, reasons = _resources(raw, plan)
        issues += reasons + board_issues(raw)
        if not memory_ok or not power_ok:
            issues.append('INCOMPLETE_OWNED_RESOURCE_OR_POWER_EVIDENCE')
        pool_source = next(s['frozen']['path'] for s in context['execution_sources']
                           if Path(s['original_path']).name == 'runtime-inputs.jsonl')
        pool_rows = list(map(json.loads, Path(pool_source).read_text().splitlines()))
        pool = {r['pool_id']: r for r in pool_rows}
        if len(pool) != len(pool_rows):
            issues.append('DUPLICATE_RUNTIME_POOL_IDENTITIES')
        primers = raw.get('priming_request_records', [])
        for row in primers:
            if not raw['startup_end_ns'] <= row['scheduled_time_ns'] <= row['completion_time_ns'] <= raw['phase_start_ns']:
                issues.append('PRIMING_OUTSIDE_DECLARED_PREMEASUREMENT_PHASE')
            boundaries = [row.get('response_collector_processing_time_ns'),
                          row.get('previous_handler_next_request_ready', {}).get('collector_processing_time_ns')]
            if any(not stamp(t) or t > raw['phase_start_ns'] for t in boundaries):
                issues.append('PRIMING_READER_BOUNDARIES_NOT_COMPLETE_BEFORE_MEASUREMENT')
        for alias, family in RESIDENTS.items():
            binding = context['worker_bindings'][family]
            spec = json.loads(Path(binding['spec']).read_text())
            ready = raw['ready'][alias]
            if (ready.get('spec_sha256') != receipt(binding['spec'])['sha256']
                    or ready.get('clock') != 'time.perf_counter_ns'
                    or ready.get('worker') != spec['candidate']['slot']
                    or not set(plan['aliases'][alias]['supported_tasks']) <= set(ready.get('tasks', []))):
                issues.append('READY_SPEC_TASK_OR_CLOCK_IDENTITY_FAILED:' + alias)
            adapter = None if family == 'ocr' else load_cpu_adapter(binding, spec)
            rows = [r for r in primers + raw['request_records'] if r['model'] == alias]
            previous_end = previous_ready = None
            anchors = []
            for row in rows:
                found, paired = validate_request(row, ready['pid'])
                issues += found
                anchors += paired
                issues += validate_output(row, pool[row['pool_id']], spec, adapter)
                response = row['worker_response']
                begin, end = response.get('start_perf_ns'), response.get('end_perf_ns')
                if previous_end is not None and (begin < previous_end or begin < previous_ready):
                    issues.append('OWNER_FIFO_OR_NEXT_READY_ORDER_VIOLATION:' + row['request_id'])
                previous_end = end
                previous_ready = row.get('previous_handler_next_request_ready', {}).get('perf_ns', end)
            owners[alias] = dict(pid=ready['pid'], requests=len(rows), phase_anchors=len(anchors),
                                measured_requests=sum(r['phase'] == 'measurement' for r in rows),
                                GPU_cursor='UNKNOWN', kernel_evidence='UNKNOWN_UNPROFILED_DOMAIN')
        catalog = json.loads(Path(plan['catalog']['path']).read_text())
        totals = recompute_quality(raw, catalog)
        quality = scored.get('quality_evidence', {})
        expected_sources = [plan['catalog'], receipt(collector.__file__), receipt(ROOT / 'gsdeploy/tasks.py'),
                            receipt(ROOT / 'gsdeploy/quality.py')]
        if (quality.get('kind') != 'measured_per_request' or quality.get('errors') != []
                or quality.get('totals_by_request') != totals or quality.get('source_receipts') != expected_sources):
            issues.append('QUALITY_EVIDENCE_DIFFERS_FROM_RECOMPUTED_TRAINING_OUTPUTS')
        diagnostics = score_flow({**raw, 'quality_evidence': dict(totals_by_request=totals)}, plan)
        if not diagnostics['utility_complete'] or diagnostics['completion_ratio'] != 1:
            issues.append('INCOMPLETE_ORIGINAL_SIX_TERM_DIAGNOSTICS')
        issues += diagnostics['unknown_or_ineligible_reasons']
        if diagnostics['memory']['budget_satisfied'] is not True:
            issues.append('ORIGINAL_SAFE_MEMORY_BUDGET_EXCEEDED')
        collector.verify_sources(plan)
        if any(receipt(paths[key]) != ref for key, ref in evidence.items()):
            issues.append('EVIDENCE_CHANGED_DURING_REVIEW')
    except Exception as exc:
        issues.append('REVIEW_INCOMPLETE:' + type(exc).__name__ + ': ' + str(exc))
    sources = [Path(__file__), Path(collector.__file__), Path(__file__).with_name('smoke_verifier_v3.py'),
        Path(__file__).with_name('known_input_observed_v5.py'), Path(__file__).with_name('flow_metrics.py'),
        Path(__file__).with_name('operator_profile_campaign_v2.py'), ROOT / 'gsdeploy/tasks.py', ROOT / 'gsdeploy/quality.py']
    result = dict(schema=SCHEMA, status='PASS' if not issues else 'FAIL', issues=sorted(set(issues)),
        evidence=evidence, evaluator_sources=[receipt(p) for p in sources], owners=owners,
        request_timings=timings, six_term_diagnostics=diagnostics,
        training_only=True, validation_started=False, model_fitted=False, GPU_started=False,
        original_stability_evaluated=False, independent_prediction_accuracy='UNKNOWN',
        deployment_gain='UNKNOWN', same_sm_concurrency='UNKNOWN', owned_scheduler_attribution='UNKNOWN',
        capture_run_id='raw-sha256:' + evidence['raw']['sha256'] if 'raw' in evidence else None,
        interpretation='PASS qualifies this complete predeclared training capture only. Six terms are diagnostic; no sustained stability, independent accuracy or deployment-gain qualification.')
    save(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run', required=True, help='Supervised execution root, containing collection/raw.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(args.plan, args.run, args.output)
    print(json.dumps(dict(status=result['status'], issues=result['issues'], report=str(Path(args.output).resolve() / 'report.json'))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
