"""Read only prospective suite inputs; never inspect runtime paths or outcomes.

This audit can run during collection. It certifies frozen input consistency,
not an execution's success or startup chronology, which need a later report.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time

from .flow_capacity_catalog import condition
from .flow_capacity_transfer import _generation
from .flow_capacity_qualification import verify_protocol
from .flow_trained_validation import check_sealed
from .flow_validation import validate_plan, _sources
from .native_flow import checked
from .probe_validation import read, receipt, save
from .profiling import digest


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_structure(plan, sealed, suite, qualification, suite_path):
    """Pure complete-domain identity checks, independent of filesystem reads."""
    trace = {r['request_id']: r for r in plan['trace']}; candidates = {c['id']: c for c in plan['candidates']}
    if not trace or len(trace) != len(plan['trace']):
        raise ValueError('Nonempty unique public request cohort required')
    if plan.get('phase') != 'predictor_validation' or plan.get('kind') == 'measured':
        raise ValueError('Prospective independent predictor validation plan required')
    if sealed.get('cohort_sha256') != digest(plan['trace']):
        raise ValueError('Prediction cohort hash differs from declared input trace')
    predictions = sealed['candidates']
    if len(predictions) != len(candidates) or {p['candidate_id'] for p in predictions} != set(candidates):
        raise ValueError('Exactly one sealed prediction for each declared candidate required')
    for prediction in predictions:
        candidate = candidates[prediction['candidate_id']]; rows = prediction['requests']; ids = [r['request_id'] for r in rows]
        if len(ids) != len(trace) or set(ids) != set(trace) or len(ids) != len(set(ids)):
            raise ValueError('Every candidate must predict the complete unique offered cohort')
        metric = prediction.get('metrics')
        if (prediction.get('unknown_reason') or not metric or not metric.get('six_term_comparison_complete') or
                not _number(metric.get('cost')) or prediction.get('comparison_U') != -metric['cost'] or
                metric.get('cohort_size') != len(trace) or set(metric.get('cohort_ids', [])) != set(trace) or
                len(metric.get('cohort_ids', [])) != len(trace) or metric.get('incomplete_ids')):
            raise ValueError('Complete comparison metrics for the full cohort required')
        for row in rows:
            expected = trace[row['request_id']]
            if (row['task'], row['pool_id'], row['alias']) != (expected['task'], expected['pool_id'], candidate['routes'][row['request_id']]):
                raise ValueError('Predicted request alias/task/input differs from frozen plan')
            if any(not _number(row.get(k)) for k in ('service_ms', 'wait_ms', 'e2e_ms', 'finish_ns')):
                raise ValueError('Finite complete per-request timing required')
            if (abs(row['e2e_ms'] - row['service_ms'] - row['wait_ms']) > 2e-6 or
                    abs(row['finish_ns'] - expected['offset_ns'] - row['e2e_ms'] * 1e6) > 2):
                raise ValueError('Prediction timing arithmetic differs from public arrival')
        _generation(plan, condition(plan, candidate))
    jobs = suite['jobs']; ids = [j['candidate_id'] for j in jobs]
    if len(ids) != len(candidates) or set(ids) != set(candidates) or len(set(ids)) != len(ids):
        raise ValueError('Exactly one suite job for every declared candidate required')
    outputs = [str(Path(j['output']).resolve()) for j in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Unique exclusive candidate outputs required')
    root = Path(suite_path).resolve().parent
    if any(Path(j['output']).resolve() != root / 'runs' / j['candidate_id'] for j in jobs):
        raise ValueError('Suite output location differs from frozen generator contract')
    expected_order = [c['id'] for c in plan['candidates']]
    random.Random('independent-flow-validation-order:' + str(plan['seed'])).shuffle(expected_order)
    if ids != expected_order:
        raise ValueError('Suite job order differs from deterministic prospective generator')
    if suite.get('no_retries') is not True or suite.get('no_seed_replacement') is not True:
        raise ValueError('Explicit no retries and no seed replacement required')
    if (qualification['target_plan'] != suite['sources']['plan'] or
            qualification['target_seed'] != plan['seed']):
        raise ValueError('Qualification protocol binds a different target plan or seed')
    if (not 0 < sealed['information_cutoff_ns'] <= sealed['created_ns'] <= suite['created_ns'] or
            not 0 < qualification['created_ns'] <= suite['created_ns']):
        raise ValueError('Predictions and qualification must precede frozen suite preparation')
    return dict(candidate_count=len(candidates), requests_per_candidate=len(trace),
        complete_request_predictions=len(trace) * len(candidates), candidate_order=ids,
        full_cohort_alias_task_pool_verified=True, no_retry_protocol_verified=True,
        deterministic_input_and_suite_generators_verified=True,
        qualification_target_and_pre_suite_time_verified=True)


def audit(suite_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    suite = read(suite_path)
    if suite.get('schema') != 'gsdeploy.independent_native_flow_suite.v1':
        raise ValueError('Explicit independent flow suite required')
    if set(suite['sources']) != {'plan', 'predictions', 'training_audit', 'qualification', 'source'}:
        raise ValueError('Complete suite source bindings required')
    for ref in suite['sources'].values():
        checked(ref)
    if suite['sources']['source'] != receipt(Path(__file__).with_name('flow_validation_run.py')):
        raise ValueError('Suite generator source differs from audited entry')
    folder = Path(suite['sealed_dir']); training_audit = check_sealed(folder)
    if (suite['sources']['predictions'] != receipt(folder / 'predictions.json') or
            suite['sources']['training_audit'] != receipt(folder / 'training-audit.json')):
        raise ValueError('Suite source paths differ from sealed directory')
    sealed = read(suite['sources']['predictions']['path'])
    if sealed['sources']['plan'] != suite['sources']['plan']:
        raise ValueError('Sealed predictions bind a different suite plan')
    plan = validate_plan(read(suite['sources']['plan']['path']))
    qualification = verify_protocol(suite['sources']['qualification']['path'])
    refs = [*suite['sources'].values(), *sealed['sources'].values(),
        *sealed['training_sources'], *sealed['resource_sources'], *sealed['execution_sources'],
        *sealed['quality_sources'], *sealed['predictor_sources'], sealed['tool_source'], sealed['metric_source']]
    for ref in refs:
        checked(ref)
    # Rewalk the complete source arrays embedded in resource/quality artifacts,
    # and ensure the seal did not omit a claimed provider input source.
    for name, source_name in (('resources', 'resource_sources'), ('quality', 'quality_sources')):
        doc = read(sealed['sources'][name]['path'])
        embedded = _sources(doc)
        if embedded != sealed[source_name]:
            raise ValueError('Seal omits or changes embedded ' + name + ' sources')
        for ref in embedded:
            checked(ref)
        if doc.get('split', 'training') != 'training':
            raise ValueError('Training-only ' + name + ' evidence required')
    result = check_structure(plan, sealed, suite, qualification, suite_path)
    refs = list({(r['path'], r['sha256']): r for r in refs}.values())
    value = dict(schema='gsdeploy.independent_flow_prospective_input_audit.v1', status='PASS',
        audited_ns=time.perf_counter_ns(), suite=receipt(suite_path), source=receipt(__file__),
        source_receipts=refs, checks=result, training_verification=training_audit['verification'],
        runtime_paths_opened=False, runtime_outputs_or_logs_read=False, GPU_started_by_audit=False,
        actual_startup_chronology_assessed=False, execution_success_assessed=False,
        physical_capacity_bound_certified=False, final_gain_eligible=False,
        scope='Read-only consistency audit of frozen prospective inputs during execution; final startup ordering, no actual retry and measurements require separate outcome audit')
    save(output, value); return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--suite', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv); value = audit(args.suite, args.output)
    print(value['status'], value['checks']); return 0


if __name__ == '__main__': main()
