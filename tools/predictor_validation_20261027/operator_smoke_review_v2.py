"""CPU-only validation of observed affine metadata and its CUPTI association."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .native_collect import receipt, save
from .operator_profile_campaign_v2 import audit, stat_receipt
from .profiling import RESIDENTS
from .smoke_verifier_v3 import check_receipt


def validate_operator_rows(document, phases, *, request_id, pid):
    issues, usable = [], []
    rows = document.get('records', [])
    if (document.get('schema') != 'gsdeploy.affine_operator_observations.v2'
            or document.get('request_id') != request_id or not rows or document.get('modules', 0) <= 0
            or document.get('instrumentation_errors') or document.get('coverage_truncated') is not False
            or document.get('gpu_synchronization_added') is not False
            or document.get('tensor_values_read') is not False or document.get('native_outputs_modified') is not False):
        return ['AFFINE_OBSERVATION_INCOMPLETE'], [], {}
    counts = Counter(row.get('id') for row in rows)
    if any(value != 1 for value in counts.values()):
        issues.append('DUPLICATE_AFFINE_CALL_IDENTITIES')
    phase_map = {}
    for phase in phases:
        if phase.get('request_id') == request_id and str(phase.get('segment_id', '')).startswith('affine_'):
            key = phase.get('segment_id')
            if key in phase_map:
                issues.append('DUPLICATE_AFFINE_CUPTI_RANGE')
            phase_map[key] = phase
    if set(phase_map) != set(counts):
        issues.append('AFFINE_RECORD_PHASE_KEYS_MISMATCH')
    for row in rows:
        try:
            a, b = row['in_features'], row['out_features']
            shape, out = row['input']['shape'], row['output']['shape']
            if (type(a) is not int or type(b) is not int or a <= 0 or b <= 0
                    or not shape or shape[-1] != a or out != shape[:-1]+[b]
                    or any(type(v) is not int or v <= 0 for v in shape)
                    or row.get('operator_kind') != 'linear_declared'
                    or row.get('dense_gemm_flops') != 2*math.prod(shape[:-1])*a*b
                    or not row.get('module') or not row.get('implementation')
                    or not row.get('direct_tensors') or row.get('unsupported_reason')):
                raise ValueError('Invalid declared affine/tensor shape')
            for tensor in [row['input'], row['output']]+row['direct_tensors']:
                if (type(tensor['element_size_bytes']) is not int or tensor['element_size_bytes'] <= 0
                        or tensor['logical_bytes'] != math.prod(tensor['shape'])*tensor['element_size_bytes']
                        or not tensor.get('dtype')):
                    raise ValueError('Invalid logical tensor metadata')
            phase = phase_map[row['id']]
            if (phase.get('pid') != pid or phase.get('kind') != 'gpu'
                    or phase.get('clock_mapping_valid') is not True
                    or phase.get('start_ns') != row['start_ns'] or phase.get('end_ns') != row['end_ns']):
                raise ValueError('Invalid affine host/CUPTI range binding')
            kernels = phase.get('kernels', [])
            if any(k.get('pid') != pid or k.get('start_ns') is None or k.get('end_ns') is None for k in kernels):
                raise ValueError('Invalid affine kernel ownership/clock')
            usable.append(dict(operator_id=row['id'], module=row['module'], metadata=row,
                kernel_count=len(kernels), kernel_duration_sum_ns=phase.get('kernel_duration_sum_ns'),
                kernel_union_busy_ns=phase.get('kernel_union_busy_ns'), kernels=kernels,
                kernel_target_available=bool(kernels),
                unknown_reason=None if kernels else 'NO_KERNEL_LAUNCH_CORRELATED_TO_THIS_CALL',
                demand=None, memory_traffic=None, achieved_occupancy=None))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append('INVALID_AFFINE_METADATA_OR_BINDING:'+str(row.get('id'))+':'+str(exc))
    if not any(row['kernel_target_available'] for row in usable):
        issues.append('NO_AFFINE_CALL_WITH_REAL_CUPTI_KERNEL')
    return issues, usable, dict(records=len(rows), mapped_calls=sum(row['kernel_target_available'] for row in usable),
        matched_kernels=sum(row['kernel_count'] for row in usable),
        implementations=sorted({row.get('implementation', '') for row in rows}),
        coverage='recognized_language_affine_only_not_complete_graph')


def review(run, strict_report, output):
    run, strict_report, output = Path(run).resolve(), Path(strict_report).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    issues, reports = [], {}
    try:
        audit(run)
        strict = json.loads(strict_report.read_text())
        if (strict.get('schema') != 'gsdeploy.kernel_smoke_strict_review.v3' or strict.get('status') != 'PASS'
                or not check_receipt(strict.get('evidence', {}).get('raw'), run/'raw.json')
                or not check_receipt(strict.get('evidence', {}).get('plan'), run/'plan.json')
                or not check_receipt(strict.get('evidence', {}).get('context'), run/'context.json')):
            raise ValueError('Matching strict runtime/clock report required')
        raw = json.loads((run/'raw.json').read_text())
        context = json.loads((run/'context.json').read_text())
        if context.get('worker_variant') != 'operator_worker_v2.py' or context.get('affine_observation') is not True:
            raise ValueError('Explicit operator worker identity required')
        for reference in strict['evaluator_sources']:
            if not check_receipt(reference):
                raise ValueError('Strict evaluator source receipt changed')
        for alias in RESIDENTS:
            trace = strict['traces'][alias]
            if any(not check_receipt(trace[name]) for name in ('sqlite', 'nsys', 'parsed')):
                raise ValueError('Strict trace dependency changed')
            rows = [r for r in raw['request_records'] if r['model'] == alias]
            if len(rows) != 1:
                raise ValueError('One isolated measured request per original owner required')
            row = rows[0]
            if alias == 'ocr#0':
                reports[alias] = dict(status='UNKNOWN_OUTSIDE_LANGUAGE_AFFINE_OBSERVER_COVERAGE',
                                     request_id=row['request_id'], kernel_trace=trace['parsed'])
                continue
            semantic = row['worker_response']['semantic_forward_observations']
            document = semantic.get('affine_operator_observations', {})
            parsed = json.loads(Path(trace['parsed']['path']).read_text())
            found, pairs, summary = validate_operator_rows(document, parsed['phases'],
                request_id=row['request_id'], pid=raw['ready'][alias]['pid'])
            issues += [item+':'+alias for item in found]
            path = output/('affine-kernel-pairs-'+alias.replace('#','-')+'.json')
            save(path, dict(request_id=row['request_id'], pairs=pairs, source=trace['parsed'],
                            metadata_source=receipt(run/'raw.json'), claim='ISOLATED_SMOKE_NOT_TRAINING_DATA'))
            reports[alias] = dict(status='PASS' if not found else 'FAIL', summary=summary, pairs=receipt(path))
    except Exception as exc:
        issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    result = dict(schema='gsdeploy.operator_kernel_smoke_review.v2', status='PASS' if not issues else 'FAIL',
        issues=issues, owners=reports, strict_runtime_clock_report=receipt(strict_report),
        plan=receipt(run/'plan.json'), raw=receipt(run/'raw.json'), evaluator=receipt(__file__),
        claim='REAL_AFFINE_METADATA_TO_CUPTI_ASSOCIATION_ONLY_NOT_TRAINED_PREDICTOR',
        profiler_perturbed=True, logical_bytes_are_not_measured_traffic=True,
        task_quality='UNKNOWN', deployment_gain_qualified=False, validation_started=False)
    save(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--strict-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(args.run, args.strict_report, args.output)
    print(json.dumps(dict(status=result['status'], issues=result['issues'], report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
