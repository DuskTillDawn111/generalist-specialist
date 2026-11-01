"""CPU review of paired retained semantic timings, with no GPU trace claims."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .native_collect import receipt, save
from .operator_profile_campaign_v2 import audit
from .profiling import RESIDENTS
from .smoke_verifier_v3 import (check_receipt, load_cpu_adapter, validate_coverage,
                                 validate_output, validate_request, validate_telemetry)


def review(run, structure_review, output):
    run, structure_review, output = Path(run).resolve(), Path(structure_review).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    issues, paired = [], []
    try:
        audit(run)
        source = json.loads(structure_review.read_text())
        if source.get('schema') != 'gsdeploy.operator_kernel_smoke_review.v2' or source.get('status') != 'PASS':
            raise ValueError('Passing structure-to-kernel review required')
        if not check_receipt(source['plan']) or not check_receipt(source['raw']) or not check_receipt(source['strict_runtime_clock_report']):
            raise ValueError('Structure evidence changed')
        original_plan = json.loads(Path(source['plan']['path']).read_text())
        original_raw = json.loads(Path(source['raw']['path']).read_text())
        plan = json.loads((run/'plan.json').read_text())
        context = json.loads((run/'context.json').read_text())
        raw = json.loads((run/'raw.json').read_text())
        executed = json.loads((run/'executed-jobs.json').read_text())
        if plan['jobs'] != original_plan['jobs'] or plan.get('paired_jobs_sha256') != original_plan.get('paired_jobs_sha256'):
            issues.append('PAIRED_INPUT_OR_SCHEDULE_PLAN_MISMATCH')
        if (context.get('worker_variant') != 'semantic_worker.py' or context.get('affine_observation') is not False
                or raw.get('performance_scope') != 'unprofiled_with_phase_markers' or executed.get('trace') is not False
                or list(run.glob('trace-*.nsys-rep'))):
            issues.append('UNPROFILED_SEMANTIC_WORKER_REQUIRED')
        if raw.get('status') != 'completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors'):
            issues.append('COLLECTION_ERROR')
        if raw.get('execution_context_sha256') != plan['execution_context_sha256']:
            issues.append('EXECUTION_CONTEXT_SHA_MISMATCH')
        for field, filename in [('plan','plan.json'), ('context','context.json'), ('executed_jobs','executed-jobs.json')]:
            if not check_receipt(raw.get(field), run/filename):
                issues.append('RAW_RECEIPT_INVALID:'+field)
        # The shared coverage routine also checks a traced precondition. Here
        # the opposite precondition was checked above; reuse only its coverage.
        issues += validate_coverage(plan, raw, {**executed, 'trace':True})
        issues += validate_telemetry(raw)
        if set(raw.get('ready', {})) != set(RESIDENTS) or len({v.get('pid') for v in raw.get('ready', {}).values()}) != 4:
            issues.append('FOUR_DISTINCT_OWNER_PIDS_REQUIRED')
        if set(raw.get('shutdown', {})) != set(RESIDENTS) or any(s.get('exit_code') != 0 or s.get('forced') is not False or s.get('bye_received') is not True for s in raw.get('shutdown', {}).values()):
            issues.append('OWNED_SHUTDOWN_FAILED')
        pool_source = next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        pool = {r['pool_id']:r for r in map(json.loads, Path(pool_source).read_text().splitlines())}
        priming_keys = lambda records: Counter((r['model'],r['task'],r['pool_id']) for r in records)
        if priming_keys(raw.get('priming_request_records', [])) != priming_keys(original_raw.get('priming_request_records', [])):
            issues.append('PRIMING_INPUTS_NOT_MATCHED')
        for alias, family in RESIDENTS.items():
            binding = context['worker_bindings'][family]
            spec = json.loads(Path(binding['spec']).read_text())
            if raw['ready'][alias].get('spec_sha256') != receipt(binding['spec'])['sha256']:
                issues.append('READY_SPEC_SHA_MISMATCH:'+alias)
            adapter = None if family == 'ocr' else load_cpu_adapter(binding, spec)
            rows = [r for r in raw['request_records'] if r['model'] == alias]
            for row in raw['priming_request_records']+rows:
                if row['model'] == alias:
                    issues += validate_output(row,pool[row['pool_id']],spec,adapter)
            for row in rows:
                found, _ = validate_request(row,raw['ready'][alias]['pid'])
                issues += found
                response = row['worker_response']
                if 'affine_operator_observations' in response.get('semantic_forward_observations',{}):
                    issues.append('AFFINE_HOOKS_PRESENT_IN_NATURAL_RUN')
                matches = [o for o in original_raw['request_records'] if (o['profile_job_id'],o['model'],o['pool_id']) == (row['profile_job_id'],row['model'],row['pool_id'])]
                if len(matches) != 1:
                    issues.append('EXACT_PAIRED_REQUEST_MISSING')
                    continue
                previous = matches[0]['worker_response']
                names = ['raw_output','prediction','finish_reason','input_tokens','generated_tokens','effective_generation','request_sha256','protocol_sha256']
                same = all(previous.get(name) == response.get(name) for name in names)
                if not same:
                    issues.append('PAIRED_OUTPUT_OR_ACTUAL_TOKEN_WORK_MISMATCH:'+alias)
                paired.append(dict(alias=alias,request_id=row['request_id'],pool_id=row['pool_id'],actual_work_matched=same,
                    natural_handler_ms=response['worker_latency_ms'],natural_service_ms=response['service_ms'],
                    profiled_operator_handler_ms=previous['worker_latency_ms'],
                    descriptive_handler_ratio=previous['worker_latency_ms']/response['worker_latency_ms'],
                    natural_e2e_ms=(row['completion_time_ns']-row['scheduled_time_ns'])/1e6,
                    natural_completion_to_next_ready_ms=(row['previous_handler_next_request_ready']['perf_ns']-response['end_perf_ns'])/1e6,
                    input_tokens=response.get('input_tokens'),generated_tokens=response.get('generated_tokens')))
    except Exception as exc:
        issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    result = dict(schema='gsdeploy.matched_natural_smoke_review.v2',status='PASS' if not issues else 'FAIL',
        issues=issues,paired_requests=paired,raw=receipt(run/'raw.json'),plan=receipt(run/'plan.json'),
        structure_review=receipt(structure_review),evaluator=receipt(__file__),
        timing_scope='unprofiled_with_semantic_phase_markers_no_affine_hooks',
        caveat='One short request per owner; ratios are descriptive overhead observations, not a transferable correction factor or P95 estimate.',
        task_quality='UNKNOWN',model_fitted=False,validation_started=False,deployment_gain_qualified=False)
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--structure-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.run,args.structure_review,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
