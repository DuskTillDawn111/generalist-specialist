"""Strict startup-aware CPU review for small training patches.

One canonical parsed trace is retained per owner. No smoke fitting, validation
execution, host-gap scaling, or profiling-domain transfer is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .small_training_patch_campaign_v4 import audit
from .native_collect import receipt, save
from .operator_smoke_review_v2 import validate_operator_rows
from .profiling import RESIDENTS
from .smoke_verifier_v3 import (check_receipt, load_cpu_adapter, validate_output,
                                 validate_request, validate_telemetry, validate_parsed)
from .trace_export import export_sqlite, parse_sqlite
from .startup_telemetry_review_v4 import validate_startup_telemetry


def coverage(plan,raw,executed):
    issues=[]
    if executed.get('jobs')!=plan['jobs']:issues.append('EXECUTED_PLAN_JOBS_CHANGED')
    if any(j['split']!='training' for j in plan['jobs']):issues.append('TRAINING_ONLY_PLAN_REQUIRED')
    key=lambda r:(r.get('profile_job_id'),r.get('model'),r.get('pool_id'),r.get('task'),r.get('split'))
    expected=Counter((j['id'],a,p,j['tasks'][a],'training') for j in plan['jobs'] for a,p in j['inputs'].items())
    observed=Counter(map(key,raw.get('request_records',[])))
    if expected!=observed:issues.append('EXACT_TRAINING_REQUEST_COVERAGE_FAILED')
    rids=[r.get('request_id') for r in raw.get('request_records',[])]
    if len(set(rids))!=len(rids) or any(not isinstance(r,str) or not r for r in rids):issues.append('INVALID_REQUEST_IDS')
    if Counter(b.get('job_id') for b in raw.get('batches',[]))!=Counter(j['id'] for j in plan['jobs']):
        issues.append('EXACT_TRAINING_JOB_COVERAGE_FAILED')
    if Counter(r for b in raw.get('batches',[]) for r in b.get('request_ids',[]))!=Counter(rids):
        issues.append('EXACT_BATCH_REQUEST_COVERAGE_FAILED')
    for b in raw.get('batches',[]):
        if Counter(b.get('request_ids',[]))!=Counter(r['request_id'] for r in raw['request_records'] if r['profile_job_id']==b['job_id']):
            issues.append('BATCH_JOB_REQUEST_BINDING_FAILED')
    return issues


def review(run,output):
    run,output=Path(run).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    issues,owners=[],{}
    try:
        audit(run)
        plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
        raw=json.loads((run/'raw.json').read_text());executed=json.loads((run/'executed-jobs.json').read_text())
        if raw.get('status')!='completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors'):
            issues.append('COLLECTION_ERRORS_PRESENT')
        if raw.get('execution_context_sha256')!=plan['execution_context_sha256']:issues.append('EXECUTION_CONTEXT_MISMATCH')
        for field,name in [('plan','plan.json'),('context','context.json'),('executed_jobs','executed-jobs.json')]:
            if not check_receipt(raw.get(field),run/name):issues.append('RAW_INPUT_RECEIPT_FAILED:'+field)
        issues+=coverage(plan,raw,executed);issues+=validate_startup_telemetry(raw)
        if set(raw.get('ready',{}))!=set(RESIDENTS) or len({r.get('pid') for r in raw.get('ready',{}).values()})!=4:
            issues.append('FOUR_DISTINCT_OWNED_READY_PIDS_REQUIRED')
        if set(raw.get('shutdown',{}))!=set(RESIDENTS) or any(s.get('exit_code')!=0 or s.get('forced') is not False or s.get('bye_received') is not True for s in raw.get('shutdown',{}).values()):
            issues.append('CLEAN_OWNED_SHUTDOWN_REQUIRED')
        traced=plan['domain']!='natural'
        if executed.get('trace') is not traced:issues.append('PROFILER_DOMAIN_MISMATCH')
        if not traced and (list(run.glob('trace-*.nsys-rep')) or raw.get('performance_scope')!='unprofiled_with_phase_markers'):
            issues.append('NATURAL_DOMAIN_PROFILER_PRESENT')
        pool_source=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        pool={r['pool_id']:r for r in map(json.loads,Path(pool_source).read_text().splitlines())}
        for alias,family in RESIDENTS.items():
            binding=context['worker_bindings'][family];spec=json.loads(Path(binding['spec']).read_text())
            pid=raw['ready'][alias]['pid']
            if raw['ready'][alias].get('spec_sha256')!=receipt(binding['spec'])['sha256']:issues.append('READY_SPEC_IDENTITY_FAILED:'+alias)
            adapter=None if family=='ocr' else load_cpu_adapter(binding,spec)
            rows=[r for r in raw['request_records'] if r['model']==alias]
            for row in raw.get('priming_request_records',[])+rows:
                if row['model']==alias:issues+=validate_output(row,pool[row['pool_id']],spec,adapter)
            anchors=[]
            for row in rows:
                found,paired=validate_request(row,pid);issues+=found;anchors+=paired
                if plan['domain']!='metadata' and 'affine_operator_observations' in row['worker_response'].get('semantic_forward_observations',{}):
                    issues.append('AFFINE_HOOKS_PRESENT_OUTSIDE_METADATA_DOMAIN')
            owner=dict(request_ids=[r['request_id'] for r in rows],timing_domain=plan['domain'])
            if traced and not rows:
                owner['kernel_evidence']='NO_MEASURED_REQUESTS_FOR_RESIDENT_OWNER'
            elif traced:
                report=run/('trace-'+alias.replace('#','-')+'.nsys-rep')
                sql=output/('trace-'+alias.replace('#','-')+'.sqlite')
                export=export_sqlite(report,sql)
                parsed=parse_sqlite(sql,anchors)
                found,summary=validate_parsed(parsed,anchors,rows,pid)
                issues+=[s+':'+alias for s in found]
                parsed['sources'].append(receipt(run/'raw.json'))
                parsed_path=output/('parsed-'+alias.replace('#','-')+'.json');save(parsed_path,parsed)
                owner.update(export=export,sqlite=receipt(sql),parsed=receipt(parsed_path),kernel_summary=summary)
                if plan['domain']=='metadata' and family!='ocr':
                    mapped=[]
                    for row in rows:
                        doc=row['worker_response']['semantic_forward_observations'].get('affine_operator_observations',{})
                        found,pairs,summary=validate_operator_rows(doc,parsed['phases'],request_id=row['request_id'],pid=pid)
                        issues+=[s+':'+alias+':'+row['request_id'] for s in found]
                        # The canonical trace holds kernels. Keep only compact
                        # operator->kernel identity links in this extra index.
                        mapped.append(dict(request_id=row['request_id'],summary=summary,operators=[
                            dict(operator_id=p['operator_id'],module=p['module'],kernel_target_available=p['kernel_target_available'],
                                 kernel_keys=[dict(pid=k['pid'],correlation_id=k['correlation_id'],nsys_start_ns=k['nsys_start_ns']) for k in p['kernels']])
                            for p in pairs]))
                    path=output/('operator-kernel-index-'+alias.replace('#','-')+'.json');save(path,mapped)
                    owner['operator_kernel_index']=receipt(path)
                elif plan['domain']=='metadata':owner['affine_coverage']='UNKNOWN_OUTSIDE_LANGUAGE_OBSERVER_DOMAIN'
            else:owner['kernel_evidence']='UNKNOWN_UNPROFILED_DOMAIN'
            owners[alias]=owner
    except Exception as exc:issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    raw_receipt=receipt(run/'raw.json')
    result=dict(schema='gsdeploy.startup_training_collection_strict_review.v4',status='PASS' if not issues else 'FAIL',
        issues=issues,owners=owners,raw=receipt(run/'raw.json'),plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
        training_only=True,validation_started=False,model_fitted=False,natural_timing_transfer_validated=False,
        capture_run_id='raw-sha256:'+raw_receipt['sha256'],
        trial_job_identity='profile_job_id identifies within-capture trials; never independent startup/capture runs',
        independent_capture_count=1,minimum_training_capture_requirement_satisfied=False,
        startup_evidence_scope='raw lifecycle startup_begin/end and priming/active boundaries; NVML sampled owned-process/board peaks, no instantaneous-peak claim',
        evaluator_sources=[receipt(__file__),receipt(Path(__file__).with_name('startup_telemetry_review_v4.py')),receipt(Path(__file__).with_name('trace_export.py')),
                           receipt(Path(__file__).with_name('nvtx_interval_index_v2.py'))])
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.run,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
