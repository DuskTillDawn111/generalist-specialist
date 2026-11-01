"""Independent strict host lifecycle/startup review; CUDA mapping is UNKNOWN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .native_collect import receipt,save
from .profiling import RESIDENTS
from .small_training_patch_campaign_v4 import audit
from .training_collection_review_v3 import coverage
from .smoke_verifier_v3 import check_receipt,load_cpu_adapter,validate_output,validate_request
from .startup_telemetry_review_v4_1 import validate_startup_telemetry


def review(run,output):
    run,output=Path(run).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    issues=[];owners={}
    try:
        audit(run)
        plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
        raw=json.loads((run/'raw.json').read_text());executed=json.loads((run/'executed-jobs.json').read_text())
        if raw.get('status')!='completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors'):
            issues.append('COLLECTION_ERRORS_PRESENT')
        if raw.get('execution_context_sha256')!=plan['execution_context_sha256']:issues.append('EXECUTION_CONTEXT_MISMATCH')
        for name,path in [('raw_plan','plan.json'),('raw_context','context.json'),('raw_executed_jobs','executed-jobs.json')]:
            field=name.removeprefix('raw_')
            if not check_receipt(raw.get(field),run/path):issues.append('RAW_INPUT_RECEIPT_FAILED:'+field)
        issues+=coverage(plan,raw,executed);issues+=validate_startup_telemetry(raw)
        if set(raw.get('ready',{}))!=set(RESIDENTS) or len({v.get('pid') for v in raw['ready'].values()})!=4:
            issues.append('FOUR_DISTINCT_OWNED_READY_PIDS_REQUIRED')
        if set(raw.get('shutdown',{}))!=set(RESIDENTS) or any(s.get('exit_code')!=0 or s.get('forced') is not False or s.get('bye_received') is not True for s in raw.get('shutdown',{}).values()):
            issues.append('CLEAN_OWNED_SHUTDOWN_REQUIRED')
        frozen_pool=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        pool={r['pool_id']:r for r in map(json.loads,Path(frozen_pool).read_text().splitlines())}
        for alias,family in RESIDENTS.items():
            binding=context['worker_bindings'][family];spec=json.loads(Path(binding['spec']).read_text())
            if raw['ready'][alias].get('spec_sha256')!=receipt(binding['spec'])['sha256']:issues.append('READY_SPEC_IDENTITY_FAILED:'+alias)
            adapter=None if family=='ocr' else load_cpu_adapter(binding,spec)
            for row in raw.get('priming_request_records',[])+raw['request_records']:
                if row['model']==alias:issues+=validate_output(row,pool[row['pool_id']],spec,adapter)
            rows=[r for r in raw['request_records'] if r['model']==alias]
            for row in rows:
                found,_=validate_request(row,raw['ready'][alias]['pid']);issues+=found
            owners[alias]=dict(request_ids=[r['request_id'] for r in rows],kernel_evidence='UNKNOWN_NOT_REVIEWED_IN_HOST_STARTUP_SCOPE')
    except Exception as exc:issues.append('HOST_STARTUP_REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    result=dict(schema='gsdeploy.host_startup_collection_strict_review.v4',status='PASS' if not issues else 'FAIL',issues=issues,
        review_scope='HOST_STARTUP_LIFECYCLE_AND_NATIVE_WORK_ONLY',kernel_mapping_qualified=False,
        cross_process_CUDA_clock_qualified=False,owners=owners,raw=receipt(run/'raw.json'),
        plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
        capture_run_id='raw-sha256:'+receipt(run/'raw.json')['sha256'],independent_capture_count=1,
        minimum_training_capture_requirement_satisfied=False,training_only=True,validation_started=False,
        evaluator_sources=[receipt(__file__),receipt(Path(__file__).with_name('startup_telemetry_review_v4_1.py')),
                           receipt(Path(__file__).with_name('smoke_verifier_v3.py'))],
        interpretation='Host startup telemetry/output protocol may be used within its actual profiler domain; no kernel target, natural transfer, instantaneous peak, or deployment accuracy is qualified.')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.run,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],report=receipt(Path(args.output)/'report.json'))))


if __name__=='__main__':main()
