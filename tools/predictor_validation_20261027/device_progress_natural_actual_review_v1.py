"""Read-only CPU provenance and exact-work review of an existing natural smoke."""
import argparse
from collections import Counter
import copy
import io
import json
from pathlib import Path
import unittest

from .native_collect import context_identity, receipt, save
from .profiling import RESIDENTS
from .paired_profile_campaign_v4 import audit
from .smoke_verifier_v3 import validate_request, validate_telemetry
from .device_progress_smoke_raw_verify_v2 import checked, FROZEN, validate_request as validate_device_request


def review(run, raw_review, output):
    run=Path(run).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    inputs={name:receipt(run/name) for name in ('plan.json','context.json','raw.json','executed-jobs.json','attempt.json','smoke-report.json')}
    values={name:json.loads(Path(ref['path']).read_text()) for name,ref in inputs.items()}
    plan=values['plan.json'];context=values['context.json'];raw=values['raw.json'];executed=values['executed-jobs.json'];strict=values['smoke-report.json']
    issues=[];checks={}
    def require(condition,name):
        checks[name]=bool(condition)
        if not condition:issues.append(name)
    # Audit reads source/input receipts and pairing documents, never executes.
    require(audit(run)==plan,'original_campaign_CPU_source_and_pair_audit')
    require(plan['split']=='smoke' and plan['mode']=='natural' and not plan['fit_eligible'] and not plan['final_gain_eligible'],'natural_smoke_only')
    require(context_identity(context)==raw['execution_context_sha256']==plan['execution_context_sha256'],'context_identity')
    require(all(raw[key]==inputs[name] for key,name in [('plan','plan.json'),('context','context.json'),('executed_jobs','executed-jobs.json')]),'raw_input_receipts')
    require(executed==dict(split=None,contexts=None,limit=None,trace=False,jobs=plan['jobs']),'unfiltered_exact_natural_jobs')
    reviewed=checked(receipt(raw_review));require(reviewed['raw']==inputs['raw.json'] and reviewed['issues']==[] and reviewed['status']=='RAW_STRUCTURAL_CONTRACT_ACCEPTED_DEVICE_API_NOT_AUTHENTICATED','v2_output_and_device_contract')
    require(strict['status']=='PASS' and strict['issues']==[] and strict['raw']==inputs['raw.json'] and strict['plan']==inputs['plan.json'],'strict_campaign_report')
    execution_ref=strict['execution'];execution=checked(execution_ref)
    execution_binding=execution['profile_execution'];manifest_ref=execution_binding['manifest'];manifest=checked(manifest_ref);binding=manifest['binding']
    require(execution_binding['binding']==binding and binding['kind']=='paired' and binding['target']==str(run) and binding['mode']=='natural' and binding['plan']==inputs['plan.json'] and binding['context']==inputs['context.json'],'actual_supervisor_exact_target_binding')
    command=[binding['python'],'-m','tools.predictor_validation_20261027.profile_training_supervisor_v1','_child','--manifest',manifest_ref['path'],'--execute']
    require(execution['command']==command and execution['collection_subprocess_launched'] is True and execution['watchdog']['return_code']==0,'actual_supervisor_command_and_child_success')
    require(execution['child_reaped'] is True and execution['all_owned_descendants_exited_and_reaped'] is True and execution['experiment_lock_held_through_group_exit'] is True and execution['live_group_pids']==[] and execution['final_source_audit']=='PASS' and execution['watchdog']['cleanup']['live_survivors']==[],'supervisor_lock_source_exit_boundary')
    require(all(receipt(ref['path'])==ref for ref in manifest['sources']),'manifest_all_source_receipts')
    starts=Path(execution_ref['path']).with_name('execution-start.json');start=json.loads(starts.read_text())
    require(start['profile_execution']==execution_binding and start['command']==command and start['started_unix_ns']==execution['started_unix_ns'],'prelaunch_to_final_execution_binding')
    require(context['residents']==RESIDENTS and plan['resident_aliases']==RESIDENTS and context['execution']['physical_owners']==4 and context['execution']['batch_size']==1 and context['execution']['weight_reuse'] is False,'original_four_physical_owners')
    snapshots={Path(s['original_path']).name:s for s in context['execution_sources']}
    require(all(snapshots[name]['frozen']['sha256']==sha and receipt(snapshots[name]['original_path'])['sha256']==sha for name,sha in FROZEN.items()),'audited_observer_worker_exact_frozen_bytes')
    expected=Counter((j['id'],a,p,j['tasks'][a],j['split'],j['validation_view']) for j in plan['jobs'] for a,p in j['inputs'].items())
    observed=Counter((r['profile_job_id'],r['model'],r['pool_id'],r['task'],r['split'],r['validation_view']) for r in raw['request_records'])
    require(expected==observed and sum(expected.values())==len(raw['request_records'])==4,'exact_measured_request_multiset')
    batches=raw['batches'];require(Counter(b['job_id'] for b in batches)==Counter(j['id'] for j in plan['jobs']),'exact_batch_multiset')
    for job in plan['jobs']:
        batch=next(b for b in batches if b['job_id']==job['id']);rows=[r for r in raw['request_records'] if r['profile_job_id']==job['id']]
        require(Counter(batch['request_ids'])==Counter(r['request_id'] for r in rows),'batch_request_identity:'+job['id'])
        require(all(r['scheduled_time_ns']==batch['start_ns']+round(job['start_offsets_ms'][r['model']]*1e6) for r in rows),'exact_planned_relative_arrivals:'+job['id'])
    priming=[]
    for owner,family in RESIDENTS.items():
        tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
        selected=[x for task in tasks for x in sorted((x for x in plan['inputs'] if x['task']==task),key=lambda x:x['input_size'],reverse=True)[:3]]
        priming.extend((owner,x['pool_id'],x['task']) for x in selected)
    require(priming==[(r['model'],r['pool_id'],r['task']) for r in raw['priming_request_records']],'original_priming_exact_selection_and_order')
    host_issues=[]
    for row in raw['priming_request_records']+raw['request_records']:
        phase_row=copy.deepcopy(row);phase_row['execution_events']=[e for e in phase_row['execution_events'] if e.get('event')!='observation']
        found,_=validate_request(phase_row,raw['ready'][row['model']]['pid']);host_issues.extend(found)
        found,_=validate_device_request(row,raw['ready'][row['model']],reviewed['output_protocol_review']['policies'].get(row['model']));host_issues.extend(found)
    require(not host_issues,'original_request_PID_schedule_host_transport_checks');issues.extend(host_issues)
    telemetry=validate_telemetry(raw);require(not telemetry,'all_owned_PID_board_telemetry_envelope');issues.extend(telemetry)
    launches={}
    for owner,family in RESIDENTS.items():
        entry=copy.deepcopy(context['worker_bindings'][family]);entry['command'][1]=str(Path(__file__).with_name(context['worker_variant']).resolve())
        entry['env'].update(context['threads']);entry['env']['PYTHONUNBUFFERED']='1';launches[owner]=entry
    require(all(Path(e['command'][1]).name=='semantic_progress_worker_v1.py' and 'nsys' not in e['command'] for e in launches.values()),'source_bound_collector_worker_substitution_natural')
    require(strict['traces']=={} and strict['natural_kernel_activity']=='UNMEASURED' and context['profiler']=='none-natural-timing','no_independent_natural_CUDA_API_authentication_claim')
    modules=['tests.test_device_progress_smoke_raw_verify_v2','tests.test_device_progress_smoke_raw_verify_v1','tests.test_device_progress_observer_v1_independent_review','tests.test_device_progress_observer_v1','tests.test_smoke_verifier_v3']
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    require(result.wasSuccessful() and result.testsRun==34,'34_CPU_regressions')
    import torch
    require(not torch.cuda.is_initialized(),'CUDA_remained_uninitialized')
    for ref in [*inputs.values(),execution_ref,manifest_ref,receipt(raw_review)]:
        require(receipt(ref['path'])==ref,'unchanged_receipt:'+ref['path'])
    report=dict(schema='gsdeploy.device_progress_natural_actual_CPU_review.v1',status='PASS_LIMITED_PROVENANCE_AND_EXACT_WORK' if not issues else 'FAIL',issues=issues,checks=checks,
        inputs=inputs,execution=execution_ref,manifest=manifest_ref,execution_start=receipt(starts),raw_contract_review=receipt(raw_review),
        source_bound_expected_worker_launches=launches,launch_evidence='Frozen audited collector substitutes context.worker_variant at cmd[1]; historical /proc command lines were not retained by this raw schema.',
        exact_work=dict(measured_requests=len(raw['request_records']),priming_requests=len(priming),measured_forward_records=sum(x.get('forward_records',0) for x in reviewed['requests']),measured_queries=sum(x.get('queries',0) for x in reviewed['requests']),original_CPU_reparse_requests=len(reviewed['output_protocol_review']['requests'])),
        tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),CUDA_initialized=False),reviewer_source=receipt(__file__),
        test_sources=[receipt(Path(__file__).resolve().parents[2]/(module.replace('.','/')+'.py')) for module in modules],
        limits=['Natural collection has no independent native CUDA API trace. Source/launch receipts and raw consistency do not independently authenticate each actual CUDA Event call.',
            'Event completion applies to its recorded stream; global request completion, other streams, exact device completion timestamp, unfinished fraction and SM service remain unqualified.',
            'OCR has original READY/output/exit coverage but no device event probes. Legal explicit no-answer is protocol validity only; task quality is not evaluated.',
            'No paired equality, overhead, speedup, training/predictor qualification or validation result is asserted.'],GPU_started_by_review=False,validation_results_read=False)
    out.mkdir(exist_ok=False);save(out/'report.json',report);(out/'tests.txt').write_text(stream.getvalue());return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--raw-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(review(args.run,args.raw_review,args.output)))
