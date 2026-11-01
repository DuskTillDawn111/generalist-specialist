"""New stable-context, three-session native training wrapper.

Preparation/audit are CPU-only. Explicit execute reuses the frozen unified
collector's native process-tree profiler, GPU lease, watchdog and cleanup.
Every capture actually runs the declared eleven jobs, including one all-four
job. No source/context field is removed to manufacture equality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import uuid

from gsdeploy.resources import identity
from . import unified_corun_collect_v8_1 as collector
from . import unified_corun_review_v8_1 as native_review
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .kernel_dag_response_replication_plan_v1 import jobs_for
from .kernel_sequence_audit_v2 import verify
from .native_collect import receipt, context_identity
from .profiling import digest
from .operator_profile_campaign_v2 import shared_execution_paths, audit as source_audit
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.kernel_dag_response_replication_producer.v1'
MATCHING_SCHEMA='gsdeploy.twoFIFO_matching_owned_training.v1.continuation.run'
MATCHING_STATUS='COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT'


def need(value,reason):
    if not value:raise ValueError(reason)


def read(ref):
    verify(ref);return json.loads(Path(ref['path']).read_text())


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return receipt(path)


def validate_matrix(plan,bindings):
    expected=jobs_for(bindings)
    need(plan.get('training_only') is True and plan.get('split')=='training'
         and plan.get('domain')==collector.LIGHT and plan.get('no_retries') is True,
         'Explicit no-retry native training plan required')
    need(plan.get('jobs')==expected,'Exact eleven executable jobs required, including same-job all four')
    for owner,binding in bindings.items():
        inputs=[item for item in plan['inputs'] if item['pool_id']==binding['pool_id'] and item['task']==binding['task']]
        need(len(inputs)==1 and inputs[0]['exact_input_sha256']==binding['exact_input_sha256'],
             'Exact source input bytes/task required for every owner')
    return expected


def validate_records(plan,raw,bindings,capture_id):
    """Distinct request instances; fixed work is never confused with instance ID."""
    jobs=validate_matrix(plan,bindings)
    need(raw.get('status')=='completed' and raw.get('execution_context_sha256')==plan['execution_context_sha256'],
         'Completed matching full execution context required')
    rows=raw['request_records'];need(len(rows)==20 and len({r['request_id'] for r in rows})==20,
                                   'Full twenty distinct request instances required')
    expected={(job['id'],owner):(job['inputs'][owner],job['tasks'][owner]) for job in jobs for owner in job['inputs']}
    found={};instances=[]
    for row in rows:
        key=(row['profile_job_id'],row['model'])
        need(key in expected and key not in found and row.get('split')=='training'
             and row.get('validation_view')=='TRAINING_ONLY_NOT_VALIDATION'
             and (row['pool_id'],row['task'])==expected[key], 'Exact job/owner/input instance bijection required')
        found[key]=row['request_id']
        instances.append(dict(instance_id=capture_id+':'+row['request_id'],native_request_id=row['request_id'],
            job_id=key[0],owner=key[1],exact_input_sha256=bindings[key[1]]['exact_input_sha256']))
    need(set(found)==set(expected),'Missing measured owner/job instance')
    batches=raw['batches'];need(len(batches)==11 and len({b['job_id'] for b in batches})==11,
                               'Eleven actual runtime job batches required')
    for batch in batches:
        offered={rid for (job,_),rid in found.items() if job==batch['job_id']}
        need(offered and len(batch['request_ids'])==len(offered) and set(batch['request_ids'])==offered,
             'Actual batch request denominator differs; all-four must execute in one batch')
    return instances


def fresh_source_packet(source_run,output):
    """New context for current planner sources, preserving all old evidence.

    Historical smoke covers unchanged direct dispatch/native/model bindings
    only. New whole-domain native qualification still requires READY/spec and
    the strict review of each newly executed capture.
    """
    source_run=Path(source_run).resolve();output=Path(output).resolve()
    old_plan=receipt(source_run/'plan.json');old_context=receipt(source_run/'context.json')
    plan=read(old_plan);context=read(old_context)
    need(context_identity(context)==plan['execution_context_sha256'],'Historical full source context changed')
    historical=context['shared_native_execution_sources']
    paths={str(p) for p in shared_execution_paths(context['worker_bindings'])}
    need(paths=={r['path'] for r in historical},'Complete historical direct native source cohort differs')
    for ref in historical:verify(ref)
    allowed={str(collector.ROOT/'gsdeploy/milp.py'),str(collector.ROOT/'gsdeploy/optimizer.py')}
    drift=[]
    for item in context['execution_sources']:
        verify(item['frozen']);current=receipt(item['original_path'])
        if current['sha256']!=item['frozen']['sha256']:
            need(item['original_path'] in allowed and item['original_path'] not in paths,
                 'Only declared planner-only drift can enter a new source packet')
            drift.append(dict(original_path=item['original_path'],historical_frozen=item['frozen'],current=current))
    output.mkdir(parents=True,exist_ok=False);sources=[]
    for index,item in enumerate(context['execution_sources']):
        frozen=output/'source-snapshot'/str(index)/Path(item['original_path']).name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(item['original_path'],frozen)
        sources.append(dict(original_path=item['original_path'],frozen=receipt(frozen)))
    proof=dict(schema=SCHEMA+'.source_packet',historical_plan=old_plan,historical_context=old_context,
        unchanged_shared_native_execution_sources=historical,planner_only_drift=drift,
        complete_current_execution_sources=sources,
        smoke_scope='UNCHANGED_DIRECT_DISPATCH_NATIVE_MODEL_BINDINGS_ONLY_NEW_READY_SPEC_AND_REVIEW_REQUIRED',
        historical_full_context_equivalence_claimed=False,whole_domain_native_qualification_inherited=False,
        GPU_started=False,validation_labels_read=False)
    proof_ref=write(output/'source-rebind.json',proof)
    context=dict(context,execution_sources=sources,sources=[s['frozen'] for s in sources],
        response_replication_source_rebind=proof_ref,
        inherited_runtime_smoke_scope=proof['smoke_scope'],historical_full_context_equivalence_claimed=False)
    plan=dict(plan,execution_context_sha256=context_identity(context));plan.pop('plan_sha256',None)
    plan['plan_sha256']=digest(plan)
    write(output/'context.json',context);write(output/'plan.json',plan)
    source_audit(output)
    return output


def prepare(matrix_path,source_run,matching_manifest,output):
    matrix_ref=receipt(matrix_path);matrix=read(matrix_ref)
    need(matrix.get('schema')=='gsdeploy.kernel_dag_response_replication_plan.v1'
         and matrix.get('validation_labels_read') is False and matrix.get('captures')==3,
         'Explicit three-capture training matrix required')
    for ref in matrix['sources']:verify(ref)
    bindings=matrix['exact_owner_inputs'];need(matrix['jobs']==jobs_for(bindings),'Matrix jobs differ')
    read(matrix['parameters'])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    # Exactly one preparation creates one nonce and one source/resource snapshot.
    packet=fresh_source_packet(source_run,output/'current-source-packet')
    seed=collector.prepare(output/'single-context-seed',packet,matrix['parameters']['path'])
    seed_run=Path(seed['runs'][collector.LIGHT]['run'])
    context=json.loads((seed_run/'context.json').read_text());base=json.loads((seed_run/'plan.json').read_text())
    campaign_id=uuid.uuid4().hex
    existing={item['original_path'] for item in context['execution_sources']}
    closure=implementation_sources(__file__,Path(__file__).with_name('kernel_dag_repeated_instance_alignment_v1.py'))
    for index,ref in enumerate(closure):
        if ref['path'] in existing:continue
        frozen=output/'source-snapshot'/str(index)/Path(ref['path']).name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(ref['path'],frozen)
        context['execution_sources'].append(dict(original_path=ref['path'],frozen=receipt(frozen)))
    context['sources']=[item['frozen'] for item in context['execution_sources']]
    context['response_replication_campaign']=dict(schema=SCHEMA,campaign_id=campaign_id,matrix=matrix_ref,
        captures=3,settings_per_capture=11,measured_requests_per_capture=20,
        same_full_context_required=True,initial_offsets_ms=0)
    context_sha=context_identity(context);runs=[]
    for index in range(3):
        capture_id=campaign_id+':capture:'+str(index);run=output/('capture-'+str(index));run.mkdir()
        plan=copy.deepcopy(base);plan.update(execution_context_sha256=context_sha,
            response_replication_capture_id=capture_id,response_replication_campaign_id=campaign_id)
        plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
        validate_matrix(plan,bindings)
        runs.append(dict(capture_id=capture_id,run=str(run),plan=write(run/'plan.json',plan),
                         context=write(run/'context.json',context)))
        collector.audit(run)
    matching_ref=receipt(matching_manifest);matching=read(matching_ref)
    need(matching.get('schema')==MATCHING_SCHEMA.removesuffix('.run'),'Pinned matching producer manifest required')
    result=dict(schema=SCHEMA,status='CPU_PREPARED_NOT_LAUNCHED',campaign_id=campaign_id,matrix=matrix_ref,
        after_matching_manifest=matching_ref,
        execution_context_sha256=context_sha,runs=runs,exact_owner_inputs=bindings,
        source_receipts=strict_references([matrix_ref,matrix,context,closure,matching_ref]),
        captures=3,jobs_per_capture=11,measured_requests_per_capture=20,measured_requests_total=60,
        priming_requests_total=36,observations_per_owner_total=15,
        launch_gate='EXPLICIT_EXECUTE_AFTER_MATCHING_COMPLETION_RECEIPT',
        unused_seed_domains_not_scheduled=True,context_fields_removed=[],GPU_started=False,fit_started=False,
        validation_labels_read=False,numerical_rank_guaranteed=False)
    write(output/'campaign.json',result);return result


def audit(campaign_path):
    campaign=read(receipt(campaign_path))
    need(campaign.get('schema')==SCHEMA and campaign.get('validation_labels_read') is False
         and len(campaign['runs'])==3 and len({r['capture_id'] for r in campaign['runs']})==3,
         'Three distinct explicit capture lifecycles required')
    contexts=[]
    closure=implementation_sources(__file__,Path(__file__).with_name('kernel_dag_repeated_instance_alignment_v1.py'))
    for ref in campaign['source_receipts']:verify(ref)
    for entry in campaign['runs']:
        run=Path(entry['run']).resolve()
        need(receipt(run/'plan.json')==entry['plan'] and receipt(run/'context.json')==entry['context'],
             'Canonical lifecycle run/plan/context paths required')
        plan=read(entry['plan']);context=read(entry['context']);collector.audit(entry['run'])
        frozen={r['original_path']:r['frozen'] for r in context['execution_sources']}
        need(len(frozen)==len(context['execution_sources'])
             and context['sources']==[r['frozen'] for r in context['execution_sources']]
             and all(ref['path'] in frozen and frozen[ref['path']]['sha256']==ref['sha256'] for ref in closure),
             'Complete exact producer/collector/review/adapter module closure required')
        need(context.get('response_replication_campaign')==dict(schema=SCHEMA,campaign_id=campaign['campaign_id'],
             matrix=campaign['matrix'],captures=3,settings_per_capture=11,measured_requests_per_capture=20,
             same_full_context_required=True,initial_offsets_ms=0), 'Stable context campaign/matrix binding changed')
        validate_matrix(plan,campaign['exact_owner_inputs']);contexts.append(context)
        need(plan.get('response_replication_capture_id')==entry['capture_id']
             and plan.get('response_replication_campaign_id')==campaign['campaign_id']
             and plan['execution_context_sha256']==campaign['execution_context_sha256'],
             'Lifecycle/campaign/full context binding changed')
    need(all(c==contexts[0] for c in contexts),'Exact full context content required; no normalization')
    return campaign


def require_matching_completion(campaign,after_ref):
    after=read(after_ref)
    need(after.get('schema')==MATCHING_SCHEMA and after.get('status')==MATCHING_STATUS
         and after.get('manifest')==campaign['after_matching_manifest']
         and len(after.get('completed_attempts',[]))==15 and after.get('numerical_fit_run') is False,
         'Pinned matching fifteen-attempt native completion required before explicit execution')
    verify(campaign['after_matching_manifest'])
    manifest=read(campaign['after_matching_manifest'])
    need({a['attempt_id'] for a in after['completed_attempts']}=={a['attempt_id'] for a in manifest['attempts']},
         'Full pinned matching attempt cohort required')
    for ref in strict_references(after):verify(ref)


def execute(campaign_path,capture_id,after_ref):
    campaign=audit(campaign_path);require_matching_completion(campaign,after_ref)
    entries=[r for r in campaign['runs'] if r['capture_id']==capture_id];need(len(entries)==1,'Unknown capture lifecycle')
    entry=entries[0];run=Path(entry['run'])
    # The outer wrapper has its own exclusive no-retry intent; old execute also
    # enforces exclusive run, lease, source guard, continuous watchdog, cleanup.
    refs=strict_references([receipt(campaign_path),campaign,after_ref,read(after_ref)])
    with SourceChangeGuard(refs) as guard:
        guard.check();audit(campaign_path);require_matching_completion(campaign,after_ref)
        write(run/'response-replication-claim.json',dict(schema=SCHEMA+'.claim',capture_id=capture_id,
            campaign=receipt(campaign_path),after=after_ref,no_retries=True,guarded_sources=refs))
        result=collector.execute(run)
        guard.check();audit(campaign_path);require_matching_completion(campaign,after_ref)
        write(run/'response-replication-result.json',dict(schema=SCHEMA+'.execution',capture_id=capture_id,
            claim=receipt(run/'response-replication-claim.json'),execution=receipt(run/'execution.json'),
            status=result['status'],GPU_started=result['GPU_started'],fit_started=False,
            outer_source_guard_passed=True))
    return result


def authenticate_review(campaign_path,capture_id,native_report_path):
    """CPU adapter consumes an already exported original strict review."""
    campaign=audit(campaign_path);entry=next(r for r in campaign['runs'] if r['capture_id']==capture_id)
    strict,plan,raw=native_review.load_review(native_report_path,collector.LIGHT)
    need(strict.get('training_only') is True and strict.get('validation_started') is False
         and strict.get('model_fitted') is False and strict.get('issues')==[], 'Explicit strict native training PASS required')
    need(strict['evidence']['plan.json']==entry['plan'] and strict['evidence']['context.json']==entry['context']
         and plan.get('response_replication_capture_id')==capture_id
         and strict['trace'].get('common_session_clock_identified') is True,
         'Exact lifecycle plan/context and one common native clock required')
    run=Path(entry['run']);claim=read(receipt(run/'response-replication-claim.json'))
    execution=read(receipt(run/'response-replication-result.json'))
    need(claim.get('schema')==SCHEMA+'.claim' and claim.get('capture_id')==capture_id
         and claim.get('campaign')==receipt(campaign_path) and claim.get('no_retries') is True,
         'Exclusive source-bound campaign lifecycle claim required')
    require_matching_completion(campaign,claim['after'])
    need(claim.get('guarded_sources')==strict_references([receipt(campaign_path),campaign,
         claim['after'],read(claim['after'])]),'Complete campaign/all-plan/matching source guard binding required')
    need(execution.get('schema')==SCHEMA+'.execution' and execution.get('capture_id')==capture_id
         and execution.get('status')=='COLLECTED' and execution.get('GPU_started') is True
         and execution.get('outer_source_guard_passed') is True
         and execution.get('claim')==receipt(run/'response-replication-claim.json')
         and execution.get('execution')==strict['evidence']['execution.json'],
         'Source-bound completed native lifecycle execution required')
    instances=validate_records(plan,raw,campaign['exact_owner_inputs'],capture_id)
    result=dict(schema=SCHEMA+'.review',status='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY',capture_id=capture_id,
        campaign=receipt(campaign_path),native_review=receipt(native_report_path),raw=strict['evidence']['raw.json'],
        sqlite=strict['trace']['sqlite'],instances=instances,full_denominator=20,
        training_only=True,validation_labels_read=False,fit_started=False,GPU_started=False,
        template_equivalence='REQUIRES_REPEATED_INSTANCE_ADAPTER',numerical_qualification=False)
    return result


def review(campaign_path,capture_id,native_report_path,output):
    result=authenticate_review(campaign_path,capture_id,native_report_path)
    write(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--matrix',required=True);p.add_argument('--source-run',required=True);p.add_argument('--matching-manifest',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--campaign',required=True)
    p=sub.add_parser('execute');p.add_argument('--campaign',required=True);p.add_argument('--capture-id',required=True)
    p.add_argument('--after-receipt',required=True);p.add_argument('--execute',required=True,action='store_true')
    p=sub.add_parser('review');p.add_argument('--campaign',required=True);p.add_argument('--capture-id',required=True);p.add_argument('--native-review',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.matrix,args.source_run,args.matching_manifest,args.output)
    elif args.op=='audit':result=audit(args.campaign)
    elif args.op=='execute':result=execute(args.campaign,args.capture_id,receipt(args.after_receipt))
    else:result=review(args.campaign,args.capture_id,args.native_review,args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','campaign_id','capture_id','captures','measured_requests_total')}))


if __name__=='__main__':main()
