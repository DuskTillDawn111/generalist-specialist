"""CPU audit of actual probe children and unresolved switch PID namespaces."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt,save
from .smoke_verifier_v3 import check_receipt
from .trace_export import linux_pid


def review(run,output):
    run,output=Path(run).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((run/'plan.json').read_text());execution=json.loads((run/'execution.json').read_text())
    issues=[]
    if not check_receipt(plan['source']) or not check_receipt(execution['plan'],run/'plan.json'):
        issues.append('EXECUTED_FROZEN_PLAN_SOURCE_RECEIPT_FAILED')
    cohorts=[]
    for line in (run/'stdout.log').read_text().splitlines():
        try:doc=json.loads(line)
        except ValueError:continue
        if doc.get('event')=='COHORT_COMPLETED':cohorts.append(doc)
    if len(cohorts)!=1:raise ValueError('One actual completed cohort stdout required')
    events=cohorts[0]['children'];ready=[e for e in events if e['event']=='READY'];done=[e for e in events if e['event']=='DONE']
    pids={e['pid'] for e in ready}
    if len(ready)!=2 or len(done)!=2 or len(pids)!=2 or {e['pid'] for e in done}!=pids:
        issues.append('TWO_ACTUAL_OWNED_CHILD_IDENTITIES_REQUIRED')
    if any(e.get('matrix_multiplies',0)<=0 or e.get('observed_scalar')!=1024. or e.get('start_perf_ns',0)>=e.get('end_perf_ns',0) for e in done):
        issues.append('ACTUAL_CUDA_WORK_OUTPUT_FAILED')
    sql=run/'review/trace.sqlite';connection=sqlite3.connect('file:'+str(sql)+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    kernels=[{**dict(r),'pid':linux_pid(r['globalPid'])} for r in connection.execute('select globalPid,contextId,count(*) count,min(start) first_ns,max(end) last_ns from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid,contextId')]
    if {r['pid'] for r in kernels}!=pids or any(r['count']<=0 for r in kernels):issues.append('OWNED_CHILD_KERNEL_COVERAGE_FAILED')
    nvtx=[{**dict(r),'pid':linux_pid(r['globalTid'])} for r in connection.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'GS_CONTEXT_PROBE:%'")]
    if {r['pid'] for r in nvtx}!=pids or len(nvtx)!=2:issues.append('OWNED_CHILD_NVTX_COVERAGE_FAILED')
    switches=[dict(r) for r in connection.execute('select * from GPU_CONTEXT_SWITCH_EVENTS order by seqNo')]
    enums={r['id']:r['name'] for r in connection.execute('select * from ENUM_GPU_CTX_SWITCH')}
    switchpids={linux_pid(r['globalPid']) for r in switches}
    if not switches:issues.append('NO_ACTUAL_GPU_CONTEXT_SWITCH_EVENTS')
    if [r['seqNo'] for r in switches]!=list(range(len(switches))):issues.append('CONTEXT_SWITCH_SEQUENCE_NUMBER_GAPS_OR_DUPLICATES')
    diagnostics=[{**dict(r),'pid':linux_pid(r['globalPid'])} for r in connection.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
    alarming=[r for r in diagnostics if r['severity']>=3 or any(w in r['text'].lower() for w in ('lost','drop','permission','denied','overflow')) or r['pid'] in pids]
    if alarming:issues.append('OWNED_WORKER_OR_TRACE_DIAGNOSTIC_FAILURE')
    contexts=[dict(r) for r in connection.execute('select * from TARGET_INFO_CUDA_CONTEXT_INFO')]
    connection.close()
    mapped=switchpids==pids
    result=dict(schema='gsdeploy.context_switch_actual_work_strict_review.v4',status='CAPABILITY_CONFIRMED_OWNED_SWITCH_UNKNOWN' if not issues and not mapped else 'ACTUAL_OWNED_SWITCH_RECORDS' if not issues else 'FAIL',
        issues=issues,actual_cohort=cohorts[0],kernel_groups=kernels,nvtx_ranges=nvtx,
        actual_gpu_context_switch_event_count=len(switches),switch_pid_values=sorted(switchpids),cuda_child_pid_values=sorted(pids),
        switch_context_ids=sorted({r['contextId'] for r in switches}),cuda_context_records=contexts,
        switch_tag_counts=[dict(tag=tag,name=enums.get(tag),count=count) for tag,count in sorted(Counter(r['tag'] for r in switches).items())],
        switch_pid_namespace_bound_to_owned_children=mapped,context_switch_permission_confirmed=bool(switches),
        sequence_numbers_contiguous=not any('SEQUENCE' in i for i in issues),
        diagnostics=diagnostics,alarming_diagnostics=alarming,
        same_sm_concurrency_proven=False,actual_original_four_owner_scheduling_qualified=False,
        limitation='Switch context IDs are driver IDs; CUDA context IDs are process-local. No exported bridge relates them. Different PID sets cannot be bound by subtracting a guessed offset. No additional probe required solely to remove this UNKNOWN.',
        sources=[receipt(run/'plan.json'),receipt(run/'execution.json'),receipt(run/'stdout.log'),receipt(run/'stderr.log'),receipt(sql),plan['source']],
        evaluator=receipt(__file__))
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.run,args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','issues','actual_gpu_context_switch_event_count','switch_pid_values','cuda_child_pid_values')}))


if __name__=='__main__':main()
