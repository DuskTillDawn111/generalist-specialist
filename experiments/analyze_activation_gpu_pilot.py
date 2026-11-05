"""Audit collector pilot coverage without promoting it to training/confirmation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.energy import integrate_power


def audit(path):
    path=Path(path).resolve();raw=json.loads(path.read_text())
    rows=raw['request_records'];samples=raw.get('board_samples',[])
    issues=[];details=[]
    for row in rows:
        response=row.get('worker_response',{})
        phases=[e for e in row.get('execution_events',[]) if 'event' in e]
        spans=[];ordered=bool(phases) and len(phases)%2==0
        for a,b in zip(phases[::2],phases[1::2]):
            valid=(a.get('event')=='start' and b.get('event')=='end'
                and a.get('segment_id')==b.get('segment_id') and a.get('kind')==b.get('kind')
                and a['perf_ns']<=b['perf_ns'])
            ordered=ordered and valid
            if valid:spans.append((a['perf_ns'],b['perf_ns']))
        start,end=response.get('start_perf_ns'),response.get('end_perf_ns')
        ready=row.get('previous_handler_next_request_ready',{})
        ready_valid=(isinstance(start,int) and isinstance(end,int)
            and ready.get('pid')==raw.get('ready',{}).get(row['model'],{}).get('pid')
            and end<=ready.get('perf_ns',-1)<=ready.get('received_time_ns',-1))
        contiguous=(bool(spans) and spans[0][0]==start and spans[-1][1]==end
                    and all(a[1]==b[0] for a,b in zip(spans,spans[1:])))
        missing_ms=((end-start-sum(b-a for a,b in spans))/1e6
                    if isinstance(start,int) and isinstance(end,int) else None)
        detail=dict(request_id=row['request_id'],model=row['model'],task=row['task'],
            group=row['profile_group'],pool_id=row['pool_id'],status=row['status'],
            phase_count=len(spans),phase_pairs_ordered=ordered,explicit_ready_valid=ready_valid,
            complete_handler_coverage=contiguous,uncovered_handler_ms=missing_ms,
            handler_ms=(end-start)/1e6 if isinstance(start,int) and isinstance(end,int) else None,
            fifo_wait_after_submit_ms=(start-row['actual_submit_time_ns'])/1e6 if isinstance(start,int) else None,
            e2e_ms=(row['completion_time_ns']-row['scheduled_time_ns'])/1e6 if row.get('completion_time_ns') else None)
        details.append(detail)
        if not ordered or not ready_valid:issues.append(row['request_id']+': invalid phases or READY')
    begin,end=raw.get('startup_begin_time_ns'),raw.get('drain_end_time_ns')
    joules=integrate_power(samples,begin,end,max_gap_seconds=.5) if begin is not None and end is not None else None
    per_path={}
    for model,task in sorted({(r['model'],r['task']) for r in details}):
        selected=[r for r in details if r['model']==model and r['task']==task and r['group']!='priming']
        def values(field):return [r[field] for r in selected if r[field] is not None]
        per_path[model+':'+task]=dict(n=len(selected),
            mean_handler_ms=statistics.mean(values('handler_ms')) if values('handler_ms') else None,
            max_fifo_wait_ms=max(values('fifo_wait_after_submit_ms'),default=None))
    paired=[]
    for row in details:
        if not row['group'].startswith('corun-'):continue
        base=[x for x in details if x['group'].startswith('solo-') and x['model']==row['model']
              and x['task']==row['task'] and x['pool_id']==row['pool_id']]
        if len(base)==1 and base[0]['handler_ms']:
            paired.append(dict(group=row['group'],model=row['model'],task=row['task'],
                pool_id=row['pool_id'],solo_ms=base[0]['handler_ms'],corun_ms=row['handler_ms'],
                ratio=row['handler_ms']/base[0]['handler_ms'],independent_repeats=1))
    return dict(schema='gsdeploy.activation_gpu_pilot_audit.v1',source=dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()),status=raw['status'],
        gpu=raw['gpu'],hardware_policy='same_GPU_model; UUID_is_provenance_only',
        performance_scope=raw['performance_scope'],requests=len(rows),
        groups=dict(Counter(r['profile_group'] for r in rows)),
        explicit_ready_valid=sum(r['explicit_ready_valid'] for r in details),
        ordered_phase_requests=sum(r['phase_pairs_ordered'] for r in details),
        complete_handler_requests=sum(r['complete_handler_coverage'] for r in details),
        stage_templates_qualified=False,U=None,board_power=dict(samples=len(samples),
            startup_through_drain_joules=joules,coverage_valid=joules is not None,
            duration_seconds=(end-begin)/1e9 if begin is not None and end is not None else None,
            max_sample_gap_seconds=max(((b['time_ns']-a['time_ns'])/1e9 for a,b in zip(samples,samples[1:])),default=None),
            peak_owned_mib=max((s['owned_used_mib'] for s in samples),default=None),
            foreign_process_seen=any(s['foreign_pids'] for s in samples),errors=raw.get('telemetry_errors',[])),
        per_path=per_path,paired_single_observations=paired,request_audit=details,
        issues=issues,runtime_errors=raw.get('runtime_errors',[]),shutdown=raw.get('shutdown'),
        limitations=['Development pilot only; no independent repeat confidence interval',
            'Native inference phases mix host work, transfers and CUDA',
            'Uncovered handler intervals are measured gaps, not inferred host/GPU stages',
            'Pilot episode energy is not the H1800 normalization denominator',
            'Capacity, semantic input difficulty and replica cost remain unqualified'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=audit(args.raw);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','requests','explicit_ready_valid',
        'complete_handler_requests','board_power','stage_templates_qualified')},ensure_ascii=False))
