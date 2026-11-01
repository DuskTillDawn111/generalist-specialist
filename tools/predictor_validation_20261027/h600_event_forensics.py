"""Trace frozen forecast prefixes without changing progress, budgets or inputs."""
from __future__ import annotations
import argparse
from collections import deque
import inspect
import json
import math
from pathlib import Path
import sys
from gsdeploy.forecasting import FullHorizonModel
from .flow_validation import _provider,_resources
from .native_flow import checked,receipt,save
from .probe_validation import read


class RepeatedNumericalState(RuntimeError):pass


def trace_case(plan,candidate,model,resources):
    code=FullHorizonModel.forecast.__code__
    lines,start=inspect.getsourcelines(FullHorizonModel.forecast)
    capture=start+next(i for i,line in enumerate(lines) if 'power = self.power_provider.power' in line)
    history=deque(maxlen=12);first_zero=None;previous=None;repeated=0
    def trace(frame,event,arg):
        nonlocal first_zero,previous,repeated
        if frame.f_code is not code:return None
        if event=='line' and frame.f_lineno==capture:
            v=frame.f_locals
            row=dict(events=v['events'],clock=v['clock'],boundary=v['boundary'],delta=v['delta'],
                clock_ulp_ns=math.ulp(v['clock']),arrival_index=v['arrival_index'],
                finished_count=len(v['finishes']),arrival_time=v['arrival_time'],
                active={a:dict(request_id=r['request_id'],segment_id=r['segment_id'],
                    fraction=v['progress'][a][1],service_ms=v['rates'][a],duration_ns=v['durations'][a])
                    for a,r in v['active'].items()})
            history.append(row)
            signature=json.dumps({k:val for k,val in row.items() if k!='events'},sort_keys=True)
            repeated=repeated+1 if signature==previous else 0;previous=signature
            if row['delta']==0 and first_zero is None:first_zero=row
            if row['delta']==0 and repeated>=8:raise RepeatedNumericalState('Identical numerical state persisted for nine events')
        return trace
    queues={a:[] for a in plan['aliases']};future=[]
    for item in plan['trace']:
        row={k:item[k] for k in ('request_id','task','pool_id')};row['arrival_ns']=item['offset_ns']
        (queues[candidate['routes'][row['request_id']]] if row['arrival_ns']==0 else future).append(row)
    simulator=FullHorizonModel(plan['aliases'],_provider(model,candidate['execution_context_sha256']),
        _resources(resources,candidate['execution_context_sha256']),resources=candidate.get('resource_configuration'))
    status='COMPLETED';error=None;forecast=None
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        forecast=simulator.forecast(queues,0,round(plan['window_seconds']*1e9),future_arrivals=future,
            route_future=lambda row,pending,now:candidate['routes'][row['request_id']],
            require_energy=False,require_memory=False,
            drain_deadline_ns=round((plan['window_seconds']+plan['drain_seconds'])*1e9))
    except RepeatedNumericalState as exc:status='NUMERICAL_FIXED_POINT_PREFIX';error=str(exc)
    except Exception as exc:status='ERROR';error=type(exc).__name__+':'+str(exc)
    finally:sys.settrace(old)
    minimal=[]
    if history:
        last=history[-1]
        for alias,row in last['active'].items():
            if last['clock']+row['duration_ns']==last['clock']:
                minimal.append(dict(alias=alias,clock_ns=last['clock'],remaining_fraction=row['fraction'],
                    service_ms=row['service_ms'],remaining_duration_ns=row['duration_ns'],
                    clock_ulp_ns=last['clock_ulp_ns'],fraction_exceeds_completion_epsilon=row['fraction']>1e-10,
                    clock_plus_duration_equals_clock=True,
                    update_with_zero_delta_preserves_fraction=(row['fraction']-0./(row['service_ms']*1e6)==row['fraction'])))
    return dict(status=status,error=error,original_max_events=simulator.max_events,
        first_zero_delta=first_zero,last_events=list(history),minimal_float_reproductions=minimal,
        completed_transition_count=forecast['transition_count'] if forecast else None,
        intervention='Read-only Python tracing; abort after nine exactly repeated zero-delta states; no clock/progress/budget changes')


def run(report_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    report=read(report_path);refs=[receipt(report_path),receipt(__file__)]
    for ref in report['source_receipts']:checked(ref)
    rows=[]
    for row in report['candidates']:
        if row['seed']!=627602 or row['candidate_id'] not in ('shared_spill_low','shared_spill_high'):continue
        plan=read(checked(row['heldout_plan']));model=read(checked(row['model']))
        resources=read(checked(row['resource_prediction_source']))
        candidate=next(c for c in plan['candidates'] if c['id']==row['candidate_id'])
        refs.extend([row['heldout_plan'],row['model'],row['resource_prediction_source']])
        rows.append(dict(seed=row['seed'],candidate_id=row['candidate_id'],diagnostic=trace_case(plan,candidate,model,resources)))
    for ref in [*refs,*report['source_receipts']]:checked(ref)
    result=dict(schema='gsdeploy.H600_frozen_forecast_numeric_prefix_diagnostic.v1',cases=rows,source_receipts=refs,
        frozen_source_receipts=report['source_receipts'],actual_future_execution_used=False,
        thresholds_changed=False,original_report_replaced=False,GPU_started=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'diagnostic.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.report,args.output)
    print(json.dumps([dict(candidate_id=r['candidate_id'],status=r['diagnostic']['status'],minimal=r['diagnostic']['minimal_float_reproductions']) for r in result['cases']]))
