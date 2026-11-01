"""Prospectively sealed independent descriptive E/M validation; CPU assessment.

All contexts mean observed handler activity, not physical GPU idle/busy state.
The held-out observation is not used to fit or modify the empirical model.
Power validation uses piecewise-linear sample integration and raw owned peaks;
safe_factor is displayed exactly once, never interpreted as a capacity proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .empirical_resources import EmpiricalResourceProvider,resource_context,integrate_samples,RESIDENTS
from .profiling_analysis import map_native_collection

SCHEMA='gsdeploy.independent_empirical_resources_validation.v1'


def receipt(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def load_bound(reference):
    if receipt(reference['path'])!=reference:raise ValueError('Changed sealed resource artifact')
    return json.loads(Path(reference['path']).read_text())


def metric(pairs):
    if not pairs:return {'count':0,'mae':None,'bias_pred_minus_observed':None,'mape':None,'relative_error_count':0}
    errors=[p-o for p,o in pairs];relative=[abs(p-o)/o for p,o in pairs if o>0]
    return {'count':len(pairs),'mae':statistics.mean(abs(v) for v in errors),'bias_pred_minus_observed':statistics.mean(errors),
            'mape':statistics.mean(relative) if relative else None,'relative_error_count':len(relative)}


def seal(model_path,plan_path,catalog_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    model=json.loads(Path(model_path).read_text());plan=json.loads(Path(plan_path).read_text());catalog=json.loads(Path(catalog_path).read_text())
    safe=catalog.get('safe_factor',catalog.get('catalog',{}).get('safe_factor'))
    if type(safe) not in (int,float) or not math.isfinite(safe) or safe<1:raise ValueError('Original catalog safety factor required')
    if model['execution_context_sha256']!=plan['execution_context_sha256']:raise ValueError('Prospective resource model/plan execution mismatch')
    if not plan.get('jobs') or any(j['split']!='validation' for j in plan['jobs']):raise ValueError('Validation-only frozen plan required')
    bound={'schema':SCHEMA,'model':receipt(model_path),'plan':receipt(plan_path),'catalog':receipt(catalog_path),
           'tool_source':receipt(__file__),'frozen_ns':time.perf_counter_ns(),'safe_factor':safe,
           'acceptance':{'whole_window_energy_absolute_relative_error':.1,'raw_owned_peak_absolute_relative_error':.1},
           'scenarios':['mean','max'],'scope':'Descriptive empirical holdout thresholds, never physical capacity guarantees',
           'posthoc_model_changes_allowed':False,'GPU_started':False}
    with output.open('x') as stream:json.dump(bound,stream,indent=2);stream.write('\n')
    return bound


def evaluate(protocol_path,raw_path):
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol.get('schema')!=SCHEMA:raise ValueError('Explicit presealed resource protocol required')
    if receipt(__file__)!=protocol['tool_source']:raise ValueError('Assessment source changed after prospective sealing')
    model=load_bound(protocol['model']);plan=load_bound(protocol['plan']);load_bound(protocol['catalog'])
    raw=json.loads(Path(raw_path).read_text())
    if raw['observation_start_ns']<=protocol['frozen_ns']:raise ValueError('Resource protocol must be sealed before independent observation')
    if raw['execution_context_sha256']!=model['execution_context_sha256']:raise ValueError('Independent resource execution context mismatch')
    mapped=map_native_collection(raw_path)
    if any(t['split']!='validation' for t in mapped['traces']):raise ValueError('Independent validation telemetry required')
    expected={j['id'] for j in plan['jobs']};actual={b['job_id'] for b in raw['batches']}
    if actual!=expected or len(actual)!=len(raw['batches']):raise ValueError('Independent resource run differs from sealed job plan')
    samples=raw['board_samples'];times=[s['time_ns'] for s in samples]
    if len(samples)<2 or times!=sorted(set(times)):raise ValueError('Ordered resource samples required')
    maxgap=max(b-a for a,b in zip(times,times[1:]))
    if maxgap>2_000_000_000:raise ValueError('Resource telemetry exceeds unchanged 2s maximum gap')
    for sample in samples:
        if set(sample['owned_aliases'])!=set(RESIDENTS) or sample['foreign_pids']:raise ValueError('Complete owned residency without foreign GPU processes required')
    rows=[r for t in mapped['traces'] for r in t['requests']]
    aliases={a:{'family':f} for a,f in RESIDENTS.items()}
    provider=EmpiricalResourceProvider(model,raw['execution_context_sha256'])
    left,right=raw['observation_start_ns'],raw['observation_end_ns']
    cuts=sorted({left,right,*[r[k] for r in rows for k in ('handler_start_ns','handler_end_ns')]})
    intervals=[]
    for a,b in zip(cuts,cuts[1:]):
        active={r['alias']:r for r in rows if r['handler_start_ns']<=a and b<=r['handler_end_ns']}
        key,members=resource_context(active,aliases);intervals.append((a,b,active,key,members))
    observed_joules=integrate_samples(samples,left,right)/1e9
    observed_peak=max(s['owned_used_mib'] for s in samples if left<=s['time_ns']<=right)
    scenarios={}
    for scenario in protocol['scenarios']:
        unknown=set();sample_pairs=[];groups={};energy=0.;memory_values=[];batchpairs=[];observed_contexts={}
        for sample in samples:
            timestamp=sample['time_ns']
            if not left<=timestamp<=right:continue
            active={r['alias']:r for r in rows if r['handler_start_ns']<=timestamp<r['handler_end_ns']}
            key,members=resource_context(active,aliases);value=provider.power(active,aliases,scenario)
            observed_contexts[key]=members
            group=groups.setdefault(key,{'members':members,'power_pairs':[],'batch_peak_pairs':[],'requested_power_samples':0})
            group['requested_power_samples']+=1
            if value['known']:
                pair=(value['watts'],sample['watts']);sample_pairs.append(pair);group['power_pairs'].append(pair)
            else:unknown.add(key)
        for a,b,active,key,members in intervals:
            power=provider.power(active,aliases,scenario);memory=provider.memory(active,aliases,scenario)
            if power['known']:energy+=power['watts']*(b-a)/1e9
            else:unknown.add(key)
            if memory['known']:memory_values.append(memory['peak_memory_mib'])
            else:unknown.add(key)
        for trace in mapped['traces']:
            a,b=trace['observation_window_ns']['start'],trace['observation_window_ns']['end']
            local=[s['owned_used_mib'] for s in samples if a<=s['time_ns']<=b]
            predictions=[];seen=set();complete=True
            for c,d,active,key,members in intervals:
                if c>=b or d<=a:continue
                memory=provider.memory(active,aliases,scenario)
                if not memory['known']:complete=False;continue
                predictions.append(memory['peak_memory_mib']);seen.add(key)
            if local and predictions and complete:
                pair=(max(predictions),max(local));batchpairs.append(pair)
                for key in seen:
                    groups.setdefault(key,{'members':model['contexts'][key]['members'],'power_pairs':[],'batch_peak_pairs':[],'requested_power_samples':0})['batch_peak_pairs'].append(pair)
        prediction_peak=max(memory_values) if memory_values and not unknown else None
        predicted_energy=energy if not unknown else None
        erel=abs(predicted_energy-observed_joules)/observed_joules if predicted_energy is not None and observed_joules else None
        mrel=abs(prediction_peak-observed_peak)/observed_peak if prediction_peak is not None and observed_peak else None
        safe=protocol['safe_factor']
        scenarios[scenario]={'power_sample_error':metric(sample_pairs),'batch_sampled_peak_error':metric(batchpairs),
            'whole_window':{'observed_joules':observed_joules,'predicted_joules':predicted_energy,
                'bias_joules':predicted_energy-observed_joules if predicted_energy is not None else None,'absolute_relative_error':erel},
            'raw_owned_peak':{'observed_mib':observed_peak,'predicted_mib':prediction_peak,'absolute_relative_error':mrel,
                'underestimate_mib':max(0.,observed_peak-prediction_peak) if prediction_peak is not None else None,
                'safe_factor':safe,'safe_observed_mib':safe*observed_peak,'safe_predicted_mib':safe*prediction_peak if prediction_peak is not None else None,
                'safe_factor_application_count':1,'capacity_bound':False},
            'descriptive_acceptance':{'whole_window_energy_error_le_10pct':None if erel is None else erel<=.1,
                'raw_owned_peak_error_le_10pct':None if mrel is None else mrel<=.1},
            'unknown_context_ids':sorted(unknown),'observed_context_count':len(observed_contexts),
            'full_context_groups':{key:{'members':g['members'],'requested_power_samples':g['requested_power_samples'],
                'power_sample_error':metric(g['power_pairs']),'batch_sampled_peak_error':metric(g['batch_peak_pairs'])} for key,g in groups.items()}}
    return {'schema':SCHEMA+'.report','protocol':receipt(protocol_path),'raw':receipt(raw_path),'scenarios':scenarios,
        'measurement_window_ns':{'start':left,'end':right},'sample_count':sum(left<=s['time_ns']<=right for s in samples),
        'max_sample_gap_ns':maxgap,'max_query_duration_ns':max(s.get('query_duration_ns',0) for s in samples),
        'GPU_started':False,'capacity_bound':False,'global_pass':None,
        'scope':'Independent descriptive resource-model validation on handler-active full contexts; CPU/handler idle is not physical GPU idle; no model changes or capacity guarantee'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('seal')
    for key in ('model','plan','catalog','output'):a.add_argument('--'+key,type=Path,required=True)
    a=sub.add_parser('assess')
    for key in ('protocol','raw','output'):a.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(argv)
    if args.op=='seal':seal(args.model,args.plan,args.catalog,args.output)
    else:
        if args.output.exists():raise FileExistsError(args.output)
        result=evaluate(args.protocol,args.raw)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
