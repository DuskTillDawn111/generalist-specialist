"""CPU parser and training-only semantic mixed-phase progress model.

Forward hooks mark Python call boundaries; these are not GPU completion events.
Phases include sampling, transfers, inference postprocessing and sync according
to the explicit boundary contract. No counters, physical values, cache hits or
kernel progress are manufactured. The development model remains core-disabled.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls

from .profiling_analysis import digest,map_native_collection

SCHEMA='gsdeploy.development_semantic_phase_progress.v1'
DEFINITION={'schema':SCHEMA,
 'prepare':'actual worker START..first language forward start',
 'prefill':'first language forward start..second forward start; single-forward case ends at post_sync end',
 'decode':'second language forward start..post_sync end; absent if only one forward',
 'finalize':'post_sync end..actual worker END',
 'ocr_native':'OCR actual worker START..END',
 'scope':'Mixed native semantic wall intervals; decode includes later sampling/inference postprocessing/sync, not pure GPU or token-only time',
 'baseline':'target family/task/pool/phase solo mean from >=2 training observations',
 'context':'Full active (family,semantic phase) multiset, preserving multiplicity; direct multiway calibration only',
 'alias_sharing':'Same-family aliases share target-pool-specific progress; per-alias residuals reported',
 'resource_demand':'UNKNOWN; no SM/DRAM/L2 inference from host markers',
 'minimum_observations':2,'maximum_relative_work_error':.15,'core_enabled':False}


def _ns(value):
    if type(value) is not int or value < 0:raise ValueError('Actual integer monotonic timestamp required')
    return value


def parse_semantic_request(row, response):
    """Return continuous observed semantic phases; no missing-forward guessing."""
    start,end=_ns(row['handler_start_ns']),_ns(row['handler_end_ns'])
    if start>=end:raise ValueError('Nonempty handler required')
    if row['family']=='ocr':
        phases=[{'phase':'ocr_native','start_ns':start,'end_ns':end}]
        boundary_report={'known':False,'reason':'No language-forward semantics for OCR native executor'}
    else:
        observation=response.get('semantic_forward_observations')
        if not isinstance(observation,dict) or observation.get('schema')!='gsdeploy.semantic_forward_observation.v1' or observation.get('request_id')!=row['request_id']:
            raise ValueError('Matching explicit language-forward observation required')
        if observation.get('clock')!='time.perf_counter_ns':
            raise ValueError('Semantic boundaries must share actual worker monotonic clock')
        if observation.get('instrumentation_errors') or observation.get('gpu_synchronization_added') is not False:
            raise ValueError('Semantic hook must add no GPU synchronization and report no instrumentation errors')
        records=observation.get('records',[])
        if not records:raise ValueError('No observed forward: semantics UNKNOWN')
        previous_end=start
        for index,record in enumerate(records):
            left,right=_ns(record['start_ns']),_ns(record['end_ns'])
            if not previous_end<=left<right<=end or record.get('forward_index')!=index:
                raise ValueError('Sequential complete observed forwards within handler required')
            previous_end=right
        sync=[s for s in response.get('observed_stages',[]) if s['id']=='post_sync']
        if len(sync)!=1:raise ValueError('Exactly one actual post_sync stage required')
        boundary=_ns(sync[0]['end_ns'])
        if not previous_end<=boundary<=end:raise ValueError('post_sync must follow final observed forward')
        first=records[0]['start_ns'];second=records[1]['start_ns'] if len(records)>1 else None
        phases=[]
        if start<first:phases.append({'phase':'prepare','start_ns':start,'end_ns':first})
        phases.append({'phase':'prefill','start_ns':first,'end_ns':second if second is not None else boundary})
        if second is not None:phases.append({'phase':'decode','start_ns':second,'end_ns':boundary})
        if boundary<end:phases.append({'phase':'finalize','start_ns':boundary,'end_ns':end})
        if any(s['start_ns']>=s['end_ns'] for s in phases):raise ValueError('Nonempty actual semantic phases required')
        references=response.get('semantic_boundary_references',[])
        errors=[]
        actual={'first_forward_start':first,'post_sync_end':boundary}
        if second is not None:actual['second_forward_start']=second
        for reference in references:
            if reference['boundary'] not in actual or not reference.get('basis'):raise ValueError('Explicit matching semantic boundary reference required')
            timestamp=_ns(reference['reference_ns'])
            errors.append({'boundary':reference['boundary'],'error_ns':actual[reference['boundary']]-timestamp,
                           'basis':reference['basis'],'scope':'Cross-observation boundary discrepancy, not inferred GPU progress'})
        boundary_report={'known':bool(errors),'errors':errors,
            'reason':None if errors else 'Independent boundary reference absent; GPU/true-progress positioning error UNKNOWN',
            'forward_count':len(records),'observed_forward_call_wall_ns':sum(r['end_ns']-r['start_ns'] for r in records),
            'gpu_progress_known':False,'cache_hit_known':False,
            'token_scope':observation.get('token_scope'),'classification_scope':observation.get('classification_scope')}
    for phase in phases:
        phase.update(demand={'sm':None,'dram':None,'l2':None},kind='mixed',
                     observation_basis='Actual worker START/END and semantic forward/post_sync host monotonic boundaries')
    return {'request_id':row['request_id'],'alias':row['alias'],'family':row['family'],'task':row['task'],
            'pool_id':row['pool_id'],'handler_start_ns':start,'handler_end_ns':end,'segments':phases,
            'progress_positioning':boundary_report}


def semantic_traces(raw_path):
    mapped=map_native_collection(raw_path)
    raw=json.loads(Path(raw_path).read_text());source={r['request_id']:r for r in raw['request_records']}
    traces=[]
    for trace in mapped['traces']:
        result={k:v for k,v in trace.items() if k!='requests'}
        result['requests']=[parse_semantic_request(row,source[row['request_id']]['worker_response']) for row in trace['requests']]
        traces.append(result)
    return traces,mapped['mapping_report']


def semantic_context(active):
    members=[{'family':family,'phase':phase,'count':count} for (family,phase),count in
             sorted(collections.Counter((r['family'],s['phase']) for r,s in active).items())]
    return digest(members),members


def fit_semantic(raw_path):
    traces,mapping=semantic_traces(raw_path);training=[t for t in traces if t['split']=='training']
    if not training:raise ValueError('Explicit training observations required; validation never fitted')
    baselines,groups,contexts,unknown={},{},{},set()
    for trace in training:
        for row in trace['requests']:
            if any(other is not row and other['handler_start_ns']<row['handler_end_ns'] and row['handler_start_ns']<other['handler_end_ns'] for other in trace['requests']):continue
            for segment in row['segments']:
                key=row['family'],row['task'],row['pool_id'],segment['phase']
                baselines.setdefault(key,[]).append((segment['end_ns']-segment['start_ns'])/1e6)
    for trace in training:
        segments=[(r,s) for r in trace['requests'] for s in r['segments']]
        cuts=sorted({s[k] for _,s in segments for k in ('start_ns','end_ns')});intervals=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[(r,s) for r,s in segments if s['start_ns']<=left and right<=s['end_ns']]
            if not active:continue
            if len({r['alias'] for r,_ in active})!=len(active):raise ValueError('Overlapping same-alias requests unsupported')
            key,members=semantic_context(active);contexts[key]=members;intervals.append((left,right,key))
        for row,segment in segments:
            target=row['family'],row['task'],row['pool_id'],segment['phase'];base=baselines.get(target,[])
            if len(base)<2:unknown.add(target);continue
            baseline=statistics.mean(base);durations={}
            for left,right,key in intervals:
                if segment['start_ns']<=left and right<=segment['end_ns']:
                    durations[key]=durations.get(key,0.)+(right-left)/1e6/baseline
            groups.setdefault(target,[]).append({'id':trace['id']+':'+row['request_id']+':'+segment['phase'],'alias':row['alias'],'durations':durations})
    accepted,diagnostics=[],[]
    for target,rows in sorted(groups.items()):
        keys=sorted({k for r in rows for k in r['durations']});matrix=np.array([[r['durations'].get(k,0.) for k in keys] for r in rows])
        rank=int(np.linalg.matrix_rank(matrix));condition=float(np.linalg.cond(matrix)) if rank==len(keys) else math.inf
        diagnostic={'family':target[0],'task':target[1],'pool_id':target[2],'phase':target[3],
                    'observations':len(rows),'columns':len(keys),'rank':rank,'accepted':False,
                    'condition_number':condition if math.isfinite(condition) else None,'condition_number_infinite':not math.isfinite(condition),'per_alias_residuals':{}}
        if len(rows)>=2 and rank==len(keys):
            speeds,_=nnls(matrix,np.ones(len(rows)));errors=np.abs(matrix@speeds-1.)
            valid=all(math.isfinite(v) and v>0 for v in speeds) and max(errors)<=DEFINITION['maximum_relative_work_error']
            diagnostic.update(accepted=bool(valid),max_relative_work_error=float(max(errors)),mean_relative_work_error=float(statistics.mean(errors)),
                nonpositive_progress=bool(any(v<=0 for v in speeds)))
            for alias in sorted({r['alias'] for r in rows}):
                ix=[i for i,r in enumerate(rows) if r['alias']==alias]
                diagnostic['per_alias_residuals'][alias]={'count':len(ix),'max_relative_work_error':float(max(errors[ix])),
                    'mean_relative_work_error':float(statistics.mean(errors[ix])),'scope':'Training reconstruction, not held-out exchangeability proof'}
            if valid:
                accepted.extend({'target':{'family':target[0],'task':target[1],'pool_id':target[2],'phase':target[3]},
                                 'context_id':key,'members':contexts[key],'progress':float(speed)} for key,speed in zip(keys,speeds))
        else:diagnostic['reason']='RANK_DEFICIENT_OR_INSUFFICIENT_OBSERVATIONS'
        diagnostics.append(diagnostic)
    position=[{'request_id':r['request_id'],**r['progress_positioning']} for t in training for r in t['requests']]
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),'core_enabled':False,'GPU_started':False,
        'execution_context_sha256':training[0]['execution_context_sha256'],'mapping_report':mapping,
        'source_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (raw_path,__file__)],
        'known':bool(accepted),'accepted_target_contexts':accepted,'fit_diagnostics':diagnostics,'progress_positioning':position,
        'coverage':{'training_batches':len(training),'validation_batches_excluded':sum(t['split']=='validation' for t in traces),
                    'observed_full_contexts':len(contexts),'target_groups':len(diagnostics),'accepted_groups':sum(d['accepted'] for d in diagnostics),
                    'unknown_solo_baselines':[list(k) for k in sorted(unknown)],
                    'progress_positioning_reference_count':sum(r['known'] for r in position)},
        'solo_baselines':[{'family':k[0],'task':k[1],'pool_id':k[2],'phase':k[3],'observations':len(v),'mean_ms':statistics.mean(v)} for k,v in sorted(baselines.items())],
        'scope':'Development training reconstruction only; full context coefficients never inferred from pairwise, all physical/kernel progress and unmeasured demand remain UNKNOWN'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);result=fit_semantic(args.raw)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
