"""Training-only development model with input-specific whole-handler progress.

A handler is a MIXED CPU/transfer/CUDA/wait/cleanup wall interval. This model
never claims kernel interference attribution. It reduces only context identity:
full active family multiplicities, keeping target family/task/pool-specific
baselines and coefficients. Pairwise evidence cannot create multiway evidence.
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

from .profiling_analysis import digest, map_native_collection

SCHEMA='gsdeploy.development_input_handler_progress.v1'
DEFINITION={'schema':SCHEMA,'segment':'actual worker START..END mixed handler wall interval',
    'baseline':'family/task/pool mean of at least two independent solo training handlers',
    'target':'family/task/pool; no cross-input coefficient pooling',
    'context':'full active family-count multiset with same-family multiplicity',
    'alias_sharing':'same-family aliases share target-specific progress; report per-alias residuals and solo baseline shifts',
    'minimum_observations':2,'maximum_relative_work_error':.15,
    'resource_classification':'UNKNOWN:ERR_NVGPUCTRPERM',
    'scope':'training-only development diagnostic, not kernel attribution or accepted predictor',
    'sensitivity':'Leave-one-complete-handler-out coefficient perturbation; report identifiability and max relative coefficient change without confidence interpretation'}


def handler_context(rows):
    members=[{'family':f,'count':n} for f,n in sorted(collections.Counter(r['family'] for r in rows).items())]
    return digest(members),members


def handler_traces(raw_path):
    mapped=map_native_collection(raw_path)
    return mapped['traces'],mapped['mapping_report']


def _safe_float(value):
    value=float(value)
    return value if math.isfinite(value) else None


def fit_handler(raw_path):
    traces,mapping=handler_traces(raw_path)
    training=[t for t in traces if t['split']=='training']
    if not training:raise ValueError('Training evidence required; validation never fitted')
    baselines,alias_solo={},{}
    for trace in training:
        for row in trace['requests']:
            others=[r for r in trace['requests'] if r is not row]
            if any(r['handler_start_ns']<row['handler_end_ns'] and r['handler_end_ns']>row['handler_start_ns'] for r in others):continue
            key=row['family'],row['task'],row['pool_id']
            duration=(row['handler_end_ns']-row['handler_start_ns'])/1e6
            baselines.setdefault(key,[]).append(duration)
            alias_solo.setdefault((*key,row['alias']),[]).append(duration)
    contexts,groups,unknown={},{},set()
    for trace in training:
        rows=trace['requests']
        cuts=sorted({r[k] for r in rows for k in ('handler_start_ns','handler_end_ns')})
        intervals=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[r for r in rows if r['handler_start_ns']<=left and right<=r['handler_end_ns']]
            if not active:continue
            if len({r['alias'] for r in active})!=len(active):raise ValueError('Overlapping handlers on same alias')
            key,members=handler_context(active);contexts[key]=members;intervals.append((left,right,key))
        for row in rows:
            target=row['family'],row['task'],row['pool_id']
            base=baselines.get(target,[])
            if len(base)<DEFINITION['minimum_observations']:
                unknown.add(target);continue
            baseline=statistics.mean(base);durations={}
            for left,right,key in intervals:
                if row['handler_start_ns']<=left and right<=row['handler_end_ns']:
                    durations[key]=durations.get(key,0.)+(right-left)/1e6/baseline
            groups.setdefault(target,[]).append({'id':trace['id']+':'+row['request_id'],'alias':row['alias'],
                                                'durations':durations})
    fits,diagnostics={},[]
    for target,rows in sorted(groups.items()):
        keys=sorted({k for r in rows for k in r['durations']})
        matrix=np.array([[r['durations'].get(k,0.) for k in keys] for r in rows]);rank=int(np.linalg.matrix_rank(matrix))
        singular=np.linalg.svd(matrix,compute_uv=False)
        condition=float(singular[0]/singular[-1]) if len(singular) and rank==len(keys) and singular[-1]>0 else math.inf
        diagnostic={'family':target[0],'task':target[1],'pool_id':target[2],'observation_count':len(rows),
            'columns':len(keys),'rank':rank,'condition_number':_safe_float(condition),'condition_number_infinite':not math.isfinite(condition),
            'singular_values':[float(v) for v in singular],'contexts':keys,'accepted':False,
            'per_alias_residuals':{},'coefficient_sensitivity':{'method':DEFINITION['sensitivity'],'known':False}}
        if len(rows)>=DEFINITION['minimum_observations'] and rank==len(keys):
            speeds,_=nnls(matrix,np.ones(len(rows)));errors=np.abs(matrix@speeds-1.)
            accepted=all(math.isfinite(v) and v>0 for v in speeds) and max(errors)<=DEFINITION['maximum_relative_work_error']
            diagnostic.update(accepted=bool(accepted),max_relative_work_error=float(max(errors)),
                mean_relative_work_error=float(statistics.mean(errors)),
                diagnostic_fitted_coefficients={k:float(v) for k,v in zip(keys,speeds)},
                diagnostic_coefficients_not_usable_unless_accepted=True)
            for alias in sorted({r['alias'] for r in rows}):
                indices=[i for i,r in enumerate(rows) if r['alias']==alias]
                diagnostic['per_alias_residuals'][alias]={'count':len(indices),'max_relative_work_error':float(max(errors[indices])),
                    'mean_relative_work_error':float(statistics.mean(errors[indices])),
                    'independent_rank':int(np.linalg.matrix_rank(matrix[indices,:])),
                    'scope':'Training reconstruction only; not alias exchangeability proof or held-out error'}
            diagnostic['rejection_reasons']=(['NONPOSITIVE_PROGRESS_COEFFICIENT'] if any(v<=0 for v in speeds) else []) + (['TRAINING_WORK_ERROR_EXCEEDS_FIXED_TOLERANCE'] if max(errors)>DEFINITION['maximum_relative_work_error'] else [])
            changes=[];unknown_leaves=0;nonpositive_reference=0
            for index in range(len(rows)):
                subset=np.delete(matrix,index,axis=0)
                if np.linalg.matrix_rank(subset)<len(keys):unknown_leaves+=1;continue
                alternative,_=nnls(subset,np.ones(len(rows)-1))
                if any(v<=0 for v in speeds):nonpositive_reference+=1;continue
                changes.append(float(max(abs(alternative-speeds)/speeds)))
            diagnostic['coefficient_sensitivity'].update(known=bool(changes),leave_one_out_refits=len(changes),
                leave_one_out_unidentifiable=unknown_leaves,nonpositive_reference_prevents_relative_sensitivity=nonpositive_reference,max_relative_coefficient_change=max(changes) if changes else None,
                median_relative_coefficient_change=statistics.median(changes) if changes else None,
                scope='Training sample sensitivity only, not a calibrated confidence bound')
            if accepted:fits[target]={key:float(speed) for key,speed in zip(keys,speeds)}
        else:diagnostic['reason']='RANK_DEFICIENT_OR_INSUFFICIENT_OBSERVATIONS'
        diagnostics.append(diagnostic)
    accepted_contexts=[]
    for target,progress in fits.items():
        for key,speed in progress.items():
            accepted_contexts.append({'target':{'family':target[0],'task':target[1],'pool_id':target[2]},
                'context_id':key,'members':contexts[key],'progress':speed})
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),
        'source_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (raw_path,__file__)],
        'execution_context_sha256':training[0]['execution_context_sha256'],'mapping_report':mapping,
        'GPU_started':False,'core_enabled':False,'known':bool(fits),
        'accepted_target_contexts':accepted_contexts,'fit_diagnostics':diagnostics,
        'coverage':{'training_batches':len(training),'validation_batches_excluded':sum(t['split']=='validation' for t in traces),
                    'observed_family_contexts':len(contexts),'target_groups':len(diagnostics),'accepted_target_groups':len(fits),
                    'unknown_solo_baselines':[list(v) for v in sorted(unknown)]},
        'solo_baselines':[{'family':k[0],'task':k[1],'pool_id':k[2],'observations':len(v),'mean_ms':statistics.mean(v)} for k,v in sorted(baselines.items())],
        'solo_alias_exchange_diagnostics':[{'family':k[0],'task':k[1],'pool_id':k[2],'alias':k[3],'observations':len(v),
              'mean_ms':statistics.mean(v),'relative_to_pooled_solo_mean':statistics.mean(v)/statistics.mean(baselines[k[:3]])} for k,v in sorted(alias_solo.items())],
        'scope':'New predeclared training diagnostic. Rejected groups stay UNKNOWN; original fine v1 independent validation remains unchanged.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);result=fit_handler(args.raw)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
