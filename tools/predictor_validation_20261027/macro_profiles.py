"""Predeclared development alternative: two MIXED native wall phases.

This model is deliberately separate from the accepted exact-profile v1 domain.
Solo baselines remain family/task/pool/phase specific. Only normalized progress
is shared across input identities and exchangeable aliases by a full multiset
of (family,macro-phase). No pairwise-to-multiway extrapolation, hardware resource
classification, core integration, validation fitting or physical claim occurs.
"""
from __future__ import annotations

import argparse
import collections
import json
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls

from .profiling_analysis import digest, map_native_collection

SCHEMA = 'gsdeploy.development_macro_phase_progress.v1'
DEFINITION = {
    'schema': SCHEMA,
    'macro0': 'observed START through post_sync PHASE end; PP OCR through observed END',
    'macro1': 'post_sync PHASE end through observed END, including cleanup/final sync/control',
    'context': 'full multiset of (family,macro phase), including same-family multiplicity',
    'baseline': 'family/task/pool/macro-specific mean of >=2 training solo observations',
    'sharing_assumptions': ['Aliases of same family are exchangeable after input-specific solo normalization',
                            'Normalized progress is shared across inputs within each family/macro/context'],
    'minimum_observations': 2, 'maximum_relative_work_error': .15,
    'classification': 'UNKNOWN:ERR_NVGPUCTRPERM',
    'scope': 'MIXED native phases, not pure GPU kernels; development alternative not core-enabled',
}


def _phase_key(row, segment):
    return row['family'], row['task'], row['pool_id'], segment['phase']


def _context(active):
    counts=collections.Counter((row['family'],segment['phase']) for row,segment in active)
    members=[{'family':family,'phase':phase,'count':count} for (family,phase),count in sorted(counts.items())]
    return digest(members),members


def macro_traces(raw_path):
    mapped=map_native_collection(raw_path)
    raw=json.loads(Path(raw_path).read_text())
    originals={r['request_id']:r for r in raw['request_records']}
    traces=[]
    for trace in mapped['traces']:
        value={k:v for k,v in trace.items() if k!='requests'}
        value['requests']=[]
        for row in trace['requests']:
            source=originals[row['request_id']]
            stages=source['worker_response']['observed_stages']
            matches=[s for s in stages if s['id']=='post_sync']
            start,end=row['handler_start_ns'],row['handler_end_ns']
            if row['family']=='ocr':
                boundary=end
            elif len(matches)!=1:
                raise ValueError('Exactly one observed post_sync boundary required for non-PP executor')
            else:
                boundary=matches[0]['end_ns']
            if not start<boundary<=end:
                raise ValueError('Invalid observed macro boundary')
            segments=[{'phase':'macro0','start_ns':start,'end_ns':boundary}]
            if boundary<end:
                segments.append({'phase':'macro1','start_ns':boundary,'end_ns':end})
            value['requests'].append({k:v for k,v in row.items() if k!='segments'} | {'segments':segments})
        traces.append(value)
    return traces,mapped['mapping_report']


def fit_macro(raw_path):
    traces,mapping=macro_traces(raw_path)
    training=[t for t in traces if t['split']=='training']
    if not training:
        raise ValueError('Training observations required; validation is never fitted')
    baselines,solo_alias_durations={},{}
    for trace in training:
        for row in trace['requests']:
            others=[r for r in trace['requests'] if r is not row]
            if any(r['handler_start_ns']<row['handler_end_ns'] and r['handler_end_ns']>row['handler_start_ns'] for r in others):
                continue
            for segment in row['segments']:
                duration=(segment['end_ns']-segment['start_ns'])/1e6
                baselines.setdefault(_phase_key(row,segment),[]).append(duration)
                solo_alias_durations.setdefault((*_phase_key(row,segment),row['alias']),[]).append(duration)
    contexts,groups,unknown_baselines={}, {}, set()
    for trace in training:
        segments=[(r,s) for r in trace['requests'] for s in r['segments']]
        cuts=sorted({s[k] for _,s in segments for k in ('start_ns','end_ns')})
        intervals=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[(r,s) for r,s in segments if s['start_ns']<=left and right<=s['end_ns']]
            if not active:continue
            if len({r['alias'] for r,_ in active}) != len(active):
                raise ValueError('Independent executor cannot overlap requests on same alias')
            key,members=_context(active);contexts[key]=members
            intervals.append((left,right,key))
        for row,segment in segments:
            base_key=_phase_key(row,segment);values=baselines.get(base_key,[])
            if len(values)<2:
                unknown_baselines.add(base_key);continue
            baseline=statistics.mean(values)
            durations={}
            for left,right,key in intervals:
                if segment['start_ns']<=left and right<=segment['end_ns']:
                    durations[key]=durations.get(key,0.)+(right-left)/1e6/baseline
            groups.setdefault((row['family'],segment['phase']),[]).append(
                {'observation_id':trace['id']+':'+row['request_id']+':'+segment['phase'],
                 'alias':row['alias'],'pool_id':row['pool_id'],'durations':durations})
    fits,diagnostics={},[]
    for (family,phase),rows in sorted(groups.items()):
        keys=sorted({k for r in rows for k in r['durations']})
        matrix=np.array([[r['durations'].get(k,0.) for k in keys] for r in rows])
        rank=int(np.linalg.matrix_rank(matrix));accepted=False;speeds=None
        diagnostic={'family':family,'phase':phase,'observation_count':len(rows),'columns':len(keys),'rank':rank,
                    'accepted':False,'contexts':keys,'alias_exchange_diagnostics':{},'cross_input_sharing_diagnostics':{}}
        # Full-rank only: a regularizer must not turn ambiguity into evidence.
        if len(rows)>=DEFINITION['minimum_observations'] and rank==len(keys):
            speeds,_=nnls(matrix,np.ones(len(rows)))
            errors=np.abs(matrix@speeds-1.)
            accepted=all(math.isfinite(v) and v>0 for v in speeds) and max(errors)<=DEFINITION['maximum_relative_work_error']
            diagnostic.update(max_relative_work_error=float(max(errors)),mean_relative_work_error=float(statistics.mean(errors)))
            for name,field in [('alias_exchange_diagnostics','alias'),('cross_input_sharing_diagnostics','pool_id')]:
                for identity in sorted({r[field] for r in rows}):
                    indices=[i for i,r in enumerate(rows) if r[field]==identity]
                    submatrix=matrix[indices,:]
                    diagnostic[name][identity]={'observations':len(indices),'max_relative_work_error':float(max(errors[indices])),
                        'mean_relative_work_error':float(statistics.mean(errors[indices])),
                        'independent_context_rank':int(np.linalg.matrix_rank(submatrix)),
                        'scope':'Training reconstruction by group, not holdout or proof of exchangeability/generalization'}
        else:
            diagnostic['reason']='RANK_DEFICIENT_OR_INSUFFICIENT_OBSERVATIONS'
        if rank<len(keys):
            for name,field in [('alias_exchange_diagnostics','alias'),('cross_input_sharing_diagnostics','pool_id')]:
                for item in sorted({r[field] for r in rows}):
                    indices=[i for i,r in enumerate(rows) if r[field]==item]
                    diagnostic[name][item]={'observations':len(indices),'independent_context_rank':int(np.linalg.matrix_rank(matrix[indices,:])),
                        'max_relative_work_error':None,'mean_relative_work_error':None,
                        'scope':'Shared-context speeds not identifiable; no reconstruction or sharing validation claim'}
        diagnostic['accepted']=bool(accepted)
        if accepted:
            fits[(family,phase)]={k:float(speed) for k,speed in zip(keys,speeds)}
        diagnostics.append(diagnostic)
    supported={}
    for key,members in contexts.items():
        if all((m['family'],m['phase']) in fits and key in fits[(m['family'],m['phase'])] for m in members):
            supported[key]={'members':members,'progress':{m['family']+':'+m['phase']:fits[(m['family'],m['phase'])][key] for m in members},
                            'resource_demand':{'sm':None,'dram':None,'l2':None}}
    family_counts=collections.Counter(r['family'] for t in training for r in t['requests'])
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),
            'execution_context_sha256':training[0]['execution_context_sha256'],
            'raw_source':mapping,'source_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (raw_path,__file__)],'GPU_started':False,'core_enabled':False,'known':bool(supported),
            'supported_contexts':supported,'fit_diagnostics':diagnostics,
            'coverage':{'training_batches':len(training),'excluded_validation_batches':sum(t['split']=='validation' for t in traces),
                        'observed_contexts':len(contexts),'fully_supported_contexts':len(supported),
                        'unsupported_observed_contexts':len(set(contexts)-set(supported)),
                        'unknown_solo_baselines':[list(k) for k in sorted(unknown_baselines)],
                        'training_requests_by_family':dict(family_counts)},
            'solo_alias_exchange_diagnostics':[{'family':k[0],'task':k[1],'pool_id':k[2],'phase':k[3],'alias':k[4],
                'observations':len(v),'mean_ms':statistics.mean(v),'normalized_to_pooled_mean':statistics.mean(v)/statistics.mean(baselines[k[:4]]),
                'scope':'Training solo alias variation; pooled baseline assumes exchangeability'} for k,v in sorted(solo_alias_durations.items())],
            'solo_baselines':[{'family':k[0],'task':k[1],'pool_id':k[2],'phase':k[3],
                               'observations':len(v),'mean_ms':statistics.mean(v)} for k,v in sorted(baselines.items())],
            'scope':'Predeclared development-only sharing assumptions; no validation tuning, six-term scoring, native stability or kernel-causal classification'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=fit_macro(args.raw)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
