"""Independent solo-only evidence recovery; never fills co-resident contexts.

A target group containing unidentifiable multiway speeds must not erase actual
solo evidence. This tool fits each family/task/pool/segment SOLO separately,
using >=2 complete independent handlers with no overlapping request at all in
the unchanged all-resident deployment. No general nullspace inference occurs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from .profiling_analysis import digest, map_native_collection
from .semantic_profiles import semantic_traces

SCHEMA='gsdeploy.development_partial_solo_profiles.v1'
DEFINITION={'schema':SCHEMA,'domain':'Observed SOLO only in unchanged full residency, no other active request during complete handler',
 'targets':'family/task/pool/segment, alias exchange explicitly assumed and audited',
 'minimum_independent_handlers':2,'maximum_relative_work_error':.15,
 'fit':'NNLS one-column positive normalized-work speed, baseline is target-specific solo mean',
 'not_implemented':'general nullspace column identification; no co-resident coefficient inference',
 'scope':'Development evidence recovery; shared multiway remains UNKNOWN; prospective serial-admission validation requires matching new execution configuration',
 'core_enabled':False}


def _traces(raw_path,mode):
    if mode=='semantic':return semantic_traces(raw_path)
    if mode=='fine':
        mapped=map_native_collection(raw_path);return mapped['traces'],mapped['mapping_report']
    if mode=='handler':
        mapped=map_native_collection(raw_path)
        for trace in mapped['traces']:
            for row in trace['requests']:
                row['segments']=[{'id':'handler','kind':'mixed','start_ns':row['handler_start_ns'],'end_ns':row['handler_end_ns']}]
        return mapped['traces'],mapped['mapping_report']
    raise ValueError('Explicit fine/semantic/handler mode required')


def fit_partial_solo(raw_path, *, mode='handler'):
    traces,mapping=_traces(raw_path,mode);training=[t for t in traces if t['split']=='training']
    if not training:raise ValueError('Independent training handlers required; validation never fitted')
    groups={};seen=set();excluded=0
    for trace in training:
        for row in trace['requests']:
            rid=trace['id']+':'+row['request_id']
            if rid in seen:raise ValueError('Repeated handler observation cannot count as independent')
            seen.add(rid)
            if any(other is not row and other['handler_start_ns']<row['handler_end_ns'] and row['handler_start_ns']<other['handler_end_ns'] for other in trace['requests']):
                excluded+=1;continue
            for segment in row['segments']:
                segment_id=segment.get('phase',segment.get('id'))
                if not segment_id:raise ValueError('Explicit observed segment identity required')
                key=row['family'],row['task'],row['pool_id'],segment_id
                duration=(segment['end_ns']-segment['start_ns'])/1e6
                if not math.isfinite(duration) or duration<=0:raise ValueError('Positive complete observed solo segment required')
                groups.setdefault(key,[]).append({'handler_id':rid,'alias':row['alias'],'duration_ms':duration})
    accepted,diagnostics=[],[]
    for target,rows in sorted(groups.items()):
        baseline=statistics.mean(r['duration_ms'] for r in rows)
        normalized=[r['duration_ms']/baseline for r in rows]
        # Unique positive minimizer of sum((duration/baseline)*speed-1)^2.
        speed=sum(normalized)/sum(v*v for v in normalized)
        errors=[abs(v*speed-1.) for v in normalized]
        valid=len(rows)>=DEFINITION['minimum_independent_handlers'] and speed>0 and max(errors)<=DEFINITION['maximum_relative_work_error']
        alias={}
        for name in sorted({r['alias'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['alias']==name]
            alias[name]={'independent_handlers':len(ix),'mean_ms':statistics.mean(rows[i]['duration_ms'] for i in ix),
                         'relative_to_pooled_baseline':statistics.mean(rows[i]['duration_ms'] for i in ix)/baseline,
                         'max_relative_work_error':max(errors[i] for i in ix),'scope':'Training alias shift, no exchangeability guarantee'}
        record={'target':{'family':target[0],'task':target[1],'pool_id':target[2],'segment_id':target[3]},
            'independent_handlers':len(rows),'handler_ids':[r['handler_id'] for r in rows],'baseline_ms':baseline,
            'progress':speed,'max_relative_work_error':max(errors),'mean_relative_work_error':statistics.mean(errors),
            'accepted':bool(valid),'alias_diagnostics':alias,
            'context':{'active_family':target[0],'active_segment':target[3],'multiplicity':1,'others_active':False},
            'scope':'Actual singleton activity in fixed resident set; not compute/memory classification or multiway prediction'}
        diagnostics.append(record)
        if valid:accepted.append(record)
    return {'schema':SCHEMA,'definition':DEFINITION,'definition_sha256':digest(DEFINITION),'mode':mode,
        'source_receipts':[{'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in (raw_path,__file__)],
        'execution_context_sha256':training[0]['execution_context_sha256'],'mapping_report':mapping,
        'core_enabled':False,'GPU_started':False,'known':bool(accepted),'accepted_solo_profiles':accepted,'fit_diagnostics':diagnostics,
        'coverage':{'training_batches':len(training),'validation_batches_excluded':sum(t['split']=='validation' for t in traces),
                    'excluded_non_solo_handlers':excluded,'target_groups':len(diagnostics),'accepted_solo_groups':len(accepted),
                    'shared_contexts_inferred':0},
        'limits':['No co-resident context is inferred','No hardware demand classification','No six-term memory/energy completeness claim',
                  'Same-family alias sharing is explicit; evaluate prospectively before use','No actual serial-admission configuration transfer certificate']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--mode',choices=('fine','semantic','handler'),default='handler');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);result=fit_partial_solo(args.raw,mode=args.mode)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
