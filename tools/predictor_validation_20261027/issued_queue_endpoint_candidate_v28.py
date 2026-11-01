"""Measured issued/unfinished queue prefix plus endpoint first-passage fit.

This is a CPU training diagnostic, not a replacement physical service rate.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from . import kernel_endpoint_candidate_v27 as base
from .owner_lifecycle_mark_v21 import receipt
from .idle_owner_selection_audit_v20 import read

SCHEMA='gsdeploy.issued_queue_endpoint_first_passage_candidate.v28'
FEATURES=(*base.FEATURES,'log_issued_unfinished_own','log_issued_unfinished_others',
    'log_issued_unfinished_same_stream','own_context_x_log_own_queue','other_context_x_log_own_queue')


class IssuedQueuePrefix:
    def __init__(self,intervals):
        self.starts={key:sorted(begin for begin,end in spans) for key,spans in intervals.items()}
        self.ends={key:sorted(end for begin,end in spans) for key,spans in intervals.items()}

    def counts(self,clock):
        return {key:bisect_right(starts,clock)-bisect_right(self.ends[key],clock)
            for key,starts in self.starts.items()}


def features(prefix):
    original=base.features(prefix)
    own=math.log1p(prefix['measured_issued_unfinished_own_count'])
    other=math.log1p(prefix['measured_issued_unfinished_other_owner_count'])
    same=math.log1p(prefix['measured_issued_unfinished_same_stream_count'])
    return np.r_[original,own,other,same,original[4]*own,original[5]*own]


def fit(rows,kind):
    groups=defaultdict(list);usable=[row for row in rows if base.target(row,kind) is not None]
    for row in usable:groups[base.signature(row['prefix_features'])].append(math.log1p(base.target(row,kind)))
    means={key:float(np.mean(values)) for key,values in groups.items()}
    matrix=np.asarray([features(row['prefix_features']) for row in usable])
    labels=np.asarray([math.log1p(base.target(row,kind))-means[base.signature(row['prefix_features'])] for row in usable])
    center=matrix.mean(axis=0);scale=matrix.std(axis=0);center[0]=0.;scale[0]=1.;scale[scale<1e-12]=1.
    design=(matrix-center)/scale;penalty=np.eye(len(FEATURES));penalty[0,0]=0.
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@labels)
    return dict(target=kind,signature_mean_log1p_ns=[dict(key=list(key),value=value,n=len(groups[key])) for key,value in sorted(means.items())],
        feature_names=list(FEATURES),center=center.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
        feature_minimum=matrix.min(axis=0).tolist(),feature_maximum=matrix.max(axis=0).tolist(),
        training_eligible_n=len(usable),regularization='normalized squared log residual loss plus L2/N; intercept unpenalized',
        fallback_on_unseen_signature=False)


def prediction(model,prefix,means):
    key=base.signature(prefix)
    if key not in means:return None,None
    row=features(prefix);design=(row-np.asarray(model['center']))/np.asarray(model['scale'])
    return max(1.,math.expm1(means[key]+float(design@np.asarray(model['coefficients'])))),max(1.,math.expm1(means[key]))


def evaluate(model,rows):
    means={tuple(row['key']):row['value'] for row in model['signature_mean_log1p_ns']}
    actual,predicted,baseline=[],[],[];unknown=0;groups=defaultdict(lambda:[[],[],[]])
    for row in rows:
        value=base.target(row,model['target'])
        if value is None:continue
        prefix=row['prefix_features'];estimate,signature_baseline=prediction(model,prefix,means)
        if estimate is None:unknown+=1;continue
        actual.append(value);predicted.append(estimate);baseline.append(signature_baseline)
        context=prefix['context_event_prefix_at_cutoff'];holder=context['owner']
        relation='own' if holder==prefix['owner'] else 'other' if holder else 'unknown'
        for group in ('owner='+prefix['owner'],'phase='+prefix['semantic_phase'],'context='+relation):
            arrays=groups[group];arrays[0].append(value);arrays[1].append(estimate);arrays[2].append(signature_baseline)
    return dict(requested_eligible=sum(base.target(row,model['target']) is not None for row in rows),unknown_signatures=unknown,
        candidate=base.stats(actual,predicted),signature_baseline=base.stats(actual,baseline),
        strata=[dict(stratum=key,candidate=base.stats(*arrays[:2]),signature_baseline=base.stats(arrays[0],arrays[2])) for key,arrays in sorted(groups.items())])


def run(evidence_report,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    report=read(evidence_report)
    if report.get('training_only') is not True or report.get('validation_labels_used') is not False:
        raise ValueError('Training-only endpoint evidence required')
    refs=[receipt(evidence_report),report['kernel_rows'],report['launch_vocabulary'],receipt(__file__),receipt(base.__file__)]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Changed endpoint training source')
    vocabulary=read(report['launch_vocabulary']['path'])
    rows=[];issued=defaultdict(list);counts=Counter()
    with gzip.open(report['kernel_rows']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);prefix=row['prefix_features'];labels=row['observation_labels'];counts['source_rows']+=1
            capture,owner=row['capture'],prefix['owner'];stream_id=vocabulary[prefix['launch_signature']]['stream']
            returned,end=prefix['cutoff_ns'],labels['observed_kernel_end_ns']
            if returned<end:
                issued[capture,owner,stream_id].append((returned,end));counts['measured_issued_positive_intervals']+=1
            else:counts['already_complete_at_API_return']+=1
            if base.selected(row):
                row['source_stream_id']=stream_id;rows.append(row);counts['selected_rows']+=1
    by_capture=defaultdict(dict)
    for (capture,owner,stream_id),spans in issued.items():by_capture[capture][owner,stream_id]=spans
    indexes={capture:IssuedQueuePrefix(spans) for capture,spans in by_capture.items()}
    for row in rows:
        prefix=row['prefix_features'];current=indexes[row['capture']].counts(prefix['cutoff_ns']);owner=prefix['owner']
        prefix['measured_issued_unfinished_own_count']=sum(value for (alias,_),value in current.items() if alias==owner)
        prefix['measured_issued_unfinished_other_owner_count']=sum(value for (alias,_),value in current.items() if alias!=owner)
        prefix['measured_issued_unfinished_same_stream_count']=current.get((owner,row['source_stream_id']),0)
        if min(prefix[key] for key in ('measured_issued_unfinished_own_count','measured_issued_unfinished_other_owner_count',
                'measured_issued_unfinished_same_stream_count'))<0:raise ValueError('Negative source issued queue')
    captures=sorted(indexes)
    folds=[]
    for heldout in captures:
        training=[row for row in rows if row['capture']!=heldout];test=[row for row in rows if row['capture']==heldout]
        for kind in base.TARGETS:
            model=fit(training,kind)
            folds.append(dict(heldout_training_capture=heldout,target=kind,training_eligible_n=model['training_eligible_n'],**evaluate(model,test)))
    models=[fit(rows,kind) for kind in base.TARGETS]
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Endpoint issued-queue source bytes changed')
    output.mkdir(parents=True)
    sample_path=output/'selected-issued-queue-prefix-labels.jsonl.gz'
    with gzip.open(sample_path,'wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    model=dict(schema=SCHEMA,training_only=True,models=models,implementation=receipt(__file__),sources=refs,
        source_sample=receipt(sample_path),sample_rule=f'fixed SHA256 modulo {base.SAMPLE_MODULUS}',
        queue_scope='Measured kernel launch API return through observed END, by owner and stream; API-in-progress and unmeasured CUDA operations excluded',
        physical_service_fit=False,reference_work_rate=False)
    (output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    result=dict(schema=SCHEMA,status='ISSUED_QUEUE_ENDPOINT_TRAINING_DIAGNOSTIC',counts=dict(counts),folds=folds,
        model=receipt(output/'model.json'),sample=receipt(sample_path),source_receipts_before=refs,source_receipts_after=after,
        limitation='Offline timestamp-prefix lower bounds for measured issued kernel queue, not full CUDA readiness or runtime observer delivery. Two captures share request templates. No ongoing conditional survival update or queue-engine integration.',
        GPU_started=False,validation_labels_used=False,independent_accuracy_validated=False,
        integrated_into_forecast=False,physical_service_fit=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();value=run(args.evidence_report,args.output)
    print(json.dumps(dict(counts=value['counts'],folds=[{key:value for key,value in fold.items() if key!='strata'} for fold in value['folds']]),indent=2))
