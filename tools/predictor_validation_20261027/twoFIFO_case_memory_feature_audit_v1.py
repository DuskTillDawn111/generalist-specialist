"""Feature-only structural audit; never access owned-memory/power labels."""
import copy
import json
import math
from pathlib import Path
import numpy as np

from .stage_response_features_v1 import receipt,checked,ordered_events
from .implementation_source_closure_v27 import implementation_sources,strict_references

TEXT_OWNERS=('text#0','text#1','generalist_2#0')


def feature_rows(raw):
    # Whitelist request/event fields before computation. No board_samples,
    # resource facts, output correctness or native memory target is accessed.
    fields=('request_id','model','task','pool_id','profile_job_id')
    def view(row):return dict({k:row[k] for k in fields},execution_events=[dict(e) for e in row['execution_events']])
    priming=[view(r) for r in raw['priming_request_records']];measured=[view(r) for r in raw['request_records']]
    rows=priming+measured;by_request={};missing=[]
    for row in rows:
        if row['model'] not in TEXT_OWNERS:continue
        prompt=None;maximum=0
        for event in row['execution_events']:
            if event.get('event')!='start' or event.get('semantic_phase') not in ('prefill','decode'):continue
            b,s,index=(event.get(k) for k in ('input_batch_size','input_sequence_length','forward_index'))
            if any(type(v) is not int or v<0 for v in (b,s,index)):
                missing.append(row['request_id']);continue
            if event['semantic_phase']=='prefill':prompt=s;demand=b*s
            elif prompt is not None:demand=b*(prompt+index)
            else:missing.append(row['request_id']);continue
            maximum=max(maximum,demand)
        by_request[row['request_id']]=maximum
    initial={o:max([by_request[r['request_id']] for r in priming if r['model']==o],default=0) for o in TEXT_OWNERS}
    prior=copy.deepcopy(initial);events=ordered_events(rows);result=[]
    for window in raw['measurement_windows']:
        cid=window['case_id'];start,end=window['start_ns'],window['fixed_drain_deadline_ns']
        case=[r for r in measured if r['profile_job_id']==cid]
        demands={o:max([by_request[r['request_id']] for r in case if r['model']==o],default=0) for o in TEXT_OWNERS}
        complete={o:max(prior[o],demands[o]) for o in TEXT_OWNERS}
        excess={o:max(0,demands[o]-prior[o]) for o in TEXT_OWNERS}
        active={};concurrent=prefill=0;exposure=0;last=start
        for event in events:
            at=event['collector_processing_time_ns']
            if at>end:break
            if at>start:
                n=len(active);f=sum(p in ('prefill','ocr_inference') for p in active.values())
                exposure+=(at-last)*(n*(n-1)/2+f);last=at
            phase=event.get('semantic_phase');segment=event.get('segment_id');owner=event['owner']
            if event.get('event')=='start' and (segment in ('native_inference','native_ocr') or phase in ('prefill','decode')):
                active[owner]='entry' if segment=='native_inference' else 'ocr_inference' if segment=='native_ocr' else phase
            elif event.get('event')=='end' and segment in ('native_inference','native_ocr'):active.pop(owner,None)
            if at>=start:
                concurrent=max(concurrent,len(active));prefill=max(prefill,sum(p in ('prefill','ocr_inference') for p in active.values()))
        n=len(active);f=sum(p in ('prefill','ocr_inference') for p in active.values());exposure+=(end-last)*(n*(n-1)/2+f)
        result.append(dict(case_id=cid,prior_shape_envelope=copy.deepcopy(prior),case_shape_max=demands,
            complete_prior_plus_case_envelope=complete,positive_case_excess_over_prior=excess,
            C=concurrent,F=prefill,R_over_horizon=exposure/(end-start)))
        prior=complete
    return dict(initial_priming_shape_envelope=initial,rows=result,missing_shape_request_ids=sorted(set(missing)))


def matrix_audit(rows,kind):
    if kind=='old_cumulative_H':
        columns=['intercept']+['log1p_cumulative_H_'+o for o in TEXT_OWNERS]
        matrix=[[1.]+[math.log1p(r['complete_prior_plus_case_envelope'][o]) for o in TEXT_OWNERS] for r in rows]
    elif kind=='case_C_F_R':
        columns=['intercept','C','F','R_over_horizon'];matrix=[[1.,r['C'],r['F'],r['R_over_horizon']] for r in rows]
    elif kind=='case_demand_C_R':
        columns=['intercept','sum_log1p_case_H','C','R_over_horizon']
        matrix=[[1.,sum(math.log1p(r['case_shape_max'][o]) for o in TEXT_OWNERS),r['C'],r['R_over_horizon']] for r in rows]
    else:raise ValueError('Exact pre-fit feature family required')
    a=np.array(matrix,dtype=float);scaled=a.copy()
    for i in range(1,a.shape[1]):
        norm=np.linalg.norm(a[:,i]);scaled[:,i]/=norm if norm else 1
    sv=np.linalg.svd(scaled,compute_uv=False)
    return dict(columns=columns,rows=len(rows),numeric_rank=int(np.linalg.matrix_rank(scaled)),singular_values=sv.tolist(),
        unique_values_per_column={name:len(set(a[:,i].tolist())) for i,name in enumerate(columns)},
        memory_target_used=False)


def run(raw_path,output):
    ref=receipt(raw_path);raw=checked(ref);features=feature_rows(raw)
    result=dict(schema='gsdeploy.twoFIFO_case_memory_feature_only_audit.v1',raw_provenance=ref,features=features,
        matrices={kind:matrix_audit(features['rows'],kind) for kind in ('old_cumulative_H','case_C_F_R','case_demand_C_R')},
        board_samples_accessed=False,memory_target_used=False,power_target_used=False,native_memory_fitted=False,
        actual_future_trajectory_used_for_feature_structure_only=True,prospective_prediction_validated=False,
        validation_labels_read=False,GPU_started=False,sources=strict_references(implementation_sources(__file__)))
    out=Path(output);out.mkdir(parents=True,exist_ok=False);path=out/'report.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return receipt(path)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.raw,args.output)))
