"""CPU strict coverage and bounded supplemental-capture eligibility analysis."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from .kernel_baseline_intake_v3 import checked,_verify
from .native_collect import receipt,save


def review(fit_path,output,selected_pool_id):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=receipt(fit_path);fit=checked(source);doc=checked(fit['models']);requests=checked(fit['request_sequences'])
    _verify(fit['observations']);points=defaultdict(list)
    with Path(fit['observations']['path']).open() as handle:
        for line in handle:
            row=json.loads(line);points[(row['group_id'],tuple(sorted(row['features'].items())))].append(row)
    impossible=Counter();conflicting=[]
    for (group_id,features),rows in points.items():
        low=min(r['duration_ns'] for r in rows);high=max(r['duration_ns'] for r in rows)
        if 0.75*high>1.25*low:
            impossible['exact_feature_points']+=1;impossible['observations']+=len(rows)
            conflicting.append(dict(group_id=group_id,feature_values=dict(features),min_duration_ns=low,max_duration_ns=high,
                observation_count=len(rows),capture_run_count=len({r['run_id'] for r in rows})))
    summaries=[];plan=[]
    for request in requests:
        phase=defaultdict(Counter);counts=Counter();group_counts=Counter()
        for row in request['ordered_rows']:
            group=doc['groups'][row['group_id']];model=doc['models'][row['group_id']]
            label='ACCEPTED_TRAINING_FIT_AT_ACTUAL_TRAINING_POINT' if model['fit_accepted'] else '+'.join(model['reasons'])
            counts[label]+=1;phase[group['semantic_phase']][label]+=1;group_counts[row['group_id']]+=1
        summaries.append(dict(run_id=request['run_id'],alias=request['alias'],pool_id=request['pool_id'],request_id=request['request_id'],
            kernel_count=len(request['ordered_rows']),strict_counts=dict(counts),phase_counts={k:dict(v) for k,v in phase.items()},
            scope='Actual training feature points are in their own feature hull; accepted training fit is not independent validation'))
        if request['pool_id']==selected_pool_id:
            for key,count in group_counts.items():
                model=doc['models'][key]
                if model['reasons']==['INSUFFICIENT_OBSERVATIONS_OR_RUNS']:
                    needed=next(extra for extra in range(1,4) if model['observation_count']+extra*count>=3 and model['run_count']+extra>=2)
                    plan.append(dict(group_id=key,kernel_name=doc['groups'][key]['kernel_name'],
                        phase=doc['groups'][key]['semantic_phase'],current_observations=model['observation_count'],current_runs=model['run_count'],
                        affected_launches_per_exact_repeat=count,minimum_additional_same_work_captures_for_eligibility=needed,
                        eligibility_does_not_guarantee_residual_pass=True))
    save(output/'exact-feature-duration-conflicts.json',conflicting)
    result=dict(schema='gsdeploy.semantic_kernel_strict_training_coverage_review.v5',fit=source,requests=summaries,
        selected_pool_id=selected_pool_id,selected_pool_supplemental_eligibility_plan=plan,
        one_extra_exact_repeat_eligible_launches=sum(p['affected_launches_per_exact_repeat'] for p in plan if p['minimum_additional_same_work_captures_for_eligibility']==1),
        two_extra_exact_repeats_eligible_launches=sum(p['affected_launches_per_exact_repeat'] for p in plan if p['minimum_additional_same_work_captures_for_eligibility']==2),
        exact_feature_impossibility_counts=dict(impossible),conflicts=receipt(output/'exact-feature-duration-conflicts.json'),
        recommendation='Do not launch an unbounded shape campaign. One second963capture can test repeatability and make64newAmpere+5decode reduction launches eligible; three singleton launches still need a third observation. It cannot repair existing7121selected-request residual failures or identify unseen1025dispatch/resource domains. Audit source-derived non-affine shape/stride/cache metadata and residual family first.',
        numerical_thresholds_unchanged=True,additional_GPU_capture_started=False,model_refitted=False,
        heldout_validation_performed=False,independent_end_to_end20percent_passed=False,
        implementation_sources=[receipt(__file__)])
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-pool-id',required=True)
    a=p.parse_args();r=review(a.fit,a.output,a.selected_pool_id)
    print({k:r[k] for k in ('one_extra_exact_repeat_eligible_launches','two_extra_exact_repeats_eligible_launches','exact_feature_impossibility_counts')})


if __name__=='__main__':main()
