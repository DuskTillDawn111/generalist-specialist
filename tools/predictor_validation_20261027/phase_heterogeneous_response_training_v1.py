"""Fixed small phase/family response design, full three training captures.

All response slopes are freely fitted; no empirical slowdown multipliers,
cohort/context lookup, unsupported macro deletion or validation data.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from . import macro_joint_response_fit_v1 as base
from . import macro_projection_contrast_training_v1 as numeric
from .implementation_source_closure_v27 import strict_references,implementation_sources

SCHEMA='gsdeploy.phase_heterogeneous_response_training.v1'
CATEGORIES=('text:entry','text:prefill','text:decode','text:ready_tail',
    'generalist_2:entry','generalist_2:prefill','generalist_2:decode','generalist_2:ready_tail',
    'ocr:entry','ocr:ocr_inference','ocr:ready_tail')
COUNT='peer_visible_macro_owner_count';SAME='peer_same_visible_macro_phase_owner_count'
DESIGNS={
    'global_phase_aware':(COUNT,SAME),
    'own_family_phase_count_global_same':tuple('count@'+c for c in CATEGORIES)+(SAME,),
    'own_family_phase_count_and_same':tuple(k+'@'+c for c in CATEGORIES for k in ('count','same')),
}


def category(owner,phase):
    result=owner.split('#')[0]+':'+phase
    base.require(result in CATEGORIES,'Explicit fixed family/macro-phase training support required')
    return result


def vector(owner,phase,state,features):
    cat=category(owner,phase);n=state[COUNT];same=state[SAME]
    base.require(type(n) is int and type(same) is int and 0<=same<=n,'Exact processed peer counts required')
    return [n if f==COUNT else same if f==SAME else
        (n if f.startswith('count@') else same) if f.split('@',1)[1]==cat else 0 for f in features]


def prepared(documents,features):
    rows=base.prepared(documents,());macros=[m for d in documents for r in d['requests'] for m in r['macros']]
    base.require(len(rows)==len(macros),'Complete macro denominator required')
    for row,macro in zip(rows,macros):
        cat=category(macro['owner'],macro['semantic_phase']);totals=defaultdict(float);ordered=[];durations=[]
        for piece in macro['causal_covariate_path']:
            values=tuple(vector(macro['owner'],macro['semantic_phase'],piece['state'],features))
            duration=piece['duration_ns']/1e6;totals[values]+=duration;ordered.append(values);durations.append(duration)
        endpoint=ordered[-1];keys=[k for k in totals if k!=endpoint]+[endpoint]
        row.update(category=cat,durations=np.asarray([totals[k] for k in keys]),
            matrix=np.asarray(keys,dtype=float).reshape(len(keys),len(features)),
            ordered_durations=np.asarray(durations),ordered_matrix=np.asarray(ordered,dtype=float).reshape(len(ordered),len(features)))
    return rows


def residual_diagnostics(documents,fit):
    rows=base.prepared(documents,(COUNT,SAME));beta=np.asarray(fit['coefficients']);mu={tuple(x['group']):x['log_reference_ms'] for x in fit['baselines']}
    macros=[m for d in documents for r in d['requests'] for m in r['macros']];metadata={};sources=[]
    for doc in documents:
        review=base.checked(doc['sources'][0]);raw_ref=review['inputs']['raw.json'];raw=base.checked(raw_ref);plan=base.checked(raw['plan'])
        jobs=base.checked(raw['executed_jobs'])['jobs'];byjob={j['id']:j for j in jobs}
        order={r['request_id']:i for i,r in enumerate(sorted(raw['request_records'],key=lambda r:r['observed_worker_start_time_ns']))}
        for row in raw['request_records']:
            job=byjob[row['profile_job_id']]
            # Context/job IDs are diagnostic strata only, never fitted inputs.
            metadata[raw_ref['sha256'],row['request_id']]=dict(source_job_id=row['profile_job_id'],
                diagnostic_job_fields={k:v for k,v in job.items() if k not in ('id','inputs','tasks')},
                source_worker_start_rank=order[row['request_id']])
        sources.extend([raw_ref,raw['plan'],raw['executed_jobs']])
    records=[]
    for row,macro in zip(rows,macros):
        logv=float(np.logaddexp.reduce(np.log(row['durations'])-row['matrix']@beta));residual=logv-mu[row['group']]
        duration=sum(p['duration_ns'] for p in macro['causal_covariate_path']);peerphases=defaultdict(float)
        for piece in macro['causal_covariate_path']:
            for peer in piece['state']['peer_macros']:peerphases[peer['phase']]+=piece['duration_ns']/duration
        records.append(dict(capture=row['capture'],request_id_for_diagnostics_only=row['request_id'],owner=row['owner'],
            macro_id=row['macro_id'],phase=row['phase'],category=category(row['owner'],row['phase']),
            log_reference_work_residual=residual,
            average_peer_count=sum(p['duration_ns']*p['state'][COUNT] for p in macro['causal_covariate_path'])/duration,
            average_same_phase_count=sum(p['duration_ns']*p['state'][SAME] for p in macro['causal_covariate_path'])/duration,
            peer_phase_exposure_ratios=dict(peerphases),**metadata[row['capture'],row['request_id']]))
    def grouped(keys):
        groups=defaultdict(list)
        for r in records:groups[tuple(r[k] for k in keys)].append(r)
        return [dict(**dict(zip(keys,key)),rows=len(rs),mean_log_residual=float(np.mean([r['log_reference_work_residual'] for r in rs])),
            RMS_log_residual=float(np.sqrt(np.mean([r['log_reference_work_residual']**2 for r in rs])))) for key,rs in sorted(groups.items(),key=lambda x:str(x[0]))]
    return dict(records=records,by_category=grouped(['category']),by_category_capture=grouped(['category','capture']),
        by_source_job_diagnostic_only=grouped(['source_job_id']),context_ids_are_prediction_inputs=False),sources


def run(paths,prior_fit_path,output):
    prior_ref=base.receipt(prior_fit_path);prior=base.checked(prior_ref)
    base.require(prior['schema']=='gsdeploy.macro_joint_response_second_order_training_leaveout.v1'
        and prior['validation_labels_read'] is False,'Qualified original phase-aware training response report required')
    for ref in prior['sources']:base.require(base.receipt(ref['path'])==ref,'Prior source closure changed')
    docs=[];sources=[prior_ref,*prior['sources']]
    for path in paths:
        doc,closure=base.verified_document(path);docs.append(doc);sources.extend(closure)
    captures=sorted({m['capture_sha256'] for d in docs for r in d['requests'] for m in r['macros']})
    base.require(len(docs)==len(captures)==3,'Three complete distinct training captures required')
    original=next(c for c in prior['candidates'] if c['candidate']=='visible_phase_aware')
    diagnostics,refs=residual_diagnostics(docs,original);sources.extend(refs)
    results=[]
    for name,features in DESIGNS.items():
        rows=prepared(docs,features);full,beta,mu,variance=numeric.fit(rows,features);folds=[];pooled=[]
        for heldout in captures:
            fitted,b,m,v=numeric.fit([r for r in rows if r['capture']!=heldout],features)
            evaluated=base.evaluate([r for r in rows if r['capture']==heldout],b,m,v)
            pooled.extend(evaluated['records']);folds.append(dict(heldout_capture_sha256=heldout,training_fit=fitted,evaluation=evaluated))
        full.update(candidate=name,whole_capture_leaveout=folds,pooled_leaveout=base.summarize(pooled),
            whole_capture_numerically_qualified=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification'] for f in folds)),
            all_macro_nodes_retained=len(rows)==1002,coefficients_are_free_real=True)
        results.append(full)
    closure=strict_references([sources,implementation_sources(__file__)])
    base.require(all(base.receipt(r['path'])==r for r in closure),'Source changed during phase-heterogeneous training')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'residual-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    report=dict(schema=SCHEMA,status='FIXED_PHASE_FAMILY_RESPONSE_DESIGN_WHOLE_TRAINING_CAPTURE_DIAGNOSTICS',
        sources=closure,candidates=results,residual_diagnostics=base.receipt(output/'residual-diagnostics.json'),
        design_fixed_before_this_fit=True,capture_replication_units=3,macro_rows=1002,
        conditional_NLL_does_not_qualify_generated_queue=True,context_or_job_lookup_used_for_fit=False,
        main_queue_provider_eligible=False,independent_validation_read=False,validation_labels_read=False,GPU_started=False,
        final_gain_eligible=False,selected_for_prospective_provider=None,
        next_step='Compare numerically qualified candidates on whole-capture generated-path request ensembles with unknown denominators retained.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return base.receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training-macros',nargs='+',required=True);p.add_argument('--prior-fit',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.training_macros,a.prior_fit,a.output)))
