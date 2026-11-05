"""Training-only comparisons of causal input/activity descriptors.

This is an exploratory diagnostic, never an admitted forecasting provider.
In particular, elapsed-call features below do not constitute observed kernels.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.forecasting.native_context import categories
from gsdeploy.forecasting.native_order import NativeOrderLaw
from gsdeploy.forecasting.native_training import digest


def observations(fit):
    """Sweep verified complete spans, preserving causal owner order and starts."""
    template=fit['profiles'];profiles={p['id']:p for p in template['profiles']}
    segments={(p['id'],s['id']):s for p in profiles.values() for s in p['segments']}
    order_spec=template.get('native_order_law')
    order_law=NativeOrderLaw(order_spec,template['execution_context_sha256']) if order_spec else None
    result=[]
    for doc in fit['normalized_runs']:
        if doc['split']!='training':raise ValueError('Training-only diagnosis required')
        events=defaultdict(list);targets=[];orders=defaultdict(int)
        for row in doc['requests']:
            ordinal=orders[row['alias']];orders[row['alias']]+=1
            for s in row['segments']:
                segment=segments[row['profile_id'],s['segment_id']]
                if segment['kind']!='gpu':continue
                target=dict(alias=row['alias'],profile_id=row['profile_id'],segment_id=s['segment_id'],
                    start_ns=s['start_ns'],end_ns=s['end_ns'],ordinal=ordinal,
                    id=doc['run_id']+':'+row['request_id']+':'+s['segment_id'],run_id=doc['run_id'],intervals=[])
                baseline=segment['duration_ms']['mean']
                if order_law:baseline*=order_law.factor(profiles[row['profile_id']],segment,ordinal)
                target['baseline_ms']=baseline;targets.append(target)
                events[s['start_ns']].append(('start',target));events[s['end_ns']].append(('end',target))
        active={};previous=None
        for timestamp in sorted(events):
            if previous is not None and timestamp>previous:
                snapshot=[{k:t[k] for k in ('alias','profile_id','start_ns','baseline_ms')} for t in active.values()]
                for target in active.values():
                    target['intervals'].append(dict(start_ns=previous,end_ns=timestamp,active=snapshot))
            for kind,target in sorted(events[timestamp],key=lambda pair:pair[0]):
                if kind=='end':
                    if active.pop(target['alias']) is not target:raise ValueError('Native owner overlap')
                else:
                    if target['alias'] in active:raise ValueError('Concurrent calls on one owner')
                    active[target['alias']]=target
            previous=timestamp
        result.extend(targets)
    return profiles,result


def diagnose(fit):
    profiles,rows=observations(fit);names=categories(profiles)
    median={key:float(np.median([s['duration_ms']['mean'] for p in profiles.values()
        if (p['family'],p['task'])==key for s in p['segments'] if s['kind']=='gpu'])) for key in names}
    grouped=defaultdict(list)
    for row in rows:
        p=profiles[row['profile_id']];grouped[p['family'],p['task'],row['segment_id']].append(row)
    reports=[]
    for key,cohort in sorted(grouped.items()):
        expanded=[];weights=[];owner=[]
        for i,row in enumerate(cohort):
            target_log=math.log(row['baseline_ms']/median[key[:2]])
            for interval in row['intervals']:
                midpoint=(interval['start_ns']+interval['end_ns'])/2
                counts=np.zeros(len(names));size=np.zeros(len(names));age=np.zeros(len(names))
                for other in interval['active']:
                    if other['alias']==row['alias']:continue
                    p=profiles[other['profile_id']];j=names.index((p['family'],p['task']))
                    counts[j]+=1;size[j]+=math.log(other['baseline_ms']/median[names[j]])
                    # Causal age from observed start and the training solo baseline.
                    # Actual finish, response tokens and future clocks are absent.
                    age[j]+=math.log1p(max(0,(midpoint-other['start_ns'])/1e6/other['baseline_ms']))
                own_age=math.log1p(max(0,(midpoint-row['start_ns'])/1e6/row['baseline_ms']))
                expanded.append((counts,size,age,target_log,own_age))
                weights.append((interval['end_ns']-interval['start_ns'])/1e6/row['baseline_ms']);owner.append(i)
        owner=np.array(owner);weights=np.array(weights);target=np.ones(len(cohort))
        used=[i for i in range(len(names)) if any(x[0][i] for x in expanded)]
        for mode in ('activity','input','input_and_causal_age','input_and_age_interactions'):
            matrix=[]
            pairs=[(i,j) for i in used for j in used if i<j and any(z[0][i]*z[0][j] for z in expanded)]
            for counts,size,age,target_log,own_age in expanded:
                x=[1.,*[counts[i] for i in used],*[counts[i]*counts[j] for i in used for j in used if i<j and
                    any(z[0][i]*z[0][j] for z in expanded)]]
                if mode!='activity':x.extend([counts[i]*target_log for i in used]);x.extend([size[i] for i in used])
                if mode in ('input_and_causal_age','input_and_age_interactions'):x.extend([age[i] for i in used])
                if mode=='input_and_age_interactions':
                    x.extend([age[i]**2 for i in used]);x.extend([counts[i]*own_age for i in used])
                    x.extend([age[i]*counts[j] for i,j in pairs]);x.extend([counts[i]*age[j] for i,j in pairs])
                    x.extend([age[i]*target_log for i in used])
                matrix.append(x)
            matrix=np.array(matrix)
            nonzero=np.max(np.abs(matrix),axis=0)>1e-10;matrix=matrix[:,nonzero]
            def residual(beta):return np.bincount(owner,weights=weights*np.exp(matrix@beta),minlength=len(cohort))-target
            def jac(beta):
                z=(weights*np.exp(matrix@beta))[:,None]*matrix
                return np.array([np.bincount(owner,weights=z[:,j],minlength=len(cohort)) for j in range(z.shape[1])]).T
            result=least_squares(residual,np.zeros(matrix.shape[1]),jac=jac,bounds=(-5.,5.),max_nfev=500)
            errors=np.abs(residual(result.x));runs=sorted({r['run_id'] for r in cohort})
            reports.append(dict(family=key[0],task=key[1],segment_id=key[2],descriptor=mode,
                parameters=matrix.shape[1],rank=int(np.linalg.matrix_rank(result.jac)),active_bound_indices=np.flatnonzero(result.active_mask).tolist(),
                optimizer_success=bool(result.success),max_work_error=float(max(errors)),p95_work_error=float(np.quantile(errors,.95)),
                mean_work_error=float(np.mean(errors)),independent_runs=runs,
                worst=[dict(id=cohort[i]['id'],error=float(errors[i])) for i in np.argsort(errors)[-5:][::-1]],
                executable_provider=False,admission_enabled=False))
            print(json.dumps(reports[-1]),flush=True)
    return dict(schema='gsdeploy.native_descriptor_diagnosis.v1',fit_sha256=digest(fit),
        split='training',admission_enabled=False,accuracy_qualified=False,
        age_feature_scope='host elapsed native-call time; no kernel-stage or physical-cause claim',reports=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();report=diagnose(json.loads(args.fit.read_text()))
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
