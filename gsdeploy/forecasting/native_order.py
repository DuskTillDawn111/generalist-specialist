"""Empirical post-prime request-order effects; no cache/DVFS causality claim."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from ..forecast_errors import UnknownEvidence

SCHEMA='gsdeploy.native_post_prime_order.v1'


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def request_orders(raw):
    counts=defaultdict(int);orders={}
    for row in raw['request_records']:
        alias=row['model'];rid=row['request_id']
        if rid in orders:raise ValueError('Unique native order request identity required')
        orders[rid]=counts[alias];counts[alias]+=1
    return orders


class NativeOrderLaw:
    def __init__(self,spec,context):
        if (spec.get('schema')!=SCHEMA or spec.get('status')!='FITTED_UNVALIDATED'
                or spec.get('execution_context_sha256')!=context
                or spec.get('descriptor')!='post_prime_owner_request_ordinal'
                or spec.get('physical_cause_identified') is not False):
            raise ValueError('Explicit matched empirical native order law required')
        self.groups={}
        for g in spec['groups']:
            key=tuple(g[k] for k in ('family','task','segment_id'))
            if (key in self.groups or not g['accepted'] or len(set(g['independent_run_ids']))<2
                    or not 0<g['tau_requests']<=32 or type(g['saturation_ordinal']) is not int
                    or g['saturation_ordinal']<1 or not math.isfinite(g['coefficient'])
                    or abs(g['coefficient'])>2):
                raise ValueError('Identifiable independently repeated native order group required')
            self.groups[key]=g

    def factor(self,profile,segment,ordinal):
        group=self.groups.get((profile['family'],profile['task'],segment['id']))
        if group is None:return 1.
        if type(ordinal) is not int or ordinal<0:
            raise UnknownEvidence('UNKNOWN_POST_PRIME_OWNER_REQUEST_ORDINAL')
        feature=math.exp(-min(ordinal,group['saturation_ordinal'])/group['tau_requests'])
        return math.exp(group['coefficient']*feature)


def fit_order_baselines(template,solo_bundles,*,families=('ocr',),max_relative_error=.2):
    """Fit input intercepts and order trend using separately launched solo runs.

    The decay scale is selected by leave-one-input-out log-ratio prediction.
    The capped descriptor is an explicit saturation assumption. Confirmation
    data cannot enter this entry point or select its decay scale.
    """
    from .native_training import validate_capture,stage_layout
    result=copy.deepcopy(template);profile_map={(p['family'],p['task'],p['pool_id']):p for p in result['profiles']}
    grouped=defaultdict(list);seen=set()
    for bundle in solo_bundles:
        if (validate_capture(bundle)!=template['execution_context_sha256'] or bundle['plan']['mode']!='solo'
                or bundle['raw']['run_id'] in seen):
            raise ValueError('Distinct matched training solo runs required for order effects')
        raw=bundle['raw'];seen.add(raw['run_id']);orders=request_orders(raw)
        for row in raw['request_records']:
            family=raw['resident_aliases'][row['model']]
            if family not in families:continue
            profile=profile_map[(family,row['task'],row['pool_id'])]
            for span in stage_layout(row):
                if span['kind']=='gpu':grouped[(family,row['task'],span['id'])].append(dict(
                    run_id=raw['run_id'],profile_id=profile['id'],ordinal=orders[row['request_id']],
                    duration_ms=(span['end_ns']-span['start_ns'])/1e6,request_id=row['request_id']))
    groups=[];diagnostics=[]
    for (family,task,segment_id),rows in sorted(grouped.items()):
        by_input=defaultdict(list)
        for row in rows:by_input[row['profile_id']].append(row)
        if len(seen)<2 or any(len({r['run_id'] for r in rs})<2 for rs in by_input.values()):
            raise ValueError('At least two independently repeated observations per order input required')
        candidates=[]
        for tau in (1.,2.,4.,8.,16.):
            differences=[]
            for ident,rs in by_input.items():
                # Independent-run contrasts remove the unknown input intercept.
                for other in rs[1:]:
                    differences.append((ident,math.exp(-min(other['ordinal'],32)/tau)-math.exp(-min(rs[0]['ordinal'],32)/tau),
                                        math.log(other['duration_ms']/rs[0]['duration_ms'])))
            x=np.array([r[1] for r in differences]);y=np.array([r[2] for r in differences]);den=float(x@x)
            if den<1e-10:continue
            beta=float(x@y/den);cv=[];valid=True
            for ident in by_input:
                keep=np.array([r[0]!=ident for r in differences]);d=float(x[keep]@x[keep])
                if d<1e-10:valid=False;break
                held_beta=float(x[keep]@y[keep]/d)
                cv.extend((y[~keep]-held_beta*x[~keep]).tolist())
            if valid:
                errors=[]
                for rs in by_input.values():
                    intercept=float(np.mean([math.log(r['duration_ms'])-beta*math.exp(-min(r['ordinal'],32)/tau) for r in rs]))
                    errors.extend(abs(r['duration_ms']/math.exp(intercept+beta*math.exp(-min(r['ordinal'],32)/tau))-1) for r in rs)
                candidates.append(dict(tau_requests=tau,coefficient=beta,cv_log_ratio_rmse=float(np.sqrt(np.mean(np.array(cv)**2))),
                    maximum_solo_residual=max(errors),solo_gate_passed=bool(abs(beta)<=2 and max(errors)<=max_relative_error)))
        diagnostic=dict(family=family,task=task,segment_id=segment_id,accepted=False,candidates=candidates)
        if not candidates:
            diagnostic['reason']='UNIDENTIFIABLE_ORDER_CONTRAST';diagnostics.append(diagnostic);continue
        eligible=[c for c in candidates if c['solo_gate_passed']]
        chosen=min(eligible or candidates,key=lambda r:r['cv_log_ratio_rmse']);beta=chosen['coefficient'];tau=chosen['tau_requests']
        diagnostic['selection_policy']='minimum leave-input-out contrast error among candidates satisfying the unchanged solo residual gate'
        input_baselines={};errors=[]
        for ident,rs in by_input.items():
            baseline=math.exp(float(np.mean([math.log(r['duration_ms'])-beta*math.exp(-min(r['ordinal'],32)/tau) for r in rs])))
            input_baselines[ident]=baseline
            errors.extend(abs(r['duration_ms']/(baseline*math.exp(beta*math.exp(-min(r['ordinal'],32)/tau)))-1) for r in rs)
        diagnostic.update(chosen=chosen,max_relative_error=max(errors),independent_run_ids=sorted(seen),inputs=len(by_input))
        if abs(beta)>2 or max(errors)>max_relative_error:
            diagnostic['reason']='EXCESSIVE_ORDER_BASELINE_RESIDUAL';diagnostics.append(diagnostic);continue
        diagnostic['accepted']=True;diagnostics.append(diagnostic)
        groups.append(dict(family=family,task=task,segment_id=segment_id,accepted=True,**chosen,
            saturation_ordinal=32,independent_run_ids=sorted(seen),input_baseline_ms=input_baselines,
            observations=rows,max_relative_error=max(errors)))
        for profile in result['profiles']:
            if profile['id'] in input_baselines:
                for segment in profile['segments']:
                    if segment['id']==segment_id:segment['duration_ms']['mean']=input_baselines[profile['id']]
    result['native_order_law']=dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
        execution_context_sha256=template['execution_context_sha256'],descriptor='post_prime_owner_request_ordinal',
        physical_cause_identified=False,groups=groups,fit_diagnostics=diagnostics,
        assumption='exponential request-order descriptor saturates after ordinal 32; independent validation required')
    NativeOrderLaw(result['native_order_law'],template['execution_context_sha256'])
    return result
