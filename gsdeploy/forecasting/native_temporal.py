"""Empirical causal native-call age and input descriptors.

Age bins describe host elapsed time within an observed native call. They are
not kernel phases. Only training receipts fit the law; heldout replay decides
whether this more flexible development model is useful.
"""
from __future__ import annotations
from collections import defaultdict
import math

import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares, minimize

from ..forecast_errors import UnknownEvidence
from .native_context import categories,within_support
from .native_order import NativeOrderLaw

SCHEMA='gsdeploy.native_causal_age_progress.v1'
BOUNDED_SCHEMA='gsdeploy.native_causal_age_progress.bounded.v1'
# An empirical numerical discretization, not a subdivision into measured stages.
LEGACY_AGE_EDGES=[round(i*.1,10) for i in range(11)]+[round(1+i*.25,10) for i in range(1,29)]
AGE_EDGES=LEGACY_AGE_EDGES+[round(8+i*.25,10) for i in range(1,33)]
# Explicit alternative basis: the descriptor becomes constant after one solo
# duration. This is an empirical modeling assumption, not observed tail support
# for the original basis. The numerical domain still ends at 16 solo durations.
BOUNDED_AGE_EDGES=[round(i*.1,10) for i in range(11)]+[16.]
BOUNDED_POLICY='constant_after_one_solo_duration_unknown_at_sixteen'


def age_bin(age,edges=AGE_EDGES):
    if not math.isfinite(age) or age<0:raise UnknownEvidence('UNKNOWN_NATIVE_CALL_AGE')
    # Source clocks are integral ns; a rounded bin boundary may differ by <1 ns.
    i=int(np.searchsorted(edges,age+1e-8,side='right')-1)
    if i>=len(edges)-1:raise UnknownEvidence('NATIVE_CALL_AGE_OUTSIDE_DESCRIPTOR_DOMAIN')
    return i


def features(target,active,profiles,names,medians,*,age_edges=AGE_EDGES):
    p=profiles[target['profile_id']];key=(p['family'],p['task']);counts=np.zeros(len(names));size=counts.copy();age=counts.copy()
    baseline=target['native_baseline_ms'];own=age_bin(target['native_age_ms']/baseline,age_edges)
    for alias,other in active.items():
        if alias==target['alias']:continue
        q=profiles[other['profile_id']];segment=next(s for s in q['segments'] if s['id']==other['segment_id'])
        if segment['kind']!='gpu':continue
        j=names.index((q['family'],q['task']));counts[j]+=1
        size[j]+=math.log(other['native_baseline_ms']/medians[j])
        age[j]+=math.log1p(age_edges[age_bin(other['native_age_ms']/other['native_baseline_ms'],age_edges)])
    return dict(count=counts,size=size,age=age,own_age=math.log1p(age_edges[own]),
        target_size=math.log(baseline/medians[names.index(key)]),profile_id=p['id'])


def feature_value(x,term):
    kind,*args=term
    if kind=='one':return 1.
    if kind in ('count','size','age'):return float(x[kind][args[0]])
    if kind=='pair':return x['count'][args[0]]*x['count'][args[1]]
    if kind=='target_size_count':return x['target_size']*x['count'][args[0]]
    if kind=='target_size_age':return x['target_size']*x['age'][args[0]]
    if kind=='own_age_count':return x['own_age']*x['count'][args[0]]
    if kind=='age2':return x['age'][args[0]]**2
    if kind=='age_count':return x['age'][args[0]]*x['count'][args[1]]
    if kind=='input':return float(x['profile_id']==args[0])
    raise ValueError('Unknown native causal descriptor')


def training_rows(fit,*,age_edges=AGE_EDGES):
    template=fit['profiles'];profiles={p['id']:p for p in template['profiles']}
    segments={(p['id'],s['id']):s for p in profiles.values() for s in p['segments']}
    spec=template.get('native_order_law');order=NativeOrderLaw(spec,template['execution_context_sha256']) if spec else None
    result=[]
    for doc in fit['normalized_runs']:
        if doc['split']!='training':raise ValueError('Temporal fit accepts training only')
        events=defaultdict(list);targets=[];orders=defaultdict(int)
        for row in doc['requests']:
            ordinal=orders[row['alias']];orders[row['alias']]+=1
            for s in row['segments']:
                segment=segments[row['profile_id'],s['segment_id']]
                if segment['kind']!='gpu':continue
                baseline=segment['duration_ms']['mean']
                if order:baseline*=order.factor(profiles[row['profile_id']],segment,ordinal)
                t=dict(alias=row['alias'],profile_id=row['profile_id'],segment_id=s['segment_id'],
                    start_ns=s['start_ns'],end_ns=s['end_ns'],native_baseline_ms=baseline,
                    id=doc['run_id']+':'+row['request_id']+':'+s['segment_id'],run_id=doc['run_id'],intervals=[])
                targets.append(t);events[s['start_ns']].append(('start',t));events[s['end_ns']].append(('end',t))
                for edge in age_edges[1:]:
                    at=s['start_ns']+round(edge*baseline*1e6)
                    if at<s['end_ns']:events[at]  # explicit numerical age event
        active={};previous=None
        for at in sorted(events):
            if previous is not None and at>previous:
                snapshot={a:dict(alias=a,profile_id=t['profile_id'],segment_id=t['segment_id'],
                    native_baseline_ms=t['native_baseline_ms'],native_age_ms=(previous-t['start_ns'])/1e6)
                    for a,t in active.items()}
                for t in active.values():t['intervals'].append(((at-previous)/1e6/t['native_baseline_ms'],snapshot))
            for kind,t in sorted(events[at],key=lambda z:z[0]):
                if kind=='end':
                    if active.pop(t['alias']) is not t:raise ValueError('Native owner span mismatch')
                else:
                    if t['alias'] in active:raise ValueError('Native owner overlap')
                    active[t['alias']]=t
            previous=at
        result.extend(targets)
    return profiles,result


def fit_temporal_progress(fit,*,max_work_error=.2,ridge=.01,timing_objective=False,age_descriptor='strict-v1'):
    if age_descriptor not in ('strict-v1','bounded-terminal-v1'):
        raise ValueError('Explicit known causal age descriptor required')
    bounded=age_descriptor=='bounded-terminal-v1'
    age_edges=BOUNDED_AGE_EDGES if bounded else AGE_EDGES
    profiles,rows=training_rows(fit,age_edges=age_edges);names=categories(profiles)
    medians=[float(np.median([s['duration_ms']['mean'] for p in profiles.values()
        if (p['family'],p['task'])==key for s in p['segments'] if s['kind']=='gpu'])) for key in names]
    grouped=defaultdict(list)
    for row in rows:
        p=profiles[row['profile_id']];grouped[p['family'],p['task'],row['segment_id']].append(row)
    groups=[];diagnostics=[]
    for (family,task,segment_id),cohort in sorted(grouped.items()):
        xs=[];ws=[];owners=[];input_runs=defaultdict(set);vertex_runs=defaultdict(set)
        for i,row in enumerate(cohort):
            input_runs[row['profile_id']].add(row['run_id'])
            for weight,active in row['intervals']:
                x=features(active[row['alias']],active,profiles,names,medians,age_edges=age_edges)
                xs.append(x);ws.append(weight);owners.append(i);vertex_runs[tuple(x['count'])].add(row['run_id'])
        used=[i for i in range(len(names)) if any(x['count'][i] for x in xs)]
        pairs=[(i,j) for i in used for j in used if i<j and any(x['count'][i]*x['count'][j] for x in xs)]
        terms=[['one']]+[['count',i] for i in used]+[['pair',i,j] for i,j in pairs]
        terms+=[[kind,i] for kind in ('target_size_count','size','age','age2','own_age_count','target_size_age') for i in used]
        terms+=[['age_count',i,j] for i,j in pairs]+[['age_count',j,i] for i,j in pairs]
        # Input effects remain finite-pool empirical parameters, anchored by both
        # independently repeated solo and overlap observations, with shrinkage.
        terms+=[['input',ident] for ident in sorted(input_runs)[1:]]
        matrix=np.array([[feature_value(x,t) for t in terms] for x in xs]);owner=np.array(owners);weight=np.array(ws)
        # Remove exact linear redundancies before optimization; report them.
        _,r,piv=qr(matrix,mode='economic',pivoting=True);rank=int(np.linalg.matrix_rank(r))
        keep=sorted(piv[:rank]);removed=[terms[i] for i in range(len(terms)) if i not in keep]
        matrix=matrix[:,keep];terms=[terms[i] for i in keep]
        last_indices=[max(np.flatnonzero(owner==i)) for i in range(len(cohort))]
        terminal_matrix=matrix[last_indices];duration_ratio=np.bincount(owner,weights=weight,minlength=len(cohort))
        def residual(beta):return np.bincount(owner,weights=weight*np.exp(matrix@beta),minlength=len(cohort))-1.
        def jac(beta):
            z=(weight*np.exp(matrix@beta))[:,None]*matrix
            return np.array([np.bincount(owner,weights=z[:,j],minlength=len(cohort)) for j in range(z.shape[1])]).T
        reg=math.sqrt(ridge)
        def time_proxy(beta):return residual(beta)/duration_ratio*np.exp(-terminal_matrix@beta)
        def time_jac(beta):
            scale=np.exp(-terminal_matrix@beta)/duration_ratio
            return scale[:,None]*jac(beta)-time_proxy(beta)[:,None]*terminal_matrix
        def objective(beta):return np.r_[.5*residual(beta),time_proxy(beta),reg*beta] if timing_objective else np.r_[residual(beta),reg*beta]
        def objective_jac(beta):return np.vstack((.5*jac(beta),time_jac(beta),reg*np.eye(len(beta)))) if timing_objective else np.vstack((jac(beta),reg*np.eye(len(beta))))
        solved=least_squares(objective,np.zeros(len(terms)),jac=objective_jac,bounds=(-5.,5.),max_nfev=600)
        beta=solved.x;method='ridge_terminal_time_scaled_work' if timing_objective else 'ridge_integrated_work';success=bool(solved.success)
        # The original maximum-work gate is unchanged. Minimax fitting makes
        # that objective explicit instead of relaxing it for noisy observations.
        if max(abs(residual(beta)))>max_work_error:
            def fun(z):return z[-1]+1e-4*float(z[:-1]@z[:-1])+(float(np.mean(time_proxy(z[:-1])**2)) if timing_objective else 0.)
            def grad(z):return np.r_[2e-4*z[:-1]+(2*time_jac(z[:-1]).T@time_proxy(z[:-1])/len(cohort) if timing_objective else 0.),1.]
            def cons(z):
                e=residual(z[:-1]);return np.r_[z[-1]-e,z[-1]+e]
            def cons_jac(z):
                j=jac(z[:-1]);return np.vstack((np.c_[-j,np.ones(len(cohort))],np.c_[j,np.ones(len(cohort))]))
            opt=minimize(fun,np.r_[beta,max(abs(residual(beta)))],jac=grad,method='SLSQP',
                constraints=[dict(type='ineq',fun=cons,jac=cons_jac)],
                bounds=[(-5.,5.)]*len(beta)+[(0.,5.)],options=dict(maxiter=600,ftol=1e-9))
            if opt.success:beta=opt.x[:-1];method='regularized_minimax_with_terminal_time' if timing_objective else 'regularized_minimax_integrated_work';success=True
        errors=np.abs(residual(beta));jrank=int(np.linalg.matrix_rank(jac(beta)))
        active_bounds=np.flatnonzero(np.abs(beta)>=5.-1e-5).tolist()
        repeated=min(map(len,input_runs.values()))>=2 and min(map(len,vertex_runs.values()))>=2
        accepted=bool(success and repeated and jrank==len(terms) and not active_bounds and max(errors)<=max_work_error)
        diagnostic=dict(family=family,task=task,segment_id=segment_id,accepted=accepted,
            observations=len(cohort),parameters=len(terms),rank=jrank,removed_redundant_terms=removed,
            max_relative_work_error=float(max(errors)),p95_work_error=float(np.quantile(errors,.95)),
            optimizer_success=success,method=method,active_bound_indices=active_bounds,
            terminal_time_proxy_max=float(max(abs(time_proxy(beta)))),terminal_time_proxy_p95=float(np.quantile(abs(time_proxy(beta)),.95)),
            independent_run_ids=sorted({r['run_id'] for r in cohort}),independently_repeated_inputs_and_counts=repeated,
            worst_observations=[dict(id=cohort[i]['id'],relative_work_error=float(errors[i])) for i in np.argsort(errors)[-5:][::-1]])
        diagnostics.append(diagnostic)
        if accepted:
            # An explicit empirical support envelope, not a physical guarantee.
            envelope={field:[[float(min(x[field][i] for x in xs)),float(max(x[field][i] for x in xs))]
                for i in range(len(names))] for field in ('size','age')}
            groups.append(dict(family=family,task=task,segment_id=segment_id,terms=terms,coefficients=beta.tolist(),
                support_vertices=[list(v) for v in sorted(vertex_runs)],feature_envelope=envelope,
                own_age_envelope=[float(min(x['own_age'] for x in xs)),float(max(x['own_age'] for x in xs))],
                profile_ids=sorted(input_runs),independent_run_ids=diagnostic['independent_run_ids']))
    result=dict(schema=BOUNDED_SCHEMA if bounded else SCHEMA,status='FITTED_UNVALIDATED',admission_enabled=False,physical_bound=False,
        execution_context_sha256=fit['profiles']['execution_context_sha256'],categories=[list(x) for x in names],
        medians_ms=medians,age_edges=age_edges,ridge=ridge,timing_objective=timing_objective,groups=groups,fit_diagnostics=diagnostics,
        descriptor_scope='finite-input empirical host-call-age bins; not measured kernel phases',
        support_policy='repeated count hull and measured marginal feature envelopes; joint accuracy requires heldout replay')
    if bounded:
        result['age_descriptor_policy']=BOUNDED_POLICY
    return result


class NativeTemporalProgress:
    def __init__(self,spec,profiles,context):
        bounded=spec.get('schema')==BOUNDED_SCHEMA
        descriptor_valid=(spec.get('age_edges')==BOUNDED_AGE_EDGES and spec.get('age_descriptor_policy')==BOUNDED_POLICY) if bounded else (spec.get('schema')==SCHEMA and spec.get('age_edges') in (AGE_EDGES,LEGACY_AGE_EDGES) and 'age_descriptor_policy' not in spec)
        if (not descriptor_valid or spec.get('status')!='FITTED_UNVALIDATED'
                or spec.get('execution_context_sha256')!=context
                or spec.get('physical_bound') is not False or spec.get('admission_enabled') is not False):
            raise ValueError('Matched explicit development temporal progress law required')
        self.spec=spec;self.profiles=profiles;self.names=[tuple(x) for x in spec['categories']]
        if self.names!=categories(profiles):raise ValueError('Temporal activity categories differ')
        self.groups={}
        for g in spec['groups']:
            key=(g['family'],g['task'],g['segment_id'])
            if (key in self.groups or len(set(g['independent_run_ids']))<2
                    or len(g['terms'])!=len(g['coefficients']) or any(not math.isfinite(b) for b in g['coefficients'])):
                raise ValueError('Repeated identifiable finite temporal group required')
            self.groups[key]=g

    def progress(self,alias,member,active):
        p=self.profiles[member['profile_id']];s=next(s for s in p['segments'] if s['id']==member['segment_id'])
        if s['kind']!='gpu':return 1.
        g=self.groups.get((p['family'],p['task'],s['id']))
        if g is None:raise UnknownEvidence('UNKNOWN_NATIVE_TEMPORAL_GROUP')
        if p['id'] not in g['profile_ids']:raise UnknownEvidence('UNKNOWN_NATIVE_TEMPORAL_INPUT')
        try:x=features(dict(member,alias=alias),active,self.profiles,self.names,self.spec['medians_ms'],age_edges=self.spec['age_edges'])
        except KeyError as exc:raise UnknownEvidence('UNKNOWN_CAUSAL_NATIVE_START_OR_BASELINE') from exc
        if not within_support(x['count'].tolist(),g['support_vertices']):
            raise UnknownEvidence('NATIVE_TEMPORAL_COUNTS_OUTSIDE_SUPPORT')
        for field in ('size','age'):
            for i,(v,(lo,hi)) in enumerate(zip(x[field],g['feature_envelope'][field])):
                if v<lo-1e-8 or v>hi+1e-8:
                    raise UnknownEvidence('NATIVE_TEMPORAL_FEATURE_OUTSIDE_TRAINING_ENVELOPE:'+field+
                        ':target='+p['family']+'/'+p['task']+':feature='+'/'.join(self.names[i])+
                        f':value={v:.8g}:range={lo:.8g},{hi:.8g}')
        lo,hi=g['own_age_envelope']
        if not lo-1e-8<=x['own_age']<=hi+1e-8:raise UnknownEvidence('NATIVE_TEMPORAL_TARGET_AGE_OUTSIDE_SUPPORT')
        speed=math.exp(sum(feature_value(x,t)*b for t,b in zip(g['terms'],g['coefficients'])))
        if not math.isfinite(speed) or speed<=0:raise UnknownEvidence('INVALID_NATIVE_TEMPORAL_PROGRESS')
        return speed
