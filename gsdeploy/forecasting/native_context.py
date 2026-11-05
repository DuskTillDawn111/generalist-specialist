"""Empirical native-call interference, pooled across pinned solo input templates.

Native inference spans mix host work, transfers and CUDA. Categorical activity
is an explicit alternative to unavailable SM/DRAM/L2 counters; it does not
estimate those counters. Accuracy and physical bounds are not granted by fit.
"""
from __future__ import annotations
import copy
import json
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from ..forecast_errors import UnknownEvidence
from ..energy import integrate_power
from ..linear_solver import linprog
from .profiles import ProfileProvider,context_key
from .native_order import NativeOrderLaw,identity as order_identity

SCHEMA='gsdeploy.native_activity_context_law.v1'


def categories(profiles):
    return sorted({(p['family'],p['task']) for p in profiles.values()
                   if any(s['kind']=='gpu' for s in p['segments'])})


def counts(active,profiles,names,omit=None):
    result=[0.]*len(names)
    for alias,member in active.items():
        if alias==omit:continue
        profile=profiles[member['profile_id']]
        segment=next(s for s in profile['segments'] if s['id']==member['segment_id'])
        if segment['kind']=='gpu':
            result[names.index((profile['family'],profile['task']))]+=1.
    return result


def design(values,terms):
    return np.array([math.prod(values[i] for i in term) for term in terms],dtype=float)


def within_support(values,vertices):
    if list(values) in vertices:return True
    if not vertices:return False
    matrix=np.array([[1.,*v] for v in vertices],dtype=float).T
    rhs=np.array([1.,*values],dtype=float)
    fit=linprog(np.zeros(len(vertices)),A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs',
        options={'time_limit':.1,'primal_feasibility_tolerance':1e-9})
    return bool(fit.success and np.max(np.abs(matrix@fit.x-rhs))<=1e-8)


class NativeContextLaw:
    def __init__(self,spec,profiles,execution_context_sha256):
        if (spec.get('schema')!=SCHEMA or spec.get('status')!='FITTED_UNVALIDATED'
                or spec.get('execution_context_sha256')!=execution_context_sha256
                or spec.get('non_gpu_progress_assumption')!='pinned_solo_duration'
                or spec.get('native_scope')!='mixed_host_transfers_CUDA'
                or spec.get('physical_bound') is not False):
            raise ValueError('Explicit matched empirical native activity law required')
        self.spec=copy.deepcopy(spec);self.profiles=profiles
        self.names=[tuple(row) for row in spec['categories']]
        if self.names!=categories(profiles):raise ValueError('Native activity category mismatch')
        self.groups={}
        temporal=spec.get('temporal_progress')
        self.temporal_progress=None
        if temporal is not None:
            if spec['progress_groups']:raise ValueError('Choose one native progress descriptor')
            from .native_temporal import NativeTemporalProgress
            self.temporal_progress=NativeTemporalProgress(temporal,profiles,execution_context_sha256)
        for group in spec['progress_groups']:
            key=tuple(group[k] for k in ('family','task','segment_id'))
            if key in self.groups or len(group['independent_run_ids'])<2:
                raise ValueError('Distinct, independently repeated native progress groups required')
            terms=group['terms'];beta=group['coefficients']
            if (len(terms)!=len(beta) or not terms or terms[0]!=[]
                    or any(not math.isfinite(b) for b in beta)
                    or any(any(type(i) is not int or not 0<=i<len(self.names) for i in t) for t in terms)):
                raise ValueError('Invalid native activity progress coefficients')
            self.groups[key]=group
        self.support_cache={}
        self._last_temporal_evaluation=None

    def progress(self,alias,member,active):
        if self.temporal_progress is not None:return self.temporal_progress.progress(alias,member,active)
        profile=self.profiles[member['profile_id']]
        segment=next(s for s in profile['segments'] if s['id']==member['segment_id'])
        if segment['kind']!='gpu':return 1.
        group=self.groups.get((profile['family'],profile['task'],segment['id']))
        if group is None:raise UnknownEvidence('UNKNOWN_NATIVE_PROGRESS_GROUP')
        x=counts(active,self.profiles,self.names,omit=alias)
        cache_key=(profile['family'],profile['task'],segment['id'],tuple(x))
        supported=self.support_cache.get(cache_key)
        if supported is None:
            supported=within_support(x,group['support_vertices'])
            self.support_cache[cache_key]=supported
        if not supported:raise UnknownEvidence('UNKNOWN_NATIVE_ACTIVITY_OUTSIDE_TRAINING_HULL')
        speed=math.exp(float(design(x,group['terms'])@np.array(group['coefficients'])))
        if not math.isfinite(speed) or speed<=0:raise UnknownEvidence('INVALID_NATIVE_ACTIVITY_PROGRESS')
        return speed

    def evaluate(self,active,scenario,key):
        if scenario!='mean':raise UnknownEvidence('UNKNOWN_NATIVE_ACTIVITY_SCENARIO:'+scenario)
        signature=None
        if self.temporal_progress is not None:
            from .native_temporal import age_bin
            signature=tuple((alias,member['profile_id'],member['segment_id'],
                member.get('native_baseline_ms'),age_bin(member['native_age_ms']/member['native_baseline_ms'],self.temporal_progress.spec['age_edges'])
                if 'native_age_ms' in member else None) for alias,member in sorted(active.items()))
            if self._last_temporal_evaluation is not None and self._last_temporal_evaluation[0]==signature:
                return self._last_temporal_evaluation[1]
        progress={alias:self.progress(alias,member,active) for alias,member in active.items()}
        resources=self.spec.get('resources')
        watts=memory=None
        if resources is not None:
            x=counts(active,self.profiles,self.names)
            if within_support(x,resources['support_vertices']):
                watts=float(design(x,resources['terms'])@np.array(resources['coefficients']))
                if not math.isfinite(watts) or watts<0:watts=None
                memory=resources['resident_observed_peak_mib']
        evidence=json.dumps(dict(schema=SCHEMA,physical_bound=False,
            independent_accuracy_qualified=False,context_key=key,
            native_scope='mixed_host_transfers_CUDA',non_gpu_progress='pinned_solo_duration'),sort_keys=True)
        result=dict(progress=progress,board_watts=watts,peak_memory_mib=memory),evidence
        if signature is not None:self._last_temporal_evaluation=(signature,result)
        return result


class NativeContextProvider(ProfileProvider):
    """Development adapter; production integration follows independent tests."""
    def __init__(self,document,execution_context_sha256,**kwargs):
        # SearchSession JSON nodes intentionally remain immutable under deepcopy.
        # Build a new outer mapping instead of mutating the caller's frozen node.
        base={k:copy.deepcopy(v) for k,v in document.items() if k not in {'native_activity_law','native_order_law','native_idle_age_energy'}}
        law=document['native_activity_law']
        if 'context_interpolation' in base:
            raise ValueError('Choose one explicit empirical context law')
        super().__init__(base,execution_context_sha256,**kwargs)
        self.native_law=NativeContextLaw(law,self.profiles,execution_context_sha256)
        energy=document.get('native_idle_age_energy')
        self.requires_energy_history=energy is not None
        self.idle_age_energy=None
        if energy is not None:
            from .native_energy import NativeIdleAgeEnergy
            if (energy.get('execution_context_sha256')!=execution_context_sha256
                    or energy.get('categories')!=law['categories']):
                raise ValueError('Causal energy law differs from the native execution context')
            self.idle_age_energy=NativeIdleAgeEnergy(energy)
        self.requires_native_call_age=self.native_law.temporal_progress is not None
        order=document.get('native_order_law');self.native_order_law=NativeOrderLaw(order,execution_context_sha256) if order else None
        if law.get('native_order_law_sha256')!=(order_identity(order) if order else None):
            raise ValueError('Native activity fit differs from its bound request-order baseline')

    def energy_interval(self,active,left_ns,right_ns,last_native_end_ns):
        if self.idle_age_energy is None:
            raise ValueError('No causal energy law was selected')
        x=counts(active,self.profiles,self.native_law.names)
        return self.idle_age_energy.integrate([(left_ns,right_ns,x)],last_native_end_ns)

    def plan(self,row,aliases,scenario):
        if self.requires_native_call_age and (row.get('forecast_cursor') is not None or row.get('head_age_ms',0)>0):
            raise UnknownEvidence('UNKNOWN_CAUSAL_NATIVE_START_FOR_CONTINUATION')
        result=super().plan(row,aliases,scenario)
        if self.native_order_law is not None:
            profile=self.profiles[result['profile_id']]
            for segment in result['segments']:
                segment['duration_ms']*=self.native_order_law.factor(profile,segment,row.get('native_order'))
        return result

    def native_clock_event(self,active,at_ns):
        """Publish only causal elapsed state and the next numerical age event."""
        if not self.requires_native_call_age:return None
        from .native_temporal import age_bin
        edges=self.native_law.temporal_progress.spec['age_edges']
        next_event=None
        for row in active.values():
            if row['segment_kind']!='gpu':continue
            start=row.get('native_segment_start_ns')
            if type(start) is not int or start>at_ns:raise UnknownEvidence('UNKNOWN_CAUSAL_NATIVE_START')
            baseline=row['segment_duration_ms'];age_ms=(at_ns-start)/1e6
            row.update(native_age_ms=age_ms,native_baseline_ms=baseline)
            index=age_bin(age_ms/baseline,edges)
            event=max(at_ns+1,start+round(edges[index+1]*baseline*1e6))
            next_event=event if next_event is None else min(next_event,event)
        return next_event

    def _context(self,active,aliases,scenario):
        # Reuse the exact provider's request identity validation. Only an absent
        # context can invoke the declared law; existing incomplete rows stay unknown.
        try:return super()._context(active,aliases,scenario)
        except UnknownEvidence:
            key=context_key(active)
            if key in self.document['contexts']:raise
            return self.native_law.evaluate(active,scenario,key)


def fit_native_progress(template,observations,*,minimum_runs=2,max_work_error=.2):
    """Fit positive activity-dependent progress from complete changing overlaps.

    Observations add run_id to the complete stage exporter rows. Each training
    input and activity vertex must recur in independent runs. Unidentifiable
    groups are reported and omitted, never replaced by guessed coefficients.
    """
    if type(minimum_runs) is not int or minimum_runs<2:raise ValueError('At least two runs required')
    profiles={p['id']:p for p in template['profiles']};names=categories(profiles)
    order=template.get('native_order_law');order_law=NativeOrderLaw(order,template['execution_context_sha256']) if order else None
    groups=defaultdict(list);seen=set()
    for row in observations:
        if (row.get('split')!='training' or row.get('execution_context_sha256')!=template['execution_context_sha256']
                or not row.get('run_id') or not row.get('id') or row['id'] in seen):
            raise ValueError('Distinct execution-bound training observations required')
        seen.add(row['id']);profile=profiles[row['profile_id']]
        segment=next(s for s in profile['segments'] if s['id']==row['segment_id'])
        if segment['kind']!='gpu':continue
        baseline=segment['duration_ms']['mean']
        if order_law is not None:baseline*=order_law.factor(profile,segment,row.get('native_order'))
        if not math.isfinite(baseline) or baseline<=0:raise ValueError('Positive pinned solo duration required')
        intervals=[]
        for interval in row['intervals']:
            duration=interval['duration_ms']
            if not math.isfinite(duration) or duration<=0:raise ValueError('Positive complete interval required')
            member=interval['members'].get(row['target_alias'])
            if member!={'profile_id':profile['id'],'segment_id':segment['id']}:
                raise ValueError('Observed native target is absent from interval')
            intervals.append((duration/baseline,counts(interval['members'],profiles,names,omit=row['target_alias'])))
        if not intervals:raise ValueError('Complete nonempty overlap intervals required')
        groups[(profile['family'],profile['task'],segment['id'])].append((row,intervals))
    fitted=[];diagnostics=[]
    for (family,task,segment_id),rows in sorted(groups.items()):
        vertices=sorted({tuple(x) for _,intervals in rows for _,x in intervals})
        used=[i for i in range(len(names)) if any(v[i] for v in vertices)]
        terms=[[]]+[[i] for i in used]+[[i,j] for i in used for j in used if i<j and any(v[i]*v[j] for v in vertices)]
        run_ids={r['run_id'] for r,_ in rows};vertex_runs=defaultdict(set);input_runs=defaultdict(set)
        for row,intervals in rows:
            input_runs[row['profile_id']].add(row['run_id'])
            for _,x in intervals:vertex_runs[tuple(x)].add(row['run_id'])
        diagnostic=dict(family=family,task=task,segment_id=segment_id,observations=len(rows),
            independent_run_ids=sorted(run_ids),parameters=len(terms),accepted=False)
        if (len(rows)<len(terms) or min(map(len,input_runs.values()))<minimum_runs
                or min(map(len,vertex_runs.values()))<minimum_runs):
            diagnostic['reason']='INSUFFICIENT_INDEPENDENT_INPUT_OR_ACTIVITY_REPETITIONS'
            diagnostics.append(diagnostic);continue
        xs=[np.array([design(x,terms) for _,x in intervals]) for _,intervals in rows]
        ws=[np.array([duration for duration,_ in intervals]) for _,intervals in rows]
        def residual(beta):return np.array([float(w@np.exp(x@beta))-1 for x,w in zip(xs,ws)])
        def jacobian(beta):return np.array([((w*np.exp(x@beta))[:,None]*x).sum(axis=0) for x,w in zip(xs,ws)])
        fit=least_squares(residual,np.zeros(len(terms)),jac=jacobian,bounds=(-5.,5.),max_nfev=1000)
        rank=int(np.linalg.matrix_rank(fit.jac));error=float(max(abs(residual(fit.x))))
        diagnostic.update(rank=rank,max_relative_work_error=error,optimizer_success=bool(fit.success))
        work_errors=residual(fit.x)
        diagnostic['worst_observations']=[dict(id=row['id'],run_id=row['run_id'],profile_id=row['profile_id'],
            relative_work_error=float(value)) for (row,_),value in sorted(zip(rows,work_errors),
            key=lambda pair:abs(pair[1]),reverse=True)[:10]]
        by_run=defaultdict(list)
        for (row,_),value in zip(rows,work_errors):by_run[row['run_id']].append(abs(float(value)))
        diagnostic['error_by_independent_run']={run:dict(observations=len(values),
            mean_relative_work_error=float(np.mean(values)),max_relative_work_error=max(values))
            for run,values in sorted(by_run.items())}
        if not fit.success or rank!=len(terms) or error>max_work_error or any(fit.active_mask):
            diagnostic['reason']='UNIDENTIFIABLE_OR_EXCESSIVE_TRAINING_WORK_ERROR'
            diagnostics.append(diagnostic);continue
        diagnostic['accepted']=True;diagnostics.append(diagnostic)
        fitted.append(dict(family=family,task=task,segment_id=segment_id,terms=terms,
            coefficients=fit.x.tolist(),support_vertices=[list(v) for v in vertices],
            independent_run_ids=sorted(run_ids),observation_ids=[r['id'] for r,_ in rows]))
    return dict(schema=SCHEMA,status='FITTED_UNVALIDATED',
        execution_context_sha256=template['execution_context_sha256'],categories=[list(x) for x in names],
        native_scope='mixed_host_transfers_CUDA',non_gpu_progress_assumption='pinned_solo_duration',
        physical_bound=False,progress_groups=fitted,resources=None,fit_diagnostics=diagnostics,
        **({'native_order_law_sha256':order_identity(order)} if order else {}))


def fit_native_resources(template,runs,*,block_seconds=5.,minimum_runs=2,max_gap_seconds=.25):
    """Fit whole-board energy over blocks, using integrated activity exposure.

    A block can cross many stage transitions. It is not an independent repeat,
    and instantaneous NVML lag is not turned into per-kernel power. Coefficients
    may have either sign; no positive worker-cost assumption is imposed.
    """
    if (minimum_runs<2 or block_seconds<=0 or max_gap_seconds<=0
            or len({r['run_id'] for r in runs})!=len(runs)):
        raise ValueError('Distinct repeated resource runs and positive covered blocks required')
    profiles={p['id']:p for p in template['profiles']};names=categories(profiles)
    vertices=set();blocks=[];run_ids=set();peak=0.;all_windows=[]
    for run in runs:
        if (run.get('split')!='training' or run.get('execution_context_sha256')!=template['execution_context_sha256']
                or not run.get('run_id') or run.get('telemetry_errors')
                or any(s.get('foreign_pids') for s in run['board_samples'])):
            raise ValueError('Matched training-only whole-board telemetry required')
        start,end=run['observation_window_ns']
        if type(start) is not int or type(end) is not int or end<=start:
            raise ValueError('Covered common-clock resource window required')
        spans=run['spans'];events=defaultdict(list)
        for span in spans:
            if not start<=span['start_ns']<span['end_ns']<=end:
                raise ValueError('Stage outside declared resource observation window')
            events[span['start_ns']].append(('start',span))
            events[span['end_ns']].append(('end',span))
        active={};windows=[];previous=start
        # END before START at a shared boundary preserves exclusive owner stages.
        for timestamp in sorted({start,end,*events}):
            if timestamp>previous:
                x=counts(active,profiles,names);vertices.add(tuple(x))
                windows.append((previous,timestamp,x))
            for kind,span in sorted(events.get(timestamp,[]),key=lambda item:item[0]):
                alias=span['alias'];member=dict(profile_id=span['profile_id'],segment_id=span['segment_id'])
                if kind=='end':
                    if active.get(alias)!=member:raise ValueError('Unmatched resource stage end')
                    active.pop(alias)
                else:
                    if alias in active:raise ValueError('Overlapping resource stages for one owner')
                    active[alias]=member
            previous=timestamp
        if active:raise ValueError('Incomplete resource roster')
        run_ids.add(run['run_id']);all_windows.append((run,windows))
        peak=max(peak,max((s['owned_used_mib'] for s in run['board_samples'] if start<=s['time_ns']<=end),default=0))
    used=[i for i in range(len(names)) if any(v[i] for v in vertices)]
    terms=[[]]+[[i] for i in used]+[[i,j] for i in used for j in used if i<j and any(v[i]*v[j] for v in vertices)]
    for run,windows in all_windows:
        start,end=run['observation_window_ns'];left=start
        while left<end:
            right=min(end,left+round(block_seconds*1e9))
            if (right-left)/1e9<min(.5,block_seconds):break
            joules=integrate_power(run['board_samples'],left,right,max_gap_seconds=max_gap_seconds)
            if joules is None:raise ValueError('Uncovered board-power block')
            exposure=np.zeros(len(terms))
            for a,b,x in windows:
                duration=max(0,min(b,right)-max(a,left))/1e9
                if duration:exposure+=duration*design(x,terms)
            blocks.append((run['run_id'],left,right,exposure,joules))
            left=right
    diagnostic=dict(independent_run_ids=sorted(run_ids),blocks=len(blocks),parameters=len(terms),accepted=False,
        scope='training integrated board energy; block count is not independent sample size')
    if len(run_ids)<minimum_runs or len(blocks)<len(terms):
        diagnostic['reason']='INSUFFICIENT_INDEPENDENT_RESOURCE_RUNS';return None,diagnostic
    matrix=np.array([b[3] for b in blocks]);target=np.array([b[4] for b in blocks])
    rank=int(np.linalg.matrix_rank(matrix));diagnostic['rank']=rank
    if rank!=len(terms):
        diagnostic['reason']='UNIDENTIFIABLE_RESOURCE_ENERGY';return None,diagnostic
    beta=np.linalg.lstsq(matrix,target,rcond=None)[0]
    if any(float(design(v,terms)@beta)<0 for v in vertices):
        diagnostic['reason']='NEGATIVE_POWER_INSIDE_OBSERVED_SUPPORT';return None,diagnostic
    errors=abs(matrix@beta-target)/np.maximum(target,1e-12)
    diagnostic.update(accepted=True,max_relative_block_energy_error=float(max(errors)),
        mean_relative_block_energy_error=float(np.mean(errors)))
    resource=dict(terms=terms,coefficients=beta.tolist(),support_vertices=[list(v) for v in sorted(vertices)],
        independent_run_ids=sorted(run_ids),resident_observed_peak_mib=peak,
        memory_scope='maximum observed owned peak across matched training runs; not a physical bound',
        energy_scope='whole_GPU_board_integrated_activity_regression')
    return resource,diagnostic
