"""Causal resident-alias routing from visible outstanding work and frozen data.

This local L+Q surrogate is not complete utility or a resource/stability proof.
The host must snapshot *all* queued and submitted requests under its lock,
ordered by (enqueued_ns, request_id), and remove received completions. Workers'
actual start times are unobservable. No submitted request is moved by this code.
"""
from __future__ import annotations

import math


def _number(value,*,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def _timestamp(value):
    return type(value) is int and value>=0


class QueueAwarePolicy:
    """Two ablations share one implementation; latency sets Q weight to zero.

    aliases: {alias: {family: str, supported_tasks: [task,...]}}
    service_ms: {family: {task: {pool_id: positive estimated milliseconds}}}
    quality_loss: {family: {task: nonnegative historical aggregate loss}}
    snapshot: {alias: {outstanding: [{request_id, task, pool_id, enqueued_ns,
                actual_submit_ns: int|None}, ...], last_completed_ns: int|None}}

    Service estimates may describe input difficulty but must come from frozen
    training evidence. Quality is aggregate only; this class accepts no per-input
    quality, references or true answers. Extra event labels are never inspected.
    """

    def __init__(self,aliases,service_ms,quality_loss,weights,latency_refs_ms,*,
                 mode='quality_aware',overdue_remaining_ms=5.):
        if mode not in {'latency','quality_aware'}:raise ValueError('Unknown policy mode')
        if not _number(overdue_remaining_ms,positive=True):raise ValueError('Positive overdue floor required')
        for term in ['L','Q']:
            if not _number(weights.get(term)):raise ValueError('Finite nonnegative L/Q weights required')
        self.aliases={}
        for alias,spec in aliases.items():
            if not isinstance(alias,str) or not isinstance(spec.get('family'),str) or not isinstance(spec.get('supported_tasks'),(list,tuple,set,frozenset)):
                raise ValueError('Invalid resident alias specification')
            self.aliases[alias]=dict(family=spec['family'],supported_tasks=frozenset(spec['supported_tasks']))
        if not self.aliases:raise ValueError('At least one resident alias required')
        # Copy only allowed mappings. No references, observations or answers are
        # accepted as inputs to scoring or stored as part of audit evidence.
        self.service_ms={family:{task:dict(values) for task,values in tasks.items()} for family,tasks in service_ms.items()}
        self.quality_loss={family:dict(tasks) for family,tasks in quality_loss.items()}
        self.weights=dict(L=weights['L'],Q=weights['Q'] if mode=='quality_aware' else 0.)
        self.latency_refs_ms=dict(latency_refs_ms);self.mode=mode;self.overdue_remaining_ms=float(overdue_remaining_ms)

    def _service(self,family,task,pool_id):
        value=self.service_ms.get(family,{}).get(task,{}).get(pool_id)
        return float(value) if _number(value,positive=True) else None

    def _waiting(self,alias,now_ns,snapshot):
        if alias not in snapshot:return None,dict(status='UNKNOWN_MISSING_ALIAS_SNAPSHOT',outstanding=[])
        state=snapshot[alias]
        if not isinstance(state,dict) or not isinstance(state.get('outstanding'),list):
            return None,dict(status='UNKNOWN_INVALID_ALIAS_SNAPSHOT',outstanding=[])
        last=state.get('last_completed_ns')
        if last is not None and (not _timestamp(last) or last>now_ns):
            return None,dict(status='UNKNOWN_INVALID_PAST_COMPLETION_TIMESTAMP',outstanding=[])
        pending=[];seen=set();family=self.aliases[alias]['family'];tasks=self.aliases[alias]['supported_tasks']
        for raw in state['outstanding']:
            # Deliberate whitelist: future/completed/worker_latency fields have
            # no effect on ordering, waiting estimates, candidates or audit.
            if not isinstance(raw,dict):return None,dict(status='UNKNOWN_INVALID_OUTSTANDING_RECORD',outstanding=pending)
            item={k:raw.get(k) for k in ['request_id','task','pool_id','enqueued_ns','actual_submit_ns']}
            rid=item['request_id'];enqueued=item['enqueued_ns'];submit=item['actual_submit_ns']
            if not isinstance(rid,str) or rid in seen or not _timestamp(enqueued):
                return None,dict(status='UNKNOWN_INVALID_OUTSTANDING_ID_OR_ENQUEUE',outstanding=pending)
            # Planned arrivals cannot create load before enqueue. Ignore their
            # submit/service labels as well, even if the future task is unknown.
            if enqueued>now_ns:continue
            if submit is not None and (not _timestamp(submit) or submit<enqueued or submit>now_ns):
                return None,dict(status='UNKNOWN_INVALID_ACTUAL_SUBMIT_TIMESTAMP',outstanding=pending)
            if item['task'] not in tasks:
                return None,dict(status='UNKNOWN_UNSUPPORTED_OUTSTANDING_TASK',outstanding=pending)
            seen.add(rid);item['predicted_service_ms']=self._service(family,item['task'],item['pool_id'])
            if item['predicted_service_ms'] is None:
                return None,dict(status='UNKNOWN_OUTSTANDING_SERVICE',outstanding=[*pending,item])
            pending.append(item)
        expected=sorted(pending,key=lambda r:(r['enqueued_ns'],r['request_id']))
        if pending!=expected:return None,dict(status='UNKNOWN_OUTSTANDING_NOT_FIFO_ORDERED',outstanding=pending)
        waiting=0.
        for i,item in enumerate(pending):
            service=item['predicted_service_ms']
            if i==0 and item['actual_submit_ns'] is not None:
                start=max(item['actual_submit_ns'],last if last is not None else item['actual_submit_ns'])
                elapsed=(now_ns-start)/1e6;remaining=max(self.overdue_remaining_ms,service-elapsed)
                item.update(predicted_start_ns=start,elapsed_since_inferred_start_ms=elapsed,
                    predicted_remaining_ms=remaining,overdue_floor_applied=service-elapsed<self.overdue_remaining_ms,
                    estimate_basis='Submitted FIFO head; start=max(host submit, last host-received completion). Actual worker start unobserved.')
            else:
                remaining=service
                item.update(predicted_remaining_ms=remaining,overdue_floor_applied=False,
                    estimate_basis='Full predicted service for queued head or FIFO suffix; no elapsed-time subtraction.')
            waiting+=remaining
        return waiting,dict(status='ESTIMATED_FROM_ALL_VISIBLE_OUTSTANDING',outstanding=pending,
            last_completed_ns=last,queued_count=sum(r['actual_submit_ns'] is None for r in pending),
            submitted_count=sum(r['actual_submit_ns'] is not None for r in pending),
            overdue_remaining_ms=self.overdue_remaining_ms,worker_start_observable=False)

    def choose(self,task,pool_id,now_ns,pending_snapshot):
        if not _timestamp(now_ns):raise ValueError('now_ns must be nonnegative integer host time')
        if not isinstance(pending_snapshot,dict):raise ValueError('Alias snapshot mapping required')
        candidates=[]
        for alias,spec in sorted(self.aliases.items()):
            candidate=dict(alias=alias,family=spec['family'],supported=task in spec['supported_tasks'],
                score=None,predicted_waiting_ms=None,predicted_own_service_ms=None,predicted_completion_ms=None,
                aggregate_quality_loss=None,latency_cost=None,quality_cost=None,unknown_reasons=[])
            if not candidate['supported']:
                candidate.update(status='UNSUPPORTED_TASK');candidates.append(candidate);continue
            own=self._service(spec['family'],task,pool_id);reference=self.latency_refs_ms.get(task)
            quality=self.quality_loss.get(spec['family'],{}).get(task)
            quality=float(quality) if _number(quality) else None
            waiting,evidence=self._waiting(alias,now_ns,pending_snapshot)
            candidate.update(predicted_own_service_ms=own,predicted_waiting_ms=waiting,aggregate_quality_loss=quality,waiting_evidence=evidence)
            reasons=candidate['unknown_reasons']
            if own is None:reasons.append('UNKNOWN_OWN_INPUT_SERVICE')
            if waiting is None:reasons.append(evidence['status'])
            if not _number(reference,positive=True):reasons.append('UNKNOWN_LATENCY_REFERENCE')
            if self.weights['Q']>0 and quality is None:reasons.append('UNKNOWN_AGGREGATE_QUALITY')
            if not reasons:
                completion=waiting+own;latency_cost=self.weights['L']*completion/reference
                quality_cost=self.weights['Q']*quality if self.weights['Q']>0 else 0.
                candidate.update(status='KNOWN_LOCAL_SURROGATE',predicted_completion_ms=completion,
                    predicted_finish_ns=now_ns+round(completion*1e6),latency_cost=latency_cost,quality_cost=quality_cost,
                    score=latency_cost+quality_cost)
            else:candidate.update(status='UNKNOWN_CANDIDATE')
            candidates.append(candidate)
        eligible=[c for c in candidates if c['score'] is not None]
        chosen=min(eligible,key=lambda c:(c['score'],c['predicted_completion_ms'],c['alias'])) if eligible else None
        return dict(alias=chosen['alias'] if chosen else None,status='CHOSEN' if chosen else 'UNKNOWN_NO_COMPARABLE_RESIDENT_ALIAS',
            task=task,pool_id=pool_id,now_ns=now_ns,mode=self.mode,weights=dict(self.weights),audit=candidates,
            unknown_supported_candidates=sum(c['status']=='UNKNOWN_CANDIDATE' for c in candidates),
            choice_scope='Minimum known local L+Q surrogate among resident supported aliases; UNKNOWN candidates cannot be ruled out.',
            objective='wL*(predicted_waiting_ms+predicted_own_service_ms)/task_latency_reference_ms + effective_wQ*aggregate_family_task_quality_loss',
            full_U=None,absolute_quality_gate=False,submitted_work_migrated=False,
            trigger='NO_KNOWN_CANDIDATE' if chosen is None else 'MINIMUM_LOCAL_SURROGATE')
