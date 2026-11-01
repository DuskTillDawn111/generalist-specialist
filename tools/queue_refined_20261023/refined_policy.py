"""Causal calibrated routing and a short-queue global incremental proxy.

Neither score is full deployment U. Outstanding host/pipe work is a concurrency
proxy, not observed active GPU kernels. FIFO-head ages are inferred from host
submission/last received completion, not measured worker service ages. Missing
T, resource/startup effects and uncalibrated future interference stay UNKNOWN.
"""
from __future__ import annotations
import copy
import math
import json

from tools.queue_refined_20261023.calibration import visible_context


def _finite(v,positive=False):
    return type(v) in (int,float) and math.isfinite(v) and (v>0 if positive else v>=0)

def _ns(v):return type(v) is int and v>=0

def _p95(values):
    values=sorted(values);p=(len(values)-1)*.95;lo=math.floor(p);hi=math.ceil(p)
    return values[lo]+(values[hi]-values[lo])*(p-lo)

class RefinedPolicy:
    """ServiceModel supports predict(...), residual(...); no future outcomes.

    modes: conservative minimizes calibrated own completion+aggregate Q;
    global_incremental minimizes the scenario-minus-baseline proxy L/P/Q/E of
    all currently outstanding requests plus the newly arriving request.
    Energy rates are optional frozen historical joules per service millisecond.
    A missing rate is UNKNOWN, never silently claimed as a calibrated zero.
    """
    def __init__(self,aliases,service_model,quality_loss,weights,latency_refs_ms,workload,*,
                 mode='conservative',margin_ms=250.,risk_quantile=.9,
                 concurrency_context=True,allow_spill=True,energy_proxy_joules_per_service_ms=None,
                 energy_reference_watts=450.,horizon_seconds=600.):
        if mode not in {'conservative','global_incremental'}:raise ValueError('Unknown refined mode')
        if not _finite(margin_ms):raise ValueError('Nonnegative finite margin required')
        if not _finite(risk_quantile) or not 0<risk_quantile<1:raise ValueError('Risk quantile must be in (0,1)')
        if type(concurrency_context) is not bool:raise ValueError('Explicit boolean concurrency context required')
        if type(allow_spill) is not bool:raise ValueError('Explicit boolean spill eligibility required')
        if not _finite(energy_reference_watts,True) or not _finite(horizon_seconds,True):raise ValueError('Positive energy normalization required')
        for t in ('Q','L','P','E'):
            if not _finite(weights.get(t,0.)):raise ValueError('Nonnegative finite proxy weights required')
        if not aliases or any(not isinstance(a,str) or not isinstance(v.get('family'),str) or not isinstance(v.get('supported_tasks'),(list,tuple,set,frozenset)) for a,v in aliases.items()):raise ValueError('Resident aliases required')
        if not workload or any(not _finite(v,True) for v in workload.values()):raise ValueError('Positive declared workload required')
        self.aliases={a:dict(family=v['family'],supported_tasks=frozenset(v['supported_tasks'])) for a,v in aliases.items()}
        if hasattr(service_model,'validate_aliases'):service_model.validate_aliases(self.aliases)
        self.allow_spill=allow_spill
        self.model=service_model;self.quality_loss=copy.deepcopy(quality_loss);self.weights={t:weights.get(t,0.) for t in ('Q','L','P','E')}
        self.refs=dict(latency_refs_ms);self.workload=dict(workload);self.total=sum(workload.values());self.mode=mode
        self.margin_ms=float(margin_ms);self.risk_quantile=float(risk_quantile);self.concurrency_context=concurrency_context
        self._prediction_cache={}
        self.expected_counts={t:rate*float(horizon_seconds) for t,rate in self.workload.items()}
        self.tail_counts={t:max(1,math.ceil(.05*n)) for t,n in self.expected_counts.items()}
        self.energy_rates=copy.deepcopy(energy_proxy_joules_per_service_ms);self.energy_reference_watts=float(energy_reference_watts);self.horizon_seconds=float(horizon_seconds)

    def warm_cache(self,pool_ids=None):
        """Precompute frozen distributions outside measurement, no trace reads.

        With no pool_ids, enumerate only historical family/task/input keys.
        Explicit pool_ids is {task:[pool_id,...]} from a frozen input catalog.
        Residual ages are not enumerated; they share the precomputed service
        distribution and are queried causally at actual dispatch time.
        """
        triples=set()
        if pool_ids is None:
            document=getattr(self.model,'document',{})
            for encoded in document.get('strata',{}).get('family_task_input',{}):
                family,task,pool=json.loads(encoded)
                if any(v['family']==family and task in v['supported_tasks'] for v in self.aliases.values()):triples.add((family,task,pool))
        else:
            for spec in self.aliases.values():
                for task in spec['supported_tasks']:
                    for pool in pool_ids.get(task,[]):triples.add((spec['family'],task,pool))
        contexts=[dict(other_active_bin=b,g2_visible_outstanding=g) for b in ('0','1','2plus') for g in (False,True)] if self.concurrency_context else [None]
        known=unknown=0
        for family,task,pool in sorted(triples):
            for context in contexts:
                result=self.model.predict(family,task,pool,context=context,quantile=self.risk_quantile)
                if _finite(result.get('conservative_ms'),True):known+=1
                else:unknown+=1
        return dict(frozen_path_inputs=len(triples),contexts=len(contexts),known_predictions=known,unknown_predictions=unknown,
                    future_arrivals_read=False,residual_age_precomputed=False)

    def _snapshot(self,snapshot,now):
        out={};seen=set()
        for alias,spec in self.aliases.items():
            state=snapshot.get(alias)
            if not isinstance(state,dict) or not isinstance(state.get('outstanding'),list):raise ValueError('UNKNOWN_MISSING_ALIAS_SNAPSHOT:'+alias)
            last=state.get('last_completed_ns')
            if last is not None and (not _ns(last) or last>now):raise ValueError('INVALID_LAST_COMPLETED_NS')
            rows=[]
            for raw in state['outstanding']:
                if not isinstance(raw,dict):raise ValueError('INVALID_OUTSTANDING_ROW')
                r={k:raw.get(k) for k in ('request_id','task','pool_id','enqueued_ns','actual_submit_ns')}
                if not _ns(r['enqueued_ns']):raise ValueError('INVALID_ENQUEUED_NS')
                if r['enqueued_ns']>now:continue
                if not isinstance(r['request_id'],str) or r['request_id'] in seen:raise ValueError('DUPLICATE_OR_INVALID_REQUEST_ID')
                if r['task'] not in spec['supported_tasks'] or r['task'] not in self.workload:raise ValueError('UNSUPPORTED_OUTSTANDING_TASK')
                submit=r['actual_submit_ns']
                if submit is not None and (not _ns(submit) or submit<r['enqueued_ns'] or submit>now):raise ValueError('INVALID_SUBMIT_NS')
                seen.add(r['request_id']);rows.append(r)
            if rows!=sorted(rows,key=lambda r:(r['enqueued_ns'],r['request_id'])):raise ValueError('NON_FIFO_SNAPSHOT')
            out[alias]=dict(outstanding=rows,last_completed_ns=last)
        return out

    def _context(self,snapshot,alias):
        if self.concurrency_context:
            # Training contexts are dispatch snapshots BEFORE this request is
            # enqueued. The hypothetical new G2 work affects other aliases,
            # while its own target context excludes that synthetic own row.
            own=snapshot[alias];rows=own['outstanding']
            if any(r.get('_refined_current') is True for r in rows):
                context_snapshot=dict(snapshot);context_snapshot[alias]=dict(own,outstanding=[r for r in rows if r.get('_refined_current') is not True])
                return visible_context(context_snapshot,alias,self.aliases)
            return visible_context(snapshot,alias,self.aliases)
        return None

    def _prediction(self,family,task,pool_id,context,age=None):
        key=(family,task,pool_id,tuple(sorted((context or {}).items())),age,self.risk_quantile)
        if key in self._prediction_cache:return self._prediction_cache[key]
        if age is None:raw=self.model.predict(family,task,pool_id,context=context,quantile=self.risk_quantile)
        else:raw=self.model.residual(family,task,pool_id,age_ms=age,context=context,quantile=self.risk_quantile)
        value=raw.get('conservative_ms')
        if not _finite(value,True):return None,dict(status='UNKNOWN_CALIBRATED_SERVICE',model=raw)
        result=(float(value),dict(status='CALIBRATED_PROXY',model=raw,context=context,
            remaining_age_basis='max(host submit,last host-received completion); actual worker start unobserved' if age is not None else None))
        self._prediction_cache[key]=result;return result

    def _forecast(self,snapshot,now,new_alias=None,new_task=None,new_pool=None):
        scenario=copy.deepcopy(snapshot)
        if new_alias is not None:
            # Unique synthetic row is internal only, never sent to the model as
            # a reference or an answer; append preserves actual dispatch FIFO.
            scenario[new_alias]['outstanding'].append(dict(request_id='__refined_current_arrival__',task=new_task,pool_id=new_pool,enqueued_ns=now,actual_submit_ns=None,_refined_current=True))
        latencies={t:[] for t in self.workload};qualities={t:[] for t in self.workload};energy=0.;energy_known=self.energy_rates is not None
        alias_audit={};current=None;unknown=[]
        for alias,spec in sorted(self.aliases.items()):
            context=self._context(scenario,alias);clock=0.;rows=[];family=spec['family'];last=scenario[alias]['last_completed_ns']
            for i,r in enumerate(scenario[alias]['outstanding']):
                synthetic=r.get('_refined_current') is True
                age=None
                if i==0 and r['actual_submit_ns'] is not None:
                    start=max(r['actual_submit_ns'],last if last is not None else r['actual_submit_ns']);age=(now-start)/1e6
                service,evidence=self._prediction(family,r['task'],r['pool_id'],context,age)
                if service is None:unknown.append('UNKNOWN_SERVICE:'+alias+':'+r['task']);continue
                clock+=service;elapsed=(now-r['enqueued_ns'])/1e6;lat=elapsed+clock
                latencies[r['task']].append(lat);q=self.quality_loss.get(family,{}).get(r['task'])
                if _finite(q):qualities[r['task']].append(float(q))
                else:unknown.append('UNKNOWN_AGGREGATE_QUALITY:'+family+':'+r['task'])
                rate=(self.energy_rates or {}).get(family,{}).get(r['task'])
                if _finite(rate):energy+=service*rate
                else:energy_known=False
                record=dict(request_id=r['request_id'],task=r['task'],pool_id=r['pool_id'],predicted_remaining_service_ms=service,
                    inferred_service_age_ms=age,predicted_remaining_completion_ms=clock,predicted_host_e2e_ms=lat,service_evidence=evidence)
                rows.append(record)
                if synthetic:current=dict(completion_ms=clock,own_service_ms=service,waiting_ms=clock-service,quality_loss=q,context=context,evidence=evidence)
            alias_audit[alias]=dict(family=family,context=context,requests=rows,visible_outstanding_count=len(scenario[alias]['outstanding']))
        terms={t:0. for t in ('Q','L','P')};per_task={}
        for t,v in latencies.items():
            if not v:continue
            ref=self.refs.get(t)
            if not _finite(ref,True):unknown.append('UNKNOWN_TASK_REFERENCE:'+t);continue
            alpha=self.workload[t]/self.total;mean=sum(v)/len(v);p95=_p95(v);qv=qualities[t]
            expected=self.expected_counts[t];tail_k=self.tail_counts[t]
            if len(qv)!=len(v):unknown.append('INCOMPLETE_QUALITY:'+t)
            q=sum(qv)/len(qv) if len(qv)==len(v) else None
            # Fixed expected full-window denominators prevent a low-cost new
            # request from diluting existing pending latency or quality loss.
            # The fixed-K sum is append-monotone. Unobserved slots are zero
            # placeholders, NOT a claim about true completed/future latencies.
            tail_sum=sum(sorted(v,reverse=True)[:tail_k])
            terms['L']+=alpha*sum(v)/expected/ref
            terms['P']+=alpha*tail_sum/tail_k/ref
            if q is not None:terms['Q']+=alpha*sum(qv)/expected
            per_task[t]=dict(requests=len(v),mean_host_e2e_proxy_ms=mean,p95_host_e2e_proxy_ms=p95,
                aggregate_Q_proxy=q,expected_window_requests=expected,fixed_tail_slots=tail_k,
                pending_latency_sum_ms=sum(v),pending_tail_sum_ms=tail_sum,
                P_scope='Fixed expected-window upper5percent tail-sum contribution; not measured or predicted full-window P95')
        terms['E']=energy/(self.energy_reference_watts*self.horizon_seconds) if energy_known else None
        known_cost=sum(self.weights[t]*terms[t] for t in ('Q','L','P')) if not unknown else None
        if known_cost is not None and terms['E'] is not None:known_cost+=self.weights['E']*terms['E']
        return dict(known_proxy_cost=known_cost,terms=terms,per_task=per_task,current=current,aliases=alias_audit,
                    normalization='L/Q sums divided by fixed expected full-window N; P largest fixed K=ceil(.05*N) pending latency sum divided by K with absent slots zero',
                    unknown_reasons=unknown,energy_proxy_known=energy_known)

    def choose(self,task,pool_id,now_ns,pending_snapshot):
        self._prediction_cache={}
        if not _ns(now_ns):raise ValueError('Nonnegative integer timestamp required')
        if task not in self.workload:raise ValueError('Task outside declared workload')
        try:snapshot=self._snapshot(pending_snapshot,now_ns)
        except ValueError as exc:return self._result(task,pool_id,now_ns,[],None,[str(exc)])
        baseline=self._forecast(snapshot,now_ns) if self.mode=='global_incremental' else None
        audit=[]
        for alias,spec in sorted(self.aliases.items()):
            record=dict(alias=alias,family=spec['family'],supported=task in spec['supported_tasks'],score=None,
                predicted_completion_ms=None,predicted_waiting_ms=None,predicted_own_service_ms=None,unknown_reasons=[])
            if not record['supported']:record['status']='UNSUPPORTED_TASK';audit.append(record);continue
            if not self.allow_spill and task=='text' and spec['family']!='text' and any(v['family']=='text' for v in self.aliases.values()):
                record.update(status='SPILL_DISABLED_CONTROL',candidate_eligible=False);audit.append(record);continue
            forecast=self._forecast(snapshot,now_ns,alias,task,pool_id);current=forecast['current'];reasons=record['unknown_reasons']
            if current is None:reasons.append('UNKNOWN_CURRENT_PREDICTION')
            if forecast['known_proxy_cost'] is None:reasons.extend(forecast['unknown_reasons'])
            ref=self.refs.get(task)
            if not _finite(ref,True):reasons.append('UNKNOWN_LATENCY_REFERENCE')
            if current is not None:
                q=current['quality_loss']
                if not _finite(q):reasons.append('UNKNOWN_AGGREGATE_QUALITY')
                record.update(predicted_completion_ms=current['completion_ms'],predicted_waiting_ms=current['waiting_ms'],
                    predicted_own_service_ms=current['own_service_ms'],aggregate_quality_loss=q,current_service_evidence=current['evidence'])
            margin=self.margin_ms if spec['family'].startswith('generalist_') and any(v['family']==task for v in self.aliases.values()) else 0.
            if not reasons:
                if self.mode=='conservative':
                    margin_cost=self.weights['L']*margin/ref
                    score=self.weights['L']*current['completion_ms']/ref+self.weights['Q']*current['quality_loss']+margin_cost
                    parts=dict(L=self.weights['L']*current['completion_ms']/ref,Q=self.weights['Q']*current['quality_loss'],margin=margin_cost)
                else:
                    if baseline['known_proxy_cost'] is None:reasons.extend(baseline['unknown_reasons'])
                    margin_cost=self.weights['L']*self.workload[task]/self.total*margin/ref/self.expected_counts[task]
                    score=forecast['known_proxy_cost']-baseline['known_proxy_cost']+margin_cost if not reasons else None
                    parts={t:(self.weights[t]*(forecast['terms'][t]-baseline['terms'][t]) if forecast['terms'][t] is not None and baseline['terms'][t] is not None else None) for t in ('Q','L','P','E')};parts['margin']=margin_cost
                record.update(score=score,status='KNOWN_PARTIAL_PROXY' if score is not None else 'UNKNOWN_CANDIDATE',
                    score_contributions=parts,margin_ms=margin,forecast=forecast,baseline=baseline,
                    same_resident_memory_delta=0.,full_memory_cost=None,energy_proxy_known=forecast['energy_proxy_known'],
                    unknown_full_U_terms=['T','M_absolute','E_startup','future_shared_gpu_interference']+([] if forecast['energy_proxy_known'] else ['E_dynamic']))
            else:record.update(status='UNKNOWN_CANDIDATE',forecast=forecast)
            audit.append(record)
        known=[r for r in audit if r['score'] is not None];chosen=min(known,key=lambda r:(r['score'],r['predicted_completion_ms'],r['alias'])) if known else None
        if chosen:
            specialist=[r for r in known if r['family']==task]
            best_special=min(specialist,key=lambda r:(r['score'],r['alias'])) if specialist else None
            for r in audit:
                r['delta_proxy_cost_vs_best_specialist']=r['score']-best_special['score'] if r['score'] is not None and best_special else None
        return self._result(task,pool_id,now_ns,audit,chosen,[])

    def _result(self,task,pool_id,now,audit,chosen,errors):
        return dict(alias=chosen['alias'] if chosen else None,status='CHOSEN' if chosen else 'UNKNOWN_NO_COMPARABLE_RESIDENT_ALIAS',
            mode=self.mode,task=task,pool_id=pool_id,now_ns=now,audit=audit,unknown_reasons=errors,
            risk_quantile=self.risk_quantile,margin_ms=self.margin_ms,concurrency_context=self.concurrency_context,allow_spill=self.allow_spill,
            objective='Calibrated conservative L+Q completion proxy' if self.mode=='conservative' else 'Scenario-minus-baseline fixed-window-denominator pending L/Q, fixed-K tail-sum P, and E proxy',
            snapshot_scope='All visible host/pipe outstanding; not actual worker active concurrency',
            horizon_scope='Current outstanding plus current request only; fixed N=workload*horizon and K=max(1,ceil(.05*N)); no future arrivals',
            P_scope='Pending upper-tail sum surrogate with fixed expected-window K; not whole-window P95 or exact delta U',
            residual_scope='Conditional historical residual at inferred host age; tail extrapolation remains unverified',
            full_U=None,absolute_quality_gate=False,submitted_work_migrated=False,
            unknown_supported_candidates=sum(r.get('status')=='UNKNOWN_CANDIDATE' for r in audit),
            trigger='NO_KNOWN_CANDIDATE' if chosen is None else 'MINIMUM_KNOWN_PARTIAL_PROXY')
