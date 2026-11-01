"""Causal observed-prefix utility routing with explicit evidence gates.

Q/T/L/P/E use the identical observed cohort for every hypothetical destination.
Resident-size accounting has a common proxy constant. Actual peak-memory M,
future arrivals, true counterfactual service and physical six-term U remain
UNKNOWN. No answers or per-input evaluation labels enter this module.
"""
from __future__ import annotations
import copy
import math
import statistics
from .model import CoupledEventModel,UnknownEvidence,finite


def percentile(values,q=.95):
    values=sorted(values);position=(len(values)-1)*q;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


class CoupledUtilityPolicy:
    def __init__(self,aliases,service_provider,quality_loss,weights,latency_refs_ms,workload,*,
        power_provider=None,scenarios=('mean','p90'),allow_spill=True,utility_margin=0.,
        energy_reference_watts=450.,exposure_rate_rps=None,exposure_burst=1.,cooldown_ms=0.,
        drift_ratio_limit=2.,memory_delta_penalty_upper_norm=None,maximum_pending_for_forecast=32):
        self.aliases=copy.deepcopy(aliases);self.quality_loss=copy.deepcopy(quality_loss)
        self.weights=dict(weights);self.refs=dict(latency_refs_ms);self.workload=dict(workload)
        if set(weights)!=set('QTLPME') or any(not finite(v) or v<0 for v in weights.values()):raise ValueError('Explicit nonnegative six weights required')
        if abs(sum(weights.values())-1)>1e-12:raise ValueError('Weights must sum to1')
        if not scenarios or len(set(scenarios))!=len(scenarios):raise ValueError('Explicit distinct service scenarios required')
        if not finite(energy_reference_watts) or energy_reference_watts<=0:raise ValueError('Positive energy reference required')
        if any(not finite(v) or v<=0 for v in workload.values()) or any(not finite(latency_refs_ms.get(t)) or latency_refs_ms[t]<=0 for t in workload):raise ValueError('Positive workload/reference required')
        for value in (utility_margin,cooldown_ms):
            if not finite(value) or value<0:raise ValueError('Nonnegative margin/cooldown required')
        if exposure_rate_rps is not None and (not finite(exposure_rate_rps) or exposure_rate_rps<=0):raise ValueError('Positive exposure rate required')
        if not finite(exposure_burst) or exposure_burst<1:raise ValueError('At least one burst token required')
        if not finite(drift_ratio_limit) or drift_ratio_limit<=1:raise ValueError('Drift threshold >1 required')
        if memory_delta_penalty_upper_norm is not None and (not finite(memory_delta_penalty_upper_norm) or memory_delta_penalty_upper_norm<0):raise ValueError('Explicit nonnegative memory penalty bound required')
        if type(maximum_pending_for_forecast) is not int or maximum_pending_for_forecast<1:raise ValueError('Positive integer pending forecast budget required')
        self.model=CoupledEventModel(aliases,service_provider,power_provider)
        self.scenarios=tuple(scenarios);self.allow_spill=allow_spill;self.margin=utility_margin
        self.energy_reference_watts=energy_reference_watts;self.exposure_rate_rps=exposure_rate_rps
        self.burst=exposure_burst;self.tokens=exposure_burst;self.last_refill_ns=None;self.cooldown_ns=round(cooldown_ms*1e6)
        self.drift_ratio_limit=drift_ratio_limit;self.cooldown_until_ns=0;self.memory_bound=memory_delta_penalty_upper_norm
        self.maximum_pending_for_forecast=maximum_pending_for_forecast
        self.completed={};self.phase_start_ns=None;self.deadline_ns=None;self.current=None;self.sequence=0;self.feedback=[]
        self.predicted_own={};self.admitted={}

    def warm_cache(self):return None
    def start_phase(self,start_ns,deadline_ns):
        if type(start_ns) is not int or type(deadline_ns) is not int or deadline_ns<=start_ns:raise ValueError('Valid phase start/deadline required')
        if self.completed or self.sequence:raise ValueError('Fresh policy per phase required')
        self.phase_start_ns=start_ns;self.deadline_ns=deadline_ns
    def set_arrival(self,request_id,arrival_ns):
        if not isinstance(request_id,str) or type(arrival_ns) is not int:raise ValueError('Explicit current arrival identity required')
        self.current=dict(request_id=request_id,arrival_ns=arrival_ns)

    def observe_admission(self,record):
        """Record metadata for fixed backgrounds and committed text alike."""
        alias=record.get('alias',record.get('model'));rid=record.get('request_id')
        arrival=record.get('arrival_ns',record.get('scheduled_time_ns',record.get('enqueued_ns')))
        if alias not in self.aliases or not isinstance(rid,str) or type(arrival) is not int:raise ValueError('Invalid committed admission identity')
        if rid in self.admitted:raise ValueError('Duplicate committed admission')
        self.admitted[rid]=dict(request_id=rid,alias=alias,task=record['task'],pool_id=record['pool_id'],arrival_ns=arrival)

    def observe_completed(self,records):
        """Whitelist host timing and actual service only, never outputs/labels."""
        allowed=('request_id','alias','task','pool_id','arrival_ns','completion_ns','service_ms')
        for source in records:
            source=dict(source);rid=source.get('request_id');prior=self.admitted.get(rid,{})
            source.update(alias=source.get('alias',source.get('model',prior.get('alias'))),
                arrival_ns=source.get('arrival_ns',prior.get('arrival_ns',source.get('enqueued_ns'))),
                completion_ns=source.get('completion_ns',source.get('completion_time_ns')),
                service_ms=source.get('service_ms',source.get('worker_service_ms')))
            r={k:source.get(k,prior.get(k)) for k in allowed};rid=r['request_id']
            if rid in self.completed:continue
            if (not isinstance(rid,str) or r['alias'] not in self.aliases or type(r['arrival_ns']) is not int
                or type(r['completion_ns']) is not int or r['completion_ns']<r['arrival_ns']):raise ValueError('Invalid completed timing/identity')
            self.completed[rid]=r
            predicted=self.predicted_own.pop(rid,None);actual=r['service_ms']
            if predicted and finite(actual) and actual>0:
                ratio=actual/predicted
                flagged=ratio>self.drift_ratio_limit
                if flagged:self.cooldown_until_ns=max(self.cooldown_until_ns,r['completion_ns']+self.cooldown_ns)
                self.feedback.append(dict(request_id=rid,actual_to_predicted_ratio=ratio,drift_triggered=flagged,
                    cooldown_until_ns=self.cooldown_until_ns,rule='Actual completed worker service / frozen predicted own-service; no online refit'))

    def _queues(self,snapshot,now):
        if set(snapshot)!=set(self.aliases):raise ValueError('Full resident snapshot required')
        queues={};ids=set(self.completed)
        for alias,state in snapshot.items():
            rows=[]
            for i,source in enumerate(state.get('outstanding',[])):
                rid=source['request_id']
                if rid in ids:raise ValueError('Duplicate completed/pending identity')
                ids.add(rid)
                arrival=source.get('arrival_ns',self.admitted.get(rid,{}).get('arrival_ns',source.get('enqueued_ns')))
                if type(arrival) is not int or arrival>now:raise ValueError('Future arrival in snapshot')
                row=dict(request_id=rid,task=source['task'],pool_id=source['pool_id'],arrival_ns=arrival,alias=alias)
                if i==0:
                    submitted=source.get('actual_submit_ns')
                    if submitted is None:submitted=source.get('enqueued_ns',arrival)
                    head_start=max(submitted,state.get('last_completed_ns') or submitted)
                    row['head_age_ms']=(now-head_start)/1e6
                    row['head_age_scope']='Host-visible submit/prior-completion proxy, not observed GPU service start or reconstructed historical work'
                rows.append(row)
            queues[alias]=rows
        return queues

    def _terms(self,forecast,current_rows):
        rows=[]
        for r in self.completed.values():rows.append(dict(r,finish_ns=r['completion_ns']))
        rows.extend(forecast['finishes'].values())
        expected=set(self.completed)|{r['request_id'] for vv in current_rows.values() for r in vv}
        if len(rows)!=len(expected) or {r['request_id'] for r in rows}!=expected:raise ValueError('Candidate cohort mismatch')
        total=sum(self.workload.values());Q=L=P=0.;counts={}
        for task,rate in self.workload.items():
            cohort=[r for r in rows if r['task']==task];counts[task]=len(cohort)
            if not cohort:raise UnknownEvidence('NO_OBSERVED_PREFIX_FOR_TASK:'+task)
            quality=[];lat=[]
            for r in cohort:
                family=self.aliases[r['alias']]['family'];loss=self.quality_loss.get(family,{}).get(task)
                if not finite(loss) or loss<0:raise UnknownEvidence('UNKNOWN_AGGREGATE_QUALITY:'+family+':'+task)
                quality.append(loss);lat.append((r['finish_ns']-r['arrival_ns'])/1e6)
            alpha=rate/total;Q+=alpha*statistics.mean(quality);L+=alpha*statistics.mean(lat)/self.refs[task];P+=alpha*percentile(lat)/self.refs[task]
        T=1-sum(r['finish_ns']<=self.deadline_ns for r in rows)/len(rows)
        horizon=(self.deadline_ns-self.phase_start_ns)/1e9
        energy=forecast['board_energy_joules']
        if not finite(energy):raise UnknownEvidence('UNKNOWN_FUTURE_BOARD_ENERGY')
        E=energy/(self.energy_reference_watts*horizon)
        terms=dict(Q=Q,T=T,L=L,P=P,M=None,E=E)
        # Completed-window energy and startup are common to every current
        # hypothetical choice. Their absolute values remain unobserved here.
        cost=sum(self.weights[t]*terms[t] for t in 'QTLPE')
        return dict(terms=terms,cost=cost,cohort_size=len(rows),cohort_ids=sorted(expected),task_counts=counts,
            quality_scope='Frozen family/task aggregate loss forecast, not measured per-input accuracy',
            E_scope='Predicted remaining measurement-window whole-board energy; observed past/startup common additive terms omitted',
            M_absolute=None,M_delta=None,resident_size_proxy_delta=0.,actual_peak_memory_invariance_proved=False,
            full_U=None,scope='Exact metric arithmetic only for this identical observed-prefix forecast, not full future workload or physical six-term U')

    def _forecast(self,queues,now_ns,deadline_ns,scenario):
        """Extension point; historical observed-prefix behavior stays explicit."""
        return self.model.forecast(queues,now_ns,deadline_ns,scenario)

    def _own_service(self,forecast,current,alias,transition):
        return transition['service_ms'][alias]

    def choose(self,task,pool_id,now_ns,pending_snapshot):
        if type(now_ns) is not int:raise ValueError('Integer host time required')
        self.sequence+=1;current=self.current or dict(request_id=f'coupled-current:{self.sequence}',arrival_ns=now_ns);self.current=None
        if current['arrival_ns']>now_ns:raise ValueError('Future current arrival')
        if any(r['completion_ns']>now_ns for r in self.completed.values()):raise ValueError('Future completion feedback')
        queues=self._queues(pending_snapshot,now_ns)
        if current['request_id'] in self.completed or any(current['request_id']==r['request_id'] for v in queues.values() for r in v):raise ValueError('Duplicate current identity')
        specialists=sorted(a for a,s in self.aliases.items() if s['family']=='text' and task in s['supported_tasks'])
        if not specialists:raise ValueError('Resident text specialist fallback required')
        fallback=min(specialists,key=lambda a:(len(queues[a]),a));audit=[]
        gates=[]
        if self.phase_start_ns is None:gates.append('UNKNOWN_PHASE_DEADLINE')
        elif now_ns<self.phase_start_ns or now_ns>self.deadline_ns:gates.append('OUTSIDE_MEASUREMENT_ARRIVAL_WINDOW')
        if not self.allow_spill:gates.append('SPILL_DISABLED_CONTROL')
        if now_ns<self.cooldown_until_ns:gates.append('COMPLETION_FEEDBACK_DRIFT_COOLDOWN')
        if self.last_refill_ns is not None and now_ns<self.last_refill_ns:raise ValueError('Nonmonotonic routing clock')
        if self.exposure_rate_rps is not None:
            elapsed=0. if self.last_refill_ns is None else (now_ns-self.last_refill_ns)/1e9
            self.tokens=min(self.burst,self.tokens+elapsed*self.exposure_rate_rps)
            if self.tokens<1:gates.append('EXPOSURE_BUDGET_EXHAUSTED')
        self.last_refill_ns=now_ns
        pending_count=sum(len(rows) for rows in queues.values())
        budget_exceeded=pending_count>self.maximum_pending_for_forecast
        if budget_exceeded:gates.append('DETERMINISTIC_PENDING_FORECAST_BUDGET_EXCEEDED')
        candidates=[] if budget_exceeded else sorted(a for a,s in self.aliases.items() if task in s['supported_tasks'])
        for alias in candidates:
            entry=dict(alias=alias,family=self.aliases[alias]['family'],score=None,scenarios={},unknown_reasons=[])
            if self.phase_start_ns is None:entry['unknown_reasons'].append('UNKNOWN_PHASE_DEADLINE')
            else:
                for scenario in self.scenarios:
                    variant=copy.deepcopy(queues);variant[alias].append(dict(current,task=task,pool_id=pool_id,alias=alias,head_age_ms=0.))
                    try:
                        forecast=self._forecast(variant,now_ns,self.deadline_ns,scenario)
                        metric=self._terms(forecast,variant)
                        transition=next(v for v in forecast['transitions'] if current['request_id'] in v['active_requests'].values())
                        metric['predicted_own_service_ms']=self._own_service(forecast,current,alias,transition)
                        metric['predicted_completion_ms']=(forecast['finishes'][current['request_id']]['finish_ns']-current['arrival_ns'])/1e6
                        metric['competition_events']=forecast['transitions'];entry['scenarios'][scenario]=metric
                    except UnknownEvidence as exc:entry['unknown_reasons'].append(scenario+':'+str(exc))
            if len(entry['scenarios'])==len(self.scenarios):entry['score']=max(v['cost'] for v in entry['scenarios'].values())
            audit.append(entry)
        known_special=[r for r in audit if r['alias'] in specialists and r['score'] is not None]
        baseline=min(known_special,key=lambda r:(r['score'],r['alias'])) if known_special else None
        chosen=baseline;eligible_spill=[]
        if baseline and not gates:
            for entry in audit:
                if entry['alias'] in specialists or entry['score'] is None:continue
                deltas={s:entry['scenarios'][s]['cost']-baseline['scenarios'][s]['cost'] for s in self.scenarios}
                extra=self.weights['M']*self.memory_bound if self.memory_bound is not None else 0.
                entry['scenario_delta_cost_vs_selected_specialist']=deltas
                entry['memory_penalty_bound_applied']=self.memory_bound
                if max(deltas.values())+extra < -self.margin:eligible_spill.append(entry)
            if eligible_spill:chosen=min(eligible_spill,key=lambda r:(max(r['scenario_delta_cost_vs_selected_specialist'].values()),r['alias']))
        alias=chosen['alias'] if chosen else fallback;spill=alias not in specialists
        if spill and self.exposure_rate_rps is not None:self.tokens-=1
        if chosen:self.predicted_own[current['request_id']]=chosen['scenarios'][self.scenarios[0]]['predicted_own_service_ms']
        if not chosen and not budget_exceeded:gates.append('NO_COMPLETE_SHARED_RESOURCE_FORECAST')
        return dict(alias=alias,status='SPECIALIST_COMPUTE_BUDGET_FALLBACK' if budget_exceeded else ('CHOSEN' if chosen else 'SPECIALIST_EVIDENCE_FALLBACK'),
            trigger='CONDITIONAL_PREFIX_UTILITY_SPILL' if spill else 'SPECIALIST_CONTROL_OR_EVIDENCE_FALLBACK',
            task=task,pool_id=pool_id,now_ns=now_ns,audit=audit,gates=gates,actual_spill_selected=spill,
            baseline_specialist=baseline['alias'] if baseline else fallback,tokens_remaining=self.tokens,
            cooldown_until_ns=self.cooldown_until_ns,feedback=list(self.feedback[-10:]),
            forecast_computation_budget=dict(maximum_pending_for_forecast=self.maximum_pending_for_forecast,
                pending_count=pending_count,exceeded=budget_exceeded,forecast_skipped=budget_exceeded,
                cohort_not_truncated=True,wall_clock_timeout_used=False),
            full_U=None,M_absolute=None,M_delta=None,actual_peak_memory_invariance_proved=False,
            conditional_proxy_terms='Q/T/L/P and remaining whole-board E; resident-size-only M proxy constant',
            unknown_terms=['Physical peak M and route-dependent workspace','Absolute past/startup energy','Future arrivals','Physical interference counterfactual truth'],
            objective_scope='Same observed-prefix cohort, common measurement deadline and exact task empirical P95 arithmetic; conditional proxy only',
            memory_comparison_scope='Only resident-size proxy difference is0. Actual measured peak-memory difference is UNKNOWN; no six-term improvement certificate.',
            physical_six_term_improvement_certified=False,answers_or_future_arrivals_used=False)
