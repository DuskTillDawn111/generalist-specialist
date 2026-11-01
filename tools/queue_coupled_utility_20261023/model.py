"""Causal shared-rate FIFO forecasts for the observed cohort only.

Every active-set transition reprices every active head. Service providers and
whole-board power evidence are injected; absent evidence is UNKNOWN. These are
conditional numerical forecasts, never a physical interference certificate.
"""
from __future__ import annotations
from collections import deque
import copy
import math
from gsdeploy.forecast_errors import UnknownEvidence


def finite(x):return type(x) in (int,float) and math.isfinite(x)


class CoupledEventModel:
    """Provider API: service/residual/power(request, active, aliases, scenario).

    service(request, active, aliases, scenario) returns service_ms, known,
    context_supported. residual adds age_ms after request and returns
    remaining_ms, known, tail_supported. power(active, aliases, scenario)
    returns watts, known. active maps aliases to their FIFO head records.
    Predicted whole-board energy is integrated once, up to the measurement
    deadline; post-deadline drain is omitted to match the original E window.
    """
    def __init__(self,aliases,service_provider,power_provider=None,*,max_events=10000):
        self.aliases=copy.deepcopy(aliases);self.service_provider=service_provider
        self.power_provider=power_provider or service_provider;self.max_events=max_events

    def _service(self,row,active,scenario):
        value=self.service_provider.service(row,active,self.aliases,scenario)
        if (value.get('known') is not True or value.get('context_supported') is not True
            or not finite(value.get('service_ms')) or value['service_ms']<=0):
            raise UnknownEvidence('UNKNOWN_SHARED_SERVICE_CONTEXT:'+row['alias']+':'+row['task'])
        return value

    def forecast(self,queues,now_ns,deadline_ns,scenario='mean',*,require_energy=True):
        if type(now_ns) is not int or type(deadline_ns) is not int:raise ValueError('Explicit integer phase times required')
        pending={a:deque(copy.deepcopy(queues.get(a,[]))) for a in self.aliases}
        seen=set()
        for alias,rows in pending.items():
            for row in rows:
                row['alias']=alias;rid=row['request_id']
                if rid in seen:raise ValueError('Duplicate pending identity')
                seen.add(rid)
                if row['task'] not in self.aliases[alias]['supported_tasks']:raise ValueError('Unsupported pending route')
                if not finite(row.get('arrival_ns')) or row['arrival_ns']>now_ns:raise ValueError('Future/invalid arrival in observed cohort')
        active={a:q[0] for a,q in pending.items() if q};remaining={};evidence=[]
        for alias,row in active.items():
            service=self._service(row,active,scenario)
            age=row.get('head_age_ms',0.)
            if not finite(age) or age<0:raise ValueError('Invalid inferred head age')
            if age>0:
                residual=self.service_provider.residual(row,age,active,self.aliases,scenario)
                if (residual.get('known') is not True or residual.get('tail_supported') is not True
                    or not finite(residual.get('remaining_ms')) or residual['remaining_ms']<=0):
                    raise UnknownEvidence('UNKNOWN_CONDITIONAL_RESIDUAL_TAIL:'+alias)
                remaining[alias]=residual['remaining_ms']/service['service_ms']
                evidence.append(dict(alias=alias,kind='INITIAL_RESIDUAL',evidence=residual))
            else:remaining[alias]=1.
        clock=float(now_ns);deadline=float(deadline_ns);energy=0.;energy_known=True;finishes={};transitions=[];events=0
        while active:
            events+=1
            if events>self.max_events:raise UnknownEvidence('EVENT_LIMIT')
            rates={a:self._service(row,active,scenario) for a,row in active.items()}
            power=self.power_provider.power(active,self.aliases,scenario)
            if power.get('known') is not True or not finite(power.get('watts')) or power['watts']<0:
                energy_known=False
                if require_energy:raise UnknownEvidence('UNKNOWN_WHOLE_BOARD_POWER_CONTEXT')
            durations={a:remaining[a]*rates[a]['service_ms']*1e6 for a in active}
            delta=min(durations.values())
            if not finite(delta) or delta<=0:raise UnknownEvidence('INVALID_COMPLETION_EVENT')
            until=clock+delta
            if energy_known:energy+=power['watts']*max(0.,min(until,deadline)-min(clock,deadline))/1e9
            for alias in active:remaining[alias]=max(0.,remaining[alias]-delta/(rates[alias]['service_ms']*1e6))
            completed=[a for a,d in durations.items() if abs(d-delta)<=max(1e-6,abs(delta)*1e-12)]
            transitions.append(dict(time_ns=clock,until_ns=until,active_requests={a:r['request_id'] for a,r in active.items()},
                service_ms={a:r['service_ms'] for a,r in rates.items()},board_watts=power.get('watts') if power.get('known') else None,
                repriced_aliases=sorted(active),completed_aliases=sorted(completed)))
            clock=until
            for alias in completed:
                row=pending[alias].popleft();finishes[row['request_id']]=dict(row,finish_ns=clock)
                if pending[alias]:active[alias]=pending[alias][0];remaining[alias]=1.
                else:active.pop(alias);remaining.pop(alias)
        # The whole board remains resident until the phase deadline.
        if clock<deadline:
            idle=self.power_provider.power({},self.aliases,scenario)
            if idle.get('known') is not True or not finite(idle.get('watts')) or idle['watts']<0:
                energy_known=False
                if require_energy:raise UnknownEvidence('UNKNOWN_IDLE_BOARD_POWER')
            if energy_known:energy+=idle['watts']*(deadline-clock)/1e9
        return dict(finishes=finishes,board_energy_joules=energy if energy_known else None,energy_known=energy_known,
            transitions=transitions,initial_residual_evidence=evidence,drain_finish_ns=clock,
            scope='Observed-cohort conditional shared-rate proxy; all active paths repriced on each start/finish; no future arrivals or physical guarantee')
