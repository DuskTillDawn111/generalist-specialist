"""Versioned DAG engine with atomic simulated-state checkpoint/continuation.

Positive activity completion intervals are rounded upward to a whole nanosecond
before being added to the integer absolute clock. Provider service requirements
remain unmodified; service consumption uses actual integer elapsed intervals.
This prevents floating-point absolute-clock cancellation without relaxing budgets.

Distinct streams are never summed into one segment. The default timing provider
is explicitly profiled replay, not physical prediction. Callers can inject
independently fitted node durations and host-release rules without altering the
observed GPU activity target. GPU scheduling/copy service remain hypotheses.
"""
from __future__ import annotations

import collections
import copy
import heapq
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_checkpoint_v7 import STATE_FIELDS, contract, export_checkpoint, import_checkpoint


class DagFullHorizonModel:
    def __init__(self,aliases,plans,*,allow_profiled_replay=False,
                 allow_independent_copy_hypothesis=False,allow_context_time_slice_hypothesis=False,
                 allow_no_implicit_default_stream_hypothesis=False,
                 lifecycle_provider=None,scenario='mean',duration_provider=None,release_provider=None,
                 power_provider=None,memory_provider=None,require_energy=False,require_memory=False,
                 max_events=500000,queue_diagnostics=None,checkpoint_binding=None):
        if not all(type(v) is bool for v in (allow_profiled_replay,allow_independent_copy_hypothesis,
                                             allow_context_time_slice_hypothesis,allow_no_implicit_default_stream_hypothesis)):
            raise ValueError('Explicit boolean DAG model hypothesis flags required')
        self.aliases=aliases;self.plans=plans;self.allow_profiled_replay=allow_profiled_replay
        self.allow_independent_copy_hypothesis=allow_independent_copy_hypothesis
        self.allow_context_time_slice_hypothesis=allow_context_time_slice_hypothesis
        self.allow_no_implicit_default_stream_hypothesis=allow_no_implicit_default_stream_hypothesis
        self.lifecycle_provider=lifecycle_provider;self.scenario=scenario
        self.duration_provider=duration_provider;self.release_provider=release_provider;self.max_events=max_events
        self.power_provider=power_provider;self.memory_provider=memory_provider
        self.require_energy=require_energy;self.require_memory=require_memory
        self.queue_diagnostics=queue_diagnostics
        self.checkpoint_binding=copy.deepcopy(checkpoint_binding)

    def forecast(self,requests,now_ns,deadline_ns,*,drain_deadline_ns=None,
                 checkpoint=None,pause_after_event_count=None):
        if type(now_ns) is not int or type(deadline_ns) is not int or now_ns>deadline_ns:
            raise ValueError('Integer measurement window required')
        if pause_after_event_count is not None and (type(pause_after_event_count) is not int or pause_after_event_count<=0):
            raise ValueError('Positive absolute event count required for atomic pause')
        checkpoint_contract=(contract(self,requests,now_ns,deadline_ns,drain_deadline_ns)
            if checkpoint is not None or pause_after_event_count is not None else None)
        arrivals=[];cohort={}
        for index,row in enumerate(copy.deepcopy(requests)):
            if (row['request_id'] in cohort or row['alias'] not in self.aliases or type(row['arrival_ns']) is not int
                    or row['task'] not in self.aliases[row['alias']]['supported_tasks']
                    or not now_ns<=row['arrival_ns']<deadline_ns):
                raise ValueError('Distinct public request identity/arrival/route required')
            cohort[row['request_id']]=row;heapq.heappush(arrivals,(row['arrival_ns'],index,row))
        pending={a:collections.deque() for a in self.aliases};running={};active={};ready=[]
        finish_events=[];wake_events=[];unready={};finishes={};starts={};queue_wait={}
        sequence=0;clock=now_ns;event_count=0;maximum_parallel=0;maximum_owner_parallel=collections.Counter()
        all_completed_nodes=0;max_waiting=0;receipt_pending={};release_hypotheses=[]
        energy=0.;peak=0.;energy_known=self.power_provider is not None;memory_known=self.memory_provider is not None
        resource_transitions=[]

        if checkpoint is not None:
            restored=import_checkpoint(checkpoint,self,checkpoint_contract)
            arrivals=restored['arrivals']
            cohort=restored['cohort']
            pending=restored['pending']
            running=restored['running']
            active=restored['active']
            ready=restored['ready']
            finish_events=restored['finish_events']
            wake_events=restored['wake_events']
            unready=restored['unready']
            finishes=restored['finishes']
            starts=restored['starts']
            queue_wait=restored['queue_wait']
            sequence=restored['sequence']
            clock=restored['clock']
            event_count=restored['event_count']
            maximum_parallel=restored['maximum_parallel']
            maximum_owner_parallel=restored['maximum_owner_parallel']
            all_completed_nodes=restored['all_completed_nodes']
            max_waiting=restored['max_waiting']
            receipt_pending=restored['receipt_pending']
            release_hypotheses=restored['release_hypotheses']
            energy=restored['energy']
            peak=restored['peak']
            energy_known=restored['energy_known']
            memory_known=restored['memory_known']
            resource_transitions=restored['resource_transitions']
            if pause_after_event_count is not None and pause_after_event_count<=event_count:
                raise ValueError('Continuation pause must follow consumed event count')

        def push_ready(owner,node):
            nonlocal sequence
            state=running[owner];node_id=node['id'];release=node.get('release_ns',0)
            if self.release_provider is not None:
                release=self.release_provider(node,state['row'])
            if isinstance(release,dict):
                if set(release)!={'not_before_offset_ns','after_predecessors_ns'}:
                    raise UnknownEvidence('EXPLICIT_DAG_HOST_SUPPLY_BOUNDARIES_REQUIRED')
                offset,delay=release['not_before_offset_ns'],release['after_predecessors_ns']
                if any(type(value) is not int or value<0 for value in (offset,delay)):
                    raise UnknownEvidence('NONNEGATIVE_INTEGER_DAG_HOST_SUPPLY_REQUIRED')
                release_at=max(state['start_ns']+offset,clock+delay)
            else:
                if type(release) is not int or release<0:raise UnknownEvidence('DAG_HOST_RELEASE_MODEL_REQUIRED')
                release_at=state['start_ns']+release
            sequence+=1;heapq.heappush(ready,(release_at,sequence,owner,node_id))

        def complete_node(owner,node_id):
            nonlocal all_completed_nodes
            state=running[owner]
            if node_id in state['done']:raise ValueError('Duplicate DAG node completion')
            state['done'].add(node_id);all_completed_nodes+=1
            for successor in state['successors'][node_id]:
                state['incoming'][successor]-=1
                if state['incoming'][successor]==0:push_ready(owner,state['nodes'][successor])

        def start_requests():
            for owner in self.aliases:
                if owner in running or owner in unready or not pending[owner]:continue
                row=pending[owner].popleft();key=owner,row['task'],row['pool_id']
                plan=self.plans.get(key)
                if plan is None:raise UnknownEvidence('UNKNOWN_DAG_INPUT:'+owner+':'+row['pool_id'])
                if plan.get('cyclic') or plan.get('unresolved') or not plan.get('nodes'):
                    raise UnknownEvidence('UNRESOLVED_REQUEST_CUDA_DAG')
                if (str(plan.get('default_stream_semantics','')).startswith('UNMEASURED')
                        and not self.allow_no_implicit_default_stream_hypothesis):
                    raise UnknownEvidence('DEFAULT_STREAM_SEMANTICS_HYPOTHESIS_OPT_IN_REQUIRED')
                nodes={n['id']:n for n in plan['nodes']}
                if len(nodes)!=len(plan['nodes']):raise ValueError('Duplicate request DAG node')
                incoming={n:0 for n in nodes};successors={n:[] for n in nodes}
                for edge in plan['edges']:
                    before,after=edge['before'],edge['after']
                    if before not in nodes or after not in nodes:raise ValueError('Foreign request DAG dependency')
                    incoming[after]+=1;successors[before].append(after)
                running[owner]=dict(row=row,start_ns=clock,nodes=nodes,incoming=incoming,successors=successors,done=set())
                starts[row['request_id']]=clock;queue_wait[row['request_id']]=clock-row['arrival_ns']
                if self.queue_diagnostics is not None:self.queue_diagnostics.mark_started(dict(row,start_ns=clock))
                for node_id,count in incoming.items():
                    if count==0:push_ready(owner,nodes[node_id])

        def finish_requests():
            nonlocal sequence
            for owner,state in list(running.items()):
                if len(state['done'])!=len(state['nodes']):continue
                row=state['row'];delays=dict(parent_receipt_ms=0.,worker_next_ready_ms=0.)
                if self.lifecycle_provider is not None:
                    delays=self.lifecycle_provider.delays(row,self.aliases,self.scenario)
                    if delays.get('known') is not True:raise UnknownEvidence('UNKNOWN_DAG_POST_HANDLER_LIFECYCLE')
                for field in ('parent_receipt_ms','worker_next_ready_ms'):
                    value=delays[field]
                    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                        raise UnknownEvidence('UNKNOWN_DAG_POST_HANDLER_BOUNDARY:'+field)
                receipt_at=clock+round(delays['parent_receipt_ms']*1e6)
                ready_at=clock+round(delays['worker_next_ready_ms']*1e6)
                record=dict(row,finish_ns=receipt_at,handler_end_ns=clock,start_ns=state['start_ns'],
                            worker_next_ready_ns=ready_at,queue_wait_ns=queue_wait[row['request_id']])
                receipt_pending[row['request_id']]=record;sequence+=1
                heapq.heappush(finish_events,(receipt_at,sequence,row['request_id']))
                unready[owner]=ready_at;heapq.heappush(wake_events,(ready_at,owner));del running[owner]

        while arrivals or any(pending.values()) or running or ready or active or finish_events or wake_events or clock<deadline_ns:
            if drain_deadline_ns is not None and clock>=drain_deadline_ns:break
            event_count+=1
            if event_count>self.max_events:raise UnknownEvidence('DAG_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION')
            while arrivals and arrivals[0][0]<=clock:
                _,_,row=heapq.heappop(arrivals);pending[row['alias']].append(row)
                if self.queue_diagnostics is not None:self.queue_diagnostics.register(row)
            while finish_events and finish_events[0][0]<=clock:
                _,_,rid=heapq.heappop(finish_events);finishes[rid]=receipt_pending.pop(rid)
            while wake_events and wake_events[0][0]<=clock:
                at,owner=heapq.heappop(wake_events)
                if unready.get(owner)==at:del unready[owner]
            start_requests()
            while ready and ready[0][0]<=clock:
                _,_,owner,node_id=heapq.heappop(ready);state=running[owner];node=state['nodes'][node_id]
                kind=node['kind']
                if kind in ('event_record','event_wait','host_sync','handler_end'):
                    complete_node(owner,node_id);continue
                if kind not in ('gpu','copy','memset'):raise ValueError('Unsupported CUDA DAG activity kind')
                if kind in ('copy','memset') and not self.allow_independent_copy_hypothesis:
                    raise UnknownEvidence('DAG_COPY_RESOURCE_SERVICE_MODEL_REQUIRED')
                if self.duration_provider is None:
                    if not self.allow_profiled_replay:raise UnknownEvidence('FITTED_DAG_ACTIVITY_DURATION_PROVIDER_REQUIRED')
                    duration=node['duration_ns']
                else:duration=self.duration_provider(node,state['row'])
                if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:
                    raise UnknownEvidence('UNKNOWN_DAG_NODE_DURATION')
                active[owner,node_id]=dict(kind=kind,remaining_ns=float(duration))
            finish_requests();start_requests()
            maximum_parallel=max(maximum_parallel,len(active))
            counts=collections.Counter(owner for owner,_ in active)
            for owner,count in counts.items():maximum_owner_parallel[owner]=max(maximum_owner_parallel[owner],count)
            max_waiting=max(max_waiting,sum(len(q) for q in pending.values()))
            if (clock>=deadline_ns and not arrivals and not any(pending.values())
                    and not running and not ready and not active and not finish_events and not wake_events):
                if self.queue_diagnostics is not None:
                    self.queue_diagnostics.observe(clock,0,pending,running,active,{})
                break
            gpu_owners={owner for (owner,_),node in active.items() if node['kind']=='gpu'}
            if len(gpu_owners)>1 and not self.allow_context_time_slice_hypothesis:
                raise UnknownEvidence('CROSS_CONTEXT_DAG_SCHEDULER_MODEL_REQUIRED')
            rates={key:(1./len(gpu_owners) if node['kind']=='gpu' else 1.) for key,node in active.items()}
            # Round relative service intervals before absolute-clock addition.
            # Completion uses this same interval, avoiding residual epsilons.
            finish_ticks={}
            for key,node in active.items():
                interval=node['remaining_ns']/rates[key]
                if not math.isfinite(interval) or interval<=0:
                    raise UnknownEvidence('INVALID_DAG_REMAINING_SERVICE_INTERVAL')
                finish_ticks[key]=max(1,math.ceil(interval))
            next_finish=clock+min(finish_ticks.values()) if finish_ticks else math.inf
            boundary=min(next_finish,arrivals[0][0] if arrivals else math.inf,
                ready[0][0] if ready else math.inf,finish_events[0][0] if finish_events else math.inf,
                wake_events[0][0] if wake_events else math.inf,
                deadline_ns if clock<deadline_ns else math.inf,
                drain_deadline_ns if drain_deadline_ns is not None else math.inf)
            if not math.isfinite(boundary):
                if running:raise UnknownEvidence('DAG_DEPENDENCY_DEADLOCK')
                break
            if type(boundary) is not int:raise ValueError('Integer CUDA DAG event boundary required')
            delta=boundary-clock
            if delta<0:raise ValueError('CUDA DAG event chronology inversion')
            if self.queue_diagnostics is not None:
                self.queue_diagnostics.observe(clock,delta,pending,running,active,rates)
            if delta>0:
                # The training resource state is handler-active ownership, not
                # the subset of owners whose CUDA nodes happen to be ready.
                handlers={owner:state['row'] for owner,state in running.items()}
                watts=None;memory_mib=None
                if clock<deadline_ns:
                    value=self.power_provider.power(handlers,self.aliases,self.scenario) if self.power_provider else {}
                    watts=value.get('watts')
                    if value.get('known') is not True or type(watts) not in (int,float) or not math.isfinite(watts) or watts<0:
                        energy_known=False;watts=None
                        if self.require_energy:raise UnknownEvidence('UNKNOWN_DAG_WHOLE_BOARD_POWER_CONTEXT')
                    else:energy+=watts*min(delta,deadline_ns-clock)/1e9
                value=self.memory_provider.memory(handlers,self.aliases,self.scenario) if self.memory_provider else {}
                memory_mib=value.get('peak_memory_mib')
                if value.get('known') is not True or type(memory_mib) not in (int,float) or not math.isfinite(memory_mib) or memory_mib<0:
                    memory_known=False;memory_mib=None
                    if self.require_memory:raise UnknownEvidence('UNKNOWN_DAG_WHOLE_BOARD_PEAK_MEMORY_CONTEXT')
                else:peak=max(peak,memory_mib)
                resource_transitions.append(dict(time_ns=clock,until_ns=boundary,
                    handler_active_aliases=sorted(handlers),board_watts=watts,peak_memory_mib=memory_mib))
            completed=[]
            for key,node in active.items():
                node['remaining_ns']-=delta*rates[key]
                if finish_ticks[key]<=delta:completed.append(key)
            clock=boundary
            for owner,node_id in completed:
                del active[owner,node_id];complete_node(owner,node_id)
            if pause_after_event_count is not None and event_count>=pause_after_event_count:
                # No artificial time boundary: pausing cannot change ceil quantization.
                local_state=locals()
                state={field:local_state[field] for field in STATE_FIELDS}
                return dict(status='PAUSED_SIMULATED_ATOMIC_BOUNDARY',
                    checkpoint=export_checkpoint(state,self.queue_diagnostics,checkpoint_contract),
                    physical_progress_observed=False,visible_state_correction_applied=False)
        return dict(cohort=cohort,finishes=finishes,board_energy_joules=energy if energy_known else None,
            peak_memory_mib=peak if memory_known else None,energy_known=energy_known,memory_known=memory_known,
            resource_transitions=resource_transitions,energy_window_ns=[now_ns,deadline_ns],
            memory_window_ns=[now_ns,clock],memory_safe_factor_applied=False,
            starts=starts,queue_wait_ns=queue_wait,events=event_count,completed_dag_nodes=all_completed_nodes,
            max_simultaneous_activity_nodes=maximum_parallel,max_simultaneous_nodes_per_owner=dict(maximum_owner_parallel),
            queue_diagnostics=(dict(max_waiting_requests=max_waiting) if self.queue_diagnostics is None
                else dict(self.queue_diagnostics.result(cohort),max_waiting_requests=max_waiting)),
            event_time_resolution_ns=1,
            activity_completion_quantization='ceil_positive_relative_interval_to_integer_ns',
            assumptions=['Equal context time slicing; intra-owner ready kernels may progress concurrently',
                         'Independent copy/memset engine service; DRAM interference unmodeled',
                         'No GPU same-SM concurrency guarantee'],
            duration_source='injected_fitted_provider' if self.duration_provider else 'profiled_replay_diagnostic',
            release_source='injected_host_release_provider' if self.release_provider else 'profiled_api_issue_replay_diagnostic',
            unseen_input_supported=False,physical_accuracy_validated=False,deployment_qualified=False)
