"""Isolated exact-cutoff simulated lifecycle continuation, never native assimilation.

Numerical DAG closures below are copied from immutable v21; no runtime patching,
future result replay, new admission cap, or changed utility/stability semantics.
Only exact supported v49/v24 schedulers and declared stateless providers resume.
"""
from __future__ import annotations
import collections
import copy
import heapq
import math
from .implementation_source_closure_v27 import implementation_sources
from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import ActiveGpuNode, validate_decision
from .dag_checkpoint_v9 import request_start_bound
from .owner_lifecycle_mark_v21 import engine_states, receipt
from .owner_lifecycle_engine_v21 import LifecycleDagPhaseModel as V21Model, LifecycleSchedulingView
from .lifecycle_engine_checkpoint_v59 import (
    STATE_FIELDS,checkpoint_contract,decode_checkpoint,commit_checkpoint,pause_result,
    diagnostic_state,install_diagnostic_state,integrate_resources)

class ResumableLifecycleDagPhaseModel(V21Model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._source_receipts=(*self._source_receipts,receipt(__file__))

    def forecast(self,*args,**kwargs):
        if type(self) is not ResumableLifecycleDagPhaseModel:
            raise ValueError('Unsupported lifecycle engine subclass state')
        sources=implementation_sources(__file__)
        try:
            return super().forecast(*args,**kwargs)
        finally:
            if any(receipt(ref['path'])!=ref for ref in sources):
                raise UnknownEvidence('LIFECYCLE_RESUME_IMPLEMENTATION_SOURCE_BYTES_CHANGED')

    def _forecast(self,requests,now_ns,deadline_ns,*,drain_deadline_ns=None,
                 checkpoint=None,pause_after_event_count=None,pause_at_or_after_ns=None,
                 pause_at_ns=None):
        if type(now_ns) is not int or type(deadline_ns) is not int or now_ns>deadline_ns:
            raise ValueError('Integer measurement window required')
        if pause_after_event_count is not None or pause_at_or_after_ns is not None:
            raise ValueError('v59 requires an exact pause_at_ns, not a later event cutoff')
        if pause_at_ns is not None and (type(pause_at_ns) is not int or not now_ns<=pause_at_ns<deadline_ns):
            raise ValueError('Exact integer pause must lie inside the original measurement window')
        if self._forecast_invoked:
            raise ValueError('A fresh lifecycle engine is required for each forecast or resume')
        contract=checkpoint_contract(self,requests,now_ns,deadline_ns,drain_deadline_ns)
        restored=None
        if checkpoint is not None:
            restored=decode_checkpoint(self,checkpoint,contract)
            if pause_at_ns is not None and pause_at_ns<=restored['clock']:
                raise ValueError('Next exact cutoff must follow restored clock')
        # No destination state changes occur before all import validation passes.
        if restored is not None:
            commit_checkpoint(self,restored)
        self._forecast_invoked=True
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
        owner_last_ready={owner:now_ns for owner in self.aliases}
        advance=None
        node_completion_ns={}
        if restored is not None:
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
            owner_last_ready=restored['owner_last_ready']
            advance=restored['advance']
            node_completion_ns=restored['node_completion_ns']
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
            node_completion_ns[owner,state['row']['request_id'],node_id]=clock
            for successor in state['successors'][node_id]:
                state['incoming'][successor]-=1
                if state['incoming'][successor]==0:push_ready(owner,state['nodes'][successor])

        def start_requests():
            for owner in self.aliases:
                if owner in running or owner in unready or not pending[owner]:continue
                head=pending[owner][0]
                if request_start_bound(self,head,owner_ready_ns=owner_last_ready[owner])>clock:continue
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
                owner_last_ready[owner]=ready_at

        def release_ready_nodes():
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

        def settle_current_clock():
            nonlocal event_count
            # Parent receipt, worker-ready, dispatch and zero-work DAG nodes
            # form a same-clock closure. The scheduler sees its final frontier,
            # never a transient owner disappearance between consecutive jobs.
            while True:
                while finish_events and finish_events[0][0]<=clock:
                    _,_,rid=heapq.heappop(finish_events);finishes[rid]=receipt_pending.pop(rid)
                while wake_events and wake_events[0][0]<=clock:
                    at,owner=heapq.heappop(wake_events)
                    if unready.get(owner)==at:del unready[owner]
                start_requests();release_ready_nodes();finish_requests()
                due=(finish_events and finish_events[0][0]<=clock
                    or wake_events and wake_events[0][0]<=clock
                    or ready and ready[0][0]<=clock)
                dispatchable=any(queue and owner not in running and owner not in unready
                    and request_start_bound(self,queue[0],owner_ready_ns=owner_last_ready[owner])<=clock
                    for owner,queue in pending.items())
                if not due and not dispatchable:break
                # Closing zero-time work is real simulator work too. Retain
                # the same fail-closed budget as repeated outer event loops.
                event_count+=1
                if event_count>self.max_events:
                    raise UnknownEvidence('DAG_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION')

        while arrivals or any(pending.values()) or running or ready or active or finish_events or wake_events or clock<deadline_ns:
            if advance is None:
                event_count+=1
                if event_count>self.max_events:raise UnknownEvidence('DAG_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION')
                while arrivals and arrivals[0][0]<=clock:
                    _,_,row=heapq.heappop(arrivals);pending[row['alias']].append(row)
                    if self.queue_diagnostics is not None:self.queue_diagnostics.register(row)
                settle_current_clock()
                maximum_parallel=max(maximum_parallel,len(active))
                counts=collections.Counter(owner for owner,_ in active)
                for owner,count in counts.items():maximum_owner_parallel[owner]=max(maximum_owner_parallel[owner],count)
                max_waiting=max(max_waiting,sum(len(q) for q in pending.values()))
                # Complete/receive all work exactly at the inclusive drain cutoff,
                # then stop before scheduling or advancing beyond that clock.
                if drain_deadline_ns is not None and clock>=drain_deadline_ns:
                    if self.queue_diagnostics is not None:
                        self.queue_diagnostics.observe(clock,0,pending,running,active,{key:0. for key in active})
                    break
                if (clock>=deadline_ns and not arrivals and not any(pending.values())
                        and not running and not ready and not active and not finish_events and not wake_events):
                    if self.queue_diagnostics is not None:
                        self.queue_diagnostics.observe(clock,0,pending,running,active,{key:0. for key in active})
                    break
                phase_nodes=[]
                for key,item in active.items():
                    if item['kind']!='gpu':continue
                    owner,node_id=key;state=running[owner];row=state['row'];node=state['nodes'][node_id]
                    phase=(self.phase_provider(node,row) if self.phase_provider is not None else 'UNBOUND_PHASE')
                    if not isinstance(phase,str) or not phase:raise UnknownEvidence('EXACT_DAG_SEMANTIC_PHASE_REQUIRED')
                    phase_nodes.append(ActiveGpuNode(key,item['remaining_ns'],phase,
                        row['request_id'],row['task'],row['pool_id']))
                lifecycle=engine_states(self.aliases,running,active)
                view=LifecycleSchedulingView(clock,tuple(phase_nodes),
                    tuple(sorted(lifecycle.items())),
                    tuple((owner,state['row']['request_id'],state['start_ns']) for owner,state in sorted(running.items())),
                    tuple((owner,tuple(row['request_id'] for row in queue)) for owner,queue in sorted(pending.items())))
                decision=self.scheduler.decide(view)
                gpu_rates,next_transition=validate_decision(view,decision)
                rates={key:(gpu_rates[key] if item['kind']=='gpu' else 1.) for key,item in active.items()}
                # Round relative service intervals before absolute-clock addition.
                # Completion uses this same interval, avoiding residual epsilons.
                finish_ticks={}
                for key,node in active.items():
                    if rates[key]==0:continue
                    interval=node['remaining_ns']/rates[key]
                    if not math.isfinite(interval) or interval<=0:
                        raise UnknownEvidence('INVALID_DAG_REMAINING_SERVICE_INTERVAL')
                    finish_ticks[key]=max(1,math.ceil(interval))
                next_finish=clock+min(finish_ticks.values()) if finish_ticks else math.inf
                next_request_start=min((request_start_bound(self,queue[0],owner_ready_ns=owner_last_ready[owner])
                    for owner,queue in pending.items() if queue and owner not in running and owner not in unready
                    and request_start_bound(self,queue[0],owner_ready_ns=owner_last_ready[owner])>clock),default=math.inf)
                boundary=min(next_finish,next_request_start,next_transition if next_transition is not None else math.inf,
                    arrivals[0][0] if arrivals else math.inf,
                    ready[0][0] if ready else math.inf,finish_events[0][0] if finish_events else math.inf,
                    wake_events[0][0] if wake_events else math.inf,
                    deadline_ns if clock<deadline_ns else math.inf,
                    drain_deadline_ns if drain_deadline_ns is not None else math.inf)
                if not math.isfinite(boundary):
                    if running:raise UnknownEvidence('PHASE_SCHEDULER_NO_PROGRESS_OR_DAG_DEPENDENCY_DEADLOCK')
                    break
                if type(boundary) is not int:raise ValueError('Integer CUDA DAG event boundary required')
                advance=dict(origin_ns=clock,boundary_ns=boundary,rates=copy.deepcopy(rates),
                    finish_ticks=finish_ticks,origin_active=copy.deepcopy(active),
                    origin_energy=energy,origin_peak=peak,origin_energy_known=energy_known,
                    origin_memory_known=memory_known,origin_resource_count=len(resource_transitions),
                    diagnostics_origin=diagnostic_state(self.queue_diagnostics),resource=None)
            else:
                boundary=advance['boundary_ns'];rates=advance['rates'];finish_ticks=advance['finish_ticks']
            if pause_at_ns==clock:
                snapshot={key:locals_state for key,locals_state in locals().items() if key in STATE_FIELDS}
                return pause_result(self,snapshot,contract)
            target=min(boundary,pause_at_ns) if pause_at_ns is not None and pause_at_ns>clock else boundary
            if target<clock:raise ValueError('CUDA DAG event chronology inversion')
            origin=advance['origin_ns'];total_delta=target-origin
            # Recompute only the unfinished constant-rate interval from its own
            # origin. This is integration, never replay of arrivals or RESULTs.
            if self.queue_diagnostics is not None:
                install_diagnostic_state(self.queue_diagnostics,advance['diagnostics_origin'])
                self.queue_diagnostics.observe(origin,total_delta,pending,running,advance['origin_active'],rates)
            energy,peak,energy_known,memory_known=integrate_resources(self,advance,
                resource_transitions,running,target,deadline_ns)
            completed=[]
            for key,node in active.items():
                node['remaining_ns']=advance['origin_active'][key]['remaining_ns']-total_delta*rates[key]
                if key in finish_ticks and finish_ticks[key]<=total_delta:completed.append(key)
            clock=target
            if target<boundary:
                snapshot={key:locals_state for key,locals_state in locals().items() if key in STATE_FIELDS}
                return pause_result(self,snapshot,contract)
            for owner,node_id in completed:
                del active[owner,node_id];complete_node(owner,node_id)
            advance=None
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
            scheduler_provenance=self.scheduler.provenance(),
            scheduler_checkpoint_supported=True,owner_lifecycle_snapshot_supported=True,
            issued_dependency_queue_complete=False,
            assumptions=['Injected per-node reference-work rates; no SM utilization or physical service identification',
                         'Independent copy/memset engine service; DRAM interference unmodeled',
                         'No GPU same-SM concurrency guarantee'],
            duration_source='injected_fitted_provider' if self.duration_provider else 'profiled_replay_diagnostic',
            release_source='injected_host_release_provider' if self.release_provider else 'profiled_api_issue_replay_diagnostic',
            unseen_input_supported=False,physical_accuracy_validated=False,deployment_qualified=False)
