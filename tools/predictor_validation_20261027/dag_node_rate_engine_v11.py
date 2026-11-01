"""v11 node-rate DAG consumer, explicitly derived from the retained v10 core.

Numerical event processing follows the frozen runtime v8 same-clock closure.
Only the scheduler/rate protocol, explicit non-GPU rates and work accounting change.
The original v10 file is not modified; this is an unvalidated model consumer.
"""
from __future__ import annotations
import collections
import copy
import heapq
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView
from .dag_node_rate_scheduler_v11 import CheckpointNodeScheduler,validate_decision
from .dag_frozen_model_v10 import FrozenDagModel
from .dag_checkpoint_v9 import request_start_bound
from .dag_node_rate_checkpoint_v11 import STATE_FIELDS,contract,export_checkpoint,import_checkpoint

class DagNodeRateCheckpointModel:
    def __init__(self,aliases,plans,*,allow_profiled_replay=False,
                 allow_independent_copy_hypothesis=False,allow_context_time_slice_hypothesis=False,
                 allow_no_implicit_default_stream_hypothesis=False,
                 lifecycle_provider=None,scenario='mean',duration_provider=None,release_provider=None,
                 power_provider=None,memory_provider=None,require_energy=False,require_memory=False,
                 max_events=500000,queue_diagnostics=None,scheduler=None,phase_provider=None,
                 checkpoint_binding=None,request_start_provider=None,frozen_model=None,non_gpu_rate_provider=None):
        if not all(type(v) is bool for v in (allow_profiled_replay,allow_independent_copy_hypothesis,
                                             allow_context_time_slice_hypothesis,allow_no_implicit_default_stream_hypothesis)):
            raise ValueError('Explicit boolean DAG model hypothesis flags required')
        self._frozen_model=FrozenDagModel(aliases,plans) if frozen_model is None else frozen_model
        if type(self._frozen_model) is not FrozenDagModel:
            raise ValueError("Explicit immutable original DAG model required")
        if frozen_model is not None and (aliases is not frozen_model.aliases or plans is not frozen_model.plans):
            raise ValueError('Shared frozen model must retain its original aliases and plans')
        self.aliases=self._frozen_model.aliases;self.plans=self._frozen_model.plans;self.allow_profiled_replay=allow_profiled_replay
        self.allow_independent_copy_hypothesis=allow_independent_copy_hypothesis
        self.allow_context_time_slice_hypothesis=allow_context_time_slice_hypothesis
        self.allow_no_implicit_default_stream_hypothesis=allow_no_implicit_default_stream_hypothesis
        self.lifecycle_provider=lifecycle_provider;self.scenario=scenario
        self.duration_provider=duration_provider;self.release_provider=release_provider;self.max_events=max_events
        self.power_provider=power_provider;self.memory_provider=memory_provider
        self.require_energy=require_energy;self.require_memory=require_memory
        self.queue_diagnostics=queue_diagnostics
        self.scheduler=CheckpointNodeScheduler.fresh_from(scheduler)
        self.non_gpu_rate_provider=non_gpu_rate_provider
        if release_provider is None or request_start_provider is None or lifecycle_provider is None:
            raise UnknownEvidence('EXPLICIT_HOST_START_RELEASE_AND_LIFECYCLE_LAWS_REQUIRED')
        self.phase_provider=phase_provider
        self.checkpoint_binding=copy.deepcopy(checkpoint_binding)
        self.request_start_provider=request_start_provider

    def forecast(self,requests,now_ns,deadline_ns,*,drain_deadline_ns=None,
                 checkpoint=None,pause_after_event_count=None,pause_at_or_after_ns=None):
        if type(now_ns) is not int or type(deadline_ns) is not int or now_ns>deadline_ns:
            raise ValueError('Integer measurement window required')
        if pause_after_event_count is not None and (type(pause_after_event_count) is not int or pause_after_event_count<=0):
            raise ValueError('Positive absolute event count required for atomic pause')
        if pause_at_or_after_ns is not None and (type(pause_at_or_after_ns) is not int or pause_at_or_after_ns<=now_ns):
            raise ValueError('Atomic pause time must follow original phase start')
        checkpoint_contract=contract(self,requests,now_ns,deadline_ns,drain_deadline_ns)
        if checkpoint is None and (self.scheduler.last_clock_ns is not None or self.scheduler.decision_count):
            raise ValueError('New forecast requires an empty scheduler; use an explicit checkpoint to continue')
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
        resource_transitions=[];node_work={}
        owner_last_ready={owner:now_ns for owner in self.aliases}
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
            node_work=restored['node_work']
            if pause_after_event_count is not None and pause_after_event_count<=event_count:
                raise ValueError('Continuation pause must follow consumed absolute event count')
            if pause_at_or_after_ns is not None and pause_at_or_after_ns<=clock:
                raise ValueError('Continuation atomic pause time must follow checkpoint clock')
            for field in ('finishes','receipt_pending'):
                for row in restored[field].values():
                    owner=row['alias'];owner_last_ready[owner]=max(owner_last_ready[owner],row['worker_next_ready_ns'])

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
                node_work[state['row']['request_id'],node_id]=dict(reference_ns=float(duration),consumed_ns=0.,remaining_ns=float(duration))

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
                    self.queue_diagnostics.observe(clock,0,pending,running,active,{})
                break
            if (clock>=deadline_ns and not arrivals and not any(pending.values())
                    and not running and not ready and not active and not finish_events and not wake_events):
                if self.queue_diagnostics is not None:
                    self.queue_diagnostics.observe(clock,0,pending,running,active,{})
                break
            phase_nodes=[]
            for key,item in active.items():
                if item['kind']!='gpu':continue
                owner,node_id=key;state=running[owner];row=state['row'];node=state['nodes'][node_id]
                phase=(self.phase_provider(node,row) if self.phase_provider is not None else 'UNBOUND_PHASE')
                if not isinstance(phase,str) or not phase:raise UnknownEvidence('EXACT_DAG_SEMANTIC_PHASE_REQUIRED')
                phase_nodes.append(ActiveGpuNode(key,item['remaining_ns'],phase,
                    row['request_id'],row['task'],row['pool_id']))
            view=SchedulingView(clock,tuple(phase_nodes))
            decision=self.scheduler.decide(view)
            gpu_rates,next_transition=validate_decision(view,decision)
            rates=dict(gpu_rates)
            for key,item in active.items():
                if item['kind']=='gpu':continue
                if self.non_gpu_rate_provider is None:
                    raise UnknownEvidence('EXPLICIT_COPY_MEMSET_REFERENCE_RATE_LAW_REQUIRED')
                owner,node_id=key;state=running[owner]
                law=self.non_gpu_rate_provider(state['nodes'][node_id],state['row'],clock,tuple(active))
                if type(law) is not dict or set(law)!={'rate','next_transition_ns'}:
                    raise UnknownEvidence('EXPLICIT_COPY_RATE_AND_TRANSITION_REQUIRED')
                rate,transition=law['rate'],law['next_transition_ns']
                if type(rate) not in (int,float) or not math.isfinite(rate) or rate<0:
                    raise UnknownEvidence('FINITE_NONNEGATIVE_COPY_REFERENCE_RATE_REQUIRED')
                if transition is not None and (type(transition) is not int or transition<=clock):
                    raise UnknownEvidence('STRICTLY_FUTURE_COPY_TRANSITION_REQUIRED')
                rates[key]=rate
                if transition is not None:next_transition=min(next_transition,transition) if next_transition is not None else transition
            next_request_start=min((request_start_bound(self,queue[0],owner_ready_ns=owner_last_ready[owner])
                for owner,queue in pending.items() if queue and owner not in running and owner not in unready
                and request_start_bound(self,queue[0],owner_ready_ns=owner_last_ready[owner])>clock),default=math.inf)
            public_or_generated_transition=min(
                next_request_start,
                arrivals[0][0] if arrivals else math.inf,
                ready[0][0] if ready else math.inf,
                finish_events[0][0] if finish_events else math.inf,
                wake_events[0][0] if wake_events else math.inf,
                next_transition if next_transition is not None else math.inf)
            if rates and all(rate==0 for rate in rates.values()) and not math.isfinite(public_or_generated_transition):
                raise UnknownEvidence('NODE_RATE_ZERO_PROGRESS_WITHOUT_FUTURE_TRANSITION')
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
                work=node_work[running[key[0]]['row']['request_id'],key[1]]
                consumed=min(node['remaining_ns'],delta*rates[key])
                node['remaining_ns']=max(0.,node['remaining_ns']-consumed)
                work['remaining_ns']=node['remaining_ns']
                work['consumed_ns']=work['reference_ns']-work['remaining_ns']
                if key in finish_ticks and finish_ticks[key]<=delta:completed.append(key)
            clock=boundary
            for owner,node_id in completed:
                del active[owner,node_id];complete_node(owner,node_id)
            if ((pause_after_event_count is not None and event_count>=pause_after_event_count)
                    or (pause_at_or_after_ns is not None and clock>=pause_at_or_after_ns)):
                local_state=locals();state={field:local_state[field] for field in STATE_FIELDS}
                return dict(status='PAUSED_SIMULATED_ATOMIC_BOUNDARY',
                    checkpoint=export_checkpoint(state,self,checkpoint_contract),
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
            scheduler_provenance=self.scheduler.provenance(),
            scheduler_checkpoint_supported=True,
            reference_work_accounting=node_work,
            node_rates_may_exceed_one=True,target_future_END_read=False,
            native_sources_authenticated=False,model_fit=False,GPU_started=False,
            assumptions=['Injected per-node reference-work rates; no SM utilization or physical service identification',
                         'Explicit supplied copy/memset reference-clock law; host START/release/lifecycle laws required',
                         'No GPU same-SM concurrency guarantee'],
            duration_source='injected_fitted_provider' if self.duration_provider else 'profiled_replay_diagnostic',
            release_source='injected_host_release_provider' if self.release_provider else 'profiled_api_issue_replay_diagnostic',
            unseen_input_supported=False,physical_accuracy_validated=False,deployment_qualified=False)
