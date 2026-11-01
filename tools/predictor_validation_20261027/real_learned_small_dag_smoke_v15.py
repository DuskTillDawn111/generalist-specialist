"""Actual learned components plus explicitly synthetic tiny DAG mechanics.

No original work/host provider initialization, validation labels, GPU execution,
protocol seal, resource estimates, utility, or physical interference claims.
"""
from pathlib import Path
import argparse
import copy
import json
import time

from .context_initial_prior_v15 import JointInitialContextPrior
from .context_next_owner_v14 import NextOwnerCandidate
from .cumulative_save_hazard_v14 import CumulativeSaveHazardController
from .dag_phase_scheduler_v8 import DagPhaseFullHorizonModel
from .handoff_candidate_v14 import HandoffCandidate
from .learned_context_oracle_v15 import LearnedContextOracle,LearnedContextScheduler
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .profiling import digest

MODEL_PATHS=dict(initial='artifacts/predictor-context-initial-prior-v15-20261028-training01/model.json',
    next_owner='artifacts/predictor-context-next-owner-v14-20261028-training02/model.json',
    handoff='artifacts/predictor-handoff-candidate-v14-20261028-training02/model.json',
    save='artifacts/predictor-save-hazard-v13-20261028-training01/age_additive-model.json')
POOL='tydiqa-new-native-e0ff7b578f9c742a6697469a'
START=1_000_000_000


class RecordingScheduler(LearnedContextScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.calls=[]

    def decide(self,view):
        record=dict(clock_ns=view.clock_ns,nodes=[dict(key=list(n.key),request_id=n.request_id,
            remaining_reference_work_ns=n.remaining_reference_work_ns,phase=n.phase) for n in view.active_gpu_nodes])
        self.calls.append(record);began=time.perf_counter()
        try:
            decision=super().decide(view)
            record.update(gpu_rates=[dict(key=list(k),rate=v) for k,v in decision.gpu_rates.items()],
                          next_transition_ns=decision.next_transition_ns)
            return decision
        except Exception as exc:
            record['error']=dict(type=type(exc).__name__,reason=str(exc));raise
        finally:
            record.update(held_owner=self.held_owner,restore_start_ns=self.restore_start_ns,
                actual_attained_age_ns=None if self.restore_start_ns is None else view.clock_ns-self.restore_start_ns,
                service_available=self.service_available,controller_result=copy.deepcopy(self.oracle.hazard),
                call_seconds=time.perf_counter()-began)


def cases():
    return [dict(name='single_request_short_GPU',requests=[('text#0',0)],gpu_ns=[100_000],host_gap_ns=0),
            dict(name='two_requests_switching',requests=[('text#0',0),('text#1',0)],gpu_ns=[600_000],host_gap_ns=0),
            dict(name='two_requests_host_gap',requests=[('text#0',0),('text#1',50_000)],gpu_ns=[20_000,300_000],host_gap_ns=100_000),
            dict(name='supported_entry_single_request',requests=[('text#1',0)],gpu_ns=[100_000],host_gap_ns=0),
            dict(name='supported_entry_two_requests_switching',requests=[('text#1',0),('text#0',50_000)],gpu_ns=[600_000],host_gap_ns=0),
            dict(name='supported_entry_two_requests_host_gap',requests=[('text#1',0),('text#0',50_000)],gpu_ns=[20_000,300_000],host_gap_ns=100_000),
            dict(name='actual_gap_scenario_branch',requests=[('text#0',0)],gpu_ns=[100_000],host_gap_ns=0,initial_quantile=.375),
            dict(name='actual_gap_bounds_only_rejects_point',requests=[('text#0',0)],gpu_ns=[100_000],host_gap_ns=0,initial_quantile=.375,initial_readiness_rule='bounds_only')]


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    refs={k:receipt(p) for k,p in MODEL_PATHS.items()}
    paths=[__file__,Path(__file__).with_name('learned_context_oracle_v15.py'),
           Path(__file__).with_name('learned_context_oracle_v14.py'),
           Path(__file__).with_name('learned_availability_forecast_v14.py'),
           Path(__file__).with_name('cumulative_save_hazard_v14.py'),
           Path(__file__).with_name('held_context_scheduler_v15.py'),
           Path(__file__).with_name('held_context_scheduler_v14.py'),
           Path(__file__).with_name('dag_phase_scheduler_v8.py')]
    sources=[receipt(p) for p in paths]
    save(out/'declaration.json',dict(cases=cases(),model_sources=refs,implementation_sources=sources,
        duration_and_release='Explicit synthetic DAG only; actual trained scheduling components',
        all_original_four_residents_preserved=True,active_request_owners=['text#0','text#1'],
        quantile=.5,readiness='lower bound reference availability hypothesis',
        empty_frontier='NO_QUERY_PAUSE',training_only=True,GPU_started=False))
    models={k:checked(r) for k,r in refs.items()}
    initial=JointInitialContextPrior(models['initial']);mark=NextOwnerCandidate(models['next_owner'])
    handoff=HandoffCandidate(models['handoff'])
    controller=CumulativeSaveHazardController(refs['save'],allow_unvalidated_approximation=True,empty_frontier_policy='NO_QUERY_PAUSE')
    initialization_seconds=time.perf_counter()-began
    setup=copy.deepcopy(models['initial']['scenarios'][0]['public_setup'])
    aliases={alias:dict(family=family,supported_tasks=['ocr'] if family=='ocr' else ['text'])
             for alias,family in setup['residents'].items()}
    identity=digest(dict(models=refs,solver=controller.provenance(),implementations=sources))
    reports=[]
    for case in cases():
        rows=[dict(request_id=case['name']+'-'+str(i),alias=owner,task='text',pool_id=POOL,arrival_ns=START+offset)
              for i,(owner,offset) in enumerate(case['requests'])]
        workload=[[r['alias'],r['task'],r['pool_id'],r['arrival_ns']-START] for r in rows]
        oracle=LearnedContextOracle(initial_prior=initial,save_controller=controller,next_owner=mark,
            handoff=handoff,public_setup=setup,public_workload=workload,source_identity=identity,
            initial_quantile=case.get('initial_quantile',.5),initial_readiness_rule=case.get('initial_readiness_rule','lower_bound_hypothesis'),
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
        scheduler=RecordingScheduler(aliases,oracle=oracle)
        plans={}
        for owner,_ in case['requests']:
            nodes=[dict(id='gpu'+str(i),kind='gpu',duration_ns=work,release_ns=0) for i,work in enumerate(case['gpu_ns'])]
            edges=[]
            if case['host_gap_ns']:
                edges=[dict(before='gpu0',after='gpu1')]
            plans[owner,'text',POOL]=dict(nodes=nodes,edges=edges)
        engine=DagPhaseFullHorizonModel(aliases,plans,scheduler=scheduler,allow_profiled_replay=True,
            phase_provider=lambda n,r:'outside_language_forwards',allow_independent_copy_hypothesis=True,
            release_provider=lambda n,r:dict(not_before_offset_ns=0,
                after_predecessors_ns=case['host_gap_ns'] if n['id']=='gpu1' else 0),
            allow_no_implicit_default_stream_hypothesis=True,max_events=1000)
        start=time.perf_counter();result=None;error=None
        try:result=engine.forecast(rows,START,START+10_000_000,drain_deadline_ns=START+20_000_000)
        except Exception as exc:error=dict(type=type(exc).__name__,reason=str(exc))
        elapsed=time.perf_counter()-start
        invariants=[]
        for record in scheduler.calls:
            restore=record['restore_start_ns'];clock=record['clock_ns']
            if restore is not None:
                invariants.append(record['actual_attained_age_ns']==clock-restore and restore<=clock)
            for rate in record.get('gpu_rates',[]):
                if rate['rate']>0:
                    invariants.append(record['service_available'] and rate['key'][0]==record['held_owner'])
            h=record['controller_result']
            if h is not None:
                invariants.append(h['state']['origin_restore_ns']==h['state']['restore_ns'])
                invariants.append(h['state']['observation_ns']>=h['state']['restore_ns'])
                invariants.append(h['preentry_hazard_integrated'] is False)
        transitions=scheduler.transitions
        service_checks=[]
        for i,event in enumerate(transitions):
            if event['kind']=='SERVICE_READY':
                preceding=[e for e in transitions[:i] if e['kind']=='RESTORE_START' and e['owner']==event['owner']]
                service_checks.append(bool(preceding) and preceding[-1]['clock_ns']<=event['clock_ns'])
        detail=dict(case=case,result=result,error=error,trace=scheduler.calls,scheduler=scheduler.provenance(),
                    oracle=oracle.provenance(),elapsed_seconds=elapsed,invariant_checks=invariants,
                    service_event_checks=service_checks)
        save(out/(case['name']+'.json'),detail)
        reports.append(dict(case=case['name'],status='CPU_MECHANICAL_SMOKE_COMPLETED' if error is None else 'FIRST_UNSUPPORTED_OR_ERROR',
            first_unsupported=error,detail=receipt(out/(case['name']+'.json')),elapsed_seconds=elapsed,
            scheduler_calls=len(scheduler.calls),scheduler_call_seconds=sum(r['call_seconds'] for r in scheduler.calls),
            modelled_switches=len(oracle.decisions),completed_requests=len(result['finishes']) if result else None,
            all_age_service_invariants_pass=all(invariants) and all(service_checks) if invariants else None,
            age_service_invariant_count=len(invariants)+len(service_checks),
            initial_decision=scheduler.initial_decision,
            maximum_restore_age_ns=max((r['actual_attained_age_ns'] or 0 for r in scheduler.calls),default=0)))
    controller.assert_sources_unchanged()
    changed=[r for r in [*sources,*refs.values()] if receipt(r['path'])!=r]
    report=dict(status='REAL_LEARNED_COMPONENTS_SMALL_SYNTHETIC_DAG_SMOKE_COMPLETE',
        model_sources=refs,implementation_sources=sources,changed_sources=changed,source_closure_unchanged=not changed,
        initialization_seconds=initialization_seconds,cases=reports,elapsed_seconds=time.perf_counter()-began,
        original_base_work_and_host_providers_initialized=False,real_trained_components_used=True,
        synthetic_work_and_host_DAG=True,validation_labels_used=False,independent_prediction_validation=False,
        physical_interference_validated=False,full_U_computed=False,protocol_sealed=False,GPU_started=False)
    save(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.output);print(json.dumps({k:r[k] for k in ('status','initialization_seconds','cases','elapsed_seconds')}))
