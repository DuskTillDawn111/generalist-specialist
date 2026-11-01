"""CPU known-input FullHorizon diagnostic; never independent prediction validation."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from gsdeploy.forecasting.simulator import FullHorizonModel
from .launch_supply_residual_v2 import LaunchSupplyResidualProvider
from .native_collect import receipt


def replay(ordered_path,residual_path,profile_ids,*,device_progress_hypothesis='equal_context_time_slice',
           context_switch_ms=0.,arrival_spacing_ms=0.,composite_path=None,lifecycle_path=None,
           quality_path=None,objective_plan_path=None):
    ordered=json.loads(Path(ordered_path).read_text());residual=json.loads(Path(residual_path).read_text())
    chosen=[p for p in ordered['profiles'] if p['id'] in profile_ids]
    if len(chosen)!=len(set(profile_ids)) or len({p['alias'] for p in chosen})!=len(chosen):
        raise ValueError('Choose existing exact profiles with distinct resident aliases')
    ordered['profiles']=chosen;ordered.pop('sequence_audit',None);gc.collect()
    aliases={p['alias']:dict(family=p['family'],supported_tasks=[p['task']]) for p in chosen}
    # Evidence string explicitly grants only a model hypothesis, never CUDA capability proof.
    ordered['concurrency_evidence']='EXPLICIT_UNVALIDATED_CROSS_CONTEXT_SCHEDULER_HYPOTHESIS'
    provider=LaunchSupplyResidualProvider(ordered,residual,ordered['execution_context_sha256'],
        allow_independent_host_supply_hypothesis=True,allow_analytical_hypothesis=True,
        device_progress_hypothesis=device_progress_hypothesis,context_switch_ms=context_switch_ms)
    lifecycle=None
    if composite_path is not None or lifecycle_path is not None:
        if composite_path is None or lifecycle_path is None:raise ValueError('Composite and lifecycle artifact required together')
        from .composite_execution_v2 import CompositeComponentAdapter
        from gsdeploy.forecasting.lifecycle import LifecycleProvider
        contract=json.loads(Path(composite_path).read_text())
        life_document=json.loads(Path(lifecycle_path).read_text())
        lifecycle=LifecycleProvider(life_document,life_document['execution_context_sha256'])
        provider=CompositeComponentAdapter(provider,contract,'service',residual_path)
        lifecycle=CompositeComponentAdapter(lifecycle,contract,'lifecycle',lifecycle_path)
    queues={p['alias']:[dict(request_id='known-input:'+p['id'],task=p['task'],pool_id=p['pool_id'],arrival_ns=0)]
            for p in chosen}
    if arrival_spacing_ms:
        # Additional requests exercise FIFO+future arrival recovery with the exact same known input.
        future=[dict(request_id='known-input-repeat:'+p['id'],task=p['task'],pool_id=p['pool_id'],
                     arrival_ns=round(arrival_spacing_ms*1e6)) for p in chosen]
        # Same task/input can exist at multiple owners; route explicitly by generated request identity.
        routes={'known-input-repeat:'+p['id']:p['alias'] for p in chosen}
        route=lambda row,*_:routes[row['request_id']]
    else:future=[];route=None
    result=FullHorizonModel(aliases,provider,lifecycle_provider=lifecycle,max_events=500000).forecast(queues,0,10**10,scenario='profiled',
        future_arrivals=future,route_future=route,require_energy=False,require_memory=False,
        trace_request_id='omit_large_transitions_from_diagnostic')
    scoring=None;scoring_sources=[]
    if quality_path is not None or objective_plan_path is not None:
        if quality_path is None or objective_plan_path is None:raise ValueError('Quality and objective plan required together')
        from gsdeploy.forecasting.metrics import score_forecast
        quality=json.loads(Path(quality_path).read_text());objective=json.loads(Path(objective_plan_path).read_text())
        tasks={p['task'] for p in chosen}
        scoring=score_forecast(result,[],aliases,{t:objective['workload_rps'][t] for t in tasks},
            {t:objective['latency_reference_ms'][t] for t in tasks},quality,
            phase_start_ns=0,now_ns=0,deadline_ns=10**10,memory_budget_mib=objective['memory_budget_mib'],
            energy_reference_watts=objective['energy_reference_watts'],weights=objective['weights'],
            memory_safe_factor=objective['memory_safe_factor'],observed_board_energy_joules=0.)
        scoring['scope']='Known-input diagnostic task subset only; absent workload tasks are not evaluated'
        scoring['configured_workload_tasks']=list(objective['workload_rps'])
        scoring_sources=[receipt(quality_path),receipt(objective_plan_path)]
    return dict(schema='gsdeploy.known_input_supply_full_horizon_diagnostic.v2',
        sources=[receipt(ordered_path),receipt(residual_path),receipt(__file__)]+
                ([receipt(composite_path),receipt(lifecycle_path)] if composite_path else []),profile_ids=profile_ids,
        scoring=scoring,scoring_sources=scoring_sources,
        lifecycle_included=lifecycle is not None,
        device_progress_hypothesis=device_progress_hypothesis,context_switch_ms=context_switch_ms,
        forecast={k:v for k,v in result.items() if k not in ('transitions','evidence')},
        real_measured_training_inputs_used=True,GPU_started=False,
        prediction_validation=False,unseen_input_evaluated=False,in_sample_calibration=True,
        natural_timing_validated=False,physically_validated=False,deployment_qualified=False,
        assumptions=['Host supply independent across owners','Device copy interference absent',
                     'Profiled kernel timing domain retained','GPU scheduler hypothesis unvalidated'],
        claim='EXECUTABLE_KNOWN_INPUT_MODEL_DIAGNOSTIC_NOT_ACCURACY_OR_GAIN_REPORT')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordered',required=True);parser.add_argument('--residual',required=True)
    parser.add_argument('--profile',action='append',required=True)
    parser.add_argument('--device-progress',choices=['static_packing','equal_context_time_slice'],default='equal_context_time_slice')
    parser.add_argument('--context-switch-ms',type=float,default=0.)
    parser.add_argument('--arrival-spacing-ms',type=float,default=0.)
    parser.add_argument('--composite');parser.add_argument('--lifecycle')
    parser.add_argument('--quality');parser.add_argument('--objective-plan')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=replay(args.ordered,args.residual,args.profile,device_progress_hypothesis=args.device_progress,
                  context_switch_ms=args.context_switch_ms,arrival_spacing_ms=args.arrival_spacing_ms,
                  composite_path=args.composite,lifecycle_path=args.lifecycle,
                  quality_path=args.quality,objective_plan_path=args.objective_plan)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(finishes=result['forecast']['finishes'],claim=result['claim'])))


if __name__=='__main__':main()
