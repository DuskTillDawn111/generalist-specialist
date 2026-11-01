"""Actual request DAG FIFO/future-arrival diagnostic, with explicit replay flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cuda_dag_simulator_v2 import DagFullHorizonModel
from .native_collect import receipt
from .kernel_sequence_audit_v2 import verify


def replay(path,*,quality_path=None,objective_path=None):
    document=json.loads(Path(path).read_text())
    if document.get('schema')!='gsdeploy.request_cuda_dag_profiles.v2' or document.get('training_only') is not True:
        raise ValueError('Source-bound real training request DAG document required')
    for source in document['sources']:verify(source)
    profiles=document['profiles'];aliases={p['alias']:dict(family=p['family'],supported_tasks=[p['task']]) for p in profiles}
    plans={(p['alias'],p['task'],p['pool_id']):p for p in profiles};requests=[];counts={}
    for p in profiles:
        count=counts.get(p['alias'],0);counts[p['alias']]=count+1
        requests.append(dict(request_id='known-dag:'+p['request_id'],alias=p['alias'],task=p['task'],pool_id=p['pool_id'],
                             arrival_ns=count*50000000))
    model=DagFullHorizonModel(aliases,plans,allow_profiled_replay=True,allow_independent_copy_hypothesis=True,
        allow_context_time_slice_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True)
    forecast=model.forecast(requests,0,10**10)
    scoring=None
    if quality_path or objective_path:
        if not quality_path or not objective_path:raise ValueError('Quality and objective sources required together')
        from gsdeploy.forecasting.metrics import score_forecast
        quality=json.loads(Path(quality_path).read_text());objective=json.loads(Path(objective_path).read_text())
        tasks={r['task'] for r in requests}
        scoring=score_forecast(forecast,[],aliases,{t:objective['workload_rps'][t] for t in tasks},
            {t:objective['latency_reference_ms'][t] for t in tasks},quality,phase_start_ns=0,now_ns=0,deadline_ns=10**10,
            memory_budget_mib=objective['memory_budget_mib'],energy_reference_watts=objective['energy_reference_watts'],
            weights=objective['weights'],memory_safe_factor=objective['memory_safe_factor'],observed_board_energy_joules=0.)
        scoring['scope']='Known-input text+OCR diagnostic task subset, not full four-task workload'
    return dict(schema='gsdeploy.request_cuda_dag_full_horizon_diagnostic.v2',source=receipt(path),
        evaluator=receipt(__file__),forecast=forecast,scoring=scoring,
        kernel_durations_are_profiled_replay=True,host_issue_release_is_profiled_replay=True,
        lifecycle_scope='Instant handoff diagnostic; strict lifecycle must use the composite-bound producer',
        GPU_started=False,independent_validation=False,unseen_input_validation=False,
        physical_accuracy_validated=False,deployment_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',required=True);parser.add_argument('--quality');parser.add_argument('--objective')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=replay(args.profiles,quality_path=args.quality,objective_path=args.objective)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(finishes={r:row['finish_ns'] for r,row in result['forecast']['finishes'].items()},
        max_parallel=result['forecast']['max_simultaneous_nodes_per_owner'],scoring=result['scoring']['terms'] if result['scoring'] else None)))


if __name__=='__main__':main()
