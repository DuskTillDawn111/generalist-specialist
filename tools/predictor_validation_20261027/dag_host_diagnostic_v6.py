"""CPU-only fitted host coverage plus trained known-input forecast diagnosis.

Training templates are diagnostic inputs, never a heldout validation cohort.
All declared requests are preserved if any dependency remains UNKNOWN.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_host_release_v5 import DagHostReleaseProvider,SCENARIOS
from .forecast_gate_v3 import execute_bundle
from .native_collect import receipt,save


def diagnose(bundle_path,host_path,output,resource_path=None):
    bundle=json.loads(Path(bundle_path).read_text());host=json.loads(Path(host_path).read_text())
    dag=json.loads(Path(host['dag']['path']).read_text())
    if receipt(host['dag']['path'])!=host['dag']:raise ValueError('Changed training DAG')
    aliases=bundle['forecast_defaults']['aliases'];plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    rows=[dict(request_id=f'known-training-template:{i}',alias=p['alias'],task=p['task'],pool_id=p['pool_id'],arrival_ns=0)
          for i,p in enumerate(dag['profiles'])]
    queues={alias:[row for row in rows if row['alias']==alias] for alias in aliases}
    public=dict(phase_start_ns=0,now_ns=0,deadline_ns=10_000_000_000,drain_deadline_ns=120_000_000_000,
        queues=queues,future_workload=dict(rates={row['task']:1. for row in rows},
        input_pools={task:sorted({r['pool_id'] for r in rows if r['task']==task}) for task in {r['task'] for r in rows}},
        process='declared_schedule',declared_arrivals=[]))
    bundle['forecast_defaults']['latency_reference_ms']={task:bundle['forecast_defaults']['latency_reference_ms'][task] for task in public['future_workload']['rates']}
    bundle['forecast_defaults'].update(provider_kind='cuda_dag_host_supply_v4',route_initial_with_continuation_policy=False)
    bundle['artifacts'].update(dag=host['dag'],dag_host_release=receipt(host_path))
    bundle.update(allow_host_supply_identification_hypothesis=True,allow_host_supply_pooling_hypothesis=True)
    if resource_path:
        bundle['artifacts']['natural_resources']=receipt(resource_path)
        bundle['allow_conditional_resource_hypothesis']=True
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);results=[]
    for scenario in SCENARIOS:
        model=copy.deepcopy(bundle);model['host_supply_scenario']=scenario
        release=DagHostReleaseProvider(host,bundle['execution_context_sha256'],plans,aliases,scenario=scenario,
            allow_identification_hypothesis=True,allow_cross_input_pooling_hypothesis=True)
        coverage=[]
        for row in rows:
            plan=plans[row['alias'],row['task'],row['pool_id']]
            known=0;unknown=[]
            for node in plan['nodes']:
                try:release(node,row);known+=1
                except UnknownEvidence as exc:unknown.append(str(exc))
            coverage.append(dict(row,host_release_known_nodes=known,total_dag_nodes=len(plan['nodes']),unknown_reasons=sorted(set(unknown))))
        save(out/f'bundle-{scenario}.json',model)
        try:forecast=execute_bundle(model,public,allow_trained_hypothesis=True)
        except UnknownEvidence as exc:forecast=dict(status='UNKNOWN',reason=str(exc),full_declared_requests=rows)
        save(out/f'forecast-{scenario}.json',forecast)
        results.append(dict(scenario=scenario,host_release_coverage=coverage,forecast=receipt(out/f'forecast-{scenario}.json'),
            forecast_status=forecast.get('status','MODEL_EXECUTED'),forecast_reason=forecast.get('reason')))
    save(out/'public.json',public)
    result=dict(schema='gsdeploy.known_training_dag_host_wiring_diagnostic.v6',results=results,
        input_bundle=receipt(bundle_path),host_model=receipt(host_path),sources=[receipt(__file__)],
        actual_declared_template_requests=len(rows),declared_objective_task_subset=sorted(public['future_workload']['rates']),
        training_template_diagnostic_only=True,
        independent_validation_performed=False,prediction_accuracy_established=False,physical_gain_certified=False,
        measured_target_fallback_used=False,GPU_started=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True)
    p.add_argument('--host-model',required=True);p.add_argument('--resources');p.add_argument('--output',required=True)
    a=p.parse_args();result=diagnose(a.bundle,a.host_model,a.output,a.resources)
    print(json.dumps(dict(requests=result['actual_declared_template_requests'],
        scenarios=[dict(scenario=r['scenario'],status=r['forecast_status'],reason=r['forecast_reason']) for r in result['results']])))


if __name__=='__main__':main()
