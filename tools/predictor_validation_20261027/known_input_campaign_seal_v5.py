"""Bind exact-content pilot models, seal all forecasts, and require completeness.

This command is CPU only. An incomplete seal cannot authorize collection.
Every output directory is exclusive; reruns preserve earlier evidence.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

from .forecast_gate_v2 import INPUT_SCHEMA,_public_cohort
from .forecast_gate_v3 import prediction_source_paths
from .historical_evidence_v4 import resolve_closure
from .known_input_gate_v5 import freeze_gate_protocol,_verify_protocol
from .known_input_forecast_gate_v5 import run_and_seal
from .known_input_native_flow_v5 import collection_source_paths,validate_plan
from .native_collect import receipt,save
from .startup_prediction_v6 import prediction as startup_prediction


def check_complete(protocol,predictions):
    candidates={c['id'] for c in protocol['candidates']}
    cells={(c['run_id'],c['candidate_id']):c for c in predictions['cells']}
    expected={(run,cid) for run in protocol['validation_domain'] for cid in candidates}
    if set(cells)!=expected:raise ValueError('Incomplete candidate/run forecast domain')
    missing=[]
    for key,cell in cells.items():
        domain={r['request_id'] for r in protocol['validation_domain'][key[0]]}
        if {r['request_id'] for r in cell['requests']}!=domain:missing.append([*key,'request_domain'])
        for row in cell['requests']:
            for field in ('service_ms','queue_ms','e2e_ms'):
                if row[field] is None:missing.append([*key,row['request_id'],field])
        for term in 'QTLPME':
            if cell.get('terms') is None or cell['terms'][term] is None:missing.append([*key,term])
        samples={s['offset_ns']:s['depth'] for s in cell.get('queue_depth_samples',[])}
        if set(samples)!=set(protocol['queue_sample_offsets_ns']) or any(v is None for v in samples.values()):
            missing.append([*key,'queue_grid'])
    if missing:raise ValueError('Incomplete predictions prohibit collection: '+json.dumps(missing[:10]))
    return dict(status='COMPLETE_PRESEALED_PREDICTIONS',candidate_cells=len(cells),
        request_rows=sum(len(c['requests']) for c in cells.values()),all_six_terms_known=True,
        complete_service_queue_e2e_and_queue_grid=True,prediction_accuracy_established=False,
        physical_gain_certified=False,GPU_started=False)


def public_inputs(plan,candidate):
    initial={alias:[] for alias in plan['aliases']};future=[];routes={}
    for offered in plan['trace']:
        row={key:offered[key] for key in ('request_id','pool_id','task')}
        alias=candidate['routes'][row['request_id']];routes[row['request_id']]=alias
        if offered['offset_ns']==0:initial[alias].append(dict(row,alias=alias,arrival_ns=0))
        else:future.append(dict(row,offset_ns=offered['offset_ns']))
    public=dict(phase_start_ns=0,now_ns=0,deadline_ns=round(plan['window_seconds']*1e9),
        drain_deadline_ns=round((plan['window_seconds']+plan['drain_seconds'])*1e9),queues=initial,
        future_workload=dict(rates=plan['workload_rps'],input_pools={task:sorted({r['pool_id'] for r in plan['trace'] if r['task']==task}) for task in plan['workload_rps']},
            process='declared_schedule',declared_arrivals=future))
    document=dict(schema=INPUT_SCHEMA,forecast_inputs=public)
    _public_cohort(document)
    return document,routes


def prepare_and_seal(plan_path,training_index,startup_model,generation_model,generation_support,output):
    plan=json.loads(Path(plan_path).read_text());validate_plan(plan)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(plan['base_model_bundle']['path']).read_text())
    if receipt(plan['base_model_bundle']['path'])!=plan['base_model_bundle']:raise ValueError('Changed base pilot model')
    run='fresh-known-input-text-ocr-co-run-v5-20261028-01';jobs=[];model_paths={};candidates=[]
    for candidate in plan['candidates']:
        cid=candidate['id'];bundle=copy.deepcopy(base)
        defaults=bundle['forecast_defaults'];defaults.update(continuation_routing_rps=candidate['routing_weights'],
            latency_reference_ms=plan['latency_reference_ms'],memory_budget_mib=plan['memory_budget_mib'],
            energy_reference_watts=plan['energy_reference_watts'],memory_safe_factor=plan['memory_safe_factor'],
            route_initial_with_continuation_policy=True,max_events=5_000_000,resources=None)
        bundle.update(weights=plan['weights'],startup_amortization_seconds=plan['startup_amortization_seconds'],
            training_indices=[receipt(training_index)],host_supply_scenario=plan['primary_host_supply_scenario'],
            startup_prediction_provider_kind='verified_prephase_transfer_v6',startup_candidate_id=cid,
            allow_native_priming_transfer_hypothesis=True,allow_exact_content_generation_hypothesis=True)
        bundle['artifacts'].update(startup_training=receipt(startup_model),flow_plan=receipt(plan_path),
            generation_work=receipt(generation_model),generation_dag_support=receipt(generation_support))
        bundle['startup_prediction']=startup_prediction(startup_model,plan_path,cid,allow_native_priming_transfer_hypothesis=True)
        save(out/f'bundle-{cid}.json',bundle);model_paths[cid]=str(out/f'bundle-{cid}.json')
        candidates.append(dict(id=cid,model_id=cid,execution_context_sha256=candidate['execution_context_sha256']))
        public,routes=public_inputs(plan,candidate)
        save(out/f'public-{cid}.json',public)
        save(out/f'routes-{cid}.json',routes)
        jobs.append(dict(run_id=run,candidate_id=cid,public_inputs_path=str(out/f'public-{cid}.json')))
    save(out/'jobs.json',jobs)
    manifest=base['artifacts'].get('historical_resolution');resolution=resolve_closure(list(model_paths.values()),manifest['path'] if manifest else None)
    sources={r['path'] for r in resolution['materialized_sources']}
    sources.update(str(p) for p in prediction_source_paths());sources.update(str(p) for p in collection_source_paths())
    sources.update([str(Path(plan_path).resolve()),str(Path(training_index).resolve()),str(out/'jobs.json')])
    sources.update(job['public_inputs_path'] for job in jobs)
    context=json.loads(Path(plan['context']['path']).read_text())
    catalog=[s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(catalog)!=1:raise ValueError('One immutable target public input catalog required')
    protocol=freeze_gate_protocol(model_paths=model_paths,source_paths=sorted(sources),training_data_paths=[training_index],
        candidates=candidates,baseline_candidate_id=plan['baseline_candidate_id'],
        validation_domain=[dict(run_id=run,requests=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']])],
        thresholds=plan['validation_thresholds'],created_unix_ns=time.time_ns(),weights=plan['weights'],
        queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],public_input_catalog_path=catalog[0])
    save(out/'protocol.json',protocol);save(out/'source-resolution.json',resolution)
    result=run_and_seal(out/'protocol.json',jobs,out/'predictions',allow_trained_hypothesis=True)
    sealed=json.loads(Path(result['predictions']['path']).read_text());complete=check_complete(protocol,sealed)
    _verify_protocol(protocol);save(out/'complete-prediction-review.json',complete)
    return dict(protocol=receipt(out/'protocol.json'),predictions=result['predictions'],
        completeness_review=receipt(out/'complete-prediction-review.json'),run_id=run,GPU_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','training-index','startup-model','generation-model','generation-support','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(prepare_and_seal(a.plan,a.training_index,a.startup_model,a.generation_model,a.generation_support,a.output)))


if __name__=='__main__':main()
