"""Prepare a fresh text/OCR co-run pilot using fixed trained public contents.

This creates reviewable CPU plans only. Model sealing and GPU execution are
separate. All four owners and the original twelve priming requests remain.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from gsdeploy.routing import Router
from .known_input_native_flow_v5 import SCHEMA,collection_source_paths,resource_configuration,validate_plan
from .native_collect import receipt,save
from .profiling import digest,RESIDENTS


def prepare(bundle_path,priming_profile_plan,output):
    base=json.loads(Path(bundle_path).read_text());contract=json.loads(Path(base['artifacts']['composite']['path']).read_text())
    context_ref=contract['target']['context'];context=json.loads(Path(context_ref['path']).read_text())
    if receipt(context_ref['path'])!=context_ref:raise ValueError('Changed target context')
    priming=json.loads(Path(priming_profile_plan).read_text())
    if len(priming['inputs'])!=8 or {p['task'] for p in priming['inputs']}!={'text','image','asr','ocr'}:
        raise ValueError('The original full eight-input priming profile plan is required')
    dag=json.loads(Path(base['artifacts']['dag']['path']).read_text())
    inputs={task:sorted({p['pool_id'] for p in dag['profiles'] if p['task']==task}) for task in ('text','ocr')}
    if any(len(values)!=2 for values in inputs.values()):raise ValueError('Exactly two trained inputs per measured task required')
    rates=dict(text=1.85,ocr=.05);duration=30.;phases=[(0.,10.,.5),(10.,20.,2.),(20.,30.,.5)];trace=[]
    for task,rate in rates.items():
        index=0;threshold=0. if task=='text' else .73
        while threshold<rate*duration:
            accumulated=0.
            for start,end,multiplier in phases:
                offered=(end-start)*rate*multiplier
                if threshold<accumulated+offered:
                    offset=start+(threshold-accumulated)/(rate*multiplier);break
                accumulated+=offered
            trace.append(dict(offset_ns=round(offset*1e9),task=task,pool_id=inputs[task][index%2]))
            index+=1;threshold+=1.
    trace.sort(key=lambda row:(row['offset_ns'],row['task']))
    for index,row in enumerate(trace):row['request_id']='measurement:'+str(index)
    candidates=[]
    for name,fraction,spill in [('balanced',.5,0.),('skewed',.25,0.),('spill_low',.5,1/6),('spill_high',.5,1/3)]:
        text={'text#0':(1-spill)*fraction,'text#1':(1-spill)*(1-fraction)}
        if spill:text['generalist_2#0']=spill
        weights=dict(text=text,ocr={'ocr#0':1.});router=Router(weights)
        candidates.append(dict(id='shared_'+name,execution_context_sha256=contract['target']['execution_context_sha256'],
            routing_weights=weights,routes={r['request_id']:router.choose(r['task']) for r in trace},
            allow_spill=bool(spill),resource_configuration=resource_configuration()))
    collector=Path(__file__).with_name('known_input_native_flow_v5.py');catalog=Path(__file__).resolve().parents[2]/'artifacts/static-study-v6-20261021/inputs/catalog.json'
    defaults=base['forecast_defaults']
    plan=dict(schema=SCHEMA,kind='prospective_known_input_conditional_pilot',split='validation',
        phase='known_input_predictor_validation',validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
        scope='Fresh heldout co-run requests of trained text/OCR contents; four residents retained; not four-task validation or final routing gain evidence.',
        unseen_input_generalization_claimed=False,OCR_tail_population_generalization_claimed=False,
        execution_context_sha256=contract['target']['execution_context_sha256'],context=context_ref,
        profile_plan=receipt(priming_profile_plan),catalog=receipt(catalog),runner_source=receipt(collector),
        collector_sources=[receipt(p) for p in collection_source_paths()],protocol_generator=receipt(__file__),
        aliases=copy.deepcopy(defaults['aliases']),window_seconds=duration,drain_seconds=120.,warmup_seconds=0.,
        arrival_process='declared_piecewise_periodic',arrival_rate_phases=[dict(start_seconds=a,end_seconds=b,multiplier=m) for a,b,m in phases],
        workload_rps=rates,latency_reference_ms={task:defaults['latency_reference_ms'][task] for task in rates},
        weights=base['weights'],memory_budget_mib=defaults['memory_budget_mib'],energy_reference_watts=defaults['energy_reference_watts'],
        memory_safe_factor=defaults['memory_safe_factor'],startup_amortization_seconds=base['startup_amortization_seconds'],
        trace=trace,trace_sha256=digest(trace),candidates=candidates,baseline_candidate_id='shared_balanced',
        queue_sample_offsets_ns=[round(i*.3*1e9) for i in range(101)],queue_depth_boundary='waiting_until_handler_start',
        primary_host_supply_scenario='host_issue_depth_supply',sensitivity_host_supply_scenario='front_loaded_supply',
        host_scenario_selected_after_observations=False,
        validation_thresholds=dict(service_mare_max=.2,e2e_mare_max=.2,p95_mare_max=.2,
            queue_depth_mae_max=1.,delta_sign_accuracy_min=1.,pair_rank_accuracy_min=1.),
        base_model_bundle=receipt(bundle_path),full_priming_retained=True,GPU_started=False)
    plan['plan_sha256']=digest(plan);validate_plan(plan)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);save(out/'plan.json',plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','priming-profile-plan','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();plan=prepare(a.bundle,a.priming_profile_plan,a.output)
    print(json.dumps(dict(requests=len(plan['trace']),tasks=sorted(plan['workload_rps']),candidates=len(plan['candidates']),GPU_started=False)))


if __name__=='__main__':main()
