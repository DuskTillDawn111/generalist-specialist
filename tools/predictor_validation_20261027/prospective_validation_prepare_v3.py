"""Prepare a public disjoint-input v3 design; never freeze or start collection.

The design uses public input bytes, passing training reviews and the previously
declared objective/candidate policies. Validation predictions/outcomes are not
read. A draft is deliberately not accepted as a frozen prediction-gate protocol.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from gsdeploy.forecasting.prediction_gate import DATA_SCHEMA
from gsdeploy.routing import Router
from .forecast_gate_v2 import INPUT_SCHEMA, _write
from .forecast_gate_v3 import BUNDLE_SCHEMA
from .kernel_domain_bridge_review_v3 import load_report
from .kernel_profile_campaign_v2 import input_identity
from .native_collect import receipt

TASKS=('text','image','asr','ocr')
SELECTION_SALT='prospective-disjoint-v3-20261028:'
DESIGN_SCHEMA='gsdeploy.prospective_disjoint_validation_design.v3'


def declared_training_inputs(plan):
    inputs={row['pool_id']:{key:row[key] for key in ('task','pool_id','exact_input_sha256')} for row in plan['inputs']}
    additional=plan.get('additional_job_bound_training_input')
    if additional is not None:
        ref=additional['selected_input']
        if receipt(ref['path'])!=ref:raise ValueError('Changed additional selected training input')
        row=json.loads(Path(ref['path']).read_text())
        if row['pool_id']!=additional['pool_id']:raise ValueError('Additional training pool identity mismatch')
        value=dict(task=row['task_type'],pool_id=row['pool_id'],exact_input_sha256=input_identity(row))
        if inputs.setdefault(row['pool_id'],value)!=value:raise ValueError('Conflicting additional training content')
    return inputs


def training_domain(review_paths):
    records=[];captures=[];inputs={};seen=set()
    for path in review_paths:
        report=json.loads(Path(path).read_text())
        plan=json.loads(Path(report['plan']['path']).read_text())
        loader=load_report
        if report.get('schema')=='gsdeploy.startup_training_collection_strict_review.v4':
            from .small_training_domain_bridge_review_v4 import load_report as loader
        report,plan,raw=loader(path,plan['domain'])
        run='raw-sha256:'+report['raw']['sha256']
        if run in seen:raise ValueError('A repeated raw is not another capture')
        seen.add(run)
        if report.get('training_only') is not True or raw.get('status')!='completed':
            raise ValueError('Completed strict training capture required')
        captures.append(dict(run_id=run,domain=plan['domain'],review=receipt(path),
                             raw=report['raw'],plan=report['plan']))
        planned=declared_training_inputs(plan)
        ids=set()
        for row in raw['request_records']:
            if row.get('split')!='training' or row.get('status')!='completed':
                raise ValueError('Every fitted capture row must be completed training')
            if row['request_id'] in ids:raise ValueError('Duplicate training request identity')
            ids.add(row['request_id'])
            value=planned[row['pool_id']]
            public={key:value[key] for key in ('task','pool_id','exact_input_sha256')}
            if public['task']!=row['task']:raise ValueError('Training task differs from declared input')
            previous=inputs.setdefault(row['pool_id'],public)
            if previous!=public:raise ValueError('A training input ID changed content or task')
            records.append(dict(run_id=run,request_id=row['request_id'],pool_id=row['pool_id'],task=row['task']))
    if not records:raise ValueError('Nonempty complete model training lineage required')
    return dict(schema=DATA_SCHEMA,split='training',records=records),captures,list(inputs.values())


def select_public(rows,training_inputs,*,per_task=4):
    ids={row['pool_id'] for row in training_inputs}
    hashes={row['exact_input_sha256'] for row in training_inputs}
    selected={};manifest=[]
    for task in TASKS:
        eligible=sorted((row for row in rows if row['task_type']==task and row['pool_id'] not in ids),
                        key=lambda row:hashlib.sha256((SELECTION_SALT+row['pool_id']).encode()).hexdigest())
        used=set();chosen=[]
        for row in eligible:
            sha=input_identity(row)
            if sha in hashes or sha in used:continue
            used.add(sha);chosen.append(row)
            item=dict(task=task,pool_id=row['pool_id'],exact_input_sha256=sha,
                      selection_key=hashlib.sha256((SELECTION_SALT+row['pool_id']).encode()).hexdigest())
            sample=row['sample']
            if task=='text':item['public_characters']=len(sample['context'])+len(sample['question'])
            else:
                item['input_file']=receipt(sample['input'])
                if item['input_file']['sha256']!=sha:raise ValueError('Public input changed during hashing')
            manifest.append(item)
            if len(chosen)==per_task:break
        if len(chosen)!=per_task:raise ValueError('Insufficient training-disjoint public content for '+task)
        selected[task]=[row['pool_id'] for row in chosen]
    return selected,manifest


def nonstationary_clock(intensity_seconds):
    """Inverse integrated intensity for .5x / 2x / .5x over three 30s phases."""
    if intensity_seconds<15:return intensity_seconds/.5
    if intensity_seconds<75:return 30+(intensity_seconds-15)/2
    return 60+(intensity_seconds-75)/.5


def public_trace(rates,pools):
    result=[]
    for task in TASKS:
        phase={'text':0.,'image':.19,'asr':.47,'ocr':.73}[task]
        index=0
        while (phase+index)/rates[task]<90:
            seconds=nonstationary_clock((phase+index)/rates[task])
            result.append(dict(offset_ns=round(seconds*1e9),task=task,
                               pool_id=pools[task][index%len(pools[task])]))
            index+=1
    result.sort(key=lambda row:(row['offset_ns'],row['task']))
    for index,row in enumerate(result):row['request_id']='measurement:'+str(index)
    return result


def candidate_public(candidate,trace,rates,aliases):
    router=Router(candidate['routing_weights']);queues={alias:[] for alias in aliases};routes={}
    for row in trace:
        alias=router.choose(row['task']);routes[row['request_id']]=alias
        if row['offset_ns']==0:
            queues[alias].append({key:row[key] for key in ('request_id','task','pool_id')} |
                                 dict(arrival_ns=0,alias=alias))
    public=dict(phase_start_ns=0,now_ns=0,deadline_ns=90000000000,drain_deadline_ns=210000000000,
        queues=queues,future_workload=dict(rates=rates,input_pools={task:sorted({row['pool_id'] for row in trace
            if row['task']==task}) for task in TASKS},process='declared_schedule',
            declared_arrivals=[copy.deepcopy(row) for row in trace if row['offset_ns']>0]))
    return dict(schema=INPUT_SCHEMA,forecast_inputs=public),routes


def prepare(configuration,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    objective=json.loads(Path(configuration['objective_plan']).read_text())
    index,captures,training_inputs=training_domain(configuration['training_reviews'])
    pool_path=Path(configuration['runtime_inputs']).resolve()
    rows=[json.loads(line) for line in pool_path.read_text().splitlines()]
    exclusions={item['pool_id']:item for item in training_inputs}
    planned_exclusions=[]
    for path in configuration.get('training_exclusion_plan_paths',[]):
        planned=json.loads(Path(path).read_text())
        if planned.get('split')!='training':raise ValueError('Only declared training plans may reserve exclusion inputs')
        planned_exclusions.append(receipt(path))
        for item in declared_training_inputs(planned).values():
            value={key:item[key] for key in ('task','pool_id','exact_input_sha256')}
            previous=exclusions.setdefault(item['pool_id'],value)
            if previous!=value:raise ValueError('A reserved training input ID changed content or task')
    pools,inputs=select_public(rows,list(exclusions.values()))
    trace=public_trace(objective['workload_rps'],pools)
    _write(output/'training-index.json',index)
    _write(output/'public-domain.json',dict(schema='gsdeploy.public_disjoint_input_reservation.v3',
        split='prospective_validation_reserved',selection_rule=SELECTION_SALT+'pool_id SHA256 ascending, unique exact input content',
        source=receipt(pool_path),inputs=inputs,training_inputs=training_inputs,
        additional_reserved_training_inputs=[item for pool,item in exclusions.items() if pool not in {r['pool_id'] for r in training_inputs}],
        training_exclusion_plans=planned_exclusions,
        disjoint_pool_ids=True,disjoint_exact_input_content=True,
        scope='Disjoint from every indexed training capture and declared future training exclusion plan; not a claim about all historical project data',
        outputs_or_validation_timings_read=False,labels_used_for_selection=False,GPU_started=False))
    _write(output/'trace.json',trace)
    artifacts={name:receipt(path) for name,path in configuration['artifacts'].items()}
    contract=json.loads(Path(artifacts['composite']['path']).read_text())
    target=contract['target']['execution_context_sha256'];aliases=objective['aliases']
    run='prospective-nonstationary-disjoint-v3-20261028-01';candidates=[];jobs=[];model_paths={}
    for previous in objective['candidates']:
        if previous['id'] not in ('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high'):continue
        candidate=copy.deepcopy(previous)
        if candidate['resource_configuration']['mode']!='shared':raise ValueError('Only original shared policies in this design')
        public,routes=candidate_public(candidate,trace,objective['workload_rps'],aliases)
        cid=candidate['id'];public_path=output/('public-'+cid+'.json')
        _write(public_path,public)
        defaults={key:copy.deepcopy(objective[key]) for key in ('latency_reference_ms','memory_budget_mib',
            'energy_reference_watts','memory_safe_factor','aliases')}
        defaults.update(continuation_routing_rps=candidate['routing_weights'],resources=candidate['resource_configuration'],
            provider_kind='cuda_dag_v2',execution_contract='strict_lifecycle_v2',service_scenario='profiled',
            route_initial_with_continuation_policy=True,require_energy=False,require_memory=False,max_events=500000)
        bundle=dict(schema=BUNDLE_SCHEMA,kind='trained_hypothesis',execution_context_sha256=target,
            forecast_defaults=defaults,artifacts=artifacts,
            baseline_intakes=[receipt(path) for path in configuration['baseline_intakes']],
            weights=objective['weights'],training_indices=[receipt(output/'training-index.json')],
            training_lineage=[receipt(path) for path in configuration['training_reviews']],
            startup_amortization_seconds=objective['startup_amortization_seconds'],
            kernel_duration_mode='trained_kernel_baseline_v3',
            hypothesis_options=dict(allow_independent_host_supply_hypothesis=True,
                device_progress_hypothesis='equal_context_time_slice',context_switch_ms=0.),
            dag_hypothesis_options=dict(allow_independent_copy_hypothesis=True,
                allow_context_time_slice_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True),
            draft_only=True,physical_accuracy_validated=False,physical_gain_certified=False)
        if configuration.get('lifecycle_provider_kind') is not None:
            bundle.update(lifecycle_provider_kind=configuration['lifecycle_provider_kind'],
                          allow_pooling_hypothesis=configuration.get('allow_pooling_hypothesis',False))
        model_path=output/('bundle-'+cid+'.json');_write(model_path,bundle)
        model_paths[cid]=str(model_path)
        candidates.append(dict(id=cid,model_id=cid,execution_context_sha256=target))
        jobs.append(dict(run_id=run,candidate_id=cid,public_inputs_path=str(public_path)))
        _write(output/('routes-'+cid+'.json'),routes)
    if len(candidates)!=4:raise ValueError('All four original candidate policies required')
    _write(output/'jobs.json',jobs)
    thresholds=dict(service_mare_max=.2,e2e_mare_max=.2,p95_mare_max=.2,queue_depth_mae_max=1.,
                    delta_sign_accuracy_min=1.,pair_rank_accuracy_min=1.)
    design=dict(schema=DESIGN_SCHEMA,status='DESIGN_DRAFT_NOT_FROZEN',not_for_gpu_execution=True,
        source_configuration=configuration,preparer_source=receipt(__file__),
        objective_source=receipt(configuration['objective_plan']),
        execution_context_sha256=target,model_paths=model_paths,
        training_data_paths=[str(output/'training-index.json')],training_captures=captures,
        public_domain=receipt(output/'public-domain.json'),trace=receipt(output/'trace.json'),jobs=receipt(output/'jobs.json'),
        candidates=candidates,baseline_candidate_id='shared_balanced',
        validation_domain=[dict(run_id=run,requests=[{key:row[key] for key in ('request_id','pool_id','task')} for row in trace])],
        thresholds=thresholds,threshold_policy='Original predeclared .2/1/1/1 values; no added queue_ms gate or retuning',
        queue_sample_offsets_ns=[900000000*i for i in range(101)],weights=objective['weights'],utility_tie_tolerance=0.,
        normalizers={key:objective[key] for key in ('latency_reference_ms','memory_budget_mib','energy_reference_watts',
                                                 'memory_safe_factor','startup_amortization_seconds')},
        arrival_design=dict(kind='public_piecewise_nonstationary',duration_seconds=90.,
            phases=[dict(start_seconds=a,end_seconds=b,rate_multiplier=f) for a,b,f in ((0,30,.5),(30,60,2.),(60,90,.5))],
            time_average_task_rps=objective['workload_rps'],same_trace_and_inputs_for_every_candidate=True),
        frozen_protocol_created=False,predictions_sealed=False,GPU_started=False,
        readiness_blockers=['Source/model closure is still under development and must be frozen as one snapshot',
            'Unseen-input executable kernel/dependency templates are not trained',
            'Non-affine, OCR and copy/memset trained-duration coverage is incomplete',
            'Natural launch-supply generalization and joint host/device scheduler model require independent validation',
            'Board-energy, startup and owned-memory prediction must be complete for delta-U sign/ranking',
            'The target context currently declares only text/OCR; all-task readiness and prospective source-bound collector plan need review'],
        required_gate_order=['complete_prediction_domain','complete_native_cohort','service_and_parent_E2E',
                             'each_task_P95_and_queue','six_term_delta_sign_and_rank'])
    _write(output/'design.json',design)
    return design


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=prepare(json.loads(Path(args.configuration).read_text()),args.output)
    print(json.dumps(dict(status=result['status'],design=receipt(Path(args.output)/'design.json'))))


if __name__=='__main__':main()
