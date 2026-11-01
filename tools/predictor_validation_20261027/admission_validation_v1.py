"""Independent admission2 validation, fixed seed628201; never training evidence.

CPU prepare/seal/report only. Fixed shared Gamma service-law transfer is a
hypothesis; resource/capacity/quality use the separate admission2 T8 campaign.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
import time

from gsdeploy.forecast_errors import UnknownEvidence
from . import admission_training_v1 as training
from . import admission_capacity_v1 as capacity
from . import admission_resources_v1 as resources
from . import admission_quality_v1 as quality
from . import flow_validation as flow
from . import flow_gate, flow_crossfit_report, flow_profiles, mechanism_audit, training_run_audit
from . import h600_event_predict_v2 as numerical
from . import h600_admission_pilot_v1 as activation_source
from . import gamma_h600_v2 as gamma
from .h600_event_simulator_v2 import FullHorizonModel
from .ocr_trained_validation import LIMITS
from .flow_eligibility import native_stability, _routing
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

SCHEMA='gsdeploy.admission2_independent_validation.v1'
SEED=628201


def sources():
    return capacity.unique_refs([receipt(p) for p in (__file__,resources.__file__,quality.__file__,
        flow.__file__,flow_gate.__file__,flow_crossfit_report.__file__,flow_profiles.__file__,
        mechanism_audit.__file__,training_run_audit.__file__,numerical.__file__,
        activation_source.__file__,FullHorizonModel.forecast.__code__.co_filename,
        native_stability.__code__.co_filename)]+capacity.sources()+gamma.sources())


def reservation(path):
    value=read(path)
    if (value.get('schema')!='gsdeploy.H600_admission_next_stage_role_reservation.v1'
            or value.get('training')!=dict(seeds=list(capacity.SEEDS),mode='admission2',
                routes=['balanced','skewed','spill_low','spill_high'],runs=8,requests=9600)
            or value.get('independent')!=dict(seed=SEED,mode='admission2',
                routes=['balanced','skewed','spill_low','spill_high'],runs=4,requests=4800)
            or value.get('existing_shared_failure_preserved') is not True
            or value.get('qualified_positive_final_gain_eligible') is not False):
        raise ValueError('Exact finite T8/V4 role reservation required')
    checked(value['service_hypothesis']['model'])
    return value


def threshold(path):
    value=read(path)
    if (value.get('schema')!='gsdeploy.training_cv_selection_threshold_source_addendum.v1'
            or value.get('thresholds')!=LIMITS
            or read(checked(value['threshold_plan']))['validation_thresholds']!=LIMITS):
        raise ValueError('Original immutable prediction thresholds required')
    return value


def _project(native, campaign, created_ns, reservation_ref, threshold_ref):
    plan=copy.deepcopy(native)
    for key in ('plan_sha256','prephase_memory_evidence','startup_comparison_policy'):
        plan.pop(key,None)
    plan['candidates']=[c for cid in capacity.ROUTES for c in plan['candidates'] if c['id']==cid]
    for candidate in plan['candidates']:
        candidate.pop('predicted_startup_joules',None);candidate.pop('predicted_prephase_peak_mib',None)
    plan.update(created_ns=created_ns,kind='prospective_capacity_validation',phase='predictor_validation',
        split='validation',not_for_training=True,validation_tuning=False,
        validation_entry=receipt(__file__),controller_binding=campaign['controller_binding'],
        role_reservation=reservation_ref,threshold_source_addendum=threshold_ref,
        validation_thresholds=copy.deepcopy(LIMITS),baseline_candidate_id=capacity.ROUTES[0],
        final_gain_eligible=False,resource_model_transfer='NOT_AUTHORIZED',
        service_model_transfer='Fixed shared Gamma law hypothesis; no admission or validation service refit',
        scope='Independent finite known-input admission2 V4 block; no new-input population generalization claim')
    plan['plan_sha256']=digest(plan)
    return plan


def _fixed_plan(plan):
    aliases={a:dict(family=f,supported_tasks=(['text','image','asr','ocr'] if f=='generalist_2'
        else ['ocr'] if f=='ocr' else ['text'])) for a,f in training.RESIDENTS.items()}
    if (plan['seed']!=SEED or plan['workload_rps']!=training.POLICY['workload_rps']
            or plan['runner_source']!=receipt(training.native_flow.__file__)
            or plan['profile_plan']!=receipt(training.PROFILE/'plan.json')
            or plan['context']!=receipt(training.PROFILE/'context.json')
            or plan['catalog']!=receipt(training.native_flow.BASE/'catalog.json')
            or plan['latency_reference_ms']!=dict(text=1000,image=250,asr=3000,ocr=10000)
            or plan['energy_reference_watts']!=450. or plan['memory_safe_factor']!=1.110854
            or plan['memory_budget_mib']!=49140 or plan['startup_amortization_seconds']!=3600.
            or plan['weights']!={k:1/6 for k in 'QTLPME'} or plan['aliases']!=aliases
            or 'input_generation_contract' in plan
            or plan['arrival_process']!='declared_periodic'
            or plan['queue_sample_offsets_ns']!=[round(i*6.*1e9) for i in range(101)]):
        raise ValueError('Original fixed independent constants/catalog/domain required')
    training._context(plan['profile_plan'],plan['context'])
    for candidate in plan['candidates']:
        if (candidate['routing_weights']!=training._weights(candidate['id'])
                or candidate['execution_context_sha256']!=plan['execution_context_sha256']):
            raise ValueError('Original fixed admission route weights/context required')
        capacity._generation(plan,capacity.condition(plan,candidate));_routing(plan,candidate)


def prepare(reservation_path,campaign_path,capacity_protocol_path,resource_protocol_path,
            quality_protocol_path,threshold_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    role=reservation(reservation_path);campaign=training.verify(campaign_path);threshold(threshold_path)
    bindings=[capacity.verify_protocol(capacity_protocol_path),resources.verify_protocol(resource_protocol_path),
        quality.verify_protocol(quality_protocol_path)]
    if any(p['campaign']!=receipt(campaign_path) for p in bindings):
        raise ValueError('All mode training protocols must bind the same exact T8 campaign')
    output.mkdir(parents=True,exist_ok=False)
    training.native_flow.prepare(Path(campaign['profile_plan']['path']).parent,output/'target',
        seed=SEED,duration=600.,drain=300.,warmup=0.,include_admission=True)
    folder=output/'target';(folder/'plan.json').rename(folder/'native-prepared-plan.json')
    plan=_project(read(folder/'native-prepared-plan.json'),campaign,time.perf_counter_ns(),
        receipt(reservation_path),receipt(threshold_path));save(folder/'plan.json',plan)
    refs=training._refs([sources(),receipt(reservation_path),receipt(campaign_path),receipt(threshold_path),
        receipt(capacity_protocol_path),receipt(resource_protocol_path),receipt(quality_protocol_path),
        receipt(folder/'plan.json'),receipt(folder/'native-prepared-plan.json'),role,bindings,plan])
    value=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        role_reservation=receipt(reservation_path),campaign=receipt(campaign_path),
        capacity_protocol=receipt(capacity_protocol_path),resource_protocol=receipt(resource_protocol_path),
        quality_protocol=receipt(quality_protocol_path),threshold_addendum=receipt(threshold_path),
        target_plan=receipt(folder/'plan.json'),native_preparation=receipt(folder/'native-prepared-plan.json'),
        model=role['service_hypothesis']['model'],validation_thresholds=copy.deepcopy(LIMITS),
        source_receipts=capacity.unique_refs(refs),training_outcomes_read=False,independent_outcomes_read=False,
        not_for_training=True,final_gain_eligible=False,GPU_started=False)
    value['protocol_sha256']=digest(value);save(output/'protocol.json',value)
    audit_protocol(output/'protocol.json');return value


def audit_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.protocol' or value.get('source')!=receipt(__file__)
            or value.get('validation_thresholds')!=LIMITS or value.get('not_for_training') is not True
            or value.get('final_gain_eligible') is not False or value.get('independent_outcomes_read') is not False
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Exact admission independent validation protocol required')
    capacity.unique_refs(value['source_receipts'])
    role=reservation(checked(value['role_reservation']));campaign=training.verify(checked(value['campaign']))
    threshold(checked(value['threshold_addendum']))
    bindings=[capacity.verify_protocol(checked(value['capacity_protocol'])),
        resources.verify_protocol(checked(value['resource_protocol'])),quality.verify_protocol(checked(value['quality_protocol']))]
    if (any(p['campaign']!=value['campaign'] for p in bindings)
            or value['model']!=role['service_hypothesis']['model']):
        raise ValueError('Validation model or common admission T8 training contract changed')
    plan=capacity.target_plan(checked(value['target_plan']))
    native=read(checked(value['native_preparation']))
    if (native.get('plan_sha256')!=digest({k:v for k,v in native.items() if k!='plan_sha256'})
            or plan!=_project(native,campaign,plan['created_ns'],value['role_reservation'],value['threshold_addendum'])
            or len(plan['trace'])!=1200 or not campaign['created_ns']<plan['created_ns']<value['created_ns']):
        raise ValueError('Exact prospective seed628201 public native projection required')
    flow.validate_plan(plan)
    _fixed_plan(plan)
    required=training._refs([sources(),role,bindings,plan,*[value[k] for k in
        ('role_reservation','campaign','capacity_protocol','resource_protocol','quality_protocol',
         'threshold_addendum','target_plan','native_preparation','model')]])
    if not {(r['path'],r['sha256']) for r in required}<={(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Independent protocol omitted required sources')
    return value


def _model(path):
    verification=gamma.verify_model(path)
    if (verification.get('verified') is not True or verification.get('heldout_seed') is not None
            or verification.get('training_flow_count')!=16 or verification.get('validation_sources_used')!=[]):
        raise ValueError('Frozen full16 shared Gamma model required; no admission service refit')
    return verification


def predict(plan,candidate,model,resource_document,quality_document,overrides):
    resource=resources.provider(resource_document,plan,candidate)
    q=quality.predict(quality_document,plan,candidate)
    overlaid=copy.deepcopy(candidate)
    for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):overlaid[key]=overrides[key]
    # The checked resource is used directly: generic shared construction does
    # not replace the route/controller-aware applicability check.
    end=round(plan['window_seconds']*1e9);queues={a:[] for a in plan['aliases']};future=[]
    for request in plan['trace']:
        row={k:request[k] for k in ('request_id','task','pool_id')};row['arrival_ns']=request['offset_ns']
        (queues[candidate['routes'][row['request_id']]] if row['arrival_ns']==0 else future).append(row)
    forecast=metric=None;reason=None
    try:
        forecast=FullHorizonModel(plan['aliases'],flow._provider(model,candidate['execution_context_sha256']),
            resource,memory_provider=resource,resources=candidate['resource_configuration'],
            max_requests=plan.get('max_forecast_requests',20000)).forecast(queues,0,end,
                future_arrivals=future,route_future=lambda r,p,t:candidate['routes'][r['request_id']],
                require_energy=False,require_memory=False,drain_deadline_ns=end+round(plan['drain_seconds']*1e9))
        metric=flow._score(plan,forecast,quality_document['loss_by_family_task'],
            startup_joules=overlaid['predicted_startup_joules'],historical_peak=overlaid['predicted_prephase_peak_mib'])
        mq=metric['terms']['Q'];pq=q['Q']
        if (mq is None)!=(pq is None) or (mq is not None and not math.isclose(mq,pq,rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('Route quality prediction differs from original scoring')
    except (UnknownEvidence,ValueError) as exc:reason=type(exc).__name__+':'+str(exc)
    rows=[]
    for request in plan['trace']:
        finish=forecast['finishes'].get(request['request_id']) if forecast else None
        rows.append(dict(request_id=request['request_id'],task=request['task'],pool_id=request['pool_id'],
            alias=candidate['routes'][request['request_id']],
            service_ms=(finish['finish_ns']-finish['start_ns'])/1e6 if finish else None,
            wait_ms=(finish['start_ns']-request['offset_ns'])/1e6 if finish else None,
            e2e_ms=(finish['finish_ns']-request['offset_ns'])/1e6 if finish else None,
            finish_ns=finish['finish_ns'] if finish else None))
    return dict(candidate_id=candidate['id'],requests=rows,metrics=metric,
        comparison_U=-metric['cost'] if metric and metric['absolute_utility_complete'] else None,
        queue_samples=flow._queue_samples(plan,forecast) if forecast else [],unknown_reason=reason,
        comparison_scope='Original six-term U; mode/route training Q/E/M, fixed Gamma service transfer hypothesis',
        quality_prediction=q,predicted_candidate_resource_estimates={k:overlaid[k] for k in
            ('predicted_startup_joules','predicted_prephase_peak_mib')},
        forecast_numerics='selected_completion_event_v2',numerics_source=receipt(FullHorizonModel.forecast.__code__.co_filename),
        original_event_budget=100000,not_for_training=True)


def seal(protocol_path,resource_manifest_path,quality_manifest_path,capacity_manifest_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=audit_protocol(protocol_path);plan=read(checked(protocol['target_plan']))
    rm=resources.verify_manifest(resource_manifest_path);qm=quality.verify_manifest(quality_manifest_path)
    cm=capacity.verify_manifest(capacity_manifest_path);verification=_model(checked(protocol['model']))
    if (rm['protocol']!=protocol['resource_protocol'] or qm['protocol']!=protocol['quality_protocol']
            or cm['protocol']!=protocol['capacity_protocol']):
        raise ValueError('Fitted models must bind the frozen admission training protocols')
    model=read(checked(protocol['model']));predictions=[]
    refs=training._refs([protocol,receipt(protocol_path),receipt(resource_manifest_path),receipt(quality_manifest_path),
        receipt(capacity_manifest_path),rm,qm,cm,sources(),flow._sources(model)])
    cutoff=time.perf_counter_ns()
    for candidate in plan['candidates']:
        cid=candidate['id'];rd=read(checked(rm['full']['route_resources'][cid]));qd=read(checked(qm['full']['route_quality'][cid]))
        predictions.append(predict(plan,candidate,model,rd,qd,rm['full']['candidate_overrides'][cid]))
    capacity.unique_refs(refs);audit_protocol(protocol_path)
    output.mkdir(parents=True,exist_ok=False)
    capacity.seal(protocol['target_plan']['path'],protocol['capacity_protocol']['path'],output/'capacity-seal.json')
    value=dict(schema='gsdeploy.sealed_native_flow_predictions.v1',entry_schema=SCHEMA+'.seal',
        created_ns=time.perf_counter_ns(),information_cutoff_ns=cutoff,
        sources=dict(plan=protocol['target_plan'],model=protocol['model'],resources=receipt(resource_manifest_path),quality=receipt(quality_manifest_path)),
        training_sources=flow._sources(model),resource_sources=rm['source_receipts'],quality_sources=qm['source_receipts'],
        execution_sources=training._refs(plan),predictor_sources=sources(),tool_source=receipt(__file__),
        metric_source=receipt(Path(__file__).with_name('flow_metrics.py')),candidates=predictions,
        cohort_sha256=flow.cohort_sha(plan),gate_protocol=LIMITS,phase='predictor_validation',
        not_for_training=True,final_gain_eligible=False,GPU_started=False,actual_future_trace_used=False)
    save(output/'predictions.json',value)
    audit=dict(schema=SCHEMA+'.seal_audit',protocol=receipt(protocol_path),predictions=receipt(output/'predictions.json'),
        resource_manifest=receipt(resource_manifest_path),quality_manifest=receipt(quality_manifest_path),
        capacity_manifest=receipt(capacity_manifest_path),capacity_seal=receipt(output/'capacity-seal.json'),
        model_verification=verification,source_receipts=capacity.unique_refs(refs),source=receipt(__file__),
        not_for_training=True,final_gain_eligible=False,GPU_started=False)
    audit['audit_sha256']=digest(audit);save(output/'seal-audit.json',audit)
    return value


def check_sealed(folder):
    folder=Path(folder);audit=read(folder/'seal-audit.json')
    if (audit.get('schema')!=SCHEMA+'.seal_audit' or audit.get('source')!=receipt(__file__)
            or audit.get('predictions')!=receipt(folder/'predictions.json')
            or audit.get('audit_sha256')!=digest({k:v for k,v in audit.items() if k!='audit_sha256'})):
        raise ValueError('Exact independent admission prediction/source seal required')
    capacity.unique_refs(training._refs(audit));protocol=audit_protocol(checked(audit['protocol']))
    rm=resources.verify_manifest(checked(audit['resource_manifest']));qm=quality.verify_manifest(checked(audit['quality_manifest']))
    cm=capacity.verify_manifest(checked(audit['capacity_manifest']));cs=capacity.verify_seal(checked(audit['capacity_seal']))
    sealed=read(folder/'predictions.json');plan=read(checked(protocol['target_plan']))
    if (sealed.get('entry_schema')!=SCHEMA+'.seal' or sealed['sources']!=dict(plan=protocol['target_plan'],model=protocol['model'],
            resources=audit['resource_manifest'],quality=audit['quality_manifest'])
            or rm['protocol']!=protocol['resource_protocol'] or qm['protocol']!=protocol['quality_protocol']
            or cm['protocol']!=protocol['capacity_protocol'] or cs['target_plan']!=protocol['target_plan']
            or not protocol['created_ns']<=sealed['information_cutoff_ns']<=sealed['created_ns']
            or cs['created_ns']>sealed['created_ns'] or audit['model_verification']!=_model(checked(protocol['model']))
            or tuple(p['candidate_id'] for p in sealed['candidates'])!=capacity.ROUTES
            or sealed.get('not_for_training') is not True or sealed.get('final_gain_eligible') is not False):
        raise ValueError('Independent admission seal differs from bound training/public inputs')
    capacity.unique_refs(training._refs(sealed))
    # Verify numerical forecasts too; a rehashed audit cannot invent predictions.
    model=read(checked(protocol['model']))
    for candidate,predicted in zip(plan['candidates'],sealed['candidates']):
        cid=candidate['id']
        expected=predict(plan,candidate,model,read(checked(rm['full']['route_resources'][cid])),
            read(checked(qm['full']['route_quality'][cid])),rm['full']['candidate_overrides'][cid])
        if expected!=predicted:raise ValueError('Sealed admission prediction differs from fresh numerical rollout')
    return audit


def actual_evidence(raw_path,protocol,sealed):
    """Independent native semantic audit, no training or shared mapping call."""
    raw_path=Path(raw_path).resolve()
    source=receipt(raw_path);scored_ref=receipt(raw_path.with_name('scored-raw.json'))
    invocation_ref=receipt(raw_path.with_name('invocation.json'));plan_ref=copy.deepcopy(protocol['target_plan'])
    entry_sources=sources()
    raw=read(checked(source));plan=capacity.target_plan(checked(plan_ref));_fixed_plan(plan)
    if (any(k in raw for k in ('native_raw','derived_from','derivation','collection_role'))
            or raw.get('plan')!=protocol['target_plan'] or raw.get('split') not in (None,'validation')
            or type(sealed.get('created_ns')) is not int or sealed['created_ns']<plan['created_ns']
            or type(raw.get('startup_begin_ns')) is not int or raw['startup_begin_ns']<=sealed['created_ns']):
        raise ValueError('Original independent native raw must start after complete forecast seal')
    candidate=next((c for c in plan['candidates'] if c['id']==raw.get('candidate_id')),None)
    if candidate is None:raise ValueError('Foreign independent admission candidate')
    identity=capacity.controller_identity(plan,candidate)
    objective={k:plan[k] for k in ('workload_rps','latency_reference_ms','weights','memory_safe_factor',
        'memory_budget_mib','energy_reference_watts','startup_amortization_seconds')}
    if raw.get('effective_objective')!=objective:
        raise ValueError('Actual independent native objective differs from original six terms')
    if read(checked(invocation_ref))!=dict(plan=protocol['target_plan'],candidate_id=candidate['id'],runner=plan['runner_source']):
        raise ValueError('Independent invocation/runner/plan mismatch')
    audit=mechanism_audit.audit(raw_path)
    profile,context=training._context(plan['profile_plan'],plan['context'])
    if ([(r['model'],r['task'],r['pool_id']) for r in raw['priming_request_records']]!=training._priming(profile)
            or raw['warmup_request_records']):raise ValueError('Exact complete largest-three priming/warmup0 required')
    ready=raw['ready']
    for alias,info in ready.items():
        if (info.get('clock')!='time.perf_counter_ns' or
                info.get('spec_sha256')!=context['worker_bindings'][training.RESIDENTS[alias]]['spec_sha256']):
            raise ValueError('Independent native READY clock/spec differs from worker context')
    samples,gap=training._telemetry(raw)
    startup,left,deadline,right=(raw[k] for k in ('startup_begin_ns','phase_start_ns','deadline_ns','observation_end_ns'))
    if right>deadline+round(plan['drain_seconds']*1e9):
        raise ValueError('Independent observation exceeds original bounded drain')
    by_id={r['request_id']:r for r in raw['request_records']};scope='admission_independent:'+source['sha256']
    semantic=[]
    for offered in plan['trace']:
        row=by_id[offered['request_id']]
        if row.get('split') not in (None,'validation'):raise ValueError('Training/final rows cannot become independent validation')
        item=flow_profiles._request(row,ready,context,split='validation',scope_id=scope,left=left,right=right)
        item.update(not_for_training=True,controller_sha256=identity);semantic.append(item)
    if any(r['parent_receipt_ns']>deadline+round(plan['drain_seconds']*1e9) for r in semantic):
        raise ValueError('Independent parent receipt exceeds bounded drain')
    history=[]
    for key in ('priming_request_records','warmup_request_records'):
        for row in raw[key]:
            if row.get('split') not in (None,'excluded_prephase_history','validation'):
                raise ValueError('Training/final history cannot become independent validation')
            item=flow_profiles._request(row,ready,context,split='excluded_prephase_history',scope_id=scope,left=startup,right=left)
            item['not_for_training']=True;history.append(item)
    for alias in training.RESIDENTS:
        rows=[r for r in semantic if r['alias']==alias]
        if rows!=sorted(rows,key=lambda r:r['handler_start_ns']):raise ValueError('Independent offered FIFO order changed')
        ordered=sorted((r for r in semantic+history if r['alias']==alias),key=lambda r:r['handler_start_ns'])
        if any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('Independent actual handlers overlap on same alias')
    scored=read(checked(scored_ref))
    if (scored.get('native_raw')!=source
            or {k:v for k,v in scored.items() if k not in ('native_raw','quality_evidence')}!=raw
            or set(scored)!=set(raw)|{'native_raw','quality_evidence'}):
        raise ValueError('Scoring changed immutable native observations')
    expected=[plan['catalog'],receipt(training.native_flow.__file__),
        receipt(training_run_audit.score_result.__code__.co_filename),receipt(training_run_audit.sample_totals.__code__.co_filename)]
    if scored['quality_evidence'].get('source_receipts')!=expected:
        raise ValueError('Exact frozen independent quality scorer/catalog sources required')
    quality_audit=training_run_audit.verify_quality_annotation(scored,plan,read(checked(plan['catalog'])))
    actual_score=flow.observed(plan,candidate,scored)
    stability=native_stability(plan,candidate,raw,read(checked(plan['catalog'])))
    observations=[dict(request_id=r['native_request_id'],alias=r['alias'],family=r['family'],task=r['task'],pool_id=r['pool_id'],
        worker_service_ms=(r['handler_end_ns']-r['handler_start_ns'])/1e6) for r in semantic]
    refs=capacity.unique_refs(training._refs([source,scored_ref,invocation_ref,plan_ref,protocol,plan,context,entry_sources,
        scored['quality_evidence']]))
    # All return receipts are the ones captured before parsing, never fresh
    # replacements after a concurrent file change.
    for ref in refs:checked(ref)
    return dict(candidate_id=candidate['id'],native_raw=source,scored_raw=scored_ref,source_receipts=refs,
        controller_sha256=identity,semantic_validation_trace=dict(split='validation',not_for_training=True,requests=semantic),
        resource_validation_evidence=dict(split='validation',not_for_training=True,sample_count=len(samples),max_gap_ns=gap),
        service_observations=observations,quality_audit=quality_audit,actual_metrics=actual_score['metrics'],mechanism_audit=audit,
        activation=activation_source.activation(audit,candidate),original_stability=stability,
        requested=len(plan['trace']),known=len(semantic),not_for_training=True,final_gain_eligible=False,GPU_started=False)


def report(sealed_dir,raw_paths,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    audit=check_sealed(sealed_dir);protocol=audit_protocol(audit['protocol']['path'])
    sealed=read(Path(sealed_dir)/'predictions.json');plan=read(checked(protocol['target_plan']))
    evidence=[actual_evidence(p,protocol,sealed) for p in raw_paths]
    if len({e['candidate_id'] for e in evidence})!=len(evidence):raise ValueError('Duplicate independent candidate')
    cs=capacity.verify_seal(audit['capacity_seal']['path']);predicted={p['candidate_id']:p['prediction'] for p in cs['predictions']}
    qualification=[]
    for item in evidence:
        cid=item['candidate_id'];prediction=predicted[cid]
        qualification.append(qualify_capacity(prediction,item))
    value=flow.report(sealed_dir,[e['scored_raw']['path'] for e in evidence],output)
    # Preserve the frozen report as written; qualification is a separately bound
    # addendum and never rewrites native annotations or upgrades the old report.
    gate=flow_gate.assess(plan,value)
    save(output/'gate.json',{**gate,'report':receipt(output/'report.json'),'protocol':audit['protocol'],
        'final_gain_eligible':False,'not_for_training':True})
    save(output/'capacity-qualification.json',dict(schema=SCHEMA+'.capacity_qualification',
        capacity_seal=audit['capacity_seal'],rows=qualification,not_for_training=True,
        independent_units=1,physical_capacity_bound=False,final_gain_eligible=False,GPU_started=False))
    for item in evidence:capacity.unique_refs(item['source_receipts'])
    check_sealed(sealed_dir)
    report_audit=dict(schema=SCHEMA+'.report_audit',report=receipt(output/'report.json'),
        gate=receipt(output/'gate.json'),capacity_qualification=receipt(output/'capacity-qualification.json'),
        seal_audit=receipt(Path(sealed_dir)/'seal-audit.json'),source=receipt(__file__),
        evidence=evidence,not_for_training=True,final_gain_eligible=False,GPU_started=False)
    report_audit['audit_sha256']=digest(report_audit);save(output/'validation-audit.json',report_audit)
    return value


def qualify_capacity(prediction,evidence):
    """Join already strict independent evidence with frozen mean/rho checks."""
    numerical_check=capacity.assess_independent_service(prediction,evidence['service_observations'])
    complete=(evidence['known']==evidence['requested']==1200
        and evidence['quality_audit']['count']==1200 and evidence['quality_audit']['independently_recomputed'] is True
        and evidence['actual_metrics']['six_term_comparison_complete'] is True)
    applicability=numerical_check['mean_service_numerical_threshold_passed'] if complete else None
    admitted=(applicability and prediction['rho_threshold_satisfied'] and evidence['original_stability']['original_stability_passed']) if applicability is not None else None
    return dict(candidate_id=evidence['candidate_id'],mean_service_checks=numerical_check['mean_service_checks'],
        statistical_service_applicability=applicability,original_rho_threshold_satisfied=prediction['rho_threshold_satisfied'],
        original_stability=evidence['original_stability'],statistical_original_rho_planning_admission_passed=admitted,
        complete_independent_evidence=complete,global_permit_capacity_qualified=None,
        original_measured_profile_resolver_admission=None,final_gain_eligible=False,not_for_training=True)


def verify_report(report_path):
    """Recompute observations, old scoring/gate and qualification without fitting."""
    report_path=Path(report_path).resolve();folder=report_path.parent
    audit=read(folder/'validation-audit.json')
    if (audit.get('schema')!=SCHEMA+'.report_audit' or audit.get('source')!=receipt(__file__)
            or audit.get('report')!=receipt(report_path) or audit.get('not_for_training') is not True
            or audit.get('final_gain_eligible') is not False
            or audit.get('audit_sha256')!=digest({k:v for k,v in audit.items() if k!='audit_sha256'})):
        raise ValueError('Exact independent admission report audit required')
    capacity.unique_refs(training._refs(audit))
    sealed_dir=Path(checked(audit['seal_audit'])).parent
    seal_audit=check_sealed(sealed_dir);protocol=audit_protocol(seal_audit['protocol']['path'])
    sealed=read(sealed_dir/'predictions.json');plan=read(checked(protocol['target_plan']))
    evidence=[actual_evidence(checked(item['native_raw']),protocol,sealed) for item in audit['evidence']]
    if evidence!=audit['evidence']:raise ValueError('Independent report evidence differs from native recomputation')
    with tempfile.TemporaryDirectory(prefix='admission-validation-report-audit-') as directory:
        expected=flow.report(sealed_dir,[item['scored_raw']['path'] for item in evidence],Path(directory)/'report')
    observed=read(report_path)
    if observed!=expected:raise ValueError('Independent report differs from original scoring recomputation')
    expected_gate={**flow_gate.assess(plan,expected),'report':receipt(report_path),'protocol':seal_audit['protocol'],
        'final_gain_eligible':False,'not_for_training':True}
    if read(checked(audit['gate']))!=expected_gate:
        raise ValueError('Independent frozen gate differs from recomputation')
    cs=capacity.verify_seal(checked(seal_audit['capacity_seal']));predictions={p['candidate_id']:p['prediction'] for p in cs['predictions']}
    expected_qualification=dict(schema=SCHEMA+'.capacity_qualification',capacity_seal=seal_audit['capacity_seal'],
        rows=[qualify_capacity(predictions[item['candidate_id']],item) for item in evidence],
        not_for_training=True,independent_units=1,physical_capacity_bound=False,final_gain_eligible=False,GPU_started=False)
    if read(checked(audit['capacity_qualification']))!=expected_qualification:
        raise ValueError('Independent capacity qualification differs from recomputation')
    return dict(schema=SCHEMA+'.report_verification',report=receipt(report_path),verified=True,
        numerical_and_native_recomputed=True,source=receipt(__file__),not_for_training=True,
        final_gain_eligible=False,GPU_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for key in ('reservation','campaign','capacity-protocol','resource-protocol','quality-protocol','threshold-addendum','output'):
        a.add_argument('--'+key,required=True)
    a=sub.add_parser('seal')
    for key in ('protocol','resource-manifest','quality-manifest','capacity-manifest','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True);a.add_argument('--raw',action='append',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--sealed-dir',required=True)
    a=sub.add_parser('verify-report');a.add_argument('--report',required=True)
    args=p.parse_args()
    if args.op=='prepare':result=prepare(args.reservation,args.campaign,args.capacity_protocol,args.resource_protocol,args.quality_protocol,args.threshold_addendum,args.output)
    elif args.op=='seal':result=seal(args.protocol,args.resource_manifest,args.quality_manifest,args.capacity_manifest,args.output)
    elif args.op=='report':result=report(args.sealed_dir,args.raw,args.output)
    elif args.op=='verify':result=check_sealed(args.sealed_dir)
    else:result=verify_report(args.report)
    print(json.dumps(dict(schema=result['schema'],GPU_started=False)))


if __name__=='__main__':main()
