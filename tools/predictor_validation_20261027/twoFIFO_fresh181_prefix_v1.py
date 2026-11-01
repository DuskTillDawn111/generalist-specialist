"""Owned pre-cohort prefix and durable prediction barrier for fresh181.

Only actual prior priming rows enter the settled seed. Future request records
and target response labels are not inputs. A source-bound predeclared forecast
is a validation hypothesis, not a statement of accuracy or deployment fitness.
"""
from __future__ import annotations
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import time

from . import twoFIFO_matching_execute_v1 as matching
from . import twoFIFO_settled_history_initial_v1 as settled
from .natural_macro_resources_candidate_v1 import priming_identity
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import strict_references
from .twoFIFO_training_manifest_v1 import require,digest,OWNERS

SCHEMA='gsdeploy.twoFIFO_fresh181_precohort_prefix.v1'
GATE_SCHEMA=SCHEMA+'.prediction_gate'
FORECAST_SCHEMA=SCHEMA+'.predeclared_forecast'
SEAL_SCHEMA=SCHEMA+'.prediction_seal'


def verify(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and receipt(ref['path'])==ref,
            'Immutable fresh prefix or forecast evidence changed')
    return ref


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return receipt(path)


def producers(plan,snapshot):
    return {owner:plan['execution_id']+':'+owner+':stdout:pid'+str(snapshot['ready'][owner]['pid'])
            for owner in OWNERS}


def inspect_native_ownership(snapshot,evidence,*,collector_pid=None):
    """Recheck stored live ancestry; no process or GPU probing occurs here."""
    from .device_progress_actual_review_v3 import valid_live_ownership
    require(set(evidence)==set(OWNERS),'Every resident needs actual owned READY ancestry')
    collectors=set()
    for owner in OWNERS:
        values=evidence[owner]
        require(type(values) is list and len(values)==1,'One pinned actual READY owner required')
        item=values[0];pid=item.get('collector_pid')
        require(type(pid) is int and pid>0 and (collector_pid is None or pid==collector_pid)
            and item.get('alias')==owner and item.get('pid')==snapshot['ready'][owner]['pid']
            and valid_live_ownership(item,pid,pid),'Actual READY pid/launcher/collector ancestry required')
        collectors.add(pid)
    require(len(collectors)==1,'One collector lifetime for all actual residents required')
    return next(iter(collectors))


def derive(plan,snapshot,epoch_ns):
    """Pure strict rederivation; cannot fill missing clocks from a final raw."""
    require(type(epoch_ns) is int and type(snapshot) is dict
        and type(snapshot.get('snapshot_time_ns')) is int and snapshot['snapshot_time_ns']<epoch_ns,
        'Actual immutable prefix must precede the first-arrival origin')
    rows=snapshot.get('request_records')
    require(type(rows) is list and [r.get('request_id') for r in rows]==[r['request_id'] for r in plan['priming']],
        'Only all eighteen prior priming records, no future cohort rows, may enter the prefix')
    require(set(snapshot.get('ready',{}))==set(OWNERS),'Complete actual worker roster required')
    for row,planned in zip(rows,plan['priming']):
        require(row.get('model')==planned['owner'] and row.get('task')==planned['task']
            and row.get('pool_id')==planned['pool_id'],'Actual priming public semantics differ')
    context=checked(verify(plan['context']))
    require(all(snapshot['ready'][owner].get('spec_sha256')==
        context['worker_bindings'][owner.split('#')[0]]['spec_sha256'] for owner in OWNERS),
        'Actual READY specification must equal the authenticated matching worker specification')
    semantics=matching.planned_priming_semantics(plan,context)
    actual=priming_identity(dict(priming_request_records=rows),context)
    require(actual==semantics['sha256'],'Actual ordered priming contents differ from the public plan')
    roster=producers(plan,snapshot)
    seed=settled.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],
        forecast_origin_ns=epoch_ns,owner_producers=roster)
    # This records source binding, not native timing/fit accuracy qualification.
    return dict(seed=seed,owner_producers=roster,actual_priming_semantics_sha256=actual,
        priming_request_ids=[r['request_id'] for r in rows])


def make_prefix(plan_ref,plan,manifest_ref,snapshot,epoch_ns,ownership_evidence,output,*,backend):
    verify(plan_ref);verify(manifest_ref)
    require(backend in ('native','CPU_TEST_ONLY'),'Explicit observed backend kind required')
    require(set(ownership_evidence)==set(OWNERS) and all(ownership_evidence[o] for o in OWNERS),
        'Actual owned READY evidence is required for every resident')
    if backend=='native':inspect_native_ownership(snapshot,ownership_evidence)
    value=derive(plan,snapshot,epoch_ns)
    return save(output,dict(schema=SCHEMA,status='OWNED_SETTLED_PRECOHORT_PREFIX_NOT_ACCURACY_QUALIFICATION',
        plan=plan_ref,manifest=manifest_ref,context=plan['context'],execution_id=plan['execution_id'],
        candidate_id=plan['candidate_id'],backend=backend,prefix_snapshot=copy.deepcopy(snapshot),
        forecast_origin_ns=epoch_ns,public_calendar_sha256=plan['cases'][0]['public_arrivals_sha256'],
        ready_ownership_evidence=copy.deepcopy(ownership_evidence),**value,
        prefix_source_binding='ACTUAL_OWNED_RUNTIME_SNAPSHOT_BEFORE_ANY_COHORT_SEND',
        target_future_labels_read=False,current_cohort_contains_prior_labels=False,
        physical_progress_observed=False,independent_accuracy_qualified=False,
        source_authenticated_by_independent_execution_review=False,
        resource_horizon_support='UNKNOWN_MATCHING_TRAINING10_PLUS60_DOES_NOT_CERTIFY90_PLUS120',
        source_bindings=plan['sources']))


def audit_prefix(prefix_ref,plan_ref,plan,manifest_ref,*,allow_cpu_fixture=False):
    doc=checked(verify(prefix_ref));verify(plan_ref);verify(manifest_ref)
    require(doc.get('schema')==SCHEMA and doc.get('status')=='OWNED_SETTLED_PRECOHORT_PREFIX_NOT_ACCURACY_QUALIFICATION'
        and doc.get('plan')==plan_ref and doc.get('manifest')==manifest_ref
        and doc.get('context')==plan['context'] and doc.get('execution_id')==plan['execution_id']
        and doc.get('candidate_id')==plan['candidate_id']
        and doc.get('backend') in (('native','CPU_TEST_ONLY') if allow_cpu_fixture else ('native',))
        and doc.get('public_calendar_sha256')==plan['cases'][0]['public_arrivals_sha256']
        and doc.get('target_future_labels_read') is False
        and doc.get('current_cohort_contains_prior_labels') is False
        and doc.get('physical_progress_observed') is False
        and doc.get('independent_accuracy_qualified') is False
        and doc.get('source_authenticated_by_independent_execution_review') is False
        and doc.get('prefix_source_binding')=='ACTUAL_OWNED_RUNTIME_SNAPSHOT_BEFORE_ANY_COHORT_SEND'
        and doc.get('resource_horizon_support')=='UNKNOWN_MATCHING_TRAINING10_PLUS60_DOES_NOT_CERTIFY90_PLUS120'
        and doc.get('source_bindings')==plan['sources'], 'Exact fresh owned-prefix role/source binding required')
    actual=derive(plan,doc['prefix_snapshot'],doc['forecast_origin_ns'])
    require(all(doc.get(k)==v for k,v in actual.items()),'Settled seed and actual priming must rederive from prefix alone')
    require(set(doc.get('ready_ownership_evidence',{}))==set(OWNERS)
        and all(doc['ready_ownership_evidence'][o] for o in OWNERS),'Complete owned prefix provenance required')
    if doc['backend']=='native':inspect_native_ownership(doc['prefix_snapshot'],doc['ready_ownership_evidence'])
    for ref in plan['sources']:verify(ref)
    return doc


def _gate_document(prefix_ref,plan_ref,plan,manifest_ref,prefix):
    epoch=prefix['forecast_origin_ns'];contract=checked(verify(plan['public_contract']))
    objective=checked(verify(plan['objective']))
    require(objective['weights']==contract['weights']=={k:1/6 for k in 'QTLPME'}
        and objective['stability_policy']=='existing_thresholds_unchanged',
        'The immutable equal-six objective and original stability are required')
    return dict(schema=GATE_SCHEMA,plan=plan_ref,manifest=manifest_ref,prefix=prefix_ref,
        candidate_id=plan['candidate_id'],execution_id=plan['execution_id'],forecast_origin_ns=epoch,
        measurement_end_ns=epoch+plan['measurement_ns'],fixed_drain_deadline_ns=epoch+plan['measurement_ns']+plan['drain_ns'],
        public_requests=[dict(request_id=a['request_id'],owner=a['owner'],task=a['task'],pool_id=a['pool_id'],
            arrival_ns=epoch+a['arrival_ns']) for a in plan['cases'][0]['arrivals']],
        model_sources=plan['forecast_barrier']['model_sources'],forecast_generator_sources=plan['forecast_barrier']['generator_sources'],
        original_objective=plan['objective'],original_public_contract=plan['public_contract'],
        six_term_weights=contract['weights'],stability_policy=objective['stability_policy'],
        expected_seal_path=plan['forecast_barrier']['seal_path'],target_future_labels_read=False,
        gate_has_no_target_raw_or_future_response_input=True,resources_and_fullU_qualified=False)


def gate_request(prefix_ref,plan_ref,plan,manifest_ref,output,*,allow_cpu_fixture=False):
    prefix=audit_prefix(prefix_ref,plan_ref,plan,manifest_ref,allow_cpu_fixture=allow_cpu_fixture)
    return save(output,_gate_document(prefix_ref,plan_ref,plan,manifest_ref,prefix))


def audit_gate(gate_ref,*,allow_cpu_fixture=False):
    gate=checked(verify(gate_ref));plan=checked(verify(gate['plan']))
    value=audit_prefix(gate['prefix'],gate['plan'],plan,gate['manifest'],allow_cpu_fixture=allow_cpu_fixture)
    require(gate==_gate_document(gate['prefix'],gate['plan'],plan,gate['manifest'],value),
        'Exact prefix-bound complete calendar, objective and prediction gate required')
    require(str(Path(gate_ref['path']).resolve())==plan['forecast_barrier']['gate_request_path'],
        'Canonical owned prediction gate required')
    return gate


def validate_forecast(document,gate):
    require(type(document) is dict and document.get('schema')==FORECAST_SCHEMA
        and set(document)=={'schema','validation_labels_read','target_future_labels_read','gate_request_sha256',
            'public_requests','model_sources','forecast_generator_sources','resource_horizon_qualified','full_U_qualified','draws'}
        and document.get('validation_labels_read') is False and document.get('target_future_labels_read') is False
        and document.get('gate_request_sha256')==digest(gate)
        and document.get('public_requests')==gate['public_requests']
        and document.get('model_sources')==gate['model_sources']
        and document.get('forecast_generator_sources')==gate['forecast_generator_sources'],
        'Only the complete predeclared source-bound forecast may pass the pre-cohort gate')
    draws=document.get('draws');require(type(draws) is list and draws,'Explicit fixed forecast draws required')
    weights=[];ids=[r['request_id'] for r in gate['public_requests']]
    require(len({d.get('draw_id') for d in draws})==len(draws),'Distinct fixed draw IDs required')
    for draw in draws:
        require(type(draw) is dict and set(draw)=={'draw_id','weight','status','requests'}
            and type(draw['draw_id']) is str and draw['draw_id'],'Closed fixed draw schema required')
        weight=draw.get('weight');require(type(weight) is dict and set(weight)=={'numerator','denominator'}
            and all(type(v) is int for v in weight.values()) and weight['numerator']>0 and weight['denominator']>0,
            'Exact positive fixed draw probability required')
        weights.append(Fraction(weight['numerator'],weight['denominator']))
        require(draw.get('status') in ('UNKNOWN_WORLD','GENERATED_WORLD')
            and type(draw.get('requests')) is list and [r.get('request_id') for r in draw['requests']]==ids,
            'Every forecast world must retain the exact complete offered cohort')
        if draw['status']=='GENERATED_WORLD':
            require(gate['model_sources'] and gate['forecast_generator_sources'],
                'Numeric forecast requires predeclared model and generator source receipts')
            require(any(type(r.get('completion_time_ns')) is int for r in draw['requests']),
                'An all-UNKNOWN completion vector cannot be relabeled GENERATED_WORLD')
        for pred,public in zip(draw['requests'],gate['public_requests']):
            require(type(pred) is dict and set(pred)=={'request_id','completion_time_ns'},
                'Forecast request projection contains only its public ID and completion; extra semantics rejected')
            if draw['status']=='UNKNOWN_WORLD':
                require(pred.get('completion_time_ns') is None,'UNKNOWN mass cannot supply a numeric completion')
            else:
                at=pred.get('completion_time_ns')
                require(at is None or type(at) is int and public['arrival_ns']<=at<=gate['fixed_drain_deadline_ns'],
                    'Predicted completion must retain incomplete requests and the original fixed horizon')
    require(sum(weights)==1,'Original full forecast probability mass required')
    require(document.get('resource_horizon_qualified') is False and document.get('full_U_qualified') is False,
        'This bridge cannot certify unsupported90+120 resources or full utility')


def seal_prediction(gate_request_path,prediction_path,output,*,clock_ns=time.perf_counter_ns):
    """External producer API; call after forecasting, before the first arrival.

    Reads only public gate/prefix and the forecast artifact; no target raw path.
    Sealing is durable evidence of order, not independent proof of generator correctness.
    """
    gate_ref=receipt(gate_request_path)
    cpu_fixture=clock_ns is not time.perf_counter_ns
    gate=audit_gate(gate_ref,allow_cpu_fixture=cpu_fixture)
    require(gate.get('schema')==GATE_SCHEMA and str(Path(output).resolve())==gate['expected_seal_path'],
        'Canonical predeclared prediction seal path required')
    pred_ref=receipt(prediction_path);prediction=checked(verify(pred_ref));validate_forecast(prediction,gate)
    require(cpu_fixture or any(d['status']=='GENERATED_WORLD' for d in prediction['draws']),
        'An all-UNKNOWN interface fixture cannot consume native validation requests')
    for ref in strict_references([gate['plan'],gate['manifest'],gate['prefix'],gate['model_sources'],gate['forecast_generator_sources']]):verify(ref)
    now=clock_ns();require(type(now) is int and now<gate['forecast_origin_ns'], 'Prediction must seal before the first-arrival origin')
    result=save(output,dict(schema=SEAL_SCHEMA,gate_request=gate_ref,prediction=pred_ref,
        plan=gate['plan'],prefix=gate['prefix'],execution_id=gate['execution_id'],candidate_id=gate['candidate_id'],
        forecast_origin_ns=gate['forecast_origin_ns'],seal_begin_ns=now,
        target_future_labels_read=False,all_offered_requests_and_probability_mass_retained=True,
        generated_forecast_correctness_independently_verified=False,independent_accuracy_qualified=False,
        CPU_TEST_ONLY=cpu_fixture))
    require(clock_ns()<gate['forecast_origin_ns'],'Prediction persistence crossed the first-arrival boundary')
    return result


def audit_seal(seal_ref,gate_ref,*,now_ns,allow_cpu_fixture=False):
    gate=audit_gate(gate_ref,allow_cpu_fixture=allow_cpu_fixture);seal=checked(verify(seal_ref))
    require(seal.get('schema')==SEAL_SCHEMA and seal.get('gate_request')==gate_ref
        and seal.get('plan')==gate['plan'] and seal.get('prefix')==gate['prefix']
        and seal.get('execution_id')==gate['execution_id'] and seal.get('candidate_id')==gate['candidate_id']
        and seal.get('forecast_origin_ns')==gate['forecast_origin_ns']
        and type(seal.get('seal_begin_ns')) is int and seal['seal_begin_ns']<=now_ns<gate['forecast_origin_ns']
        and seal.get('CPU_TEST_ONLY') in ((True,False) if allow_cpu_fixture else (False,))
        and seal.get('target_future_labels_read') is False,
        'Durable prediction seal must be accepted before the exact first-arrival origin')
    verify(gate['prefix']);verify(gate['plan']);verify(gate['manifest'])
    forecast=checked(verify(seal['prediction']));validate_forecast(forecast,gate)
    require(allow_cpu_fixture or any(d['status']=='GENERATED_WORLD' for d in forecast['draws']),
        'Native validation requires an actual predeclared generated forecast, not an all-UNKNOWN interface test')
    for ref in gate['model_sources']+gate['forecast_generator_sources']:verify(ref)
    return seal
