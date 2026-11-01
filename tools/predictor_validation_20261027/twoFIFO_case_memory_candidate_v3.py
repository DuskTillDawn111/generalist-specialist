"""Four-column case sampled-memory hypothesis, isolated from production paths.

Native fitting is gated on all twelve reviewed attempts. Actual-trajectory fits
are explanatory only. Generated features accept donor sidecars/worlds, never
heldout future execution rows or memory targets. P0 is a fixed cumulative offset.
"""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import numpy as np

from . import twoFIFO_resource_provider_v3 as resource
from . import twoFIFO_case_memory_feature_audit_v1 as explanatory
from . import twoFIFO_prospective_memory_v1 as generated
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_case_sampled_memory_candidate.v3'
ROW_SCHEMA=SCHEMA+'.actual_trajectory_row'
MODEL_SCHEMA=SCHEMA+'.fixed_offset_model'
COLUMNS=('intercept','sum_log1p_case_H','C','R_over_horizon')
require=resource.require


def feature_vector(case):
    demands=case['case_shape_max']
    if any(not resource.legacy.finite(demands.get(o)) for o in explanatory.TEXT_OWNERS):return None
    if not resource.legacy.finite(case.get('C')) or not resource.legacy.finite(case.get('R_over_horizon')):return None
    return [1.,sum(math.log1p(demands[o]) for o in explanatory.TEXT_OWNERS),float(case['C']),float(case['R_over_horizon'])]


def _public_keys(requests):return sorted({(r['owner'],r['task'],r['pool_id']) for r in requests})


def support_context(prior_descriptor,case_shape_max,public_keys):
    """Predeclared task/shape domain, separate from four fitted columns."""
    envelope={o:0 for o in explanatory.TEXT_OWNERS}
    for row in prior_descriptor['prior_processed_shape_highwater']:
        owner=row['owner']
        if owner not in envelope:continue
        b,s,index=(row.get(k) for k in ('max_input_batch_size','max_input_sequence_length','max_forward_index'))
        if any(type(v) is not int or v<0 for v in (b,s,index)):envelope[owner]=None
        elif envelope[owner] is not None:envelope[owner]=max(envelope[owner],b*(s+index))
    current_known=all(resource.legacy.finite(case_shape_max.get(o)) for o in explanatory.TEXT_OWNERS)
    departures={o:(case_shape_max[o]>envelope[o] if current_known and envelope[o] is not None else None) for o in explanatory.TEXT_OWNERS}
    task_state=dict(owner_tasks=[list(k) for k in sorted({(k[0],k[1]) for k in public_keys})],
        owners_with_current_text_forward=[o for o in explanatory.TEXT_OWNERS if resource.legacy.finite(case_shape_max.get(o)) and case_shape_max[o]>0],
        owner_prior_envelope_departures=departures)
    return dict(prior_owner_shape_envelope=envelope,task_state=task_state,
        task_state_key=resource.digest(task_state),known=current_known and all(v is not None for v in envelope.values()))


def _context_known(context):
    if not isinstance(context,dict) or context.get('known') is not True:return False
    if set(context)!={'prior_owner_shape_envelope','task_state','task_state_key','known'}:return False
    envelope=context['prior_owner_shape_envelope'];state=context['task_state']
    if not isinstance(envelope,dict) or set(envelope)!=set(explanatory.TEXT_OWNERS):return False
    if not all(resource.legacy.finite(v) for v in envelope.values()):return False
    if not isinstance(state,dict) or set(state)!={'owner_tasks','owners_with_current_text_forward','owner_prior_envelope_departures'}:return False
    tasks=state['owner_tasks'];forwards=state['owners_with_current_text_forward'];departures=state['owner_prior_envelope_departures']
    if not isinstance(tasks,list) or any(not isinstance(k,list) or len(k)!=2 or any(type(v) is not str for v in k) for k in tasks):return False
    if tasks!=[list(k) for k in sorted({tuple(k) for k in tasks})]:return False
    if not isinstance(forwards,list) or forwards!=[o for o in explanatory.TEXT_OWNERS if o in forwards]:return False
    if not isinstance(departures,dict):return False
    if set(state['owner_prior_envelope_departures'])!=set(explanatory.TEXT_OWNERS):return False
    if any(type(v) is not bool for v in state['owner_prior_envelope_departures'].values()):return False
    return resource.digest(state)==context['task_state_key']


def actual_rows_from_verified(raw,plan,context,native_table):
    """Explanatory feature adapter; caller authenticates native intake first."""
    feature_doc=explanatory.feature_rows(raw);feature_rows=feature_doc['rows'];missing_shape=set(feature_doc['missing_shape_request_ids'])
    cases={c['case_id']:c for c in native_table['measured_case_resources']['cases']}
    windows={w['case_id']:w for w in raw['measurement_windows']};startup=next(b['time_ns'] for b in raw['lifecycle_boundaries'] if b['boundary']=='startup_begin')
    protocol_rows=raw['priming_request_records']+raw['request_records'];results=[]
    for case in feature_rows:
        cid=case['case_id'];window=windows[cid];prior=[]
        for sample in raw['board_samples']:
            _,finish=resource.legacy.poll_bracket(sample)
            if startup<=sample['time_ns'] and finish<window['start_ns']:prior.append(sample['owned_used_mib'])
        p0=max(prior) if prior else None;target=cases[cid]['startup_through_this_drain_owned_sampled_peak_mib']
        future_rows=[r for r in raw['request_records'] if r['profile_job_id']==cid]
        public=[dict(owner=r['model'],task=r['task'],pool_id=r['pool_id']) for r in future_rows]
        descriptor=generated.shapes.candidate_at(protocol_rows,plan['resident_owners'],window['start_ns'])
        polls=[s for s in raw['board_samples'] if window['start_ns']<=s['time_ns']<=window['fixed_drain_deadline_ns']]
        gaps=[b['time_ns']-a['time_ns'] for a,b in zip(polls,polls[1:])]
        feature=feature_vector(case)
        if missing_shape&{r['request_id'] for r in future_rows}:feature=None
        results.append(dict(schema=ROW_SCHEMA,capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],case_id=cid,
            physical_execution_identity=native_table['physical_execution_identity'],resident_owners=plan['resident_owners'],
            features=feature,feature_details=case,P0_mib=p0,target_absolute_sampled_peak_mib=target,
            missing_actual_shape_request_ids=sorted(missing_shape&{r['request_id'] for r in future_rows}),
            public_keys=[list(k) for k in _public_keys(public)],prior_shape_descriptor=descriptor,
            support_context=support_context(descriptor,case['case_shape_max'],_public_keys(public)),
            forecast_origin_ns=window['start_ns'],fixed_drain_deadline_ns=window['fixed_drain_deadline_ns'],
            observation_domain=dict(nominal_poll_cadence_ns=20_000_000,horizon_ns=window['fixed_drain_deadline_ns']-window['start_ns']),
            coverage_diagnostic=dict(case_owned_polls=len(polls),largest_midpoint_gap_ns=max(gaps) if gaps else None),
            actual_future_trajectory_used=True,prospective_queue_memory_validation=False,
            target_scope='original_startup_through_this_fixed_drain_owned_sampled_peak',safe_factor_applied=False))
    return results


def _row_known(row):
    values=row.get('features');p0=row.get('P0_mib');target=row.get('target_absolute_sampled_peak_mib')
    return bool(type(values) is list and len(values)==4 and values[0]==1 and all(resource.legacy.finite(v) for v in values)
        and resource.legacy.finite(p0) and resource.legacy.finite(target) and target>=p0
        and _context_known(row.get('support_context')))


def fit_rows(rows,*,startup_donors=None):
    """Pure CPU table helper. It does not authenticate native rows."""
    require(rows and all(r.get('schema')==ROW_SCHEMA for r in rows),'Exact v3 explanatory rows required')
    require(len({(r['attempt_id'],r['case_id']) for r in rows})==len(rows),'Distinct original case rows required')
    blocks=sorted({r['capture_id'] for r in rows});require(len(blocks)>=2,'Two complete donor blocks required')
    attempts=sorted({r['attempt_id'] for r in rows})
    require(all({r['attempt_id'] for r in rows if r['capture_id']==b}=={b+'-shard-'+str(i) for i in (1,2,3,4)} for b in blocks),
        'Four independent startup attempts per whole block required')
    physical={r['physical_execution_identity'] for r in rows};require(len(physical)==1,'One physical execution domain required')
    require(all(r['resident_owners']==rows[0]['resident_owners'] for r in rows),'One resident owner roster required')
    groups=defaultdict(int)
    for row in rows:groups[row['attempt_id']]+=1
    known=[r for r in rows if _row_known(r)];coeff=None;rank=0;singular=[];weights=[];geometry=None;numeric_failure=None
    if known:
        weights=[1/(len(blocks)*4*groups[r['attempt_id']]) for r in known]
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                x=np.array([r['features'] for r in known],float);y=np.array([(r['target_absolute_sampled_peak_mib']-r['P0_mib'])/1024 for r in known],float)
                w=np.sqrt(weights);a=x*w[:,None];scales=np.linalg.norm(a,axis=0);scaled=a/np.where(scales>0,scales,1)
                solved,_,rank,sv=np.linalg.lstsq(scaled,y*w,rcond=None);rank=int(rank)
                if not all(np.isfinite(v).all() for v in (x,y,w,a,scales,scaled,solved,sv)):raise FloatingPointError('Nonfinite weighted fit')
                singular=sv.tolist()
                if rank==4:
                    coefficients=solved/np.where(scales>0,scales,1)
                    inverse=np.linalg.inv(a.T@a);_,_,vt=np.linalg.svd(a,full_matrices=False)
                    projector=vt.T@vt;leverage=np.einsum('ij,jk,ik->i',x,inverse,x)
                    if not all(np.isfinite(v).all() for v in (coefficients,inverse,projector,leverage)):raise FloatingPointError('Nonfinite fit geometry')
                    coeff=coefficients.tolist()
                    geometry=dict(weighted_design_gram_inverse=inverse.tolist(),row_space_projector=projector.tolist(),
                        training_leverage_min=float(min(leverage)),training_leverage_max=float(max(leverage)),
                        leverage_is_diagnostic_not_error_selected_support_threshold=True)
        except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as error:
            numeric_failure=type(error).__name__;coeff=None;rank=0;singular=[];geometry=None
    bounds=None
    if known:
        bounds=dict(features=[dict(min=min(r['features'][i] for r in known),max=max(r['features'][i] for r in known)) for i in range(4)],
            P0_mib=dict(min=min(r['P0_mib'] for r in known),max=max(r['P0_mib'] for r in known)))
        bounds['prior_owner_shape_envelope']={o:dict(min=min(r['support_context']['prior_owner_shape_envelope'][o] for r in known),
            max=max(r['support_context']['prior_owner_shape_envelope'][o] for r in known)) for o in explanatory.TEXT_OWNERS}
        bounds['exact_task_states']={r['support_context']['task_state_key']:copy.deepcopy(r['support_context']['task_state']) for r in known}
    donors=copy.deepcopy(startup_donors or {})
    require(all(v['training_block_id'] in blocks and k in attempts for k,v in donors.items()),'Startup donor outside training blocks/attempts')
    status=('FITTED_CONDITIONAL_ON_KNOWN_ROWS_ONLY_EXPLANATORY_HYPOTHESIS' if len(known)<len(rows) else 'FITTED_CONDITIONAL_EXPLANATORY_HYPOTHESIS') if coeff is not None else ('UNKNOWN_MEMORY_FIT_NUMERICAL_FAILURE' if numeric_failure else 'UNKNOWN_RANK_DEFICIENT_OR_NO_KNOWN_ROWS')
    return dict(schema=MODEL_SCHEMA,status=status,
        columns=list(COLUMNS),coefficients=coeff,numeric_rank=rank,singular_values=singular,
        fit_numerical_failure=numeric_failure,fit_numerically_succeeded=coeff is not None,
        training_block_ids=blocks,training_attempt_ids=attempts,physical_execution_identity=next(iter(physical)),
        resident_owners=rows[0]['resident_owners'],startup_donors=donors,
        feature_support=bounds,design_geometry=geometry,public_keys=sorted({tuple(k) for r in known for k in r['public_keys']}),
        observation_domains=list({json.dumps(r['observation_domain'],sort_keys=True):r['observation_domain'] for r in known}.values()),
        training_rows=copy.deepcopy(rows),total_training_rows=len(rows),known_training_rows=len(known),unknown_training_rows=len(rows)-len(known),
        full_training_denominator_supported=len(known)==len(rows),native_authenticity_established_by_table_helper=False,
        row_weights=[dict(attempt_id=r['attempt_id'],case_id=r['case_id'],weight=weight) for r,weight in zip(known,weights)],
        P0_coefficient_fixed=1,model_is_actual_trajectory_explanatory=True,
        prospective_donor_generated_rank_verified=False,conditional_feature_state_sufficient=False,
        sampled_peak_is_capacity_bound=False,probability_law_qualified=False,original_safe_factor_applied=False,
        main_queue_provider_eligible=False,resource_prediction_qualified=False,final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def _predict(model,features,p0,keys,observation_domain,context):
    reasons=[]
    if model.get('schema')!=MODEL_SCHEMA or model.get('columns')!=list(COLUMNS):raise ValueError('Exact fixed four-column model required')
    if model['coefficients'] is None or model['numeric_rank']!=4:reasons.append('MEMORY_FIT_NUMERICAL_FAILURE' if model.get('fit_numerical_failure') else 'RANK_DEFICIENT_MODEL')
    elif type(model['coefficients']) is not list or len(model['coefficients'])!=4:raise ValueError('Exactly four fixed-offset coefficients required')
    if not (type(features) is list and len(features)==4 and features[0]==1 and all(resource.legacy.finite(v) for v in features)):
        reasons.append('MISSING_OR_NONFINITE_GENERATED_FEATURES')
    if not resource.legacy.finite(p0):reasons.append('MISSING_CAUSAL_OR_GENERATED_P0')
    if not set(tuple(k) for k in keys)<=set(tuple(k) for k in model['public_keys']):reasons.append('UNKNOWN_PUBLIC_INPUT_DOMAIN')
    if observation_domain not in model['observation_domains']:reasons.append('UNKNOWN_OBSERVATION_DOMAIN')
    if not _context_known(context):reasons.append('UNKNOWN_PRIOR_SHAPE_OR_EXACT_TASK_STATE')
    support=model['feature_support']
    if support is None:reasons.append('NO_TRAINING_FEATURE_SUPPORT')
    elif not reasons:
        require(len(support['features'])==4,'Exactly four feature support ranges required')
        for bound in support['features']+[support['P0_mib']]+list(support['prior_owner_shape_envelope'].values()):
            require(set(bound)=={'min','max'} and resource.legacy.finite(bound['min']) and resource.legacy.finite(bound['max']) and 0<=bound['min']<=bound['max'],
                'Closed nonnegative ordered support ranges required')
        if any(not bound['min']<=value<=bound['max'] for value,bound in zip(features,support['features'])):reasons.append('GENERATED_FEATURE_OUTSIDE_DONOR_RANGE')
        if not support['P0_mib']['min']<=p0<=support['P0_mib']['max']:reasons.append('P0_OUTSIDE_DONOR_RANGE')
        if any(not bound['min']<=context['prior_owner_shape_envelope'][o]<=bound['max'] for o,bound in support['prior_owner_shape_envelope'].items()):reasons.append('PRIOR_SHAPE_ENVELOPE_OUTSIDE_DONOR_RANGE')
        state=context['task_state'];key=context['task_state_key']
        require(resource.digest(state)==key,'Task state body/hash differs')
        if support['exact_task_states'].get(key)!=state:reasons.append('UNKNOWN_EXACT_TASK_STATE_OR_ENVELOPE_DEPARTURE_PATTERN')
    diagnostic=None
    if model.get('design_geometry') is not None and type(features) is list and len(features)==4 and all(resource.legacy.finite(v) for v in features):
        geo=model['design_geometry'];inverse=np.array(geo['weighted_design_gram_inverse']);projector=np.array(geo['row_space_projector']);x=np.array(features)
        require(inverse.shape==projector.shape==(4,4) and np.isfinite(inverse).all() and np.isfinite(projector).all(),'Finite four-column design geometry required')
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                leverage=float(x@inverse@x);distance=float(np.linalg.norm(x-projector@x))
                if not all(math.isfinite(v) for v in (leverage,distance)):raise FloatingPointError('Nonfinite design diagnostic')
                diagnostic=dict(design_span_distance=distance,leverage=leverage,
                    training_leverage_max=geo['training_leverage_max'],leverage_is_diagnostic_only=True)
        except (FloatingPointError,OverflowError):
            reasons.append('NONFINITE_DESIGN_GEOMETRY_ARITHMETIC')
            diagnostic=dict(design_span_distance=None,leverage=None,leverage_is_diagnostic_only=True,
                status='UNKNOWN_NONFINITE_DESIGN_GEOMETRY_ARITHMETIC')
    if reasons:return dict(status='UNKNOWN',unknown_reasons=reasons,peak_memory_mib=None,support_diagnostic=diagnostic)
    require(all(type(v) in (float,int) and math.isfinite(v) for v in model['coefficients']),'Finite fixed-offset coefficients required')
    try:
        value=p0+1024*sum(a*b for a,b in zip(features,model['coefficients']));peak=max(p0,value)
        finite=math.isfinite(value) and math.isfinite(peak)
    except OverflowError:finite=False
    if not finite:return dict(status='UNKNOWN',unknown_reasons=['UNKNOWN_NONFINITE_MEMORY_ARITHMETIC'],peak_memory_mib=None,support_diagnostic=diagnostic)
    return dict(status='EMPIRICAL_CONDITIONAL_ABSOLUTE_CASE_PEAK',unknown_reasons=[],peak_memory_mib=peak,unclipped_absolute_prediction_mib=value,support_diagnostic=diagnostic)


def evaluate_explanatory_rows(rows,model):
    require(not {r['capture_id'] for r in rows}&set(model['training_block_ids']),'Whole heldout block entered donor training')
    result=[]
    for row in rows:
        require(row['physical_execution_identity']==model['physical_execution_identity'],'Heldout physical domain differs')
        prediction=_predict(model,row['features'],row['P0_mib'],row['public_keys'],row['observation_domain'],row.get('support_context'))
        estimate=prediction['peak_memory_mib'];truth=row['target_absolute_sampled_peak_mib']
        error=estimate-truth if estimate is not None and resource.legacy.finite(truth) else None
        result.append(dict(attempt_id=row['attempt_id'],case_id=row['case_id'],prediction=prediction,
            actual_absolute_sampled_peak_mib=truth,P0_mib=row['P0_mib'],signed_error_mib=error,
            underestimated=error<0 if error is not None else None,
            P0_baseline_signed_error_mib=row['P0_mib']-truth if resource.legacy.finite(row['P0_mib']) and resource.legacy.finite(truth) else None))
    return dict(rows=result,all_rows_retained=len(result)==len(rows),heldout_cases=len(rows),
        known_predictions=sum(r['signed_error_mib'] is not None for r in result),
        unknown_predictions=sum(r['signed_error_mib'] is None for r in result),
        underprediction_count=sum(r['underestimated'] is True for r in result),
        actual_future_features_conditioned=True,prospective_queue_memory_prediction_validated=False)


def _reject_future_labels(value):
    forbidden={'board_samples','owned_used_mib','sampled_owned_mib','target_absolute_sampled_peak_mib','peak_memory_mib',
        'absolute_sampled_peak_target_mib','worker_response','request_records','priming_request_records'}
    if isinstance(value,dict):
        require(not forbidden&set(value),'Realized future execution/memory labels cannot enter generated features')
        for child in value.values():_reject_future_labels(child)
    elif isinstance(value,(list,tuple)):
        for child in value:_reject_future_labels(child)


def generated_case_features(sidecar,world,public_requests,*,model,memory_origin,observation_domain):
    """Donor-only generation API: no raw/native future row or memory target arg."""
    _reject_future_labels([sidecar,world,public_requests]);world=generated.engine.decode(world)
    require(model.get('schema')==MODEL_SCHEMA,'Exact v3 model required')
    require(sidecar['physical_execution_identity']==model['physical_execution_identity'],'Generated physical domain differs')
    require(sidecar['slots'] and {s['training_block_id'] for s in sidecar['slots']}<=set(model['training_block_ids']),
        'Selected complete request vectors must come only from donor blocks')
    start=generated.rational(world['original_window'][0]);cutoff=generated.rational(world['cutoff_ns'])
    base=dict(schema=SCHEMA+'.generated_features',status='UNKNOWN',unknown_reasons=[],features=None,P0_mib=None,
        physical_execution_identity=model['physical_execution_identity'],training_block_ids=model['training_block_ids'],
        public_keys=[list(k) for k in _public_keys(public_requests)],observation_domain=copy.deepcopy(observation_domain),
        realized_future_memory_labels_used=False,actual_future_features_used=False,
        donor_generated_full_or_fold_rank_verified=False,origin_independently_authenticated=False)
    if memory_origin is None:return dict(base,unknown_reasons=['MISSING_EXPLICIT_MEMORY_ORIGIN'])
    if memory_origin.get('scope')=='resident_observed_before_origin':
        require(set(memory_origin)=={'scope','P0_mib','shape_descriptor','last_observed_poll_end_ns'},'Closed resident origin required')
        require(generated.rational(memory_origin['last_observed_poll_end_ns'])<start,'Resident P0 must be available strictly before origin')
        descriptor=memory_origin['shape_descriptor'];p0=memory_origin['P0_mib']
    elif memory_origin.get('scope')=='selected_training_donor_after_original_priming':
        require(set(memory_origin)=={'scope','donor_attempt_id'},'Closed generated startup donor origin required')
        donor=model['startup_donors'].get(memory_origin['donor_attempt_id'])
        require(donor is not None and donor['training_block_id'] in model['training_block_ids'],'Startup donor outside whole training fold')
        require({s['training_block_id'] for s in sidecar['slots']}=={donor['training_block_id']},'Startup and request vectors must share whole-block latent')
        descriptor=donor['shape_descriptor'];p0=donor['P0_mib']
    else:raise ValueError('Explicit resident or generated original-priming origin required; queue seed is not memory state')
    base['P0_mib']=p0;base['memory_origin_scope']=memory_origin['scope']
    if not resource.legacy.finite(p0) or descriptor is None:return dict(base,unknown_reasons=['UNKNOWN_INITIAL_PEAK_OR_SHAPE_HISTORY'])
    # Missing generated notices are not repaired from native heldout outcomes.
    trace=world.get('trace')
    if not isinstance(trace,list):return dict(base,unknown_reasons=['MISSING_GENERATED_EVENT_TRACE'])
    processed=[r for r in trace if r.get('kind')=='processed_notification'];reader=[r for r in trace if r.get('kind')=='protocol_reader_available']
    latent={(r['request_id'],r['identity']) for r in trace if r.get('kind')=='latent_occurrence'}
    reader_map={(r['request_id'],r['identity']):r for r in reader}
    if any((r['request_id'],r['identity']) not in latent or (r['request_id'],r['identity']) not in reader_map
        or reader_map[(r['request_id'],r['identity'])]['processing_ns']!=r['available_ns'] for r in processed):
        return dict(base,unknown_reasons=['MISSING_GENERATED_LATENT_OR_PROCESSING_EVENT'])
    expected={(r['request_id'],r['identity']) for r in reader if r['processing_ns']<=cutoff}
    if expected!={(r['request_id'],r['identity']) for r in processed}:return dict(base,unknown_reasons=['MISSING_GENERATED_PROCESSED_NOTICE'])
    if any(not any(r.get('kind')=='latent_occurrence' and r.get('request_id')==rid and r.get('action')=='result' for r in trace)
        for rid in world['RESULT_receipts']):return dict(base,unknown_reasons=['MISSING_GENERATED_RESULT_EVENT'])
    slots={(*s['public_key'],s['public_key_arrival_ordinal']):s for s in sidecar['slots']};ordinals=defaultdict(int);requests={};slot_by_request={}
    for _,r in sorted(enumerate(public_requests),key=lambda pair:(pair[1]['arrival_ns'],pair[0])):
        key=(r['owner'],r['task'],r['pool_id']);slot=(*key,ordinals[key]);ordinals[key]+=1;requests[r['request_id']]=r;slot_by_request[r['request_id']]=slot
    for rid in world['READY_occurrences']:
        if rid not in slot_by_request:return dict(base,unknown_reasons=['UNKNOWN_GENERATED_READY_PUBLIC_REQUEST'])
        needed={m['generated_mark_identity'] for n in slots[slot_by_request[rid]]['nodes'] for m in n['marks']}
        if not needed<={identity for request_id,identity in latent if request_id==rid}:
            return dict(base,unknown_reasons=['MISSING_COMPLETED_VECTOR_GENERATED_MARK'])
    intervals=generated.engine.decode(generated.generated_shape_intervals(sidecar,world,public_requests,
        resident_owners=tuple(model['resident_owners']),initial_descriptor=descriptor))
    shapes={o:0 for o in explanatory.TEXT_OWNERS};prompts={};missing=[]
    for notice in intervals['processed_shape_marks']:
        rid=notice['request_id'];owner=requests[rid]['owner']
        if owner not in shapes:continue
        current=next((r for r in notice['state_after']['shape_state']['current_processed_owner_task_exact_input_phase_shape'] if r['owner']==owner),None)
        if not current or current['phase'] not in ('prefill','decode'):continue
        shape=current['actual_BEGIN_shape'];b,s,index=(shape.get(k) for k in ('input_batch_size','input_sequence_length','forward_index'))
        if any(type(v) is not int or v<0 for v in (b,s,index)):missing.append(rid);continue
        if current['phase']=='prefill':prompts[rid]=s;demand=b*s
        elif rid in prompts:demand=b*(prompts[rid]+index)
        else:missing.append(rid);continue
        shapes[owner]=max(shapes[owner],demand)
    if missing:return dict(base,unknown_reasons=['MISSING_DONOR_GENERATED_TEXT_SHAPE_AXES'],missing_generated_shape_request_ids=sorted(set(missing)))
    concurrent=0;exposure=0
    for interval in intervals['intervals']:
        active=interval['state']['shape_state']['current_processed_owner_task_exact_input_phase_shape'];n=len(active)
        f=sum(a['phase'] in ('prefill','ocr_inference') for a in active);concurrent=max(concurrent,n)
        exposure+=float(interval['right_ns']-interval['left_ns'])*(n*(n-1)/2+f)
    horizon=cutoff-start
    if horizon<=0:return dict(base,unknown_reasons=['EMPTY_GENERATED_HORIZON'])
    require(observation_domain['horizon_ns']==horizon,'Declared sampling horizon differs from generated case')
    feature_details=dict(case_shape_max=shapes,C=concurrent,R_over_horizon=exposure/float(horizon))
    base.update(status='DONOR_ONLY_GENERATED_CASE_FEATURES',unknown_reasons=[],features=feature_vector(feature_details),
        support_context=support_context(descriptor,shapes,base['public_keys']),
        feature_details=feature_details,generated_processing_events=len(processed),shape_trace_sha256=generated.digest(intervals),
        generated_shape_intervals=intervals,sidecar_sha256=generated.digest(sidecar))
    return base


def predict_generated_case(model,features,*,query_times=()):
    require(features.get('schema')==SCHEMA+'.generated_features','Exact donor generated features required')
    require(features['physical_execution_identity']==model['physical_execution_identity']
        and features['training_block_ids']==model['training_block_ids'],'Generated feature/fold domain differs')
    prediction=_predict(model,features['features'],features['P0_mib'],features['public_keys'],features['observation_domain'],features.get('support_context'))
    if features['status']!='DONOR_ONLY_GENERATED_CASE_FEATURES':prediction=dict(status='UNKNOWN',peak_memory_mib=None,unknown_reasons=features['unknown_reasons'])
    return dict(schema=SCHEMA+'.prediction',prediction=prediction,queries=[dict(at_ns=at,prediction=copy.deepcopy(prediction)) for at in query_times],
        absolute_case_peak_evaluated_once=True,P0_coefficient_fixed=1,repeated_queries_accumulate_memory=False,
        original_safe_factor_applied=False,actual_future_features_used=False,conditional_memory_model_sufficiency_established=False,
        main_queue_provider_eligible=False,resource_prediction_qualified=False,capacity_bound=False,
        probability_law_qualified=False,final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def run(campaign_path,review_paths,output):
    native_tables,sources=resource.load_complete(campaign_path,review_paths)
    require(len(native_tables)==12 and len({t['capture_id'] for t in native_tables})==3,'Exactly twelve complete native attempts required')
    rows=[];donors={}
    for table in native_tables:
        raw=checked(table['raw']);plan=checked(table['plan']);context=checked(table['context'])
        attempt_rows=actual_rows_from_verified(raw,plan,context,table);rows.extend(attempt_rows)
        require([r['case_id'] for r in attempt_rows]==[c['case_id'] for c in plan['cases']],
            'Every original ordered case in each native attempt required')
        donors[plan['attempt_id']]=dict(training_block_id=plan['capture_id'],P0_mib=table['premeasurement_owned_sampled_peak_mib'],
            shape_descriptor=attempt_rows[0]['prior_shape_descriptor'])
    require(len(rows)==150,'All 150 original case attempts required')
    model=fit_rows(rows,startup_donors=donors);folds=[]
    for block in model['training_block_ids']:
        train=[r for r in rows if r['capture_id']!=block];heldout=[r for r in rows if r['capture_id']==block]
        fold=fit_rows(train,startup_donors={k:v for k,v in donors.items() if v['training_block_id']!=block})
        folds.append(dict(heldout_block_id=block,training_model=fold,heldout_attempt_ids=sorted({r['attempt_id'] for r in heldout}),
            actual_trajectory_conditional_evaluation=evaluate_explanatory_rows(heldout,fold)))
    result=dict(schema=SCHEMA+'.actual_trajectory_training_report',model=model,whole_block_leaveout=folds,
        all_150_rows_retained=True,all_three_folds_retained=True,actual_future_trajectory_fit_is_explanatory_only=True,
        donor_generated_path_predictions_run=False,transplant_witness_used_as_generated_rank=False,
        main_queue_provider_eligible=False,resource_prediction_qualified=False,final_gain_eligible=False,
        original_safe_factor_applied=False,validation_labels_read=False,GPU_started=False,
        sources=strict_references([sources,implementation_sources(__file__)]))
    require(all(receipt(r['path'])==r for r in result['sources']),'Case memory candidate source changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);path=out/'report.json'
    path.write_text(json.dumps(generated.encode(result),indent=2,allow_nan=False)+'\n');return receipt(path)
