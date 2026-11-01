"""Exact finite synthetic joint-posterior reference, never a runtime provider.

World masses are explicit CPU fixtures, not fitted/calibrated probabilities.
A missing END constrains notification availability, not physical response age.
No native state, main predictor gate, DAG cursor, SM clock or RNG is touched.
"""
from __future__ import annotations
import math

SCHEMA='gsdeploy.joint_response_notification_finite_reference.v1'
ENDPOINTS={'reader_receipt','parent_processing'}
WORLD_FIELDS={'id','mass','begin_ns','response_ns','shape_template_id','BEGIN','END'}
DELAY_FIELDS={'reader_delay_ns','processing_extra_ns'}
EVENT_FIELDS={'edge','occurrence_ns','available_ns','shape_template_id'}


def stamp(value):return type(value) is int and value>=0


def visible_notifications(envelopes,cutoff_ns):
    """Availability metadata is read first; future bodies are never examined."""
    if not stamp(cutoff_ns):raise ValueError('Integer cutoff required')
    visible=[]
    for envelope in envelopes:
        at=envelope['available_ns']
        if not stamp(at):raise ValueError('Explicit availability metadata required')
        if at>cutoff_ns:continue
        body=envelope['body']
        if type(body) is not dict or set(body)!={'edge','occurrence_ns','shape_template_id'}:
            raise ValueError('Exact visible notification body required')
        visible.append(dict(body,available_ns=at))
    return sorted(visible,key=lambda row:(row['available_ns'],row['edge']))


def _events(world,endpoint):
    if type(world) is not dict or set(world)!=WORLD_FIELDS:
        raise ValueError('Complete finite joint-world fields required')
    if (type(world['id']) is not str or not world['id'] or type(world['mass']) not in (int,float)
            or not math.isfinite(world['mass']) or world['mass']<0
            or not stamp(world['begin_ns']) or not stamp(world['response_ns'])
            or type(world['shape_template_id']) is not str or not world['shape_template_id']):
        raise ValueError('Finite explicit mass, response clock and prior template identity required')
    events=[]
    for edge,at in (('BEGIN',world['begin_ns']),('END',world['begin_ns']+world['response_ns'])):
        delay=world[edge]
        if delay is None:continue # explicit latent loss; no default loss probability
        if (type(delay) is not dict or set(delay)!=DELAY_FIELDS
                or not all(stamp(x) for x in delay.values())):
            raise ValueError('Explicit joint reader/processing delays or loss required')
        available=at+delay['reader_delay_ns']
        if endpoint=='parent_processing':available+=delay['processing_extra_ns']
        events.append(dict(edge=edge,occurrence_ns=at,available_ns=available,
            shape_template_id=world['shape_template_id'] if edge=='BEGIN' else None))
    return events


def condition(worlds,events,*,cutoff_ns,visibility_endpoint,registry_complete_through_ns,
              delivery_policy='PER_STAGE_FIFO_WITH_EXPLICIT_LOSS'):
    """Condition a finite joint fixture on the complete visible BEGIN/END set.

    This computes reference arithmetic only. It does not supply a learned prior,
    stage eligibility, unknown future covariate path, or native assimilation.
    """
    if visibility_endpoint not in ENDPOINTS or not stamp(cutoff_ns):
        raise ValueError('Frozen visibility endpoint and integer cutoff required')
    if not stamp(registry_complete_through_ns) or registry_complete_through_ns<cutoff_ns:
        return dict(schema=SCHEMA,status='UNKNOWN_INCOMPLETE_VISIBLE_REGISTRY',native_adapter_eligible=False)
    if delivery_policy!='PER_STAGE_FIFO_WITH_EXPLICIT_LOSS':raise ValueError('Unsupported delivery-order contract')
    visible={}
    for event in events:
        if (type(event) is not dict or set(event)!=EVENT_FIELDS or event['edge'] not in ('BEGIN','END')
                or not stamp(event['occurrence_ns']) or not stamp(event['available_ns'])
                or not event['occurrence_ns']<=event['available_ns']<=cutoff_ns
                or event['edge'] in visible):
            raise ValueError('Unique already-visible stage notification facts required')
        visible[event['edge']]=event
    if 'BEGIN' in visible and 'END' in visible:
        if (visible['END']['occurrence_ns']<visible['BEGIN']['occurrence_ns']
                or visible['END']['available_ns']<visible['BEGIN']['available_ns']):
            raise ValueError('Observed per-stage FIFO clock contradiction')
    selected=[];ids=set()
    for world in worlds:
        model_events=_events(world,visibility_endpoint)
        if world['id'] in ids:raise ValueError('Distinct joint-world identity required')
        ids.add(world['id'])
        by_edge={e['edge']:e for e in model_events}
        if ('BEGIN' in by_edge and 'END' in by_edge
                and by_edge['END']['available_ns']<by_edge['BEGIN']['available_ns']):
            raise ValueError('Joint delay world violates declared per-stage FIFO')
        predicted_visible={e['edge']:e for e in model_events if e['available_ns']<=cutoff_ns}
        if predicted_visible!=visible or world['mass']==0:continue
        end=world['begin_ns']+world['response_ns']
        end_notice=by_edge.get('END',{}).get('available_ns')
        selected.append(dict(id=world['id'],mass=world['mass'],
            response_completed=end<=cutoff_ns,
            latent_remaining_host_response_ns=max(0,end-cutoff_ns),
            remaining_END_notice_ns=None if end_notice is None else max(0,end_notice-cutoff_ns),
            END_notification_lost=end_notice is None))
    total=sum(row['mass'] for row in selected)
    if not total:return dict(schema=SCHEMA,status='UNKNOWN_ZERO_JOINT_SUPPORT',native_adapter_eligible=False)
    for row in selected:row['conditional_mass']=row.pop('mass')/total
    return dict(schema=SCHEMA,status='EXACT_FINITE_CPU_REFERENCE_ONLY',cutoff_ns=cutoff_ns,
        visibility_endpoint=visibility_endpoint,worlds=selected,
        completed_response_mass=sum(r['conditional_mass'] for r in selected if r['response_completed']),
        ongoing_response_mass=sum(r['conditional_mass'] for r in selected if not r['response_completed']),
        mean_latent_remaining_host_response_ns=sum(r['conditional_mass']*r['latent_remaining_host_response_ns'] for r in selected),
        END_lost_mass=sum(r['conditional_mass'] for r in selected if r['END_notification_lost']),
        notification_absence_implies_unfinished=False,response_completed_is_observed_fact=False,
        conditional_probability_calibrated=False,fixture_prior_is_trained_model=False,
        all_request_GPU_work_complete=None,physical_SM_service_ns=None,kernel_cursor=None,
        native_adapter_eligible=False,main_queue_provider_eligible=False,validation_labels_read=False)
