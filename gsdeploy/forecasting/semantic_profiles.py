"""Continuous host-call subsegments derived from explicit nested observations.

All subdivisions of an original mixed native span keep that mixed meaning.
CUDA event durations are retained as diagnostics and never replace host-clock
boundaries or turn an unlabelled gap into pure CPU execution.
"""
from collections import defaultdict

from ..forecast_errors import UnknownEvidence
from .native_training import stage_layout


def split_semantic_native(row):
    coarse = stage_layout(row)
    response = row['worker_response']
    events = response.get('semantic_phase_events')
    if response.get('semantic_phase_schema') != 'gsdeploy.semantic_native_observation.v1' or not events:
        raise UnknownEvidence('UNKNOWN_SEMANTIC_NATIVE_OBSERVATIONS')
    for event in events:
        if (event.get('request_id') != row['request_id']
                or type(event.get('host_start_ns')) is not int or type(event.get('host_end_ns')) is not int
                or event['host_start_ns'] >= event['host_end_ns']
                or not event.get('component') or not event.get('role')
                or not event.get('device_event_complete')
                or event.get('device_elapsed_ms') is None or event['device_elapsed_ms'] < 0
                or event.get('stream_id') != event.get('end_current_stream_id')):
            raise ValueError('Complete causal host-call and same-stream CUDA event observations required')
    result, assigned = [], set()
    for span in coarse:
        if span['kind'] != 'gpu':
            result.append(dict(span, semantic_role=span['id'], semantic_observation=False))
            continue
        contained = [(i,e) for i,e in enumerate(events)
                     if span['start_ns'] <= e['host_start_ns'] < e['host_end_ns'] <= span['end_ns']]
        for i,_ in contained:
            if i in assigned:raise ValueError('Semantic call belongs to multiple native roots')
            assigned.add(i)
        for i,(_,a) in enumerate(contained):
            for _,b in contained[i+1:]:
                left,right=max(a['host_start_ns'],b['host_start_ns']),min(a['host_end_ns'],b['host_end_ns'])
                if left<right and not (
                        a['host_start_ns']<=b['host_start_ns'] and b['host_end_ns']<=a['host_end_ns'] or
                        b['host_start_ns']<=a['host_start_ns'] and a['host_end_ns']<=b['host_end_ns']):
                    raise UnknownEvidence('CONCURRENT_NONNESTED_SEMANTIC_CALLS_REQUIRE_A_DIFFERENT_EXECUTION_MODEL')
        cuts=sorted({span['start_ns'],span['end_ns'],*[e[k] for _,e in contained for k in ('host_start_ns','host_end_ns')]})
        parts=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[(i,e) for i,e in contained if e['host_start_ns']<=left and right<=e['host_end_ns']]
            if active:
                shortest=min(e['host_end_ns']-e['host_start_ns'] for _,e in active)
                inner=[(i,e) for i,e in active if e['host_end_ns']-e['host_start_ns']==shortest]
                if len(inner)!=1:raise UnknownEvidence('AMBIGUOUS_IDENTICAL_SEMANTIC_CALL_BOUNDARIES')
                source,event=inner[0];role=event['role']
            else:
                source,role=None,'native_gap'
            if parts and parts[-1]['source_event_index']==source and parts[-1]['semantic_role']==role:
                parts[-1]['end_ns']=right
            else:
                parts.append(dict(start_ns=left,end_ns=right,semantic_role=role,source_event_index=source,
                                  kind='gpu',semantic_observation=source is not None,
                                  mixed_host_transfers_CUDA=True))
        for index,part in enumerate(parts):
            part['id']=f'{span["id"]}.semantic.{index:04d}.{part["semantic_role"]}'
            result.append(part)
    if assigned!=set(range(len(events))):
        raise ValueError('Semantic forward is outside its measured native span')
    if (result[0]['start_ns']!=coarse[0]['start_ns'] or result[-1]['end_ns']!=coarse[-1]['end_ns']
            or any(a['end_ns']!=b['start_ns'] for a,b in zip(result,result[1:]))):
        raise ValueError('Derived semantic layout must cover the complete original handler')
    return result


def layout_roles(layout):
    """An explicit input-specific sequence; no universal token count is assumed."""
    return [(s['id'],s['kind'],s['semantic_role']) for s in layout]
