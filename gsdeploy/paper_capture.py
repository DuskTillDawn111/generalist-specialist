"""Convert explicit complete native stage records to the training contract.

This is an offline adapter, not a collector. It neither segments legacy total
latency nor fabricates host gaps, device timestamps, counters or READY events.
"""

import copy

from .forecasting.lifecycle import export_training_observations
from .paper_training import OBS_SCHEMA, export_segments
from .resources import identity, validate_resources

CAPTURE_SCHEMA='gsdeploy.complete_stage_capture.v1'


def normalize_capture(capture,entry):
    required={'schema','split','kind','source','run_id','execution_context_sha256',
              'clock_contract','roster_complete','resident_aliases','observation_window_ns',
              'resource_configuration_sha256','raw','resource_observations'}
    if (set(capture)!=required or capture['schema']!=CAPTURE_SCHEMA or capture['split']!='training'
            or capture['kind'] not in {'measured','cpu_fixture'} or capture['roster_complete'] is not True):
        raise ValueError('Explicit complete training capture required')
    aliases={f'{m}#{i}':{'family':m} for m,n in entry['deployment'].items() for i in range(n)}
    if (sorted(capture['resident_aliases'])!=sorted(aliases)
            or len(capture['resident_aliases'])!=len(aliases)
            or capture['execution_context_sha256']!=entry['profiles']['execution_context_sha256']
            or capture['kind']!=entry['profiles']['provenance']['kind']):
        raise ValueError('Capture resident/execution/provenance identity mismatch')
    resources=validate_resources(entry['resources'],aliases) if 'resources' in entry else None
    if capture['resource_configuration_sha256']!=(identity(resources) if resources is not None else None):
        raise ValueError('Capture resource configuration differs from its stage template')
    raw=capture['raw']
    if (raw.get('kind')!=capture['kind'] or raw.get('execution_context_sha256')!=capture['execution_context_sha256']
            or raw.get('admission_enabled')!=(resources is not None and resources['mode']=='request_admission')):
        raise ValueError('Native capture execution/control mode mismatch')
    lifecycle=export_training_observations(raw,aliases,execution_context_sha256=capture['execution_context_sha256'],
        clock_contract=capture['clock_contract'],source=capture['source'])
    profiles={(p['family'],p['task'],p['pool_id']):p for p in entry['profiles']['profiles']}
    requests=[]
    for row in raw['request_records']:
        profile=profiles.get((aliases[row['model']]['family'],row['task'],row['pool_id']))
        if profile is None:raise ValueError('Native input has no execution-bound segment template')
        # A handler START notification can accompany phase events. Retain only
        # the marker after requiring it to agree with the explicit response.
        events=row.get('execution_events',[]);phases=[]
        for event in events:
            if (event.get('request_id')!=row['request_id'] or type(event.get('perf_ns')) is not int
                    or type(event.get('received_time_ns')) is not int
                    or event['received_time_ns']<event['perf_ns']):
                raise ValueError('Native phase identity/common-clock visibility invalid')
            if 'event' not in event:
                if event['perf_ns']!=row['worker_response'].get('start_perf_ns'):
                    raise ValueError('Native handler start markers disagree')
            else:phases.append(event)
        if len(phases)!=2*len(profile['segments']):
            raise ValueError('All host/GPU/sync/cleanup segment boundaries must be explicitly captured')
        spans=[]
        for segment,start,end in zip(profile['segments'],phases[::2],phases[1::2]):
            if any(e.get('segment_id')!=segment['id'] or e.get('kind')!=segment['kind'] for e in (start,end)):
                raise ValueError('Native ordered segment identity/kind differs from its template')
            if start.get('event')!='start' or end.get('event')!='end':
                raise ValueError('Native stage events are incomplete or out of order')
            spans.append(dict(segment_id=segment['id'],start_ns=start['perf_ns'],end_ns=end['perf_ns']))
        if (spans[0]['start_ns']!=row['worker_response'].get('start_perf_ns')
                or spans[-1]['end_ns']!=row['worker_response'].get('end_perf_ns')):
            raise ValueError('Complete segments must cover the full native handler; gaps are not inferred')
        requests.append(dict(request_id=row['request_id'],alias=row['model'],profile_id=profile['id'],segments=spans))
    # Lifecycle identities are run-specific even when sources name a shared
    # file containing several independent runs.
    for row,native in zip(lifecycle,raw['request_records']):row['id']=capture['run_id']+':'+native['request_id']
    result=dict(schema=OBS_SCHEMA,split='training',kind=capture['kind'],source=capture['source'],
        run_id=capture['run_id'],execution_context_sha256=capture['execution_context_sha256'],
        clock='host_perf_counter_ns',same_host_verified=True,roster_complete=True,
        observation_window_ns=copy.deepcopy(capture['observation_window_ns']),requests=requests,
        resource_observations=copy.deepcopy(capture['resource_observations']),lifecycle_observations=lifecycle)
    export_segments(result,entry['profiles'],{a:v['family'] for a,v in aliases.items()})
    return result
