"""Strict offline adapter for the complete native activity collector.

Solo input baselines and categorical interference remain empirical. This
pipeline never labels native-call spans as pure device time or releases a
model for algorithm admission solely because its training residual is small.
"""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
from collections import Counter, defaultdict

from ..paper_capture import CAPTURE_SCHEMA, normalize_capture
from ..paper_training import export_segments
from .lifecycle import fit_lifecycle
from .native_context import NativeContextProvider, fit_native_progress, fit_native_resources
from .profiles import PROFILE_SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_capture(bundle,*,allowed_splits=('training',)):
    """Accept only completed collector training cohorts under their saved plan."""
    raw,plan,context=(bundle[k] for k in ('raw','plan','context'))
    if (raw.get('schema')!='gsdeploy.native_activity_capture.v1'
            or raw.get('status')!='completed' or raw.get('split') not in allowed_splits
            or raw.get('kind') not in {'measured','cpu_fixture'}
            or raw.get('kind')=='measured' and raw.get('backend')!='native'
            or plan.get('split')!=raw.get('split') or plan.get('mode') not in {'solo','overlap'}
            or raw.get('admission_enabled') is not False
            or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors')
            or not bundle.get('source') or not raw.get('run_id')
            or raw['run_id']!=plan.get('run_id')):
        raise ValueError('Completed training-only native collector capture required')
    identity=digest(context)
    if (raw.get('execution_context_sha256')!=identity or plan.get('execution_context_sha256')!=identity
            or raw.get('resident_aliases')!=context.get('resident_aliases')
            or plan.get('aliases')!=context.get('resident_aliases')
            or not context.get('resident_aliases')):
        raise ValueError('Native collector plan/context identity mismatch')
    aliases=context['resident_aliases']
    if set(raw.get('ready',{}))!=set(aliases) or set(raw.get('shutdown',{}))!=set(aliases):
        raise ValueError('Complete resident lifecycle roster required')
    for alias in aliases:
        ready,shutdown=raw['ready'][alias],raw['shutdown'][alias]
        if (ready.get('clock')!='time.perf_counter_ns' or type(ready.get('pid')) is not int
                or ready['pid']<=0 or shutdown.get('forced') is not False
                or shutdown.get('exit_code')!=0 or shutdown.get('bye_received') is not True):
            raise ValueError('Explicit native common-clock READY and clean shutdown required')
    expected=Counter((j['id'],r['alias'],r['task'],r['pool_id'])
                     for j in plan['jobs'] for r in j['requests'])
    observed=Counter((r.get('profile_job_id'),r.get('model'),r.get('task'),r.get('pool_id'))
                     for r in raw['request_records'])
    if not expected or expected!=observed:
        raise ValueError('Native collector cohort differs from its complete planned input jobs')
    rows={r['request_id']:r for r in raw['request_records']}
    batches=raw.get('batches',[]);seen=set()
    if len(rows)!=len(raw['request_records']) or len(batches)!=len(plan['jobs']):
        raise ValueError('Unique complete native batch roster required')
    for job,batch in zip(plan['jobs'],batches):
        if job['id']!=batch['id'] or seen&set(batch['request_ids']):
            raise ValueError('Native batch order or unique request ownership changed')
        seen.update(batch['request_ids'])
        requests=[rows[rid] for rid in batch['request_ids']]
        planned=sorted(job['requests'],key=lambda r:r.get('offset_ms',0))
        if len(requests)!=len(planned):raise ValueError('Native batch cohort changed')
        for offered,row in zip(planned,requests):
            if ((row['model'],row['task'],row['pool_id'],row['profile_job_id'])!=
                    (offered['alias'],offered['task'],offered['pool_id'],job['id'])
                    or row['scheduled_time_ns']!=batch['start_ns']+round(offered.get('offset_ms',0)*1e6)):
                raise ValueError('Native input identity or prescribed arrival timing changed')
    if seen!=set(rows):raise ValueError('Native batches omit observed requests')
    for row in raw['request_records']:
        if row.get('phase')!='training' or row.get('status')!='completed' or row['model'] not in aliases:
            raise ValueError('Completed planned training request required')
        boundary=row.get('previous_handler_next_request_ready',{})
        if boundary.get('pid')!=raw['ready'][row['model']]['pid']:
            raise ValueError('Request READY must belong to its resident worker')
    start,end=raw['observation_start_ns'],raw['observation_end_ns']
    if type(start) is not int or type(end) is not int or start>=end:
        raise ValueError('Complete native observation window required')
    return identity


def stage_layout(row):
    """Use explicit PHASE boundaries, cross-checking the worker result spans."""
    events=[e for e in row['execution_events'] if 'event' in e]
    if not events or len(events)%2:raise ValueError('Complete explicit native phases required')
    spans=[]
    for start,end in zip(events[::2],events[1::2]):
        if (start.get('event')!='start' or end.get('event')!='end'
                or (start.get('segment_id'),start.get('kind'))!=(end.get('segment_id'),end.get('kind'))
                or start['kind'] not in {'host','gpu','sync','cleanup'}
                or type(start.get('perf_ns')) is not int or type(end.get('perf_ns')) is not int
                or start['perf_ns']>=end['perf_ns']
                or spans and spans[-1]['end_ns']!=start['perf_ns']):
            raise ValueError('Contiguous ordered native phase boundaries required')
        spans.append(dict(id=start['segment_id'],kind=start['kind'],start_ns=start['perf_ns'],end_ns=end['perf_ns']))
    response=row['worker_response']
    reported=[{k:s[k] for k in ('id','kind','start_ns','end_ns')} for s in response['observed_stages']]
    if (spans!=reported or spans[0]['start_ns']!=response['start_perf_ns']
            or spans[-1]['end_ns']!=response['end_perf_ns']):
        raise ValueError('Native PHASE and full-handler result spans disagree')
    return spans


def build_template(solo_bundles, pool_items, *, minimum_runs=2):
    """Pin every supported input to independently repeated, nonoverlapping solo work."""
    if minimum_runs<2 or len(solo_bundles)<minimum_runs:
        raise ValueError('Independently repeated complete solo runs required')
    identities={validate_capture(b) for b in solo_bundles}
    if len(identities)!=1 or len({b['raw']['run_id'] for b in solo_bundles})!=len(solo_bundles):
        raise ValueError('Distinct solo runs under one execution context required')
    kinds={b['raw']['kind'] for b in solo_bundles}
    if len(kinds)!=1:raise ValueError('Homogeneous physical/fixture provenance required')
    aliases=solo_bundles[0]['context']['resident_aliases']
    tasks=defaultdict(set)
    for item in pool_items:tasks[item['task_type']].add(item['pool_id'])
    expected={(alias,task,pool) for alias,family in aliases.items()
              for task,inputs in tasks.items() if family=='generalist_2' or family==task for pool in inputs}
    baselines=defaultdict(list);layouts={};run_ids=defaultdict(set)
    for bundle in solo_bundles:
        if bundle['plan']['mode']!='solo':raise ValueError('Solo baseline cannot use co-run captures')
        actual=Counter((r['model'],r['task'],r['pool_id']) for r in bundle['raw']['request_records'])
        if set(actual)!=expected or any(n!=1 for n in actual.values()):
            raise ValueError('Each solo run must cover every resident alias and full frozen input pool once')
        ranges=[]
        for row in bundle['raw']['request_records']:
            spans=stage_layout(row);ranges.append((spans[0]['start_ns'],spans[-1]['end_ns']))
            key=(aliases[row['model']],row['task'],row['pool_id'])
            layout=[(s['id'],s['kind']) for s in spans]
            if key in layouts and layouts[key]!=layout:raise ValueError('Native stage layout changed between solo rounds')
            layouts[key]=layout;baselines[key].append([(s['end_ns']-s['start_ns'])/1e6 for s in spans])
            run_ids[key].add(bundle['raw']['run_id'])
        ranges.sort()
        if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):
            raise ValueError('Solo input references contain concurrent native handlers')
    profiles=[];baseline_evidence=[]
    for key,values in sorted(baselines.items()):
        if len(run_ids[key])<minimum_runs:raise ValueError('Input baseline lacks independent solo repeats')
        profiles.append(dict(id=digest(key),family=key[0],task=key[1],pool_id=key[2],
            execution=dict(batch_size=1,backend='complete_native_calls',weight_reuse=False,
                           execution_context_sha256=next(iter(identities))),
            segments=[dict(id=name,kind=kind,duration_ms={'mean':statistics.mean(v[i] for v in values)},
                           demand=dict.fromkeys(('sm','dram','l2'),None if kind=='gpu' else 0.))
                      for i,(name,kind) in enumerate(layouts[key])]))
        baseline_evidence.append(dict(input=list(key),independent_run_ids=sorted(run_ids[key]),
                                      stage_duration_vectors_ms=values))
    template=dict(schema=PROFILE_SCHEMA,execution_context_sha256=next(iter(identities)),
        provenance=dict(kind=next(iter(kinds)),split='training',sources=sorted(b['source'] for b in solo_bundles)),
        profiles=profiles,contexts={})
    return template,baseline_evidence


def normalize_native(bundle,template):
    identity=validate_capture(bundle)
    if identity!=template['execution_context_sha256']:raise ValueError('Native template execution mismatch')
    aliases=bundle['context']['resident_aliases'];deployment=Counter(aliases.values())
    raw=copy.deepcopy(bundle['raw'])
    # The collector labels the cohort and phase. The lifecycle contract uses
    # an explicit per-row split as well; this copies the existing declaration.
    for row in raw['request_records']:
        row['split']=raw['split'];stage_layout(row)
    source=bundle['source']
    capture=dict(schema=CAPTURE_SCHEMA,kind=raw['kind'],split='training',source=source,run_id=raw['run_id'],
        execution_context_sha256=identity,roster_complete=True,resident_aliases=list(aliases),
        observation_window_ns=[raw['observation_start_ns'],raw['observation_end_ns']],
        resource_configuration_sha256=None,resource_observations=[],raw=raw,
        clock_contract=dict(handler='host_perf_counter_ns',runtime='host_perf_counter_ns',
            worker_next_ready='host_perf_counter_ns',same_host_verified=True,
            source=source+':owned local subprocesses; saved complete-worker and runtime perf_counter sources'))
    return normalize_capture(capture,dict(deployment=dict(deployment),profiles=template))


def fit_native_captures(bundles,pool_items,*,max_work_error=.2,block_seconds=5.,order_aware=False):
    if len({b['raw']['run_id'] for b in bundles})!=len(bundles):
        raise ValueError('Distinct independent capture run ids required')
    solo=[b for b in bundles if b['plan']['mode']=='solo']
    template,baseline_evidence=build_template(solo,pool_items)
    if order_aware:
        from .native_order import fit_order_baselines
        template=fit_order_baselines(template,solo,max_relative_error=max_work_error)
    observations=[];lifecycle=[];resource_runs=[];documents=[]
    for bundle in bundles:
        document=normalize_native(bundle,template);documents.append(document)
        rows,_=export_segments(document,template,bundle['context']['resident_aliases'])
        if order_aware:
            from .native_order import request_orders
            orders=request_orders(bundle['raw'])
            by_request={document['run_id']+':'+r['request_id']+':'+s['segment_id']:orders[r['request_id']]
                        for r in document['requests'] for s in r['segments']}
            observations.extend(dict(row,run_id=document['run_id'],native_order=by_request[row['id']]) for row in rows)
        else:observations.extend(dict(row,run_id=document['run_id']) for row in rows)
        lifecycle.extend(document['lifecycle_observations'])
        spans=[dict(alias=r['alias'],profile_id=r['profile_id'],**s)
               for r in document['requests'] for s in r['segments']]
        resource_runs.append(dict(run_id=document['run_id'],split='training',
            execution_context_sha256=template['execution_context_sha256'],
            observation_window_ns=document['observation_window_ns'],spans=spans,
            telemetry_errors=bundle['raw']['telemetry_errors'],board_samples=bundle['raw']['board_samples']))
    template['provenance']['sources']=sorted(b['source'] for b in bundles)
    law=fit_native_progress(template,observations,max_work_error=max_work_error)
    law['resources'],resource_diagnostic=fit_native_resources(template,resource_runs,block_seconds=block_seconds)
    template['native_activity_law']=law
    life=fit_lifecycle(lifecycle,template['execution_context_sha256'])
    NativeContextProvider(template,template['execution_context_sha256'],allow_cpu_fixture=True)
    return dict(schema='gsdeploy.native_activity_fit.v1',status='FITTED_UNVALIDATED',
        admission_enabled=False,independent_accuracy_qualified=False,profiles=template,lifecycle=life,
        baseline_evidence=baseline_evidence,resource_diagnostic=resource_diagnostic,
        normalized_runs=documents,source_receipts=[b['source'] for b in bundles])
