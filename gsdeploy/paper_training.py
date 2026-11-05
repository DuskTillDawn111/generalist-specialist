"""Training-only pipeline from complete stage observations to executable laws.

Clock-aligned stage spans must come from an explicit producer; legacy handler
totals are never split into invented GPU/host stages. Identifiable progress is
fitted with the existing work-conservation estimator. Power and memory use
separate observed full-context tuples. Missing contexts remain unsupported.
"""

import copy
import math
import statistics

from .forecasting.profiles import PROFILE_SCHEMA, ProfileProvider, context_key, fit_progress
from .forecasting.lifecycle import fit_lifecycle
from .forecast_errors import UnknownEvidence
from .paper_forecast import MODE, canonical, digest, validate_spec

OBS_SCHEMA = 'gsdeploy.complete_stage_training.v1'


def _ns(value):
    if type(value) is not int or value<0:
        raise ValueError('Common-clock integer stage timestamp required')
    return value


def export_segments(document, template, aliases):
    required = {'schema','split','kind','source','run_id','execution_context_sha256',
                'clock','same_host_verified','roster_complete','requests','resource_observations',
                'lifecycle_observations','observation_window_ns'}
    if (set(document)!=required or document['schema']!=OBS_SCHEMA
            or document['split']!='training' or document['kind'] not in {'measured','cpu_fixture'}
            or document['clock']!='host_perf_counter_ns' or document['same_host_verified'] is not True
            or document['roster_complete'] is not True
            or document['execution_context_sha256']!=template['execution_context_sha256']
            or any(not isinstance(document[k],str) or not document[k] for k in ('source','run_id'))):
        raise ValueError('Complete training roster, execution identity, source and verified common clock required')
    window = document['observation_window_ns']
    if not isinstance(window,list) or len(window)!=2 or _ns(window[0])>=_ns(window[1]):
        raise ValueError('Complete common-clock observation window required')
    profiles = {p['id']:p for p in template['profiles']}
    spans,seen = [],set()
    for row in document['requests']:
        if (set(row)!={'request_id','alias','profile_id','segments'}
                or not isinstance(row['request_id'],str) or not row['request_id'] or row['request_id'] in seen
                or row['alias'] not in aliases or row['profile_id'] not in profiles):
            raise ValueError('Unique complete stage request identity required')
        seen.add(row['request_id'])
        profile = profiles[row['profile_id']]
        if profile['family']!=aliases[row['alias']]:
            raise ValueError('Stage profile family differs from resident owner')
        if [s['segment_id'] for s in row['segments']]!=[s['id'] for s in profile['segments']]:
            raise ValueError('All ordered segments including host/sync/cleanup must be observed')
        previous = None
        for s in row['segments']:
            if set(s)!={'segment_id','start_ns','end_ns'}:
                raise ValueError('Explicit stage span fields required')
            start,end = _ns(s['start_ns']),_ns(s['end_ns'])
            if start<window[0] or end>window[1] or end<=start or previous is not None and start!=previous:
                raise ValueError('Stage clock/order/gap invalid; missing host work cannot be dropped')
            previous = end
            spans.append(dict(alias=row['alias'],profile_id=row['profile_id'],segment_id=s['segment_id'],
                              start_ns=start,end_ns=end,request_id=row['request_id']))
    if not spans:
        raise ValueError('Nonempty complete stage training observations required')
    for alias in aliases:
        owned=sorted([s for s in spans if s['alias']==alias],key=lambda s:s['start_ns'])
        if any(a['end_ns']>b['start_ns'] for a,b in zip(owned,owned[1:])):
            raise ValueError('An owner cannot execute two batch1 stages simultaneously')
    for resource in document['resource_observations']:
        if (set(resource)!={'id','members','board_watts','peak_memory_mib','scope','start_ns','end_ns'}
                or not window[0]<=_ns(resource['start_ns'])<_ns(resource['end_ns'])<=window[1]):
            raise ValueError('Resource observations require a covered common-clock window')
        start,end=resource['start_ns'],resource['end_ns']
        cuts=sorted({start,end}|{s[k] for s in spans for k in ('start_ns','end_ns') if start<s[k]<end})
        for a,b in zip(cuts,cuts[1:]):
            members={s['alias']:dict(profile_id=s['profile_id'],segment_id=s['segment_id'])
                     for s in spans if s['start_ns']<=a and s['end_ns']>=b}
            if members!=resource['members']:
                raise ValueError('Resource window crosses an unreported active-stage transition')
    endpoints = sorted({s[k] for s in spans for k in ('start_ns','end_ns')})
    observations,contexts = [],{}
    for target in spans:
        cuts = [t for t in endpoints if target['start_ns']<=t<=target['end_ns']]
        intervals = []
        for start,end in zip(cuts,cuts[1:]):
            members = {s['alias']:{'profile_id':s['profile_id'],'segment_id':s['segment_id']}
                       for s in spans if s['start_ns']<=start and s['end_ns']>=end}
            key = context_key(members)
            contexts[key] = dict(members=members,evidence=document['source'],scenarios={
                'mean':dict(progress={},board_watts=None,peak_memory_mib=None)})
            intervals.append(dict(duration_ms=(end-start)/1e6,members=members))
        observations.append(dict(id=f"{document['run_id']}:{target['request_id']}:{target['segment_id']}",
            split='training',execution_context_sha256=document['execution_context_sha256'],
            target_alias=target['alias'],profile_id=target['profile_id'],segment_id=target['segment_id'],
            intervals=intervals))
    return observations,contexts


def fit_deployment(entry, documents, *, minimum_observations=2, max_relative_work_error=.15,
                   interpolate_contexts=False):
    if not documents or len({d['run_id'] for d in documents})!=len(documents):
        raise ValueError('Distinct nonempty training run identities required')
    template = copy.deepcopy(entry['profiles'])
    if template['schema']!=PROFILE_SCHEMA:
        raise ValueError('Explicit per-input baseline segment template required')
    context = template['execution_context_sha256']
    kinds = {d['kind'] for d in documents}
    if kinds!={template['provenance']['kind']}:
        raise ValueError('Baseline and observations must use homogeneous measured/fixture provenance')
    aliases={f'{m}#{i}':m for m,n in entry['deployment'].items() for i in range(n)}
    template['contexts']={}
    template.pop('preparation_contexts',None)  # This entry point is batch1 only.
    template.pop('context_interpolation',None)  # Never inherit a stale fitted law.
    segments,lifecycle,resources=[],[],{}
    progress_runs,life_runs={},{}
    ids=set()
    for d in documents:
        observed,contexts=export_segments(d,template,aliases)
        segments.extend(observed);template['contexts'].update(contexts)
        for row in observed:
            for interval in row['intervals']:
                key=(row['target_alias'],row['profile_id'],row['segment_id'],context_key(interval['members']))
                progress_runs.setdefault(key,set()).add(d['run_id'])
        request_lookup={d['run_id']+':'+r['request_id']:r for r in d['requests']}
        for row in d['lifecycle_observations']:
            if (row.get('kind')!=d['kind'] or row.get('source')!=d['source']
                    or row.get('execution_context_sha256')!=context):
                raise ValueError('Lifecycle observations must retain their training source and execution identity')
            request=request_lookup.get(row.get('id'))
            profile=next((p for p in template['profiles'] if request and p['id']==request['profile_id']),None)
            if (profile is None or any(row.get(k)!=profile[k] for k in ('family','task','pool_id'))
                    or row.get('timestamps',{}).get('handler_end_ns')!=request['segments'][-1]['end_ns']):
                raise ValueError('Lifecycle identity and handler_end must match the complete observed request')
            life_runs.setdefault(tuple(row[k] for k in ('family','task','pool_id')),set()).add(d['run_id'])
            lifecycle.append(row)
        for row in d['resource_observations']:
            if (set(row)!={'id','members','board_watts','peak_memory_mib','scope','start_ns','end_ns'}
                    or not row['id'] or row['id'] in ids
                    or row['scope']!='whole_board_and_all_resident_workers'):
                raise ValueError('Unique measured complete-context board/memory observation required')
            ids.add(row['id'])
            for field in ('board_watts','peak_memory_mib'):
                value=row[field]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Observed nonnegative finite resources required')
            key=context_key(row['members'])
            template['contexts'].setdefault(key,dict(members=copy.deepcopy(row['members']),
                evidence=d['source'],scenarios={'mean':dict(progress={},board_watts=None,peak_memory_mib=None)}))
            resources.setdefault(key,[]).append((d['run_id'],d['source'],row))
    template['provenance']=dict(kind=next(iter(kinds)),split='training',
        sources=sorted(set(template['provenance']['sources'])|{d['source'] for d in documents}))
    # fit_progress clears stale rates; resources have already been cleared.
    fitted=fit_progress(template,segments,minimum_observations=minimum_observations,
                        max_relative_work_error=max_relative_work_error)
    missing=[]
    for group in fitted['fit_diagnostics']['groups']:
        counts={key:len(progress_runs.get((group['alias'],group['profile_id'],group['segment_id'],key),set()))
                for key in group['contexts']}
        group['independent_runs_per_context']=counts
        if any(n<2 for n in counts.values()):
            group.update(accepted=False,reason='INSUFFICIENT_INDEPENDENT_PROGRESS_RUNS')
            for key in group['contexts']:
                fitted['contexts'].get(key,{}).get('scenarios',{}).get('mean',{}).get('progress',{}).pop(group['alias'],None)
    for ctx in fitted['contexts'].values():
        mean=ctx['scenarios'].get('mean')
        if mean is not None and set(mean['progress'])!=set(ctx['members']):
            del ctx['scenarios']['mean']
    fitted['contexts']={k:v for k,v in fitted['contexts'].items() if v['scenarios']}
    for key,ctx in fitted['contexts'].items():
        rows=resources.get(key,[])
        if len(rows)>=minimum_observations and len({run for run,source,row in rows})>=2:
            ctx['scenarios']['mean']['board_watts']=statistics.mean(row['board_watts'] for run,source,row in rows)
            ctx['scenarios']['mean']['peak_memory_mib']=max(row['peak_memory_mib'] for run,source,row in rows)
            ctx['evidence']=canonical(sorted({source for run,source,row in rows}))
        else:
            missing.append(dict(context_key=key,reason='INSUFFICIENT_INDEPENDENT_RESOURCE_OBSERVATIONS'))
    # Idle must exist with explicit resources. It has no progress coefficients.
    idle=context_key({})
    if idle not in fitted['contexts']:
        missing.append(dict(context_key=idle,reason='MISSING_IDLE_RESIDENT_RESOURCE_CONTEXT'))
    life=copy.deepcopy(entry['lifecycle'])
    life['profiles']=[]
    life['provenance']=copy.deepcopy(fitted['provenance'])
    grouped={}
    for row in lifecycle:
        grouped.setdefault(tuple(row[k] for k in ('family','task','pool_id','permit_scope')),[]).append(row)
    for template_life in entry['lifecycle']['profiles']:
        key=tuple(template_life[k] for k in ('family','task','pool_id','permit_scope'))
        rows=grouped.pop(key,[])
        try:
            if not rows:
                raise UnknownEvidence('NO_LIFECYCLE_OBSERVATIONS')
            trained=fit_lifecycle(rows,context,minimum_observations=minimum_observations)['profiles'][0]
        except UnknownEvidence:
            trained={k:template_life[k] for k in ('family','task','pool_id','permit_scope')}
            trained.update(scenarios={'mean':dict(parent_receipt_ms=None,permit_release_ms=None,worker_next_ready_ms=None)},
                           evidence=[d['source'] for d in documents])
        life['profiles'].append(trained)
    if grouped:
        raise ValueError('Lifecycle training permit scope differs from the declared execution template')
    life_keys={tuple(p[k] for k in ('family','task','pool_id')) for p in life['profiles']}
    for profile in template['profiles']:
        if tuple(profile[k] for k in ('family','task','pool_id')) not in life_keys:
            missing.append(dict(input=[profile[k] for k in ('family','task','pool_id')],reason='MISSING_LIFECYCLE_INPUT_TEMPLATE'))
    for profile in life['profiles']:
        delays=profile['scenarios']['mean']
        if len(life_runs.get(tuple(profile[k] for k in ('family','task','pool_id')),set()))<2:
            for field in delays:
                delays[field]=None
        required=['parent_receipt_ms','worker_next_ready_ms']
        if profile['permit_scope']=='request_admission':
            required.append('permit_release_ms')
        if any(delays[k] is None for k in required):
            missing.append(dict(input=[profile[k] for k in ('family','task','pool_id')],reason='UNKNOWN_LIFECYCLE_BOUNDARY'))
    rejected=[g for g in fitted['fit_diagnostics']['groups'] if not g['accepted']]
    if interpolate_contexts:
        from .forecasting.context_law import SCHEMA
        fitted['context_interpolation']=dict(schema=SCHEMA,method='convex_observed_resource_features',scenarios=['mean'])
    ProfileProvider(fitted,context,allow_cpu_fixture=next(iter(kinds))=='cpu_fixture')
    result={**copy.deepcopy(entry),'profiles':fitted,'lifecycle':life}
    report=dict(status='PARTIAL_TRAINING_LAWS' if missing or rejected else 'FITTED_DECLARED_CONTEXTS_ONLY',
        training_runs=len(documents),segment_observations=len(segments),
        identified_contexts=len(fitted['contexts']),rejected_progress_groups=rejected,missing=missing,
        supported_scope='exact observed inputs/contexts plus opt-in within-hull interpolation; unsupported remains UNKNOWN',
        context_interpolation_enabled=interpolate_contexts,
        interpolation_scope='same target and execution; observed feature hull only; no physical guarantee' if interpolate_contexts else None,
        independent_accuracy_qualified=False,physical_resource_bound=False,
        source_sha256=digest(documents))
    return result,report


def fit_config(config, training_by_deployment, **options):
    """Produce a new explicit nominal model, never mutate/reseal old evidence."""
    raw=copy.deepcopy(config.as_dict())
    if raw['catalog'].get('performance_mode')!=MODE:
        raise ValueError('Training requires an explicit paper predictor configuration')
    supported={canonical(e['deployment']) for e in raw['catalog'][MODE]['deployments']}
    if set(training_by_deployment)-supported:
        raise ValueError('Training contains an undeclared or noncanonical deployment identity')
    entries,reports=[],[]
    for entry in raw['catalog'][MODE]['deployments']:
        documents=training_by_deployment.get(canonical(entry['deployment']))
        if documents is None:
            continue  # Output domain only includes explicitly trained entries.
        fitted,report=fit_deployment(entry,documents,**options)
        entries.append(fitted);reports.append(dict(deployment=entry['deployment'],**report))
    if not entries:
        raise ValueError('No training deployment matches the declared execution domain')
    raw['catalog'][MODE]['deployments']=entries
    raw['catalog'][MODE]['scenarios']={'1.0':'mean'}
    validate_spec(raw['catalog'])
    return raw,dict(schema='gsdeploy.paper_training_report.v1',deployments=reports,
                    status='TRAINING_ONLY_NOT_ACCURACY_QUALIFICATION',validation_labels_read=False)
