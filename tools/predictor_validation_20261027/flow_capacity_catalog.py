"""Training-only original-interface capacity catalogs for exact flow conditions.

One catalog per complete alias-routing/resource/load/input condition prevents
family-aggregation key collisions. Fresh measured worker service is normalized
by the unchanged original g(n)/n. No validation observation is read. Outputs
contain keys only for declared training seeds; a future seed transfer requires
another prospectively bound artifact, never a measured-outcome label.
"""
from __future__ import annotations
import argparse
import collections
import copy
import inspect
import json
import math
from pathlib import Path
import statistics
import time

from gsdeploy.performance import profile_key
from .flow_eligibility import _routing, BINDING_SCHEMA, RHO_MAX, RHO_TOLERANCE
from .profiling import digest
from .probe_validation import read, receipt, save


def verify(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Changed capacity calibration source receipt')


def _number(v,positive=False):
    return type(v) in (int,float) and math.isfinite(v) and (v>0 if positive else v>=0)


def _input_domain(plan):
    verify(plan['profile_plan']);profile=read(plan['profile_plan']['path'])
    domain=collections.defaultdict(set)
    for item in profile['input_assignments']:
        if item['view']=='known_input':domain[item['task']].add(item['pool_id'])
    if set(domain)!=set(plan['workload_rps']):raise ValueError('Complete declared known-input domain required')
    return {t:sorted(ids) for t,ids in sorted(domain.items())}


def condition(plan,candidate):
    return {k:copy.deepcopy(v) for k,v in dict(execution_context_sha256=candidate['execution_context_sha256'],
        aliases=plan['aliases'],resource_configuration=candidate.get('resource_configuration'),
        routing_weights=candidate['routing_weights'],workload_rps=plan['workload_rps'],window_seconds=plan['window_seconds'],
        drain_seconds=plan['drain_seconds'],warmup_seconds=plan.get('warmup_seconds',0),
        arrival_process=plan['arrival_process'],known_input_domain=_input_domain(plan)).items()}


def training_evidence(raw_path,expected_plan=None):
    """Derive role from a pre-startup frozen training plan, preserve original raw."""
    raw_path=Path(raw_path).resolve();raw=read(raw_path);verify(raw['plan'])
    if expected_plan is not None and raw['plan']!=expected_plan:raise ValueError('Raw training plan differs from declared suite')
    plan=read(raw['plan']['path'])
    if (plan.get('phase')!='predictor_training' or plan.get('kind')!='prospective_training'
            or plan.get('split')!='training' or plan.get('validation_tuning') is not False):
        raise ValueError('Prospectively declared predictor_training plan required')
    if digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Training plan self hash mismatch')
    if not type(plan.get('created_ns')) is int or not plan['created_ns']<raw['startup_begin_ns']:
        raise ValueError('Training role must be frozen before startup')
    for key in ('protocol_generator','runner_source','context','catalog','profile_plan'):verify(plan[key])
    if (raw.get('schema')!='gsdeploy.native_flow_collection.v1' or raw.get('kind')!='measured' or raw.get('backend')!='native'
            or raw.get('status')!='completed' or raw.get('collection_complete') is not True
            or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors','qualification_issues'))):
        raise ValueError('Complete error-free native training flow required')
    candidates=[c for c in plan['candidates'] if c['id']==raw['candidate_id']]
    if len(candidates)!=1:raise ValueError('Unique declared training candidate required')
    candidate=candidates[0];aliases=plan['aliases'];domain=_input_domain(plan)
    if (raw['execution_context_sha256']!=candidate['execution_context_sha256']
            or raw.get('resource_configuration')!=candidate.get('resource_configuration')
            or raw.get('trace_sha256',raw.get('cohort_sha256'))!=digest(plan['trace'])):
        raise ValueError('Native training context/resource/trace mismatch')
    if raw['deadline_ns']-raw['phase_start_ns']!=round(plan['window_seconds']*1e9):raise ValueError('Training horizon mismatch')
    shutdown=raw.get('shutdown',{})
    if (set(shutdown)!=set(aliases) or any(v.get('forced') or v.get('exit_code')!=0 or not v.get('bye_received') for v in shutdown.values())):
        raise ValueError('Complete clean training worker shutdown evidence required')
    _routing(plan,candidate)
    actual={r['request_id']:r for r in raw['request_records']}
    if len(actual)!=len(raw['request_records']) or set(actual)!={r['request_id'] for r in plan['trace']}:
        raise ValueError('Complete unique training request cohort required')
    observations=[]
    for request in plan['trace']:
        row=actual[request['request_id']];alias=candidate['routes'][request['request_id']];task=request['task']
        if (row.get('split') not in (None,'training') or row['status']!='completed'
                or (row['model'],row['task'],row['pool_id'],row['scheduled_time_ns'])!=
                   (alias,task,request['pool_id'],raw['phase_start_ns']+request['offset_ns'])
                or request['pool_id'] not in domain[task]):
            raise ValueError('Training route/input/cohort or split mismatch')
        response=row['worker_response'];start=row['observed_worker_start_time_ns'];end=response['end_perf_ns']
        if (response['start_perf_ns']!=start or not row['scheduled_time_ns']<=start<end<=row['completion_time_ns']<=raw['observation_end_ns']):
            raise ValueError('Native training worker/parent clocks invalid')
        if response.get('success') is not True:raise ValueError('Failed inference cannot calibrate capacity service')
        observations.append({'request_id':row['request_id'],'family':aliases[alias]['family'],'alias':alias,'task':task,
            'pool_id':row['pool_id'],'worker_service_ms':(end-start)/1e6,'worker_start_ns':start,'worker_end_ns':end,
            'handler_completed_in_measurement_window':end<=raw['deadline_ns']})
    samples=raw['board_samples'];times=[s['time_ns'] for s in samples]
    if (not samples or any(type(t) is not int for t in times) or times!=sorted(set(times))
            or not times[0]<=raw['startup_begin_ns']<raw['phase_start_ns']<raw['deadline_ns']<=times[-1]
            or max((b-a for a,b in zip(times,times[1:])),default=math.inf)>2_000_000_000):
        raise ValueError('Ordered fully covered training telemetry required')
    for sample in samples:
        if (not _number(sample.get('owned_used_mib')) or sample.get('foreign_pids')
                or sample['time_ns']>=raw['startup_end_ns'] and set(sample.get('owned_aliases',[]))!=set(aliases)):
            raise ValueError('Valid all-owned training memory evidence required')
    # Only role metadata is derived. All original clocks, statuses, outputs and
    # measurements remain unchanged and the immutable native source is linked.
    derived=copy.deepcopy(raw)
    derived['request_records']=[{**r,'split':'training'} for r in raw['request_records']]
    derived.update(split='training',collection_role='predictor_training',derived_from=receipt(raw_path),
        derivation={'source':receipt(__file__),'role_source':raw['plan'],
            'rule':'Only add training role from plan.phase/kind/split frozen before native startup',
            'original_observations_modified':False})
    return plan,candidate,raw,derived,observations,max(s['owned_used_mib'] for s in samples)


def verify_catalog(path):
    """Recheck transitive native provenance and normalization arithmetic."""
    catalog=read(path) if not isinstance(path,dict) else path
    audit=catalog['native_flow_capacity_audit']
    for ref in audit['source_receipts']:verify(ref)
    independently_summed=collections.defaultdict(list);raw_peaks=[]
    for ref in catalog['native_flow_capacity_binding']['source_receipts']:
        verify(ref)
        doc=read(ref['path']) if Path(ref['path']).suffix=='.json' else {}
        if 'derived_from' in doc:
            verify(doc['derived_from']);verify(doc['derivation']['role_source']);verify(doc['derivation']['source'])
            original=read(doc['derived_from']['path'])
            if doc['request_records']!=[{**r,'split':'training'} for r in original['request_records']]:
                raise ValueError('Derived training evidence changed original observations')
            for key,value in original.items():
                if key!='request_records' and doc.get(key)!=value:
                    raise ValueError('Derived training evidence changed original raw field')
            for row in original['request_records']:
                family=audit['condition']['aliases'][row['model']]['family']
                independently_summed[(family,row['task'])].append((row['worker_response']['end_perf_ns']-row['observed_worker_start_time_ns'])/1e6)
            raw_peaks.append(max(s['owned_used_mib'] for s in original['board_samples']))
    for family,tasks in audit['service_estimates'].items():
        n=audit['deployment'][family];g=catalog['capacity_factors'][family][str(n)]
        for task,value in tasks.items():
            actual=independently_summed[(family,task)]
            if len(actual)!=value['request_count'] or not math.isclose(sum(actual),value['sum_worker_service_ms'],rel_tol=1e-12):
                raise ValueError('Computed capacity service totals differ from immutable native evidence')
            mean=value['sum_worker_service_ms']/value['request_count']
            normalized=mean*g/n
            if not math.isclose(normalized,value['family_normalized_service_ms'],rel_tol=1e-12):raise ValueError('Capacity normalization arithmetic mismatch')
            for profile in catalog['colocation_profiles'].values():
                if not math.isclose(profile['service_time_ms'][family][task],normalized,rel_tol=1e-12):raise ValueError('Catalog service differs from training-computed estimate')
    base=read(audit['base_catalog']['path']);verify(audit['base_catalog'])
    if digest(catalog['capacity_factors'])!=audit['original_capacity_factors_sha256'] or catalog['capacity_factors']!=base['capacity_factors']:
        raise ValueError('Original capacity factors changed')
    expected_peak=max(raw_peaks);expected_safe=base['safe_factor']*expected_peak
    if (catalog['safe_factor']!=base['safe_factor'] or audit['raw_owned_sampled_peak_mib']!=expected_peak
            or not math.isclose(audit['safe_memory_mib'],expected_safe,rel_tol=1e-12)
            or any(not math.isclose(p['safe_memory_mib'],expected_safe,rel_tol=1e-12) for p in catalog['colocation_profiles'].values())):
        raise ValueError('Capacity memory estimate or original safety factor changed')
    return True


def build(suite_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    suite=read(suite_path)
    if suite.get('schema')!='gsdeploy.continuous_training_suite.v1' or suite.get('no_validation_data') is not True:
        raise ValueError('Explicit training-only continuous suite required')
    verify(suite['source']);jobs=suite['jobs'];groups={};source_refs=[receipt(suite_path),suite['source'],receipt(__file__)]
    import gsdeploy.performance,gsdeploy.energy,gsdeploy.routing
    source_refs += [receipt(m.__file__) for m in (gsdeploy.performance,gsdeploy.energy,gsdeploy.routing)]
    source_refs.append(receipt(inspect.getfile(_routing)))
    raw_hashes=set()
    for job in jobs:
        verify(job['plan']);raw_path=Path(job['output'])/'raw.json';raw_ref=receipt(raw_path)
        if raw_ref['sha256'] in raw_hashes:raise ValueError('Duplicate native training raw')
        raw_hashes.add(raw_ref['sha256'])
        plan,candidate,raw,derived,observations,peak=training_evidence(raw_path,job['plan'])
        if candidate['id']!=job['candidate_id']:raise ValueError('Suite/native candidate mismatch')
        spec=condition(plan,candidate);key=digest(spec)
        group=groups.setdefault(key,{'condition':spec,'runs':[],'base_catalog':plan['catalog']})
        if group['base_catalog']!=plan['catalog']:raise ValueError('Original base catalog differs within condition')
        group['runs'].append(dict(plan=plan,candidate=candidate,raw=raw,derived=derived,observations=observations,peak=peak,
            raw_source=raw_ref,plan_source=job['plan']))
    expected_seeds=set(suite['seeds'])
    if len(expected_seeds)<2:raise ValueError('At least two declared fresh training seeds required')
    for group in groups.values():
        seeds=[r['plan']['seed'] for r in group['runs']]
        if len(seeds)!=len(set(seeds)) or set(seeds)!=expected_seeds:raise ValueError('Every condition must cover all declared independent training seeds')
    output.mkdir(parents=True,exist_ok=False);manifest=[]
    for key,group in sorted(groups.items()):
        runs=group['runs'];spec=group['condition'];folder=output/('condition-'+key[:16]);folder.mkdir()
        base=read(group['base_catalog']['path']);catalog=copy.deepcopy(base)
        deployment=dict(collections.Counter(v['family'] for v in spec['aliases'].values()))
        observations=[o for r in runs for o in r['observations']];by_task=collections.defaultdict(list)
        for o in observations:by_task[(o['family'],o['task'])].append(o)
        estimates={};service={};wrapper_refs=[];refs=[*source_refs,group['base_catalog']]
        for index,run in enumerate(runs):
            p=folder/('training-evidence-'+str(run['plan']['seed'])+'.json');save(p,run['derived']);wrapper_refs.append(receipt(p))
            refs.extend([run['raw_source'],run['plan_source'],*[run['plan'][k] for k in ('context','profile_plan','protocol_generator','runner_source')]])
        refs=list({(r['path'],r['sha256']):r for r in refs}.values())
        for (family,task),rows in sorted(by_task.items()):
            n=deployment[family];g=base['capacity_factors'][family][str(n)];mean=statistics.mean(r['worker_service_ms'] for r in rows)
            value=mean*g/n
            estimates.setdefault(family,{})[task]={'sum_worker_service_ms':sum(r['worker_service_ms'] for r in rows),
                'request_count':len(rows),'worker_mean_service_ms':mean,'replicas':n,'original_capacity_factor':g,
                'normalization_factor':g/n,'family_normalized_service_ms':value,
                'unique_input_count':len({r['pool_id'] for r in rows}),
                'drain_handler_fraction':sum(not r['handler_completed_in_measurement_window'] for r in rows)/len(rows),
                'run_worker_mean_service_ms':[statistics.mean(o['worker_service_ms'] for o in run['observations'] if (o['family'],o['task'])==(family,task)) for run in runs]}
            service.setdefault(family,{})[task]=value
        peak=max(r['peak'] for r in runs);safe=base['safe_factor']*peak
        catalog.update(performance_mode='profiled',memory_mode='profiled',colocation_profiles={})
        catalog.setdefault('measurement_context',{})['native_flow_capacity_condition']={'condition_sha256':key,**spec}
        catalog['native_flow_capacity_binding']={'schema':BINDING_SCHEMA,'split':'training',
            'execution_context_sha256':spec['execution_context_sha256'],
            'resource_configuration_sha256':digest(spec['resource_configuration']),'source_receipts':wrapper_refs,
            'scope':'Original planning rule under matched empirical training condition; native stability and prospective seed transfer remain separate'}
        catalog['native_flow_capacity_audit']={'schema':'gsdeploy.training_flow_capacity_computed_values.v1','condition':spec,
            'base_catalog':group['base_catalog'],
            'source_receipts':refs,'service_estimates':estimates,'deployment':deployment,
            'original_capacity_factors_sha256':digest(base['capacity_factors']),
            'raw_owned_sampled_peak_mib':peak,'safe_memory_mib':safe,'safe_factor':base['safe_factor'],
            'memory_safety_factor_application_count':1,'capacity_bound':False,'uncertainty_calibrated':False,
            'rho_threshold':RHO_MAX,'rho_tolerance':RHO_TOLERANCE,'training_seeds':sorted(expected_seeds),
            'training_observations':len(observations),'heldout_observations_used':False,
            'service_scope':'All declared training cohort worker END-START, including bounded drain; excludes admission wait, FIFO wait and parent transport tail',
            'normalization':'family_service_ms = mean_worker_END_minus_START_ms * original_g(n)/n; original rho formula and g(n) unchanged',
            'statistical_scope':'Request-weighted empirical training input distribution across two fresh seed realizations; independent validation and stationary capacity are not established'}
        for run in runs:
            p=run['plan'];routing=_routing(p,run['candidate']);pk=profile_key(catalog,deployment,routing,p['workload_rps'],p['window_seconds'],p['seed'],p['arrival_process'])
            catalog['colocation_profiles'][pk]={'kind':'measured','scope':'whole_gpu_colocation','coverage_valid':True,
                'service_time_ms':service,'safe_memory_mib':safe,
                'provenance':{'split':'training','condition_sha256':key,'training_seeds':sorted(expected_seeds),
                    'source_receipts':wrapper_refs,'estimator':'Pooled measured training-condition means; no heldout seed outcomes',
                    'profile_key_scope':'Declared training seed only; no new target-seed key automatically fabricated'}}
        save(folder/'catalog.json',catalog);verify_catalog(folder/'catalog.json')
        manifest.append({'condition_sha256':key,'candidate_ids':sorted({r['candidate']['id'] for r in runs}),
            'catalog':receipt(folder/'catalog.json'),'training_runs':len(runs),'observations':len(observations),
            'training_only_profile_keys':list(catalog['colocation_profiles'])})
    value={'schema':'gsdeploy.training_flow_capacity_catalog_manifest.v1','created_ns':time.perf_counter_ns(),
        'source_receipts':source_refs,'conditions':manifest,'training_runs':len(jobs),'GPU_started':False,
        'heldout_observations_used':False,'final_gain_eligible':False,
        'target_seed_transfer_enabled':False,'scope':'Original-interface numerical capacity calibration only; separate prospective transfer artifact and native independent stability required'}
    save(output/'manifest.json',value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(argv);v=build(a.suite,a.output)
    print(json.dumps({'conditions':len(v['conditions']),'training_runs':v['training_runs'],'output':str(Path(a.output).resolve()),'GPU_started':False}));return 0

if __name__=='__main__':raise SystemExit(main())
