"""Prospective global-max2 H600 training acquisition and native evidence mapping.

Prepare/verify/map are CPU only. Execute is an explicit, reviewed operation;
this module never reclassifies validation/pilot/final outcomes as training.
Base worker and controller identities are separate and both are mandatory.
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
import math
from pathlib import Path
import random
import time

from gsdeploy.energy import integrate_power
from gsdeploy.resources import validate_resources
from . import native_flow, mechanism_audit, flow_profiles, flow_validation, flow_eligibility
from .background_collect import context_identity
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest, RESIDENTS

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'artifacts/predictor-gpu-20261027/background-training-v7'
SCHEMA='gsdeploy.admission_H600_training_campaign.v1'
SEEDS=(628101,628102)
ROUTES=tuple('admission2_'+c for c in ('balanced','skewed','spill_low','spill_high'))
POLICY=dict(seeds=list(SEEDS),candidate_ids=list(ROUTES),primary_training_runs=8,
    H=600.,drain_seconds=300.,warmup_seconds=0.,
    workload_rps=dict(text=1.85,image=.05,asr=.05,ocr=.05),
    resource_mode='request_admission',global_max_active=2,per_alias_max_inflight=1,
    residency=RESIDENTS,role='primary_training',split='training',
    service_fitting='Training-only observations may support a separately declared fit; no fitting by this acquisition entry',
    no_retries=True,no_seed_replacement=True,no_validation_data=True,
    pooling='Never pool shared and admission; exact controller and complete route condition required',
    qualification='Acquisition alone grants no capacity, service transfer, predictor or final-gain PASS')


def sources():
    import gsdeploy.energy,gsdeploy.resources,gsdeploy.routing,gsdeploy.stability
    names=('admission_training_v1.py','native_flow.py','mechanism_audit.py','flow_profiles.py',
        'background_profiles.py','semantic_profiles.py','profiling_analysis.py','flow_validation.py','background_collect.py',
        'flow_capacity_catalog.py','flow_capacity_transfer.py','flow_eligibility.py')
    return [receipt(Path(__file__).with_name(n)) for n in names]+[
        receipt(m.__file__) for m in (gsdeploy.energy,gsdeploy.resources,gsdeploy.routing,gsdeploy.stability)]


def _refs(value):
    found={}
    def visit(item):
        if isinstance(item,dict):
            if set(item)=={'path','sha256'}:
                checked(item);found[(item['path'],item['sha256'])]=item
            else:
                for child in item.values():visit(child)
        elif isinstance(item,list):
            for child in item:visit(child)
    visit(value);return list(found.values())


def _context(profile_ref,context_ref):
    profile=read(checked(profile_ref));context=read(checked(context_ref))
    if (context.get('residents')!=RESIDENTS or profile.get('execution_context_sha256')!=context_identity(context)
            or profile.get('plan_sha256')!=digest({k:v for k,v in profile.items() if k!='plan_sha256'})):
        raise ValueError('Exact fixed-residency base worker/profile identity required')
    snapshots=context.get('execution_sources',[])
    if (not snapshots or context.get('sources')!=[s['frozen'] for s in snapshots]
            or len({s['original_path'] for s in snapshots})!=len(snapshots)):
        raise ValueError('Complete unique immutable base worker snapshots required')
    for source in snapshots:
        checked(source['frozen'])
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Current worker source differs from immutable base snapshot')
    for name in mechanism_audit.SEMANTIC_PATHS:
        selected=[s for s in snapshots if Path(s['original_path']).resolve()==ROOT/name]
        if len(selected)!=1 or selected[0]['frozen']['sha256']!=receipt(ROOT/name)['sha256']:
            raise ValueError('Current controller semantics absent from base source snapshots')
    return profile,context


def controller(context_ref,context_sha):
    value=dict(base_worker_context=context_ref,base_worker_context_sha256=context_sha,
        resource_configuration=native_flow.resource_configuration(2),
        resource_configuration_sha256=digest(native_flow.resource_configuration(2)),
        runtime_semantics_sources=[receipt(ROOT/n) for n in mechanism_audit.SEMANTIC_PATHS],
        training_mapper_source=receipt(__file__),audit_source=receipt(mechanism_audit.__file__))
    return {**value,'controller_sha256':digest(value)}


def _project(native,protocol_ref,created_ns):
    plan=copy.deepcopy(native)
    for key in ('plan_sha256','prephase_memory_evidence','startup_comparison_policy'):plan.pop(key,None)
    plan['candidates']=[c for cid in ROUTES for c in plan['candidates'] if c['id']==cid]
    for c in plan['candidates']:
        c.pop('predicted_prephase_peak_mib',None);c.pop('predicted_startup_joules',None)
    binding=controller(plan['context'],plan['execution_context_sha256'])
    plan.update(created_ns=created_ns,phase='predictor_training',kind='prospective_training',split='training',
        validation_tuning=False,protocol_generator=receipt(__file__),training_campaign_protocol=protocol_ref,
        campaign_role='primary_training',training_run_label='primary',measurement_role='admission_H600_training',
        controller_binding=binding,baseline_candidate_id=ROUTES[0],resource_model_transfer='NOT_AUTHORIZED',
        scope='Prospective admission2 primary training only; base worker context plus separate controller identity',
        final_gain_eligible=False)
    plan['plan_sha256']=digest(plan);return plan


def _weights(cid):
    fraction,spill={'admission2_balanced':(.5,0.),'admission2_skewed':(.25,0.),
        'admission2_spill_low':(.5,1/6),'admission2_spill_high':(.5,1/3)}[cid]
    text={'text#0':(1-spill)*fraction,'text#1':(1-spill)*(1-fraction)}
    if spill:text['generalist_2#0']=spill
    return dict(text=text,image={'generalist_2#0':1.},asr={'generalist_2#0':1.},ocr={'ocr#0':1.})


def _priming(profile):
    result=[]
    for alias,family in RESIDENTS.items():
        for task in (['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']):
            for item in sorted((r for r in profile['inputs'] if r['task']==task),key=lambda r:r['input_size'],reverse=True)[:3]:
                result.append((alias,task,item['pool_id']))
    return result


def prepare(output,profile_dir=None):
    output=Path(output).resolve();profile_dir=Path(profile_dir or PROFILE).resolve()
    if output.exists():raise FileExistsError(output)
    # The v1 acquisition domain is the original public fixed training profile.
    profile_ref=receipt(profile_dir/'plan.json');context_ref=receipt(profile_dir/'context.json')
    profile,context=_context(profile_ref,context_ref)
    if profile_dir!=PROFILE.resolve():raise ValueError('Fixed background-training-v7 profile required')
    output.mkdir(parents=True,exist_ok=False)
    refs=_refs([sources(),profile_ref,context_ref,context])
    protocol=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),policy=copy.deepcopy(POLICY),
        profile_plan=profile_ref,context=context_ref,source=receipt(__file__),source_receipts=refs,
        no_validation_data=True,independent_outcomes_read=False,GPU_started=False)
    protocol['protocol_sha256']=digest(protocol);save(output/'protocol.json',protocol)
    protocol_ref=receipt(output/'protocol.json');jobs=[];plan_refs=[]
    for seed in SEEDS:
        folder=output/('seed-'+str(seed))
        native_flow.prepare(profile_dir,folder,seed=seed,duration=600.,drain=300.,warmup=0.,include_admission=True)
        (folder/'plan.json').rename(folder/'native-prepared-plan.json')
        plan=_project(read(folder/'native-prepared-plan.json'),protocol_ref,time.perf_counter_ns())
        save(folder/'plan.json',plan);ref=receipt(folder/'plan.json');plan_refs.append(ref)
        for cid in ROUTES:
            job_id='seed-'+str(seed)+'__'+cid+'__primary'
            jobs.append(dict(job_id=job_id,run_id=job_id,seed=seed,candidate_id=cid,plan=ref,
                native_preparation=receipt(folder/'native-prepared-plan.json'),campaign_role='primary_training',
                split='training',controller_sha256=plan['controller_binding']['controller_sha256'],
                output=str((folder/'runs'/cid).resolve())))
    random.Random('admission2-training-v1-fixed-order').shuffle(jobs)
    campaign=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),policy=copy.deepcopy(POLICY),
        source=receipt(__file__),protocol=protocol_ref,profile_plan=profile_ref,context=context_ref,
        controller_binding=controller(context_ref,context_identity(context)),jobs=jobs,
        source_receipts=_refs([refs,protocol_ref,plan_refs,[j['native_preparation'] for j in jobs]]),
        no_retries=True,no_seed_replacement=True,no_validation_data=True,validation_sources_used=[],
        independent_outcomes_read=False,GPU_started=False,final_gain_eligible=False,
        expected_measurement_requests=9600,execution_requires_main_thread_review=True)
    campaign['campaign_sha256']=digest(campaign);save(output/'campaign.json',campaign)
    verify(output/'campaign.json');return campaign


def verify(campaign):
    """Audit all eight prospective job identities before any consumer uses one."""
    path=Path(campaign).resolve() if not isinstance(campaign,dict) else None
    if path is None:raise ValueError('Campaign artifact path required; unbound in-memory wrappers are forbidden')
    value=read(path)
    if (value.get('schema')!=SCHEMA or value.get('source')!=receipt(__file__) or value.get('policy')!=POLICY
            or value.get('campaign_sha256')!=digest({k:v for k,v in value.items() if k!='campaign_sha256'})
            or value.get('no_retries') is not True or value.get('no_seed_replacement') is not True
            or value.get('no_validation_data') is not True or value.get('validation_sources_used')!=[]
            or value.get('independent_outcomes_read') is not False or value.get('GPU_started') is not False
            or value.get('final_gain_eligible') is not False
            or value.get('expected_measurement_requests')!=9600 or value.get('execution_requires_main_thread_review') is not True):
        raise ValueError('Exact prospective admission2 training campaign required')
    for ref in value['source_receipts']:checked(ref)
    p=read(checked(value['protocol']))
    if (p.get('schema')!=SCHEMA+'.protocol' or p.get('policy')!=POLICY or p.get('source')!=receipt(__file__)
            or p.get('protocol_sha256')!=digest({k:v for k,v in p.items() if k!='protocol_sha256'})
            or p.get('context')!=value['context'] or p.get('profile_plan')!=value['profile_plan']
            or p.get('no_validation_data') is not True or p.get('independent_outcomes_read') is not False):
        raise ValueError('Prospective training protocol/source contract changed')
    profile,context=_context(value['profile_plan'],value['context'])
    if (value['context']!=receipt(PROFILE/'context.json') or value['profile_plan']!=receipt(PROFILE/'plan.json')
            or value.get('controller_binding')!=controller(value['context'],context_identity(context))):
        raise ValueError('Fixed base worker and separate globalmax2 controller required')
    required=_refs([sources(),value['context'],value['profile_plan'],context])
    refs={(r['path'],r['sha256']) for r in p['source_receipts']}
    if not {(r['path'],r['sha256']) for r in required}<=refs:
        raise ValueError('Training protocol omitted required worker/controller/mapper sources')
    jobs=value.get('jobs',[]);expected=[];plans={}
    for seed in SEEDS:
        folder=path.parent/('seed-'+str(seed));plan_path=folder/'plan.json'
        rawprep=folder/'native-prepared-plan.json';native=read(rawprep);plan=read(plan_path)
        if (native.get('plan_sha256')!=digest({k:v for k,v in native.items() if k!='plan_sha256'})
                or plan!=_project(native,value['protocol'],plan['created_ns'])):
            raise ValueError('Training plan must be exact role-bound native projection')
        flow_validation.validate_plan(plan)
        if (plan['seed']!=seed or plan['window_seconds']!=600. or plan['drain_seconds']!=300.
                or plan['warmup_seconds']!=0. or plan['workload_rps']!=POLICY['workload_rps']
                or len(plan['trace'])!=1200 or plan['context']!=value['context'] or plan['profile_plan']!=value['profile_plan']
                or plan['runner_source']!=receipt(native_flow.__file__)
                or tuple(c['id'] for c in plan['candidates'])!=ROUTES
                or any(type(x) is not int for x in (p['created_ns'],plan['created_ns'],value['created_ns']))
                or not p['created_ns']<plan['created_ns']<value['created_ns']):
            raise ValueError('Exact training seed/H600/drain300/warmup0/source/chronology required')
        checked(plan['catalog']);public={(r['task'],r['pool_id']) for r in profile['input_assignments'] if r['view']=='known_input'}
        if any((r['task'],r['pool_id']) not in public for r in plan['trace']):raise ValueError('Known-input public domain mismatch')
        for c in plan['candidates']:
            if (c['execution_context_sha256']!=context_identity(context)
                    or c['resource_configuration']!=native_flow.resource_configuration(2)
                    or c['routing_weights']!=_weights(c['id'])):
                raise ValueError('Shared/other-controller candidate forbidden in admission2 suite')
            from .flow_eligibility import _routing
            from .flow_capacity_catalog import condition
            from .flow_capacity_transfer import _generation
            _generation(plan,condition(plan,c))
            _routing(plan,c)
            jid='seed-'+str(seed)+'__'+c['id']+'__primary'
            expected.append(dict(job_id=jid,run_id=jid,seed=seed,candidate_id=c['id'],plan=receipt(plan_path),
                native_preparation=receipt(rawprep),campaign_role='primary_training',split='training',
                controller_sha256=plan['controller_binding']['controller_sha256'],
                output=str((folder/'runs'/c['id']).resolve())))
        plans[seed]=plan
    random.Random('admission2-training-v1-fixed-order').shuffle(expected)
    if jobs!=expected or len(jobs)!=8:raise ValueError('All eight unique fixed jobs, paths and order required')
    needed=_refs([required,value['protocol'],[j['plan'] for j in jobs],[j['native_preparation'] for j in jobs]])
    declared={(r['path'],r['sha256']) for r in value['source_receipts']}
    if not {(r['path'],r['sha256']) for r in needed}<=declared:
        raise ValueError('Campaign omitted required immutable source receipts')
    return value


def _job(job,campaign):
    value=verify(campaign)
    matches=[j for j in value['jobs'] if j['job_id']==(job if isinstance(job,str) else job.get('job_id'))]
    if len(matches)!=1 or isinstance(job,dict) and job!=matches[0]:
        raise ValueError('Exact original campaign job required; wrappers cannot change role/output/plan')
    return value,matches[0]


def _telemetry(raw):
    samples=raw['board_samples'];pid_alias={str(v['pid']):a for a,v in raw['ready'].items()}
    previous=None;gap=0
    for s in samples:
        stamp=mechanism_audit.ns(s,'time_ns')
        if previous is not None:
            if stamp<=previous:raise ValueError('Strictly ordered NVML samples required')
            gap=max(gap,stamp-previous)
        previous=stamp
        processes={str(k):v for k,v in s['processes_mib'].items()}
        for v in [s['watts'],s['owned_used_mib'],*processes.values()]:
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('Finite nonnegative telemetry required')
        if (s.get('foreign_pids') or not set(processes)<=set(pid_alias)
                or set(s.get('owned_aliases',[]))!={pid_alias[p] for p in processes}
                or abs(sum(processes.values())-s['owned_used_mib'])>1e-6
                or stamp>=raw['startup_end_ns'] and set(processes)!=set(pid_alias)):
            raise ValueError('Owned PID/alias telemetry mismatch')
    if (len(samples)<2 or gap>2_000_000_000
            or not samples[0]['time_ns']<=raw['startup_begin_ns']<=raw['observation_end_ns']<=samples[-1]['time_ns']):
        raise ValueError('Full startup-through-drain telemetry coverage with original 2s gap required')
    return samples,gap


def map_training(job,campaign):
    """Map only prospectively frozen primary training jobs; no validation adapter."""
    value,job=_job(job,campaign);campaign_ref=receipt(campaign)
    plan=read(checked(job['plan']));context=read(checked(plan['context']))
    output=Path(job['output']);raw_path=output/'raw.json';raw=read(raw_path)
    source=receipt(raw_path)
    if (any(k in raw for k in ('native_raw','derived_from','derivation','collection_role'))
            or raw.get('plan')!=job['plan'] or raw.get('candidate_id')!=job['candidate_id']
            or type(raw.get('startup_begin_ns')) is not int
            or not value['created_ns']<raw['startup_begin_ns']
            or raw.get('split') not in (None,'training')):
        raise ValueError('Immutable native training raw must start after complete campaign/role/source freeze')
    invocation_path=output/'invocation.json';invocation=read(invocation_path)
    if invocation!=dict(plan=job['plan'],candidate_id=job['candidate_id'],runner=plan['runner_source']):
        raise ValueError('Original native invocation differs from exact frozen training job')
    candidate=next(c for c in plan['candidates'] if c['id']==job['candidate_id'])
    if (raw.get('resource_configuration')!=plan['controller_binding']['resource_configuration']
            or validate_resources(raw['resource_configuration'],RESIDENTS)!=native_flow.resource_configuration(2)):
        raise ValueError('Training raw cannot inherit shared or another controller identity')
    audit=mechanism_audit.audit(raw_path)
    profile=read(checked(plan['profile_plan']))
    if ([(r['model'],r['task'],r['pool_id']) for r in raw['priming_request_records']]!=_priming(profile)
            or raw['warmup_request_records']):
        raise ValueError('Complete native largest-three priming and empty warmup0 history required')
    ready=raw['ready']
    for alias,info in ready.items():
        if (info.get('clock')!='time.perf_counter_ns'
                or info.get('spec_sha256')!=context['worker_bindings'][RESIDENTS[alias]]['spec_sha256']):
            raise ValueError('Native READY clock/spec differs from base training context')
    samples,gap=_telemetry(raw)
    startup,ready_time,left,deadline,right=(raw[k] for k in
        ('startup_begin_ns','startup_end_ns','phase_start_ns','deadline_ns','observation_end_ns'))
    by_id={r['request_id']:r for r in raw['request_records']}
    scope='admission_training:'+source['sha256'];requests=[]
    for offered in plan['trace']:
        row=by_id[offered['request_id']]
        if row.get('split') not in (None,'training'):raise ValueError('Validation/final row cannot become training')
        request=flow_profiles._request(row,ready,context,split='training',scope_id=scope,left=left,right=right)
        request.update(not_for_training=False,controller_sha256=job['controller_sha256'],
            resource_configuration_sha256=plan['controller_binding']['resource_configuration_sha256'],
            campaign_job_id=job['job_id'])
        requests.append(request)
    if any(r['parent_receipt_ns']>deadline+round(plan['drain_seconds']*1e9) for r in requests):
        raise ValueError('Training parent receipt exceeds original bounded drain')
    history=[]
    for key in ('priming_request_records','warmup_request_records'):
        for row in raw[key]:
            if row.get('split') not in (None,'excluded_prephase_history','training'):
                raise ValueError('Validation history cannot be relabelled for training')
            r=flow_profiles._request(row,ready,context,split='excluded_prephase_history',scope_id=scope,left=startup,right=left)
            r.update(not_for_service_training=True,not_for_training=True);history.append(r)
    for alias in RESIDENTS:
        rows=[r for r in requests if r['alias']==alias]
        if rows!=sorted(rows,key=lambda r:r['handler_start_ns']):raise ValueError('Per-alias offered FIFO order changed')
        all_rows=sorted((r for r in requests+history if r['alias']==alias),key=lambda r:r['handler_start_ns'])
        if any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(all_rows,all_rows[1:])):
            raise ValueError('Actual continuous native handlers overlap on same alias')
    selected_samples=[max((s for s in samples if s['time_ns']<=left),key=lambda s:s['time_ns']),
        *[s for s in samples if left<s['time_ns']<right],
        min((s for s in samples if s['time_ns']>=right),key=lambda s:s['time_ns'])]
    refs=_refs([source,job['plan'],job['native_preparation'],campaign_ref,value['source_receipts'],
        receipt(invocation_path),plan['controller_binding'],context,sources()])
    common=dict(split='training',execution_context_sha256=plan['execution_context_sha256'],
        base_worker_context=plan['context'],controller_binding=plan['controller_binding'],
        controller_sha256=job['controller_sha256'],resource_configuration=candidate['resource_configuration'],
        controller_execution_sha256=job['controller_sha256'],candidate_id=job['candidate_id'],training_seed=job['seed'],
        resource_configuration_sha256=digest(candidate['resource_configuration']),
        campaign=campaign_ref,campaign_job_id=job['job_id'],sources=refs,source_receipts=refs)
    trace=dict(**common,schema=flow_profiles.SCHEMA,kind='measured',backend='native',id=scope,
        not_for_training=False,split_basis='Immutable prospective training campaign and plan before native startup',
        clock='host_monotonic_ns',observation_window_ns=dict(start=left,end=right),
        admission_enabled=True,requests=requests,
        power_samples=[dict(time_ns=s['time_ns'],watts=s['watts']) for s in selected_samples],
        memory_samples=[{k:s[k] for k in ('time_ns','owned_used_mib','owned_aliases','foreign_pids')} for s in selected_samples],
        scope='One intact native admission2 training cohort; no artificial per-request solo trials')
    resource=dict(**common,measurement_window_ns=dict(start=left,deadline=deadline,observation_end=right),
        startup_window_ns=dict(start=startup,ready=ready_time,end=left),
        startup_joules=integrate_power(samples,startup,left),measurement_joules=integrate_power(samples,left,deadline),
        drain_joules=integrate_power(samples,deadline,right) if deadline<right else 0.,
        raw_all_sampled_owned_peak_mib=max(s['owned_used_mib'] for s in samples),
        raw_prephase_sampled_owned_peak_mib=max(s['owned_used_mib'] for s in samples if s['time_ns']<=left),
        raw_measurement_and_drain_sampled_owned_peak_mib=max(s['owned_used_mib'] for s in selected_samples),
        safe_factor_applied=False,capacity_bound=False,stage_history=history+requests,board_samples=copy.deepcopy(samples),
        scope='Raw owned sampled peaks and board energy; exact admission2 controller, no shared pooling')
    observations=[dict(request_id=r['native_request_id'],alias=r['alias'],family=r['family'],task=r['task'],pool_id=r['pool_id'],
        worker_service_ms=(r['handler_end_ns']-r['handler_start_ns'])/1e6,worker_start_ns=r['handler_start_ns'],
        service_ms=(r['handler_end_ns']-r['handler_start_ns'])/1e6,
        worker_end_ns=r['handler_end_ns'],handler_completed_in_measurement_window=r['handler_end_ns']<=deadline,
        controller_sha256=job['controller_sha256'],split='training') for r in requests]
    stability=flow_eligibility.native_stability(plan,candidate,raw,read(checked(plan['catalog'])))
    # Recheck identities after parsing so consumers cannot receive mixed versions.
    checked(source);verify(campaign)
    return dict(traces=[trace],resource_evidence=resource,service_observations=observations,
        plan=plan,candidate=candidate,native_raw=source,job=job,
        quality_evidence_status='NOT_AUDITED_BY_TRAINING_MAPPER',six_term_training_evidence_complete=False,
        mapping_report=dict(schema=SCHEMA+'.mapping',trace_count=1,continuous_cohort=True,request_count=len(requests),
            prephase_history_requests_excluded=len(history),split='training',not_for_training=False,
            candidate_id=candidate['id'],job_id=job['job_id'],controller_sha256=job['controller_sha256'],sources=refs,
            native_contract=audit,base_worker_and_controller_verified=True,full_offered_cohort_verified=True,
            actual_START_PHASE_verified=True,same_alias_FIFO_nonoverlap_verified=True,max_telemetry_gap_ns=gap,
            original_stability=stability,quality_evidence_status='NOT_AUDITED_BY_TRAINING_MAPPER',
            physical_service_transfer_validated=False,capacity_qualified=False,final_gain_eligible=False,GPU_started=False))


def _review(campaign_path,review_path):
    """Bind an explicit main-thread review; phase outcomes need not be positive."""
    value=read(review_path);campaign=verify(campaign_path)
    if (value.get('schema')!=SCHEMA+'.main_thread_review' or value.get('campaign')!=receipt(campaign_path)
            or value.get('decision')!='APPROVED_FOR_TRAINING_ONLY'
            or value.get('reviewer_role')!='main_thread' or value.get('final_gain_eligible') is not False
            or type(value.get('created_ns')) is not int or value['created_ns']<=campaign['created_ns']):
        raise ValueError('Exact campaign-bound explicit main-thread training review required')
    for key in ('stage2_report','stage2_gate','stage3_report'):checked(value[key])
    stage2=read(value['stage2_report']['path']);gate=read(value['stage2_gate']['path']);stage3=read(value['stage3_report']['path'])
    if (gate.get('report')!=value['stage2_report'] or gate.get('status') not in ('PASS','FAIL')
            or stage2.get('candidate_coverage')!={'requested':4,'observed':4}
            or len(stage2.get('candidates',[]))!=4 or any(r.get('known')!=1200 for r in stage2['candidates'])
            or stage3.get('schema')!='gsdeploy.H600_admission_service_pilot.v1.report'
            or stage3.get('status') not in ('DIAGNOSTIC_COMPLETE','DIAGNOSTIC_INCOMPLETE')
            or len(stage3.get('candidates',[]))!=3
            or {c.get('candidate_id') for c in stage3['candidates']}!={
                'shared_spill_low','admission1_spill_low','admission2_spill_low'}):
        raise ValueError('Complete stage2 assessment/gate and an explicit all-mode stage3 report must precede review')
    if (stage3['status']=='DIAGNOSTIC_INCOMPLETE' and
            (not isinstance(value.get('stage3_incomplete_training_reason'),str)
             or not value['stage3_incomplete_training_reason'].strip())):
        raise ValueError('Main-thread review must explain training after incomplete stage3; incompleteness remains preserved')
    _refs([stage2,gate,stage3])
    return value


def execute(campaign,job_id,review_path):
    value,job=_job(job_id,campaign);review=_review(campaign,review_path)
    if Path(job['output']).exists():raise FileExistsError('No retry or existing-output replacement permitted: '+job['output'])
    # Any earlier failed attempt stops this finite campaign. Earlier successful
    # jobs must be source-verified before advancing the declared order.
    for earlier in value['jobs']:
        if earlier['job_id']==job['job_id']:break
        done=Path(earlier['output'])/'training-complete.json'
        if not done.exists():raise ValueError('Predeclared job order required; preceding job is not complete')
        packet=read(done)
        if packet.get('campaign')!=receipt(campaign) or packet.get('job')!=earlier:
            raise ValueError('Prior job completion identity changed')
        checked(packet['native_raw']);map_training(earlier,campaign)
    # Atomic exclusive reservation survives any collector failure; do not retry.
    claim=Path(job['output']).parent/(job['candidate_id']+'.training-attempt.json')
    claim.parent.mkdir(parents=True,exist_ok=True)
    save(claim,dict(schema=SCHEMA+'.attempt',created_ns=time.perf_counter_ns(),job=job,
        campaign=receipt(campaign),review=receipt(review_path),source=receipt(__file__),GPU_started=False))
    _review(campaign,review_path)
    native_flow.execute(job['plan']['path'],job['candidate_id'],job['output'])
    mapped=map_training(job,campaign)
    result=dict(schema=SCHEMA+'.completed',created_ns=time.perf_counter_ns(),campaign=receipt(campaign),
        job=job,native_raw=mapped['native_raw'],review=receipt(review_path),source=receipt(__file__),
        request_count=len(mapped['service_observations']),controller_sha256=job['controller_sha256'],
        GPU_started=True,final_gain_eligible=False)
    save(Path(job['output'])/'training-complete.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--campaign',required=True)
    a=sub.add_parser('map');a.add_argument('--campaign',required=True);a.add_argument('--job-id',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('execute')
    for k in ('campaign','job-id','review'):a.add_argument('--'+k,required=True)
    a=p.parse_args()
    if a.op=='prepare':v=prepare(a.output)
    elif a.op=='verify':v=verify(a.campaign)
    elif a.op=='map':v=map_training(a.job_id,a.campaign);save(a.output,v)
    else:v=execute(a.campaign,a.job_id,a.review)
    print(json.dumps(dict(status='COMPLETE' if a.op=='execute' else a.op.upper(),
        GPU_started=a.op=='execute',final_gain_eligible=False)))


if __name__=='__main__':main()
