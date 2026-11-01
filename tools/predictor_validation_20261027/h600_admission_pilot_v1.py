"""Prospective fixed-route whole-request admission service-mechanism pilot.

Prepare/seal/report are CPU only. The separate, explicit run operation launches
GPU workers after the selected independent validation has finished. A failed
independent gate can only authorize a diagnostic pilot, never final comparison.
Shared service-law transfer is a hypothesis; no shared resource law is reused.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from . import flow_gate, flow_profiles, flow_validation, mechanism_audit, native_flow, flow_eligibility
from . import h600_selected_validation_v2 as selected
from .h600_event_simulator_v2 import FullHorizonModel
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest, RESIDENTS
from .validation import errors, percentile

SCHEMA = 'gsdeploy.H600_admission_service_pilot.v1'
IDS = ('shared_spill_low', 'admission1_spill_low', 'admission2_spill_low')
UNKNOWN = dict(startup='UNKNOWN', energy='UNKNOWN', memory='UNKNOWN',
               capacity_migration='UNKNOWN', six_term_utility='UNKNOWN', delta_U='UNKNOWN')
ROOT = Path(__file__).resolve().parents[2]
_PREREQUISITE_CACHE = {}


def _all_refs(value):
    """Collect actual receipt objects, including nested model provenance."""
    found = {}
    def visit(item):
        if isinstance(item, dict):
            if set(item) == {'path', 'sha256'}:
                checked(item); found[(item['path'], item['sha256'])] = item
            else:
                for child in item.values(): visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(value)
    return list(found.values())


def _receipt_closure(refs):
    """Snapshot every transitive JSON receipt, including execution snapshots.

    This permits reusing expensive numerical verification inside one process.
    Every hit still hashes the complete closure; changed data or code invalidates
    the cached proof. Nonreceipt path strings cannot authorize evidence.
    """
    pending = list(refs); found = {}
    while pending:
        ref = pending.pop(); key = (ref['path'],ref['sha256'])
        checked(ref)
        if key in found: continue
        found[key] = ref
        if Path(ref['path']).suffix=='.json':
            pending.extend(_all_refs(read(ref['path'])))
    return list(found.values())


def _controller(plan, candidate):
    value = dict(base_worker_context=plan['context'],
        base_worker_context_sha256=plan['execution_context_sha256'],
        resource_configuration=candidate['resource_configuration'],
        resource_configuration_sha256=digest(candidate['resource_configuration']),
        runtime_semantics_sources=[receipt(ROOT / name) for name in mechanism_audit.SEMANTIC_PATHS],
        adapter_source=receipt(__file__), audit_source=receipt(mechanism_audit.__file__))
    return {**value, 'controller_sha256': digest(value)}


def _project(original, created_ns):
    plan = copy.deepcopy(original)
    for key in ('plan_sha256', 'startup_comparison_policy', 'prephase_memory_evidence'):
        plan.pop(key, None)
    plan['candidates'] = [c for cid in IDS for c in plan['candidates'] if c['id'] == cid]
    for c in plan['candidates']:
        c.pop('predicted_prephase_peak_mib', None)
        c.pop('predicted_startup_joules', None)
    plan.update(created_ns=created_ns, phase='mechanism_validation', split='validation',
        baseline_candidate_id=IDS[0], scope='Fixed spill_low route service-only admission pilot',
        final_gain_eligible=False, resource_model_transfer='NOT_AUTHORIZED',
        unsupported_predictions=UNKNOWN.copy())
    plan['controller_bindings'] = {c['id']: _controller(plan, c) for c in plan['candidates']}
    plan['plan_sha256'] = digest(plan)
    return plan


def audit_plan(path):
    plan = flow_validation.validate_plan(read(path))
    original_path = Path(path).with_name('native-prepared-plan.json')
    original = read(original_path)
    if (digest({k:v for k,v in original.items() if k != 'plan_sha256'}) != original.get('plan_sha256')
            or plan != _project(original, plan['created_ns'])):
        raise ValueError('Pilot must be exact structural projection of retained native preparation')
    if (tuple(c['id'] for c in plan['candidates']) != IDS or plan['seed'] != 627801
            or plan['window_seconds'] != 600. or plan['drain_seconds'] != 300.
            or plan['warmup_seconds'] != 0. or len(plan['trace']) != 1200
            or plan['workload_rps'] != dict(text=1.85,image=.05,asr=.05,ocr=.05)
            or {a:v['family'] for a,v in plan['aliases'].items()} != RESIDENTS):
        raise ValueError('Fixed seed627801 H600/drain300/warmup0/residency/cohort required')
    common = plan['candidates'][0]
    for c, cap in zip(plan['candidates'], (None,1,2)):
        if (c['routes'] != common['routes'] or c['routing_weights'] != common['routing_weights']
                or c['resource_configuration'] != native_flow.resource_configuration(cap)
                or c['execution_context_sha256'] != plan['execution_context_sha256']):
            raise ValueError('Identical per-request routes and exact native controller modes required')
    context = read(checked(plan['context']))
    if (native_flow.context_identity(context) != plan['execution_context_sha256']
            or context.get('residents') != RESIDENTS):
        raise ValueError('Immutable fixed base worker identity required')
    snapshots = context.get('execution_sources', [])
    if not snapshots or context.get('sources') != [s['frozen'] for s in snapshots]:
        raise ValueError('Complete immutable base worker snapshots required')
    for source in snapshots:
        checked(source['frozen'])
        if receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
            raise ValueError('Current worker source differs from frozen base context')
    for name in mechanism_audit.SEMANTIC_PATHS:
        source = [s for s in snapshots if Path(s['original_path']).resolve() == ROOT/name]
        if len(source) != 1 or source[0]['frozen']['sha256'] != receipt(ROOT/name)['sha256']:
            raise ValueError('Base context lacks exact current controller semantics')
    for key in ('profile_plan','catalog','runner_source'): checked(plan[key])
    if plan['runner_source'] != receipt(native_flow.__file__):
        raise ValueError('Exact native runner required')
    profile = read(checked(plan['profile_plan']))
    public = {(r['task'],r['pool_id']) for r in profile['input_assignments'] if r['view']=='known_input'}
    if (profile['execution_context_sha256'] != plan['execution_context_sha256']
            or profile.get('plan_sha256') != digest({k:v for k,v in profile.items() if k!='plan_sha256'})
            or any((r['task'],r['pool_id']) not in public for r in plan['trace'])):
        raise ValueError('Exact known-input public profile domain required')
    return plan


def prepare(profile_dir, output):
    """Declare only future public inputs/controllers; reads no independent outcomes."""
    output = Path(output).resolve()
    native_flow.prepare(profile_dir, output, duration=600., warmup=0., drain=300.,
        seed=627801, include_admission=True)
    (output/'plan.json').rename(output/'native-prepared-plan.json')
    plan = _project(read(output/'native-prepared-plan.json'), time.perf_counter_ns())
    save(output/'plan.json', plan); audit_plan(output/'plan.json')
    save(output/'preparation.json', dict(schema=SCHEMA+'.preparation',
        created_ns=time.perf_counter_ns(), plan=receipt(output/'plan.json'),
        native_preparation=receipt(output/'native-prepared-plan.json'), entry_source=receipt(__file__),
        GPU_started=False, independent_outcomes_read=False, final_gain_eligible=False))
    return plan


def audit_prerequisite(protocol_path, model_path, report_path, gate_path):
    """Require complete source-bound independent evidence, preserving a failed gate."""
    direct = [receipt(p) for p in (protocol_path,model_path,report_path,gate_path,selected.__file__,__file__)]
    key = digest(direct)
    if key in _PREREQUISITE_CACHE:
        result = _PREREQUISITE_CACHE[key]
        for ref in result['source_receipts']: checked(ref)
        return copy.deepcopy(result)
    protocol = selected.audit_protocol(protocol_path)
    report = read(report_path); gate = read(gate_path)
    report_audit_path = Path(report_path).parent/'training-audit.json'
    report_audit = read(report_audit_path)
    if (report_audit.get('schema') != selected.AUDIT+'.report'
            or report_audit.get('entry_source') != receipt(selected.__file__)
            or report_audit.get('report') != receipt(report_path)
            or report_audit.get('validation_protocol') != receipt(protocol_path)):
        raise ValueError('Selected independent report audit required')
    folder = Path(checked(report_audit['sealed_audit'])).parent
    seal_audit = selected.check_sealed(folder)
    verification = seal_audit['verification']
    sealed = read(folder/'predictions.json')
    if (sealed['sources']['model'] != receipt(model_path)
            or report['sealed'] != receipt(folder/'predictions.json')
            or report.get('report_source') != receipt(flow_validation.__file__)):
        raise ValueError('Pilot must reuse exact independently validated selected model')
    _, evidence = selected.validation_evidence([checked(r) for r in report['raw_sources']], sealed)
    if evidence != report_audit.get('strict_native_validation_evidence'):
        raise ValueError('Independent strict native evidence changed')
    target = copy.deepcopy(selected.audit_plan(protocol['target_plan']['path'], protocol_path))
    target['validation_thresholds'] = selected.LIMITS
    expected = flow_gate.assess(target, report)
    expected.update(plan=protocol['target_plan'], validation_protocol=receipt(protocol_path),
        report=receipt(report_path), tool=receipt(selected.__file__), final_gain_eligible=False)
    if gate != expected or gate['status'] not in ('PASS','FAIL'):
        raise ValueError('Complete recomputed independent PASS/FAIL gate required; INCOMPLETE blocks pilot')
    rows = report['candidates']
    if (len(rows) != 4 or len({r['candidate_id'] for r in rows}) != 4
            or any(r['requested'] != 1200 or r['known'] != 1200 for r in rows)):
        raise ValueError('Complete independent four-route/full-cohort report required')
    refs = _all_refs([protocol, report, report_audit, gate, sealed,
        read(folder/'training-audit.json'), verification])
    refs = _receipt_closure([*refs,*direct,receipt(report_audit_path),*selected.predictor_sources()])
    result = dict(status=gate['status'], selected_estimator=protocol['estimator_selection'],
        model_verification=verification, protocol=receipt(protocol_path), report=receipt(report_path),
        gate=receipt(gate_path), sealed=receipt(folder/'predictions.json'), source_receipts=refs,
        scope='Diagnostic service mechanism only', final_gain_eligible=False)
    _PREREQUISITE_CACHE[key] = copy.deepcopy(result)
    return result


class UnknownResources:
    def power(self,*args): return {'known':False,'watts':None}
    def memory(self,*args): return {'known':False,'peak_memory_mib':None}


def _task_p95(plan, rows):
    result = {}
    for task in plan['workload_rps']:
        cohort = [r for r in rows if r['task']==task]
        requested = sum(r['task']==task for r in plan['trace'])
        known = sum(r.get('e2e_ms') is not None for r in cohort)
        result[task] = dict(requested=requested, known=known,
            p95_ms=percentile([r['e2e_ms'] for r in cohort]) if known==requested and requested else None)
    return result


def predict(plan, candidate, model, *, provider=None):
    """Service-only prospective rollout; unsupported resources stay UNKNOWN."""
    queues = {a:[] for a in plan['aliases']}; future = []
    for item in plan['trace']:
        row = {k:item[k] for k in ('request_id','task','pool_id')};row['arrival_ns']=item['offset_ns']
        (queues[candidate['routes'][row['request_id']]] if not row['arrival_ns'] else future).append(row)
    forecast = None; reason = None
    end = round(plan['window_seconds']*1e9)
    try:
        provider = provider or flow_validation._provider(model,candidate['execution_context_sha256'])
        unknown = UnknownResources()
        forecast = FullHorizonModel(plan['aliases'],provider,unknown,memory_provider=unknown,
            resources=candidate['resource_configuration'],max_requests=plan.get('max_forecast_requests',20000)).forecast(
                queues,0,end,future_arrivals=future,
                route_future=lambda row,pending,now:candidate['routes'][row['request_id']],
                require_energy=False,require_memory=False,
                drain_deadline_ns=end+round(plan['drain_seconds']*1e9))
    except (UnknownEvidence,ValueError) as exc:
        reason = type(exc).__name__+':'+str(exc)
    rows = []
    for item in plan['trace']:
        f = forecast['finishes'].get(item['request_id']) if forecast else None
        rows.append(dict(request_id=item['request_id'], task=item['task'],pool_id=item['pool_id'],
            alias=candidate['routes'][item['request_id']],
            service_ms=(f['finish_ns']-f['start_ns'])/1e6 if f else None,
            wait_ms=(f['start_ns']-item['offset_ns'])/1e6 if f else None,
            e2e_ms=(f['finish_ns']-item['offset_ns'])/1e6 if f else None,
            start_ns=f['start_ns'] if f else None,finish_ns=f['finish_ns'] if f else None))
    known = sum(r['e2e_ms'] is not None for r in rows)
    admission = candidate['resource_configuration']['mode']=='request_admission'
    cap = candidate['resource_configuration']['groups']['gpu']['max_active'] if admission else None
    occupied = sum((t['until_ns']-t['time_ns'])*len(t['active_requests'])/1e6
        for t in forecast['transitions']) if forecast else None
    return dict(candidate_id=candidate['id'],requested=len(rows),known=known,requests=rows,
        per_task=_task_p95(plan,rows),queue_samples=flow_validation._queue_samples(plan,forecast) if forecast else [],
        unknown_reason=reason,incomplete_ids=[r['request_id'] for r in rows if r['e2e_ms'] is None],
        execution_precheck=('UNKNOWN_PREDICTION' if forecast is None else
            'PREDICTED_COMPLETE' if known==len(rows) else 'PREDICTED_DRAIN_INFEASIBLE'),
        permit_demand=dict(handler_time_within_bounded_rollout_ms=occupied,
            complete_handler_service_sum_ms=sum(r['service_ms'] for r in rows if r['service_ms'] is not None),
            configured_global_cap=cap,available_permit_ms=cap*(plan['window_seconds']+plan['drain_seconds'])*1000 if cap else None,
            unfinished_cohort=len(rows)-known,full_cohort_demand_known=known==len(rows),
            interpretation='Handler occupancy under simulated release-at-END rule; unfinished work not extrapolated or dropped'),
        unsupported_predictions=UNKNOWN.copy(),service_transfer='SHARED_TO_ADMISSION_UNVALIDATED_HYPOTHESIS',
        service_definition='Worker START to handler END',wait_definition='Scheduled offer to worker START',
        e2e_definition='Predicted handler END minus scheduled offer; actual uses parent receipt',
        admission_model_limits='Simulated alias-order offers and release at handler END; actual ticket/release boundaries are not reconstructed',
        numerics_source=receipt(FullHorizonModel.forecast.__code__.co_filename),original_event_budget=100000)


def seal(plan_path,model_path,protocol_path,report_path,gate_path,output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    plan = audit_plan(plan_path)
    prerequisite = audit_prerequisite(protocol_path,model_path,report_path,gate_path)
    model = read(model_path)
    if model['execution_context_sha256'] != plan['execution_context_sha256']:
        raise ValueError('Selected service model and immutable base worker context differ')
    sources = dict(plan=receipt(plan_path),model=receipt(model_path),protocol=receipt(protocol_path),
        independent_report=receipt(report_path),independent_gate=receipt(gate_path))
    context = read(checked(plan['context']))
    refs = _all_refs([sources,plan,context,model,prerequisite,
        [receipt(p) for p in (Path(plan_path).with_name('native-prepared-plan.json'),
        __file__,native_flow.__file__,mechanism_audit.__file__,flow_profiles.__file__,flow_eligibility.__file__,
        ROOT/'gsdeploy/stability.py',
        Path(flow_profiles.__file__).with_name('background_profiles.py'),
        Path(flow_profiles.__file__).with_name('semantic_profiles.py'))],selected.predictor_sources()])
    cutoff = time.perf_counter_ns()
    predictions = [predict(plan,c,model) for c in plan['candidates']]
    for ref in refs: checked(ref)
    audit_plan(plan_path); audit_prerequisite(protocol_path,model_path,report_path,gate_path)
    value = dict(schema=SCHEMA+'.sealed',created_ns=time.perf_counter_ns(),information_cutoff_ns=cutoff,
        sources=sources,source_receipts=refs,entry_source=receipt(__file__),prerequisite=prerequisite,
        controller_bindings=plan['controller_bindings'],predictions=predictions,
        GPU_started=False,actual_future_outcomes_used=False,final_gain_eligible=False,
        unsupported_predictions=UNKNOWN.copy())
    value['seal_sha256']=digest(value)
    output.mkdir(parents=True,exist_ok=False);save(output/'predictions.json',value)
    return value


def check_sealed(folder):
    value = read(Path(folder)/'predictions.json')
    if (value.get('schema') != SCHEMA+'.sealed' or value.get('entry_source') != receipt(__file__)
            or value.get('seal_sha256') != digest({k:v for k,v in value.items() if k!='seal_sha256'})):
        raise ValueError('Exact immutable admission source/prediction seal required')
    for ref in value['source_receipts']: checked(ref)
    s = value['sources'];plan = audit_plan(checked(s['plan']))
    prerequisite = audit_prerequisite(checked(s['protocol']),checked(s['model']),
        checked(s['independent_report']),checked(s['independent_gate']))
    if (value['prerequisite'] != prerequisite or value['controller_bindings'] != plan['controller_bindings']
            or not plan['created_ns'] <= value['information_cutoff_ns'] <= value['created_ns']
            or tuple(p['candidate_id'] for p in value['predictions']) != IDS
            or any(p['requested'] != 1200 or len(p['requests']) != 1200 for p in value['predictions'])):
        raise ValueError('Pilot seal source/protocol/full-cohort contract changed')
    if not {(r['path'],r['sha256']) for r in _all_refs([s,plan,read(s['model']['path']),prerequisite])} <= {
            (r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Seal omitted required immutable source receipts')
    return value


def activation(audit, candidate):
    admission = candidate['resource_configuration']['mode']=='request_admission'
    output = {}
    for scope in ('measurement','whole_run'):
        rows = [r for r in audit['requests'] if scope=='whole_run' or r['phase']=='measurement']
        if not admission:
            output[scope] = dict(status='NOT_APPLICABLE',observed_peak=None,positive_wait_count=None)
            continue
        cap = candidate['resource_configuration']['groups']['gpu']['max_active']
        peak = mechanism_audit.peak([(r['grant_ns'],r['parent_receipt_ns']) for r in rows])
        waiting = [r for r in rows if r['grant_ns']>r['wait_start_ns']]
        # A positive timestamp gap alone includes acquire/clock overhead. Actual
        # contention requires *cap* other observed permits simultaneously inside
        # this wait. One other alias cannot establish blocking at max_active=2.
        competing = [r for r in waiting if any(s['alias']!=r['alias'] and
            max(r['wait_start_ns'],s['grant_ns']) < min(r['grant_ns'],s['parent_receipt_ns']) for s in rows)]
        saturated = [r for r in waiting if mechanism_audit.peak([
            (max(r['wait_start_ns'],s['grant_ns']),min(r['grant_ns'],s['parent_receipt_ns']))
            for s in rows if s['alias']!=r['alias'] and
            max(r['wait_start_ns'],s['grant_ns']) < min(r['grant_ns'],s['parent_receipt_ns'])])>=cap]
        active = peak==cap and bool(saturated)
        output[scope] = dict(status='OBSERVED_ACTIVATION' if active else 'INSUFFICIENT_ACTIVATION',
            max_active=cap,observed_peak=peak,positive_wait_count=len(waiting),
            observed_cap_reached=peak==cap,wait_with_competing_alias_count=len(competing),
            wait_under_observed_saturation_count=len(saturated),
            admission_wait_ms=mechanism_audit.summary([(r['grant_ns']-r['wait_start_ns'])/1e6 for r in rows]))
    return output


def actual_evidence(raw_path,sealed):
    """Controller-aware validation adapter; never invokes the shared-only mapper."""
    raw_path = Path(raw_path).resolve();raw = read(raw_path)
    plan = audit_plan(checked(sealed['sources']['plan']))
    if (raw.get('plan') != sealed['sources']['plan'] or raw.get('native_raw')
            or type(raw.get('startup_begin_ns')) is not int or raw['startup_begin_ns']<=sealed['created_ns']):
        raise ValueError('Immutable native raw must start after exact prospective source/prediction seal')
    candidate = next((c for c in plan['candidates'] if c['id']==raw.get('candidate_id')),None)
    if candidate is None or sealed['controller_bindings'][candidate['id']] != _controller(plan,candidate):
        raise ValueError('Actual controller identity differs from separate prospective binding')
    audit = mechanism_audit.audit(raw_path)
    context = read(checked(plan['context']));ready = raw['ready']
    invocation_path = raw_path.parent/'invocation.json';invocation = read(invocation_path)
    if invocation != dict(plan=raw['plan'],candidate_id=candidate['id'],runner=plan['runner_source']):
        raise ValueError('Prelaunch invocation/source mismatch')
    for alias,info in ready.items():
        if (info.get('clock')!='time.perf_counter_ns'
                or info.get('spec_sha256')!=context['worker_bindings'][RESIDENTS[alias]]['spec_sha256']):
            raise ValueError('Actual READY clock/spec differs from immutable worker binding')
    startup,left,right,deadline = (raw[k] for k in ('startup_begin_ns','phase_start_ns','observation_end_ns','deadline_ns'))
    samples = raw['board_samples'];previous = None;maxgap = 0
    pid_alias = {str(info['pid']):alias for alias,info in ready.items()}
    for sample in samples:
        timestamp = mechanism_audit.ns(sample,'time_ns')
        if previous is not None:
            if timestamp<=previous: raise ValueError('Strict telemetry timestamp order required')
            maxgap=max(maxgap,timestamp-previous)
        previous=timestamp
        for field in ('watts','owned_used_mib'):
            x=sample[field]
            if type(x) not in (int,float) or not math.isfinite(x) or x<0: raise ValueError('Invalid owned telemetry')
        processes = {str(k):v for k,v in sample['processes_mib'].items()}
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in processes.values())
                or sample.get('foreign_pids') or not set(processes)<=set(pid_alias)
                or set(sample.get('owned_aliases',[]))!={pid_alias[p] for p in processes}
                or abs(sum(processes.values())-sample['owned_used_mib'])>1e-6
                or timestamp>=raw['startup_end_ns'] and set(processes)!=set(pid_alias)):
            raise ValueError('Owned telemetry PID/alias/sum mismatch')
    if len(samples)<2 or maxgap>2_000_000_000 or not samples[0]['time_ns']<=startup<=right<=samples[-1]['time_ns']:
        raise ValueError('Complete startup-through-drain telemetry bracketing required')
    by_id = {r['request_id']:r for r in raw['request_records']}
    scope = 'admission_validation:'+receipt(raw_path)['sha256']
    semantic = [flow_profiles._request(by_id[r['request_id']],ready,context,split='validation',
        scope_id=scope,left=left,right=right) for r in plan['trace']]
    for r in semantic: r['not_for_training']=True
    if any(r['parent_receipt_ns']>deadline+round(plan['drain_seconds']*1e9) for r in semantic):
        raise ValueError('Parent receipt outside declared bounded drain')
    history = [flow_profiles._request(r,ready,context,split='excluded_prephase_history',
        scope_id=scope,left=startup,right=left) for key in ('priming_request_records','warmup_request_records') for r in raw[key]]
    for alias in RESIDENTS:
        ordered = [r for r in semantic if r['alias']==alias]
        if ordered!=sorted(ordered,key=lambda r:r['handler_start_ns']): raise ValueError('Per-worker offered FIFO changed')
        all_rows = sorted((r for r in semantic+history if r['alias']==alias),key=lambda r:r['handler_start_ns'])
        if any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(all_rows,all_rows[1:])):
            raise ValueError('Per-worker actual handlers overlap')
    rows = [dict(request_id=r['native_request_id'],task=r['task'],pool_id=r['pool_id'],alias=r['alias'],
        service_ms=(r['handler_end_ns']-r['handler_start_ns'])/1e6,
        wait_ms=(r['handler_start_ns']-r['arrival_ns'])/1e6,
        e2e_ms=(r['parent_receipt_ns']-r['arrival_ns'])/1e6,
        start_ns=r['handler_start_ns']-left,finish_ns=r['parent_receipt_ns']-left) for r in semantic]
    queue = [dict(offset_ns=t,depth=sum(r['arrival_ns']<=left+t<r['parent_receipt_ns'] for r in semantic))
        for t in plan['queue_sample_offsets_ns']]
    stability = flow_eligibility.native_stability(plan,candidate,raw,read(checked(plan['catalog'])))
    return dict(candidate_id=candidate['id'],native_raw=receipt(raw_path),invocation=receipt(invocation_path),
        base_worker_context=plan['context'],controller_binding=plan['controller_bindings'][candidate['id']],
        requested=len(plan['trace']),known=len(rows),requests=rows,per_task=_task_p95(plan,rows),
        queue_samples=queue,mechanism_audit=audit,activation=activation(audit,candidate),
        original_stability=stability,rho_admission_passed=None,capacity_migration='UNKNOWN',
        semantic_validation_trace=dict(split='validation',not_for_training=True,requests=semantic),
        telemetry=dict(samples=len(samples),maximum_gap_ns=maxgap,attribution='Owned PID telemetry; no resource-law transfer'),
        unsupported_predictions=UNKNOWN.copy(),final_gain_eligible=False)


def run(sealed_dir, output, *, diagnostic=False):
    """Explicit GPU launch, sequential frozen order, no retries or seed replacement."""
    sealed = check_sealed(sealed_dir)
    if sealed['prerequisite']['status']=='FAIL' and not diagnostic:
        raise ValueError('Independent gate FAIL: only explicitly diagnostic mechanism execution is permitted')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    state = dict(schema=SCHEMA+'.run',created_ns=time.perf_counter_ns(),
        seal=receipt(Path(sealed_dir)/'predictions.json'),entry_source=receipt(__file__),
        diagnostic=True,independent_gate_status=sealed['prerequisite']['status'],
        candidate_order=list(IDS),attempted=[],completed=[],skipped=[],status='STARTED',
        GPU_started=False,final_gain_eligible=False)
    save(output/'started.json',state)
    try:
        for cid in IDS:
            check_sealed(sealed_dir)
            prediction = next(p for p in sealed['predictions'] if p['candidate_id']==cid)
            if prediction['execution_precheck']!='PREDICTED_COMPLETE':
                state['skipped'].append(dict(candidate_id=cid,reason=prediction['execution_precheck'],
                    requested=prediction['requested'],predicted_known=prediction['known']))
                save(output/('skipped-'+cid+'.json'),state['skipped'][-1]);continue
            state['attempted'].append(cid);state['GPU_started']=True;save(output/('attempt-'+cid+'.json'),state)
            native_flow.execute(sealed['sources']['plan']['path'],cid,output/cid)
            actual_evidence(output/cid/'raw.json',sealed)
            state['completed'].append(cid);save(output/('complete-'+cid+'.json'),state)
        check_sealed(sealed_dir);state['status']='INCOMPLETE_DECLARED_MODES_SKIPPED' if state['skipped'] else 'COMPLETED'
    except Exception as exc:
        state.update(status='INCOMPLETE_NO_RETRY',error=type(exc).__name__+':'+str(exc))
        save(output/'run-state.json',state);raise
    save(output/'run-state.json',state);return state


def report(sealed_dir, raw_paths, output):
    sealed = check_sealed(sealed_dir);plan = audit_plan(sealed['sources']['plan']['path'])
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    supplied = {}; raw_refs = []
    for path in raw_paths:
        raw=read(path);cid=raw.get('candidate_id')
        if cid not in IDS or cid in supplied: raise ValueError('No foreign or repeated pilot raw candidates permitted')
        if raw.get('plan')!=sealed['sources']['plan']: raise ValueError('Raw uses different pilot plan')
        supplied[cid]=path;raw_refs.append(receipt(path))
    candidates = []
    for prediction in sealed['predictions']:
        cid=prediction['candidate_id'];observed=None;reason=None
        if cid not in supplied:
            reason=prediction['execution_precheck'] if prediction['execution_precheck']!='PREDICTED_COMPLETE' else 'MISSING_DECLARED_CANDIDATE'
        else:
            try: observed=actual_evidence(supplied[cid],sealed)
            except (ValueError,KeyError,TypeError) as exc: reason=type(exc).__name__+':'+str(exc)
        actual={r['request_id']:r for r in observed['requests']} if observed else {}
        timing={field:errors([(r[field],actual[r['request_id']][field]) for r in prediction['requests']
            if r[field] is not None and r['request_id'] in actual]) for field in ('service_ms','wait_ms','e2e_ms')}
        predicted_queue={r['offset_ns']:r['depth'] for r in prediction['queue_samples']}
        actual_queue={r['offset_ns']:r['depth'] for r in observed['queue_samples']} if observed else {}
        queue_errors=errors([(v,actual_queue[t]) for t,v in predicted_queue.items() if t in actual_queue])
        p95={}
        for task in plan['workload_rps']:
            p=prediction['per_task'][task]['p95_ms'];o=observed['per_task'][task]['p95_ms'] if observed else None
            p95[task]=dict(predicted_ms=p,observed_ms=o,absolute_relative_error=abs(p-o)/o if p is not None and o else None)
        paired=timing['e2e_ms']['count'];complete=(paired==len(plan['trace']) and
            prediction['known']==len(plan['trace']) and observed is not None and observed['known']==len(plan['trace']))
        candidates.append(dict(candidate_id=cid,requested=len(plan['trace']),predicted_known=prediction['known'],
            observed_known=observed['known'] if observed else 0,paired_known=paired,complete=complete,
            execution_precheck=prediction['execution_precheck'],permit_demand=prediction['permit_demand'],
            timing_errors=timing,queue_error=queue_errors,per_task_p95=p95,
            activation=observed['activation'] if observed else None,evidence_error=reason,
            original_stability=observed['original_stability'] if observed else None,
            actual_evidence=observed,unsupported_predictions=UNKNOWN.copy()))
    check_sealed(sealed_dir)
    for ref in raw_refs: checked(ref)
    value=dict(schema=SCHEMA+'.report',created_ns=time.perf_counter_ns(),
        sealed=receipt(Path(sealed_dir)/'predictions.json'),entry_source=receipt(__file__),raw_sources=raw_refs,
        status='DIAGNOSTIC_COMPLETE' if all(c['complete'] for c in candidates) else 'DIAGNOSTIC_INCOMPLETE',
        complete_three_mode_mechanism_conclusion=('OBSERVED_ACTIVATION_ONLY' if
            all(c['complete'] for c in candidates) and all(c['activation']['measurement']['status']=='OBSERVED_ACTIVATION'
                for c in candidates if c['candidate_id']!=IDS[0]) else 'INCOMPLETE_OR_INSUFFICIENT_ACTIVATION'),
        independent_gate_status=sealed['prerequisite']['status'],candidates=candidates,
        requested=sum(c['requested'] for c in candidates),paired_known=sum(c['paired_known'] for c in candidates),
        no_accuracy_gate_predeclared=True,unsupported_predictions=UNKNOWN.copy(),final_gain_eligible=False,
        scope='Service transfer and observed controller activation only; no final routing gain or six-term ranking')
    output.mkdir(parents=True,exist_ok=False);save(output/'report.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for key in ('profile-dir','output'): a.add_argument('--'+key,required=True)
    a=sub.add_parser('seal')
    for key in ('plan','model','selected-protocol','independent-report','independent-gate','output'):
        a.add_argument('--'+key,required=True)
    a=sub.add_parser('check');a.add_argument('--sealed-dir',required=True)
    a=sub.add_parser('run');a.add_argument('--sealed-dir',required=True);a.add_argument('--output',required=True)
    a.add_argument('--diagnostic',action='store_true')
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True)
    a.add_argument('--raw',action='append',default=[]);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.op=='prepare': value=prepare(a.profile_dir,a.output)
    elif a.op=='seal': value=seal(a.plan,a.model,a.selected_protocol,a.independent_report,a.independent_gate,a.output)
    elif a.op=='check': value=check_sealed(a.sealed_dir)
    elif a.op=='run': value=run(a.sealed_dir,a.output,diagnostic=a.diagnostic)
    else: value=report(a.sealed_dir,a.raw,a.output)
    print(json.dumps(dict(status=value.get('status','PREPARED' if a.op=='prepare' else 'SEALED'),
        GPU_started=value.get('GPU_started',False),final_gain_eligible=False)))


if __name__=='__main__': main()
