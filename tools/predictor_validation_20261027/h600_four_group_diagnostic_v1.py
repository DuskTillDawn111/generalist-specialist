"""Prospective eight-configuration negative-result comparison, CPU preparation.

The fixed shared service predictor has failed independent checks. These final
blocks are diagnostic, never qualification or a positive routing-gain claim.
No GPU operation is exposed until upstream assessment interfaces are connected.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from . import admission_training_v1 as training
from . import flow_eligibility, flow_metrics, flow_profiles, flow_validation, mechanism_audit, native_flow
from . import h600_four_group_statistics_v1 as statistics
from .flow_capacity_catalog import condition
from .flow_capacity_transfer import _generation
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest, RESIDENTS
from .training_run_audit import verify_quality_annotation

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'gsdeploy.H600_four_group_diagnostic.v1'
SEED = 628301
IDS = tuple(m+'_'+r for m in statistics.MODES for r in statistics.ROUTES)
LOCKS = {m: dict(no_spill='balanced', allow='spill_high') for m in statistics.MODES}
# Paired modes alternate first position across successive route pairs. This
# deterministic order is fixed before any final outcome or final GPU startup.
ORDER = ('shared_spill_high', 'admission2_spill_high',
         'admission2_balanced', 'shared_balanced',
         'shared_skewed', 'admission2_skewed',
         'admission2_spill_low', 'shared_spill_low')
RESERVATION = ROOT/'artifacts/predictor-gpu-20261027/admission-next-stage-roles-v1.json'


def sources():
    import gsdeploy.metrics, gsdeploy.energy, gsdeploy.quality, gsdeploy.tasks
    return [receipt(m.__file__) for m in (training, flow_eligibility, flow_metrics,
        flow_profiles, flow_validation, mechanism_audit, native_flow, statistics,
        gsdeploy.metrics, gsdeploy.energy, gsdeploy.quality, gsdeploy.tasks)] + [
        receipt(__file__), receipt(verify_quality_annotation.__code__.co_filename),
        receipt(condition.__code__.co_filename),receipt(_generation.__code__.co_filename)]


def _binding(plan, candidate):
    if candidate['id'].startswith('admission2_'):
        return training.controller(plan['context'], plan['execution_context_sha256'])
    body = dict(base_worker_context=plan['context'],
        base_worker_context_sha256=plan['execution_context_sha256'],
        resource_configuration=candidate['resource_configuration'],
        resource_configuration_sha256=digest(candidate['resource_configuration']),
        adapter_source=receipt(__file__),
        runtime_semantics_sources=[receipt(ROOT/p) for p in mechanism_audit.SEMANTIC_PATHS])
    return {**body, 'controller_sha256': digest(body)}


def _project(original, created_ns):
    plan = copy.deepcopy(original)
    for key in ('plan_sha256', 'startup_comparison_policy', 'prephase_memory_evidence'):
        plan.pop(key, None)
    plan['candidates'] = [c for cid in IDS for c in plan['candidates'] if c['id']==cid]
    for candidate in plan['candidates']:
        candidate.pop('predicted_startup_joules', None)
        candidate.pop('predicted_prephase_peak_mib', None)
    plan.update(created_ns=created_ns, phase='final_diagnostic', split='final',
        kind='prospective_four_group_diagnostic', role_reservation=receipt(RESERVATION),
        entry_source=receipt(__file__), locked_policies=copy.deepcopy(LOCKS),
        candidate_order=list(ORDER), baseline_candidate_id='shared_balanced',
        controller_bindings={c['id']:_binding(plan,c) for c in plan['candidates']},
        no_retries=True, no_seed_replacement=True, final_gain_eligible=False,
        statistical_significance_claim=False, independent_blocks=1,
        policy_scope='Predetermined balanced/high diagnostic policies in both modes; no admission optimum asserted',
        strong_baseline_scope='Both declared no-spill routes measured; per-block strongest observed finite baseline',
        scope='Full six-term descriptive negative-result comparison; not a qualified optimized deployment')
    plan['plan_sha256'] = digest(plan)
    return plan


def audit_plan(path):
    plan = flow_validation.validate_plan(read(path))
    original_path = Path(path).with_name('native-prepared-plan.json')
    original = read(original_path)
    if (original.get('plan_sha256')!=digest({k:v for k,v in original.items() if k!='plan_sha256'})
            or plan != _project(original, plan['created_ns'])):
        raise ValueError('Exact prospectively fixed final diagnostic projection required')
    reservation = read(checked(plan['role_reservation']))
    if (reservation.get('schema')!='gsdeploy.H600_admission_next_stage_role_reservation.v1'
            or reservation['final_diagnostic']['seeds']!=[SEED]
            or reservation['final_diagnostic']['runs']!=8
            or reservation['final_diagnostic']['independent_blocks']!=1):
        raise ValueError('Disjoint eight-configuration single-block role reservation required')
    if (plan['seed']!=SEED or plan['window_seconds']!=600. or plan['drain_seconds']!=300.
            or plan['warmup_seconds']!=0. or len(plan['trace'])!=1200
            or plan['workload_rps']!=training.POLICY['workload_rps']
            or tuple(c['id'] for c in plan['candidates'])!=IDS
            or plan['runner_source']!=receipt(native_flow.__file__)
            or plan['profile_plan']!=receipt(training.PROFILE/'plan.json')
            or plan['catalog']!=receipt(native_flow.BASE/'catalog.json')
            or plan['latency_reference_ms']!=dict(text=1000,image=250,asr=3000,ocr=10000)
            or plan['energy_reference_watts']!=450.
            or plan['memory_safe_factor']!=1.110854 or plan['memory_budget_mib']!=49140
            or plan['startup_amortization_seconds']!=3600.
            or 'input_generation_contract' in plan
            or plan['queue_sample_offsets_ns']!=[round(i*6.*1e9) for i in range(101)]
            or type(plan['created_ns']) is not int or plan['created_ns']<=reservation['created_ns']
            or any(v!=1/6 for v in plan['weights'].values())):
        raise ValueError('Original fixed final cohort, constants and domains required')
    aliases={a:dict(family=f,supported_tasks=(['text','image','asr','ocr'] if f=='generalist_2'
        else ['ocr'] if f=='ocr' else ['text'])) for a,f in RESIDENTS.items()}
    if plan['aliases']!=aliases:
        raise ValueError('Exact original resident alias/task domain required')
    training._context(plan['profile_plan'],plan['context'])
    for key in ('catalog','runner_source'): checked(plan[key])
    if plan['context']!=receipt(training.PROFILE/'context.json'):
        raise ValueError('Exact fixed immutable base worker context required')
    for candidate in plan['candidates']:
        mode, route = candidate['id'].split('_',1)
        cap = None if mode=='shared' else 2
        if (candidate['resource_configuration']!=native_flow.resource_configuration(cap)
                or candidate['routing_weights']!=training._weights('admission2_'+route)
                or candidate['execution_context_sha256']!=plan['execution_context_sha256']):
            raise ValueError('Exact corresponding per-mode route and native controller required')
        _generation(plan,condition(plan,candidate))
        flow_eligibility._routing(plan,candidate)
        counterpart = next(c for c in plan['candidates']
            if c['id']==('admission2_' if mode=='shared' else 'shared_')+route)
        if counterpart['routes']!=candidate['routes']:
            raise ValueError('Matched modes must have identical per-request aliases')
    return plan


def prepare(output):
    output = Path(output).resolve()
    native_flow.prepare(training.PROFILE,output,duration=600.,drain=300.,warmup=0.,
                        seed=SEED,include_admission=True)
    (output/'plan.json').rename(output/'native-prepared-plan.json')
    plan = _project(read(output/'native-prepared-plan.json'),time.perf_counter_ns())
    save(output/'plan.json',plan)
    audit_plan(output/'plan.json')
    save(output/'preparation.json',dict(schema=SCHEMA+'.preparation',created_ns=time.perf_counter_ns(),
        plan=receipt(output/'plan.json'),source_receipts=training._refs([sources(),plan]),
        GPU_started=False,final_outcomes_read=False,final_gain_eligible=False))
    return plan


def audit_actual(plan_path, raw_path, *, seal_ns):
    """Verify a final cohort without relabelling or invoking a training mapper.

    ``seal_ns`` belongs to a separately verified final execution seal. This
    helper alone does not authenticate that seal or any capacity forecast.
    """
    plan_path,raw_path = Path(plan_path).resolve(),Path(raw_path).resolve()
    plan_ref=receipt(plan_path)
    native_ref=receipt(raw_path)
    invocation_ref=receipt(raw_path.parent/'invocation.json')
    entry_sources=sources()
    plan = audit_plan(plan_path)
    native = read(checked(native_ref))
    if (native.get('native_raw') or native.get('plan')!=plan_ref
            or type(seal_ns) is not int or seal_ns<plan['created_ns']
            or type(native.get('startup_begin_ns')) is not int
            or native['startup_begin_ns']<=seal_ns):
        raise ValueError('Unmodified native final collection must follow prospective final seal')
    cid = native.get('candidate_id')
    candidate = next((c for c in plan['candidates'] if c['id']==cid),None)
    if candidate is None: raise ValueError('Foreign final configuration')
    objective={k:plan[k] for k in ('workload_rps','latency_reference_ms','weights','memory_safe_factor',
        'memory_budget_mib','energy_reference_watts','startup_amortization_seconds')}
    if native.get('effective_objective')!=objective:
        raise ValueError('Actual native final objective differs from exact original six-term objective')
    invocation = read(checked(invocation_ref))
    if invocation!=dict(plan=plan_ref,candidate_id=cid,runner=plan['runner_source']):
        raise ValueError('Exact final native invocation required')
    contract = mechanism_audit.audit(raw_path)
    profile,context = training._context(plan['profile_plan'],plan['context'])
    if ([(r['model'],r['task'],r['pool_id']) for r in native['priming_request_records']]!=training._priming(profile)
            or native['warmup_request_records']):
        raise ValueError('Same native priming and warmup0 required across modes')
    for alias,info in native['ready'].items():
        if (info.get('clock')!='time.perf_counter_ns'
                or info.get('spec_sha256')!=context['worker_bindings'][RESIDENTS[alias]]['spec_sha256']):
            raise ValueError('Final READY clock/spec mismatch')
    samples,gap = training._telemetry(native)
    left,right = native['phase_start_ns'],native['observation_end_ns']
    if right>native['deadline_ns']+round(plan['drain_seconds']*1e9):
        raise ValueError('Final native collection exceeds unchanged bounded drain')
    scope='final_diagnostic:'+native_ref['sha256']
    semantic = [flow_profiles._request(r,native['ready'],context,split='final',
        scope_id=scope,left=left,right=right)
        for r in native['request_records']]
    history = [flow_profiles._request(r,native['ready'],context,
        split='excluded_prephase_history',scope_id=scope,
        left=native['startup_begin_ns'],right=left) for r in native['priming_request_records']]
    if any(r.get('split') not in (None,'final') for r in native['request_records']):
        raise ValueError('Final cohort cannot be a relabelled training/validation collection')
    for row in semantic: row['not_for_training']=True
    for alias in RESIDENTS:
        rows=[r for r in semantic if r['alias']==alias]
        all_rows=sorted((r for r in semantic+history if r['alias']==alias),key=lambda r:r['handler_start_ns'])
        if (rows!=sorted(rows,key=lambda r:r['handler_start_ns'])
                or any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(all_rows,all_rows[1:]))):
            raise ValueError('Final actual per-alias FIFO/nonoverlap required')
    scored_path=raw_path.parent/'scored-raw.json';scored_ref=receipt(scored_path)
    scored=read(checked(scored_ref))
    if (scored.get('native_raw')!=native_ref
            or {k:v for k,v in scored.items() if k not in ('native_raw','quality_evidence')}!=native):
        raise ValueError('Final quality annotation must bind exact immutable native raw')
    quality=verify_quality_annotation(scored,plan,read(checked(plan['catalog'])))
    if scored['quality_evidence']['source_receipts']!=[plan['catalog'],receipt(native_flow.__file__),
            receipt(ROOT/'gsdeploy/tasks.py'),receipt(ROOT/'gsdeploy/quality.py')]:
        raise ValueError('Exact native final scorer/source receipts required')
    score=flow_metrics.score_flow(scored,plan)
    if quality['count']!=1200 or len(semantic)!=1200 or not score['utility_complete']:
        raise ValueError('Full final cohort, quality and all six actual terms required')
    stability=flow_eligibility.native_stability(plan,candidate,native,read(checked(plan['catalog'])))
    refs=training._refs([entry_sources,plan_ref,plan,context,scored['quality_evidence'],native_ref,
        scored_ref,invocation_ref])
    for ref in refs: checked(ref)
    return dict(schema=SCHEMA+'.actual',source=receipt(__file__),plan=plan_ref,
        native_raw=native_ref,scored_raw=scored_ref,candidate_id=cid,
        requests=1200,known=1200,not_for_training=True,split='final',
        semantic_trace_sha256=digest(semantic),quality=quality,six_term_score=score,
        original_stability=stability,original_rho_admission_passed=None,
        source_receipts=refs,mechanism=contract,telemetry_samples=len(samples),max_gap_ns=gap,
        final_gain_eligible=False,capacity_qualification_performed=False,GPU_started=False)


def report_actual(plan_path, raw_paths, output, *, seal_ns):
    """Audit and summarize the full declared domain without granting capacity.

    The execution wrapper must authenticate the prospective final seal first.
    This lower-level CPU report explicitly lacks that separate authorization
    proof; it cannot make a qualified final-gain claim. Missing attempts remain
    visible in the complete 9600-request denominator.
    """
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    plan=audit_plan(plan_path);plan_ref=receipt(plan_path)
    evidence={};invalid=[];seen=set()
    for path in raw_paths:
        initial_ref=receipt(path);native=read(checked(initial_ref));cid=native.get('candidate_id')
        if cid not in IDS or cid in seen:raise ValueError('Only one final attempt per declared configuration')
        seen.add(cid)
        try:
            item=audit_actual(plan_path,path,seal_ns=seal_ns)
            evidence[cid]=item
        except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
            checked(initial_ref)
            invalid.append(dict(candidate_id=cid,native_raw=initial_ref,requested=1200,
                full_cohort_audited_requests=0,recorded_requests=len(native.get('request_records',[])),
                recorded_completed_requests=sum(r.get('status')=='completed' for r in native.get('request_records',[])),
                audit_error=type(exc).__name__+':'+str(exc),retained_without_retry=True,
                six_term_utility=None,original_qualification='UNKNOWN',final_gain_eligible=False))
    rows=[]
    for cid,item in evidence.items():
        mode,route=cid.split('_',1)
        rows.append(dict(mode=mode,route=route,terms=item['six_term_score']['terms'],
            requested=1200,known=item['known'],original_qualification='FAIL'
                if not item['original_stability']['original_stability_passed'] else 'UNKNOWN'))
    result=statistics.block(plan['locked_policies'],rows)
    for item in evidence.values():
        for ref in item['source_receipts']:checked(ref)
    checked(plan_ref)
    output.mkdir(parents=True,exist_ok=False)
    refs=[]
    for cid,item in evidence.items():
        path=output/(cid+'-actual-audit.json');save(path,item);refs.append(receipt(path))
    value=dict(schema=SCHEMA+'.report',created_ns=time.perf_counter_ns(),
        plan=plan_ref,source=receipt(__file__),actual_audits=refs,
        descriptive_comparison=result,requested=9600,known=result['known'],
        invalid_attempts=invalid,
        status=result['status'],not_for_training=True,independent_blocks=1,
        original_capacity_qualification='UNKNOWN_NOT_ESTABLISHED_BY_THIS_REPORT',
        final_seal_authenticated_by_this_helper=False,
        final_gain_eligible=False,statistical_significance_claim=False,
        primary_comparison='Admission2 locked_allow minus strongest observed declared no-spill baseline',
        primary_value=result['modes']['admission2']['locked_allow_minus_strong_no_spill'],
        policy_scope=plan['policy_scope'],strong_baseline_scope=plan['strong_baseline_scope'],
        source_receipts=training._refs([plan,plan_ref,sources(),evidence,invalid,refs]),GPU_started=False)
    save(output/'report.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',required=True)
    args=parser.parse_args()
    plan=prepare(args.prepare)
    print(json.dumps(dict(status='PREPARED',candidates=len(plan['candidates']),GPU_started=False)))


if __name__=='__main__': main()
