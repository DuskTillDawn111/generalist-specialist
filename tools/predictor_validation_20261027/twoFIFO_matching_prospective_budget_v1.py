"""CPU real-saved-law timing gate; no target validation labels or GPU launch.

Benchmarks both full replay passes for each original candidate using a verified
training-only prior prefix. This is an execution budget measurement, never a
substitute for the future target prefix or a prediction-accuracy experiment.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time

from . import twoFIFO_matching_prospective_bridge_v1 as core
from . import twoFIFO_prospective_contract_v1 as public
from . import twoFIFO_settled_history_initial_v1 as settled
from .natural_macro_resources_candidate_v1 import priming_identity
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_matching_real_saved_law_public181_replay_budget.v1'
require=core.require
verify=core.verify
receipt=core.receipt
checked=core.checked
FIXED_MARGIN_SECONDS=30
MULTIPLIER=1.5
STAGES=('prefix_seed_and_calendar_validation','prefix_gate_durable_persistence',
    'producer_full_replay','producer_report_durable_persistence','auditor_prefix_and_report_validation',
    'auditor_full_replay','auditor_exact_report_comparison','seal_durable_persistence_and_receipt_validation',
    'acceptance_checkpoint_durable_persistence')


def required_budget_ns(maximum):
    require(type(maximum) is int and maximum>0,'Positive measured full live workload required')
    return (3*maximum+1)//2+FIXED_MARGIN_SECONDS*1_000_000_000


def _measure(stages,name,callback):
    begin=time.perf_counter_ns();value=callback();elapsed=time.perf_counter_ns()-begin
    require(elapsed>0,'Monotonic CPU stage duration must be positive')
    stages.append(dict(stage=name,elapsed_ns=elapsed));return value


def _training_prefix(fit_report):
    def extract(bundle,intake_document):
        require(intake_document==fit_report['intake_evidence'],'Timing seed must use the saved exact15 intake')
        raw=core._raws(bundle)[0];plan=bundle['plans'][0]
        case=plan['cases'][0];history=[r for r in raw['case_histories'] if r['case_id']==case['case_id']]
        require(len(history)==1,'One actual prior prefix from first matching case required')
        snapshot=history[0]['prior_history_snapshot']
        require([r['request_id'] for r in snapshot['request_records']]==[r['request_id'] for r in plan['priming']],
            'Budget seed may contain only the actual18 training priming records')
        actual=priming_identity(raw,checked(plan['context']))
        return dict(snapshot=copy.deepcopy(snapshot),donor_plan=copy.deepcopy(plan),actual_priming_sha256=actual)
    return core.response.with_verified_bundle(fit_report['manifest']['path'],fit_report['continuation']['path'],extract)


def benchmark(fit_report_path,public_preparation_path,output,*,lead_seconds=300):
    require(type(lead_seconds) is int and 1<=lead_seconds<=1200,'Predeclared live budget in1..1200 seconds required')
    fit_ref=receipt(fit_report_path);public_ref=receipt(public_preparation_path)
    preparation=public.audit(public_preparation_path);contract_ref=preparation['artifacts']['public_contract'];contract=checked(contract_ref)
    watched=strict_references([fit_ref,checked(verify(fit_ref)),public_ref,preparation,implementation_sources(__file__)])
    with core.fresh.SourceChangeGuard(watched) as guard:
        guard.check()
        with core.open_law_session(fit_report_path) as session:
            payload=session.checked_payload();require(payload['law'] is not None,'No eligible saved law: native timing budget remains HOLD')
            require(core.support(payload['law'],contract)['complete_all_keys_in_every_donor_block'],
                'Missing donor block/template support: native timing budget remains HOLD')
            prefix=_training_prefix(payload['fit_report']);donor=prefix['donor_plan'];snapshot=prefix['snapshot']
            out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
            origin=snapshot['snapshot_time_ns']+lead_seconds*1_000_000_000
            producers={owner:donor['execution_id']+':'+owner+':stdout:pid'+str(snapshot['ready'][owner]['pid']) for owner in core.fresh.OWNERS}
            seed=settled.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],forecast_origin_ns=origin,owner_producers=producers)
            rows=[]
            for candidate in contract['candidates']:
                plan=copy.deepcopy(donor);plan.update(public_contract=contract_ref,
                    runtime_config=core.fresh.runtime_configuration(donor,candidate,contract).as_dict())
                gate=dict(forecast_origin_ns=origin,measurement_end_ns=origin+90_000_000_000,
                    fixed_drain_deadline_ns=origin+210_000_000_000,
                    public_requests=[dict(request_id=r['request_id'],owner=candidate['routes'][r['request_id']],
                        task=r['task'],pool_id=r['pool_id'],arrival_ns=origin+r['offset_ns']) for r in contract['trace']])
                input_prefix=dict(seed=seed,actual_priming_semantics_sha256=prefix['actual_priming_sha256'])
                stages=[];case_out=out/candidate['id'];live_begin=time.perf_counter_ns()
                def validate_prefix():
                    rederived=settled.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],
                        forecast_origin_ns=origin,owner_producers=producers)
                    require(rederived==seed and len(gate['public_requests'])==181,'Training prefix/calendar changed')
                    return core.fresh.digest(dict(snapshot=snapshot,gate=gate))
                _measure(stages,STAGES[0],validate_prefix)
                def persist_prefix():
                    prior=core.prefix.save(case_out/'training-prefix.json',dict(snapshot=snapshot,
                        seed=core.encode(seed),actual_priming_sha256=prefix['actual_priming_sha256'],TRAINING_ONLY=True))
                    request=core.prefix.save(case_out/'calendar.json',gate)
                    return [prior,request]
                artifacts=_measure(stages,STAGES[1],persist_prefix);runs=[];producer_value=None
                for role in ('producer','auditor'):
                    session.checked_payload();guard.check()
                    if role=='auditor':
                        def validate_report():
                            validate_prefix()
                            for ref in artifacts:verify(ref)
                            require(checked(producer_ref)['replay']==producer_value,'Persisted producer result changed')
                        _measure(stages,STAGES[4],validate_report)
                    value=_measure(stages,STAGES[2] if role=='producer' else STAGES[5],lambda:
                        core.replay_law(payload['plans'],payload['law'],payload['unknown_reason'],plan,input_prefix,gate))
                    elapsed=stages[-1]['elapsed_ns'];session.checked_payload();guard.check()
                    require(value['source_supported_hypothesis_available'] is True,
                        'All-UNKNOWN replay cannot establish a native forecast timing budget')
                    runs.append(dict(role=role,elapsed_ns=elapsed,forecast_sha256=value['forecast_sha256'],
                        generated_worlds=value['generated_worlds'],world_unknown_probability_mass=value['world_unknown_probability_mass']))
                    if role=='producer':
                        producer_value=value
                        producer_ref=_measure(stages,STAGES[3],lambda:core.prefix.save(case_out/'producer.json',
                            dict(replay=value,sources=watched,seed_scope='VERIFIED_TRAINING_PREFIX_CPU_BUDGET_ONLY')))
                        artifacts.append(producer_ref)
                    else:
                        def compare_report():require(value==checked(verify(producer_ref))['replay'],
                            'Independent complete saved-law budget replay differs')
                        _measure(stages,STAGES[6],compare_report)
                require(runs[0]['forecast_sha256']==runs[1]['forecast_sha256'],'Independent full budget replays differ')
                def persist_seal():
                    ref=core.prefix.save(case_out/'seal.json',dict(producer=producer_ref,forecast_sha256=value['forecast_sha256'],
                        prefix=artifacts[0],calendar=artifacts[1],TRAINING_PREFIX_CPU_BUDGET_ONLY=True))
                    for item in artifacts+[ref]:verify(item)
                    return ref
                seal_ref=_measure(stages,STAGES[7],persist_seal);artifacts.append(seal_ref)
                checkpoint_ref=_measure(stages,STAGES[8],lambda:core.prefix.save(case_out/'checkpoint.json',
                    dict(prefix=snapshot,public_requests=gate['public_requests'],forecast=producer_ref,seal=seal_ref,
                        sources=watched,TRAINING_PREFIX_CPU_BUDGET_ONLY=True)))
                artifacts.append(checkpoint_ref)
                measured_live_ns=time.perf_counter_ns()-live_begin
                rows.append(dict(candidate_id=candidate['id'],runs=runs,
                    combined_replay_ns=sum(r['elapsed_ns'] for r in runs),stages=stages,artifacts=artifacts,
                    combined_live_workload_ns=measured_live_ns,
                    context_guard_and_call_overhead_ns=measured_live_ns-sum(s['elapsed_ns'] for s in stages),
                    public_calendar_sha256=core.fresh.digest(gate['public_requests'])))
            maximum=max(r['combined_live_workload_ns'] for r in rows)
            required_ns=required_budget_ns(maximum)
            result=dict(schema=SCHEMA,status='REAL_SAVED_LAW_TWO_PASS_BUDGET_MEASURED_NOT_TARGET_FORECAST',
                fit=fit_ref,public_preparation=public_ref,public_contract=contract_ref,
                seed_scope='VERIFIED_MATCHING_TRAINING_FIRST_CASE18_PRIOR_ONLY_NOT_FUTURE_TARGET_PREFIX',
                training_snapshot_sha256=core.fresh.digest(snapshot),
                training_plan=dict(attempt_id=donor['attempt_id'],execution_id=donor['execution_id'],context=donor['context']),
                cases=rows,candidate_count=4,requests_per_candidate=181,draws_per_pass=12,passes_per_candidate=2,
                lead_seconds=lead_seconds,fixed_margin_seconds=FIXED_MARGIN_SECONDS,multiplier=MULTIPLIER,
                maximum_two_pass_replay_ns=max(r['combined_replay_ns'] for r in rows),
                maximum_measured_live_workload_ns=maximum,required_live_budget_ns=required_ns,
                live_workload='VERIFIED_TRAINING_PREFIX_PROXY_FULL_NUMERICAL_AND_DURABLE_IO',
                actual_target_prefix_overhead_verified=False,
                forecast_scope='QUEUE_HYPOTHESIS_ONLY_NOT_AUTHENTICATED_SIX_TERM_SCORE',
                full_U_live_pipeline_included=False,full_U_qualified=False,
                fits_predeclared_budget=required_ns<lead_seconds*1_000_000_000,
                preauthentication_timings=session.timings,preauthentication_excluded_only_before_worker_startup=True,
                clock='time.perf_counter_ns',real_saved_matching_law_used=True,model_fit=False,
                target_future_labels_read=False,independent_prediction_accuracy_qualified=False,GPU_started=False,
                sources=strict_references([watched,[c['artifacts'] for c in rows]]))
            session.checked_payload();guard.check()
        ref=core.prefix.save(out/'report.json',result);guard.check();return ref


def audit(report_path,fit_ref,*,lead_seconds):
    ref=receipt(report_path);doc=checked(verify(ref))
    require(doc.get('schema')==SCHEMA and doc.get('status')=='REAL_SAVED_LAW_TWO_PASS_BUDGET_MEASURED_NOT_TARGET_FORECAST'
        and doc.get('fit')==fit_ref and doc.get('lead_seconds')==lead_seconds
        and doc.get('candidate_count')==4 and doc.get('requests_per_candidate')==181
        and doc.get('draws_per_pass')==12 and doc.get('passes_per_candidate')==2
        and doc.get('fixed_margin_seconds')==FIXED_MARGIN_SECONDS and doc.get('multiplier')==MULTIPLIER
        and doc.get('clock')=='time.perf_counter_ns' and doc.get('real_saved_matching_law_used') is True
        and doc.get('live_workload')=='VERIFIED_TRAINING_PREFIX_PROXY_FULL_NUMERICAL_AND_DURABLE_IO'
        and doc.get('actual_target_prefix_overhead_verified') is False
        and doc.get('forecast_scope')=='QUEUE_HYPOTHESIS_ONLY_NOT_AUTHENTICATED_SIX_TERM_SCORE'
        and doc.get('full_U_live_pipeline_included') is False and doc.get('full_U_qualified') is False
        and doc.get('model_fit') is False and doc.get('target_future_labels_read') is False and doc.get('GPU_started') is False,
        'Exact real saved-law full181 two-pass timing report required')
    core.response.audit_fit(verify(fit_ref));preparation=public.audit(doc['public_preparation']['path']);contract=checked(doc['public_contract'])
    require(doc['public_contract']==preparation['artifacts']['public_contract']
        and [c['candidate_id'] for c in doc['cases']]==[c['id'] for c in contract['candidates']],
        'Budget must cover all original four candidates')
    for case in doc['cases']:
        require([r['role'] for r in case['runs']]==['producer','auditor']
            and all(type(r['elapsed_ns']) is int and r['elapsed_ns']>0 and r['generated_worlds']>0 for r in case['runs'])
            and case['runs'][0]['forecast_sha256']==case['runs'][1]['forecast_sha256']
            and case['combined_replay_ns']==sum(r['elapsed_ns'] for r in case['runs']),
            'Both actual full forecast replay timings and matching hashes required')
        require([s['stage'] for s in case['stages']]==list(STAGES)
            and all(type(s['elapsed_ns']) is int and s['elapsed_ns']>0 for s in case['stages'])
            and type(case['context_guard_and_call_overhead_ns']) is int and case['context_guard_and_call_overhead_ns']>=0
            and case['combined_live_workload_ns']==sum(s['elapsed_ns'] for s in case['stages'])+case['context_guard_and_call_overhead_ns']
            and [r['elapsed_ns'] for r in case['runs']]==[case['stages'][2]['elapsed_ns'],case['stages'][5]['elapsed_ns']]
            and len(case['artifacts'])==5,'Every prefix/replay/receipt/fsync/checkpoint timing stage required')
        for artifact in case['artifacts']:verify(artifact)
    maximum=max(c['combined_live_workload_ns'] for c in doc['cases'])
    required=required_budget_ns(maximum)
    require(doc['maximum_two_pass_replay_ns']==max(c['combined_replay_ns'] for c in doc['cases'])
        and doc['maximum_measured_live_workload_ns']==maximum and doc['required_live_budget_ns']==required
        and doc['fits_predeclared_budget'] is (required<lead_seconds*1_000_000_000),
        'Timing budget must derive from every original full replay plus fixed margin')
    expected=strict_references([fit_ref,checked(verify(fit_ref)),doc['public_preparation'],preparation,
        implementation_sources(__file__),[c['artifacts'] for c in doc['cases']]])
    require(doc['sources']==expected,'Budget implementation/source closure changed')
    for source in expected:verify(source)
    return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit-report','public-preparation','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--lead-seconds',type=int,default=300);args=parser.parse_args()
    print(benchmark(args.fit_report,args.public_preparation,args.output,lead_seconds=args.lead_seconds))
