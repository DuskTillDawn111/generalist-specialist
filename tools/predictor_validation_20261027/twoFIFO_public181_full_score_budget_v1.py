"""Real training-only full-sixterm dual-replay and durable-IO CPU budget.

Preauthentication occurs before the measured live interval. Actual native4
still enforces its snapshot+300s deadline; this proxy never certifies accuracy,
never reads future validation labels, and never extends that fixed lead.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
from pathlib import Path
import time

from . import twoFIFO_public181_expected_score_v2 as expected
from . import twoFIFO_matching_prospective_budget_v1 as old
from . import twoFIFO_matching_prospective_bridge_v1 as core
from . import twoFIFO_prospective_contract_v1 as public
from . import twoFIFO_settled_history_initial_v1 as settled
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_public181_full_six_term_replay_budget.v1'
require=core.require
verify=core.verify
receipt=core.receipt
checked=core.checked
STAGES=('prefix_seed_and_calendar_validation','prefix_gate_durable_persistence',
    'producer_full_queue_replay_and_QME_original_score_composition','producer_report_durable_persistence',
    'auditor_prefix_and_report_validation','auditor_full_queue_replay_and_QME_original_score_composition',
    'auditor_exact_report_comparison','sixterm_seal_durable_persistence_and_receipt_validation',
    'acceptance_checkpoint_durable_persistence')
FIXED_MARGIN_SECONDS=30
MULTIPLIER=1.5
required_budget_ns=old.required_budget_ns


def _measure(stages,name,callback):
    begin=time.perf_counter_ns();value=callback();elapsed=time.perf_counter_ns()-begin
    require(elapsed>0,'Positive CPU workload duration required')
    stages.append(dict(stage=name,elapsed_ns=elapsed));return value


def _proxy_inputs(resource_payload,training,contract,contract_ref,candidate):
    donors=[checked(verify(item['plan'])) for item in resource_payload['manifest_document']['attempts']
        if item['candidate_id']==candidate['id']]
    require(len(donors)==2,'Two source-bound resource donors per candidate required')
    plan=copy.deepcopy(checked(verify(donors[0]['reference_validation_plan'])))
    plan.update(shared_dispatch_protocol=expected.shared_dispatch_protocol(),evaluation_protocol=expected.EVALUATION_PROTOCOL)
    plan['forecast_barrier'].update(lead_seconds=300,prior_READY_settle_timeout_seconds=10)
    snapshot=training['snapshot'];origin=snapshot['snapshot_time_ns']+300_000_000_000
    donor=training['donor_plan']
    producers={owner:donor['execution_id']+':'+owner+':stdout:pid'+str(snapshot['ready'][owner]['pid']) for owner in core.fresh.OWNERS}
    seed=settled.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],forecast_origin_ns=origin,owner_producers=producers)
    prefix=dict(seed=seed,actual_priming_semantics_sha256=training['actual_priming_sha256'],
        prefix_snapshot=snapshot,backend='CPU_TRAINING_PREFIX_PROXY_NOT_TARGET')
    gate=dict(forecast_origin_ns=origin,measurement_end_ns=origin+90_000_000_000,
        fixed_drain_deadline_ns=origin+210_000_000_000,
        public_requests=[dict(request_id=r['request_id'],owner=candidate['routes'][r['request_id']],
            task=r['task'],pool_id=r['pool_id'],arrival_ns=origin+r['offset_ns']) for r in contract['trace']])
    require(plan['public_contract']==contract_ref and plan['candidate']==candidate,
        'Budget public candidate and native resource domain must match exactly')
    return plan,prefix,gate


def benchmark(fit_report_path,public_preparation_path,resource_manifest_path,resource_review_paths,output,*,lead_seconds=300):
    require(type(lead_seconds) is int and lead_seconds==300,'Shared protocol fixes exactly300s; insufficient budget means HOLD')
    fit_ref=receipt(fit_report_path);public_ref=receipt(public_preparation_path);resource_ref=receipt(resource_manifest_path)
    review_refs=[receipt(path) for path in resource_review_paths]
    require(len(review_refs)==8 and len({r['path'] for r in review_refs})==8,'All8 actual native resource reviews required for full-score timing')
    preparation=public.audit(public_preparation_path);contract_ref=preparation['artifacts']['public_contract'];contract=checked(verify(contract_ref))
    watched=strict_references([fit_ref,checked(verify(fit_ref)),public_ref,preparation,
        resource_ref,checked(verify(resource_ref)),review_refs,[checked(verify(ref)) for ref in review_refs],implementation_sources(__file__)])
    with core.fresh.SourceChangeGuard(watched) as guard:
        with ExitStack() as stack:
            preauth_begin=time.perf_counter_ns()
            laws=[stack.enter_context(core.open_law_session(fit_report_path)) for _ in range(2)]
            resources=[stack.enter_context(expected.open_resource_session(resource_manifest_path,resource_review_paths)) for _ in range(2)]
            preauth_ns=time.perf_counter_ns()-preauth_begin
            payload=laws[0].checked_payload();require(payload['law'] is not None,'Missing real saved queue law')
            require(core.support(payload['law'],contract)['complete_all_keys_in_every_donor_block'],'Unsupported full181 queue law')
            training=old._training_prefix(payload['fit_report']);snapshot=training['snapshot'];rows=[]
            out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
            for candidate in contract['candidates']:
                resource_payload=resources[0].checked_payload()
                plan,prefix,gate=_proxy_inputs(resource_payload,training,contract,contract_ref,candidate)
                stages=[];case_out=out/candidate['id'];case_out.mkdir();live_begin=time.perf_counter_ns()
                def validate_prefix():
                    again=_proxy_inputs(resource_payload,training,contract,contract_ref,candidate)
                    require(again==(plan,prefix,gate),'Training-only prefix/candidate proxy changed')
                    require(len(gate['public_requests'])==181,'All181 public arrivals retained')
                _measure(stages,STAGES[0],validate_prefix)
                def persist_prefix():
                    prefix_ref=expected.save(case_out/'prefix.json',core.encode(dict(prefix=prefix,TRAINING_ONLY=True)))
                    gate_ref=expected.save(case_out/'gate.json',dict(gate=gate,prefix=prefix_ref,TRAINING_ONLY=True))
                    return [prefix_ref,gate_ref]
                artifacts=_measure(stages,STAGES[1],persist_prefix);runs=[];producer_ref=None
                for index,role in enumerate(('producer','auditor')):
                    law=laws[index];resource=resources[index]
                    def full_pass():
                        lp=law.checked_payload();rp=resource.checked_payload();guard.check()
                        replay=core.replay_law(lp['plans'],lp['law'],lp['unknown_reason'],plan,prefix,gate)
                        require(replay['source_supported_hypothesis_available'] is True,'All-UNKNOWN queue replay cannot establish full-score timing')
                        queue=dict(replay=replay,sources=watched)
                        composed=expected.compose(rp,plan,prefix,gate,queue,configuration_ref=fit_ref,target_manifest_ref=public_ref)
                        require(composed['transfer']['structural_transfer_supported']
                            and composed['score']['all_six_marginal_expectations_complete'],
                            'Full-score benchmark requires complete actual native Q/M/E support, not unknown fast-path')
                        law.checked_payload();resource.checked_payload();guard.check()
                        return core.encode(dict(schema=SCHEMA+'.training_proxy_sixterm_report',
                            queue=queue,composition=composed,hypothesis=expected.HCPU,
                            evaluation_protocol=expected.EVALUATION_PROTOCOL,source_evidence_authenticated=False,
                            target_future_labels_read=False,full_U_accuracy_qualified=False,TRAINING_ONLY=True))
                    if index:
                        def validate_report():
                            validate_prefix()
                            for ref in artifacts:verify(ref)
                            require(checked(verify(producer_ref))==producer_value,'Persisted full six-term producer changed')
                        _measure(stages,STAGES[4],validate_report)
                    value=_measure(stages,STAGES[2] if index==0 else STAGES[5],full_pass)
                    runs.append(dict(role=role,elapsed_ns=stages[-1]['elapsed_ns'],
                        report_sha256=core.fresh.digest(value),generated_worlds=12,resource_plans_retained=8,
                        all_six_expectations_complete=True))
                    if index==0:
                        producer_value=value
                        producer_ref=_measure(stages,STAGES[3],lambda:expected.save(case_out/'producer.json',value))
                        artifacts.append(producer_ref)
                    else:
                        def compare_report():require(value==checked(verify(producer_ref)),'Independent six-term full replay differs')
                        _measure(stages,STAGES[6],compare_report)
                def persist_seal():
                    ref=expected.save(case_out/'seal.json',dict(schema=expected.SCHEMA+'.training_proxy_seal',
                        expected_report=producer_ref,prefix=artifacts[0],gate=artifacts[1],
                        hypothesis=expected.HCPU,evaluation_protocol=expected.EVALUATION_PROTOCOL,
                        conditional_mean_available=True,independent_accuracy_qualified=False,TRAINING_ONLY=True))
                    for artifact in artifacts+[ref]:verify(artifact)
                    return ref
                seal_ref=_measure(stages,STAGES[7],persist_seal);artifacts.append(seal_ref)
                checkpoint_ref=_measure(stages,STAGES[8],lambda:expected.save(case_out/'checkpoint.json',dict(
                    snapshot=snapshot,public_requests=gate['public_requests'],report=producer_ref,seal=seal_ref,
                    sources=watched,TRAINING_ONLY=True)))
                artifacts.append(checkpoint_ref)
                measured=time.perf_counter_ns()-live_begin
                rows.append(dict(candidate_id=candidate['id'],runs=runs,stages=stages,artifacts=artifacts,
                    combined_replay_ns=sum(r['elapsed_ns'] for r in runs),combined_live_workload_ns=measured,
                    context_guard_and_call_overhead_ns=measured-sum(s['elapsed_ns'] for s in stages),
                    public_calendar_sha256=core.fresh.digest(gate['public_requests'])))
            maximum=max(c['combined_live_workload_ns'] for c in rows);required=required_budget_ns(maximum)
            result=dict(schema=SCHEMA,status='REAL_TRAINING_SOURCE_FULL_SIX_TERM_DUAL_REPLAY_CPU_PROXY_BUDGET',
                fit=fit_ref,public_preparation=public_ref,public_contract=contract_ref,
                resource_manifest=resource_ref,resource_reviews=review_refs,
                evaluation_protocol=expected.EVALUATION_PROTOCOL,hypothesis=expected.HCPU,
                cases=rows,candidate_count=4,requests_per_candidate=181,draws_per_pass=12,passes_per_candidate=2,
                resource_attempts_retained=8,lead_seconds=300,fixed_margin_seconds=FIXED_MARGIN_SECONDS,multiplier=MULTIPLIER,
                maximum_measured_live_workload_ns=maximum,required_live_budget_ns=required,
                fits_predeclared_budget=required<300_000_000_000,preauthentication_elapsed_ns=preauth_ns,
                preauthentication_excluded_only_before_worker_startup=True,
                live_workload='VERIFIED_TRAINING_PREFIX_PROXY_FULL_SIXTERM_NUMERICAL_AND_DURABLE_IO',
                actual_target_prefix_overhead_verified=False,full_U_live_pipeline_included=True,
                full_U_accuracy_qualified=False,full_U_qualified=False,clock='time.perf_counter_ns',
                real_saved_matching_law_used=True,model_fit=False,target_future_labels_read=False,
                independent_prediction_accuracy_qualified=False,GPU_started=False,
                sources=strict_references([watched,[c['artifacts'] for c in rows]]))
            guard.check();ref=expected.save(out/'report.json',result);guard.check();return ref


def audit(report_path,fit_ref,*,lead_seconds=300,resource_manifest_ref=None,resource_review_refs=None):
    ref=receipt(report_path);doc=checked(verify(ref));core.response.audit_fit(verify(fit_ref))
    preparation=public.audit(doc['public_preparation']['path']);contract=checked(verify(doc['public_contract']))
    require(doc.get('schema')==SCHEMA and doc.get('status')=='REAL_TRAINING_SOURCE_FULL_SIX_TERM_DUAL_REPLAY_CPU_PROXY_BUDGET'
        and doc['fit']==fit_ref and type(lead_seconds) is int and doc['lead_seconds']==lead_seconds==300
        and doc['public_contract']==preparation['artifacts']['public_contract']
        and doc['evaluation_protocol']==expected.EVALUATION_PROTOCOL and doc['hypothesis']==expected.HCPU
        and doc['candidate_count']==4 and doc['requests_per_candidate']==181 and doc['draws_per_pass']==12
        and doc['passes_per_candidate']==2 and doc['resource_attempts_retained']==8
        and doc['fixed_margin_seconds']==30 and doc['multiplier']==1.5 and doc['clock']=='time.perf_counter_ns'
        and doc['actual_target_prefix_overhead_verified'] is False and doc['full_U_live_pipeline_included'] is True
        and doc['full_U_accuracy_qualified'] is False and doc['full_U_qualified'] is False
        and doc['live_workload']=='VERIFIED_TRAINING_PREFIX_PROXY_FULL_SIXTERM_NUMERICAL_AND_DURABLE_IO'
        and doc['real_saved_matching_law_used'] is True and doc['model_fit'] is False
        and doc['target_future_labels_read'] is False and doc['GPU_started'] is False
        and doc['independent_prediction_accuracy_qualified'] is False
        and type(doc['preauthentication_elapsed_ns']) is int and doc['preauthentication_elapsed_ns']>0
        and doc['preauthentication_excluded_only_before_worker_startup'] is True,
        'Exact real-training full-sixterm dual replay and IO budget required')
    require(resource_manifest_ref is None or doc['resource_manifest']==resource_manifest_ref,'Budget resource manifest differs')
    require(resource_review_refs is None or doc['resource_reviews']==resource_review_refs,'Budget native resource reviews differ')
    with expected.open_resource_session(doc['resource_manifest']['path'],[r['path'] for r in doc['resource_reviews']]) as resource:
        payload=resource.checked_payload()
        require(len(doc['resource_reviews'])==8 and payload['intake']['unknown_resource_attempts']==0,
            'Full-sixterm budget requires every native resource attempt')
    require([r['candidate_id'] for r in doc['cases']]==[c['id'] for c in contract['candidates']],'All4 candidates required')
    for case in doc['cases']:
        require([r['role'] for r in case['runs']]==['producer','auditor']
            and all(type(r['elapsed_ns']) is int and r['elapsed_ns']>0 and r['generated_worlds']==12
                and r['resource_plans_retained']==8 and r['all_six_expectations_complete'] is True for r in case['runs'])
            and case['runs'][0]['report_sha256']==case['runs'][1]['report_sha256']
            and [s['stage'] for s in case['stages']]==list(STAGES)
            and all(type(s['elapsed_ns']) is int and s['elapsed_ns']>0 for s in case['stages'])
            and [r['elapsed_ns'] for r in case['runs']]==[case['stages'][2]['elapsed_ns'],case['stages'][5]['elapsed_ns']]
            and case['combined_replay_ns']==sum(r['elapsed_ns'] for r in case['runs'])
            and type(case['context_guard_and_call_overhead_ns']) is int and case['context_guard_and_call_overhead_ns']>=0
            and case['combined_live_workload_ns']==sum(s['elapsed_ns'] for s in case['stages'])+case['context_guard_and_call_overhead_ns']
            and len(case['artifacts'])==5,'All numerical and durable IO stages required')
        for artifact in case['artifacts']:verify(artifact)
        producer=checked(case['artifacts'][2]);require(core.fresh.digest(producer)==case['runs'][0]['report_sha256'],
            'Timing replay digest must bind the durable full six-term output')
        require(producer['composition']['score']['all_six_marginal_expectations_complete'] is True,
            'UNKNOWN fast-path is not a full-sixterm benchmark')
    maximum=max(c['combined_live_workload_ns'] for c in doc['cases']);required=required_budget_ns(maximum)
    require(doc['maximum_measured_live_workload_ns']==maximum and doc['required_live_budget_ns']==required
        and doc['fits_predeclared_budget'] is (required<300_000_000_000),'Fixed300s full-pipeline margin must rederive')
    watched=strict_references([fit_ref,checked(verify(fit_ref)),doc['public_preparation'],preparation,
        doc['resource_manifest'],checked(verify(doc['resource_manifest'])),doc['resource_reviews'],
        [checked(verify(r)) for r in doc['resource_reviews']],implementation_sources(__file__)])
    require(doc['sources']==strict_references([watched,[c['artifacts'] for c in doc['cases']]]),'Full-score budget source closure changed')
    for source in doc['sources']:verify(source)
    verify(ref);return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit-report','public-preparation','resource-manifest','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--resource-review',action='append',default=[]);args=parser.parse_args()
    print(benchmark(args.fit_report,args.public_preparation,args.resource_manifest,args.resource_review,args.output))
