"""CPU training-component checks of exact-input host supply on the real DAG axis.

Isolated endpoints fit the host model; SSG endpoints do not. SSG checks are only
host-component checks: other duration components may have trained on SSG light
captures. No result is a fresh prospective validation or final utility claim.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from .cuda_dag_simulator_v6 import DagFullHorizonModel
from .dag_host_release_v5 import DagHostReleaseProvider as PooledHost
from .dag_host_release_v7 import DagHostReleaseProvider,checked,binding_identity,public_binding,SCENARIOS
from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
from .kernel_semantic_query_v5 import ExactSourceSemanticHypothesisResolver
from .kernel_two_capture_provisional_v5 import ExactSourceTwoCaptureProvisionalResolver
from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from .native_collect import receipt,save
from .native_execution_projection_v5 import native_execution_projection


class NumericDurations:
    """One exact template cache, with strict and hypothesis counts kept apart."""
    def __init__(self,plans,semantic_path,copy_path,provisional_path=None):
        self.plans=plans;self.cache={};self.coverage={}
        self.kernel=ExactSourceSemanticHypothesisResolver(semantic_path)
        if provisional_path:
            self.kernel=ExactSourceTwoCaptureProvisionalResolver(provisional_path,
                allow_two_capture_provisional_hypothesis=True,base_resolver=self.kernel)
        self.copy=ExactSourceCopyMemsetHypothesisResolver(copy_path)

    def __call__(self,node,row):
        key=row['alias'],row['task'],row['pool_id'];cachekey=(*key,node['id'])
        if cachekey not in self.cache:
            result=(self.kernel(node['kernel']) if node['kind']=='gpu' else
                    self.copy(node,self.plans[key]['source_sqlite']))
            c=self.coverage.setdefault(key,dict(queried_nodes=0,strict_qualified_nodes=0,
                numeric_hypothesis_nodes=0,unavailable_nodes=0,strict_failure_reasons=Counter(),
                predicted_activity_sum_ns=0.,strict_predicted_activity_sum_ns=0.,
                by_kind={},query_count_scope='unique_used_template_activity_nodes'))
            c['queried_nodes']+=1;c['strict_qualified_nodes']+=result.get('strict_known') is True
            c['strict_failure_reasons'].update(result.get('strict_failure_reasons',[]))
            if result.get('numeric_hypothesis_available') is not True:
                c['unavailable_nodes']+=1;raise UnknownEvidence('TRAINING_DIAGNOSTIC_NUMERIC_DURATION_UNKNOWN:'+node['id'])
            if result.get('hypothesis_only') is not True or result.get('deployment_qualified') is not False:
                raise ValueError('Numerical duration must preserve hypothesis-only scope')
            duration=result['duration_ns'];c['numeric_hypothesis_nodes']+=1
            c['predicted_activity_sum_ns']+=duration
            if result.get('strict_known') is True:c['strict_predicted_activity_sum_ns']+=duration
            kind=c['by_kind'].setdefault(node['kind'],dict(nodes=0,predicted_sum_ns=0.,strict_nodes=0))
            kind['nodes']+=1;kind['predicted_sum_ns']+=duration;kind['strict_nodes']+=result.get('strict_known') is True
            self.cache[cachekey]=duration
        return self.cache[cachekey]


def checked_capture(inventory_ref,host):
    inventory=checked(inventory_ref);raw=checked(inventory['raw']);plan=checked(inventory['plan']);context=checked(inventory['context'])
    if (plan.get('training_only') is not True or raw.get('kind')!='measured' or raw.get('status')!='completed'
            or raw.get('execution_context_sha256')!=inventory['original_execution_context_sha256']):
        raise ValueError('Original complete measured training capture required')
    projected=native_execution_projection(context)['projection_sha256']
    if projected!={b['native_execution_projection_sha256'] for b in host['public_bindings']}.pop():
        raise ValueError('Training component check requires identical native source projection')
    if any(r.get('split')!='training' or r.get('status')!='completed' for r in raw['request_records']):
        raise ValueError('All diagnostic rows must be completed training')
    return inventory,raw,plan,context


def jobs_for_capture(raw,plan):
    jobs={job['id']:job for job in plan['jobs']};grouped=defaultdict(list)
    for row in raw['request_records']:
        if row['profile_job_id'] not in jobs:raise ValueError('Foreign actual training job')
        grouped[row['profile_job_id']].append(row)
    for job_id,rows in grouped.items():
        job=jobs[job_id]
        if {r['model'] for r in rows}!=set(job['inputs']) or len(rows)!=len(job['inputs']):
            raise ValueError('Full original job request cohort required')
        yield job,rows


def simulate_job(plans,aliases,rows,job,release,durations,*,max_events=500000):
    public=[dict(request_id=r['request_id'],alias=r['model'],task=r['task'],pool_id=r['pool_id'],
                 arrival_ns=round(job['start_offsets_ms'][r['model']]*1e6)) for r in rows]
    diagnostic=DagQueueDiagnostics(aliases,plans,durations,now_ns=0)
    forecast=DagFullHorizonModel(aliases,plans,duration_provider=diagnostic.duration,
        release_provider=release,queue_diagnostics=diagnostic,max_events=max_events,
        allow_independent_copy_hypothesis=True,allow_context_time_slice_hypothesis=True,
        allow_no_implicit_default_stream_hypothesis=True).forecast(public,0,30_000_000_000,
            drain_deadline_ns=120_000_000_000)
    if set(forecast['finishes'])!={r['request_id'] for r in rows}:raise UnknownEvidence('INCOMPLETE_TRAINING_DAG_JOB')
    output=[]
    for row in rows:
        rid=row['request_id'];worker=row['worker_response'];actual=(worker['end_perf_ns']-worker['start_perf_ns'])/1e6
        p=forecast['finishes'][rid];predicted=(p['handler_end_ns']-p['start_ns'])/1e6
        output.append(dict(request_id=rid,alias=row['model'],task=row['task'],pool_id=row['pool_id'],
            public_start_offset_ms=job['start_offsets_ms'][row['model']],
            predicted_handler_ms=predicted,observed_training_handler_ms=actual,
            predicted_minus_observed_ms=predicted-actual,absolute_relative_error=abs(predicted-actual)/actual,
            predicted_queue_ms=p['queue_wait_ns']/1e6,
            observed_stages=[dict(id=s.get('id'),duration_ms=(s['end_ns']-s['start_ns'])/1e6)
                             for s in worker.get('observed_stages',[]) if type(s.get('start_ns')) is int and type(s.get('end_ns')) is int],
            observed_peer_handler_overlap_ms={other['model']:max(0,min(worker['end_perf_ns'],other['worker_response']['end_perf_ns'])-
                max(worker['start_perf_ns'],other['worker_response']['start_perf_ns']))/1e6 for other in rows if other is not row}))
    return dict(status='MODEL_EXECUTED',events=forecast['events'],rows=output,queue_diagnostics=forecast['queue_diagnostics'])


def diagnose(host_path,parameters_path,semantic_path,copy_path,output,provisional_path=None):
    from .forecast_gate_v3 import prediction_source_paths
    source_before=[receipt(p) for p in prediction_source_paths()]
    evidence_paths=[host_path,parameters_path,semantic_path,copy_path]+([provisional_path] if provisional_path else [])
    evidence_before=[receipt(p) for p in evidence_paths]
    host=checked(receipt(host_path));parameters=checked(receipt(parameters_path));dag=checked(host['dag'])
    old=checked(host['source_host']);plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    aliases={p['alias']:dict(family=p['family'],supported_tasks=sorted({q['task'] for q in plans.values() if q['alias']==p['alias']})) for p in plans.values()}
    bindings={(b['alias'],b['task'],b['pool_id']):b for b in host['public_bindings']}
    durations=NumericDurations(plans,semantic_path,copy_path,provisional_path)
    # Declare all capture/scenario comparisons before opening endpoint labels.
    capture_refs=[r for r in parameters['capture_inventories'] if Path(r['path']).name.startswith(('natural_isolate_','natural_SSG_'))]
    if len(capture_refs)!=4:raise ValueError('Two original isolated plus two SSG inventories required')
    fit_sources={c['raw']['sha256'] for c in host['captures']}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks=[]
    declared=[dict(capture=r,label=Path(r['path']).stem,modes=['family_task_pooled_v5','exact_input_v7'],scenarios=list(SCENARIOS)) for r in capture_refs]
    save(out/'declared-checks.json',declared)
    providers={}
    for scenario in SCENARIOS:
        providers['family_task_pooled_v5',scenario]=PooledHost(old,host['execution_context_sha256'],plans,aliases,
            scenario=scenario,allow_identification_hypothesis=True,allow_cross_input_pooling_hypothesis=True)
        providers['exact_input_v7',scenario]=DagHostReleaseProvider(host,host['execution_context_sha256'],plans,aliases,
            scenario=scenario,allow_identification_hypothesis=True,allow_exact_input_conditioning_hypothesis=True,public_bindings=bindings)
    for ref in capture_refs:
        inv,raw,plan,context=checked_capture(ref,host)
        is_isolate=inv['label'].startswith('natural_isolate_')
        if (inv['raw']['sha256'] in fit_sources)!=is_isolate:
            raise ValueError('Isolated fit versus SSG component-check split changed')
        jobs=list(jobs_for_capture(raw,plan))
        for job,rows in jobs:
            for row in rows:
                key=row['model'],row['task'],row['pool_id'];profile=plans[key]
                binding,_=public_binding(profile,row,host['dag'],context,host['light_execution_context_sha256'],host['execution_context_sha256'])
                if binding_identity(binding)!=bindings[key]:raise ValueError('Training component check actual work/content differs from isolated binding')
            for mode in ('family_task_pooled_v5','exact_input_v7'):
                for scenario in SCENARIOS:
                    release=providers[mode,scenario]
                    try:result=simulate_job(plans,aliases,rows,job,release,durations)
                    except UnknownEvidence as exc:result=dict(status='UNKNOWN',reason=str(exc),full_request_ids=[r['request_id'] for r in rows])
                    result.update(capture_label=inv['label'],capture_raw=inv['raw'],original_natural_context_sha256=raw['execution_context_sha256'],
                        host_original_context_sha256=host['execution_context_sha256'],native_projection_transfer_hypothesis=True,
                        job_id=job['id'],mode=mode,scenario=scenario,isolated_fit_endpoint=is_isolate,
                        SSG_endpoint_used_for_host_fit=False,full_pipeline_independent_validation=False)
                    checks.append(result)
        save(out/(inv['label']+'.json'),dict(results=[c for c in checks if c['capture_label']==inv['label']]))
    summaries=[]
    for label in sorted({c['capture_label'] for c in checks}):
        for mode in ('family_task_pooled_v5','exact_input_v7'):
            for scenario in SCENARIOS:
                matching=[c for c in checks if (c['capture_label'],c['mode'],c['scenario'])==(label,mode,scenario)]
                rows=[r for c in matching for r in c.get('rows',[])]
                summaries.append(dict(capture_label=label,mode=mode,scenario=scenario,request_count=len(rows),
                    unknown_jobs=sum(c['status']=='UNKNOWN' for c in matching),
                    service_mare=statistics.mean(r['absolute_relative_error'] for r in rows) if rows else None))
    coverage=[dict(alias=k[0],task=k[1],pool_id=k[2],**v) for k,v in durations.coverage.items()]
    source_after=[receipt(p) for p in prediction_source_paths()]
    evidence_after=[receipt(p) for p in evidence_paths]
    if source_before!=source_after or evidence_before!=evidence_after:
        raise ValueError('Diagnostic producer or source-bound model changed during execution')
    report=dict(schema='gsdeploy.training_component_exact_input_host_diagnostic.v7',
        host=receipt(host_path),parameters=receipt(parameters_path),semantic_fit=receipt(semantic_path),copy_fit=receipt(copy_path),
        two_capture_provisional_fit=receipt(provisional_path) if provisional_path else None,
        declared_checks=receipt(out/'declared-checks.json'),summary=summaries,baseline_coverage=coverage,
        original_kernel_strict_rejections_preserved=True,missing_numeric_service_not_zero=True,
        scenario_selected_from_endpoint_results=False,SSG_endpoints_used_for_fit=False,
        source_endpoint_checks_are_training_identity=True,SSG_is_host_component_check_only=True,
        post_handler_lifecycle_omitted=True,e2e_and_six_term_utility_evaluated=False,
        whole_capture_host_holdout_supported=False,whole_capture_host_holdout_missing='Third independent isolated capture needed to retain two after holding one out',
        unseen_input_supported=False,independent_validation_performed=False,physical_accuracy_established=False,
        physical_gain_certified=False,validation_refit=False,GPU_started=False,
        exact_input_host_parameters_rederived_from_original_raws=True,
        production_source_before=source_before,production_source_after=source_after,
        production_source_bytes_unchanged=True,model_evidence_before=evidence_before,model_evidence_after=evidence_after,
        sources=[receipt(__file__),receipt(Path(__file__).with_name('dag_host_release_v7.py'))])
    save(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('host-model','parameters','semantic-fit','copy-fit','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--provisional-fit');a=p.parse_args()
    r=diagnose(a.host_model,a.parameters,a.semantic_fit,a.copy_fit,a.output,a.provisional_fit)
    print(json.dumps(dict(checks=r['summary'],GPU_started=False)))

if __name__=='__main__':main()
