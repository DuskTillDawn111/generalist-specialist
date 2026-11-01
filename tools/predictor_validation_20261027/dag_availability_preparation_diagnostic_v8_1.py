"""Execute whole original request DAGs with fitted availability and solo work.

Each predeclared job retains every request. Unsupported policy states are
reported for the whole job, never dropped or replaced by equal context rates.
Natural endpoints are training component checks, not a prospective validation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from .dag_availability_complete_fit_v8 import derive
from .dag_host_release_v7 import checked
from .dag_preparation_host_release_v8 import DagPreparationHostReleaseProvider,SCENARIO
from .dag_phase_scheduler_v8 import DagPhaseFullHorizonModel
from .multi_native_composite_v5 import MultiNativeProvider
from .native_collect import receipt,save
from .solo_reference_work_v8 import SoloOrdinalReferenceWork
from .static_dag_phase_binding_v8 import StaticDagPhaseBinding
from .trained_dag_availability_v8 import TrainedDagAvailabilityScheduler
from .unified_corun_review_v8_1 import load_review,NATURAL


def diagnose(model_path,parameters_path,bundle_path,review_paths,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    model_ref=receipt(model_path);model=checked(model_ref)
    regenerated=derive([r['path'] for r in model['source_DAG_reports']])
    if json.loads(json.dumps(regenerated))!=model:
        raise ValueError('Availability parameters fail exact source rederivation')
    parameters=checked(receipt(parameters_path));host=checked(parameters['host_model']);dag=checked(host['dag'])
    bundle=checked(receipt(bundle_path));plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    aliases={p['alias']:dict(family=p['family'],supported_tasks=sorted({q['task'] for q in plans.values() if q['alias']==p['alias']})) for p in plans.values()}
    bindings={(b['alias'],b['task'],b['pool_id']):b for b in host['public_bindings']}
    scenario=bundle['host_supply_scenario']
    lifecycle=MultiNativeProvider(checked(bundle['composite']),bundle['forecast_defaults']['aliases'],host_scenario=scenario)
    release=DagPreparationHostReleaseProvider(host,host['execution_context_sha256'],plans,aliases,scenario=SCENARIO,
        allow_preparation_alignment_hypothesis=True,public_bindings=bindings)
    copy=ExactSourceCopyMemsetHypothesisResolver(bundle['copy_memset_fit']['path'])
    if copy.fit_source!=bundle['copy_memset_fit']:raise ValueError('Changed exact copy service hypothesis')
    def copy_work(node,row):
        result=copy(node,plans[row['alias'],row['task'],row['pool_id']]['source_sqlite'])
        if result.get('numeric_hypothesis_available') is not True:
            raise UnknownEvidence('EXACT_COPY_WORK_REFERENCE_UNAVAILABLE')
        return result['duration_ns']
    work=SoloOrdinalReferenceWork(host['dag'],parameters['solo_light_reviews'],copy_duration_provider=copy_work)
    phases=StaticDagPhaseBinding(host['dag'],parameters['solo_light_reviews'][:1])
    sources=[model_ref,receipt(parameters_path),receipt(bundle_path),receipt(__file__),receipt(Path(__file__).with_name('dag_phase_scheduler_v8.py'))]
    captures=[];declared=[]
    for path in review_paths:
        report,plan,raw=load_review(path,NATURAL);byjob=defaultdict(list)
        for row in raw['request_records']:byjob[row['profile_job_id']].append(row)
        for job in plan['jobs']:
            rows=byjob[job['id']]
            if set(r['model'] for r in rows)!=set(job['inputs']) or len(rows)!=len(job['inputs']):
                raise ValueError('Complete original request job cohort required')
            declared.append(dict(review=receipt(path),job_id=job['id'],request_ids=[r['request_id'] for r in rows]))
        captures.append((receipt(path),report,plan,raw,byjob));sources.extend([receipt(path),*report['evidence'].values()])
    save(out/'declared-cohorts.json',declared);checks=[]
    for ref,report,plan,raw,byjob in captures:
        for job in plan['jobs']:
            rows=byjob[job['id']];public=[dict(request_id=r['request_id'],alias=r['model'],task=r['task'],pool_id=r['pool_id'],arrival_ns=round(job['start_offsets_ms'][r['model']]*1e6)) for r in rows]
            scheduler=TrainedDagAvailabilityScheduler(model['policy'],allow_unvalidated_training_candidate=True)
            engine=DagPhaseFullHorizonModel(aliases,plans,duration_provider=work,release_provider=release,
                phase_provider=phases,scheduler=scheduler,lifecycle_provider=lifecycle,
                allow_independent_copy_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True,max_events=1_000_000)
            try:
                forecast=engine.forecast(public,0,10_000_000_000,drain_deadline_ns=120_000_000_000)
                if set(forecast['finishes'])!={r['request_id'] for r in rows}:raise UnknownEvidence('INCOMPLETE_FULL_DAG_COHORT')
                result=[]
                for row in rows:
                    p=forecast['finishes'][row['request_id']];w=row['worker_response']
                    handler=(w['end_perf_ns']-w['start_perf_ns'])/1e6
                    observed_receipt=(row['completion_time_ns']-row['actual_submit_time_ns'])/1e6
                    predicted_handler=(p['handler_end_ns']-p['start_ns'])/1e6
                    predicted_receipt=(p['finish_ns']-p['arrival_ns'])/1e6
                    result.append(dict(request_id=row['request_id'],alias=row['model'],task=row['task'],pool_id=row['pool_id'],
                        predicted_handler_ms=predicted_handler,observed_handler_ms=handler,handler_absolute_relative_error=abs(predicted_handler-handler)/handler,
                        predicted_receipt_latency_ms=predicted_receipt,observed_submit_to_receipt_ms=observed_receipt,
                        receipt_absolute_relative_error=abs(predicted_receipt-observed_receipt)/observed_receipt,
                        predicted_request_queue_ms=p['queue_wait_ns']/1e6,observed_worker_queue_ms=row['worker_queue_wait_ms']))
                check=dict(status='FULL_DAG_EXECUTED',rows=result,events=forecast['events'],forecast=forecast)
            except UnknownEvidence as exc:
                check=dict(status='UNKNOWN_COMPLETE_JOB',reason=str(exc),full_request_ids=[r['request_id'] for r in rows],
                    scheduler_transitions=scheduler.transitions)
            check.update(source_review=ref,job_id=job['id'],job_name=job['name'],request_count=len(rows));checks.append(check)
            save(out/(ref['sha256'][:12]+'-'+job['id'][:12]+'.json'),check)
            print(json.dumps(dict(job=job['name'],status=check['status'],reason=check.get('reason'))),flush=True)
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Full-DAG diagnostic source changed')
    rows=[r for c in checks for r in c.get('rows',[])]
    result=dict(schema='gsdeploy.preparation_conditioned_availability_whole_DAG_diagnostic.v8_1',status='TRAINING_COMPONENT_DIAGNOSTIC_COMPLETE',
        sources=sources,declared_cohorts=receipt(out/'declared-cohorts.json'),checks=[{k:v for k,v in c.items() if k!='forecast'} for c in checks],
        complete_jobs=sum(c['status']=='FULL_DAG_EXECUTED' for c in checks),unknown_jobs=sum(c['status']=='UNKNOWN_COMPLETE_JOB' for c in checks),
        full_declared_requests=sum(c['request_count'] for c in checks),numeric_requests=len(rows),
        handler_mare=statistics.mean(r['handler_absolute_relative_error'] for r in rows) if rows else None,
        receipt_mare=statistics.mean(r['receipt_absolute_relative_error'] for r in rows) if rows else None,
        solo_reference_work_provenance=work.provenance(),static_phase_provenance=phases.provenance(),
        elapsed_seconds=time.monotonic()-start,cohort_filtering_used=False,equal_context_fallback_used=False,
        original_host_lifecycle_copy_transfer_is_hypothesis=True,preparation_alignment_is_hypothesis=True,
        independent_prediction_validation=False,full_utility_evaluated=False,physical_accuracy_validated=False,GPU_started=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','parameters','bundle','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--natural-review',action='append',required=True);a=p.parse_args()
    r=diagnose(a.model,a.parameters,a.bundle,a.natural_review,a.output)
    print(json.dumps({k:r[k] for k in ('status','complete_jobs','unknown_jobs','full_declared_requests','numeric_requests','handler_mare','receipt_mare')}))


if __name__=='__main__':main()
