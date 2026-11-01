"""Reviewed native queue training -> rederived joint support -> busy DAG CPU replay.

This is a training diagnostic. Collector-domain transfer is an explicit source
projection hypothesis; no new timing equivalence or predictor validation is
asserted. Every public request and original arrival/window is retained.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from .dag_checkpoint_v9 import receipt
from .dag_busy_support_v9 import fit_support,verify_support,JointLifecycleFromSupport
from .dag_busy_posterior_v8 import condition_busy_scenarios
from .dag_visible_observer_v8 import observe_cutoff
from .cuda_dag_simulator_v9 import DagFullHorizonModel
from .native_execution_projection_v5 import FIELDS,native_execution_projection
from .multi_native_composite_v5 import MultiNativeProvider,checked
from .component_generation_support_v7 import ComponentGenerationSupport
from .component_host_supply_v8 import ComponentHostSupply
from .dag_host_diagnostic_v7 import NumericDurations
from .background_collect import context_identity


def queue_context_bridge(bundle,target_context_ref):
    """Permit only one documented domain-label difference in native projection."""
    composite=checked(bundle['composite']);source_ref=composite['target']['context']
    source=checked(source_ref);target=checked(target_context_ref)
    missing=[k for k in (*FIELDS,'large_weight_stats','execution_sources') if k not in target]
    if missing:
        raise ValueError('Explicit target queue native context fields required:'+','.join(missing))
    baseline=native_execution_projection(source)
    if (source['measurement_domain']!='natural' or target['measurement_domain']!='natural_predeclared_queue_training_v8'
            or target.get('queue_training') is not True or target.get('training_only') is not True):
        raise ValueError('Original natural and explicitly declared queue-training domains required')
    mismatches=[k for k in FIELDS if k!='measurement_domain' and source.get(k)!=target.get(k)]
    catalogs=[]
    for context in (source,target):
        selected=[s['frozen'] for s in context['execution_sources']
                  if Path(s['original_path']).name=='runtime-inputs.jsonl']
        if len(selected)!=1 or receipt(selected[0]['path'])!=selected[0]:
            raise ValueError('One unchanged source-bound public runtime input catalog required')
        catalogs.append(selected[0])
    if catalogs[0]['sha256']!=catalogs[1]['sha256']:
        mismatches.append('exact_public_runtime_input_catalog_content_sha256')
    if source['large_weight_stats']['sha256']!=target['large_weight_stats']['sha256']:
        mismatches.append('large_weight_stats_content_sha256')
    for ref in target['shared_native_execution_sources']:
        if receipt(ref['path'])!=ref:
            raise ValueError('Queue bridge native execution source changed')
    checked(target['large_weight_stats'])
    if mismatches:
        raise UnknownEvidence('QUEUE_CONTEXT_NATIVE_PROJECTION_MISMATCH:'+','.join(mismatches))
    return dict(schema='gsdeploy.explicit_natural_to_queue_training_projection_hypothesis.v9',
        source_context=source_ref,target_context=copy.deepcopy(target_context_ref),
        source_full_context_sha256=baseline['original_full_context_sha256'],
        target_full_context_sha256=context_identity(target),
        source_native_projection_sha256=baseline['projection_sha256'],
        exact_public_input_catalogs=dict(source=catalogs[0],target=catalogs[1]),
        same_pool_id_different_content_allowed=False,
        identical_native_fields=[k for k in FIELDS if k!='measurement_domain'],
        allowed_difference=dict(field='measurement_domain',source='natural',target=target['measurement_domain']),
        collector_invariance_is_hypothesis=True,full_context_equality_claimed=False,
        natural_timing_transfer_validated=False,physical_accuracy_validated=False,GPU_started=False)


class FrozenDurations:
    def __init__(self,values,evidence):self.values=values;self.evidence=evidence
    def __call__(self,node,row):return self.values[row['alias'],row['task'],row['pool_id'],node['id']]


class FrozenRelease:
    def __init__(self,values,evidence):self.values=values;self.evidence=evidence
    def __call__(self,node,row):return copy.deepcopy(self.values[row['alias'],row['task'],row['pool_id'],node['id']])


def frozen_training_providers(bundle,public_rows):
    """Materialize original trained template queries, never target endpoints."""
    aliases=bundle['forecast_defaults']['aliases']
    for field in ('semantic_fit','copy_memset_fit','two_capture_provisional_fit'):
        checked(bundle[field])
    adapter=MultiNativeProvider(checked(bundle['composite']),aliases,host_scenario=bundle['host_supply_scenario'])
    generation=ComponentGenerationSupport(checked(bundle['generation']['component_support']),adapter,
        allow_exact_content_generation_hypothesis=True)
    support=generation.check(public_rows)
    release=ComponentHostSupply(checked(bundle['host_supply_components']),adapter,aliases,generation.public_bindings,
        scenario=bundle['host_supply_scenario'],expected_composite=bundle['composite'])
    durations=NumericDurations(adapter.plans,bundle['semantic_fit']['path'],bundle['copy_memset_fit']['path'],
        bundle['two_capture_provisional_fit']['path'])
    numeric={};host={};work={}
    evidence=[bundle[k] for k in ('composite','host_supply_components','semantic_fit','copy_memset_fit','two_capture_provisional_fit')]
    selected={}
    for row in public_rows:selected.setdefault((row['alias'],row['task'],row['pool_id']),row)
    for k,row in selected.items():
        if k not in adapter.plans:raise UnknownEvidence('NO_EXACT_PUBLIC_TRAINING_DAG:'+str(k))
        total=0.
        for node in adapter.plans[k]['nodes']:
            host[(*k,node['id'])]=release(node,row)
            if node['kind'] in ('gpu','copy','memset'):
                value=durations(node,row);numeric[(*k,node['id'])]=value;total+=value
        work[json.dumps(list(k),separators=(',',':'))]=dict(solo_activity_work_ms=total/1e6,
            scope='Training template activity demand sum; hypotheses retained; not co-run wall time, E2E capacity or measured throughput',
            evidence=evidence,physical_accuracy_validated=False)
    return adapter,FrozenDurations(numeric,evidence),FrozenRelease(host,evidence),dict(
        generation_support=support,solo_activity_work=work,duration_coverage={str(k):v for k,v in durations.coverage.items()})


def replay(review_paths,bundle_path,output,*,maximum_cutoffs=3,execute_dag=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    review_refs=[receipt(p) for p in review_paths];bundle_ref=receipt(bundle_path);bundle=checked(bundle_ref)
    support=fit_support(review_refs);verify_support(support)
    (output/'support.json').write_text(json.dumps(support,indent=2,sort_keys=True)+'\n')
    runs=[]
    for review_ref in review_refs:
        report=checked(review_ref);raw_ref=report['evidence']['raw'];raw=checked(raw_ref);plan=checked(report['evidence']['plan'])
        bridge=queue_context_bridge(bundle,raw['context'])
        if bridge['target_full_context_sha256']!=support['execution_context_sha256']:
            raise ValueError('Joint model and explicit queue transfer target context differ')
        start=raw['phase_start_ns'];deadline=raw['deadline_ns'];drain=deadline+round(plan['drain_seconds']*1e9)
        rows=[dict(request_id=r['request_id'],alias=r['alias'],task=r['task'],pool_id=r['pool_id'],arrival_ns=start+r['offset_ns']) for r in plan['trace']]
        adapter,duration,release,provider_report=frozen_training_providers(bundle,rows)
        (output/(raw['run_id']+'-provider-diagnostic.json')).write_text(json.dumps(provider_report,indent=2,sort_keys=True)+'\n')
        checks=[]
        for index,offset in enumerate(plan['cutoff_offsets_ns'][:maximum_cutoffs]):
            snapshot=observe_cutoff(raw_ref['path'],rows,plan['aliases'],execution_context_sha256=support['execution_context_sha256'],
                phase_start_ns=start,deadline_ns=deadline,drain_deadline_ns=drain,cutoff_ns=start+offset)
            snapshot_path=output/(raw['run_id']+'-cutoff'+str(index)+'-snapshot.json')
            snapshot_path.write_text(json.dumps(snapshot,indent=2,sort_keys=True)+'\n')
            def factory(*,scenario_id,now_ns,request_start_provider):
                lifecycle=JointLifecycleFromSupport(support,scenario_id,snapshot['full_public_cohort'])
                return DagFullHorizonModel(plan['aliases'],adapter.plans,duration_provider=duration,release_provider=release,
                    lifecycle_provider=lifecycle,request_start_provider=request_start_provider,max_events=5_000_000,
                    allow_independent_copy_hypothesis=True,allow_context_time_slice_hypothesis=True,
                    allow_no_implicit_default_stream_hypothesis=True,
                    checkpoint_binding=dict(execution_context_sha256=support['execution_context_sha256'],
                        evidence=[*support['sources'],bundle_ref,*duration.evidence]))
            result=(condition_busy_scenarios(snapshot,support,factory,capture_latent_checkpoints=False,
                score_kwargs=dict(rates=plan['workload_rps'],refs=plan['latency_reference_ms'],quality_loss=adapter.quality,
                    memory_budget_mib=plan['memory_budget_mib'],energy_reference_watts=plan['energy_reference_watts'],
                    memory_safe_factor=plan['memory_safe_factor'],weights=plan['weights'])) if execute_dag else
                dict(status='CPU_PREPARED_ACTUAL_DAG_NOT_EXECUTED',unknown_reasons=snapshot['unknown_reasons'],candidates=[]))
            result_path=output/(raw['run_id']+'-cutoff'+str(index)+'-result.json')
            result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
            checks.append(dict(cutoff_ns=start+offset,status=result['status'],result=receipt(result_path),snapshot=receipt(snapshot_path),
                full_public_cohort=len(rows),observed_prefix=len(snapshot['completed_prefix']),candidate_count=len(result['candidates']),
                rejected=result.get('rejected',[]),unknown_reasons=result.get('unknown_reasons',[])))
        runs.append(dict(run_id=raw['run_id'],raw=raw_ref,bridge=bridge,cutoffs=checks))
    result=dict(schema='gsdeploy.native_queue_busy_training_DAG_diagnostic.v9',
        status='REVIEWED_TRAINING_DIAGNOSTIC_NOT_INDEPENDENT_PREDICTION_VALIDATION',support=receipt(output/'support.json'),
        bundle=bundle_ref,runs=runs,actual_DAG_execution_requested=execute_dag,
        source_model_context_transfer='Explicit native field equality with collector domain change remains a timing transfer hypothesis',
        independent_validation=False,posterior_calibrated=False,risk_bound_calibrated=False,deployment_utility=None,GPU_started=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',nargs='+',required=True);parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--maximum-cutoffs',type=int,default=3)
    parser.add_argument('--execute-dag',action='store_true')
    args=parser.parse_args();result=replay(args.reviews,args.bundle,args.output,
        maximum_cutoffs=args.maximum_cutoffs,execute_dag=args.execute_dag)
    print(json.dumps(dict(status=result['status'],runs=len(result['runs']),GPU_started=False)))


if __name__=='__main__':main()
