"""Seal actually executed trained-hypothesis forecasts through new providers.

This producer constructs the tools-only supply/composite or CUDA DAG model,
executes its full public cohort, invokes original score_forecast, and derives
sealed timings/terms. It accepts no supplied prediction timing or utility cells.
Unseen inputs or incomplete trained kernel coverage remain explicit UNKNOWN.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.prediction_gate import _verify_protocol,receipt,seal_predictions
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.forecasting.lifecycle import LifecycleProvider
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.routing import Router
from .forecast_gate_v2 import _read,_write,_public_cohort,_to_cell,_startup_value,MODEL_FIELDS_FORBIDDEN
from .composite_execution_v2 import CompositeComponentAdapter,verify_target_runtime
from .launch_supply_residual_v2 import LaunchSupplyResidualProvider
from .trained_supply_provider_v3 import TrainedKernelSupplyProvider

BUNDLE_SCHEMA='gsdeploy.forecast_gate_model_bundle.v3'


def prediction_source_paths():
    root=Path(__file__).resolve().parents[2]
    return [*sorted((root/'gsdeploy').rglob('*.py')),
            *sorted(Path(__file__).resolve().parent.glob('*.py'))]


def evidence_closure(paths):
    """Conservative exact-receipt JSON lineage; no raw timings become inputs."""
    pending=[receipt(p) for p in paths];seen={}
    while pending:
        reference=pending.pop();key=reference['path'],reference['sha256']
        if key in seen:continue
        if receipt(reference['path'])!=reference:
            raise ValueError('Changed model/training/source closure receipt: '+reference['path'])
        seen[key]=reference
        if Path(reference['path']).suffix!='.json':continue
        value=_read(reference['path']);stack=[value]
        while stack:
            node=stack.pop()
            if isinstance(node,dict):
                if set(node)=={'path','sha256'} and isinstance(node['path'],str) and isinstance(node['sha256'],str):
                    pending.append(node)
                else:stack.extend(node.values())
            elif isinstance(node,list):stack.extend(node)
    return list(seen.values())


def _artifact(bundle,name):
    ref=bundle['artifacts'][name]
    if receipt(ref['path'])!=ref:raise ValueError('Changed '+name+' model artifact')
    return _read(ref['path'])


def execute_bundle(bundle,public,*,allow_trained_hypothesis=False,allow_cpu_fixture=False):
    if bundle.get('schema')!=BUNDLE_SCHEMA or bundle.get('kind') not in ('trained_hypothesis','cpu_fixture'):
        raise ValueError('Explicit v3 trained-hypothesis bundle required')
    fixture=bundle['kind']=='cpu_fixture'
    if fixture and not allow_cpu_fixture:raise ValueError('CPU fixture opt-in required')
    if not fixture and not allow_trained_hypothesis:raise ValueError('Trained hypothesis opt-in required')
    defaults=bundle['forecast_defaults']
    if MODEL_FIELDS_FORBIDDEN.intersection(defaults) or defaults.get('execution_contract')!='strict_lifecycle_v2':
        raise ValueError('Only model settings and strict lifecycle are permitted in defaults')
    if 'quality_loss' in defaults:raise ValueError('Quality must come from a bound training artifact')
    contract=_artifact(bundle,'composite');ordered=_artifact(bundle,'ordered');residual=_artifact(bundle,'residual')
    lifecycle_doc=_artifact(bundle,'lifecycle');quality=_artifact(bundle,'quality')
    if bundle['execution_context_sha256']!=contract['target']['execution_context_sha256']:
        raise ValueError('Candidate must retain target collector original execution identity')
    if not fixture:
        verify_target_runtime(contract,contract['target']['context']['path'])
        quality_report=_artifact(bundle,'quality_report')
        if (quality_report.get('status')!='FITTED' or quality_report['quality_loss']!=bundle['artifacts']['quality']
                or quality_report['execution_context_sha256']!=contract['target']['execution_context_sha256']):
            raise ValueError('Quality expectation is not bound to target natural training domain')
    startup_verified=False
    if bundle.get('startup_prediction_provider_kind') is not None:
        if bundle['startup_prediction_provider_kind']!='verified_prephase_transfer_v6':
            raise ValueError('Explicit supported startup prediction verification kind required')
        from .startup_prediction_v6 import prediction as startup_prediction
        for name in ('startup_training','flow_plan'):_artifact(bundle,name)
        startup=startup_prediction(bundle['artifacts']['startup_training']['path'],bundle['artifacts']['flow_plan']['path'],
            bundle['startup_candidate_id'],allow_native_priming_transfer_hypothesis=bundle.get('allow_native_priming_transfer_hypothesis',False))
        if startup!=bundle.get('startup_prediction'):
            raise ValueError('Startup prediction differs from fully reverified training and target plan')
        flow=_artifact(bundle,'flow_plan');candidate=next(c for c in flow['candidates'] if c['id']==bundle['startup_candidate_id'])
        if candidate['routing_weights']!=defaults['continuation_routing_rps'] or defaults['resources'] is not None:
            raise ValueError('Startup-bound shared flow candidate differs from prediction routing/resources')
        startup_verified=True
    options=copy.deepcopy(bundle['hypothesis_options'])
    allowed={'allow_independent_host_supply_hypothesis','device_progress_hypothesis','context_switch_ms'}
    if set(options)-allowed:raise ValueError('Unsupported or hidden scheduling hypothesis options')
    mode=bundle.get('kernel_duration_mode')
    common=dict(allow_cpu_fixture=fixture,allow_analytical_hypothesis=True,**options)
    provider=None;baseline=None
    if mode=='trained_kernel_baseline_v3':
        from .kernel_baseline_query_v3 import ExactSourceBaselineResolver
        for ref in [bundle['artifacts']['baseline_fit'],*bundle['baseline_intakes']]:
            if receipt(ref['path'])!=ref:raise ValueError('Changed fitted baseline or intake artifact')
        baseline=ExactSourceBaselineResolver(bundle['artifacts']['baseline_fit']['path'],
            [r['path'] for r in bundle['baseline_intakes']])
        if defaults['provider_kind']=='launch_supply_composite_v2':
            provider=TrainedKernelSupplyProvider(ordered,residual,ordered['execution_context_sha256'],
                                                 baseline_resolver=baseline,**common)
    elif mode=='cpu_fixture' and fixture:
        if defaults['provider_kind']=='launch_supply_composite_v2':
            provider=LaunchSupplyResidualProvider(ordered,residual,ordered['execution_context_sha256'],**common)
    elif mode=='numeric_semantic_hypothesis_v5':
        if (defaults['provider_kind']!='cuda_dag_host_supply_v4'
                or bundle.get('allow_numeric_semantic_hypothesis') is not True):
            raise ValueError('Numeric semantic hypothesis requires explicit DAG evaluation opt-in')
        from .kernel_semantic_query_v5 import ExactSourceSemanticHypothesisResolver
        reference=bundle['artifacts']['semantic_fit']
        if receipt(reference['path'])!=reference:raise ValueError('Changed numeric semantic hypothesis artifact')
        baseline=ExactSourceSemanticHypothesisResolver(reference['path'])
        if 'two_capture_provisional_fit' in bundle['artifacts']:
            if bundle.get('allow_two_capture_provisional_hypothesis') is not True:
                raise ValueError('Two-capture provisional evaluation requires its own explicit opt-in')
            from .kernel_two_capture_provisional_v5 import ExactSourceTwoCaptureProvisionalResolver
            ref=bundle['artifacts']['two_capture_provisional_fit']
            if receipt(ref['path'])!=ref:raise ValueError('Changed provisional two-capture hypothesis artifact')
            baseline=ExactSourceTwoCaptureProvisionalResolver(ref['path'],
                allow_two_capture_provisional_hypothesis=True,base_resolver=baseline)
    else:raise ValueError('Production v3 uses trained kernel predictions, never measured-target fallback')
    coverage=getattr(provider,'baseline_coverage',None);generation_support=None
    if mode=='numeric_semantic_hypothesis_v5':
        coverage=dict(kind='numeric_semantic_hypothesis_evaluation_only',queried_nodes=0,
            strict_qualified_nodes=0,numeric_hypothesis_nodes=0,unavailable_nodes=0,
            strict_failure_reason_counts={},strict_fit_gate_changed=False,deployment_qualified=False)
    copy_resolver=None;copy_coverage=None
    if 'copy_memset_fit' in bundle['artifacts']:
        if (defaults['provider_kind']!='cuda_dag_host_supply_v4'
                or bundle.get('copy_memset_duration_mode')!='numeric_copy_memset_hypothesis_v5'
                or bundle.get('allow_copy_memset_numeric_hypothesis') is not True):
            raise ValueError('Copy/memset numerical hypotheses require explicit DAG evaluation opt-in')
        from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
        ref=bundle['artifacts']['copy_memset_fit']
        if receipt(ref['path'])!=ref:raise ValueError('Changed copy/memset hypothesis artifact')
        copy_resolver=ExactSourceCopyMemsetHypothesisResolver(ref['path'])
        copy_coverage=dict(kind='numeric_copy_memset_hypothesis_evaluation_only',queried_nodes=0,
            strict_qualified_nodes=0,numeric_hypothesis_nodes=0,unavailable_nodes=0,
            strict_failure_reason_counts={},strict_fit_gate_changed=False,deployment_qualified=False)
    if provider is not None:
        provider=CompositeComponentAdapter(provider,contract,'service',bundle['artifacts']['residual']['path'])
    lifecycle_kind=bundle.get('lifecycle_provider_kind','exact_training_input_v2')
    if lifecycle_kind=='exact_training_input_v2':
        lifecycle=LifecycleProvider(lifecycle_doc,lifecycle_doc['execution_context_sha256'],allow_cpu_fixture=fixture)
    elif lifecycle_kind=='pooled_joint_mean_hypothesis_v4':
        from .pooled_lifecycle_v4 import PooledLifecycleProvider
        if 'lifecycle_model_adaptation' not in contract:
            raise ValueError('Pooled lifecycle requires its own derived composite identity')
        lifecycle=PooledLifecycleProvider(lifecycle_doc,lifecycle_doc['execution_context_sha256'],
            allow_cpu_fixture=fixture,allow_pooling_hypothesis=bundle.get('allow_pooling_hypothesis',False))
    else:raise ValueError('Explicit supported lifecycle provider kind required')
    lifecycle=CompositeComponentAdapter(lifecycle,contract,'lifecycle',bundle['artifacts']['lifecycle']['path'])
    resource_provider=None
    if 'natural_resources' in bundle['artifacts']:
        from .natural_resources_v5 import NaturalResourceProvider
        resource_doc=_artifact(bundle,'natural_resources')
        if resource_doc.get('schema')=='gsdeploy.natural_direct_state_resource_hypothesis.v6':
            from .natural_resources_v6 import NaturalResourceProvider
        target_ref=contract['target']['context']
        if receipt(target_ref['path'])!=target_ref:raise ValueError('Changed natural resource target context')
        resource_provider=NaturalResourceProvider(resource_doc,_read(target_ref['path']),
            allow_conditional_resource_hypothesis=bundle.get('allow_conditional_resource_hypothesis',False),
            resource_scenario=bundle.get('resource_scenario','mean'))
        if resource_provider.execution_context_sha256!=bundle['execution_context_sha256']:
            raise ValueError('Natural resources must retain exact original target identity')
    aliases=defaults['aliases'];start=public['phase_start_ns'];now=public['now_ns'];deadline=public['deadline_ns']
    workload=FutureWorkload(**public['future_workload']);suffix=workload.suffix(start,now,deadline)
    router=Router(defaults['continuation_routing_rps'])
    if defaults.get('route_initial_with_continuation_policy',False):
        # Continue the same deterministic policy after its phase-start choices;
        # otherwise the first text request would be routed twice as choice one.
        initial=sorted((row for queue in public['queues'].values() for row in queue),
                       key=lambda row:(row['arrival_ns'],row['task'],row['request_id']))
        for row in initial:
            if router.choose(row['task'])!=row['alias']:
                raise ValueError('Initial route differs from declared whole-cohort routing policy')
    if defaults['provider_kind']=='launch_supply_composite_v2':
        forecast=FullHorizonModel(aliases,provider,power_provider=resource_provider,memory_provider=resource_provider,
            resources=defaults['resources'],lifecycle_provider=lifecycle,
            max_events=defaults.get('max_events',500000)).forecast(public['queues'],now,deadline,
            defaults.get('service_scenario','profiled'),future_arrivals=suffix['arrivals'],
            route_future=lambda row,queues,clock:router.choose(row['task']),
            drain_deadline_ns=public['drain_deadline_ns'],require_energy=defaults.get('require_energy',False),
            require_memory=defaults.get('require_memory',False),trace_request_id='omit_large_transition_payload')
    elif defaults['provider_kind'] in ('cuda_dag_v2','cuda_dag_host_supply_v4'):
        if defaults['provider_kind']=='cuda_dag_host_supply_v4':
            from .cuda_dag_simulator_v6 import DagFullHorizonModel
        else:
            from .cuda_dag_simulator_v2 import DagFullHorizonModel
        dag=_artifact(bundle,'dag')
        if (dag.get('schema')!='gsdeploy.request_cuda_dag_profiles.v2' or dag.get('training_only') is not True
                or dag['execution_context_sha256']!=ordered['execution_context_sha256']):
            raise ValueError('DAG source must retain the bound light training domain')
        if defaults['resources'] is not None and defaults['resources'].get('mode')=='request_admission':
            raise UnknownEvidence('DAG_REQUEST_ADMISSION_EXECUTION_NOT_IMPLEMENTED')
        dag_options=bundle.get('dag_hypothesis_options',{})
        if set(dag_options)-{'allow_independent_copy_hypothesis','allow_context_time_slice_hypothesis',
                             'allow_no_implicit_default_stream_hypothesis'}:
            raise ValueError('Explicit supported DAG hypotheses required')
        def duration(node,row):
            if fixture and mode!='numeric_semantic_hypothesis_v5':return node['duration_ns']
            if node['kind']!='gpu':
                if copy_resolver is None:raise UnknownEvidence('TRAINED_COPY_MEMSET_DURATION_MODEL_REQUIRED')
                source=plans[row['alias'],row['task'],row['pool_id']]['source_sqlite']
                result=copy_resolver(node,source);copy_coverage['queried_nodes']+=1
                if result.get('hypothesis_only') is not True or result.get('deployment_qualified') is not False:
                    raise ValueError('Copy/memset evaluation cannot claim physical qualification')
                copy_coverage['strict_qualified_nodes']+=result.get('strict_known') is True
                for reason in result.get('strict_failure_reasons',[]):
                    reason=str(reason);counts=copy_coverage['strict_failure_reason_counts'];counts[reason]=counts.get(reason,0)+1
                if result.get('numeric_hypothesis_available') is not True:
                    copy_coverage['unavailable_nodes']+=1
                    raise UnknownEvidence('NUMERIC_COPY_MEMSET_DAG_HYPOTHESIS_UNAVAILABLE:'+node['id'])
                copy_coverage['numeric_hypothesis_nodes']+=1
                return result['duration_ns']
            result=baseline(node['kernel'])
            if mode=='numeric_semantic_hypothesis_v5':
                coverage['queried_nodes']+=1
                if result.get('hypothesis_only') is not True or result.get('deployment_qualified') is not False:
                    raise ValueError('Numeric evaluation cannot claim physical qualification')
                coverage['strict_qualified_nodes']+=result.get('strict_known') is True
                for reason in result.get('strict_failure_reasons',[]):
                    reason=str(reason);counts=coverage['strict_failure_reason_counts'];counts[reason]=counts.get(reason,0)+1
                if result.get('numeric_hypothesis_available') is not True:
                    coverage['unavailable_nodes']+=1
                    raise UnknownEvidence('NUMERIC_SEMANTIC_DAG_HYPOTHESIS_UNAVAILABLE:'+node['id'])
                coverage['numeric_hypothesis_nodes']+=1
                return result['duration_ns']
            if result.get('known') is not True:raise UnknownEvidence('TRAINED_DAG_KERNEL_BASELINE_UNKNOWN:'+node['id'])
            return result['duration_ns']
        def release(node,row):
            if fixture:return node.get('release_ns',0)
            # Observed API issue offsets and the observed terminal handler anchor
            # belong to replay diagnostics. Fitted GPU service does not make
            # those release times a prospective natural host-supply model.
            raise UnknownEvidence('TRAINED_DAG_HOST_RELEASE_MODEL_REQUIRED:'+node['kind'])
        plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
        if len(plans)!=len(dag['profiles']):raise ValueError('Unique DAG template input/owner identities required')
        if defaults['provider_kind']=='cuda_dag_host_supply_v4':
            from .dag_host_release_v4 import DagHostReleaseProvider
            host=_artifact(bundle,'dag_host_release')
            if host.get('schema')=='gsdeploy.dag_host_supply_training_hypothesis.v5':
                from .dag_host_release_v5 import DagHostReleaseProvider
            if (host.get('light_execution_context_sha256')!=dag['execution_context_sha256']
                    or host.get('dag')!=bundle['artifacts']['dag']):
                raise ValueError('Host supply model must bind the same light-domain DAG source')
            release=DagHostReleaseProvider(host,bundle['execution_context_sha256'],plans,aliases,
                scenario=bundle.get('host_supply_scenario'),
                allow_identification_hypothesis=bundle.get('allow_host_supply_identification_hypothesis',False),
                allow_cross_input_pooling_hypothesis=bundle.get('allow_host_supply_pooling_hypothesis',False))
        rows=[copy.deepcopy(r) for queue in public['queues'].values() for r in queue]
        rows.extend(dict(r,alias=router.choose(r['task'])) for r in suffix['arrivals'])
        if 'generation_work' in bundle['artifacts']:
            from .known_input_generation_v6 import check as check_generation
            generation_support=check_generation(bundle,contract,aliases,rows)
        queue_diagnostics=None
        if defaults['provider_kind']=='cuda_dag_host_supply_v4':
            from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
            for record in (coverage,copy_coverage):
                if record is not None:record['query_count_scope']='unique_used_template_nodes_cached_for_all_requests'
            queue_diagnostics=DagQueueDiagnostics(aliases,plans,duration,now_ns=now)
            duration=queue_diagnostics.duration
        forecast=DagFullHorizonModel(aliases,plans,lifecycle_provider=lifecycle,
            scenario=defaults.get('service_scenario','profiled'),duration_provider=duration,
            release_provider=release,
            **(dict(power_provider=resource_provider,memory_provider=resource_provider,
                queue_diagnostics=queue_diagnostics,
                require_energy=defaults.get('require_energy',False),require_memory=defaults.get('require_memory',False))
                if defaults['provider_kind']=='cuda_dag_host_supply_v4' else {}),
            max_events=defaults.get('max_events',500000),**dag_options).forecast(
                rows,now,deadline,drain_deadline_ns=public['drain_deadline_ns'])
    else:raise ValueError('Unsupported v3 provider_kind; DAG uses an explicit separate implementation')
    metric=score_forecast(forecast,[],aliases,workload.rates,defaults['latency_reference_ms'],quality,
        phase_start_ns=start,now_ns=now,deadline_ns=deadline,memory_budget_mib=defaults['memory_budget_mib'],
        energy_reference_watts=defaults['energy_reference_watts'],memory_safe_factor=defaults['memory_safe_factor'],
        weights=bundle['weights'],observed_peak_memory_mib=_startup_value(bundle,'peak_memory_mib'),
        observed_board_energy_joules=0.)
    return dict(schema='gsdeploy.trained_hypothesis_full_horizon_result.v3',forecast=forecast,metrics=metric,
        baseline_coverage=coverage,copy_memset_coverage=copy_coverage,
        target_execution_context_sha256=bundle['execution_context_sha256'],
        model_composite_execution_sha256=contract['composite_execution_sha256'],
        provider_kind=defaults['provider_kind'],kernel_duration_mode=mode,
        host_supply_scenario=bundle.get('host_supply_scenario'),generation_work_support=generation_support,
        startup_prediction_verified=startup_verified,
        measured_target_fallback_used=False,physical_accuracy_validated=False,physical_gain_certified=False,GPU_started=False)


def run_and_seal(protocol_path,jobs,output_dir,*,allow_trained_hypothesis=False,allow_cpu_fixture=False):
    protocol=_read(protocol_path);_verify_protocol(protocol)
    if any(receipt(p) not in protocol['sources'] for p in prediction_source_paths()):
        raise ValueError('Freeze complete v3 package/tools producer source closure before prediction')
    candidates={c['id']:c for c in protocol['candidates']};expected={(r,c) for r in protocol['validation_domain'] for c in candidates}
    keyed={}
    for job in jobs:
        if set(job)!={'run_id','candidate_id','public_inputs_path'}:raise ValueError('Explicit public forecast job required')
        key=job['run_id'],job['candidate_id']
        if key not in expected or key in keyed or receipt(job['public_inputs_path']) not in protocol['sources']:
            raise ValueError('Duplicate/foreign/unfrozen forecast job')
        keyed[key]=job
    if set(keyed)!=expected:raise ValueError('Complete predeclared Cartesian forecast coverage required')
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    cells=[];audits=[];objectives={};shared_cohorts={}
    for index,(key,job) in enumerate(sorted(keyed.items())):
        run,cid=key;candidate=candidates[cid];model_ref=protocol['models'][candidate['model_id']];bundle=_read(model_ref['path'])
        if bundle.get('execution_context_sha256')!=candidate['execution_context_sha256']:
            raise ValueError('Bundle/candidate original target execution mismatch')
        resolution=None;historical_ref=bundle.get('artifacts',{}).get('historical_resolution')
        if historical_ref is not None:
            if receipt(historical_ref['path'])!=historical_ref:raise ValueError('Changed historical resolution manifest')
            from .historical_evidence_v4 import resolve_closure
            resolution=resolve_closure([model_ref['path']],historical_ref['path'])
            closure=resolution['materialized_sources']
        else:closure=evidence_closure([model_ref['path']])
        bound=[*protocol['sources'],*protocol['models'].values(),*protocol['training_data']]
        if any(ref not in bound for ref in closure):raise ValueError('Freeze every model/training/source lineage receipt')
        if not bundle.get('training_indices') or any(ref not in protocol['training_data'] for ref in bundle['training_indices']):
            raise ValueError('Bundle complete training index must be protocol training data')
        if bundle['weights']!=protocol['weights']:raise ValueError('Original six utility weights differ')
        defaults=bundle['forecast_defaults']
        objective={k:defaults[k] for k in ('latency_reference_ms','memory_budget_mib','energy_reference_watts','memory_safe_factor')}
        objective['startup_amortization_seconds']=bundle['startup_amortization_seconds']
        if run in objectives and objectives[run]!=objective:raise ValueError('Paired candidates changed original five normalizers')
        objectives[run]=objective
        public,cohort=_public_cohort(_read(job['public_inputs_path']))
        actual={rid:{k:r[k] for k in ('request_id','task','pool_id')} for rid,r in cohort.items()}
        if actual!={r['request_id']:r for r in protocol['validation_domain'][run]}:
            raise ValueError('Forecast public cohort differs from full validation domain')
        shared={k:public[k] for k in ('phase_start_ns','now_ns','deadline_ns','drain_deadline_ns','future_workload')}
        shared['initial_requests']=sorted(({k:r[k] for k in ('request_id','task','pool_id','arrival_ns')}
            for rows in public['queues'].values() for r in rows),key=lambda r:r['request_id'])
        if run in shared_cohorts and shared_cohorts[run]!=shared:raise ValueError('Paired candidate arrival/workload changed')
        shared_cohorts[run]=shared
        try:
            result=execute_bundle(bundle,public,allow_trained_hypothesis=allow_trained_hypothesis,allow_cpu_fixture=allow_cpu_fixture)
            if set(result['forecast']['cohort'])!=set(cohort):raise ValueError('New producer dropped public cohort requests')
        except UnknownEvidence as exc:
            result=dict(status='UNKNOWN',reason=str(exc),GPU_started=False,full_public_cohort=actual)
        path=output/f'forecast-{index:04d}.json';_write(path,result)
        cells.append(_to_cell(result,cohort,candidate,run,protocol,bundle,public['phase_start_ns']))
        audits.append(dict(run_id=run,candidate_id=cid,model=model_ref,public_inputs=receipt(job['public_inputs_path']),
            result=receipt(path),kind=bundle['kind'],provider_kind=defaults['provider_kind'],model_source_closure=closure,
            historical_evidence_resolution=resolution))
    # Refuse source edits during the CPU producer, before making a seal.
    _verify_protocol(protocol)
    sealed=seal_predictions(protocol,cells,created_unix_ns=time.time_ns());_write(output/'predictions.json',sealed)
    report=dict(schema='gsdeploy.forecast_gate_production_receipt.v3',protocol=receipt(protocol_path),
        predictions=receipt(output/'predictions.json'),producer=receipt(__file__),jobs=audits,
        GPU_started=False,accuracy_established=False,physical_gain_certified=False,
        allow_trained_hypothesis=allow_trained_hypothesis,allow_cpu_fixture=allow_cpu_fixture)
    _write(output/'production-receipt.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--jobs',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--allow-trained-hypothesis',action='store_true');parser.add_argument('--allow-cpu-fixture',action='store_true')
    args=parser.parse_args();result=run_and_seal(args.protocol,_read(args.jobs),args.output,
        allow_trained_hypothesis=args.allow_trained_hypothesis,allow_cpu_fixture=args.allow_cpu_fixture)
    print(json.dumps(result['predictions']))


if __name__=='__main__':main()
