"""Fresh four-process v16 CPU prediction production under the original v5 gate.

This command requires complete numeric preflight, freezes operational sources,
then constructs four independent compiled providers. No preflight cells are
sealed and no GPU collector is started.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from . import learned_campaign_parallel_v16 as parallel
from . import learned_campaign_seal_v15 as previous
from .known_input_gate_v5 import _cells,_verify_protocol,freeze_gate_protocol,seal_predictions
from .known_input_campaign_seal_v5 import check_complete
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .prospective_flow_contract_v11 import bind_public,gate_cell,original_flow_defaults

SCHEMA='gsdeploy.parallel_learned_campaign_prediction_production.v16'


def _numeric_preflight(path,plan_ref,config_ref,plan):
    evidence=previous._numeric_preflight(path,plan_ref,config_ref,plan)
    report=checked(evidence['report'])
    if report.get('CPU_representation_version')!=16:
        raise ValueError('Actual v16 compiled-provider numeric preflight required')
    for reference in evidence['results']:
        result=checked(reference);diagnostic=result.get('static_provider_compilation',{})
        if (result.get('CPU_representation_version')!=16
                or diagnostic.get('nodes_or_edges_dropped') is not False
                or diagnostic.get('fitted_values_changed') is not False):
            raise ValueError('Complete original graph and fitted values in v16 preflight required')
    return evidence


def prepare_and_seal(plan_path,training_index_path,configuration_path,numeric_preflight_report_path,
                     output,*,historical_manifest_path=None,worker_timeout_seconds=7200.):
    plan_ref=receipt(plan_path);config_ref=receipt(configuration_path)
    plan=checked(plan_ref);config=checked(config_ref);previous._preserved_plan(plan)
    if config.get('schema')!=previous.MODEL_SCHEMA:
        raise ValueError('Actual v15 learned configuration with v16 compiled provider required')
    preflight=_numeric_preflight(numeric_preflight_report_path,plan_ref,config_ref,plan)
    training=previous._complete_index(training_index_path)
    bundle=checked(config['base_model_configuration']['component_bundle'])
    defaults=original_flow_defaults(plan,bundle['forecast_defaults']);context=checked(plan['context'])
    catalog=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(catalog)!=1:raise ValueError('One immutable original public input-content catalog required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);jobs=[];spec_refs=[]
    for candidate in plan['candidates']:
        cid=candidate['id'];directory=Path(plan_ref['path']).parent
        public_ref=receipt(directory/('public-'+cid+'.json'));routes_ref=receipt(directory/('routes-'+cid+'.json'))
        bind_public(plan,cid,checked(public_ref)['forecast_inputs'],checked(routes_ref),defaults,plan['weights'])
        jobs.append(dict(run_id=plan['run_id'],candidate_id=cid,public_inputs=public_ref,routes=routes_ref))
        spec=dict(schema=parallel.SPEC_SCHEMA,protocol_path=str(out/'protocol.json'),plan=plan_ref,
            configuration=config_ref,candidate_id=cid,public_inputs=public_ref,routes=routes_ref,
            training_index=training['index'],output_directory=str(out/'workers'/cid),max_events=parallel.MAX_EVENTS)
        path=out/('worker-spec-'+cid+'.json');save(path,spec);spec_refs.append(receipt(path))
    save(out/'jobs.json',jobs)
    roots=[plan_ref['path'],config_ref['path'],training['index']['path'],*[r['path'] for r in training['strict_reviews']]]
    resolution=previous._operational_training_closure(roots,training,historical_manifest_path)
    save(out/'source-resolution.json',resolution)
    sources={r['path'] for r in resolution['materialized_sources']}
    sources.update(str(p.resolve()) for p in previous.prediction_source_paths())
    sources.update(str(p.resolve()) for p in previous.collection_source_paths())
    sources.update(s['original_path'] for s in context['execution_sources'])
    sources.update([str(Path(__file__).resolve()),str(Path(parallel.__file__).resolve()),str(out/'jobs.json'),
        str(out/'source-resolution.json'),preflight['report']['path'],preflight['source_before']['path'],
        preflight['source_after']['path'],training['report']['path']])
    sources.update(r['path'] for r in [*spec_refs,*preflight['results'],*training['artifacts'].values()])
    sources.update(j[k]['path'] for j in jobs for k in ('public_inputs','routes'))
    for source in context['execution_sources']:
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Prepared execution source differs from frozen original bytes')
    protocol=freeze_gate_protocol(model_paths={'learned_context_compiled_v16':config_ref['path']},
        source_paths=sorted(sources),training_data_paths=[training['index']['path']],
        candidates=[dict(id=c['id'],model_id='learned_context_compiled_v16',
            execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
        baseline_candidate_id=plan['baseline_candidate_id'],
        validation_domain=[dict(run_id=plan['run_id'],requests=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']])],
        thresholds=plan['validation_thresholds'],created_unix_ns=time.time_ns(),weights=plan['weights'],
        utility_tie_tolerance=0.,queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],public_input_catalog_path=catalog[0]['path'])
    save(out/'protocol.json',protocol);protocol_ref=receipt(out/'protocol.json')
    before=protocol['sources'];save(out/'source-before.json',before);_verify_protocol(protocol)
    seal=None;workers=[]
    try:
        execution=parallel.run_parallel(spec_refs,protocol_ref,timeout_seconds=worker_timeout_seconds)
        save(out/'parallel-execution.json',execution)
        if (execution['status']!='ALL_FOUR_CPU_WORKERS_EXITED_ZERO'
                or execution.get('all_workers_reaped') is not True or len(execution['workers'])!=4):
            raise ValueError('Four successful reaped fresh CPU workers required; '+str(execution.get('error')))
        cells=[];pids=set()
        for candidate,job,spec_ref,worker in zip(plan['candidates'],jobs,spec_refs,execution['workers']):
            if (worker['candidate_id']!=candidate['id'] or worker['spec']!=spec_ref
                    or worker['exit_code']!=0 or worker.get('receipt') is None):
                raise ValueError('Fresh worker candidate/spec/exit binding changed')
            summary=checked(worker['receipt']);workers.append(summary)
            if (summary.get('schema')!=parallel.WORKER_SCHEMA or summary.get('status')!=parallel.COMPLETE
                    or summary.get('candidate_id')!=candidate['id'] or summary.get('protocol')!=protocol_ref
                    or summary.get('spec')!=spec_ref or summary.get('preflight_cells_reused') is not False
                    or any(summary.get(k) is not True for k in ('frozen_protocol_verified_before',
                        'frozen_protocol_verified_after','provider_constructed_in_worker_after_freeze',
                        'actual_availability_training_raw_all_in_frozen_index'))
                    or type(summary.get('pid')) is not int or summary['pid']<=0 or summary['pid'] in pids
                    or summary['pid']!=worker.get('pid') or summary['pid']==os.getpid()
                    or not summary.get('availability_training_audit')
                    or any(a['raw']['sha256'] not in training['raw_sha256'] for a in summary['availability_training_audit'])):
                raise ValueError('Fresh independent source-verified worker receipt required')
            pids.add(summary['pid']);result=checked(summary['result'])
            public=checked(job['public_inputs'])['forecast_inputs'];routes=checked(job['routes'])
            rows=bind_public(plan,candidate['id'],public,routes,defaults,plan['weights'])
            diagnostic=result.get('static_provider_compilation',{})
            if (result.get('status')!=parallel.COMPLETE or result.get('full_public_cohort')!=rows
                    or result.get('plan')!=plan_ref or result.get('configuration')!=config
                    or result.get('candidate_id')!=candidate['id'] or result.get('CPU_representation_version')!=16
                    or diagnostic.get('nodes_or_edges_dropped') is not False
                    or diagnostic.get('fitted_values_changed') is not False):
                raise ValueError('Every fresh v16 worker must retain the complete original cohort')
            cells.append(gate_cell(result,rows,candidate,plan['run_id'],plan['queue_sample_offsets_ns'],
                phase_start_ns=public['phase_start_ns']))
        _cells(cells,protocol);complete=check_complete(protocol,dict(cells=cells))
        _verify_protocol(protocol);after=[receipt(r['path']) for r in before]
        if after!=before:raise ValueError('Frozen sources changed during fresh parallel execution')
        save(out/'source-after.json',after);save(out/'complete-prediction-review.json',complete)
        seal=seal_predictions(protocol,cells,created_unix_ns=time.time_ns());save(out/'predictions.json',seal)
        _verify_protocol(protocol)
    except Exception as exc:
        # A final source check can fail after writing the seal. Quarantine it so
        # the normal collector path cannot consume a failed production result.
        if (out/'predictions.json').exists():
            (out/'predictions.json').rename(out/'rejected-predictions.json')
        if not (out/'source-after.json').exists():
            observations=[]
            for reference in before:
                try:observations.append(receipt(reference['path']))
                except Exception as source_exc:
                    observations.append(dict(path=reference['path'],verification_error=str(source_exc)))
            save(out/'source-after.json',observations)
        save(out/'failed-production.json',dict(schema=SCHEMA,status='FAILED_PROHIBITS_COLLECTION',
            reason=str(exc),exception=type(exc).__name__,protocol=protocol_ref,worker_audits=workers,
            prediction_seal_written=seal is not None,GPU_started=False))
        raise
    report=dict(schema=SCHEMA,status='COMPLETE_FRESH_PARALLEL_LEARNED_PREDICTIONS_SEALED',
        protocol=protocol_ref,predictions=receipt(out/'predictions.json'),producer=receipt(__file__),
        parallel_helper=receipt(parallel.__file__),parallel_execution=receipt(out/'parallel-execution.json'),
        completeness_review=receipt(out/'complete-prediction-review.json'),source_resolution=receipt(out/'source-resolution.json'),
        numeric_preflight=preflight,training_index_audit=training,worker_audits=workers,
        requests_per_candidate=57,candidate_count=4,independent_CPU_worker_count=4,
        fresh_providers_constructed_after_protocol_freeze=True,preflight_cells_reused=False,
        frozen_sources_unchanged=True,original_thresholds_weights_and_cohort_preserved=True,
        full_historical_transitive_closure_verified=False,operational_training_closure_with_declared_historical_boundaries=True,
        collection_started=False,GPU_started=False,independent_accuracy_established=False,physical_gain_certified=False)
    save(out/'production-receipt.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','training-index','configuration','numeric-preflight-report','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--historical-manifest');parser.add_argument('--worker-timeout-seconds',type=float,default=7200.)
    args=parser.parse_args()
    print(json.dumps(prepare_and_seal(args.plan,args.training_index,args.configuration,args.numeric_preflight_report,
        args.output,historical_manifest_path=args.historical_manifest,worker_timeout_seconds=args.worker_timeout_seconds)))
