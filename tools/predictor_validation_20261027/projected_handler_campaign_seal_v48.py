"""CPU artifact binding of explicitly declared complete forecasts to the v5 gate.

This is explicitly not the v16 fresh-provider production contract. Existing
preflight cells are independently reprojected, then sealed only after every
numeric prerequisite and an independent review pass. No GPU entry exists.
"""
from __future__ import annotations
import argparse
import gzip
import importlib
import json
import math
from pathlib import Path
import sys
import time

from . import learned_campaign_seal_v15 as legacy
from . import whole57_joint_training_v23 as training23
from . import unknown_preflight_archive_v1 as io
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .known_input_gate_v5 import DATA_SCHEMA,_cells,_verify_protocol,freeze_gate_protocol,seal_predictions
from .known_input_campaign_seal_v5 import check_complete
from .known_input_native_flow_v11 import collection_source_paths
from .known_input_observed_v5 import _bindings
from .native_collect import receipt
from .prospective_flow_contract_v11 import bind_public,gate_cell,original_flow_defaults,score_complete_forecast
from .projected_handler_forecast_v44 import SCHEMA as MODEL_SCHEMA
from .training_source_guard_v11 import availability_training_sources

SCHEMA='gsdeploy.projected_handler_artifact_binding_campaign.v48'
PREFLIGHT_SCHEMA='gsdeploy.projected_worker_handler_context_full_cohort_CPU_preflight.v44'
MODEL_ID='projected_worker_handler_artifact_bound_v48'
COMPLETE='COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST'
SELECTED={'schema','status','reason','configuration','plan','candidate_id','full_public_cohort',
 'execution_context_sha256','metrics','startup_prediction','CPU_representation_version',
 'lifecycle_scheduler_version','static_provider_compilation','GPU_started',
 'global_request_concurrency_cap_used','physical_exclusive_gate_claimed',
 'forecast.cohort','forecast.starts','forecast.finishes','forecast.peak_memory_mib','forecast.board_energy_joules'}


def forecast_contract(path=None):
    contract=checked(receipt(path)) if path is not None else dict(
        schema='gsdeploy.projected_handler_artifact_contract.v48',forecast_schema=MODEL_SCHEMA,
        preflight_schema=PREFLIGHT_SCHEMA,lifecycle_scheduler_version=44,
        factory_source=receipt(Path(__file__).with_name('projected_handler_forecast_v44.py')),
        preflight_source=receipt(Path(__file__).with_name('projected_handler_preflight_v44.py')))
    if (set(contract)!={'schema','forecast_schema','preflight_schema','lifecycle_scheduler_version','factory_source','preflight_source'}
            or contract['schema']!='gsdeploy.projected_handler_artifact_contract.v48'
            or any(not isinstance(contract[k],str) or not contract[k] for k in ('forecast_schema','preflight_schema'))
            or type(contract['lifecycle_scheduler_version']) is not int or contract['lifecycle_scheduler_version']<44):
        raise ValueError('Explicit exact factory/preflight schema and immutable implementation contract required')
    checked_bytes(contract['factory_source']);checked_bytes(contract['preflight_source'])
    return contract


def checked(ref):
    if set(ref)!={'path','sha256'} or receipt(ref['path'])!=ref:raise ValueError('Exact immutable reference required')
    return json.loads(Path(ref['path']).read_text())


def parser_runtime(directory):
    directory=Path(directory).resolve()
    if not (directory/'ijson/__init__.py').is_file():raise ValueError('Explicit installed streaming parser runtime required')
    sys.path.insert(0,str(directory));module=importlib.import_module('ijson')
    if not Path(module.__file__).resolve().is_relative_to(directory):raise ValueError('Streaming parser resolved outside frozen runtime')
    refs=[receipt(p) for p in sorted((directory/'ijson').rglob('*')) if p.is_file() and p.suffix in ('.py','.so')]
    return module,refs


def forecast_projection(summary,parser):
    ref=summary['result'];path=Path(ref['artifact']['path']);before=io.file_stat(path)
    if receipt(path)!=ref['artifact'] or ref.get('complete_lossless_roundtrip_verified') is not True:
        raise ValueError('Exact roundtrip gzip forecast artifact required')
    io.verify_gzip(path,ref['uncompressed_sha256'],ref['uncompressed_bytes'])
    result={};builder=None;selected=None;depth=0;maps=[]
    with gzip.open(path,'rb') as stream:
        for prefix,event,value in parser.parse(stream,use_float=True):
            if event=='start_map':maps.append(set())
            elif event=='map_key':
                if value in maps[-1]:raise ValueError('Duplicate JSON map key in forecast')
                maps[-1].add(value)
            elif event=='end_map':maps.pop()
            if builder is not None:
                builder.event(event,value)
                if event in ('start_map','start_array'):depth+=1
                elif event in ('end_map','end_array'):depth-=1
                if depth==0:
                    result[selected]=builder.value;builder=None
            elif prefix in SELECTED and event!='map_key':
                if prefix in result:raise ValueError('Repeated selected forecast field')
                if event in ('start_map','start_array'):
                    selected=prefix;builder=parser.common.ObjectBuilder();builder.event(event,value);depth=1
                elif event in ('string','number','boolean','null'):result[prefix]=value
    if builder is not None:raise ValueError('Truncated selected forecast')
    if io.file_stat(path)!=before or receipt(path)!=ref['artifact']:raise ValueError('Forecast changed during independent projection')
    result['forecast']={k.split('.',1)[1]:result.pop(k) for k in list(result) if k.startswith('forecast.')}
    return result


def numeric_summary(report,contract=None):
    schema=PREFLIGHT_SCHEMA if contract is None else contract['preflight_schema']
    if (report.get('schema')!=schema or report.get('status')!='FOUR_FULL_COHORT_CPU_PREFLIGHTS_EXECUTED'
            or report.get('source_bytes_unchanged') is not True or report.get('all_original_six_term_metrics_available') is not True
            or report.get('GPU_started') is not False or report.get('validation_labels_read') is not False
            or report.get('collection_eligible') is not False or report.get('protocol_sealed') is not False
            or len(report.get('cells',[]))!=4 or {c['candidate_id'] for c in report['cells']}!=legacy.CANDIDATES):
        raise ValueError('Completed unchanged-source four-candidate declared unsealed preflight required')
    for cell in report['cells']:
        if (cell.get('status')!=COMPLETE or cell.get('cohort_size')!=57 or cell.get('starts')!=57
                or cell.get('receipts')!=57 or cell.get('metrics') is None):
            raise ValueError('Any UNKNOWN, incomplete cohort or missing numeric metrics prohibits sealing')


def quality_from_bundle(bundle):
    composite=checked(bundle['composite']);quality={};refs=[bundle['composite']]
    for entry in composite['components']:
        spec=entry['spec'];source=checked(spec['quality']);refs.append(spec['quality'])
        for family,tasks in source.items():
            for task,value in tasks.items():
                if task not in spec['tasks']:continue
                current=quality.setdefault(family,{})
                if task in current and current[task]!=value:raise ValueError('Conflicting frozen quality components')
                current[task]=value
    return quality,refs


def verify_projection(result,summary,stored_cell,plan,plan_ref,config,rows,candidate,phase_start,quality,contract=None):
    schema=MODEL_SCHEMA if contract is None else contract['forecast_schema']
    version=44 if contract is None else contract['lifecycle_scheduler_version']
    if (result.get('schema')!=schema or result.get('status')!=COMPLETE or result.get('configuration')!=config
            or result.get('plan')!=plan_ref or result.get('candidate_id')!=candidate['id']
            or result.get('execution_context_sha256')!=plan['execution_context_sha256']
            or result.get('full_public_cohort')!=rows or len(rows)!=57
            or result.get('CPU_representation_version')!=16 or result.get('lifecycle_scheduler_version')!=version
            or result.get('GPU_started') is not False or result.get('global_request_concurrency_cap_used') is not False
            or result.get('physical_exclusive_gate_claimed') is not False
            or result['static_provider_compilation'].get('nodes_or_edges_dropped') is not False
            or result['static_provider_compilation'].get('fitted_values_changed') is not False):
        raise ValueError('Exact complete declared model, graph, context and 57 cohort required')
    expected={r['request_id'] for r in rows};forecast=result['forecast']
    if set(forecast.get('starts',{}))!=expected or set(forecast.get('finishes',{}))!=expected:
        raise ValueError('Every declared request must have a numeric start and receipt')
    metrics=score_complete_forecast(forecast,rows,plan,quality,result['startup_prediction'],phase_start_ns=phase_start)
    if metrics!=result['metrics'] or result['metrics']!=summary['metrics']:
        raise ValueError('Independent original six-term reprojection differs from sealed-artifact prerequisite')
    cell=gate_cell(dict(result,metrics=metrics),rows,candidate,plan['run_id'],plan['queue_sample_offsets_ns'],phase_start_ns=phase_start)
    if cell!=stored_cell:raise ValueError('Original gate-cell projection differs from unsealed cell')
    return cell


def supplemental_training(config,legacy_training):
    """Explicit v23 raw union; risk validation captures are never inputs."""
    model=checked(config['observable_context_model']);rows=[];refs=[config['observable_context_model']];captures=[]
    if model.get('training_only') is not True or model.get('validation_labels_used') is not False or len(model.get('captures',{}))!=4:
        raise ValueError('Four explicit training-only context captures required')
    routes=set()
    for sha,capture in model['captures'].items():
        route=capture['route'];routes.add(route)
        review_path=io.ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01'/training23.LIGHT/route/'report.json'
        review_ref=receipt(review_path);review=checked(review_ref);evidence=review['evidence']
        plan=checked(evidence['plan.json'])
        if plan.get('split')!='training' or plan.get('training_only') is not True:raise ValueError('Training qualification required before reading raw')
        raw=checked(evidence['raw.json']);training23.audit(Path(evidence['plan.json']['path']).parent)
        if (review.get('schema')!='gsdeploy.whole57_joint_training_review.v23' or review.get('status')!='PASS'
                or review.get('issues') or review.get('requests')!=57 or review.get('training_only') is not True
                or evidence['raw.json']['sha256']!=sha or raw.get('status')!='completed'
                or raw.get('training_only') is not True or raw.get('domain')!=training23.LIGHT
                or len(raw['request_records'])!=57 or len(raw['priming_request_records'])!=12
                or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))):
            raise ValueError('Context model capture must bind actual strict whole57 training review')
        projection=checked(capture['report'])
        if projection.get('capture')!=sha or projection.get('training_only') is not True or projection.get('validation_labels_used') is not False:
            raise ValueError('Projected context training capture differs from reviewed raw')
        refs.extend([review_ref,*evidence.values(),capture['report'],*projection['sources']])
        for row in raw['request_records']+raw['priming_request_records']:
            if row.get('status')!='completed':raise ValueError('All measured+priming training requests must complete')
            rows.append(dict(run_id='raw-sha256:'+sha,**{k:row[k] for k in ('request_id','pool_id','task')}))
        captures.append(dict(raw=evidence['raw.json'],review=review_ref,projection=capture['report'],measured=57,priming=12))
    if routes!={'balanced','skewed','spill_low','spill_high'}:raise ValueError('Four unique context training routes required')
    base=checked(legacy_training['index']);union=base['records']+rows
    keys=[(r['run_id'],r['request_id']) for r in union]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate supplemental training identity')
    return dict(schema=DATA_SCHEMA,split='training',records=union),strict_references(refs),captures


def availability_training(config,training):
    base=config['base_v15_configuration']['base_model_configuration']
    manifest=checked(training['artifacts']['source-manifest.json'])
    if base['component_bundle'] not in manifest['configuration']['model_lineage_roots']:
        raise ValueError('Component bundle must be covered by strict original raw training lineage')
    model=checked(base['availability_model'])
    audits=availability_training_sources(model['source_DAG_reports'])
    if not audits or any(a['raw']['sha256'] not in training['raw_sha256'] for a in audits):
        raise ValueError('Actual availability training raw omitted from strict complete index')
    return audits,strict_references([base['availability_model'],audits])


def additional_execution_sources(path):
    if path is None:return []
    manifest=checked(receipt(path))
    if (set(manifest)!={'schema','sources'} or manifest['schema']!='gsdeploy.additional_execution_sources.v48'
            or not isinstance(manifest['sources'],list) or not manifest['sources']):
        raise ValueError('Explicit nonempty additional execution source receipt manifest required')
    references=[receipt(path)]
    for ref in manifest['sources']:
        checked_bytes(ref)
        if Path(ref['path']).suffix!='.py':raise ValueError('Additional execution entries must be Python implementations')
        references.extend(implementation_sources(ref['path']))
        if ref not in references:raise ValueError('Additional execution entry differs from static import closure')
    return strict_references(references)


def prepare(plan_path,configuration_path,preflight_report_path,legacy_training_index,output,parser_root,contract_path=None,execution_sources_path=None):
    out=Path(output).absolute()
    if out.exists():raise FileExistsError(out)
    plan_ref=receipt(plan_path);config_ref=receipt(configuration_path);preflight_ref=receipt(preflight_report_path)
    contract=forecast_contract(contract_path)
    plan=checked(plan_ref);config=checked(config_ref);legacy._preserved_plan(plan)
    if config.get('schema')!=contract['forecast_schema']:raise ValueError('Exact declared forecast configuration required')
    report=checked(preflight_ref);numeric_summary(report,contract) # Fail before any seal, parser or new output on UNKNOWN.
    root=Path(preflight_report_path).absolute().parent
    before_ref=receipt(root/'source-before.json');after_ref=receipt(root/'source-after.json')
    before=checked(before_ref);after=checked(after_ref)
    if not before or before!=after:raise ValueError('Exact unchanged producer source-before/after required')
    for ref in before:checked_bytes(ref)
    if plan_ref not in before or config_ref not in before:raise ValueError('Original preflight plan/config not frozen')
    if any(contract[k] not in before for k in ('factory_source','preflight_source')):
        raise ValueError('Declared exact factory/preflight implementations were not frozen during prediction')
    parser,parser_refs=parser_runtime(parser_root)
    execution_sources=additional_execution_sources(execution_sources_path)
    training=legacy._complete_index(legacy_training_index)
    availability_audit,availability_refs=availability_training(config,training)
    index,extra_refs,captures=supplemental_training(config,training)
    bundle_ref=config['base_v15_configuration']['base_model_configuration']['component_bundle'];bundle=checked(bundle_ref)
    defaults=original_flow_defaults(plan,bundle['forecast_defaults']);quality,quality_refs=quality_from_bundle(bundle)
    cells=[];artifacts=[]
    for summary in report['cells']:
        cid=summary['candidate_id'];candidate=next(c for c in plan['candidates'] if c['id']==cid)
        public_ref=receipt(Path(plan_path).absolute().parent/('public-'+cid+'.json'));routes_ref=receipt(Path(plan_path).absolute().parent/('routes-'+cid+'.json'))
        if public_ref not in before or routes_ref not in before:raise ValueError('Candidate public/routes not frozen during prediction')
        public=checked(public_ref)['forecast_inputs'];routes=checked(routes_ref)
        rows=bind_public(plan,cid,public,routes,defaults,plan['weights'])
        cell_ref=receipt(root/(cid+'-unsealed-cell.json'));stored=checked(cell_ref)
        projected=forecast_projection(summary,parser)
        cells.append(verify_projection(projected,summary,stored,plan,plan_ref,config,rows,candidate,public['phase_start_ns'],quality,contract))
        artifacts.extend([summary['result']['artifact'],cell_ref,public_ref,routes_ref])
    gate_contract=dict(candidates=plan['candidates'],validation_domain={plan['run_id']:[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']]},queue_sample_offsets_ns=plan['queue_sample_offsets_ns'])
    _cells(cells,gate_contract);complete=check_complete(gate_contract,dict(cells=cells))
    context=checked(plan['context'])
    catalog=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(catalog)!=1:raise ValueError('Exactly one immutable public input-content catalog required')
    sources=strict_references([before,extra_refs,availability_refs,execution_sources,parser_refs,quality_refs,bundle_ref,preflight_ref,before_ref,after_ref,
        plan_ref,config_ref,training['index'],training['report'],list(training['artifacts'].values()),training['strict_reviews'],
        artifacts,plan['context'],catalog,contract,([] if contract_path is None else [receipt(contract_path)]),implementation_sources(__file__),*[receipt(p) for p in collection_source_paths()],
        [s['frozen'] for s in context['execution_sources']],*[receipt(s['original_path']) for s in context['execution_sources']]])
    for ref in sources:checked_bytes(ref)
    for source in context['execution_sources']:
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:raise ValueError('Original prepared execution source drift')
    out.mkdir(parents=True,exist_ok=False)
    io.save_synced(out/'training-index.json',index);io.save_synced(out/'verified-cells.json',dict(cells=cells))
    io.save_synced(out/'source-before.json',sources)
    result=dict(schema=SCHEMA,status='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL',
        plan=plan_ref,configuration=config_ref,forecast_contract=contract,numeric_preflight=preflight_ref,producer_sources_before=before_ref,producer_sources_after=after_ref,
        source_before=receipt(out/'source-before.json'),sources=sources,training_index=receipt(out/'training-index.json'),
        legacy_training_audit=training,availability_training_audit=availability_audit,supplemental_training_captures=captures,verified_cells=receipt(out/'verified-cells.json'),
        completeness=complete,public_catalog=catalog[0],parser_sources=parser_refs,additional_execution_sources=execution_sources,producer=receipt(__file__),
        preflight_cells_reused=True,provider_constructed_after_freeze=False,all_four_results_independently_reprojected=True,
        original_thresholds_weights_cohorts_contexts_preserved=True,GPU_started=False,protocol_sealed=False,
        physical_prediction_accuracy_validated=False,validation_outcomes_read=False)
    io.save_synced(out/'preparation.json',result);return result


def checked_bytes(ref):
    if set(ref)!={'path','sha256'} or receipt(ref['path'])!=ref:raise ValueError('Frozen source bytes changed: '+str(ref))


def seal(preparation_path,independent_review_path,output):
    preparation_ref=receipt(preparation_path);p=checked(preparation_ref);review_ref=receipt(independent_review_path);review=checked(review_ref)
    if (p.get('schema')!=SCHEMA or p.get('status')!='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL'
            or p.get('all_four_results_independently_reprojected') is not True
            or review.get('schema')!='gsdeploy.projected_handler_artifact_binding_independent_review.v48'
            or review.get('status')!='PASS' or review.get('preparation')!=preparation_ref
            or review.get('complete_four_forecasts_reprojected') is not True or review.get('issues')!=[]):
        raise ValueError('Exact complete preparation and independent artifact review required before seal')
    for ref in p['sources']:checked_bytes(ref)
    plan=checked(p['plan']);legacy._preserved_plan(plan);cells=checked(p['verified_cells'])['cells']
    sources=strict_references([p['sources'],preparation_ref,review_ref,p['verified_cells'],p['training_index'],p['source_before']])
    out=Path(output).absolute();out.mkdir(parents=True,exist_ok=False)
    protocol=freeze_gate_protocol(model_paths={MODEL_ID:p['configuration']['path']},source_paths=[r['path'] for r in sources],
        training_data_paths=[p['training_index']['path']],candidates=[dict(id=c['id'],model_id=MODEL_ID,execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
        baseline_candidate_id=plan['baseline_candidate_id'],validation_domain=[dict(run_id=plan['run_id'],requests=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']])],
        thresholds=plan['validation_thresholds'],created_unix_ns=time.time_ns(),weights=plan['weights'],utility_tie_tolerance=0.,
        queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],public_input_catalog_path=p['public_catalog']['path'])
    io.save_synced(out/'protocol.json',protocol);_cells(cells,protocol);complete=check_complete(protocol,dict(cells=cells))
    seal_written=False
    try:
        _verify_protocol(protocol);predictions=seal_predictions(protocol,cells,created_unix_ns=time.time_ns())
        io.save_synced(out/'predictions.json',predictions);seal_written=True
        _verify_protocol(protocol);_bindings(out/'protocol.json',out/'predictions.json',p['plan']['path'],plan['run_id'])
        after=[receipt(r['path']) for r in sources]
        if after!=sources:raise ValueError('Source drift during artifact-bound seal')
        io.save_synced(out/'source-after.json',after)
        result=dict(schema=SCHEMA,status='ARTIFACT_BOUND_PREDICTIONS_SEALED_REQUIRES_POSTSEAL_GPU_VALIDATION',
            protocol=receipt(out/'protocol.json'),predictions=receipt(out/'predictions.json'),preparation=preparation_ref,
            independent_review=review_ref,completeness=complete,preflight_cells_reused=True,provider_constructed_after_freeze=False,
            original_v5_precollection_bindings_passed=True,all_GPU_must_start_after_seal=True,GPU_started=False,
            physical_prediction_accuracy_validated=False)
        io.save_synced(out/'production-receipt.json',result);return result
    except BaseException as exc:
        if seal_written:(out/'predictions.json').rename(out/'rejected-predictions.json')
        io.save_synced(out/'failed-production.json',dict(status='FAILED_PROHIBITS_COLLECTION',reason=str(exc),GPU_started=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    command=sub.add_parser('prepare')
    for key in ('plan','configuration','preflight-report','legacy-training-index','output','parser-root'):command.add_argument('--'+key,required=True)
    command.add_argument('--forecast-contract')
    command.add_argument('--additional-execution-sources')
    command=sub.add_parser('seal')
    for key in ('preparation','independent-review','output'):command.add_argument('--'+key,required=True)
    command.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    result=prepare(args.plan,args.configuration,args.preflight_report,args.legacy_training_index,args.output,args.parser_root,args.forecast_contract,args.additional_execution_sources) if args.op=='prepare' else seal(args.preparation,args.independent_review,args.output)
    print(json.dumps(dict(status=result['status'],GPU_started=False)))
