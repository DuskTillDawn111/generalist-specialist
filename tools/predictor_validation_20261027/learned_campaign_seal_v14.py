"""Freeze and freshly execute the original learned 4-by-57 validation campaign.

A complete numeric preflight is a prerequisite, never a source of sealed cells.
After protocol freeze this CPU-only command constructs the actual learned
provider and executes every complete declared cohort again. Collection remains
the separate v11 collector's job, with the original v5 precollection checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .complete_training_index_v12 import review_capture, validate_index
from .forecast_gate_v3 import prediction_source_paths
from .historical_evidence_v4 import resolve_closure
from .known_input_campaign_seal_v5 import check_complete
from .known_input_gate_v5 import _cells, _verify_protocol, freeze_gate_protocol, seal_predictions
from .known_input_native_flow_v11 import collection_source_paths, validate_plan
from .learned_availability_forecast_v14 import LearnedAvailabilityForecast, SCHEMA as MODEL_SCHEMA
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .prospective_flow_contract_v11 import bind_public, gate_cell, original_flow_defaults

SCHEMA = 'gsdeploy.learned_campaign_prediction_production.v14'
CANDIDATES = {'shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high'}
COMPLETE = 'COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST'
MAX_EVENTS = 5_000_000


def _read(path):
    return json.loads(Path(path).read_text())


def _preserved_plan(plan):
    validate_plan(plan)
    if (len(plan['trace']) != 57 or {c['id'] for c in plan['candidates']} != CANDIDATES
            or len(plan['candidates']) != 4 or not isinstance(plan.get('run_id'), str)
            or not plan['run_id'] or plan.get('split') != 'validation'):
        raise ValueError('Original prepared four-candidate 57-request validation plan required')
    original = checked(plan['public_contract_source'])
    validate_plan(original)
    for key in ('aliases', 'window_seconds', 'drain_seconds', 'warmup_seconds',
                'workload_rps', 'latency_reference_ms', 'weights', 'memory_budget_mib',
                'energy_reference_watts', 'memory_safe_factor', 'startup_amortization_seconds',
                'baseline_candidate_id', 'queue_sample_offsets_ns', 'queue_depth_boundary',
                'validation_thresholds', 'arrival_process', 'arrival_rate_phases'):
        if plan.get(key) != original.get(key):
            raise ValueError('Original public campaign contract changed: '+key)
    signature = lambda p: [{k:r[k] for k in ('offset_ns', 'task', 'pool_id')} for r in p['trace']]
    if signature(plan) != signature(original):
        raise ValueError('Original complete public arrival/input trace changed')
    old = {c['id']:c for c in original['candidates']}
    if set(old) != CANDIDATES:
        raise ValueError('Original four routing candidates required')
    for candidate in plan['candidates']:
        source = old[candidate['id']]
        settings = lambda c: {k:v for k,v in c.items() if k not in ('routes', 'execution_context_sha256')}
        if (settings(candidate) != settings(source)
                or [candidate['routes'][r['request_id']] for r in plan['trace']]
                != [source['routes'][r['request_id']] for r in original['trace']]):
            raise ValueError('Original routing/resource candidate changed')


def _complete_index(index_path):
    """Recheck supplied strict captures and their exact measured+priming union."""
    index_ref = receipt(index_path); index = checked(index_ref); validate_index(index)
    report_ref = receipt(Path(index_path).with_name('report.json')); report = checked(report_ref)
    if (report.get('schema') != 'gsdeploy.complete_training_index_audit.v12'
            or report.get('status') != 'INDEX_BUILT_PUBLIC_DISJOINT_WITH_EXPLICIT_LINEAGE_BOUNDARIES'
            or report.get('unresolved_raw_lineage_count') != 0
            or report.get('validation_raw_or_labels_read') is not False
            or report['artifacts'].get('training-index.json') != index_ref):
        raise ValueError('Complete source-audited training index with no unresolved raw lineage required')
    for ref in report['artifacts'].values():
        if receipt(ref['path']) != ref:
            raise ValueError('Complete training-index audit artifact changed')
    manifest = checked(report['artifacts']['source-manifest.json'])
    recovered = []; raw_ids = set(); strict_reviews = []
    for capture in manifest['captures']:
        if capture['raw']['sha256'] in raw_ids or not capture['reviews']:
            raise ValueError('Unique strict-reviewed original training captures required')
        records, actual, _ = review_capture(capture['reviews'][0])
        for key in ('run_id', 'raw', 'plan', 'context', 'measured_requests', 'priming_requests'):
            if actual[key] != capture[key]:
                raise ValueError('Complete index capture differs from strict original source')
        for ref in capture['reviews']:
            if receipt(ref['path']) != ref:
                raise ValueError('Training review source changed')
            strict_reviews.append(ref)
        recovered.extend(records); raw_ids.add(capture['raw']['sha256'])
    signature = lambda rows: sorted(tuple(sorted(r.items())) for r in rows)
    if (len(raw_ids) < 2 or signature(recovered) != signature(index['records'])
            or report.get('records') != len(recovered)
            or report.get('independent_original_raw_runs') != len(raw_ids)):
        raise ValueError('Training index must contain every strict measured and priming request exactly')
    return dict(index=index_ref, report=report_ref, artifacts=report['artifacts'],
                raw_sha256=sorted(raw_ids), records=len(recovered), strict_reviews=strict_reviews)


def _numeric_preflight(path, plan_ref, config_ref, plan):
    """Read preflight metadata/forecasts solely to validate the prerequisite."""
    ref = receipt(path); report = checked(ref)
    if (report.get('status') != 'FULL_PUBLIC_PREFLIGHT_EXECUTED'
            or report.get('plan') != plan_ref or report.get('configuration') != config_ref
            or report.get('core_and_public_sources_unchanged') is not True
            or report.get('training_sources_guarded') is not True
            or report.get('full_four_candidate_forecast_executed') is not True
            or report.get('validation_outcomes_read') is not False
            or report.get('GPU_started') is not False
            or report.get('completeness', {}).get('status') != 'COMPLETE_PRESEALED_PREDICTIONS'):
        raise ValueError('Complete numeric four-candidate preflight of this exact plan/configuration required')
    parent = Path(path).resolve().parent
    before_ref, after_ref = receipt(parent/'source-before.json'), receipt(parent/'source-after.json')
    before, after = checked(before_ref), checked(after_ref)
    if not before or before != after or any(receipt(r['path']) != r for r in before):
        raise ValueError('Numeric preflight source identities changed')
    summaries = report['candidates']
    if len(summaries) != 4 or {r['candidate_id'] for r in summaries} != CANDIDATES:
        raise ValueError('Complete numeric preflight candidate domain required')
    # These cells are deliberately local and discarded, never returned or sealed.
    cells = []; result_refs = []; candidates = {c['id']:c for c in plan['candidates']}
    configuration = checked(config_ref)
    bundle = checked(configuration['base_model_configuration']['component_bundle'])
    defaults = original_flow_defaults(plan, bundle['forecast_defaults'])
    for summary in summaries:
        result = checked(summary['result'])
        if (summary.get('status') != COMPLETE or result.get('status') != COMPLETE
                or summary.get('full_declared_requests') != 57 or result.get('plan') != plan_ref
                or result.get('candidate_id') != summary['candidate_id']
                or result.get('configuration') != configuration):
            raise ValueError('Every preflight candidate must be complete with the frozen configuration')
        public = _read(Path(plan_ref['path']).with_name('public-'+summary['candidate_id']+'.json'))['forecast_inputs']
        routes = _read(Path(plan_ref['path']).with_name('routes-'+summary['candidate_id']+'.json'))
        rows = bind_public(plan, summary['candidate_id'], public, routes, defaults, plan['weights'])
        if len(rows) != 57 or result['full_public_cohort'] != rows:
            raise ValueError('Complete numeric preflight cohort required')
        cells.append(gate_cell(result, rows, candidates[summary['candidate_id']], plan['run_id'],
                               plan['queue_sample_offsets_ns'], phase_start_ns=public['phase_start_ns']))
        result_refs.append(summary['result'])
    domain = {plan['run_id']:[{k:r[k] for k in ('request_id', 'pool_id', 'task')} for r in plan['trace']]}
    contract = dict(candidates=plan['candidates'], validation_domain=domain,
                    queue_sample_offsets_ns=plan['queue_sample_offsets_ns'])
    _cells(cells, contract); check_complete(contract, dict(cells=cells))
    return dict(report=ref, source_before=before_ref, source_after=after_ref,
                results=result_refs, numeric_cells_reused_for_seal=False)


def prepare_and_seal(plan_path, training_index_path, configuration_path,
                     numeric_preflight_report_path, output, *, historical_manifest_path=None):
    plan_ref, config_ref = receipt(plan_path), receipt(configuration_path)
    plan, config = checked(plan_ref), checked(config_ref)
    _preserved_plan(plan)
    if config.get('schema') != MODEL_SCHEMA:
        raise ValueError('Actual learned availability configuration required')
    preflight = _numeric_preflight(numeric_preflight_report_path, plan_ref, config_ref, plan)
    training = _complete_index(training_index_path)
    bundle = checked(config['base_model_configuration']['component_bundle'])
    defaults = original_flow_defaults(plan, bundle['forecast_defaults'])
    context = checked(plan['context'])
    catalog = [s['frozen'] for s in context['execution_sources']
               if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(catalog) != 1:
        raise ValueError('One immutable original public input-content catalog required')
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for candidate in plan['candidates']:
        cid = candidate['id']; directory = Path(plan_ref['path']).parent
        public_ref = receipt(directory/('public-'+cid+'.json'))
        routes_ref = receipt(directory/('routes-'+cid+'.json'))
        public, routes = checked(public_ref)['forecast_inputs'], checked(routes_ref)
        bind_public(plan, cid, public, routes, defaults, plan['weights'])
        jobs.append(dict(run_id=plan['run_id'], candidate_id=cid, public_inputs=public_ref, routes=routes_ref))
    save(out/'jobs.json', jobs)
    # The index audit retains an old gate as historical metadata. Bind its
    # bytes without opening that gate or its outcomes. Expand the actual model
    # lineage and each strict training capture instead of that unrelated graph.
    roots = [plan_ref['path'], config_ref['path'], training['index']['path'],
             *[r['path'] for r in training['strict_reviews']]]
    resolution = resolve_closure(roots, historical_manifest_path)
    save(out/'source-resolution.json', resolution)
    sources = {r['path'] for r in resolution['materialized_sources']}
    sources.update(str(p.resolve()) for p in prediction_source_paths())
    sources.update(str(p.resolve()) for p in collection_source_paths())
    sources.update(s['original_path'] for s in context['execution_sources'])
    sources.update([str(Path(__file__).resolve()), str(out/'jobs.json'), str(out/'source-resolution.json'),
                    preflight['report']['path'], preflight['source_before']['path'], preflight['source_after']['path']])
    sources.update(r['path'] for r in preflight['results'])
    sources.update([training['report']['path'], *[r['path'] for r in training['artifacts'].values()]])
    sources.update(j[k]['path'] for j in jobs for k in ('public_inputs', 'routes'))
    for source in context['execution_sources']:
        if receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
            raise ValueError('Prepared execution source differs from frozen original bytes')
    protocol = freeze_gate_protocol(model_paths={'learned_context_v14':config_ref['path']},
        source_paths=sorted(sources), training_data_paths=[training['index']['path']],
        candidates=[dict(id=c['id'], model_id='learned_context_v14',
                         execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
        baseline_candidate_id=plan['baseline_candidate_id'],
        validation_domain=[dict(run_id=plan['run_id'], requests=[{k:r[k] for k in ('request_id', 'pool_id', 'task')}
                                                              for r in plan['trace']])],
        thresholds=plan['validation_thresholds'], created_unix_ns=time.time_ns(), weights=plan['weights'],
        utility_tie_tolerance=0., queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],
        public_input_catalog_path=catalog[0]['path'])
    save(out/'protocol.json', protocol)
    before = protocol['sources']; save(out/'source-before.json', before)
    _verify_protocol(protocol)
    cells = []; audits = []; seal = None
    try:
        # Provider construction and all predictions happen strictly after freeze.
        model = LearnedAvailabilityForecast(config, allow_unvalidated_training_hypothesis=True)
        if any(a['raw']['sha256'] not in training['raw_sha256'] for a in model.availability_training_audit):
            raise ValueError('Actual availability training raw omitted from complete index')
        startup_ref = model.bundle['startup']['training_model']
        for candidate, job in zip(plan['candidates'], jobs):
            _verify_protocol(protocol)
            public, routes = checked(job['public_inputs'])['forecast_inputs'], checked(job['routes'])
            rows = bind_public(plan, candidate['id'], public, routes, defaults, plan['weights'])
            result = model.execute(plan_ref, candidate['id'], public, routes, startup_ref, max_events=MAX_EVENTS)
            result_path = out/('forecast-'+candidate['id']+'.json'); save(result_path, result)
            if result.get('full_public_cohort') != rows:
                raise ValueError('Fresh learned forecast changed the complete frozen cohort')
            cells.append(gate_cell(result, rows, candidate, plan['run_id'], plan['queue_sample_offsets_ns'],
                                   phase_start_ns=public['phase_start_ns']))
            audits.append(dict(candidate_id=candidate['id'], status=result.get('status'), result=receipt(result_path)))
        _cells(cells, protocol)
        complete = check_complete(protocol, dict(cells=cells))
        if any(a['status'] != COMPLETE for a in audits):
            raise ValueError('All freshly executed learned candidates must be numeric COMPLETE')
        _verify_protocol(protocol)
        after = [receipt(r['path']) for r in before]
        if after != before:
            raise ValueError('Frozen sources changed during fresh learned execution')
        save(out/'source-after.json', after)
        save(out/'complete-prediction-review.json', complete)
        seal = seal_predictions(protocol, cells, created_unix_ns=time.time_ns())
        save(out/'predictions.json', seal)
        _verify_protocol(protocol)
    except Exception as exc:
        if not (out/'source-after.json').exists():
            save(out/'source-after.json', [receipt(r['path']) for r in before])
        save(out/'failed-production.json', dict(schema=SCHEMA, status='FAILED_PROHIBITS_COLLECTION',
            reason=str(exc), exception=type(exc).__name__, protocol=receipt(out/'protocol.json'),
            candidate_audits=audits, prediction_seal_written=seal is not None,
            GPU_started=False, independent_accuracy_established=False))
        raise
    report = dict(schema=SCHEMA, status='COMPLETE_FRESH_LEARNED_PREDICTIONS_SEALED',
        protocol=receipt(out/'protocol.json'), predictions=receipt(out/'predictions.json'),
        completeness_review=receipt(out/'complete-prediction-review.json'), source_resolution=receipt(out/'source-resolution.json'),
        producer=receipt(__file__), run_id=plan['run_id'], jobs=audits, training_index_audit=training,
        numeric_preflight=preflight, frozen_sources_unchanged=True, predictions_executed_after_protocol_freeze=True,
        preflight_cells_reused=False, original_thresholds_weights_and_cohort_preserved=True,
        requests_per_candidate=57, candidate_count=4, collection_started=False, GPU_started=False,
        independent_accuracy_established=False, physical_gain_certified=False)
    save(out/'production-receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('plan', 'training-index', 'configuration', 'numeric-preflight-report', 'output'):
        parser.add_argument('--'+field, required=True)
    parser.add_argument('--historical-manifest')
    args = parser.parse_args()
    result = prepare_and_seal(args.plan, args.training_index, args.configuration,
        args.numeric_preflight_report, args.output, historical_manifest_path=args.historical_manifest)
    print(json.dumps({k:result[k] for k in ('status', 'protocol', 'predictions', 'run_id', 'GPU_started')}))


if __name__ == '__main__':
    main()
