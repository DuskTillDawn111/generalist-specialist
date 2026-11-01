"""Independently re-read a real v55 preparation before seal; CPU only.

This is an artifact review, not an implementation review. PASS is written only
after all four complete gzip artifacts, original scores, source snapshots and
the exact original training identity union have been independently checked.
The v55 producer's projection, verification and supplemental-index functions
are intentionally not used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from .complete_training_index_v12 import records_for_capture, validate_index
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .known_input_campaign_seal_v5 import check_complete
from .known_input_gate_v5 import DATA_SCHEMA, _cells
from .known_input_native_flow_v11 import collection_source_paths
from .learned_campaign_seal_v15 import _complete_index, _preserved_plan
from .native_collect import receipt
from .prospective_flow_contract_v11 import bind_public, gate_cell, original_flow_defaults, score_complete_forecast
from .training_source_guard_v11 import availability_training_sources
from .whole57_joint_training_v23 import LIGHT, audit as audit_whole57

SCHEMA = 'gsdeploy.projected_handler_artifact_binding_independent_review.v55'
CAMPAIGN = 'gsdeploy.projected_handler_artifact_binding_campaign.v55'
COMPLETE = 'COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST'
CANDIDATES = ('shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high')
FIELDS = {'schema', 'status', 'configuration', 'plan', 'candidate_id', 'full_public_cohort',
    'execution_context_sha256', 'metrics', 'startup_prediction', 'CPU_representation_version',
    'lifecycle_scheduler_version', 'static_provider_compilation', 'GPU_started',
    'global_request_concurrency_cap_used', 'physical_exclusive_gate_claimed',
    'forecast.cohort', 'forecast.starts', 'forecast.finishes', 'forecast.peak_memory_mib',
    'forecast.board_energy_joules'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and receipt(ref['path']) == ref,
            'Exact immutable path/SHA reference required')
    return ref


def read(ref):
    exact(ref)
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON map key')
            value[key] = item
        return value
    result = json.loads(Path(ref['path']).read_text(), object_pairs_hook=unique,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    exact(ref)
    return result


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def parser_runtime(path):
    root = Path(path).resolve()
    require((root/'ijson/__init__.py').is_file(), 'Explicit streaming parser runtime required')
    sys.path.insert(0, str(root)); parser = importlib.import_module('ijson')
    require(Path(parser.__file__).resolve().is_relative_to(root), 'Parser runtime resolved outside declared directory')
    sources = [receipt(p) for p in sorted((root/'ijson').rglob('*')) if p.is_file() and p.suffix in ('.py', '.so')]
    return parser, sources


class HashingReader:
    def __init__(self, stream):
        self.stream, self.sha, self.size = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size); self.sha.update(data); self.size += len(data)
        return data


def project_gzip(summary, parser):
    """One full streaming parse checks CRC, uncompressed hash and all JSON keys."""
    ref = summary['result']; artifact = exact(ref['artifact']); path = Path(artifact['path'])
    before = path.stat()
    require(ref.get('complete_lossless_roundtrip_verified') is True, 'Declared complete lossless artifact required')
    result, builder, selected, depth, maps = {}, None, None, 0, []
    with gzip.open(path, 'rb') as decompressed:
        reader = HashingReader(decompressed)
        for prefix, event, value in parser.parse(reader, use_float=True):
            if event == 'start_map': maps.append(set())
            elif event == 'map_key':
                require(value not in maps[-1], 'Duplicate JSON map key in complete forecast')
                maps[-1].add(value)
            elif event == 'end_map': maps.pop()
            if builder is not None:
                builder.event(event, value)
                if event in ('start_map', 'start_array'): depth += 1
                elif event in ('end_map', 'end_array'): depth -= 1
                if depth == 0:
                    result[selected] = builder.value; builder = None
            elif prefix in FIELDS and event != 'map_key':
                require(prefix not in result, 'Repeated selected forecast field')
                if event in ('start_map', 'start_array'):
                    selected = prefix; builder = parser.common.ObjectBuilder()
                    builder.event(event, value); depth = 1
                elif event in ('string', 'number', 'boolean', 'null'):
                    result[prefix] = value
        # Force an EOF read even if the selected fields occurred very early.
        require(reader.read(1) == b'', 'Parser must consume complete forecast')
    require(builder is None and not maps and set(result) == FIELDS, 'Complete required forecast projection missing')
    require(reader.sha.hexdigest() == ref['uncompressed_sha256'] and reader.size == ref['uncompressed_bytes'],
            'Full uncompressed forecast SHA/length differs')
    exact(artifact)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Forecast changed during full read')
    result['forecast'] = {k.split('.', 1)[1]:result.pop(k) for k in list(result) if k.startswith('forecast.')}
    return result


def source_snapshots(preparation):
    sources = preparation['sources']
    require(sources and strict_references(sources) == sources and read(preparation['source_before']) == sources,
            'Preparation source-before must equal complete canonical sources')
    for ref in sources: exact(ref)
    before, after = read(preparation['producer_sources_before']), read(preparation['producer_sources_after'])
    require(before and before == after and strict_references(before) == before, 'Producer snapshots must be exact unchanged canonical lists')
    for ref in before:
        exact(ref); require(ref in sources, 'Original producer source omitted from preparation')
    contract = preparation['forecast_contract']
    require(contract['schema'] == 'gsdeploy.projected_handler_artifact_contract.v55', 'Exact v55 artifact contract required')
    required = implementation_sources(contract['factory_source']['path'], contract['preflight_source']['path'])
    require(all(ref in before for ref in required), 'Complete factory/preflight closure not frozen before forecasts')
    require(all(contract[k] in before for k in ('factory_source', 'preflight_source')), 'Exact declared producer entries not frozen')
    require(preparation['producer'] in sources, 'Preparation producer not frozen')
    require(all(ref in sources for ref in implementation_sources(preparation['producer']['path'])),
            'Complete preparation producer closure omitted')
    require(all(receipt(p) in sources for p in collection_source_paths()), 'Actual native collector closure omitted')
    return before, sources


def independent_training(preparation, config, plan):
    supplied = preparation['legacy_training_audit']
    legacy = _complete_index(supplied['index']['path'])
    require(legacy == supplied, 'Original strict training-index audit changed')
    original = read(legacy['artifacts']['source-manifest.json'])
    physical = read(config['observable_context_model']); idle = read(config['idle_SAVE_model'])
    require(physical.get('training_only') is True and physical.get('validation_labels_used') is False
            and len(physical.get('captures', {})) == 4 and idle['captures'] == physical['captures'],
            'Exact same four training-only physical/idle captures required')
    rows, raw_ids, evidence = [], set(), []
    extras = {c['raw']['sha256']:c for c in preparation['supplemental_training_captures']}
    require(set(extras) == set(physical['captures']) and len(extras) == 4, 'Exact four supplemental captures required')
    entries = [(c, False) for c in original['captures']] + [(extras[c], True) for c in sorted(extras)]
    for capture, supplemental in entries:
        raw_ref = capture['raw']; require(raw_ref['sha256'] not in raw_ids, 'Duplicate original training raw')
        if supplemental:
            review = read(capture['review']); plan_ref = review['evidence']['plan.json']; context_ref = review['evidence']['context.json']
            require(review.get('status') == 'PASS' and review.get('issues') == [] and review.get('training_only') is True,
                    'Strict supplemental training review required')
            require(review['evidence']['raw.json'] == raw_ref, 'Supplemental review raw identity changed')
        else:
            plan_ref, context_ref = capture['plan'], capture['context']
        training_plan, context = read(plan_ref), read(context_ref)
        require(training_plan.get('split') == 'training' and training_plan.get('training_only') is True
                and context.get('training_only') is True, 'Qualify explicit original training role before raw open')
        raw = read(raw_ref)
        require(raw.get('status') == 'completed'
                and not raw.get('not_for_training') and raw.get('split') in (None, 'training'), 'Completed original training raw required')
        recovered, _, _ = records_for_capture(raw, training_plan, raw_ref)
        measured, priming = raw['request_records'], raw['priming_request_records']
        require(len(measured) == capture.get('measured_requests', capture.get('measured'))
                and len(priming) == capture.get('priming_requests', capture.get('priming')), 'Exact original measured plus priming population required')
        if supplemental:
            projected = read(capture['projection']); bound = physical['captures'][raw_ref['sha256']]
            require(bound['report'] == capture['projection'] and projected['capture'] == raw_ref['sha256']
                    and projected.get('training_only') is True and projected.get('validation_labels_used') is False
                    and projected['schema'] == 'gsdeploy.whole57_fixed_median_worker_handler_context_risk.v41'
                    and training_plan['domain'] == LIGHT and raw['domain'] == LIGHT
                    and training_plan['route_id'] == bound['route'] == raw['route_id']
                    and len(measured) == 57 and len(priming) == 12, 'Original v41 training projection/raw route differs')
            audit_whole57(Path(plan_ref['path']).parent)
        for row in measured:
            require(row.get('status') == 'completed' and row.get('split') == 'training', 'Every measured row must remain explicitly training')
        for row in priming:
            require(row.get('status') == 'completed' and row.get('split') in (None, 'training')
                    and not row.get('not_for_training'), 'Only qualified original priming permits absent split')
        rows.extend(recovered)
        raw_ids.add(raw_ref['sha256']); evidence.extend([raw_ref, plan_ref, context_ref])
    index = read(preparation['training_index']); validate_index(index)
    signature = lambda values:Counter(tuple(sorted(r.items())) for r in values)
    require(len(rows) == 1281 and signature(rows) == signature(index['records']), 'Exact independently reconstructed 1281 training identities required')
    require(len({(r['run_id'], r['request_id']) for r in rows}) == len(rows), 'Duplicate training identity')
    require(plan['run_id'] not in {r['run_id'] for r in rows}
            and not ({r['request_id'] for r in plan['trace']} & {r['request_id'] for r in rows}), 'Fresh validation identities overlap training')
    base = config['base_v15_configuration']['base_model_configuration']
    require(base['component_bundle'] in original['configuration']['model_lineage_roots'], 'Actual component lineage not covered by complete original index')
    availability = read(base['availability_model'])
    audits = availability_training_sources(availability['source_DAG_reports'])
    require(audits == preparation['availability_training_audit'] and audits
            and all(a['raw']['sha256'] in legacy['raw_sha256'] for a in audits), 'Actual availability raw lineage differs')
    require(all(ref in preparation['sources'] for ref in strict_references([config, physical, idle, audits, evidence])),
            'Numerical model/raw training dependencies omitted from preparation sources')
    return dict(records=len(rows), original_raw_captures=len(raw_ids), legacy_records=legacy['records'],
                supplemental_records=sum(c['measured']+c['priming'] for c in extras.values()),
                training_roles_checked_before_raw_read=True, exact_independent_identity_union=True)


def quality_bundle(bundle):
    composite = read(bundle['composite']); quality, sources = {}, [bundle['composite']]
    for entry in composite['components']:
        spec = entry['spec']; fitted = read(spec['quality']); sources.append(spec['quality'])
        for family, tasks in fitted.items():
            for task, value in tasks.items():
                if task not in spec['tasks']: continue
                previous = quality.setdefault(family, {})
                require(task not in previous or previous[task] == value, 'Conflicting exact quality training component')
                previous[task] = value
    return quality, sources


def verify_forecast(projected, summary, config, plan_ref, plan, candidate, rows, quality, phase_start, contract):
    require(projected['status'] == summary['status'] == COMPLETE and projected['schema'] == contract['forecast_schema']
            and projected['configuration'] == config and projected['plan'] == plan_ref
            and projected['candidate_id'] == candidate['id'] and projected['full_public_cohort'] == rows
            and len(rows) == 57 and projected['execution_context_sha256'] == plan['execution_context_sha256']
            and projected['CPU_representation_version'] == 16
            and projected['lifecycle_scheduler_version'] == contract['lifecycle_scheduler_version']
            and projected['GPU_started'] is False and projected['global_request_concurrency_cap_used'] is False
            and projected['physical_exclusive_gate_claimed'] is False
            and projected['static_provider_compilation']['nodes_or_edges_dropped'] is False
            and projected['static_provider_compilation']['fitted_values_changed'] is False,
            'Complete exact original model/config/cohort/graph forecast required')
    forecast = projected['forecast']; expected = {r['request_id']:r for r in rows}
    require(forecast['cohort'] == expected and set(forecast['starts']) == set(expected)
            and set(forecast['finishes']) == set(expected), 'All 57 cohort/START/receipt identities required')
    for rid, start in forecast['starts'].items():
        require(type(start) is int and start == forecast['finishes'][rid].get('start_ns'), 'Exact integer START must equal completion START')
    metrics = score_complete_forecast(forecast, rows, plan, quality, projected['startup_prediction'], phase_start_ns=phase_start)
    require(metrics == projected['metrics'] == summary['metrics'], 'Independent original six-term metrics differ')
    cell = gate_cell(dict(projected, metrics=metrics), rows, candidate, plan['run_id'], plan['queue_sample_offsets_ns'], phase_start_ns=phase_start)
    require(all(type(v) in (int, float) and math.isfinite(v) for v in cell['terms'].values()), 'Every original six-term value must be finite')
    return cell


def review(preparation_path, parser_root, output):
    out = Path(output).absolute(); require(not out.exists(), 'Exclusive independent review output required')
    own_sources = implementation_sources(__file__)
    test_ref = receipt(Path(__file__).resolve().parents[2]/'tests/test_production_preparation_review_v55.py')
    preparation_ref = receipt(preparation_path); p = read(preparation_ref)
    require(p.get('schema') == CAMPAIGN and p.get('status') == 'CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL'
            and p.get('GPU_started') is False and p.get('protocol_sealed') is False
            and p.get('validation_outcomes_read') is False, 'Actual unsealed CPU-only v55 preparation required')
    before, sources = source_snapshots(p)
    plan, config = read(p['plan']), read(p['configuration']); _preserved_plan(plan)
    require(p['plan'] in before and p['configuration'] in before and config['schema'] == p['forecast_contract']['forecast_schema'],
            'Exact original plan and declared configuration not frozen before prediction')
    training = independent_training(p, config, plan)
    preflight = read(p['numeric_preflight']); contract = p['forecast_contract']
    summaries = preflight['cells']
    require(preflight['schema'] == contract['preflight_schema'] and preflight['status'] == 'FOUR_FULL_COHORT_CPU_PREFLIGHTS_EXECUTED'
            and preflight.get('source_bytes_unchanged') is True and preflight.get('GPU_started') is False
            and preflight.get('validation_labels_read') is False and preflight.get('protocol_sealed') is False
            and preflight.get('collection_eligible') is False and len(summaries) == 4
            and {c['candidate_id'] for c in summaries} == set(CANDIDATES)
            and all(c['status'] == COMPLETE and c['cohort_size'] == c['starts'] == c['receipts'] == 57 for c in summaries),
            'Actual four numeric original 57-request forecasts required')
    parser, parser_sources = parser_runtime(parser_root)
    require(all(ref in sources for ref in parser_sources), 'Independent parser runtime must equal frozen parser bytes')
    bundle_ref = config['base_v15_configuration']['base_model_configuration']['component_bundle']; bundle = read(bundle_ref)
    defaults = original_flow_defaults(plan, bundle['forecast_defaults']); quality, quality_sources = quality_bundle(bundle)
    require(all(ref in sources for ref in quality_sources+[bundle_ref]), 'Original quality bundle sources omitted')
    stored = read(p['verified_cells'])['cells']; cells, artifacts = [], []
    by_summary = {c['candidate_id']:c for c in summaries}
    root = Path(p['numeric_preflight']['path']).parent; prepared = Path(p['plan']['path']).parent
    for candidate in plan['candidates']:
        cid = candidate['id']; summary = by_summary[cid]
        public_ref, routes_ref = receipt(prepared/('public-'+cid+'.json')), receipt(prepared/('routes-'+cid+'.json'))
        require(public_ref in before and routes_ref in before, 'Exact candidate public inputs/routes omitted before prediction')
        public, routes = read(public_ref)['forecast_inputs'], read(routes_ref)
        rows = bind_public(plan, cid, public, routes, defaults, plan['weights'])
        projected = project_gzip(summary, parser)
        cell = verify_forecast(projected, summary, config, p['plan'], plan, candidate, rows, quality, public['phase_start_ns'], contract)
        cell_ref = receipt(root/(cid+'-unsealed-cell.json'))
        require(cell == read(cell_ref) and cell in stored, 'Independently projected gate cell differs from prepared/original cell')
        require(summary['result']['artifact'] in sources and cell_ref in sources, 'Actual forecast/cell artifact omitted from source snapshot')
        cells.append(cell); artifacts.append(dict(candidate_id=cid, artifact=summary['result'], public=public_ref, routes=routes_ref,
            original_unsealed_cell=cell_ref, requests=len(rows), all_numeric_START_values_verified=True,
            original_six_term_metrics_equal=True, original_gate_cell_equal=True))
        print(json.dumps(dict(event='INDEPENDENT_FORECAST_REPROJECTED', candidate_id=cid, requests=len(rows))), flush=True)
    require(Counter(json.dumps(c, sort_keys=True) for c in cells) == Counter(json.dumps(c, sort_keys=True) for c in stored),
            'Prepared verified-cell list differs from complete independent Cartesian product')
    gate = dict(candidates=plan['candidates'], validation_domain={plan['run_id']:[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']]},
                queue_sample_offsets_ns=plan['queue_sample_offsets_ns'])
    _cells(cells, gate); complete = check_complete(gate, dict(cells=cells))
    require(complete == p['completeness'], 'Prepared completeness differs from independent original gate')
    context = read(plan['context'])
    for source in context['execution_sources']:
        exact(source['frozen']); require(receipt(source['original_path'])['sha256'] == source['frozen']['sha256'], 'Original execution source drift')
        require(source['frozen'] in sources and receipt(source['original_path']) in sources,
                'Original and frozen execution source omitted from preparation')
    require([s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')] == [p['public_catalog']],
            'Exact one public runtime content catalog required')
    for ref in sources: exact(ref)
    exact(preparation_ref)
    for ref in own_sources+[test_ref]: exact(ref)
    result = dict(schema=SCHEMA, status='PASS', issues=[], preparation=preparation_ref,
        complete_four_forecasts_reprojected=True, forecast_reviews=artifacts, training_audit=training,
        completeness=complete, checked_sources=sources, reviewer_sources=own_sources, reviewer_tests=[test_ref],
        source_bytes_unchanged=True, full_gzip_CRC_SHA_length_verified=True,
        GPU_started=False, validation_labels_read=False, physical_prediction_accuracy_validated=False,
        provider_constructed_after_freeze=False, preflight_cells_reused=True,
        reviewed_unix_ns=time.time_ns(), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Actual preflight artifact binding, not fresh provider execution after protocol freeze.',
                    'PASS does not establish GPU prediction accuracy, idle physical mark support or six-term gains.'])
    out.mkdir(parents=True, exist_ok=False); save(out/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True); parser.add_argument('--parser-root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = review(args.preparation, args.parser_root, args.output)
    print(json.dumps(dict(status=result['status'], preparation=result['preparation'])))
