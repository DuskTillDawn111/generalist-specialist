"""Compose original-context DAG components without rewriting source models.

The native projection is an explicit collector-invariance hypothesis. Light and
natural projections remain different domains. This module never launches CUDA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from gsdeploy.forecasting.lifecycle import export_training_observations
from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result
from . import dag_host_release_v5 as host
from .kernel_domain_bridge_review_v3 import load_report as load_v3
from .historical_evidence_v4 import HistoricalReceiptResolver, resolve_closure
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .pooled_lifecycle_v4 import PooledLifecycleProvider, fit as fit_pooled_lifecycle
from .natural_quality_fit_v2 import expected_totals

SCHEMA = 'gsdeploy.multi_native_composite_execution_contract.v5'
COMPONENT_FIELDS = {'id', 'tasks', 'dag', 'dag_review', 'light_context', 'natural_context',
                    'host_supply', 'lifecycle', 'quality', 'quality_report', 'historical_resolution'}
OPT_INS = {'collector_invariance', 'light_to_natural_transfer',
           'partial_host_identification', 'family_task_pooling'}


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Changed or malformed multi-native source receipt')
    return json.loads(Path(ref['path']).read_text())


def context(ref, domain, *, require_live=False):
    doc = checked(ref)
    projection = native_execution_projection(doc)
    if projection['measurement_domain'] != domain:
        raise ValueError('Multi-native component measurement domain mismatch')
    for source in doc['execution_sources']:
        frozen = source['frozen']
        if receipt(frozen['path']) != frozen:
            raise ValueError('Changed original frozen execution source')
        if require_live and receipt(source['original_path'])['sha256'] != frozen['sha256']:
            raise ValueError('Actual target execution source differs from frozen bytes')
    return doc, projection


def _load_training(ref):
    report = checked(ref)
    loader = load_v3
    if report.get('schema') == 'gsdeploy.startup_training_collection_strict_review.v4':
        from .small_training_domain_bridge_review_v4 import load_report
        loader = load_report
    return loader(ref['path'], 'natural')


def _fit_host(dag_ref, light_ref, natural_ref, review_refs):
    dag = checked(dag_ref)
    _, light = context(light_ref, 'light-kernel')
    _, natural = context(natural_ref, 'natural')
    if (dag.get('schema') != 'gsdeploy.request_cuda_dag_profiles.v2'
            or dag.get('training_only') is not True
            or dag.get('execution_context_sha256') != light['original_full_context_sha256']):
        raise ValueError('Original light-context training DAG required')
    plans = {(p['alias'], p['task'], p['pool_id']): p for p in dag['profiles']}
    if not plans or len(plans) != len(dag['profiles']):
        raise ValueError('Unique exact-input DAG profiles required')
    captures, observations, rejected, records, seen = [], [], [], [], set()
    for ref in review_refs:
        review, plan, raw = _load_training(ref)
        run = 'raw-sha256:' + review['raw']['sha256']
        if (run in seen or review.get('training_only') is not True
                or raw.get('kind') != 'measured' or raw.get('status') != 'completed'
                or raw.get('execution_context_sha256') != natural['original_full_context_sha256']
                or plan.get('execution_context_sha256') != natural['original_full_context_sha256']):
            raise ValueError('Independent completed original-context natural training captures required')
        _, capture_context = context(review['context'], 'natural')
        if capture_context['original_full_context_sha256'] != natural['original_full_context_sha256']:
            raise ValueError('Natural review context differs from original component')
        seen.add(run)
        captures.append(dict(review=ref, raw=review['raw'], context=review['context'], run_id=run))
        ids = set()
        for row in raw['request_records']:
            if row.get('split') != 'training' or row.get('status') != 'completed' or row['request_id'] in ids:
                raise ValueError('Unique completed training rows required')
            ids.add(row['request_id'])
            records.append(dict(run_id=run, request_id=row['request_id'], pool_id=row['pool_id'], task=row['task']))
            profile = plans.get((row['model'], row['task'], row['pool_id']))
            if profile is None:
                rejected.append(dict(source=run, request_id=row['request_id'], reason='NO_MATCHING_TRAINING_DAG_TEMPLATE'))
                continue
            try:
                value = host.fit_observation(profile, row)
            except UnknownEvidence as exc:
                rejected.append(dict(source=run, request_id=row['request_id'], reason=str(exc)))
                continue
            observations.append(dict(value, source=run, request_id=row['request_id']))
    groups = host.aggregate(observations)
    required = {(p['family'], p['task']) for p in plans.values()}
    supported = {(g['family'], g['task']) for g in groups if g['supported']}
    if len(captures) < 2 or rejected or not required <= supported:
        raise UnknownEvidence('COMPLETE_TWO_CAPTURE_TWO_INPUT_HOST_COMPONENT_REQUIRED')
    return dict(schema=host.SCHEMA, kind='partially_identified_dag_host_supply_hypothesis',
        status='FITTED_CONDITIONAL_HYPOTHESES', dag=dag_ref,
        execution_context_sha256=natural['original_full_context_sha256'],
        light_execution_context_sha256=light['original_full_context_sha256'],
        captures=captures, groups=groups, observations=observations, rejected=rejected,
        supported_group_count=len(supported), independent_natural_capture_count=len(captures),
        fitted_scenarios=list(host.SCENARIOS), identification_complete=False,
        observed_ranges_are_confidence_bounds=False, measured_handler_end_used_as_future_anchor=False,
        profiled_issue_times_used_as_future_release=False, kernel_durations_rescaled=False,
        natural_timing_accuracy_validated=False, deployment_qualified=False, GPU_started=False), records


def fit_host_component(dag_path, light_context_path, natural_context_path, natural_reviews, output):
    """Fit real v5 host parameters and a complete training index, exclusively.

    Inputs retain their original IDs. The model precedes the final composition,
    avoiding a cyclic model/contract receipt. No feasibility report is a model.
    """
    inputs = dict(dag=receipt(dag_path), light_context=receipt(light_context_path),
                  natural_context=receipt(natural_context_path),
                  natural_reviews=[receipt(p) for p in natural_reviews])
    model, records = _fit_host(inputs['dag'], inputs['light_context'], inputs['natural_context'], inputs['natural_reviews'])
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'training-index.json', dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
                                        split='training', records=records))
    model.update(fit_inputs=inputs, training_index=receipt(out / 'training-index.json'),
                 sources=[receipt(__file__), receipt(host.__file__)])
    save(out / 'model.json', model)
    return model


def _verify_host(spec, model):
    if not model.get('sources'):
        raise ValueError('Source-bound host fitting implementation required')
    for ref in model['sources']:
        if receipt(ref['path']) != ref:
            raise ValueError('Changed host fitting implementation source')
    refs = [c['review'] for c in model['captures']]
    expected, records = _fit_host(spec['dag'], spec['light_context'], spec['natural_context'], refs)
    for name in ('schema', 'dag', 'execution_context_sha256', 'light_execution_context_sha256',
                 'captures', 'groups', 'observations', 'rejected'):
        if model.get(name) != expected[name]:
            raise ValueError('Host model differs from recomputed original training evidence: ' + name)
    index = checked(model['training_index'])
    if (index.get('schema') != 'gsdeploy.independent_prediction_gate_training_index.v2'
            or index.get('split') != 'training' or index.get('records') != records):
        raise ValueError('Host training index does not exactly cover real captures')


def _verify_training_components(spec, lifecycle, quality, report, natural_id):
    """Verify nested bytes and rederive numerical handoff/quality from training.

    Closure starts only at the declared components. Historical implementations
    need exact byte-identical snapshots, never a blanket exemption.
    """
    history = spec['historical_resolution']
    if history is not None:
        checked(history)
    resolver = HistoricalReceiptResolver(history['path'] if history else None)
    resolve_closure([spec[k]['path'] for k in ('lifecycle', 'quality_report', 'dag_review')],
                    history['path'] if history else None)
    def read(ref):
        return checked(resolver.resolve(ref))
    sources = lifecycle.get('sources', [])
    if len(sources) < 2:
        raise ValueError('Pooled lifecycle requires original observations and training report')
    observations = read(sources[0])['observations']
    training = read(sources[1])
    if (training.get('execution_context_sha256') != natural_id
            or training.get('observations') != sources[0]):
        raise ValueError('Lifecycle observations differ from original training report')
    if training.get('schema') not in ('gsdeploy.natural_training_lifecycle_fit.v2',
                                     'gsdeploy.natural_training_handoff_quality_fit.v4'):
        raise ValueError('Original natural lifecycle fit report required')
    clock_ref = training['clock_evidence']; clock = read(clock_ref)
    if clock.get('schema') != 'gsdeploy.natural_training_host_clock_evidence.v2' or not clock.get('sources'):
        raise ValueError('Original source-bound host clock evidence required')
    contract = dict(clock['clock_contract'], source='sha256:' + clock_ref['sha256'])
    natural = read(spec['natural_context'])
    aliases = {a: dict(family=f) for a, f in natural['residents'].items()}
    actual_observations, raw_rows, runs = [], [], set()
    for capture in training['captures']:
        raw = read(capture['raw']); run = 'raw-sha256:' + capture['raw']['sha256']
        if run in runs or capture['run_id'] != run or capture['raw'] not in clock['sources']:
            raise ValueError('Independent lifecycle capture and exact clock source required')
        runs.add(run)
        actual_observations.extend(export_training_observations(dict(raw, split='training', admission_enabled=False),
            aliases, execution_context_sha256=natural_id, clock_contract=contract, source=run))
        raw_rows.extend((run, row) for row in raw['request_records'])
    if len(runs) < 2 or observations != actual_observations:
        raise ValueError('Lifecycle observations differ from actual training raw boundaries')
    expected = fit_pooled_lifecycle(observations, natural_id)
    if any(lifecycle.get(key) != value for key, value in expected.items()):
        raise ValueError('Pooled lifecycle differs from recomputed original training observations')
    totals_ref = report.get('training_totals')
    if totals_ref is None:
        raise ValueError('Quality requires original per-request training totals')
    totals = read(totals_ref)['records']
    if 'catalog' in report:
        catalog = read(report['catalog'])
    else:
        candidates = []
        for ref in report.get('sources', []):
            if Path(ref['path']).suffix == '.json':
                value = read(ref)
                if isinstance(value, dict) and 'references' in value.get('quality', {}):
                    candidates.append(value)
        if len(candidates) != 1:
            raise ValueError('One original quality reference catalog required')
        catalog = candidates[0]
    actual_totals = []
    for run, row in raw_rows:
        response = copy.deepcopy(row['worker_response']); task = row['task']; pool = row['pool_id']
        if task == 'ocr':
            response['prediction'] = page_ocr_text(response['prediction'])
        numerator, denominator = sample_totals(task, score_result(catalog['quality']['references'][task][pool], response))
        actual_totals.append(dict(run_id=run, request_id=row['request_id'], family=aliases[row['model']]['family'],
            task=task, pool_id=pool, numerator=numerator, denominator=denominator))
    if totals != actual_totals:
        raise ValueError('Quality totals differ from actual completed training responses and references')
    index = read(report['training_index'])
    expected_records = [{k:r[k] for k in ('run_id', 'request_id', 'pool_id', 'task')} for r in totals]
    if (index.get('split') != 'training' or index.get('records') != expected_records
            or len({(r['run_id'], r['request_id']) for r in totals}) != len(totals)):
        raise ValueError('Quality training index must exactly cover unique actual totals')
    grouped = defaultdict(list)
    for row in totals:
        grouped[row['family'], row['task']].append(row)
    rebuilt = {}
    evidence = 'sha256:' + totals_ref['sha256']
    for (family, task), rows in grouped.items():
        pools = defaultdict(list)
        for row in rows:
            pools[row['pool_id']].append(row)
        rebuilt.setdefault(family, {})[task] = dict(
            by_input={pool: expected_totals(members, evidence) for pool, members in sorted(pools.items())},
            fallback=expected_totals(rows, evidence))
    if quality != rebuilt:
        raise ValueError('Quality differs from recomputed per-request training totals')


def _components(specs, target_ref, *, require_live=False):
    target, target_id = context(target_ref, 'natural', require_live=require_live)
    entries, routes, ids, light_projection = [], set(), set(), None
    for spec in specs:
        if set(spec) != COMPONENT_FIELDS or not spec['id'] or spec['id'] in ids:
            raise ValueError('Explicit unique multi-native component specification required')
        ids.add(spec['id'])
        _, light = context(spec['light_context'], 'light-kernel')
        _, natural = context(spec['natural_context'], 'natural')
        if light_projection is None:
            light_projection = light['projection_sha256']
        if (light['projection_sha256'] != light_projection
                or natural['projection_sha256'] != target_id['projection_sha256']):
            raise ValueError('Components must share native projection separately within each domain')
        dag = checked(spec['dag'])
        if (dag.get('schema') != 'gsdeploy.request_cuda_dag_profiles.v2' or dag.get('training_only') is not True
                or dag.get('execution_context_sha256') != light['original_full_context_sha256']):
            raise ValueError('Component DAG original light identity mismatch')
        tasks = spec['tasks']
        if not tasks or len(set(tasks)) != len(tasks) or {p['task'] for p in dag['profiles']} != set(tasks):
            raise ValueError('Declared component tasks must exactly cover its DAG profiles')
        dag_review = checked(spec['dag_review'])
        if dag_review.get('schema') != 'gsdeploy.multitask_dag_independent_cpu_review.v5':
            raise ValueError('Source-bound canonical DAG coverage review required')
        reviewed = [p for p in dag_review['profiles'] if p['source_dag'] == spec['dag']]
        reviewed_keys = [(p['alias'], p['task'], p['pool_id']) for p in reviewed]
        if (len(reviewed_keys) != len(set(reviewed_keys))
                or set(reviewed_keys) != {(p['alias'], p['task'], p['pool_id']) for p in dag['profiles']}
                or any(p['original_full_context_sha256'] != light['original_full_context_sha256']
                       or p.get('complete_canonical_kernel_identity_match') is not True for p in reviewed)):
            raise ValueError('Complete original-source canonical DAG denominator required')
        for profile in dag['profiles']:
            key = profile['alias'], profile['task'], profile['pool_id']
            if key in routes:
                raise ValueError('Duplicate multi-native alias/task/input route')
            routes.add(key)
            if target['residents'].get(profile['alias']) != profile['family']:
                raise ValueError('DAG owner family differs from target resident')
            if profile.get('cyclic') or profile.get('unresolved'):
                raise UnknownEvidence('INCOMPLETE_OR_CYCLIC_MULTI_NATIVE_DAG')
            host.structure(profile)  # Full topology/sink validation, no kernel-duration substitution.
        lifecycle = checked(spec['lifecycle'])
        report = checked(spec['quality_report'])
        quality = checked(spec['quality'])
        if (lifecycle.get('execution_context_sha256') != natural['original_full_context_sha256']
                or report.get('execution_context_sha256') != natural['original_full_context_sha256']
                or report.get('status') != 'FITTED' or report.get('quality_loss') != spec['quality']):
            raise ValueError('Lifecycle/quality must retain original natural component binding')
        for profile in dag['profiles']:
            if profile['task'] not in quality.get(profile['family'], {}):
                raise UnknownEvidence('MISSING_COMPONENT_TRAINING_QUALITY')
        model = checked(spec['host_supply'])
        _verify_host(spec, model)
        _verify_training_components(spec, lifecycle, quality, report, natural['original_full_context_sha256'])
        entries.append(dict(spec=copy.deepcopy(spec), original_light_context_sha256=light['original_full_context_sha256'],
            original_natural_context_sha256=natural['original_full_context_sha256']))
    if not entries:
        raise ValueError('At least one original-context component required')
    return target_id, light_projection, entries


def compose(specs, target_context_path, *, hypotheses):
    if set(hypotheses) != OPT_INS or any(v is not True for v in hypotheses.values()):
        raise ValueError('All multi-native transfer/identification/pooling hypotheses require explicit opt-in')
    target_ref = receipt(target_context_path)
    target, light_projection, entries = _components(specs, target_ref, require_live=True)
    doc = dict(schema=SCHEMA, target=dict(context=target_ref,
        execution_context_sha256=target['original_full_context_sha256']), components=entries,
        light_native_projection_sha256=light_projection,
        natural_native_projection_sha256=target['projection_sha256'], hypotheses=dict(hypotheses),
        sources=[receipt(__file__), receipt(host.__file__)], original_contexts_rewritten=False,
        original_models_rewritten=False, physical_accuracy_validated=False, deployment_qualified=False)
    doc['composite_execution_sha256'] = identity(doc)
    return doc


class MultiNativeProvider:
    """Dispatch host and lifecycle by exact original DAG route; never relabel."""
    def __init__(self, document, aliases, *, host_scenario, allow_cpu_fixture=False):
        if (document.get('schema') != SCHEMA or identity({k:v for k,v in document.items()
                if k != 'composite_execution_sha256'}) != document.get('composite_execution_sha256')):
            raise ValueError('Unchanged multi-native composition contract required')
        if set(document.get('hypotheses', {})) != OPT_INS or any(v is not True for v in document['hypotheses'].values()):
            raise ValueError('Explicit multi-native hypotheses required')
        for ref in document['sources']:
            if receipt(ref['path']) != ref:
                raise ValueError('Changed multi-native implementation source')
        target, light, entries = _components([e['spec'] for e in document['components']], document['target']['context'], require_live=True)
        if (entries != document['components'] or target['original_full_context_sha256'] != document['target']['execution_context_sha256']
                or target['projection_sha256'] != document['natural_native_projection_sha256']
                or light != document['light_native_projection_sha256']):
            raise ValueError('Recomputed multi-native source identities differ')
        self.execution_context_sha256 = document['composite_execution_sha256']
        self.target_execution_context_sha256 = target['original_full_context_sha256']
        self.plans, self.routes, self.components, self.quality = {}, {}, {}, {}
        self.physical_evidence = False
        for entry in entries:
            spec = entry['spec']; cid = spec['id']; dag = checked(spec['dag'])
            plans = {(p['alias'], p['task'], p['pool_id']): p for p in dag['profiles']}
            for key, profile in plans.items():
                if (key[0] not in aliases or aliases[key[0]]['family'] != profile['family']
                        or key[1] not in aliases[key[0]]['supported_tasks']):
                    raise ValueError('Declared alias configuration differs from original DAG owner')
                self.plans[key] = profile; self.routes[key] = cid
            natural = entry['original_natural_context_sha256']
            release = host.DagHostReleaseProvider(checked(spec['host_supply']), natural, plans, aliases,
                scenario=host_scenario, allow_identification_hypothesis=True, allow_cross_input_pooling_hypothesis=True)
            lifecycle = PooledLifecycleProvider(checked(spec['lifecycle']), natural,
                allow_cpu_fixture=allow_cpu_fixture, allow_pooling_hypothesis=True)
            self.components[cid] = dict(entry=entry, release=release, lifecycle=lifecycle)
            for family, tasks in checked(spec['quality']).items():
                for task, value in tasks.items():
                    if task not in spec['tasks']:
                        continue
                    existing = self.quality.setdefault(family, {})
                    if task in existing and existing[task] != value:
                        raise ValueError('Conflicting original family/task training quality')
                    existing[task] = value

    def component(self, row):
        key = row['alias'], row['task'], row['pool_id']
        if key not in self.routes:
            raise UnknownEvidence('NO_EXACT_MULTI_NATIVE_DAG_ROUTE')
        return self.components[self.routes[key]]

    def release(self, node, row):
        # Engine requires exactly these two numeric fields; provenance is separate.
        return self.component(row)['release'](node, row)

    def delays(self, row, aliases, scenario):
        component = self.component(row)
        result = component['lifecycle'].delays(row, aliases, 'mean' if scenario == 'profiled' else scenario)
        return dict(result, composite_execution_sha256=self.execution_context_sha256,
            original_execution_context_sha256=component['entry']['original_natural_context_sha256'],
            natural_timing_validated=False)

    def provenance(self, row):
        entry = self.component(row)['entry']
        return dict(component_id=entry['spec']['id'], dag=entry['spec']['dag'],
            original_light_context_sha256=entry['original_light_context_sha256'],
            original_natural_context_sha256=entry['original_natural_context_sha256'],
            composite_execution_sha256=self.execution_context_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit-host')
    for key in ('dag', 'light-context', 'natural-context', 'output'):
        fit.add_argument('--' + key, required=True)
    fit.add_argument('--natural-review', action='append', required=True)
    build = sub.add_parser('compose')
    for key in ('specification', 'target-context', 'output'):
        build.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if args.command == 'fit-host':
        fit_host_component(args.dag, args.light_context, args.natural_context, args.natural_review, args.output)
    else:
        spec = json.loads(Path(args.specification).read_text())
        save(Path(args.output), compose(spec['components'], args.target_context, hypotheses=spec['hypotheses']))


if __name__ == '__main__':
    main()
