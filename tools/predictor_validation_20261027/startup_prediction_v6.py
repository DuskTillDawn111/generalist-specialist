"""Recompute v5 startup evidence and bind predictions to an executable flow plan.

Loading/priming means remain hypotheses. Warmup, admission, changed input
contents and changed executing sources cannot silently reuse these estimates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from gsdeploy.resources import identity
from . import known_input_native_flow_v5 as flow
from . import startup_prediction_v5 as original
from .generation_work_v4 import input_content_identity
from .historical_evidence_v4 import HistoricalReceiptResolver, resolve_closure
from .native_collect import context_identity, receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .operator_profile_campaign_v2 import stat_receipt
from .profiling import RESIDENTS, digest

SCHEMA = 'gsdeploy.flow_bound_premeasurement_startup_prediction.v6'


def finite_nonnegative(value, field):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative startup ' + field + ' required')
    return value


def read(ref, resolver=None):
    resolved = resolver.resolve(ref) if resolver else ref
    if receipt(resolved['path']) != resolved:
        raise ValueError('Changed startup evidence: ' + resolved['path'])
    return json.loads(Path(resolved['path']).read_text())


def public_catalog(context, *, resolver=None, require_live=False):
    sources = [s for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(sources) != 1:
        raise ValueError('One source-bound actual runtime public catalog required')
    source = sources[0]
    frozen = resolver.resolve(source['frozen']) if resolver else source['frozen']
    if receipt(frozen['path']) != frozen:
        raise ValueError('Changed immutable runtime input catalog')
    if require_live and receipt(source['original_path'])['sha256'] != frozen['sha256']:
        raise ValueError('Actual executing public catalog differs from frozen input source')
    rows = list(map(json.loads, Path(frozen['path']).read_text().splitlines()))
    if len({r['pool_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate public catalog input identity')
    return {r['pool_id']: r for r in rows}, dict(frozen=source['frozen'], original_path=source['original_path'])


def priming_content(inputs, residents, catalog, *, verify_media=True):
    if residents != RESIDENTS:
        raise ValueError('Original four physical owners required')
    if len({r['pool_id'] for r in inputs}) != len(inputs):
        raise ValueError('Distinct declared priming inputs required')
    declarations = {r['pool_id']: r for r in inputs}
    result, media_checked = [], {}
    for planned in original.planned_priming(inputs, residents):
        declaration = declarations[planned['pool_id']]
        pool = catalog[planned['pool_id']]
        if pool['task_type'] != planned['task']:
            raise ValueError('Priming task differs from actual catalog content')
        sample = pool['sample']
        if planned['task'] == 'text':
            exact = digest({k: sample[k] for k in ('context', 'question')})
            size = len(sample['context']) + len(sample['question'])
        else:
            exact = sample['input_sha256']
            if verify_media:
                path = sample['input']
                if path not in media_checked:
                    media_checked[path] = receipt(path)['sha256']
                if media_checked[path] != exact:
                    raise ValueError('Actual priming media bytes differ from declared public content')
            size = sample.get('duration_s', Path(sample['input']).stat().st_size)
        if declaration.get('exact_input_sha256') != exact or declaration.get('input_size') != size:
            raise ValueError('Declared priming hash/size differs from real public input content')
        result.append(dict(planned, public_input_content_sha256=input_content_identity(pool),
                           exact_input_sha256=exact, input_size=size))
    if len(result) != 12:
        raise ValueError('Exactly the original twelve priming requests required')
    return result


def verify_model(model_path):
    model_ref = receipt(model_path)
    model = read(model_ref)
    if model.get('schema') != original.SCHEMA:
        raise ValueError('Original immutable v5 premeasurement training model required')
    if not model.get('implementation_sources'):
        raise ValueError('Source-bound v5 startup implementation required')
    for source in model['implementation_sources']:
        if receipt(source['path']) != source:
            raise ValueError('Changed v5 startup training implementation')
    resolution = model.get('historical_resolution')
    if resolution:
        read(resolution)
    resolver = HistoricalReceiptResolver(resolution['path'] if resolution else None)
    if not model.get('sources'):
        raise ValueError('Actual independent strict startup training reviews required')
    closure = resolve_closure([r['path'] for r in model['sources']], resolution['path'] if resolution else None)
    if closure != model['evidence_closure']:
        raise ValueError('Startup training closure differs')
    observations, contexts, seen = [], [], set()
    projection, priming_id, content_id = None, None, None
    contents = None
    for source in model['sources']:
        report = read(source, resolver)
        if (report.get('schema') != 'gsdeploy.startup_training_collection_strict_review.v4'
                or report.get('status') != 'PASS' or report.get('issues') or not report.get('evaluator_sources')):
            raise ValueError('Actual strict PASS startup training review required')
        for evaluator in report['evaluator_sources']:
            resolver.resolve(evaluator)
        raw, plan, context = (read(report[k], resolver) for k in ('raw', 'plan', 'context'))
        run = 'raw-sha256:' + report['raw']['sha256']
        if run in seen or report.get('capture_run_id') != run:
            raise ValueError('Actual unique independent startup raw captures required')
        seen.add(run)
        if (plan.get('domain') != 'natural' or plan.get('split') != 'training' or not plan.get('jobs')
                or any(j.get('split') != 'training' for j in plan['jobs'])
                or any(r.get('split') != 'training' or r.get('status') != 'completed' for r in raw['request_records'])):
            raise ValueError('Exclusively completed natural training requests required')
        native = native_execution_projection(context)
        full = native['original_full_context_sha256']
        if (native['measurement_domain'] != 'natural' or raw['execution_context_sha256'] != full
                or plan['execution_context_sha256'] != full):
            raise ValueError('Actual native full-context training identity mismatch')
        if projection is None:
            projection = native['projection_sha256']
        if projection != native['projection_sha256']:
            raise ValueError('Different native startup training projections')
        priming = raw['priming_request_records']
        if not priming or any(r.get('status') != 'completed' for r in priming):
            raise ValueError('Actual completed training priming required')
        actual = [{k: r[k] for k in ('model', 'task', 'pool_id')} for r in priming]
        if actual != original.planned_priming(plan['inputs'], context['residents']):
            raise ValueError('Actual training priming differs from declared collector rule')
        current = identity(actual)
        catalog, _ = public_catalog(context, resolver=resolver)
        current_contents = priming_content(plan['inputs'], context['residents'], catalog)
        if priming_id is None:
            priming_id, content_id, contents = current, digest(current_contents), current_contents
        if current != priming_id or digest(current_contents) != content_id:
            raise ValueError('Training priming identities or actual public contents differ')
        observations.append(dict(run_id=run, raw=report['raw'], **original.targets(raw)))
        contexts.append(dict(context=report['context'], native_projection=native))
    if len(observations) < 2:
        raise ValueError('At least two actual independent startup captures required')
    means = {k: statistics.mean(o[k] for o in observations) for k in ('energy_joules', 'peak_memory_mib')}
    for field, value in model['startup_prediction'].items():
        finite_nonnegative(value, field)
    expected = dict(status='FITTED_CONDITIONAL_TRAINING_MEANS', native_projection_sha256=projection,
        priming_configuration_sha256=priming_id, independent_capture_count=len(observations),
        contexts=contexts, observations=observations, startup_prediction=means,
        energy_window='startup_begin_through_active_begin_including_priming',
        peak_window='startup_begin_through_active_begin_owned_sampled_peak',
        energy_includes_priming=True, safety_factor_applied=False, startup_amortization_seconds=3600.)
    if any(model.get(k) != v for k, v in expected.items()):
        raise ValueError('V5 startup model differs from recomputed actual targets, means or identities')
    return model, dict(model=model_ref, recomputed_startup_prediction=means,
        independent_capture_count=len(observations), native_projection_sha256=projection,
        training_full_context_sha256s=sorted({c['native_projection']['original_full_context_sha256'] for c in contexts}),
        priming_content=contents, priming_content_sha256=content_id, evidence_closure=closure)


def validate_target(flowplan_path, candidate_id):
    plan_ref = receipt(flowplan_path)
    plan = read(plan_ref)
    flow.validate_plan(plan)
    if type(plan.get('warmup_seconds')) not in (int, float) or plan['warmup_seconds'] != 0:
        raise ValueError('Startup transfer requires exactly zero warmup seconds')
    candidates = [c for c in plan['candidates'] if c['id'] == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Exact declared target candidate required')
    candidate = candidates[0]
    if candidate['resource_configuration'] != flow.resource_configuration():
        raise ValueError('Only the original shared/no-admission resource configuration is supported')
    if plan['runner_source'] != receipt(flow.__file__):
        raise ValueError('Actual known-input flow runner source differs')
    expected_sources = [receipt(p) for p in flow.collection_source_paths()]
    if plan.get('collector_sources') != expected_sources:
        raise ValueError('Complete actual collector source closure differs')
    for ref in expected_sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Changed actual flow collector source')
    catalog_ref = receipt(flow.BASE / 'catalog.json')
    if plan['catalog'] != catalog_ref:
        raise ValueError('Actual collector catalog path/bytes differ')
    catalog_doc = read(catalog_ref)
    if plan.get('memory_safe_factor') != catalog_doc['safe_factor']:
        raise ValueError('Original catalog memory safety factor required')
    context, profile = read(plan['context']), read(plan['profile_plan'])
    # The collector uses target context to execute and this separate profile's
    # inputs to prime. Preserve both full identities instead of equating them.
    profile_context_ref = receipt(Path(plan['profile_plan']['path']).with_name('context.json'))
    profile_context = read(profile_context_ref)
    for name, ctx in (('target', context), ('profile', profile_context)):
        for source in ctx['execution_sources']:
            if receipt(source['frozen']['path']) != source['frozen'] or receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
                raise ValueError('Changed actual ' + name + ' worker/source closure')
        for stat in read(ctx['large_weight_stats']):
            if stat_receipt(stat['path']) != stat:
                raise ValueError('Changed actual worker weight stat')
    native = native_execution_projection(context)
    profile_native = native_execution_projection(profile_context)
    if (plan['execution_context_sha256'] != native['original_full_context_sha256']
            or candidate['execution_context_sha256'] != native['original_full_context_sha256']
            or profile['execution_context_sha256'] != profile_native['original_full_context_sha256']
            or native['projection_sha256'] != profile_native['projection_sha256']
            or native['measurement_domain'] != 'natural'):
        raise ValueError('Target/profile original full contexts or native projections differ')
    if {a: row['family'] for a, row in plan['aliases'].items()} != RESIDENTS:
        raise ValueError('Actual original four resident owners required')
    if digest({k: v for k, v in profile.items() if k != 'plan_sha256'}) != profile.get('plan_sha256'):
        raise ValueError('Actual priming profile plan hash differs')
    pools, pool_ref = public_catalog(context, require_live=True)
    profile_pools, profile_pool_ref = public_catalog(profile_context, require_live=True)
    if pool_ref['original_path'] != str(flow.BASE / 'runtime-inputs.jsonl') or pools != profile_pools:
        raise ValueError('Actual collector runtime public catalog differs from priming profile')
    contents = priming_content(profile['inputs'], context['residents'], pools)
    return plan, dict(plan=plan_ref, context=plan['context'], profile_plan=plan['profile_plan'],
        profile_context=profile_context_ref, catalog=catalog_ref, runtime_catalog=pool_ref,
        profile_runtime_catalog=profile_pool_ref, runner=plan['runner_source'], collector_sources=expected_sources,
        candidate_id=candidate_id, candidate_sha256=digest(candidate),
        resource_configuration=candidate['resource_configuration'], native_projection=native,
        profile_native_projection=profile_native, actual_priming_content=contents,
        priming_content_sha256=digest(contents), warmup_seconds=0.)


def prediction(model_path_v5, flowplan_path, candidate_id, *, allow_native_priming_transfer_hypothesis=False):
    if not allow_native_priming_transfer_hypothesis:
        raise ValueError('Explicit native projection and exact priming-content transfer hypothesis required')
    model, training = verify_model(model_path_v5)
    plan, target = validate_target(flowplan_path, candidate_id)
    if target['native_projection']['projection_sha256'] != training['native_projection_sha256']:
        raise ValueError('Future flow native projection differs from actual startup training')
    if target['priming_content_sha256'] != training['priming_content_sha256']:
        raise ValueError('Future flow priming actual public contents differ from startup training')
    if plan['startup_amortization_seconds'] != model['startup_amortization_seconds']:
        raise ValueError('Original startup amortization must match trained scoring contract')
    implementations = [receipt(__file__), receipt(original.__file__), receipt(flow.__file__),
        receipt(Path(__file__).with_name('historical_evidence_v4.py')),
        receipt(Path(__file__).with_name('native_execution_projection_v5.py')),
        receipt(Path(__file__).with_name('flow_validation.py')),
        receipt(Path(__file__).with_name('generation_work_v4.py'))]
    return dict(schema=SCHEMA, basis='frozen_training_prediction',
        **training['recomputed_startup_prediction'],
        evidence=[training['model'], target['plan'], target['context'], target['profile_plan'],
                  target['profile_context'], target['catalog'], *target['collector_sources'], *implementations],
        execution_context_sha256=target['native_projection']['original_full_context_sha256'],
        native_projection_sha256=training['native_projection_sha256'],
        training_full_context_sha256s=training['training_full_context_sha256s'],
        profile_execution_context_sha256=target['profile_native_projection']['original_full_context_sha256'],
        priming_configuration_sha256=model['priming_configuration_sha256'],
        priming_content_sha256=training['priming_content_sha256'],
        candidate_id=candidate_id, flow_plan=target['plan'], target_contract=target,
        training_reverification=training, energy_includes_priming=True, energy_includes_warmup=False,
        warmup_seconds=0., safety_factor_applied=False, startup_amortization_seconds=3600.,
        collector_invariance_is_hypothesis=True, exact_priming_content_transfer_is_hypothesis=True,
        full_context_equality_claimed=False, physical_accuracy_validated=False, capacity_bound=False,
        continuous_peak_known=False, GPU_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--flow-plan', required=True)
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-native-priming-transfer-hypothesis', action='store_true', required=True)
    args = parser.parse_args()
    predictions = {candidate: prediction(args.model, args.flow_plan, candidate,
        allow_native_priming_transfer_hypothesis=args.allow_native_priming_transfer_hypothesis) for candidate in args.candidate}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'predictions.json', dict(schema=SCHEMA, predictions=predictions, GPU_started=False))
    print(json.dumps(dict(status='SOURCE_REVERIFIED_FLOW_BOUND_PREDICTIONS', candidates=list(predictions))))


if __name__ == '__main__':
    main()
