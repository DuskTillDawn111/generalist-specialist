"""Predict original-score premeasurement energy, including actual priming.

The v4 loading-only energy target is retained. This separate model integrates
startup_begin through active_begin, as flow_metrics does, and permits collector
transfer only under an explicit matching native projection/priming hypothesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from gsdeploy.resources import identity
from .empirical_resources import integrate_samples
from .historical_evidence_v4 import resolve_closure
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .small_training_domain_bridge_review_v4 import load_report
from .startup_prediction_fit_v4 import targets as loading_targets

SCHEMA = 'gsdeploy.premeasurement_startup_prediction_hypothesis.v5'


def targets(raw):
    result = loading_targets(raw)
    start, end = result['peak_window_ns']['start'], result['peak_window_ns']['end']
    result.update(loading_only_energy_joules=result['energy_joules'],
                  energy_joules=integrate_samples(raw['board_samples'], start, end) / 1e9,
                  energy_window_ns=dict(start=start, end=end),
                  premeasurement_seconds=(end-start) / 1e9,
                  energy_includes_priming=True, active_or_drain_energy_included=False)
    return result


def planned_priming(inputs, residents):
    result = []
    for alias, family in residents.items():
        tasks = ['text', 'image', 'asr'] if family == 'generalist_2' else ['ocr' if family == 'ocr' else 'text']
        for task in tasks:
            selected = sorted((r for r in inputs if r['task'] == task), key=lambda r:r['input_size'], reverse=True)[:3]
            if not selected:
                raise ValueError('Every resident priming task requires declared inputs')
            result.extend(dict(model=alias, task=task, pool_id=r['pool_id']) for r in selected)
    return result


def fit(review_paths, output, *, historical_resolution_path=None):
    observations, contexts, sources, seen = [], [], [], set()
    projection, priming_id = None, None
    for path in review_paths:
        report, plan, raw = load_report(path, 'natural')
        source = report['raw']
        if source['sha256'] in seen:
            raise ValueError('Repeated raw is not an independent startup capture')
        seen.add(source['sha256'])
        if (plan.get('split') != 'training' or any(j.get('split') != 'training' for j in plan['jobs'])
                or any(r.get('split') != 'training' or r.get('status') != 'completed' for r in raw['request_records'])):
            raise ValueError('Completed exclusively training startup capture required')
        context = json.loads(Path(report['context']['path']).read_text())
        native = native_execution_projection(context)
        if native['measurement_domain'] != 'natural':
            raise ValueError('Natural premeasurement training required')
        if projection is None:
            projection = native['projection_sha256']
        if projection != native['projection_sha256'] or raw['execution_context_sha256'] != native['original_full_context_sha256']:
            raise ValueError('Identical native projection and actual original full context required')
        priming = raw['priming_request_records']
        if not priming or any(r.get('status') != 'completed' for r in priming):
            raise ValueError('Complete actual priming required')
        actual = [{k:r[k] for k in ('model','task','pool_id')} for r in priming]
        if actual != planned_priming(plan['inputs'], context['residents']):
            raise ValueError('Actual priming differs from future collector planning rule')
        current = identity(actual)
        if priming_id is None:
            priming_id = current
        if current != priming_id:
            raise ValueError('Identical actual priming configuration required')
        observations.append(dict(run_id='raw-sha256:' + source['sha256'], raw=source, **targets(raw)))
        contexts.append(dict(context=report['context'], native_projection=native))
        sources.append(receipt(path))
    if len(observations) < 2:
        raise ValueError('At least two independent actual startup captures required')
    closure = resolve_closure([r['path'] for r in sources], historical_resolution_path)
    model = dict(schema=SCHEMA, status='FITTED_CONDITIONAL_TRAINING_MEANS',
        native_projection_sha256=projection, priming_configuration_sha256=priming_id,
        contexts=contexts, observations=observations, independent_capture_count=len(observations),
        startup_prediction={k:statistics.mean(r[k] for r in observations) for k in ('energy_joules','peak_memory_mib')},
        energy_window='startup_begin_through_active_begin_including_priming',
        peak_window='startup_begin_through_active_begin_owned_sampled_peak',
        energy_includes_priming=True, safety_factor_applied=False, startup_amortization_seconds=3600.,
        sources=sources, evidence_closure=closure,
        historical_resolution=receipt(historical_resolution_path) if historical_resolution_path else None,
        implementation_sources=[receipt(__file__), receipt(Path(__file__).with_name('startup_prediction_fit_v4.py')),
            receipt(Path(__file__).with_name('empirical_resources.py')), receipt(Path(__file__).with_name('native_execution_projection_v5.py')),
            receipt(Path(__file__).with_name('historical_evidence_v4.py'))],
        scope='Conditional training means for matching native execution and exact priming; not a capacity/confidence bound',
        collector_invariance_is_hypothesis=True, physical_accuracy_validated=False, GPU_started=False)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'model.json', model)
    return model


def prediction(model_path, target_context_path, target_inputs, *, allow_native_projection_transfer=False):
    if not allow_native_projection_transfer:
        raise ValueError('Explicit startup native projection/priming transfer hypothesis required')
    ref = receipt(model_path)
    model = json.loads(Path(model_path).read_text())
    if model.get('schema') != SCHEMA or model.get('independent_capture_count', 0) < 2:
        raise ValueError('Actual independent startup training model required')
    for source in model['implementation_sources']:
        if receipt(source['path']) != source:
            raise ValueError('Changed startup prediction implementation')
    resolution = model['historical_resolution']
    if resolution is not None and receipt(resolution['path']) != resolution:
        raise ValueError('Changed startup historical resolution')
    closure = resolve_closure([r['path'] for r in model['sources']], resolution['path'] if resolution else None)
    if closure != model['evidence_closure']:
        raise ValueError('Changed startup training evidence closure')
    context_ref = receipt(target_context_path)
    context = json.loads(Path(target_context_path).read_text())
    native = native_execution_projection(context)
    if native['projection_sha256'] != model['native_projection_sha256']:
        raise ValueError('Startup target native execution differs')
    priming = planned_priming(target_inputs, context['residents'])
    if identity(priming) != model['priming_configuration_sha256']:
        raise ValueError('Startup target priming differs from actual training')
    return dict(basis='frozen_training_prediction', evidence=[ref, context_ref],
        **model['startup_prediction'], execution_context_sha256=native['original_full_context_sha256'],
        native_projection_sha256=native['projection_sha256'], priming_configuration_sha256=identity(priming),
        energy_includes_priming=True, safety_factor_applied=False, startup_amortization_seconds=3600.,
        collector_invariance_is_hypothesis=True, physical_accuracy_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', action='append', required=True)
    parser.add_argument('--historical-resolution')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model = fit(args.review, args.output, historical_resolution_path=args.historical_resolution)
    print(json.dumps(dict(captures=model['independent_capture_count'], startup_prediction=model['startup_prediction'],
                          energy_includes_priming=True)))


if __name__ == '__main__':
    main()
