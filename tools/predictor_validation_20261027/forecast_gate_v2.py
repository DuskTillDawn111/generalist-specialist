"""Seal actual FullHorizonModel forecasts before independent GPU collection.

Frozen model bundles contain training-derived parameters. Separate frozen public
input files contain only the declared workload. No observed validation output
is an input to this CPU prediction command.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.__main__ import forecast_request
from gsdeploy.forecasting.prediction_gate import _verify_protocol, receipt, seal_predictions
from gsdeploy.forecasting.workload import number

BUNDLE_SCHEMA = 'gsdeploy.forecast_gate_model_bundle.v2'
INPUT_SCHEMA = 'gsdeploy.forecast_gate_public_inputs.v2'
PUBLIC_FIELDS = {'phase_start_ns', 'now_ns', 'deadline_ns', 'drain_deadline_ns',
                 'future_workload', 'queues'}
MODEL_FIELDS_FORBIDDEN = PUBLIC_FIELDS | {'schema', 'execution_context_sha256', 'completed',
    'handoff_state', 'visible_host_state', 'admission_state', 'observed_last_arrivals_ns',
    'observed_peak_memory_mib', 'observed_board_energy_joules'}


def prediction_source_paths():
    """Conservative source closure, including scorer, provider and simulator."""
    root = Path(__file__).resolve().parents[2]
    return [Path(__file__).resolve(), *sorted((root/'gsdeploy').rglob('*.py'))]


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _public_cohort(inputs):
    if inputs.get('schema') != INPUT_SCHEMA or set(inputs) != {'schema', 'forecast_inputs'}:
        raise ValueError('Only frozen public forecast inputs accepted')
    public = inputs['forecast_inputs']
    if set(public) != PUBLIC_FIELDS:
        raise ValueError('Explicit prospective whole-run public fields required')
    if (public['now_ns'] != public['phase_start_ns']
            or public['future_workload'].get('process') != 'declared_schedule'):
        raise ValueError('Whole-run gate requires a declared public schedule at phase start')
    start, deadline = public['phase_start_ns'], public['deadline_ns']
    if any(type(v) is not int for v in (start, deadline)) or not 0 <= start < deadline:
        raise ValueError('Valid public measurement window required')
    result = {}

    def add(row):
        if (row['request_id'] in result or type(row['arrival_ns']) is not int
                or not start <= row['arrival_ns'] < deadline):
            raise ValueError('Duplicate or outside-window public request')
        result[row['request_id']] = row

    for alias, rows in public['queues'].items():
        for row in rows:
            if set(row) != {'request_id', 'task', 'pool_id', 'arrival_ns', 'alias'} or row['alias'] != alias:
                raise ValueError('Initial queue accepts public request identity/arrival/route only')
            if row['arrival_ns'] != start:
                raise ValueError('Only already-arrived phase-start requests belong in initial queues')
            add(copy.deepcopy(row))
    for row in public['future_workload']['declared_arrivals']:
        if set(row) != {'request_id', 'task', 'pool_id', 'offset_ns'}:
            raise ValueError('Declared schedule accepts public identity/time only')
        if type(row['offset_ns']) is not int or row['offset_ns'] <= 0:
            raise ValueError('Phase-start requests must be explicit initial queues')
        add({k: row[k] for k in ('request_id', 'task', 'pool_id')} | {'arrival_ns': start + row['offset_ns']})
    return public, result


def _startup_value(bundle, key):
    evidence = bundle.get('startup_prediction', {})
    if not evidence.get('evidence') or evidence.get('basis') != 'frozen_training_prediction':
        return None
    value = evidence.get(key)
    if value is not None:
        number(value, 'training startup '+key)
    return value


def _to_cell(result, cohort, candidate, run_id, protocol, bundle, phase_start_ns):
    rows, finishes = [], result.get('forecast', {}).get('finishes', {})
    for rid, public in cohort.items():
        finish = finishes.get(rid, {})
        handler_start, handler_end, receipt_ns = (finish.get(k) for k in ('start_ns', 'handler_end_ns', 'finish_ns'))
        if all(type(v) is int for v in (handler_start, handler_end, receipt_ns)):
            if not public['arrival_ns'] <= handler_start <= handler_end <= receipt_ns:
                raise ValueError('Predicted handler/receipt chronology invalid')
            service, queue, e2e = ((handler_end-handler_start)/1e6,
                                  (handler_start-public['arrival_ns'])/1e6,
                                  (receipt_ns-public['arrival_ns'])/1e6)
        else:
            service = queue = e2e = None
        rows.append({k: public[k] for k in ('request_id', 'task', 'pool_id')} |
                    dict(service_ms=service, queue_ms=queue, e2e_ms=e2e))
    terms = copy.deepcopy(result.get('metrics', {}).get('terms'))
    if terms is not None:
        startup_j = _startup_value(bundle, 'energy_joules')
        amortization = bundle.get('startup_amortization_seconds')
        if startup_j is None or amortization is None:
            terms['E'] = None
        elif terms['E'] is not None:
            number(amortization, 'startup amortization seconds', positive=True)
            # Original E = measurement J/(W*H) + startup J/(W*A).
            # Measurement service model integrates only through the deadline.
            watts = bundle['forecast_defaults']['energy_reference_watts']
            number(watts, 'energy reference watts', positive=True)
            terms['E'] += startup_j/(watts*amortization)
    samples = []
    for offset in protocol['queue_sample_offsets_ns']:
        clock, depth, known = phase_start_ns+offset, 0, True
        for rid, public in cohort.items():
            if public['arrival_ns'] > clock:
                continue
            handler_start = finishes.get(rid, {}).get('start_ns')
            if type(handler_start) is not int:
                known = False
            elif clock < handler_start:
                depth += 1
        samples.append(dict(offset_ns=offset, depth=depth if known else None))
    return dict(run_id=run_id, candidate_id=candidate['id'],
                execution_context_sha256=candidate['execution_context_sha256'],
                requests=rows, terms=terms, queue_depth_samples=samples)


def run_and_seal(protocol_path, jobs, output_dir, *, allow_cpu_fixture=False,
                 allow_analytical_hypothesis=False, allow_profiled_baselines=False):
    """Execute all candidate/run forecasts and seal their derived gate cells.

    jobs: [{run_id, candidate_id, public_inputs_path}], with exact Cartesian
    coverage. Each public input file and this producer must be protocol sources.
    """
    protocol = _read(protocol_path)
    _verify_protocol(protocol)
    sources = protocol['sources']
    if any(receipt(path) not in sources for path in prediction_source_paths()):
        raise ValueError('Forecast producer and complete package source closure must be frozen before prediction')
    candidates = {c['id']: c for c in protocol['candidates']}
    expected = {(r, c) for r in protocol['validation_domain'] for c in candidates}
    keyed = {}
    for job in jobs:
        if set(job) != {'run_id', 'candidate_id', 'public_inputs_path'}:
            raise ValueError('Explicit run/candidate/public input job required')
        key = job['run_id'], job['candidate_id']
        if key not in expected or key in keyed:
            raise ValueError('Duplicate/foreign forecast job')
        if receipt(job['public_inputs_path']) not in sources:
            raise ValueError('Public workload file must be in frozen protocol sources')
        keyed[key] = job
    if set(keyed) != expected:
        raise ValueError('Every predeclared run/candidate forecast is required')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells, audits, run_public, run_objective = [], [], {}, {}
    for index, (key, job) in enumerate(sorted(keyed.items())):
        run_id, cid = key
        candidate = candidates[cid]
        model_ref = protocol['models'][candidate['model_id']]
        bundle = _read(model_ref['path'])
        if (bundle.get('schema') != BUNDLE_SCHEMA
                or bundle.get('execution_context_sha256') != candidate['execution_context_sha256']
                or bundle.get('kind') not in ('trained_model', 'cpu_fixture')):
            raise ValueError('Explicit bound forecast model bundle required')
        if bundle['kind'] == 'cpu_fixture' and not allow_cpu_fixture:
            raise ValueError('CPU fixture cannot enter a physical prediction seal')
        defaults = bundle['forecast_defaults']
        if MODEL_FIELDS_FORBIDDEN.intersection(defaults):
            raise ValueError('Model defaults cannot supply workload/history/observed validation fields')
        if defaults.get('execution_contract') != 'strict_lifecycle_v2':
            raise ValueError('New prediction gate requires actual lifecycle model')
        objective = {k: defaults[k] for k in ('latency_reference_ms', 'memory_budget_mib',
                     'energy_reference_watts', 'memory_safe_factor')}
        objective['startup_amortization_seconds'] = bundle.get('startup_amortization_seconds')
        if run_id in run_objective and objective != run_objective[run_id]:
            raise ValueError('Paired candidates must share original objective normalizers')
        run_objective[run_id] = objective
        if bundle['kind'] != 'cpu_fixture' and defaults.get('provider_kind') != 'kernel_resource_v2':
            raise ValueError('Paper predictor gate requires the kernel provider, not an empirical context table')
        public, cohort = _public_cohort(_read(job['public_inputs_path']))
        domain = {r['request_id']: r for r in protocol['validation_domain'][run_id]}
        actual = {rid: {k: row[k] for k in ('request_id', 'pool_id', 'task')} for rid, row in cohort.items()}
        if actual != domain:
            raise ValueError('Public forecast cohort differs from the entire frozen validation domain')
        # Candidates may differ in route/owner assignment only. The arrivals,
        # task weights, timing windows and input identities must be identical.
        shared = {k: public[k] for k in ('phase_start_ns', 'now_ns', 'deadline_ns', 'drain_deadline_ns', 'future_workload')}
        shared['initial_requests'] = sorted(({k: r[k] for k in ('request_id', 'pool_id', 'task', 'arrival_ns')}
                                            for rows in public['queues'].values() for r in rows), key=lambda r:r['request_id'])
        if run_id in run_public and shared != run_public[run_id]:
            raise ValueError('Candidate workload/arrival cohort differs within a paired run')
        run_public[run_id] = shared
        request = copy.deepcopy(defaults) | copy.deepcopy(public)
        request.update(schema='gsdeploy.full_horizon_forecast_request.v1',
                       execution_context_sha256=candidate['execution_context_sha256'],
                       completed=[], observed_board_energy_joules=0.,
                       observed_peak_memory_mib=_startup_value(bundle, 'peak_memory_mib'))
        try:
            result = forecast_request(request, allow_cpu_fixture=allow_cpu_fixture,
                                      allow_analytical_hypothesis=allow_analytical_hypothesis,
                                      allow_profiled_baselines=allow_profiled_baselines)
            if result['forecast']['cohort'].keys() != cohort.keys():
                raise ValueError('Simulator did not retain the complete public cohort')
        except UnknownEvidence as exc:
            result = dict(status='UNKNOWN', reason=str(exc), GPU_started=False)
        result_path = output/f'forecast-{index:04d}.json'
        _write(result_path, result)
        cells.append(_to_cell(result, cohort, candidate, run_id, protocol, bundle, public['phase_start_ns']))
        audits.append(dict(run_id=run_id, candidate_id=cid, model=model_ref,
                           public_inputs=receipt(job['public_inputs_path']), result=receipt(result_path),
                           startup_scope='training-predicted startup peak and amortized board energy; no validation observation',
                           producer='actual forecast_request -> FullHorizonModel -> score_forecast'))
    sealed = seal_predictions(protocol, cells, created_unix_ns=time.time_ns())
    _write(output/'predictions.json', sealed)
    report = dict(schema='gsdeploy.forecast_gate_production_receipt.v2', protocol=receipt(protocol_path),
                  predictions=receipt(output/'predictions.json'), producer=receipt(__file__), jobs=audits,
                  GPU_started=False, allow_cpu_fixture=allow_cpu_fixture,
                  allow_analytical_hypothesis=allow_analytical_hypothesis,
                  allow_profiled_baselines=allow_profiled_baselines,
                  accuracy_established=False, physical_gain_certified=False)
    _write(output/'production-receipt.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--jobs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-cpu-fixture', action='store_true')
    parser.add_argument('--allow-analytical-hypothesis', action='store_true')
    parser.add_argument('--allow-profiled-baselines', action='store_true')
    args = parser.parse_args(argv)
    result = run_and_seal(args.protocol, _read(args.jobs), args.output,
        allow_cpu_fixture=args.allow_cpu_fixture, allow_analytical_hypothesis=args.allow_analytical_hypothesis,
        allow_profiled_baselines=args.allow_profiled_baselines)
    print(json.dumps(dict(predictions=result['predictions'], GPU_started=False)))


if __name__ == '__main__':
    main()
