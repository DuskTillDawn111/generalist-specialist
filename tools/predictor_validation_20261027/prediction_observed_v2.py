"""CPU-only native-flow observations and prospective collection checks for v2.

No fitting, GPU execution, or automatic qualification occurs here. The current
campaign's six-term contract is inherited, including its owned sampled peak.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

from gsdeploy.forecasting.prediction_gate import (
    OBSERVED_SCHEMA, PREDICTION_SCHEMA, _cells, _verify_protocol, receipt,
)
from gsdeploy.resources import identity
from .flow_metrics import score_flow
from .flow_validation import validate_plan

PRECOLLECTION_SCHEMA = 'gsdeploy.prediction_precollection_check.v2'
ADAPTER_SCHEMA = 'gsdeploy.prediction_observation_adapter.v2'
METRIC_CONTRACT = 'inherited_campaign_startup_through_drain_owned_peak_v2'
QUEUE_BOUNDARIES = {'waiting_until_handler_start'}


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _verify(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Source receipt changed or malformed')


def _ns(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _bindings(protocol_path, predictions_path, plan_path, run_id):
    protocol, predictions, plan = map(_read, (protocol_path, predictions_path, plan_path))
    _verify_protocol(protocol)
    if (predictions.get('schema') != PREDICTION_SCHEMA
            or predictions.get('protocol_sha256') != protocol['protocol_sha256']
            or predictions.get('prediction_sha256') != identity({k: v for k, v in predictions.items() if k != 'prediction_sha256'})
            or not _ns(predictions.get('created_unix_ns'))
            or predictions['created_unix_ns'] < protocol['created_unix_ns']):
        raise ValueError('Frozen prediction/protocol identity or chronology changed')
    _cells(predictions['cells'], protocol)
    validate_plan(plan)
    if receipt(plan_path) not in protocol['sources']:
        raise ValueError('Flow plan must be bound before prediction seal')
    if run_id not in protocol['validation_domain']:
        raise ValueError('Run is outside frozen validation domain')
    domain = sorted(protocol['validation_domain'][run_id], key=lambda r: r['request_id'])
    offered = sorted(({k: r[k] for k in ('request_id', 'pool_id', 'task')} for r in plan['trace']), key=lambda r: r['request_id'])
    if (domain != offered or plan['weights'] != protocol['weights']
            or plan['baseline_candidate_id'] != protocol['baseline_candidate_id']):
        raise ValueError('Frozen input cohort or six weights differ from flow plan')
    if list(plan.get('queue_sample_offsets_ns', [])) != protocol['queue_sample_offsets_ns']:
        raise ValueError('Queue grid differs from frozen protocol')
    if protocol['queue_sample_offsets_ns'] and plan.get('queue_depth_boundary') not in QUEUE_BOUNDARIES:
        raise ValueError('Prospective queue depth boundary required')
    for ref in plan.get('execution_sources', []):
        _verify(ref)
    for key in ('context', 'runner_source', 'profile_plan', 'catalog'):
        if key in plan:
            _verify(plan[key])
    if 'context' in plan:
        context = _read(plan['context']['path'])
        for source in context.get('execution_sources', []):
            _verify(source['frozen'])
            if receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
                raise ValueError('Frozen runtime source/input changed')
    gate_candidates = {c['id']: c for c in protocol['candidates']}
    for candidate in plan['candidates']:
        if (candidate['id'] not in gate_candidates or candidate['execution_context_sha256']
                != gate_candidates[candidate['id']]['execution_context_sha256']):
            raise ValueError('Candidate differs from frozen execution context')
    if {c['id'] for c in plan['candidates']} != set(gate_candidates):
        raise ValueError('Flow/gate candidate domains differ')
    return protocol, predictions, plan


def verify_before_collection(protocol_path, predictions_path, plan_path, *, run_id,
                             candidate_id, output_root):
    """Check and save an exclusive receipt before a caller starts any collection.

    Caller must then stamp the real collection start in raw, bind this receipt,
    and retain the original raw. Local Unix chronology is not an external
    timestamp authority. This function never starts a worker.
    """
    protocol, predictions, plan = _bindings(protocol_path, predictions_path, plan_path, run_id)
    if candidate_id not in {c['id'] for c in plan['candidates']}:
        raise ValueError('Unknown collection candidate')
    if not any(c['run_id'] == run_id and c['candidate_id'] == candidate_id for c in predictions['cells']):
        raise ValueError('Selected candidate must have a presealed prediction cell')
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Collection output must be new or empty: ' + str(output))
    now = time.time_ns()
    if now <= predictions['created_unix_ns']:
        raise ValueError('Precollection check must follow prediction seal')
    check_path = output.parent/(output.name+'.precollection.json')
    result = dict(schema=PRECOLLECTION_SCHEMA, checked_unix_ns=now, receipt_path=str(check_path),
        run_id=run_id, candidate_id=candidate_id, output_root=str(output),
        protocol=receipt(protocol_path), predictions=receipt(predictions_path), plan=receipt(plan_path),
        adapter_source=receipt(__file__), scorer_source=receipt(Path(__file__).with_name('flow_metrics.py')),
        protocol_sha256=protocol['protocol_sha256'], prediction_sha256=predictions['prediction_sha256'],
        GPU_started=False, claim='PRECOLLECTION_CHECK_NOT_PROOF_OF_MODEL_FITTING_OR_EXTERNAL_TIME')
    result['check_sha256'] = identity(result)
    # Leave the collection output absent so an exclusive native collector can
    # create it itself. The sibling check file is also exclusive/no retry.
    output.parent.mkdir(parents=True, exist_ok=True)
    _save(check_path, result)
    return result


def _chronology(raw, raw_path, precollection_path, protocol_path, predictions_path, plan_path, run_id, cid):
    """Missing chronology is unknown; contradicting existing evidence is refused."""
    if not precollection_path or not raw.get('precollection'):
        return None, ['MISSING_PRECOLLECTION_RECEIPT']
    pre = _read(precollection_path)
    if (pre.get('schema') != PRECOLLECTION_SCHEMA or pre.get('check_sha256')
            != identity({k: v for k, v in pre.items() if k != 'check_sha256'})):
        raise ValueError('Precollection check changed')
    for key, path in (('protocol', protocol_path), ('predictions', predictions_path), ('plan', plan_path)):
        if pre.get(key) != receipt(path):
            raise ValueError('Precollection evidence binding differs')
    for key in ('adapter_source', 'scorer_source'):
        _verify(pre[key])
    if (not _ns(pre.get('checked_unix_ns'))
            or pre['checked_unix_ns'] <= _read(predictions_path)['created_unix_ns']):
        raise ValueError('Precollection check does not follow prediction seal')
    if (pre.get('run_id'), pre.get('candidate_id')) != (run_id, cid):
        raise ValueError('Precollection run/candidate mismatch')
    _verify(raw['precollection'])
    if raw['precollection'] != receipt(precollection_path):
        raise ValueError('Raw does not bind its actual precollection check')
    # Annotated quality files may be outside the collection directory, but must
    # retain and bind the native raw that was saved there.
    origin = Path(raw_path).resolve()
    if raw.get('native_raw'):
        _verify(raw['native_raw'])
        native = _read(raw['native_raw']['path'])
        if native != {k: v for k, v in raw.items() if k not in {'quality_evidence', 'native_raw'}}:
            raise ValueError('Annotated observations changed original native raw')
        origin = Path(raw['native_raw']['path']).resolve()
    if origin.parent != Path(pre['output_root']).resolve():
        raise ValueError('Observed raw is outside checked collection directory')
    started = raw.get('collection_started_unix_ns')
    if not _ns(started):
        return None, ['MISSING_REAL_COLLECTION_UNIX_START']
    if started < pre['checked_unix_ns']:
        raise ValueError('Collection started before prospective check')
    return started, []


def _timings(raw, plan, candidate):
    start, end, observed_end = (raw.get(k) for k in ('phase_start_ns', 'deadline_ns', 'observation_end_ns'))
    startup, ready = raw.get('startup_begin_ns'), raw.get('startup_end_ns')
    if not all(_ns(v) for v in (start, end, observed_end, startup, ready)):
        raise ValueError('Native startup/measurement/drain boundaries required')
    if not startup <= ready <= start < end <= observed_end or end != start + round(plan['window_seconds']*1e9):
        raise ValueError('Native window chronology differs from plan')
    # Polling/shutdown observation can extend past the drain. Completion
    # eligibility below still uses the exact predeclared cutoff, with no slack.
    actual = {r['request_id']: r for r in raw['request_records']}
    if len(actual) != len(raw['request_records']) or set(actual) != {r['request_id'] for r in plan['trace']}:
        raise ValueError('Missing/duplicate/foreign offered request identity')
    results, reasons = [], []
    arrivals, starts, receipts = {}, {}, {}
    for offered in plan['trace']:
        rid, alias = offered['request_id'], candidate['routes'][offered['request_id']]
        row = actual[rid]
        arrival = start + offered['offset_ns']
        if (row.get('pool_id'), row.get('task'), row.get('model'), row.get('scheduled_time_ns')) != (offered['pool_id'], offered['task'], alias, arrival):
            raise ValueError('Actual route/input/scheduled arrival differs from plan')
        arrivals[rid] = arrival
        timing = {k: offered[k] for k in ('request_id', 'pool_id', 'task')}
        timing.update(service_ms=None, queue_ms=None, e2e_ms=None)
        response = row.get('worker_response', {})
        handler_start, handler_end = response.get('start_perf_ns'), response.get('end_perf_ns')
        events = [e for e in row.get('execution_events', []) if 'event' not in e]
        ready_pid = raw.get('ready', {}).get(alias, {}).get('pid')
        if len(events) > 1:
            raise ValueError('Duplicate START boundary')
        if events and ready_pid is not None:
            event = events[0]
            if event.get('pid') != ready_pid or event.get('request_id') != rid:
                raise ValueError('START owner/request mismatch')
            stamp, received = event.get('perf_ns'), event.get('received_time_ns')
            if not _ns(stamp) or not _ns(received) or not arrival <= stamp <= received:
                raise ValueError('Invalid START clock/order')
            if handler_start is not None and handler_start != stamp:
                raise ValueError('START and response handler boundary disagree')
            if row.get('observed_worker_start_time_ns') != stamp:
                raise ValueError('Stored worker start differs from explicit START')
            starts[rid] = stamp
            timing['queue_ms'] = (stamp-arrival)/1e6
        else:
            starts[rid] = None
            reasons.append('MISSING_ACTUAL_START:'+rid)
        receipt_ns = row.get('response_reader_event_enqueue_time_ns')
        if row.get('status') == 'completed' and receipt_ns is not None:
            if not _ns(receipt_ns) or row.get('completion_time_ns') != receipt_ns:
                raise ValueError('Parent scoring boundary differs from actual reader receipt')
            if not arrival <= receipt_ns <= min(observed_end, end + round(plan['drain_seconds']*1e9)):
                raise ValueError('Receipt outside offered/bounded drain interval')
            receipts[rid] = receipt_ns
            timing['e2e_ms'] = (receipt_ns-arrival)/1e6
        else:
            receipts[rid] = None
            reasons.append('MISSING_ACTUAL_PARENT_RECEIPT:'+rid)
        if starts[rid] is not None and handler_end is not None:
            if not _ns(handler_end) or not starts[rid] <= handler_end <= observed_end:
                raise ValueError('Handler end order invalid')
            if receipts[rid] is not None and handler_end > receipts[rid]:
                raise ValueError('Handler end follows parent receipt')
            timing['service_ms'] = (handler_end-starts[rid])/1e6
        else:
            reasons.append('MISSING_ACTUAL_HANDLER_INTERVAL:'+rid)
        results.append(timing)
    return results, reasons, arrivals, starts, receipts


def _resources(raw, plan):
    """Verify owned PID arithmetic before allowing score_flow to consume it."""
    reasons, samples = [], raw.get('board_samples', [])
    pids = {a: r.get('pid') for a, r in raw.get('ready', {}).items()}
    if (set(pids) != set(plan['aliases']) or any(type(v) is not int or v <= 0 for v in pids.values())
            or len(set(pids.values())) != len(pids)):
        return False, False, ['UNKNOWN_DISTINCT_OWNED_READY_PIDS']
    if set(raw.get('resident_aliases', [])) != set(plan['aliases']):
        return False, False, ['UNKNOWN_COMPLETE_RESIDENT_ALIAS_SET']
    times = [s.get('time_ns') for s in samples]
    if any(not _ns(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Telemetry timestamps must strictly increase')
    coverage = bool(times and times[0] <= raw['startup_begin_ns'] and times[-1] >= raw['observation_end_ns'])
    if not coverage:
        reasons.append('MISSING_STARTUP_THROUGH_DRAIN_TELEMETRY_COVERAGE')
    memory_ok, power_ok = coverage, coverage
    for sample in samples:
        if not _number(sample.get('watts')):
            power_ok = False
            reasons.append('MISSING_OR_INVALID_BOARD_POWER')
        values = sample.get('processes_mib')
        if not isinstance(values, dict) or any(not _number(v) for v in values.values()):
            memory_ok = False
            reasons.append('MISSING_OWNED_PID_MEMORY_VALUES')
            continue
        try:
            values = {int(pid): v for pid, v in values.items()}
        except (ValueError, TypeError):
            raise ValueError('Invalid NVML process identity') from None
        present = {a for a, pid in pids.items() if pid in values}
        foreign = set(values)-set(pids.values())
        if foreign or sample.get('foreign_pids'):
            memory_ok = power_ok = False
            reasons.append('FOREIGN_GPU_PROCESS')
        if set(sample.get('owned_aliases', [])) != present:
            raise ValueError('Owned alias list differs from actual NVML PIDs')
        if sample['time_ns'] >= raw['startup_end_ns'] and present != set(pids):
            memory_ok = False
            reasons.append('MISSING_RESIDENT_AFTER_READY')
        owned = sum(values[pid] for pid in pids.values() if pid in values)
        if not _number(sample.get('owned_used_mib')) or abs(sample['owned_used_mib']-owned) > 1e-6:
            raise ValueError('Owned memory sum differs from actual owned PID values')
    return memory_ok, power_ok, reasons


def build_observed_run(protocol_path, predictions_path, plan_path, *, run_id,
                       collections, output_dir, allow_cpu_fixture=False):
    """Save gate-compatible measurements and a separate provenance audit.

    Missing timings/quality/resources stay null. Missing collection chronology
    prevents emission of a gate input altogether. Contradictory identities and
    timestamps are errors. Smoke/phase profiling can never enter this adapter.
    """
    protocol, predictions, plan = _bindings(protocol_path, predictions_path, plan_path, run_id)
    candidates = {c['id']: c for c in plan['candidates']}
    cells, audits, begun, kinds, seen = [], [], [], set(), set()
    for collection in collections:
        cid, raw_path = collection['candidate_id'], collection['raw_path']
        if cid not in candidates or cid in seen:
            raise ValueError('Duplicate/foreign observed candidate')
        seen.add(cid)
        raw = _read(raw_path)
        if raw.get('schema') != 'gsdeploy.native_flow_collection.v1' or raw.get('split') != 'validation':
            raise ValueError('Only independent validation native full-flow raw; no smoke/training/phase profiles')
        kind = raw.get('kind')
        if kind not in {'measured', 'cpu_fixture'} or kind == 'cpu_fixture' and not allow_cpu_fixture:
            raise ValueError('CPU fixtures require explicit opt-in; they are not physical evidence')
        if kind == 'measured' and raw.get('backend') != 'native':
            raise ValueError('Measured observations require native execution')
        if 'profile' in raw.get('performance_scope', '').lower() and raw['performance_scope'] != 'unprofiled':
            raise ValueError('Profiler-perturbed flow is not natural latency validation')
        kinds.add(kind)
        candidate = candidates[cid]
        if raw.get('candidate_id') != cid or raw.get('execution_context_sha256') != candidate['execution_context_sha256']:
            raise ValueError('Raw candidate/execution identity mismatch')
        if raw.get('plan') != receipt(plan_path):
            raise ValueError('Raw must bind frozen complete flow plan')
        _verify(raw['plan'])
        if raw.get('effective_objective') != {k: plan[k] for k in ('workload_rps', 'latency_reference_ms', 'weights',
                'memory_safe_factor', 'memory_budget_mib', 'energy_reference_watts', 'startup_amortization_seconds')}:
            raise ValueError('Actual objective differs from frozen six-term contract')
        if raw.get('resource_configuration') != candidate.get('resource_configuration'):
            raise ValueError('Actual resource configuration differs from candidate')
        if 'context' in plan and raw.get('context') != plan['context']:
            raise ValueError('Actual context receipt differs from frozen plan')
        for key in ('cohort_sha256', 'trace_sha256'):
            if key in raw and raw[key] != identity(plan['trace']):
                raise ValueError('Raw arrival cohort hash differs from frozen plan')
        started, reasons = _chronology(raw, raw_path, collection.get('precollection_path'),
            protocol_path, predictions_path, plan_path, run_id, cid)
        if started is not None:
            begun.append(started)
        timings, timing_reasons, arrivals, starts, receipts = _timings(raw, plan, candidate)
        reasons.extend(timing_reasons)
        memory_ok, power_ok, resource_reasons = _resources(raw, plan)
        reasons.extend(resource_reasons)
        safe_raw = copy.deepcopy(raw)
        # score_flow emits completed-only diagnostic L/P. A gate must instead
        # retain missing whole-cohort latency and throughput as unknown.
        complete_receipts = all(r['e2e_ms'] is not None for r in timings)
        if not complete_receipts:
            for row in safe_raw['request_records']:
                if receipts[row['request_id']] is None:
                    row['status'] = 'incomplete'
        quality = raw.get('quality_evidence', {})
        quality_refs = quality.get('source_receipts', [])
        for ref in quality_refs:
            _verify(ref)
        quality_ok = quality.get('kind') == 'measured_per_request' and bool(quality_refs) and not quality.get('errors')
        if not quality_ok:
            safe_raw['quality_evidence'] = {}
            reasons.append('MISSING_ACTUAL_QUALITY_PROVENANCE')
        elif set(quality.get('totals_by_request', {}))-set(receipts):
            raise ValueError('Foreign quality request identity')
        # Missing raw telemetry is a real unknown; helper accepts an empty list.
        safe_raw.setdefault('board_samples', [])
        diagnostic = score_flow(safe_raw, {**plan, 'resident_aliases': list(plan['aliases'])})
        terms = diagnostic['terms'].copy()
        if not complete_receipts:
            terms.update(T=None, L=None, P=None)
        if not memory_ok:
            terms['M'] = None
        if not power_ok:
            terms['E'] = None
        grid, boundary = [], plan.get('queue_depth_boundary')
        endpoints = starts
        for offset in protocol['queue_sample_offsets_ns']:
            now = raw['phase_start_ns']+offset
            relevant = [rid for rid, arrival in arrivals.items() if arrival <= now]
            known = all(endpoints[rid] is not None for rid in relevant)
            grid.append(dict(offset_ns=offset, depth=sum(endpoints[rid] > now for rid in relevant) if known else None))
        cell = dict(run_id=run_id, candidate_id=cid, execution_context_sha256=candidate['execution_context_sha256'],
                    requests=timings, terms=terms)
        if grid:
            cell['queue_depth_samples'] = grid
        cells.append(cell)
        reasons.extend(diagnostic['unknown_or_ineligible_reasons'])
        integrity = (raw.get('status') == 'completed' and raw.get('collection_complete') is True
                     and all(k in raw and not raw[k] for k in ('errors', 'runtime_errors', 'telemetry_errors', 'qualification_issues')))
        if raw.get('shutdown') and (set(raw['shutdown']) != set(plan['aliases']) or any(
                s.get('forced') or s.get('exit_code') != 0 or not s.get('bye_received') for s in raw['shutdown'].values())):
            integrity = False
        if not integrity:
            # Gate input has no eligibility field. Prevent failed collection
            # integrity from disappearing when only this cell is consumed.
            terms.update({k: None for k in terms})
            reasons.append('UNKNOWN_OR_FAILED_COLLECTION_INTEGRITY')
        rho, stability = raw.get('rho_admission_passed'), raw.get('stability', {}).get('stability_state_v2')
        audits.append(dict(candidate_id=cid, raw=receipt(raw_path), precollection=raw.get('precollection'),
            quality_sources=quality_refs, unknown_reasons=sorted(set(reasons)),
            terms_complete=all(v is not None for v in terms.values()), source_integrity_asserted=integrity,
            original_rho_assertion=rho, original_stability_assertion=stability,
            qualification_scope='Retained assertions only; this adapter does not recompute original rho/stability or grant eligibility',
            eligible_for_gain_claim=False, owned_memory_verified=memory_ok, board_power_verified=power_ok,
            resource_diagnostics=diagnostic['memory'], energy_diagnostics=diagnostic['energy']))
    if len(kinds) > 1:
        raise ValueError('CPU and physical observations cannot be mixed')
    observed = None
    if cells and len(begun) == len(cells):
        observed = dict(schema=OBSERVED_SCHEMA, split='validation', kind=next(iter(kinds)), run_id=run_id,
                        collection_started_unix_ns=min(begun), cells=cells)
    missing = sorted(set(candidates)-seen)
    complete = (observed is not None and not missing and all(a['terms_complete'] and not a['unknown_reasons'] for a in audits)
                and all(all(r[k] is not None for r in c['requests'] for k in ('service_ms', 'queue_ms', 'e2e_ms'))
                        and all(s['depth'] is not None for s in c.get('queue_depth_samples', [])) for c in cells))
    result = dict(schema=ADAPTER_SCHEMA, status='COMPLETE' if complete else 'UNKNOWN', metric_contract=METRIC_CONTRACT,
        protocol=receipt(protocol_path), predictions=receipt(predictions_path), plan=receipt(plan_path),
        adapter_source=receipt(__file__), scorer_source=receipt(Path(__file__).with_name('flow_metrics.py')),
        candidate_audits=audits, missing_candidate_ids=missing, queue_depth_boundary=plan.get('queue_depth_boundary'),
        observed_emitted=observed is not None, GPU_started=False, physical_gain_certified=False,
        original_rules_changed=False, claim='SOURCE_BOUND_OBSERVATION_ADAPTER_NOT_PREDICTOR_OR_GAIN_QUALIFICATION')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if observed is not None:
        _save(output/'observed-run.json', observed)
        result['observed'] = receipt(output/'observed-run.json')
    _save(output/'adapter-report.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    for op in ('precheck', 'adapt'):
        p = sub.add_parser(op)
        for name in ('protocol', 'predictions', 'plan', 'run-id', 'output'):
            p.add_argument('--'+name, required=True)
        if op == 'precheck':
            p.add_argument('--candidate-id', required=True)
        else:
            p.add_argument('--collections', required=True, help='JSON list of candidate_id/raw_path/precollection_path')
            p.add_argument('--allow-cpu-fixture', action='store_true')
    args = parser.parse_args(argv)
    common = dict(protocol_path=args.protocol, predictions_path=args.predictions, plan_path=args.plan, run_id=args.run_id)
    if args.op == 'precheck':
        result = verify_before_collection(**common, candidate_id=args.candidate_id, output_root=args.output)
    else:
        result = build_observed_run(**common, collections=_read(args.collections), output_dir=args.output,
                                    allow_cpu_fixture=args.allow_cpu_fixture)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
