"""Prospective statistical rho-profile qualification, without measured-key fiction.

One independent run checks transfer applicability, not a physical capacity
bound. Thresholds are sealed before target startup and never fitted here.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from . import flow_capacity_transfer as transfer
from .flow_capacity_catalog import verify
from .flow_eligibility import RHO_MAX, RHO_TOLERANCE
from .probe_validation import read, receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.prospective_statistical_rho_profile_qualification.v1'
POLICY = dict(mean_service_relative_error_max=.2,
    mean_service_error='abs(observed worker END-START mean - frozen training worker mean) / frozen training worker mean',
    populations='Every active family/task from offered routing, all declared cohort completions including bounded drain; no request discarded',
    required_candidate_coverage=1., required_request_coverage=1.,
    rho_threshold=RHO_MAX, rho_tolerance=RHO_TOLERANCE,
    clean_full_owned_telemetry_required=True, original_native_stability_required=True,
    physical_capacity_bound_required=False, old_target_measured_key_required=False,
    applicability='Exact training condition transfer except fresh seed/generated trace; valid for declared H90 condition only',
    evidence_scope='Single prospectively evaluated operating run; no confidence bound or stationary/long-horizon claim',
    thresholds_tuned_from_target=False)


def seal(transfer_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    transfer.verify_transfer(transfer_path); frozen = read(transfer_path)
    plan = read(frozen['target_plan']['path'])
    # Known conventional target locations are checked only for presence, never
    # opened. Report additionally checks actual startup against this seal.
    folder = Path(frozen['target_plan']['path']).parent
    if any((folder / name).exists() for name in ('runs', 'raw.json', 'scored-raw.json')):
        raise ValueError('Qualification must be sealed before any target collection')
    import gsdeploy.stability
    sources = [receipt(__file__), receipt(transfer.__file__),
        receipt(Path(__file__).with_name('flow_eligibility.py')), receipt(gsdeploy.stability.__file__),
        receipt(transfer_path), frozen['target_plan']]
    value = dict(schema=SCHEMA, kind='prospective_statistical_profile_qualification_protocol',
        created_ns=time.perf_counter_ns(), transfer=receipt(transfer_path), target_plan=frozen['target_plan'],
        policy=POLICY, source_receipts=sources, target_seed=plan['seed'], GPU_started=False,
        target_outcomes_read=False, physical_capacity_bound=False, old_target_resolver_modified=False,
        final_gain_eligible=False)
    value['protocol_sha256'] = digest(value); save(output, value); return value


def verify_protocol(path):
    value = read(path)
    if (value.get('schema') != SCHEMA or value.get('policy') != POLICY or
            value.get('protocol_sha256') != digest({k: v for k, v in value.items() if k != 'protocol_sha256'})):
        raise ValueError('Changed prospective statistical qualification policy')
    for ref in value['source_receipts']:
        verify(ref)
    transfer.verify_transfer(value['transfer']['path'])
    if read(value['transfer']['path'])['target_plan'] != value['target_plan']:
        raise ValueError('Qualification target differs from frozen transfer')
    return value


def _telemetry(plan, raw):
    samples = raw.get('board_samples', []); aliases = set(plan['aliases'])
    times = [s.get('time_ns') for s in samples]
    if (len(times) < 2 or any(type(t) is not int for t in times) or times != sorted(set(times)) or
            not times[0] <= raw['startup_begin_ns'] < raw['phase_start_ns'] < raw['deadline_ns'] <= times[-1] or
            max(b - a for a, b in zip(times, times[1:])) > 2_000_000_000):
        return False
    for sample in samples:
        memory = sample.get('owned_used_mib'); power = sample.get('watts')
        if (type(memory) not in (int, float) or not math.isfinite(memory) or memory < 0 or
                type(power) not in (int, float) or not math.isfinite(power) or power < 0 or
                sample.get('foreign_pids') or sample['time_ns'] >= raw['startup_end_ns'] and
                set(sample.get('owned_aliases', [])) != aliases):
            return False
    return True


def assess_rows(protocol, plan, descriptive, raws):
    """Pure CPU gate arithmetic; completeness is independent of service error."""
    rows = []; by_id = {c['id']: c for c in plan['candidates']}
    for row in descriptive['candidates']:
        cid = row['candidate_id']; raw = raws.get(cid)
        if raw is None:
            rows.append(dict(candidate_id=cid, status='NOT_RUN', statistical_profile_applicable=None,
                statistical_original_rho_planning_admission_passed=None, final_gain_eligible=False)); continue
        if raw['startup_begin_ns'] <= protocol['created_ns']:
            raise ValueError('Qualification protocol must precede actual target startup')
        prediction = row['transfer_prediction']; frozen = prediction['transfer']
        active = {(m, t) for t, weights in frozen['routing_rps_by_family'].items()
                  for m, rate in weights.items() if rate > 0}
        compared = row['independent_service_comparison']; checks = []
        for family, task in sorted(active):
            item = compared.get(family, {}).get(task); known = bool(item and item['request_count'] > 0)
            error = (abs(item['observed_mean_worker_service_ms'] - item['predicted_worker_service_ms']) /
                     item['predicted_worker_service_ms']) if known else None
            checks.append(dict(family=family, task=task, request_count=item['request_count'] if known else 0,
                predicted_worker_mean_ms=item['predicted_worker_service_ms'] if known else None,
                observed_worker_mean_ms=item['observed_mean_worker_service_ms'] if known else None,
                absolute_relative_mean_error=error, threshold=.2,
                passed=error <= .2 if known else None))
        telemetry = _telemetry(plan, raw)
        complete = (len(raw['request_records']) == len(plan['trace']) and all(r['status'] == 'completed' for r in raw['request_records']))
        applicable = complete and telemetry and all(c['passed'] is True for c in checks)
        stable = row['original_stability_passed']
        rho_pass = prediction['transferred_rho_threshold_satisfied']
        admission = applicable and stable and rho_pass
        rows.append(dict(candidate_id=cid, status='ASSESSED', mean_service_checks=checks,
            complete_cohort=complete, clean_full_owned_telemetry=telemetry,
            statistical_profile_applicable=applicable, original_native_stability_passed=stable,
            original_rho_formula_statistical_profile=frozen['rho_by_family'], original_rho_threshold_satisfied=rho_pass,
            statistical_original_rho_planning_admission_passed=admission,
            old_target_resolver=prediction['original_target_resolver'], old_target_resolver_admission=None,
            physical_capacity_bound_certified=False, final_gain_eligible=False,
            scope='Independent matched-condition statistical profile applicability plus unchanged original rho arithmetic and native stability; no measured target key or physical bound'))
    return rows


def report(protocol_path, raw_paths, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = verify_protocol(protocol_path); plan = read(protocol['target_plan']['path'])
    raws = {}
    for path in raw_paths:
        raw = read(path)
        if raw['candidate_id'] in raws:
            raise ValueError('Duplicate target candidate')
        if raw['startup_begin_ns'] <= protocol['created_ns']:
            raise ValueError('Qualification protocol must precede actual target startup')
        raws[raw['candidate_id']] = raw
    output.mkdir(parents=True, exist_ok=False)
    descriptive = transfer.report(protocol['transfer']['path'], raw_paths, output / 'transfer-report.json')
    rows = assess_rows(protocol, plan, descriptive, raws)
    value = dict(schema='gsdeploy.statistical_rho_profile_qualification_report.v1',
        status='ASSESSED' if len(raws) == len(plan['candidates']) else 'INCOMPLETE',
        protocol=receipt(protocol_path), descriptive_report=receipt(output / 'transfer-report.json'),
        report_source=receipt(__file__), candidates=rows, policy=protocol['policy'],
        independent_operating_run_count_per_candidate=1, uncertainty_calibrated=False,
        physical_capacity_bound_certified=False, final_gain_eligible=False, GPU_started=False,
        scope='Operational statistical-planning qualification only; final six-term gains and repeated confirmation remain separate')
    save(output / 'report.json', value); return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='op', required=True)
    s = sub.add_parser('seal'); s.add_argument('--transfer', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('report'); s.add_argument('--protocol', required=True); s.add_argument('--raw', action='append', required=True); s.add_argument('--output', required=True)
    args = p.parse_args(argv)
    value = seal(args.transfer, args.output) if args.op == 'seal' else report(args.protocol, args.raw, args.output)
    print(value['schema'], 'GPU_started=False'); return 0


if __name__ == '__main__':
    raise SystemExit(main())
