"""CPU-only admission2 training capacity adapter; shared evidence cannot enter.

Keeps the frozen input-standardization, original g(n)/n and rho arithmetic.
Training CV is not independent service qualification or global-permit proof.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from . import admission_training_v1 as training
from . import capacity_input_weighted_v1 as weighted
from .flow_capacity_catalog import condition
from .flow_capacity_transfer import _generation, _plan
from .flow_eligibility import RHO_MAX, RHO_TOLERANCE
from .h600_capacity_v1 import service_checks
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest

SCHEMA = 'gsdeploy.admission2_training_capacity.v1'
SEEDS = training.SEEDS
ROUTES = training.ROUTES
TARGET_SEED = 628201
POLICY = dict(training_seeds=list(SEEDS), candidate_ids=list(ROUTES),
    independent_target_seed=TARGET_SEED,
    numerical_engine=weighted.VERSION, normalization='Original g(n)/resident n; never substitute max_active=2',
    pooling='Exact controller, candidate route, input law and complete condition only; no shared, pilot or independent outcomes',
    cv='Entire seed excluded across all four routes', rho_threshold=RHO_MAX,
    rho_tolerance=RHO_TOLERANCE, mean_service_relative_error_max=.2,
    error_denominator='Predicted worker mean', independent_validation_sources_used=[],
    capacity_bound=False, final_gain_eligible=False)
ACCEPTANCE = dict(mean_service_relative_error_max=.2,denominator='predicted_worker_mean',
    original_stability_required=True,controller_aware_raw_audit_required=True,
    full_cohort_required=True,clean_owned_telemetry_required=True,
    global_permit_and_queue_validation_separate=True)


def sources():
    return [receipt(p) for p in (__file__, training.__file__, weighted.__file__,
        condition.__code__.co_filename, _generation.__code__.co_filename,
        service_checks.__code__.co_filename)] + weighted.source_files()


def unique_refs(refs):
    result = {}
    for ref in refs:
        checked(ref)
        if ref['path'] in result and result[ref['path']] != ref:
            raise ValueError('Conflicting admission training source receipts')
        result[ref['path']] = ref
    return list(result.values())


def controller_identity(plan, candidate):
    binding = plan.get('controller_binding', {})
    body = {k: v for k, v in binding.items() if k != 'controller_sha256'}
    if (candidate.get('id') not in ROUTES or not binding
            or binding.get('controller_sha256') != digest(body)
            or binding.get('base_worker_context_sha256') != candidate.get('execution_context_sha256')
            or binding.get('base_worker_context') != plan.get('context')
            or binding.get('resource_configuration') != training.native_flow.resource_configuration(2)
            or binding.get('resource_configuration_sha256') != digest(binding['resource_configuration'])
            or candidate.get('resource_configuration') != binding['resource_configuration']):
        raise ValueError('Exact separately bound admission2 controller and route required')
    return binding['controller_sha256']


def prepare(campaign_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    campaign = training.verify(campaign_path)
    value = dict(schema=SCHEMA+'.protocol', created_ns=time.perf_counter_ns(), policy=POLICY,
        campaign=receipt(campaign_path), controller_binding=campaign['controller_binding'],
        source_receipts=unique_refs([*sources(), *campaign['source_receipts']]),
        training_outcomes_read=False, independent_outcomes_read=False, GPU_started=False)
    value['protocol_sha256'] = digest(value)
    save(output, value)
    return value


def verify_protocol(path):
    value = read(path)
    if (value.get('schema') != SCHEMA+'.protocol' or value.get('policy') != POLICY
            or type(value.get('created_ns')) is not int
            or value.get('training_outcomes_read') is not False
            or value.get('independent_outcomes_read') is not False
            or value.get('protocol_sha256') != digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed admission capacity protocol')
    unique_refs([value['campaign'], *value['source_receipts']])
    campaign = training.verify(value['campaign']['path'])
    if value['controller_binding'] != campaign['controller_binding']:
        raise ValueError('Admission capacity controller differs from campaign')
    required = {(r['path'], r['sha256']) for r in [*sources(), *campaign['source_receipts']]}
    if not required <= {(r['path'], r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Capacity protocol omitted required sources')
    return value


def load_campaign(campaign_path):
    """Common strict training loader for resource and capacity consumers."""
    campaign = training.verify(campaign_path)
    if any(not (Path(j['output'])/'raw.json').is_file() for j in campaign['jobs']):
        raise ValueError('All eight admission2 primary raw files required before fitting')
    runs = []
    for job in campaign['jobs']:
        mapped = training.map_training(job, campaign_path)
        plan, candidate = mapped['plan'], mapped['candidate']
        identity = controller_identity(plan, candidate)
        evidence, traces = mapped['resource_evidence'], mapped['traces']
        if (evidence.get('split') != 'training' or evidence.get('controller_sha256') != identity
                or evidence.get('resource_configuration') != candidate['resource_configuration']
                or len(traces) != 1 or traces[0].get('split') != 'training'
                or traces[0].get('controller_sha256') != identity
                or traces[0].get('resource_configuration') != candidate['resource_configuration']
                or traces[0].get('execution_context_sha256') != candidate['execution_context_sha256']
                or evidence.get('execution_context_sha256') != candidate['execution_context_sha256']
                or mapped['job'] != job or mapped['native_raw'] != receipt(Path(job['output'])/'raw.json')):
            raise ValueError('Mapped admission training role/controller/source mismatch')
        observations = mapped['service_observations']
        by_id = {r['request_id']: r for r in observations}
        if len(by_id) != len(observations) or set(by_id) != {r['request_id'] for r in plan['trace']}:
            raise ValueError('All offered training service observations required')
        for offered in plan['trace']:
            row = by_id[offered['request_id']]
            alias = candidate['routes'][offered['request_id']]
            if (row.get('split') != 'training' or row.get('controller_sha256') != identity
                    or (row['alias'], row['family'], row['task'], row['pool_id']) !=
                    (alias, plan['aliases'][alias]['family'], offered['task'], offered['pool_id'])
                    or not weighted.finite_positive(row['worker_service_ms'])):
                raise ValueError('Training service role/controller/route/input mismatch')
        spec = condition(plan, candidate)
        original_raw = read(checked(mapped['native_raw']))
        if type(original_raw.get('startup_begin_ns')) is not int:
            raise ValueError('Original native training startup clock required')
        runs.append(dict(flow_id=job['job_id'], seed=job['seed'], candidate_id=job['candidate_id'],
            plan=plan, plan_source=job['plan'], raw_source=mapped['native_raw'],
            controller_sha256=identity, condition=spec, condition_sha256=digest(spec),
            generation=_generation(plan, spec), observations=observations,
            startup_begin_ns=original_raw['startup_begin_ns'],
            evidence=evidence, traces=traces, mapping_report=mapped['mapping_report']))
    if ({(r['seed'],r['candidate_id']) for r in runs} != {(s,c) for s in SEEDS for c in ROUTES}
            or len(runs) != 8):
        raise ValueError('Exactly two seeds by four admission2 routes required')
    for cid in ROUTES:
        selected = [r for r in runs if r['candidate_id']==cid]
        if any((r['condition'],r['generation'],r['controller_sha256']) !=
               (selected[0]['condition'],selected[0]['generation'],selected[0]['controller_sha256']) for r in selected):
            raise ValueError('Admission training seed conditions differ')
    return dict(runs=runs, source_receipts=unique_refs([
        receipt(campaign_path), *campaign['source_receipts'], *sources(),
        *[r['raw_source'] for r in runs], *[s for r in runs for s in r['evidence']['sources']]]))


def load_training(protocol_path):
    protocol = verify_protocol(protocol_path)
    bundle = load_campaign(protocol['campaign']['path'])
    if any(not protocol['created_ns'] < run['startup_begin_ns'] for run in bundle['runs']):
        raise ValueError('Capacity protocol must be frozen before every training startup')
    bundle.update(protocol=protocol,source_receipts=unique_refs([
        receipt(protocol_path),*protocol['source_receipts'],*bundle['source_receipts']]))
    return bundle


def estimate(plan, candidate, runs, catalog):
    """Public target inputs plus training only; original gate remains UNKNOWN."""
    identity = controller_identity(plan, candidate)
    selected = [r for r in runs if r.get('controller_sha256') == identity
        and r.get('candidate_id') == candidate['id']]
    result = weighted.estimate(plan, candidate, selected, catalog)
    result.update(schema=SCHEMA+'.estimate', controller_sha256=identity,
        resource_configuration=copy.deepcopy(candidate['resource_configuration']),
        candidate_id=candidate['id'], admission_policy=POLICY,
        global_permit_capacity_qualified=None,
        global_permit_scope='Original family rho does not prove coupled global permit capacity; full-horizon drain and independent queue validation still required')
    return result


def compute_fit(bundle):
    runs = bundle['runs']
    full, folds = [], []
    for run in sorted(runs, key=lambda r:(r['seed'],r['candidate_id'])):
        plan = run['plan']
        candidate = next(c for c in plan['candidates'] if c['id']==run['candidate_id'])
        catalog = read(checked(plan['catalog']))
        common = dict(seed=run['seed'], candidate_id=run['candidate_id'],
            public_plan=run['plan_source'], heldout_raw=run['raw_source'])
        full.append({**common, 'prediction':estimate(plan,candidate,runs,catalog),
            'scope':'In-sample public training reference; no independent qualification'})
        retained = [r for r in runs if r['seed'] != run['seed']]
        prediction = estimate(plan,candidate,retained,catalog)
        checks = service_checks(prediction,run['observations'])
        folds.append({**common, 'training_seeds':sorted({r['seed'] for r in retained}),
            'training_raw_receipts':[r['raw_source'] for r in retained],
            'prediction':prediction, 'service_checks':checks,
            'service_applicability_diagnostic':all(c['passed'] is True for c in checks),
            'qualification_passed':False})
    return dict(schema=SCHEMA+'.fit', policy=POLICY, full_training_references=full,
        whole_seed_cv=folds, source_receipts=bundle['source_receipts'],
        training_only=True, independent_validation_sources_used=[],
        original_rho_admission_passed=None, global_permit_capacity_qualified=None,
        physical_capacity_bound=False, final_gain_eligible=False, GPU_started=False)


def fit(protocol_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    bundle = load_training(protocol_path)
    document = compute_fit(bundle)
    # Recheck frozen sources before producing any fitted artifact.
    unique_refs(bundle['source_receipts'])
    output.mkdir(parents=True, exist_ok=False)
    save(output/'capacity-training-fit.json', document)
    value = dict(schema=SCHEMA+'.manifest', created_ns=time.perf_counter_ns(),
        protocol=receipt(protocol_path), policy=POLICY,
        fit=receipt(output/'capacity-training-fit.json'), source_receipts=bundle['source_receipts'],
        independent_validation_sources_used=[], final_gain_eligible=False, GPU_started=False)
    value['manifest_sha256'] = digest(value)
    save(output/'manifest.json', value)
    return value


def verify_manifest(path, *, recompute=True):
    value = read(path)
    if (value.get('schema') != SCHEMA+'.manifest' or value.get('policy') != POLICY
            or value.get('independent_validation_sources_used') != []
            or value.get('final_gain_eligible') is not False or value.get('GPU_started') is not False
            or value.get('manifest_sha256') != digest({k:v for k,v in value.items() if k!='manifest_sha256'})):
        raise ValueError('Changed admission capacity manifest')
    verify_protocol(checked(value['protocol']))
    unique_refs(value['source_receipts'])
    document = read(checked(value['fit']))
    if recompute and document != compute_fit(load_training(value['protocol']['path'])):
        raise ValueError('Capacity fit differs from recomputed admission training')
    return value


def target_plan(plan_path):
    plan = _plan(plan_path)
    if (plan['seed'] != TARGET_SEED or plan.get('split') != 'validation'
            or plan.get('kind') != 'prospective_capacity_validation'
            or plan.get('phase') != 'predictor_validation'
            or tuple(c['id'] for c in plan['candidates']) != ROUTES
            or plan.get('window_seconds') != 600. or plan.get('drain_seconds') != 300.
            or plan.get('warmup_seconds') != 0.):
        raise ValueError('Disjoint prospective admission2 H600 capacity target required')
    for candidate in plan['candidates']:
        controller_identity(plan,candidate)
    return plan


def seal(plan_path, protocol_path, output):
    """Reserve a future disjoint capacity target; accepts no target outcomes.

    This freezes a statistical prediction only. Independent controller-aware
    raw mapping, original stability and coupled queue validation remain needed.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    plan = target_plan(plan_path)
    bundle = load_training(protocol_path)
    catalog = read(checked(plan['catalog']))
    predictions = [dict(candidate_id=c['id'],prediction=estimate(plan,c,bundle['runs'],catalog))
        for c in plan['candidates']]
    refs = unique_refs([receipt(plan_path),*bundle['source_receipts'],
        *training._refs([plan['controller_binding'],*[plan[k] for k in ('context','profile_plan','runner_source','catalog')]])])
    value = dict(schema=SCHEMA+'.seal',created_ns=time.perf_counter_ns(),policy=POLICY,
        target_plan=receipt(plan_path),training_protocol=receipt(protocol_path),predictions=predictions,
        source_receipts=refs,target_outcomes_used=False,independent_validation_sources_used=[],
        acceptance=copy.deepcopy(ACCEPTANCE),
        statistical_service_applicability=None,original_rho_admission_passed=None,
        global_permit_capacity_qualified=None,physical_capacity_bound=False,
        final_gain_eligible=False,GPU_started=False)
    if type(plan.get('created_ns')) is not int or value['created_ns'] <= plan['created_ns']:
        raise ValueError('Public target plan must be created before capacity seal')
    value['seal_sha256'] = digest(value)
    save(output,value)
    return value


def verify_seal(path):
    value = read(path)
    if (value.get('schema') != SCHEMA+'.seal' or value.get('policy') != POLICY
            or value.get('target_outcomes_used') is not False
            or value.get('independent_validation_sources_used') != []
            or value.get('acceptance') != ACCEPTANCE
            or value.get('statistical_service_applicability') is not None
            or value.get('original_rho_admission_passed') is not None
            or value.get('global_permit_capacity_qualified') is not None
            or value.get('final_gain_eligible') is not False or value.get('GPU_started') is not False
            or value.get('seal_sha256') != digest({k:v for k,v in value.items() if k!='seal_sha256'})):
        raise ValueError('Changed prospective admission capacity seal')
    unique_refs(value['source_receipts'])
    plan = target_plan(checked(value['target_plan']))
    if (type(value.get('created_ns')) is not int or type(plan.get('created_ns')) is not int
            or value['created_ns'] <= plan['created_ns']):
        raise ValueError('Target plan/seal chronology changed')
    bundle = load_training(checked(value['training_protocol']))
    needed = [value['target_plan'],*bundle['source_receipts'],
        *training._refs([plan['controller_binding'],*[plan[k] for k in ('context','profile_plan','runner_source','catalog')]])]
    if not {(r['path'],r['sha256']) for r in needed} <= {(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Capacity seal omitted required target or training sources')
    catalog = read(checked(plan['catalog']))
    expected = [dict(candidate_id=c['id'],prediction=estimate(plan,c,bundle['runs'],catalog))
        for c in plan['candidates']]
    if value['predictions'] != expected:
        raise ValueError('Prospective capacity predictions differ from recomputed training')
    return value


def assess_independent_service(prediction, observations):
    """Numerical adapter for an independently audited controller-aware mapper.

    This function deliberately accepts no raw-file path and performs no role
    relabelling. The caller must audit source, disjoint seed, pre-startup seal,
    controller, complete cohort and original stability before qualification.
    """
    checks = service_checks(prediction,observations)
    known = prediction.get('status') == 'ESTIMATED' and bool(checks)
    passed = all(c['passed'] is True for c in checks) if known else None
    return dict(mean_service_checks=checks,mean_service_numerical_threshold_passed=passed,
        threshold=.2,error_denominator='predicted_worker_mean',
        independent_raw_and_temporal_audit_performed=False,
        statistical_service_applicability=None,original_rho_admission_passed=None,
        global_permit_capacity_qualified=None,final_gain_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--campaign',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('fit'); p.add_argument('--protocol',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('verify'); p.add_argument('--manifest',required=True)
    p = sub.add_parser('seal'); p.add_argument('--plan',required=True); p.add_argument('--protocol',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('verify-seal'); p.add_argument('--sealed',required=True)
    args = parser.parse_args()
    if args.operation == 'prepare': result = prepare(args.campaign,args.output)
    elif args.operation == 'fit': result = fit(args.protocol,args.output)
    elif args.operation == 'verify': result = verify_manifest(args.manifest)
    elif args.operation == 'seal': result = seal(args.plan,args.protocol,args.output)
    else: result = verify_seal(args.sealed)
    print(json.dumps(dict(schema=result['schema'], GPU_started=False)))


if __name__ == '__main__':
    main()
