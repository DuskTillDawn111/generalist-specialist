"""Fresh CPU plain/bounded DFS of the sealed admission V4 route domain.

The original six-term forecast and route/controller-specific T8 providers are
reused without fitting. V4 evidence supplies feasibility gates only; stored
predicted utilities and F8 outcomes are never search objectives.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

from gsdeploy.forecast_errors import UnknownEvidence
from . import admission_validation_v1 as validation
from . import admission_capacity_v1 as capacity
from . import admission_quality_v1 as quality
from . import admission_resources_v1 as resources
from . import h600_search_core_v2 as core
from .native_flow import checked, receipt, save
from .probe_validation import read

TOLERANCE = core.TOLERANCE
SCHEMA = 'gsdeploy.admission_V4_same_domain_DFS_check.v1'


def conjunction(values):
    values = list(values)
    if any(v is False for v in values): return False
    if all(v is True for v in values): return True
    return None


def qualification(row, original):
    """No transferred mean/rho check substitutes for the original resolver."""
    row = row or {}
    checks = dict(
        complete_independent_evidence=row.get('complete_independent_evidence'),
        statistical_service_applicability=row.get('statistical_service_applicability'),
        statistical_rho_threshold=row.get('original_rho_threshold_satisfied'),
        original_stability=row.get('original_stability', {}).get('original_stability_passed'),
        global_permit_capacity_qualified=row.get('global_permit_capacity_qualified'),
        original_measured_profile_resolver_admission=row.get('original_measured_profile_resolver_admission'),
        source_bound_original_rho_admission=original.get('rho_admission_passed'))
    if any(v is not None and type(v) is not bool for v in checks.values()):
        raise ValueError('Qualification checks must be bool or UNKNOWN')
    return dict(passed=conjunction(checks.values()), checks=checks)


def bound(plan, candidate, quality_document, overrides):
    overlaid = copy.deepcopy(candidate)
    for key in ('predicted_startup_joules', 'predicted_prephase_peak_mib'):
        value = overrides[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative route resource overlays required')
        overlaid[key] = value
    if (set(plan['weights']) != set('QTLPME')
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in plan['weights'].values())
            or not math.isfinite(plan['memory_safe_factor']) or plan['memory_safe_factor'] < 0
            or not math.isfinite(plan['memory_budget_mib']) or plan['memory_budget_mib'] <= 0):
        raise ValueError('Nonnegative original six-term costs and positive memory budget required')
    return core.upper_bound(plan, overlaid, quality_document['loss_by_family_task'])


def evaluate(plan, candidate, model, resource_document, quality_document, overrides, original, gate):
    try:
        # This entrypoint enforces the route/controller-aware providers before
        # constructing a new FullHorizonModel for the entire arrival trace.
        predicted = validation.predict(plan, candidate, model, resource_document, quality_document, overrides)
    except (UnknownEvidence, ValueError) as exc:
        return dict(comparison_U=None, model_eligible=None, qualified_eligible=conjunction([None, gate['passed']]),
                    original_rho=original, qualification=gate, unknown_reasons=[type(exc).__name__ + ':' + str(exc)])
    metric = predicted.get('metrics')
    comparable = bool(metric and metric['absolute_utility_complete'] is True
                      and metric['six_term_comparison_complete'] is True)
    value = predicted['comparison_U'] if comparable else None
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError('Finite original utility required')
    constraints = dict(memory_budget_satisfied=metric.get('memory_budget_satisfied') if metric else None,
                       drain_complete=not metric['incomplete_ids'] if metric else None)
    # A known violated constraint is a definite exclusion even if another term
    # is unknown. A comparable objective is also required for membership.
    eligible = conjunction([comparable if comparable else None, *constraints.values()])
    return dict(comparison_U=value, absolute_U=metric.get('full_U') if metric else None,
                terms=metric.get('terms') if metric else None, model_eligible=eligible,
                model_constraints=constraints, qualified_eligible=conjunction([eligible, gate['passed']]),
                original_rho=original, qualification=gate,
                unknown_reasons=([predicted['unknown_reason']] if predicted.get('unknown_reason') else [])
                    + (metric.get('unknown_reasons', []) if metric else []),
                cohort_size=len(predicted['requests']), quality_prediction=predicted['quality_prediction'],
                candidate_resource_estimates=predicted['predicted_candidate_resource_estimates'])


def optimum(values, unknown, count):
    maximum = max(values.values()) if values else None
    ties = sorted(cid for cid, value in values.items() if maximum - value <= TOLERANCE) if values else []
    status = 'UNKNOWN' if unknown else 'COMPLETE' if values else 'EMPTY_DOMAIN'
    return dict(status=status, maximum_known_U=maximum, all_known_tie_ids=ties,
                certified_maximum_U=maximum if status == 'COMPLETE' else None,
                certified_all_tie_ids=ties if status == 'COMPLETE' else None,
                unknown_ids=sorted(unknown), finite_domain_size=count,
                explicit_empty_domain=status == 'EMPTY_DOMAIN')


def search(plan, model, route_resources, route_quality, overrides, gates, originals, *, bounded=False):
    started = time.perf_counter(); records = []; pruned = []
    values = {}; qualified_values = {}; unknown = []; qualified_unknown = []
    candidates = plan['candidates']
    bounds = {c['id']: bound(plan, c, route_quality[c['id']], overrides[c['id']]) for c in candidates}
    if bounded: candidates = sorted(candidates, key=lambda c: (-bounds[c['id']]['U_upper'], c['id']))
    for candidate in candidates:
        cid = candidate['id']; upper = bounds[cid]['U_upper']
        incumbent = max(values.values()) if values else None
        qualified_incumbent = max(qualified_values.values()) if qualified_values else None
        # A model-only incumbent must not prune a potentially qualified route.
        model_dominated = incumbent is not None and upper < incumbent - TOLERANCE
        qualified_dominated = (gates[cid]['passed'] is not True or
            qualified_incumbent is not None and upper < qualified_incumbent - TOLERANCE)
        if bounded and model_dominated and qualified_dominated:
            pruned.append(dict(candidate_id=cid, bound=bounds[cid], incumbent_U=incumbent,
                               qualified_incumbent_U=qualified_incumbent))
            if gates[cid]['passed'] is None: qualified_unknown.append(cid)
            continue
        result = evaluate(plan, candidate, model, route_resources[cid], route_quality[cid],
                          overrides[cid], originals[cid], gates[cid])
        if result['comparison_U'] is not None and result['comparison_U'] > upper + TOLERANCE:
            raise AssertionError('Nonnegative six-term upper bound violated')
        for field, known_values, unknown_ids in (
            ('model_eligible', values, unknown), ('qualified_eligible', qualified_values, qualified_unknown)):
            if result[field] is None: unknown_ids.append(cid)
            elif result[field] is True: known_values[cid] = result['comparison_U']
        records.append(dict(candidate_id=cid, bound=bounds[cid], result=result))
    return dict(algorithm='bounded_leaf_DFS' if bounded else 'plain_leaf_DFS',
                model=optimum(values, unknown, len(candidates)),
                original_qualified=optimum(qualified_values, qualified_unknown, len(candidates)),
                candidate_count=len(candidates), fresh_forecast_calls=len(records), bound_pruned=pruned,
                records=records, elapsed_seconds=time.perf_counter()-started)


def run(sealed_dir, validation_report, output):
    """Audit original V4 evidence then fresh-search twice; never launch a GPU."""
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    sealed_dir = Path(sealed_dir).resolve(); validation_report = Path(validation_report).resolve()
    # Capture before verification/parse. A new report or audit appearing after
    # verification must not inherit that old verification's authority.
    entry_refs = [receipt(path) for path in (validation_report,
        validation_report.parent/'validation-audit.json', sealed_dir/'seal-audit.json',
        sealed_dir/'predictions.json')]
    verification = validation.verify_report(validation_report)
    capacity.unique_refs(entry_refs)
    if verification.get('verified') is not True or verification.get('report') != entry_refs[0]:
        raise ValueError('Independent verification must bind the original report receipt')
    report_audit = read(checked(entry_refs[1]))
    if report_audit['seal_audit'] != entry_refs[2]:
        raise ValueError('V4 report must bind the exact search seal')
    # verify_report already recomputes check_sealed, all three manifests, the
    # protocol and native evidence. Read those exact audited inputs, rather
    # than repeat T8 fitting audits at every search node or pass.
    audit = read(checked(entry_refs[2]))
    protocol = read(checked(audit['protocol']))
    plan = read(checked(protocol['target_plan'])); model = read(checked(protocol['model']))
    rm = read(checked(audit['resource_manifest']))
    qm = read(checked(audit['quality_manifest']))
    cm = read(checked(audit['capacity_manifest']))
    cs = read(checked(audit['capacity_seal']))
    rows = read(checked(report_audit['capacity_qualification']))['rows']
    ids = [c['id'] for c in plan['candidates']]
    if tuple(ids) != capacity.ROUTES or len({r['candidate_id'] for r in rows}) != len(rows) or set(r['candidate_id'] for r in rows) != set(ids):
        raise ValueError('Exact four-route V4 qualification domain required')
    row_by_id = {r['candidate_id']: r for r in rows}
    route_resources = {cid: read(checked(rm['full']['route_resources'][cid])) for cid in ids}
    route_quality = {cid: read(checked(qm['full']['route_quality'][cid])) for cid in ids}
    overrides = rm['full']['candidate_overrides']
    catalog = read(checked(plan['catalog']))
    originals = {c['id']: core.original_rho(plan, c, catalog) for c in plan['candidates']}
    gates = {cid: qualification(row_by_id[cid], originals[cid]) for cid in ids}
    refs = capacity.unique_refs(validation.training._refs([audit, protocol, plan, model, rm, qm, cm, cs,
        route_resources, route_quality, report_audit, entry_refs,
        receipt(validation_report), receipt(__file__), receipt(core.__file__), receipt(core.original_rho.__code__.co_filename)]))
    plain = search(plan, model, route_resources, route_quality, overrides, gates, originals)
    capacity.unique_refs(refs)
    bounded = search(plan, model, route_resources, route_quality, overrides, gates, originals, bounded=True)
    capacity.unique_refs(refs)
    same = all(plain[k]['maximum_known_U'] == bounded[k]['maximum_known_U']
               and plain[k]['all_known_tie_ids'] == bounded[k]['all_known_tie_ids']
               for k in ('model', 'original_qualified'))
    same_domain_status = all(plain[k]['status'] == bounded[k]['status'] for k in ('model', 'original_qualified'))
    model_status = plain['model']['status']
    status = 'FAIL' if not same else 'UNKNOWN' if model_status == 'UNKNOWN' else 'EMPTY_MODEL_DOMAIN' if model_status == 'EMPTY_DOMAIN' else 'PASS'
    summary = dict(schema=SCHEMA, status=status, same_known_optima_and_ties=same,
        same_search_output=same and same_domain_status, same_domain_status=same_domain_status,
        evaluated_unknown_ids_by_search={name:{k:result[k]['unknown_ids'] for k in ('model','original_qualified')}
            for name,result in (('plain',plain),('bounded',bounded))},
        same_model_optimum_and_all_ties=same if model_status == 'COMPLETE' else None,
        model=plain['model'], original_qualified=plain['original_qualified'],
        global_permit_capacity_proof='UNKNOWN', original_gate_constrained_optimality_certified=False,
        physical_optimality_certified=False, final_gain_eligible=False, GPU_started=False,
        service_refitted=False, stored_prediction_used_as_objective=False, F8_outcomes_read=False,
        V4_outcomes_scope='Audited applicability/stability qualification only; no utility optimization or fit',
        domain='Four reserved admission2 static routes; identical V4 public full arrival/input trace, original six terms, exact T8 route/controller resources/quality',
        objective_scope='Fresh original six-term full_U; model memory budget and bounded drain constraints',
        unsearched=['other deployments', 'replica counts', 'unlisted routes', 'resource shares', 'joint batch/cache execution', 'adaptive policies'],
        upper_bound_proof='Nonnegative six-term costs: training Q lower bound and safe declared route peak/budget M lower bound; other costs >=0. Strict upper<incumbent-1e-12 preserves all ties; qualified search keeps a separate incumbent.',
        empirical_resource_scope='Training estimates, not physical upper bounds',
        validation_report_verification=verification, source_receipts=refs,
        original_rho={cid: originals[cid] for cid in ids}, qualifications=gates,
        candidate_count=len(ids), request_count=len(plan['trace']), cohort_sha256=core.cohort_sha(plan),
        pruning_count=len(bounded['bound_pruned']), forecast_calls_saved=plain['fresh_forecast_calls']-bounded['fresh_forecast_calls'],
        search_acceleration_claim=False,
        plain={k:v for k,v in plain.items() if k != 'records'},
        bounded={k:v for k,v in bounded.items() if k != 'records'})
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('summary', summary), ('plain-dfs', plain), ('bounded', bounded)):
        save(output/(name+'.json'), value)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sealed-dir', 'validation-report', 'output'): parser.add_argument('--'+key, required=True)
    args = parser.parse_args(); result = run(args.sealed_dir, args.validation_report, args.output)
    print(json.dumps({k:result[k] for k in ('status', 'model', 'original_qualified', 'pruning_count')}))
