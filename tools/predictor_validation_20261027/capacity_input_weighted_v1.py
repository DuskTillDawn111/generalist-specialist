"""Input-standardized conditional capacity estimates from training only.

The public target trace supplies input proportions, never measured service.
Means remain conditional on the complete alias/load/resource/route condition.
Original g(n), family-average rho and .8+1e-8 are preserved. Missing positive
weight input groups stay UNKNOWN; neither pooled nor solo means fill gaps.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
from pathlib import Path
import statistics
import time

from .flow_capacity_catalog import condition, training_evidence, verify
from .flow_capacity_transfer import _generation, _plan
from .flow_eligibility import _routing, RHO_MAX, RHO_TOLERANCE
from .profiling import digest
from .probe_validation import read, receipt, save


VERSION = 'gsdeploy.input_weighted_conditional_capacity.v1'
ROUTES = {'shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high'}
POLICY = {
    'stratum': ['complete_condition_sha256', 'family', 'task', 'pool_id'],
    'estimator': 'Arithmetic mean successful worker END-START within stratum',
    'target_weights': 'Public exact target trace counts conditional on routed family/task, normalized within that family/task',
    'weight_semantics': 'Finite declared workload composition, not population input law or offered arrival rate',
    'condition': 'Exact execution identity, aliases, resource configuration, alias routing weights, offered rates, H, drain, warmup, arrival process and input domain',
    'normalization': 'family_service_ms = sum_input(target_weight * training_worker_mean_ms) * original_g(n)/n',
    'rho': 'sum_task(original_offered_family_rps * family_service_ms/1000)/original_g(n)',
    'missing': 'Any positive-weight input absent in matched training is UNKNOWN; no renormalization or pooled/solo fallback',
    'alias': 'Same-family aliases remain pooled within exact alias-route condition; no new alias rho threshold',
    'rho_threshold': RHO_MAX, 'rho_tolerance': RHO_TOLERANCE,
    'capacity_bound': False, 'uncertainty_calibrated': False,
}


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def source_files():
    names = ('capacity_input_weighted_v1.py', 'flow_capacity_catalog.py', 'flow_capacity_transfer.py',
             'flow_eligibility.py', 'native_flow.py', 'profiling.py')
    sources = [receipt(Path(__file__).with_name(name)) for name in names]
    import gsdeploy.performance, gsdeploy.routing, gsdeploy.stability
    sources.extend(receipt(m.__file__) for m in (gsdeploy.performance, gsdeploy.routing, gsdeploy.stability))
    return sources


def load_training(suite_path):
    suite = read(suite_path)
    if (suite.get('schema') != 'gsdeploy.continuous_training_suite.v1' or
            suite.get('no_validation_data') is not True or len(suite['seeds']) != 2 or
            len(set(suite['seeds'])) != 2 or len(suite['jobs']) != 8):
        raise ValueError('Complete prospectively training-only two-seed/eight-flow suite required')
    verify(suite['source'])
    runs, seen = [], set()
    sources = [receipt(suite_path), suite['source'], *source_files()]
    for job in suite['jobs']:
        raw_path = Path(job['output'])/'raw.json'
        plan, candidate, raw, derived, observations, peak = training_evidence(raw_path, job['plan'])
        fid = 'seed-' + str(plan['seed']) + '__' + candidate['id']
        if (plan['seed'] not in suite['seeds'] or candidate['id'] != job['candidate_id'] or
                candidate['id'] not in ROUTES or fid in seen):
            raise ValueError('Unknown, substituted, or duplicate training flow')
        seen.add(fid)
        spec = condition(plan, candidate)
        generation = _generation(plan, spec)
        raw_ref = receipt(raw_path)
        runs.append({'flow_id': fid, 'seed': plan['seed'], 'candidate_id': candidate['id'],
                     'plan': plan, 'plan_source': job['plan'], 'raw_source': raw_ref,
                     'condition': spec, 'condition_sha256': digest(spec),
                     'generation': generation, 'observations': observations, 'raw_peak_mib': peak})
        sources.extend([raw_ref, job['plan'], *[plan[k] for k in
            ('protocol_generator', 'runner_source', 'context', 'catalog', 'profile_plan')]])
    if seen != {'seed-' + str(s) + '__' + c for s in suite['seeds'] for c in ROUTES}:
        raise ValueError('Every seed must cover all four declared routes')
    if len({r['raw_source']['sha256'] for r in runs}) != 8:
        raise ValueError('Duplicate native raw SHA across training flows')
    for route in ROUTES:
        selected = [r for r in runs if r['candidate_id'] == route]
        if (selected[0]['condition'] != selected[1]['condition'] or
                selected[0]['generation'] != selected[1]['generation']):
            raise ValueError('Training seed conditions or input generation differ')
    for ref in sources:
        verify(ref)
    return {'runs': runs, 'source_receipts': sources, 'seeds': sorted(suite['seeds'])}


def input_weights(plan, candidate):
    """Only public routed inputs; actual target records are not an argument."""
    routing = _routing(plan, candidate)
    counts = collections.defaultdict(collections.Counter)
    for row in plan['trace']:
        alias = candidate['routes'][row['request_id']]
        family = plan['aliases'][alias]['family']
        counts[(family, row['task'])][row['pool_id']] += 1
    rows = []
    for task, families in sorted(routing.items()):
        for family, rps in sorted(families.items()):
            if not rps:
                continue
            inputs = counts[(family, task)]
            total = sum(inputs.values())
            rows.append({'family': family, 'task': task, 'offered_family_rps': rps,
                         'public_target_request_count': total,
                         'weights': {pool: count/total for pool, count in sorted(inputs.items())},
                         'counts': dict(sorted(inputs.items()))})
    return rows


def estimate(plan, candidate, training_runs, catalog):
    """No target service/outcomes input; uses training with exact conditions."""
    spec = condition(plan, candidate)
    generation = _generation(plan, spec)
    selected = [r for r in training_runs if r['condition'] == spec and r['generation'] == generation]
    deployment = dict(collections.Counter(row['family'] for row in plan['aliases'].values()))
    capacities = {family: catalog['capacity_factors'][family][str(n)] for family,n in deployment.items()}
    if any(not finite_positive(g) for g in capacities.values()):
        raise ValueError('Positive original g(n) required')
    observations = [row for r in selected for row in r['observations']]
    groups = collections.defaultdict(list)
    for row in observations:
        if not finite_positive(row['worker_service_ms']):
            raise ValueError('Finite positive training worker service required')
        groups[(row['family'], row['task'], row['pool_id'])].append(row)
    estimates, normalized = [], {}
    rho = {family: 0. for family in deployment}
    for target in input_weights(plan, candidate):
        family, task = target['family'], target['task']
        strata, missing = [], []
        for pool, weight in target['weights'].items():
            rows = groups[(family, task, pool)]
            if not rows:
                missing.append(pool)
            strata.append({'pool_id': pool, 'target_weight': weight, 'training_request_count': len(rows),
                'training_alias_counts': dict(collections.Counter(r['alias'] for r in rows)),
                'training_worker_mean_ms': statistics.mean(r['worker_service_ms'] for r in rows) if rows else None})
        known = bool(target['weights']) and not missing and bool(selected)
        weighted = sum(s['target_weight']*s['training_worker_mean_ms'] for s in strata) if known else None
        pooled_rows = [r for r in observations if (r['family'],r['task']) == (family,task)]
        pooled = statistics.mean(r['worker_service_ms'] for r in pooled_rows) if pooled_rows else None
        factor = capacities[family]/deployment[family]
        service = weighted*factor if known else None
        normalized.setdefault(family,{})[task] = service
        if service is None:
            rho[family] = None
        elif rho[family] is not None:
            rho[family] += target['offered_family_rps']*service/1000/capacities[family]
        estimates.append({**target, 'status': 'ESTIMATED' if known else 'UNKNOWN',
            'unknown_reason': None if known else ('NO_MATCHED_TRAINING_CONDITION' if not selected else
                'NO_TARGET_REQUESTS_FOR_POSITIVE_OFFERED_ROUTE' if not target['weights'] else 'MISSING_POSITIVE_WEIGHT_INPUT_GROUP'),
            'missing_inputs': missing, 'strata': strata,
            'known_target_weight_mass': sum(s['target_weight'] for s in strata if s['training_worker_mean_ms'] is not None),
            'weighted_worker_service_ms': weighted, 'pooled_training_worker_mean_ms': pooled,
            'family_normalized_service_ms': service, 'original_g_n': capacities[family],
            'replicas': deployment[family], 'normalization_factor': factor})
    complete = all(row['status']=='ESTIMATED' for row in estimates)
    return {'schema':VERSION, 'condition_sha256':digest(spec), 'condition':spec,
        'status':'ESTIMATED' if complete else 'UNKNOWN', 'estimates':estimates,
        'training_flow_ids':[r['flow_id'] for r in selected],
        'training_raw_receipts':[r['raw_source'] for r in selected],
        'training_seeds':sorted({r['seed'] for r in selected}),
        'service_time_ms':normalized, 'rho_by_family':rho,
        'rho_threshold_satisfied':all(v<=RHO_MAX+RHO_TOLERANCE for v in rho.values()) if complete else None,
        'threshold':RHO_MAX, 'tolerance':RHO_TOLERANCE, 'capacity_factors':capacities,
        'original_capacity_factors':copy.deepcopy(catalog['capacity_factors']),
        'target_trace_sha256':digest(plan['trace']), 'target_outcomes_used':False,
        'kind':'input_standardized_training_condition_estimate', 'physical_capacity_bound':False,
        'target_applicability_validated':None, 'original_rho_admission_passed':None,
        'original_capacity_and_stability_passed':None, 'final_gain_eligible':False}


def evaluate(prediction, heldout_observations):
    """Heldout service is evaluation only; the estimate remains immutable."""
    rows = []
    for item in prediction['estimates']:
        family, task = item['family'], item['task']
        actual = [r['worker_service_ms'] for r in heldout_observations if (r['family'],r['task']) == (family,task)]
        observed = statistics.mean(actual) if actual else None
        weighted, pooled = item['weighted_worker_service_ms'], item['pooled_training_worker_mean_ms']
        rows.append({'family':family,'task':task,'actual_request_count':len(actual),
            'observed_worker_mean_ms':observed, 'weighted_worker_mean_ms':weighted,
            'pooled_worker_mean_ms':pooled,
            'weighted_mean_relative_error':abs(weighted-observed)/observed if weighted is not None and observed else None,
            'pooled_mean_relative_error':abs(pooled-observed)/observed if pooled is not None and observed else None,
            'weighted_signed_error_ms':weighted-observed if weighted is not None and observed else None,
            'pooled_signed_error_ms':pooled-observed if pooled is not None and observed else None,
            'unknown_reason':item['unknown_reason'], 'missing_inputs':item['missing_inputs']})
    rho_actual = {f:0. for f in prediction['capacity_factors']}
    for row, target in zip(rows,prediction['estimates']):
        f = row['family']
        if row['observed_worker_mean_ms'] is None:
            rho_actual[f] = None
        elif rho_actual[f] is not None:
            rho_actual[f] += target['offered_family_rps']*row['observed_worker_mean_ms']/1000/target['replicas']
    return {'groups':rows,'observed_service_rho_diagnostic':rho_actual,
        'observed_service_is_capacity_proof':False,'independent_validation':False,'final_gain_eligible':False}


def crossvalidate(suite_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    bundle = load_training(suite_path)
    refs = bundle['source_receipts']
    protocol = {'schema':'gsdeploy.input_weighted_capacity_training_cv_policy.v1',
        'created_ns':time.perf_counter_ns(), 'policy':POLICY, 'suite':receipt(suite_path),
        'CV_unit':'Complete seed held out across all four routes', 'target_weights_known_before_evaluation':True,
        'no_independent_results_read':True, 'GPU_started':False}
    cases = []
    for seed in bundle['seeds']:
        training = [r for r in bundle['runs'] if r['seed']!=seed]
        heldouts = [r for r in bundle['runs'] if r['seed']==seed]
        for heldout in heldouts:
            if any(r['raw_source']['sha256']==heldout['raw_source']['sha256'] for r in training):
                raise ValueError('Heldout raw leaked into training')
            plan = heldout['plan']
            candidate = next(c for c in plan['candidates'] if c['id']==heldout['candidate_id'])
            catalog = read(plan['catalog']['path'])
            predicted = estimate(plan,candidate,training,catalog)
            cases.append({'seed':seed,'flow_id':heldout['flow_id'],'candidate_id':heldout['candidate_id'],
                'heldout_raw':heldout['raw_source'],'heldout_plan':heldout['plan_source'],
                'training_flow_ids':[r['flow_id'] for r in training], 'prediction':predicted,
                'evaluation':evaluate(predicted,heldout['observations'])})
    groups = [g for case in cases for g in case['evaluation']['groups']]
    paired = [g for g in groups if g['weighted_mean_relative_error'] is not None and g['pooled_mean_relative_error'] is not None]
    summary = {'requested_groups':len(groups),'known_weighted_groups':sum(g['weighted_mean_relative_error'] is not None for g in groups),
        'unknown_weighted_groups':sum(g['weighted_mean_relative_error'] is None for g in groups),
        'paired_groups':len(paired),
        'paired_weighted_macro_mean_relative_error':statistics.mean(g['weighted_mean_relative_error'] for g in paired) if paired else None,
        'paired_pooled_macro_mean_relative_error':statistics.mean(g['pooled_mean_relative_error'] for g in paired) if paired else None,
        'weighted_better_groups':sum(g['weighted_mean_relative_error']<g['pooled_mean_relative_error'] for g in paired),
        'weighted_worse_groups':sum(g['weighted_mean_relative_error']>g['pooled_mean_relative_error'] for g in paired)}
    for ref in refs:
        verify(ref)
    report = {'schema':'gsdeploy.input_weighted_capacity_training_cv_report.v1','created_ns':time.perf_counter_ns(),
        'policy':protocol,'source_receipts':refs,'cases':cases,'summary':summary,
        'training_only':True,'retrospective_training_cross_validation':True,
        'independent_validation':False,'independent_results_read':False,'GPU_started':False,
        'original_failures_preserved':True,'qualification_passed':False,'final_gain_eligible':False}
    output.mkdir(parents=True,exist_ok=False)
    save(output/'policy.json',protocol)
    save(output/'report.json',report)
    draft = {'schema':'gsdeploy.input_weighted_capacity_prospective_qualification_draft.v1',
        'kind':'DRAFT_NOT_A_FROZEN_TARGET_QUALIFICATION','created_ns':time.perf_counter_ns(),
        'training_cv_report':receipt(output/'report.json'),'estimator_source':receipt(__file__),
        'target_plan':None,'target_seal':None,'independent_runs':None,
        'required_before_new_measurement':['Freeze estimator, original g(n)/rho/.8 and all sources',
            'Publish exact independent target plan/trace/routes; compute finite-input weights without outcomes',
            'Require every positive-weight group supported by matching training condition',
            'Freeze run count and per-group mean-service acceptance threshold before startup',
            'Recheck seal precedes every target startup and verify unchanged native stability',
            'Keep original measured-profile resolver UNKNOWN; report transfer qualification separately'],
        'unchanged_rho_threshold':RHO_MAX,'unchanged_rho_tolerance':RHO_TOLERANCE,
        'acceptance_policy':None,'physical_capacity_bound':False,'uncertainty_calibrated':False,
        'horizon_transfer_allowed':False,'resource_transfer_allowed':False,
        'failure_semantics':'Any missing input group or failed service/stability check is UNKNOWN/FAIL; no renormalization or old-failure override',
        'original_failures_preserved':True,'qualification_passed':False,'final_gain_eligible':False,
        'limits':['Input reweighting corrects sampled composition only, not workload-dependent conditional drift',
            'Family-average rho cannot detect same-family alias skew capacity bottlenecks',
            'Finite cohort including bounded drain is not a stationary service capacity guarantee',
            'Two training seeds do not calibrate a run-level uncertainty bound']}
    save(output/'prospective-qualification-draft.json',draft)
    return report


def seal(plan_path,suite_path,output):
    """Prospective public-plan estimate; no target raw or outcomes argument."""
    if Path(output).exists():
        raise FileExistsError(output)
    plan = _plan(plan_path)
    bundle = load_training(suite_path)
    if plan['seed'] in bundle['seeds']:
        raise ValueError('New target seed must be disjoint from training')
    catalog = read(plan['catalog']['path'])
    predictions = [{'candidate_id':c['id'],'prediction':estimate(plan,c,bundle['runs'],catalog)} for c in plan['candidates']]
    refs = [receipt(plan_path),*[plan[k] for k in ('catalog','context','profile_plan','runner_source')],*bundle['source_receipts']]
    for ref in refs:
        verify(ref)
    value = {'schema':'gsdeploy.prospective_input_weighted_capacity_seal.v1','created_ns':time.perf_counter_ns(),
        'target_plan':receipt(plan_path),'source_receipts':refs,'policy':POLICY,'candidates':predictions,
        'target_outcomes_used':False,'GPU_started':False,'target_applicability_validated':None,
        'acceptance_policy':None,'original_measured_profile_qualification_passed':None,'final_gain_eligible':False}
    value['seal_sha256']=digest(value)
    save(output,value)
    return value


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op',required=True)
    cv = sub.add_parser('crossvalidate')
    cv.add_argument('--suite',required=True); cv.add_argument('--output',required=True)
    prospective = sub.add_parser('seal')
    prospective.add_argument('--suite',required=True); prospective.add_argument('--plan',required=True)
    prospective.add_argument('--output',required=True)
    args = parser.parse_args()
    result = crossvalidate(args.suite,args.output) if args.op=='crossvalidate' else seal(args.plan,args.suite,args.output)
    print(json.dumps(result.get('summary',{'candidates':len(result['candidates'])}) if args.op=='seal' else result['summary']))
