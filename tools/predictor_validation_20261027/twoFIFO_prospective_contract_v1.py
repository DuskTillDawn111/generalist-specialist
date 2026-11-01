"""Freeze an existing public flow and audit planned native-template support.

CPU preparation only: historical request IDs, input contents, routes, rates and
windows are retained. This is neither a fresh execution plan nor a prediction
seal. Planned training coverage is never promoted to fitted-model support.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from .flow_validation import validate_plan
from .kernel_profile_campaign_v2 import input_identity
from .profiling import digest
from .stage_response_features_v1 import receipt, checked

SCHEMA = 'gsdeploy.twoFIFO_prospective_public_contract.v1'
OBJECTIVE = 'gsdeploy.objective_contract.v1'
CAMPAIGN = 'gsdeploy.twoFIFO_source_rebound_campaign.v1'
FIELDS = ('aliases', 'workload_rps', 'latency_reference_ms', 'weights',
    'memory_budget_mib', 'energy_reference_watts', 'memory_safe_factor',
    'startup_amortization_seconds', 'window_seconds', 'drain_seconds',
    'warmup_seconds', 'baseline_candidate_id', 'queue_sample_offsets_ns')


def need(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    return receipt(path)


def public_contract(plan, objective):
    """Project only public fields; never re-generate arrivals or choose routes."""
    projected = copy.deepcopy(plan)
    need(projected.get('schema') in ('gsdeploy.native_flow_plan.v1',
        'gsdeploy.native_known_input_flow_plan.v5'), 'Existing public native-flow plan required')
    if 'plan_sha256' in projected:
        need(projected['plan_sha256'] == digest({k:v for k,v in projected.items()
            if k != 'plan_sha256'}), 'Original public plan hash changed')
    projected['schema'] = 'gsdeploy.native_flow_plan.v1'
    projected.pop('plan_sha256', None)
    validate_plan(projected)
    need(objective.get('schema') == OBJECTIVE
        and objective.get('weights') == plan['weights'] == {k:1/6 for k in 'QTLPME'}
        and objective.get('stability_policy') == 'existing_thresholds_unchanged'
        and plan['startup_amortization_seconds'] == 3600.,
        'Original equal six-term objective and unchanged stability required')
    need([r['offset_ns'] for r in plan['trace']] == sorted(r['offset_ns'] for r in plan['trace']),
        'Chronological original arrival order required')
    result = {k:copy.deepcopy(plan[k]) for k in FIELDS}
    for key in ('arrival_process', 'arrival_rate_phases', 'validation_thresholds',
                'utility_tie_tolerance', 'queue_depth_boundary', 'startup_comparison_policy'):
        if key in plan:
            result[key] = copy.deepcopy(plan[key])
    result['trace'] = copy.deepcopy(plan['trace'])
    result['candidates'] = copy.deepcopy(plan['candidates'])
    result['source_execution_context_sha256'] = plan['execution_context_sha256']
    result['task_counts'] = dict(Counter(r['task'] for r in plan['trace']))
    result['resident_owner_count'] = len(plan['aliases'])
    result['positive_rate_tasks_without_requests'] = sorted(set(plan['workload_rps'])
        - {r['task'] for r in plan['trace']})
    return result


def candidate_calendar(contract, candidate):
    """Native queue/scorer rows for the entire offered cohort, in source order."""
    rows = [dict(request_id=r['request_id'], owner=candidate['routes'][r['request_id']],
        task=r['task'], pool_id=r['pool_id'], arrival_ns=r['offset_ns'])
        for r in contract['trace']]
    return dict(candidate_id=candidate['id'], public_requests=rows,
        original_window=[0, round(contract['window_seconds']*1e9)],
        original_drain_end_ns=round((contract['window_seconds']+contract['drain_seconds'])*1e9),
        resource_configuration=copy.deepcopy(candidate.get('resource_configuration')),
        queue_sample_offsets_ns=contract['queue_sample_offsets_ns'],
        historical_source_context=candidate['execution_context_sha256'],
        current_backend_context_authenticated=False)


def planned_support(contract, plans, declarations):
    """Every required owner/task/exact-input must occur in every donor block.

    Only measured case calendars supply candidate request vectors; priming and
    merely resident supported_tasks do not supply missing image/ASR templates.
    """
    blocks = sorted({p['capture_id'] for p in plans})
    need(blocks, 'Nonempty planned donor blocks required')
    counts = {block:Counter() for block in blocks}
    for plan in plans:
        need(plan.get('split') == 'training', 'Public training plans required')
        for case in plan['cases']:
            for row in case['arrivals']:
                key = row['owner'], row['task'], row['pool_id']
                need(row['pool_id'] in declarations
                    and declarations[row['pool_id']]['task'] == row['task'],
                    'Planned training row differs from exact context input')
                counts[plan['capture_id']][key] += 1
    required = sorted({(c['routes'][r['request_id']], r['task'], r['pool_id'])
        for c in contract['candidates'] for r in contract['trace']})
    rows = []
    for owner, task, pool in required:
        declared = declarations.get(pool)
        count = {b:counts[b][owner, task, pool] for b in blocks}
        covered = declared is not None and declared['task'] == task and all(count.values())
        rows.append(dict(owner=owner, task=task, pool_id=pool,
            planned_measured_count_by_block=count,
            training_exact_input_sha256=declared['exact_input_sha256'] if declared else None,
            status='PLANNED_IN_EVERY_BLOCK_NOT_MODEL_QUALIFIED' if covered else
                'MISSING_EXACT_NATIVE_TEMPLATE_SUPPORT',
            fitted_model_support_authenticated=False))
    return dict(block_ids=blocks, required_keys=rows,
        required_key_count=len(rows), missing_key_count=sum(r['status'].startswith('MISSING') for r in rows),
        priming_used_as_measured_template=False, declared_task_support_used_as_template=False,
        planned_coverage_is_not_fitted_support=True)


def public_input_manifest(catalog_ref, contract, declarations):
    """Re-derive each selected exact content identity from the original catalog."""
    need(receipt(catalog_ref['path']) == catalog_ref, 'Public catalog changed')
    needed = {r['pool_id']:r['task'] for r in contract['trace']}
    need(len({(r['pool_id'], r['task']) for r in contract['trace']}) == len(needed),
        'One pool identity cannot denote multiple public tasks')
    inputs = []
    seen = set()
    for line in Path(catalog_ref['path']).read_text().splitlines():
        row = json.loads(line)
        if row['pool_id'] not in needed:
            continue
        pool = row['pool_id']
        need(pool not in seen and row['task_type'] == needed[pool], 'Unique matching public input required')
        seen.add(pool)
        exact = input_identity(row)
        item = dict(pool_id=pool, task=row['task_type'], exact_input_sha256=exact)
        if row['task_type'] != 'text':
            item['input_file'] = receipt(row['sample']['input'])
        inputs.append(item)
        if pool in declarations:
            need(declarations[pool]['task'] == row['task_type']
                and declarations[pool]['exact_input_sha256'] == exact,
                'Same pool ID changed exact public input content')
    need(seen == set(needed), 'Complete original public input contents required')
    need(receipt(catalog_ref['path']) == catalog_ref, 'Public catalog changed during read')
    return inputs


def prepare(objective_path, public_plan_path, campaign_path, public_catalog_path, output):
    refs = {k:receipt(p) for k,p in dict(objective=objective_path, public_plan=public_plan_path,
        training_campaign=campaign_path, public_catalog=public_catalog_path).items()}
    objective, plan, campaign = (checked(refs[k]) for k in ('objective','public_plan','training_campaign'))
    contract = public_contract(plan, objective)
    need(campaign.get('schema') == CAMPAIGN and len(campaign.get('attempts', [])) == 12,
        'Explicit complete source-rebound twelve-plan public campaign required')
    context = checked(campaign['context'])
    plans = [checked(r) for r in campaign['attempts']]
    need(all(p['context'] == campaign['context'] for p in plans), 'Same rebound training context required')
    need(len({p['attempt_id'] for p in plans}) == 12
        and sorted(Counter(p['capture_id'] for p in plans).values()) == [4,4,4],
        'Distinct twelve attempts across three whole four-shard blocks required')
    declarations = {r['pool_id']:r for r in context['declared_inputs']}
    need(len(declarations) == len(context['declared_inputs']), 'Distinct declared input identities required')
    inputs = public_input_manifest(refs['public_catalog'], contract, declarations)
    support = planned_support(contract, plans, declarations)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = dict(public_contract=save(out/'public-contract.json', contract),
        public_inputs=save(out/'public-inputs.json', inputs),
        planned_support=save(out/'planned-support.json', support))
    calendars = {c['id']:save(out/('calendar-'+c['id']+'.json'), candidate_calendar(contract,c))
        for c in contract['candidates']}
    modules = ('known_input_gate_v5.py', 'twoFIFO_candidate_ranking_v1.py',
        'macro_six_term_bridge_v1.py', 'actual_worker_world_score_v3.py')
    report = dict(schema=SCHEMA, status='CPU_PUBLIC_CONTRACT_FROZEN_HOLD_NOT_PREDICTION_SEAL',
        inputs=refs, training_context=campaign['context'], training_plans=campaign['attempts'],
        artifacts=artifacts, candidate_calendars=calendars,
        preparer=receipt(__file__), reuse_interfaces={Path(m).stem:receipt(Path(__file__).with_name(m)) for m in modules},
        task_counts=contract['task_counts'], resident_owner_count=contract['resident_owner_count'],
        required_native_keys=support['required_key_count'], missing_native_keys=support['missing_key_count'],
        validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
        unseen_input_generalization_claimed=False,
        original_arrivals_inputs_routes_rates_windows_and_normalizers_retained=True,
        original_objective_and_stability_retained=True, historical_request_ids_retained=True,
        public_contract_frozen=True, fresh_execution_plan_created=False, predictions_sealed=False,
        numerical_fit_run=False, validation_labels_read=False, GPU_started=False,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        readiness_blockers=[
            'Historical public plan is a comparison reference; fresh execution/request identities are not sealed',
            'Original public backend/context has no authenticated bridge to source-rebound native backend',
            'Planned exact-input support is not actual qualified fitted request-vector support',
            'Full-horizon templates, transport, joint draws and startup/history/resource evidence require authentication',
            'Complete independent native quality and original stability/drain evidence are absent']
            + (['Missing exact owner/task/input support in training blocks'] if support['missing_key_count'] else [])
            + (['Positive-rate tasks missing from offered cohort'] if contract['positive_rate_tasks_without_requests'] else []),
        report_entry=dict(point_gate='known_input_gate_v5.evaluate_prediction_gate',
            paired_world_ranker='twoFIFO_candidate_ranking_v1.rank_candidates',
            checks=['complete_cohort', 'service_E2E', 'each_task_P95', 'queue_depth',
                'six_term_delta_U_sign', 'all_candidate_pair_ranks'],
            finite_world_projection_policy_predeclared=False,
            mean_world_P95_is_not_mixture_P95=True,
            unknown_worlds_or_terms_are_not_deleted=True,
            qualification_is_separate_from_candidate_ranking=True))
    save(out/'report.json', report)
    audit(out/'report.json')
    return report


def audit(path):
    """Recheck preparation bytes and public arithmetic; no fitted/validation data."""
    report = checked(receipt(path))
    need(report.get('schema') == SCHEMA
        and report.get('status') == 'CPU_PUBLIC_CONTRACT_FROZEN_HOLD_NOT_PREDICTION_SEAL'
        and all(report.get(k) is False for k in ('fresh_execution_plan_created',
            'predictions_sealed','numerical_fit_run','validation_labels_read','GPU_started',
            'main_queue_provider_eligible','final_gain_eligible')),
        'CPU public-only HOLD contract required')
    for ref in [report['preparer'], *report['reuse_interfaces'].values()]:
        need(receipt(ref['path']) == ref, 'Preparation/interface source changed')
    inputs = report['inputs']
    objective, plan, campaign = (checked(inputs[k]) for k in ('objective','public_plan','training_campaign'))
    need(receipt(inputs['public_catalog']['path']) == inputs['public_catalog'], 'Public catalog changed')
    need(report['training_context'] == campaign['context']
        and report['training_plans'] == campaign['attempts'], 'Training public lineage changed')
    context = checked(report['training_context'])
    plans = [checked(ref) for ref in report['training_plans']]
    contract = public_contract(plan, objective)
    need(checked(report['artifacts']['public_contract']) == contract, 'Original public contract changed')
    declarations = {r['pool_id']:r for r in context['declared_inputs']}
    support = planned_support(contract, plans, declarations)
    need(checked(report['artifacts']['planned_support']) == support, 'Planned support matrix changed')
    need(checked(report['artifacts']['public_inputs']) ==
        public_input_manifest(inputs['public_catalog'], contract, declarations),
        'Complete original public input manifest changed')
    need(report['task_counts'] == contract['task_counts']
        and report['resident_owner_count'] == contract['resident_owner_count']
        and report['required_native_keys'] == support['required_key_count']
        and report['missing_native_keys'] == support['missing_key_count'],
        'Public task/owner/support report summary changed')
    need(set(report['candidate_calendars']) == {c['id'] for c in contract['candidates']},
        'Complete candidate domain required')
    for candidate in contract['candidates']:
        need(checked(report['candidate_calendars'][candidate['id']]) == candidate_calendar(contract,candidate),
            'Complete original arrival/input/route cohort changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('objective','public-plan','campaign','public-catalog','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = prepare(args.objective, args.public_plan, args.campaign, args.public_catalog, args.output)
    print(json.dumps({k:report[k] for k in ('status','task_counts','required_native_keys','missing_native_keys')}))


if __name__ == '__main__':
    main()
