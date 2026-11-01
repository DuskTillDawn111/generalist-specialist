"""Startup transfer verified against the fresh-ID v11 collector.

The immutable v5 training model is rederived by the original v6 verifier.
The target must bind the actual v11 collector, complete unchanged priming,
original shared execution and native projection. This is still a conditional
transfer hypothesis, not an independent accuracy result.
"""
from __future__ import annotations
from pathlib import Path

from . import known_input_native_flow_v11 as flow
from . import startup_prediction_v5 as original
from . import startup_prediction_v6 as verification
from .startup_prediction_v6 import read, verify_model, public_catalog, priming_content
from .native_collect import receipt
from .native_execution_projection_v5 import native_execution_projection
from .operator_profile_campaign_v2 import stat_receipt
from .profiling import RESIDENTS, digest

SCHEMA = 'gsdeploy.flow_bound_premeasurement_startup_prediction.v11'


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
    implementations = [receipt(__file__), receipt(original.__file__), receipt(verification.__file__), receipt(flow.__file__),
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

