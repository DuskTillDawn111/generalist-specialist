"""Minimal missing full-state natural resource training; preparation is CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import shutil

from .native_collect import context_identity, receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .operator_profile_campaign_v2 import audit as source_audit
from .profiling import RESIDENTS, digest
from .smoke_verifier_v3 import check_receipt
from .startup_collect_v4 import collect

PLAN_SCHEMA = 'gsdeploy.generic_full_state_resource_training_plan.v5'
RESOURCE_SCHEMA = 'gsdeploy.natural_direct_state_resource_hypothesis.v5'
ALIASES = {a: dict(family=f) for a, f in RESIDENTS.items()}


def state_key(active, aliases):
    counts = Counter((aliases[alias]['family'], row['task']) for alias, row in active.items())
    members = [dict(family=f, task=t, count=n) for (f, t), n in sorted(counts.items())]
    return digest(members), members


def universe():
    states = []
    for text_count, gtask, ocr in itertools.product(range(3), (None, 'text', 'image', 'asr'), (False, True)):
        tasks = dict.fromkeys(list(RESIDENTS)[:text_count], 'text')
        if gtask:
            tasks['generalist_2#0'] = gtask
        if ocr:
            tasks['ocr#0'] = 'ocr'
        key, members = state_key({a: dict(task=t) for a, t in tasks.items()}, ALIASES)
        states.append(dict(key=key, members=members, owner_tasks=tasks))
    return states


def design(model, primers):
    if model.get('schema') != RESOURCE_SCHEMA:
        raise ValueError('Actual full-state natural resource model required')
    states = universe()
    missing = [s for s in states if not model.get('states', {}).get(s['key'], {}).get('known', False)]
    by_task = {}
    for task in ('text', 'image', 'asr', 'ocr'):
        selected = sorted((p for p in primers if p['task'] == task), key=lambda p: p['input_size'])
        if len(selected) != 2 or len({p['pool_id'] for p in selected}) != 2:
            raise ValueError('Exactly inherited short/long training primers per task required')
        by_task[task] = selected
    jobs = []
    for state in missing:
        if not state['owner_tasks']:
            raise ValueError('Idle state requires actual inter-job observation, not an empty job')
        for stratum in range(2):
            tasks = state['owner_tasks']
            job = dict(context='resource-full-state-' + state['key'][:12],
                inputs={a: by_task[t][stratum]['pool_id'] for a, t in tasks.items()},
                tasks=tasks, start_offsets_ms=dict.fromkeys(tasks, 0),
                resource_target_state_key=state['key'], resource_target_members=state['members'],
                input_stratum='short' if stratum == 0 else 'long',
                split='training', validation_view='TRAINING_ONLY_NOT_VALIDATION',
                phase='full-state-natural-resource-training')
            job['id'] = digest(job)
            jobs.append(job)
    return dict(universe=states, covered=[s for s in states if s not in missing], missing=missing, jobs=jobs)


def prepare(reference_run, resource_model_path, output):
    reference_run, output = Path(reference_run).resolve(), Path(output).resolve()
    resource_model_path = Path(resource_model_path).resolve()
    source_audit(reference_run)
    plan = json.loads((reference_run / 'plan.json').read_text())
    context = json.loads((reference_run / 'context.json').read_text())
    model = json.loads(resource_model_path.read_text())
    if plan['domain'] != 'natural' or context['collection_budget'] != dict(free_disk_reserve_bytes=12 * 2**30, total_timeout_seconds=2400):
        raise ValueError('Frozen natural reference and unchanged collection budget required')
    if not model.get('sources') or not all(check_receipt(r) for r in model['sources']):
        raise ValueError('Changed resource model training sources')
    original_projection = native_execution_projection(context)
    if original_projection['projection_sha256'] != model['native_projection_sha256']:
        raise ValueError('Resource training model and actual reference projection differ')
    selected = design(model, plan['inputs'])
    output.mkdir(parents=True, exist_ok=False)
    # Freeze new wrapper/audit imports; keep existing large snapshots as refs.
    existing = {s['original_path'] for s in context['execution_sources']}
    extras = [Path(__file__).resolve()] + [Path(__file__).with_name(name) for name in (
        'native_execution_projection_v5.py', 'kernel_baseline_intake_v3.py',
        'kernel_domain_bridge_review_v3.py')]
    for index, source in enumerate(extras):
        source = source.resolve()
        if str(source) in existing:
            continue
        frozen = output / 'source-snapshot' / str(index) / source.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(source, frozen)
        extra = dict(original_path=str(source), frozen=receipt(frozen))
        context['execution_sources'].append(extra)
        context['sources'].append(extra['frozen'])
    context['resource_training'] = dict(schema=PLAN_SCHEMA, resource_model=receipt(resource_model_path),
        intended_missing_state_count=len(selected['missing']),
        simultaneous_launch_assumption='Zero planned offsets maximize possible full-state overlap; actual handler overlap and interior samples still required.')
    projection = native_execution_projection(context)
    if projection['projection_sha256'] != original_projection['projection_sha256']:
        raise ValueError('New wrapper must preserve explicit natural native projection')
    plan.update(schema=PLAN_SCHEMA, execution_context_sha256=context_identity(context),
        shard='generic-full-state-resources', jobs=selected['jobs'], paired_jobs_sha256=digest(selected['jobs']),
        resource_training_model=receipt(resource_model_path),
        claim='NATURAL_FULL_STATE_RESOURCE_TRAINING_ONLY_NOT_SERVICE_SLOWDOWN_OR_VALIDATION')
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = digest(plan)
    save(output / 'coverage-design.json', selected)
    captures = {}
    for name in ('capture01', 'capture02'):
        run = output / name / 'natural'
        run.mkdir(parents=True)
        save(run / 'context.json', context)
        save(run / 'plan.json', plan)
        audit(run)
        captures[name] = dict(run=str(run), plan=receipt(run / 'plan.json'), context=receipt(run / 'context.json'))
    free = shutil.disk_usage(output).free
    report = dict(schema='gsdeploy.generic_full_state_resource_training_campaign.v5',
        status='PREPARED_CPU_ONLY', reference_plan=receipt(reference_run / 'plan.json'),
        reference_context=receipt(reference_run / 'context.json'), resource_model=receipt(resource_model_path),
        coverage_design=receipt(output / 'coverage-design.json'), captures=captures,
        total_universe_states=len(selected['universe']), previously_supported_states=len(selected['covered']),
        targeted_missing_states=len(selected['missing']), jobs_per_capture=len(selected['jobs']),
        requests_per_capture=sum(len(j['inputs']) for j in selected['jobs']),
        projection=projection, original_reference_projection=original_projection,
        source_snapshot_copied=False, prior_raw_copied=False,
        priming_inputs_unchanged=plan['inputs'] == json.loads((reference_run / 'plan.json').read_text())['inputs'],
        original_four_physical_owners=True, independent_capture_count=0,
        budget=dict(free_disk_bytes_at_preparation=free, free_disk_reserve_bytes=12 * 2**30,
            total_timeout_seconds_per_capture=2400, estimated_additional_bytes_two_captures=64 * 2**20,
            remaining_after_reserve_and_size_scenario_bytes=free - 12 * 2**30 - 64 * 2**20,
            previous_natural_startup_seconds=66.7, previous_priming_seconds=8.6,
            scenario_seconds_per_capture=180, scenario_is_upper_bound=False,
            enforcement='Inherited frozen collector checks at phase/job boundaries; no continuous watchdog.'),
        minimum_captures_per_actual_state=2, offsets_ms=0,
        scope='Direct resource-state training. No sum of solo/pair states; no profiler or slowdown model.',
        observed_state_gate='Each target needs genuine simultaneous handlers and interior NVML samples in both independent raw captures. Missing state stays UNKNOWN; no automatic retry or coverage claim.',
        GPU_started=False, validation_started=False)
    save(output / 'campaign.json', report)
    return report


def audit(run):
    run = Path(run).resolve()
    result = source_audit(run)
    plan = json.loads((run / 'plan.json').read_text())
    context = json.loads((run / 'context.json').read_text())
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('split') != 'training'
            or plan.get('domain') != 'natural' or plan.get('shard') != 'generic-full-state-resources'
            or context.get('measurement_domain') != 'natural' or context.get('affine_observation') is not False
            or context.get('collection_budget') != dict(free_disk_reserve_bytes=12 * 2**30, total_timeout_seconds=2400)):
        raise ValueError('Explicit generic natural resource-training plan and inherited budgets required')
    if not check_receipt(plan['resource_training_model']):
        raise ValueError('Changed resource coverage model')
    model = json.loads(Path(plan['resource_training_model']['path']).read_text())
    if model['native_projection_sha256'] != native_execution_projection(context)['projection_sha256']:
        raise ValueError('Native resource projection mismatch')
    expected = design(model, plan['inputs'])['jobs']
    if plan['jobs'] != expected or plan['paired_jobs_sha256'] != digest(expected):
        raise ValueError('Exact declared missing-state/stratum jobs required')
    unhashed = copy.deepcopy(plan)
    recorded = unhashed.pop('plan_sha256')
    if recorded != digest(unhashed):
        raise ValueError('Resource plan digest mismatch')
    return dict(result, jobs=len(expected), schema=PLAN_SCHEMA)


def execute(run):
    run = Path(run).resolve()
    audit(run)
    if shutil.disk_usage(run).free < 12 * 2**30:
        raise ValueError('Original 12 GiB free disk reserve insufficient')
    raw = collect(run, trace=False)
    return dict(status=raw['status'], raw=receipt(run / 'raw.json'), requests=len(raw['request_records']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--reference-run', required=True)
    p.add_argument('--resource-model', required=True)
    p.add_argument('--output', required=True)
    for operation in ('audit', 'execute'):
        p = sub.add_parser(operation)
        p.add_argument('--run', required=True)
        if operation == 'execute':
            p.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    result = prepare(args.reference_run, args.resource_model, args.output) if args.op == 'prepare' else audit(args.run) if args.op == 'audit' else execute(args.run)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
