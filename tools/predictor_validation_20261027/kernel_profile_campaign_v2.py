"""Original four-owner kernel profiling campaign; preparation is CPU only.

Execution delegates to the existing audited retained-native collector. All output
is exclusive, and smoke data are never training or validation observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from . import background_collect as collector
from .native_collect import ROOT, context_identity, receipt, save
from .profiling import CONTEXTS, RESIDENTS, digest
from .trace_export import export_sqlite, parse_sqlite

TASKS = ('text', 'image', 'ocr', 'asr')


def input_identity(row):
    sample = row['sample']
    if row['task_type'] == 'text':
        return digest({k: sample[k] for k in ('context', 'question')})
    return hashlib.sha256(Path(sample['input']).read_bytes()).hexdigest()


def choose_inputs(rows):
    """Separate exact content, with both partitions spanning short and long input."""
    result = {'training': {}, 'validation': {}}
    for task in TASKS:
        items = []
        seen = set()
        for row in rows:
            if row['task_type'] != task:
                continue
            sha = input_identity(row)
            if sha in seen:
                continue
            seen.add(sha)
            sample = row['sample']
            size = (len(sample['context']) + len(sample['question']) if task == 'text'
                    else sample.get('duration_s', os.path.getsize(sample['input'])))
            items.append(dict(task=task, pool_id=row['pool_id'], input_size=size,
                              exact_input_sha256=sha))
        items.sort(key=lambda x: (x['input_size'], x['pool_id']))
        if len(items) < 8:
            raise ValueError('Eight unique inputs required for ' + task)
        # Short and long, two nearby observations per stratum in each partition.
        indices = [0, 1, 2, 3, len(items)-4, len(items)-3, len(items)-2, len(items)-1]
        for split, picks in [('training', (0, 2, 4, 6)), ('validation', (1, 3, 5, 7))]:
            result[split][task] = [items[indices[i]] for i in picks]
        if {r['exact_input_sha256'] for r in result['training'][task]} & {
                r['exact_input_sha256'] for r in result['validation'][task]}:
            raise ValueError('Input content overlap')
    return result


def make_jobs(selection, split, seed, repetitions=2):
    jobs = []
    views = [('known_input_independent_trial', selection['training'])] if split == 'validation' else [('training', selection['training'])]
    if split == 'validation':
        views.append(('unseen_input', selection['validation']))
    for view, inputs in views:
        repeats = 1 if split == 'validation' else repetitions
        for index in range(4):
            cases = [(name, aliases, 'text') for name, aliases in CONTEXTS.items()]
            cases += [('S1G', ['text#1', 'generalist_2#0'], 'text')]
            for task in ('image', 'asr'):
                cases += [(name+'_'+task, CONTEXTS[name], task) for name in ('G', 'SG', 'SSG', 'SSGO')]
            for name, aliases, gtask in cases:
                tasks = {alias: ('ocr' if alias == 'ocr#0' else gtask if alias == 'generalist_2#0' else 'text') for alias in aliases}
                ids = {a: inputs[tasks[a]][index]['pool_id'] for a in aliases}
                for offset in ((0,) if len(aliases) == 1 else (0, 50, 200)):
                    for order in ([aliases] if offset == 0 else [aliases, list(reversed(aliases))]):
                        for repeat in range(repeats):
                            job = dict(context=name, inputs=ids, tasks=tasks,
                                       start_offsets_ms={a: k*offset for k, a in enumerate(order)},
                                       repeat=repeat, split=split, validation_view=view, order_seed=seed)
                            job['id'] = digest(job)
                            jobs.append(job)
    random.Random(seed).shuffle(jobs)
    return jobs


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'artifacts/static-study-v6-20261021/inputs'
    rows = [json.loads(line) for line in (base/'runtime-inputs.jsonl').read_text().splitlines()]
    selection = choose_inputs(rows)
    workers = json.loads((base/'worker-registry.json').read_text())
    # Include all current package sources: runner imports evaluation/search modules
    # even though the experiment only calls its record constructor.
    paths = set((ROOT/'gsdeploy').rglob('*.py'))
    paths.update(Path(__file__).with_name(n) for n in (
        'kernel_profile_campaign_v2.py', 'background_collect.py', 'native_collect.py',
        'observed_worker.py', 'semantic_worker.py', 'semantic_observer.py', 'profiling.py', 'trace_export.py'))
    paths.update((base/'runtime-inputs.jsonl', base/'worker-registry.json'))
    for family in set(RESIDENTS.values()):
        binding = workers[family]
        paths.add(Path(binding['spec']))
        cmd = binding['command']
        paths.update(Path(cmd[cmd.index('--source-root')+1]).rglob('*.py'))
        spec = json.loads(Path(binding['spec']).read_text())
        if spec.get('adapter_source'):
            paths.add(Path(spec['adapter_source']))
    # Freeze selected payload bytes as well as public JSON metadata.
    chosen = {r['pool_id'] for partition in selection.values() for items in partition.values() for r in items}
    paths.update(Path(r['sample']['input']) for r in rows if r['pool_id'] in chosen and 'input' in r['sample'])
    sources = []
    for index, path in enumerate(sorted(paths)):
        path = path.resolve()
        frozen = output/'source-snapshot'/str(index)/path.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(path, frozen)
        sources.append(dict(original_path=str(path), frozen=receipt(frozen)))
    context = dict(residents=RESIDENTS, sources=[s['frozen'] for s in sources], execution_sources=sources,
        worker_bindings={f: workers[f] for f in sorted(set(RESIDENTS.values()))},
        threads={k:'1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        execution=dict(batch_size=1, backend='retained-native-observed-v1', weight_reuse=False,
                       physical_owners=4, admission='default_shared_no_permit'),
        counters='UNKNOWN:ERR_NVGPUCTRPERM', worker_variant='semantic_worker.py',
        device='RTX4090-49140MiB', task_domain=list(TASKS),
        phase_scope='host call ranges plus separately correlated CUPTI kernels',
        profiler='nsys-cuda-nvtx-profiler-perturbed',
        protocol='kernel_profile_campaign_v2_original_four_owners')
    execution_sha = context_identity(context)
    save(output/'input-selection.json', selection)
    run_receipts = {}
    for split, seed in [('smoke', 2026102701), ('training', 2026102702), ('validation', 2026102703)]:
        run = output/split
        run.mkdir()
        if split == 'smoke':
            inputs = [selection['training']['text'][0], selection['training']['ocr'][0]]
            job = dict(context='SSGO', inputs={a: inputs[a == 'ocr#0']['pool_id'] for a in RESIDENTS},
                       tasks={a: 'ocr' if a == 'ocr#0' else 'text' for a in RESIDENTS},
                       start_offsets_ms=dict.fromkeys(RESIDENTS, 0), repeat=0, split='smoke',
                       validation_view='correctness_clock_only_not_fit', order_seed=seed)
            job['id'] = digest(job)
            jobs = [job]
        else:
            # Only training inputs are warmed, even in validation. No labels are read.
            inputs = [r for rows_ in selection['training'].values() for r in rows_]
            jobs = make_jobs(selection, split, seed)
        plan = dict(schema='gsdeploy.kernel_profile_campaign.v2', execution_context_sha256=execution_sha,
                    resident_aliases=RESIDENTS, inputs=inputs, jobs=jobs, seed=seed,
                    no_retries=True, GPU_started=False, input_selection=receipt(output/'input-selection.json'),
                    claim='PLAN_ONLY_NOT_GPU_EVIDENCE_NOT_UTILITY',
                    warmup_scope='training_inputs_only_excluded_from_fit',
                    permit_release='NOT_APPLICABLE_default_shared_has_no_admission_permit',
                    required_observation=['four_owned_READY_pids', 'actual_START_END', 'parent_receipt',
                        'WORKER_READY', 'CUPTI_kernel_launch_correlation', 'per_process_clock_mapping',
                        'whole_board_power', 'all_owned_sampled_memory_peak', 'UNKNOWN_missing_counters'])
        plan['plan_sha256'] = digest(plan)
        save(run/'context.json', context)
        save(run/'plan.json', plan)
        run_receipts[split] = dict(plan=receipt(run/'plan.json'), context=receipt(run/'context.json'), jobs=len(jobs))
    campaign = dict(schema='gsdeploy.kernel_profile_campaign_index.v2', execution_context_sha256=execution_sha,
                    runs=run_receipts, input_selection=receipt(output/'input-selection.json'),
                    status='PREPARED_CPU_ONLY', blockers=[], GPU_started=False,
                    validation_fit_holdout='known inputs new trials plus exact-content-disjoint inputs; no unseen-task claim',
                    numerical_gate='smoke must pass before training; validation never used to tune training')
    save(output/'campaign.json', campaign)
    return campaign


def audit(run):
    run = Path(run).resolve()
    context = json.loads((run/'context.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    for src in context['execution_sources']:
        if receipt(src['frozen']['path']) != src['frozen'] or receipt(src['original_path'])['sha256'] != src['frozen']['sha256']:
            raise ValueError('Source/input drift: '+src['original_path'])
    if context_identity(context) != plan['execution_context_sha256'] or context['residents'] != RESIDENTS:
        raise ValueError('Execution identity or four-owner mismatch')
    selection = plan['input_selection']
    if receipt(selection['path']) != selection:
        raise ValueError('Input partition drift')
    return dict(status='CPU_SOURCE_AUDIT_PASS', execution_context_sha256=plan['execution_context_sha256'], jobs=len(plan['jobs']))


def execution_gate(run):
    run = Path(run).resolve()
    audit(run)
    if run.name != 'smoke':
        smoke = json.loads((run.parent/'smoke/smoke-report.json').read_text())
        if smoke.get('status') != 'PASS':
            raise ValueError('A passing four-owner correctness/clock smoke is required')
        if run.name == 'validation':
            training = json.loads((run.parent/'training/raw.json').read_text())
            if training.get('status') != 'completed':
                raise ValueError('Training collection must complete before independent validation')
    return run


def anchors_from_records(records, pid):
    starts, anchors = {}, []
    for row in records:
        if row.get('worker_response', {}).get('start_perf_ns') is None:
            continue
        for event in row.get('execution_events', []):
            if 'event' not in event:
                continue
            key = (event['request_id'], event['kind'], event['segment_id'])
            if event['event'] == 'start':
                if key in starts:
                    raise ValueError('Duplicate phase start')
                starts[key] = event['perf_ns']
            elif event['event'] == 'end':
                anchors.append(dict(pid=pid, request_id=key[0], kind=key[1], segment_id=key[2],
                                    start_ns=starts.pop(key), end_ns=event['perf_ns']))
    if starts:
        raise ValueError('Unclosed phase anchors')
    return anchors


def verify_smoke(run):
    """CPU export/verification; subprocess nsys export never starts a GPU model."""
    run = Path(run).resolve()
    audit(run)
    raw = json.loads((run/'raw.json').read_text())
    issues, outputs = [], {}
    if raw.get('status') != 'completed' or raw.get('runtime_errors') or raw.get('telemetry_errors'):
        issues.append('RAW_OR_RUNTIME_OR_TELEMETRY_FAILED')
    if set(raw.get('ready', {})) != set(RESIDENTS) or len({r['pid'] for r in raw.get('ready', {}).values()}) != 4:
        issues.append('FOUR_DISTINCT_OWNER_PIDS_REQUIRED')
    samples = raw.get('board_samples', [])
    if any(s.get('foreign_pids') for s in samples) or not any(set(s.get('owned_aliases', [])) == set(RESIDENTS) for s in samples):
        issues.append('ALL_OWNED_MEMORY_OR_FOREIGN_PROCESS_FAILED')
    for alias in RESIDENTS:
        rows = [r for r in raw['request_records'] if r['model'] == alias]
        pid = raw['ready'][alias]['pid']
        report = run/('trace-'+alias.replace('#','-')+'.nsys-rep')
        sql = report.with_suffix('.sqlite')
        export = export_sqlite(report, sql)
        anchors = anchors_from_records(rows, pid)
        parsed = parse_sqlite(sql, anchors)
        path = run/('kernels-'+alias.replace('#','-')+'.json')
        save(path, parsed)
        outputs[alias] = dict(export=export, parsed=receipt(path))
        if parsed['unmatched_anchor_count'] or not parsed['clock_mappings'] or any(not m['valid'] for m in parsed['clock_mappings'].values()):
            issues.append('CLOCK_ANCHOR_MAPPING_FAILED:'+alias)
        for row in rows:
            rid = row['request_id']
            if not any(p['request_id'] == rid and p['kernel_count'] > 0 for p in parsed['phases']):
                issues.append('NO_CORRELATED_REQUEST_KERNEL:'+rid)
            response = row.get('worker_response', {})
            nxt = row.get('previous_handler_next_request_ready', {})
            start, end = response.get('start_perf_ns'), response.get('end_perf_ns')
            receipt_ns = row.get('response_reader_event_enqueue_time_ns')
            if (row.get('status') != 'completed' or any(type(v) is not int for v in (start, end, receipt_ns, nxt.get('perf_ns')))
                    or not start <= end <= receipt_ns or not end <= nxt['perf_ns'] <= nxt['received_time_ns']):
                issues.append('HANDLER_RECEIPT_NEXT_READY_ORDER_FAILED:'+rid)
    result = dict(schema='gsdeploy.kernel_profile_smoke_report.v2', status='PASS' if not issues else 'FAIL',
                  issues=issues, raw=receipt(run/'raw.json'), traces=outputs,
                  counters='UNKNOWN:ERR_NVGPUCTRPERM', owned_peak_scope='NVML_sampled_not_instantaneous',
                  permit_release='NOT_APPLICABLE_default_shared_has_no_admission_permit',
                  claim='CORRECTNESS_AND_CLOCK_ONLY_NOT_TRAINING_NOT_PREDICTOR_ACCURACY')
    save(run/'smoke-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    for name in ('audit', 'execute', 'verify-smoke'):
        p = sub.add_parser(name); p.add_argument('--run', required=True)
        if name == 'execute':
            p.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    if args.op == 'prepare': result = prepare(args.output)
    elif args.op == 'audit': result = audit(args.run)
    elif args.op == 'verify-smoke': result = verify_smoke(args.run)
    else:
        result = collector.collect(execution_gate(args.run), trace=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
