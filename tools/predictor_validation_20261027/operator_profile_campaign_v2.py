"""Small isolated affine/kernel proof plus matched unprofiled timing campaign.

All four original owners remain resident. New exclusive preparation only;
neither preparation nor audit starts a GPU worker. Validation is not scheduled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from . import background_collect as collector
from .kernel_profile_campaign_v2 import choose_inputs
from .native_collect import ROOT, context_identity, receipt, save
from .profiling import RESIDENTS, digest
from .smoke_verifier_v3 import check_receipt


def stat_receipt(path):
    path = Path(path).resolve()
    info = path.stat()
    return dict(path=str(path), inode=info.st_ino, device=info.st_dev, size=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def shared_execution_paths(workers):
    """The direct, executed dispatch/native path, excluding offline scoring.

    Full current sources are still frozen/audited separately. This narrower
    identity is solely to establish which old runtime smoke applies to the
    unchanged dispatch and model path when offline scorers evolve.
    """
    paths = {ROOT/'gsdeploy/resources.py'}
    paths.update(ROOT/'gsdeploy/runtime'/n for n in ('deployment.py', 'transport.py', 'schema.py', 'admission.py', 'runner.py'))
    paths.update(Path(__file__).with_name(n) for n in ('semantic_worker.py', 'semantic_observer.py', 'background_collect.py', 'native_collect.py'))
    for family in set(RESIDENTS.values()):
        binding = workers[family]
        paths.add(Path(binding['spec']))
        cmd = binding['command']
        paths.update(Path(cmd[cmd.index('--source-root')+1]).rglob('*.py'))
        spec = json.loads(Path(binding['spec']).read_text())
        if spec.get('adapter_source'):
            paths.add(Path(spec['adapter_source']))
    return {p.resolve() for p in paths}


def resource_paths(workers):
    small, large = set(), set()
    def add(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        (large if path.stat().st_size > 32*2**20 else small).add(path)
    for family in set(RESIDENTS.values()):
        spec = json.loads(Path(workers[family]['spec']).read_text())
        resources = spec['resources']
        for key in ('protocol', 'classmap', 'inference_source'):
            if resources.get(key):
                add(resources[key])
        checkpoint = resources['checkpoint']
        if isinstance(checkpoint, dict):
            root = Path(checkpoint['path'])
            for item in checkpoint['files']:
                add(root/item['name'])
        else:
            root = Path(checkpoint)
            for subdir in ('PaddlePaddle--PP-OCRv6_medium_det_safetensors', 'PaddlePaddle--PP-OCRv6_medium_rec_safetensors'):
                for path in (root/subdir).rglob('*'):
                    if path.is_file():
                        add(path)
        peft = spec['execution'].get('peft_adapter')
        if peft:
            manifest = Path(peft['manifest'])
            add(manifest)
            if receipt(manifest)['sha256'] != peft['sha256']:
                raise ValueError('Pinned PEFT manifest mismatch')
            content = json.loads(manifest.read_text())
            for item in content['files']:
                path = Path(content['path'])/item['name']
                # This adapter is modest enough to hash once; do not label a
                # checkpoint stat record as a fresh full weight hash.
                if receipt(path)['sha256'] != item['sha256']:
                    raise ValueError('Pinned PEFT file mismatch')
                add(path)
    return small, large


def prepare(output, smoke_report):
    output, smoke_report = Path(output).resolve(), Path(smoke_report).resolve()
    report = json.loads(smoke_report.read_text())
    if report.get('schema') != 'gsdeploy.kernel_smoke_strict_review.v3' or report.get('status') != 'PASS':
        raise ValueError('Strict v3 runtime/clock smoke PASS required')
    for name in ('raw', 'plan', 'context', 'executed_jobs'):
        if not check_receipt(report['evidence'][name]):
            raise ValueError('Changed strict-smoke evidence: '+name)
    base = ROOT/'artifacts/static-study-v6-20261021/inputs'
    workers = json.loads((base/'worker-registry.json').read_text())
    shared = shared_execution_paths(workers)
    historical = {s['original_path']: s['frozen']['sha256'] for s in report['evidence']['original_sources']}
    if any(historical.get(str(p)) != receipt(p)['sha256'] for p in shared):
        raise ValueError('Previous smoke direct execution dependency differs')
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (base/'runtime-inputs.jsonl').read_text().splitlines()]
    selection = choose_inputs(rows)
    primers = [selection['training']['text'][0], selection['training']['ocr'][0]]
    selected_ids = {r['pool_id'] for r in primers}
    small, large = resource_paths(workers)
    paths = set((ROOT/'gsdeploy').rglob('*.py')) | small | shared
    paths.update(Path(__file__).with_name(n) for n in (
        'operator_profile_campaign_v2.py', 'kernel_profile_campaign_v2.py', 'smoke_verifier_v3.py',
        'operator_worker_v2.py', 'operator_observer_v2.py', 'observed_worker.py', 'profiling.py', 'trace_export.py'))
    paths.update((base/'runtime-inputs.jsonl', base/'worker-registry.json'))
    paths.update(Path(r['sample']['input']) for r in rows if r['pool_id'] in selected_ids and 'input' in r['sample'])
    sources = []
    for index, path in enumerate(sorted(paths)):
        path = path.resolve()
        frozen = output/'source-snapshot'/str(index)/path.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(path, frozen)
        sources.append(dict(original_path=str(path), frozen=receipt(frozen)))
    large_stats = [stat_receipt(path) for path in sorted(large)]
    save(output/'large-weight-stats.json', large_stats)
    save(output/'input-selection.json', selection)
    common = dict(residents=RESIDENTS, sources=[s['frozen'] for s in sources], execution_sources=sources,
        worker_bindings={f: workers[f] for f in sorted(set(RESIDENTS.values()))},
        threads={k:'1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        execution=dict(batch_size=1, backend='retained-native-observed-v1', weight_reuse=False,
                       physical_owners=4, admission='default_shared_no_permit'),
        shared_native_execution_sources=[receipt(p) for p in sorted(shared)],
        large_weight_stats=receipt(output/'large-weight-stats.json'),
        large_weight_identity_scope='stat_frozen_and_checked_not_fresh_full_weight_sha256',
        inherited_runtime_smoke=receipt(smoke_report),
        counters='UNKNOWN:ERR_NVGPUCTRPERM', device='RTX4090-49140MiB', task_domain=['text', 'ocr'],
        phase_scope='semantic worker host ranges; GPU attribution requires separately traced run')
    runs = {}
    for variant in ('structure-smoke', 'natural-smoke'):
        run = output/variant
        run.mkdir()
        context = {**common, 'worker_variant': 'operator_worker_v2.py' if variant == 'structure-smoke' else 'semantic_worker.py',
                   'profiler': 'nsys-cuda-nvtx' if variant == 'structure-smoke' else 'none-natural-with-semantic-markers',
                   'affine_observation': variant == 'structure-smoke'}
        sha = context_identity(context)
        jobs = []
        for alias in RESIDENTS:
            task = 'ocr' if alias == 'ocr#0' else 'text'
            item = next(p for p in primers if p['task'] == task)
            job = dict(context='isolated-'+alias, inputs={alias:item['pool_id']}, tasks={alias:task},
                       start_offsets_ms={alias:0}, repeat=0, split='smoke',
                       validation_view='isolated_structure_correctness_not_fit')
            # Identical pair-job keys across instrumented/natural executions.
            job['id'] = digest(job)
            jobs.append(job)
        plan = dict(schema='gsdeploy.operator_profile_plan.v2', execution_context_sha256=sha,
            resident_aliases=RESIDENTS, inputs=primers, jobs=jobs, no_retries=True, GPU_started=False,
            variant=variant, paired_jobs_sha256=digest(jobs),
            input_selection=receipt(output/'input-selection.json'), inherited_runtime_smoke=receipt(smoke_report),
            claim='ISOLATED_STRUCTURE_OR_MATCHED_NATURAL_SMOKE_NOT_TRAINING_NOT_VALIDATION')
        plan['plan_sha256'] = digest(plan)
        save(run/'context.json', context); save(run/'plan.json', plan)
        runs[variant] = dict(plan=receipt(run/'plan.json'), context=receipt(run/'context.json'), jobs=len(jobs),
                             execution_context_sha256=sha)
    result = dict(schema='gsdeploy.operator_profile_campaign.v2', status='PREPARED_CPU_ONLY',
                  runs=runs, inherited_runtime_smoke=receipt(smoke_report),
                  direct_native_identity=digest(common['shared_native_execution_sources']),
                  full_sources=len(sources), large_weight_stat_records=len(large_stats),
                  no_training_or_validation_execution=True,
                  validation_gate='FROZEN_MODEL_AND_SEALED_PREDICTIONS_REQUIRED_SEPARATE_PROTOCOL')
    save(output/'campaign.json', result)
    return result


def audit(run):
    run = Path(run).resolve()
    plan = json.loads((run/'plan.json').read_text()); context = json.loads((run/'context.json').read_text())
    for source in context['execution_sources']:
        if not check_receipt(source['frozen']) or receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
            raise ValueError('Source/input changed: '+source['original_path'])
    for reference in (context['large_weight_stats'], context['inherited_runtime_smoke'], plan['input_selection']):
        if not check_receipt(reference):
            raise ValueError('Manifest/report receipt changed')
    for row in json.loads(Path(context['large_weight_stats']['path']).read_text()):
        if stat_receipt(row['path']) != row:
            raise ValueError('Weight stat changed: '+row['path'])
    if context_identity(context) != plan['execution_context_sha256']:
        raise ValueError('Execution context mismatch')
    if context['residents'] != RESIDENTS or context['execution']['physical_owners'] != 4:
        raise ValueError('Original four physical owners required')
    return dict(status='CPU_SOURCE_RESOURCE_AUDIT_PASS', execution_context_sha256=plan['execution_context_sha256'])


def execute(run):
    run = Path(run).resolve(); audit(run)
    plan = json.loads((run/'plan.json').read_text())
    if plan['variant'] == 'natural-smoke':
        report_path = run.parent/'structure-smoke/strict-review-v3/report.json'
        checked = json.loads(report_path.read_text())
        if checked.get('status') != 'PASS' or not check_receipt(checked.get('evidence', {}).get('plan'), run.parent/'structure-smoke/plan.json'):
            raise ValueError('Matching structure runtime/clock PASS is required first')
    raw = collector.collect(run, trace=plan['variant'] == 'structure-smoke')
    return dict(status=raw['status'], raw=receipt(run/'raw.json'), requests=len(raw['request_records']),
                performance_scope=raw['performance_scope'], U=None, validation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True); p.add_argument('--smoke-report', required=True)
    p = sub.add_parser('audit'); p.add_argument('--run', required=True)
    p = sub.add_parser('execute'); p.add_argument('--run', required=True); p.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    if args.op == 'prepare': result = prepare(args.output, args.smoke_report)
    elif args.op == 'audit': result = audit(args.run)
    else: result = execute(args.run)
    print(json.dumps(result, indent=2))
    return 1 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
