"""Fresh complete eight-path solo data in the qualified reconstruction runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observer-report', required=True, type=Path)
    p.add_argument('--context-capture', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int)
    args = p.parse_args()
    report = json.loads(args.observer_report.read_text())
    if (report.get('observer_repeat_check_passed') is not True or report.get('issues')
            or args.seed in (6110491, 6110492)):
        raise ValueError('Complete observer gate and unused development seed required')
    source = args.context_capture.resolve()
    context = json.loads((source/'context.json').read_text())
    pilot = json.loads((source/'plan.json').read_text())
    expected = {f+'#0': f for f in ('generalist_2', 'text', 'image', 'ocr', 'asr')}
    if (context['resident_aliases'] != expected or context.get('observer_diagnostic') != 'five_model_reconstruct_semantic'
            or context['exllama_runtime']['matmul_recons_thd'] != 1
            or pilot['execution_context_sha256'] != digest(context)):
        raise ValueError('Exact five-model semantic reconstruction context required')
    context_receipts = [r for name, r in report['sources'].items() if name.startswith('semantic-')]
    source_raw = source/'raw.json'
    receipt = str(source_raw)+':sha256:'+hashlib.sha256(source_raw.read_bytes()).hexdigest()
    if receipt not in context_receipts:
        raise ValueError('Context capture is absent from successful observer gate')
    pool_content = POOL.read_bytes()
    if hashlib.sha256(pool_content).hexdigest() != context['input_pool_sha256']:
        raise ValueError('Frozen input pool changed')
    by_task = {}
    for line in pool_content.decode().splitlines():
        item = json.loads(line)
        by_task.setdefault(item['task_type'], []).append(item)
    jobs = []
    for alias, family in sorted(expected.items()):
        for task in by_task if family == 'generalist_2' else (family,):
            for item in by_task[task]:
                jobs.append(dict(id=f'solo:{alias}:{item["pool_id"]}', requests=[
                    dict(alias=alias, task=task, pool_id=item['pool_id'], offset_ms=0.)]))
    random.Random(args.seed).shuffle(jobs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for index, old in enumerate(json.loads((source/'sources.json').read_text())):
        if any(hashlib.sha256(Path(old[key]).read_bytes()).hexdigest() != old['sha256'] for key in ('path', 'snapshot')):
            raise ValueError('Frozen reconstruction source drift: '+old['path'])
        snapshot = output/'source-snapshot'/str(index)/Path(old['path']).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(old['snapshot']).read_bytes())
        receipts.append(dict(old, snapshot=str(snapshot)))
    plan = dict(schema='gsdeploy.native_activity_training_plan.v1', mode='solo', split='training',
        execution_context_sha256=digest(context), run_id=output.name, seed=args.seed,
        aliases=expected, jobs=jobs, priming='largest_three_per_supported_task', budget_seconds=1800,
        resident_domain=dict.fromkeys(expected.values(), 1), confirmation_seeds_reserved=[6110491, 6110492],
        observer_report_sha256=hashlib.sha256(args.observer_report.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='complete fresh eight-path baseline repeat evidence; no predictor qualification from observer or solo data',
        old_runtime_templates_reusable=False, pure_gpu_stage_claim=False)
    save(output/'context.json', context)
    save(output/'sources.json', receipts)
    save(output/'plan.json', plan)
    (output/'solo-preparation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(output), jobs=len(jobs), requests=len(jobs), context=digest(context))))


if __name__ == '__main__':
    main()
