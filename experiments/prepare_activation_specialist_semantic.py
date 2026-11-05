"""Matched four-specialist observer pilots for the unresolved service phase structure."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, digest, save
from experiments.prepare_activation_resident import prepare_resident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    prepare_resident(output, args.reference.resolve(), dict(text=1,image=1,ocr=1,asr=1),
                     mode='overlap', seed=args.seed)
    plan = json.loads((output/'plan.json').read_text())
    context = json.loads((output/'context.json').read_text())
    sources = json.loads((output/'sources.json').read_text())
    extra = [Path(__file__).resolve()]
    observer = ROOT/'tools/predictor_validation_20261027/semantic_observed_worker.py'
    if args.semantic:
        extra.append(observer)
        for binding in context['worker_bindings'].values():
            binding['command'][1] = str(observer)
    for source in extra:
        content = source.read_bytes()
        snapshot = output/'observer-source-snapshot'/source.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        sources.append(dict(path=str(source), snapshot=str(snapshot), sha256=hashlib.sha256(content).hexdigest()))
    sources.sort(key=lambda source: source['path'])
    context.update(source_content=[{key: source[key] for key in ('path','sha256')} for source in sources],
        observer_diagnostic='specialists_semantic' if args.semantic else 'specialists_complete')
    by_task = {}
    for line in POOL.read_text().splitlines():
        item = json.loads(line)
        by_task.setdefault(item['task_type'], []).append(item)
    def request(task, index, offset=0.):
        return dict(alias=task+'#0', task=task, pool_id=by_task[task][index % len(by_task[task])]['pool_id'],
                    offset_ms=offset)
    jobs = []
    for task in by_task:
        for index in (0,15):
            jobs.append(dict(id=f'solo:{task}:{index}', requests=[request(task,index)]))
            for offset in (0.,250.):
                requests = [request(task,index)]+[request(peer,index+7,offset) for peer in by_task if peer!=task]
                jobs.append(dict(id=f'corun:{task}:{index}:{offset}', requests=requests))
        requests = [request(task,7)]
        for peer in by_task:
            if peer != task:
                requests.extend(request(peer,i+7,75.*i) for i in range(6))
        jobs.append(dict(id=f'sustained:{task}', requests=requests))
    random.Random(args.seed).shuffle(jobs)
    plan.update(jobs=jobs, split='development_observer_diagnostic',
        execution_context_sha256=digest(context), observer_diagnostic=context['observer_diagnostic'],
        not_for_training=True, matched_job_content_sha256=digest(jobs), performance_qualified=False,
        expected_semantic_components=dict(text='model_forward',asr='model_forward',
                                          image='image_features',ocr='ocr_detection'))
    save(output/'context.json',context)
    save(output/'sources.json',sources)
    save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output), jobs=len(jobs),
        requests=sum(len(job['requests']) for job in jobs), context=plan['execution_context_sha256'])))


if __name__ == '__main__':
    main()
