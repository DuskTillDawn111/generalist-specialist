"""Prepare matched natural control/semantic-observer development pilots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,prepare,save,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--semantic',action='store_true');args=p.parse_args();output=args.output.resolve()
    prepare(output,args.reference.resolve(),image=True,ocr_replicas=1,mode='overlap',seed=args.seed)
    plan=json.loads((output/'plan.json').read_text());context=json.loads((output/'context.json').read_text())
    sources=json.loads((output/'sources.json').read_text());extra=[Path(__file__)]
    if args.semantic:
        observer=ROOT/'tools/predictor_validation_20261027/semantic_observed_worker.py';extra.append(observer)
        for entry in context['worker_bindings'].values():entry['command'][1]=str(observer)
    for source in extra:
        content=source.read_bytes();snapshot=output/'observer-source-snapshot'/source.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        sources.append(dict(path=str(source),sha256=hashlib.sha256(content).hexdigest(),snapshot=str(snapshot)))
    sources.sort(key=lambda s:s['path'])
    context.update(source_content=[{k:s[k] for k in ('path','sha256')} for s in sources],
                   observer_diagnostic='semantic_cuda_events' if args.semantic else 'complete_control')
    by_task={}
    for line in POOL.read_text().splitlines():
        row=json.loads(line);by_task.setdefault(row['task_type'],[]).append(row)
    def request(alias,task,index,offset=0.):
        return dict(alias=alias,task=task,pool_id=by_task[task][index%len(by_task[task])]['pool_id'],offset_ms=offset)
    # Same jobs, inputs and order in each matched control/observer pair.
    jobs=[]
    for task in by_task:
        for index in (0,15):
            jobs.append(dict(id=f'solo:g2:{task}:{index}',requests=[request('generalist_2#0',task,index)]))
            for offset in (0.,250.):
                jobs.append(dict(id=f'corun:{task}:{index}:{offset}',requests=[request('generalist_2#0',task,index),
                    request('image#0','image',index+7,offset),request('ocr#0','ocr',index+11,offset)]))
        item=[request('generalist_2#0',task,7)]
        for family,count in [('image',8),('ocr',6)]:
            item.extend(request(family+'#0',family,i+7,75.*i) for i in range(count))
        jobs.append(dict(id=f'sustained:{task}',requests=item))
    for family in ('image','ocr'):
        for index in (0,15):jobs.append(dict(id=f'solo:{family}:{index}',requests=[request(family+'#0',family,index)]))
    random.Random(args.seed).shuffle(jobs)
    plan.update(jobs=jobs,split='development_observer_diagnostic',execution_context_sha256=digest(context),
                observer_diagnostic=context['observer_diagnostic'],not_for_training=True,
                matched_job_content_sha256=digest(jobs),performance_qualified=False)
    save(output/'context.json',context);save(output/'sources.json',sources);save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output),jobs=len(jobs),observer=context['observer_diagnostic'],
                          matched_job_content_sha256=plan['matched_job_content_sha256'])))


if __name__=='__main__':main()
