"""Prepare explicit resident domains for the unchanged complete native collector."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,prepare,save,digest


def prepare_resident(output,reference,deployment,*,mode='solo',seed=6110601):
    if (set(deployment)-{'generalist_2','text','image','ocr','asr'} or mode not in {'solo','overlap'}
            or any(type(n) is not int or n<1 or n>2 or m!='ocr' and n!=1 for m,n in deployment.items())):
        raise ValueError('Explicit retained resident domain required; OCR alone permits two copies')
    supported=set(('text','image','ocr','asr'))
    if 'generalist_2' not in deployment and not supported<=set(deployment):
        raise ValueError('Prepared resident domain must serve the complete four-task input pool')
    # Retain the frozen collector source set, then add all executable bindings
    # required by this newly declared resident context. Earlier contexts stay intact.
    prepare(output,reference,image=bool(deployment.get('image')),ocr_replicas=deployment.get('ocr',0),mode=mode,seed=seed)
    workers=json.loads((reference/'worker-registry.json').read_text())
    aliases={f'{m}#{i}':m for m,n in sorted(deployment.items()) for i in range(n)}
    registry={a:copy.deepcopy(workers[m]) for a,m in aliases.items()}
    receipts=json.loads((output/'sources.json').read_text());existing={r['path'] for r in receipts}
    sources={Path(__file__)}
    for binding in registry.values():
        spec_path=Path(binding['spec']);spec=json.loads(spec_path.read_text());sources.add(spec_path)
        command=binding['command'];sources.update(Path(command[command.index('--source-root')+1]).rglob('*.py'))
        if spec.get('adapter_source'):sources.add(Path(spec['adapter_source']))
        if spec['resources'].get('inference_source'):sources.add(Path(spec['resources']['inference_source']))
    for path in sorted(sources):
        if str(path) in existing:continue
        content=path.read_bytes();dest=output/'additional-source-snapshot'/str(len(receipts))/path.name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        receipts.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),snapshot=str(dest)))
    receipts.sort(key=lambda r:r['path'])
    context=json.loads((output/'context.json').read_text())
    context.update(resident_aliases=aliases,worker_bindings=registry,preparation_workflow='general_resident_v1',
                   source_content=[{k:r[k] for k in ('path','sha256')} for r in receipts])
    pool=[json.loads(line) for line in POOL.read_text().splitlines()];by_task={}
    for row in pool:by_task.setdefault(row['task_type'],[]).append(row)
    def request(alias,task,index):
        return dict(alias=alias,task=task,pool_id=by_task[task][index%len(by_task[task])]['pool_id'],offset_ms=0.)
    jobs=[]
    if mode=='solo':
        for alias,family in aliases.items():
            for task in by_task if family=='generalist_2' else (family,):
                for index,item in enumerate(by_task[task]):
                    jobs.append(dict(id=f'solo:{alias}:{item["pool_id"]}',requests=[request(alias,task,index)]))
    else:
        for ti,(task,inputs) in enumerate(by_task.items()):
            lead='generalist_2#0' if 'generalist_2' in deployment else task+'#0'
            for index,item in enumerate(inputs):
                offered=[request(lead,task,index)]
                for alias,family in aliases.items():
                    if alias!=lead:offered.append(request(alias,task if family=='generalist_2' else family,index+7*ti))
                jobs.append(dict(id=f'full:{task}:{index}',requests=offered))
        # Actual pair/triple contrasts identify reductions in the active roster.
        for n in range(2,len(aliases)):
            for subset in itertools.combinations(aliases,n):
                for index in (0,15,31):
                    offered=[request(alias,'text' if aliases[alias]=='generalist_2' else aliases[alias],index)
                             for alias in subset]
                    jobs.append(dict(id='subset:'+','.join(subset)+':'+str(index),requests=offered))
    random.Random(seed).shuffle(jobs)
    plan=json.loads((output/'plan.json').read_text());plan.update(aliases=aliases,jobs=jobs,
        execution_context_sha256=digest(context),resident_domain=deployment)
    save(output/'context.json',context);save(output/'sources.json',receipts);save(output/'plan.json',plan)
    print(json.dumps(dict(event='general_resident_prepared',output=str(output),deployment=deployment,
        jobs=len(jobs),context=digest(context))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--deployment',required=True,type=Path)
    p.add_argument('--mode',choices=['solo','overlap'],default='solo');p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();prepare_resident(args.output.resolve(),args.reference.resolve(),json.loads(args.deployment.read_text()),mode=args.mode,seed=args.seed)


if __name__=='__main__':main()
