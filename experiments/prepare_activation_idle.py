"""Matched long idle and busy-to-idle contrasts for board-energy calibration."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,prepare,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--ocr-replicas',type=int,choices=[1,2],required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args();output=args.output.resolve()
    prepare(output,args.reference.resolve(),image=True,ocr_replicas=args.ocr_replicas,mode='overlap',seed=args.seed)
    pool=[json.loads(line) for line in POOL.read_text().splitlines()];by_task={}
    for item in pool:by_task.setdefault(item['task_type'],[]).append(item)
    plan=json.loads((output/'plan.json').read_text());tasks=list(by_task);random.Random(args.seed).shuffle(tasks)
    jobs=[dict(id='resident_idle_before',requests=[],duration_ms=30000)]
    for index,task in enumerate(tasks):
        item=by_task[task][len(by_task[task])//2]
        requests=[dict(alias='generalist_2#0',task=task,pool_id=item['pool_id'],offset_ms=0.)]
        for alias,family in plan['aliases'].items():
            if family=='generalist_2':continue
            for i in range(16 if family=='image' else 6):
                item=by_task[family][i%len(by_task[family])]
                requests.append(dict(alias=alias,task=family,pool_id=item['pool_id'],offset_ms=i*75.))
        jobs.append(dict(id='controlled_busy:'+task,requests=requests))
        if index in (1,3):jobs.append(dict(id=f'resident_idle_after_busy:{index}',requests=[],duration_ms=30000))
    plan.update(jobs=jobs,diagnostic_supplement=True,
        design='three fixed 30 second resident idle windows, before and after controlled bursts',
        idle_is_not_unloaded=True,admission_enabled=False)
    save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output),jobs=len(jobs),idle_seconds=90,context=plan['execution_context_sha256'])))


if __name__=='__main__':main()
