"""Matched diagnostic traces for text/ASR pressure and fixed-domain ASR routing."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,save,digest
from experiments.prepare_activation_resident import prepare_resident
from gsdeploy.trace import make_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--hybrid',action='store_true');args=p.parse_args();output=args.output.resolve()
    deployment=dict(generalist_2=1,image=1,ocr=1,asr=1) if args.hybrid else dict(text=1,image=1,ocr=1,asr=1)
    prepare_resident(output,args.reference.resolve(),deployment,mode='overlap',seed=args.seed)
    plan=json.loads((output/'plan.json').read_text());by_task={}
    for line in POOL.read_text().splitlines():
        row=json.loads(line);by_task.setdefault(row['task_type'],[]).append(row)
    jobs=[]
    for case,asr_rate in enumerate((1.,1.2)):
        rates=dict(text=1.,image=.0748360842705697,ocr=.0748360842705697,asr=asr_rate)
        trace=make_trace(rates,45.,args.seed+100*case,'poisson');counters=dict.fromkeys(by_task,0)
        cohort=[];rng=random.Random(args.seed+100*case+10000)
        for offset,task in trace:
            n=counters[task];counters[task]+=1
            item=by_task[task][n%len(by_task[task])]
            cohort.append(dict(task=task,pool_id=item['pool_id'],offset_ms=offset*1000.,route_uniform=rng.random()))
        for alpha in ((0.,.1,.2,.25) if args.hybrid else (0.,)):
            requests=[]
            for row in cohort:
                task=row['task'];alias=task+'#0'
                if args.hybrid and (task=='text' or task=='asr' and row['route_uniform']<alpha):alias='generalist_2#0'
                requests.append(dict(alias=alias,**{k:row[k] for k in ('task','pool_id','offset_ms')}))
            jobs.append(dict(id=f'shared_pressure:asr{asr_rate}:g_asr{alpha}',requests=requests,duration_ms=45000,
                             rates=rates,asr_generalist_share=alpha,shared_trace_sha256=digest(cohort),
                             finite_probe_not_capacity_or_utility=True))
    random.Random(args.seed+7000).shuffle(jobs)
    plan.update(jobs=jobs,mode='shared_pressure_probe',split='development_shared_pressure',
                not_for_training=True,admission_enabled=False,
                design='same 45 second task/input/arrival trace across all-S and four fixed-resident hybrid ASR shares',
                no_utility_normalization=True,steady_capacity_qualified=False)
    save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output),jobs=len(jobs),requests=sum(len(j['requests']) for j in jobs),
                          deployment=deployment,context=plan['execution_context_sha256'])))


if __name__=='__main__':main()
