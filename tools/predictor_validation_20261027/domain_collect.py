"""Predeclared complete known-input task-count training supplement.

Selection uses only the training input assignment and finite fixed-residency
domain, never validation errors. Execution reuses the exact frozen v7 workers.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import shutil

from .background_collect import collect,context_identity,receipt,save
from .profiling import RESIDENTS,digest,PLAN_SCHEMA


def prepare(profile_dir,output,repetitions=2):
    profile_dir,output=Path(profile_dir).resolve(),Path(output).resolve()
    context=json.loads((profile_dir/'context.json').read_text())
    original=json.loads((profile_dir/'plan.json').read_text())
    if repetitions<2:
        raise ValueError('At least two independent training repeats required')
    known={(r['task'],r['pool_id']) for r in original['input_assignments'] if r['view']=='known_input'}
    inputs={t:sorted(p for task,p in known if task==t) for t in ('text','image','asr','ocr')}
    if not inputs or len({len(rows) for rows in inputs.values()})!=1:
        raise ValueError('Balanced known input sets required')
    jobs=[]
    for tc,gt,oc in itertools.product(range(3),(None,'text','image','asr'),range(2)):
        members=['text#'+str(i) for i in range(tc)]
        if gt:members.append('generalist_2#0')
        if oc:members.append('ocr#0')
        if not members:continue
        for index in range(len(inputs['text'])):
            tasks={a:(gt if a=='generalist_2#0' else 'ocr' if a=='ocr#0' else 'text') for a in members}
            for repeat in range(repetitions):
                job=dict(context=f'complete_t{tc}_g{gt or "idle"}_o{oc}',
                    tasks=tasks,inputs={a:inputs[t][index] for a,t in tasks.items()},
                    start_offsets_ms=dict.fromkeys(members,0),repeat=repeat,split='training',
                    validation_view='training_domain_completion',supplement='complete-known-task-domain-v8')
                job['id']=digest(job);jobs.append(job)
    random.Random('complete-known-task-domain-v8-training-only').shuffle(jobs)
    plan=dict(schema=PLAN_SCHEMA,execution_context_sha256=context_identity(context),resident_aliases=RESIDENTS,
        inputs=[r for r in original['inputs'] if (r['task'],r['pool_id']) in known],jobs=jobs,
        input_assignments=[r for r in original['input_assignments'] if (r['task'],r['pool_id']) in known],
        generator_source=receipt(__file__),source_training_plan=receipt(profile_dir/'plan.json'),
        no_retries=True,GPU_started=False,claim='TRAINING_ONLY_FULL_KNOWN_TASK_DOMAIN_NOT_VALIDATION')
    plan['plan_sha256']=digest(plan)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(profile_dir/'context.json',output/'context.json')
    save(output/'plan.json',plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare');a.add_argument('--profile-dir',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('execute');a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.op=='prepare':
        plan=prepare(args.profile_dir,args.output);print(json.dumps(dict(status='PREPARED',jobs=len(plan['jobs']))))
    else:collect(args.output,split='training')


if __name__=='__main__':main()
